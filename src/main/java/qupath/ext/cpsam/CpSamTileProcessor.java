package qupath.ext.cpsam;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.TranslateException;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.djl.DjlTools;
import qupath.lib.awt.common.BufferedImageTools;
import qupath.lib.experimental.pixels.MeasurementProcessor;
import qupath.lib.experimental.pixels.Parameters;
import qupath.lib.experimental.pixels.Processor;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ColorTransforms;
import qupath.lib.images.servers.PixelType;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;
import qupath.opencv.ops.ImageOp;
import qupath.opencv.ops.ImageOps;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Processor for individual tiles: per-channel normalize, convert to tensor, call CPSAM model.
 * Mirrors TilePredictionProcessor from InstanSeg.
 *
 * Input: Mat (OpenCV) from the ImageSupplier (ImageOps.buildImageDataOp)
 * Output: NDArray[] (mask) → converted to Mat for output handler
 *
 * Note: resize-by-diameter, padding for transformer, and mask crop/rescale
 * are handled internally by the TorchScript model.
 */
class CpSamTileProcessor implements Processor<Mat, Mat, NDArray[]> {

    private static final Logger logger = LoggerFactory.getLogger(CpSamTileProcessor.class);
    private static final double EPSILON = 1e-6;

    private final BlockingQueue<Predictor<NDList, NDList>> predictors;
    private final Collection<ColorTransforms.ColorTransform> channels;
    private final double lowPercentile;
    private final double highPercentile;

    private final NDManager ndManager;
    private final NDArray diameterTensor;
    private final NDArray cellprobTensor;
    private final NDArray flowTensor;
    private final NDArray niterTensor;
    private final NDArray batchSizeTensor;

    private final AtomicLong nPixelsProcessed = new AtomicLong(0);
    private final AtomicInteger nTilesProcessed = new AtomicInteger(0);
    private final AtomicInteger nTilesFailed = new AtomicInteger(0);
    private final AtomicBoolean wasInterrupted = new AtomicBoolean(false);
    private final Map<ROI, ImageOp> normalization = Collections.synchronizedMap(new WeakHashMap<>());

    CpSamTileProcessor(BlockingQueue<Predictor<NDList, NDList>> predictors,
                       Collection<? extends ColorTransforms.ColorTransform> channels, NDManager ndManager,
                       double diameter, float cellprobThreshold, float flowThreshold,
                       int niter, int batchSize) {
        this.predictors = predictors;
        this.channels = List.copyOf(channels);
        this.lowPercentile = 1.0;
        this.highPercentile = 99.0;
        this.ndManager = ndManager;
        this.diameterTensor = ndManager.create((float) diameter);
        this.cellprobTensor = ndManager.create(cellprobThreshold);
        this.flowTensor = ndManager.create(flowThreshold);
        this.niterTensor = ndManager.create((long) niter);
        this.batchSizeTensor = ndManager.create(batchSize);
    }

    public int getTilesProcessedCount() { return nTilesProcessed.get(); }
    public int getTilesFailedCount() { return nTilesFailed.get(); }
    public long getPixelsProcessedCount() { return nPixelsProcessed.get(); }
    public boolean wasInterrupted() { return wasInterrupted.get(); }

    @Override
    public NDArray[] process(Parameters<Mat, Mat> params) throws IOException {
        Mat mat = params.getImage();
        boolean verboseLogging = CpSamPreferences.verboseLoggingProperty().get();
        long tileStart = System.currentTimeMillis();

        if (verboseLogging) {
            logger.info("Tile start: region={}, mat={}x{}, channels={}, type={}",
                    params.getRegionRequest(), mat.cols(), mat.rows(), mat.channels(), mat.type());
        }

        ImageOp preprocessing = normalization.computeIfAbsent(
                params.getParent().getROI(),
                roi -> getNormalization(params.getImageData(), roi, channels, lowPercentile, highPercentile));

        mat = preprocessing.apply(mat);
        NDArray batchInput = matToBatchInput(mat, ndManager);

        if (verboseLogging) {
            logger.info("Tile model input BCHW: shape={}, dtype={}", batchInput.getShape(), batchInput.getDataType());
            logPerChannelMinMax("Tile normalized BCHW", batchInput.squeeze(0));
        }

        NDList modelInput = new NDList(batchInput, diameterTensor, cellprobTensor, flowTensor, niterTensor, batchSizeTensor);

        if (verboseLogging) {
            logger.info("Tile scalar params: diameter={}, cellprob={}, flow={}, niter={}, batchSize={}",
                    diameterTensor.getFloat(), cellprobTensor.getFloat(), flowTensor.getFloat(), niterTensor.getLong(), batchSizeTensor.getInt());
        }

        Predictor<NDList, NDList> predictor = null;
        try {
            predictor = predictors.take();
            NDList output = predictor.predict(modelInput);
            // Release input tensor immediately — inference is done
            batchInput.close();
            // Copy mask to float32, then release the raw GPU output
            NDArray maskND = output.get(0).squeeze(0).toType(DataType.FLOAT32, true); // [B,H,W] → [H,W] float32
            output.close();

            if (verboseLogging) {
                logger.info("Tile model output: shape={}, dtype={}, min={}, max={}, elapsedMs={}",
                        maskND.getShape(), maskND.getDataType(), maskND.min().getFloat(), maskND.max().getFloat(),
                        System.currentTimeMillis() - tileStart);
            }

            nPixelsProcessed.addAndGet((long) mat.rows() * mat.cols());
            nTilesProcessed.incrementAndGet();

            return new NDArray[]{maskND};

        } catch (TranslateException e) {
            nTilesFailed.incrementAndGet();
            logger.error("Error in prediction for tile [{}x{}]", mat.cols(), mat.rows(), e);
            return null;
        } catch (InterruptedException e) {
            wasInterrupted.set(true);
            nTilesFailed.incrementAndGet();
            logger.debug("Prediction interrupted", e);
            return null;
        } finally {
            if (predictor != null) {
                try {
                    predictors.put(predictor);
                } catch (InterruptedException e) {
                    logger.warn("Tiling interrupted");
                }
            }
        }
    }

    /**
     * Convert a normalized float32 Mat directly to the model's expected BCHW tensor.
     *
     * NOTE: DjlTools.matToNDArray with layout "CHW" routes through opencv_dnn.blobFromImage()
     * and is safe here because preprocessing has already converted the tile to float32.
     */
    private static NDArray matToBatchInput(Mat mat, NDManager manager) {
        return DjlTools.matToNDArray(manager, mat, "CHW")
                .toType(DataType.FLOAT32, false)
                .expandDims(0);
    }

    private static ImageOp getNormalization(
            ImageData<BufferedImage> imageData,
            ROI roi,
            Collection<ColorTransforms.ColorTransform> channels,
            double lowPerc,
            double highPerc) {

        try {
            BufferedImage image;
            double downsample = Math.max(1, Math.max(roi.getBoundsWidth(), roi.getBoundsHeight()) / 1024);
            var request = RegionRequest.createInstance(imageData.getServerPath(), downsample, roi);
            image = imageData.getServer().readRegion(request);

            var params = channels.stream().map(colorTransform -> {
                var mask = BufferedImageTools.createROIMask(image.getWidth(), image.getHeight(), roi, request);
                float[] maskPix = ColorTransforms.createChannelExtractor(0).extractChannel(null, mask, null);
                float[] fpix = colorTransform.extractChannel(imageData.getServer(), image, null);

                int ind = 0;
                for (int i = 0; i < maskPix.length; i++) {
                    if (maskPix[i] == 255) {
                        fpix[ind] = fpix[i];
                        ind++;
                    }
                }

                double[] usePixels = new double[ind];
                for (int i = 0; i < ind; i++) {
                    usePixels[i] = fpix[i];
                }

                double lo = MeasurementProcessor.Functions.percentile(lowPerc).apply(usePixels);
                double hi = MeasurementProcessor.Functions.percentile(highPerc).apply(usePixels);
                double scale = hi > lo ? 1.0 / (hi - lo + EPSILON) : 0.0;
                double offset = -lo * scale;
                return new double[]{offset, scale, lo, hi};
            }).toList();

            return ImageOps.Core.sequential(
                    ImageOps.Core.ensureType(PixelType.FLOAT32),
                    ImageOps.Core.multiply(params.stream().mapToDouble(e -> e[1]).toArray()),
                    ImageOps.Core.add(params.stream().mapToDouble(e -> e[0]).toArray())
            );
        } catch (Exception e) {
            logger.error("Error preparing cached CPSAM normalization", e);
        }

        return ImageOps.Core.sequential(
                ImageOps.Core.ensureType(PixelType.FLOAT32),
                ImageOps.Normalize.percentile(lowPerc, highPerc, true, EPSILON)
        );
    }

    private static void logPerChannelMinMax(String label, NDArray chw) {
        int channels = (int) chw.getShape().get(0);
        for (int ch = 0; ch < channels; ch++) {
            NDArray channel = chw.get(ch);
            logger.info("{} channel {}: min={}, max={}", label, ch, channel.min().getFloat(), channel.max().getFloat());
        }
    }
}
