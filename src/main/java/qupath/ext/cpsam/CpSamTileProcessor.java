package qupath.ext.cpsam;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslateException;
import ij.IJ;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.experimental.pixels.Parameters;
import qupath.lib.experimental.pixels.Processor;
import qupath.lib.images.servers.ColorTransforms;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;
import qupath.opencv.ops.ImageOp;
import qupath.opencv.tools.OpenCVTools;

import java.io.IOException;
import java.nio.file.Path;
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

    private final BlockingQueue<Predictor<NDList, NDList>> predictors;
    private final Collection<ColorTransforms.ColorTransform> channels;
    private final double lowPercentile;
    private final double highPercentile;
    private final double normalizationDownsample; // NaN → derive from normalizationMaxDimension
    private final int normalizationMaxDimension;

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

    /** Directory to save normalized tiles into before inference, or {@code null} to skip saving. */
    private final Path saveDir;
    private final AtomicInteger saveTileIndex = new AtomicInteger(0);

    CpSamTileProcessor(BlockingQueue<Predictor<NDList, NDList>> predictors,
                       Collection<? extends ColorTransforms.ColorTransform> channels, NDManager ndManager,
                       double diameter, float cellprobThreshold, float flowThreshold,
                       int niter, int batchSize,
                       double normalizationDownsample, int normalizationMaxDimension,
                       double normalizationLowPercentile, double normalizationHighPercentile,
                       Path saveDir) {
        this.predictors = predictors;
        this.channels = List.copyOf(channels);
        this.lowPercentile = normalizationLowPercentile;
        this.highPercentile = normalizationHighPercentile;
        this.normalizationDownsample = normalizationDownsample;
        this.normalizationMaxDimension = normalizationMaxDimension;
        this.ndManager = ndManager;
        this.diameterTensor = ndManager.create((float) diameter);
        this.cellprobTensor = ndManager.create(cellprobThreshold);
        this.flowTensor = ndManager.create(flowThreshold);
        this.niterTensor = ndManager.create((long) niter);
        this.batchSizeTensor = ndManager.create(batchSize);
        this.saveDir = saveDir;
    }

    public int getTilesProcessedCount() { return nTilesProcessed.get(); }
    public int getTilesFailedCount() { return nTilesFailed.get(); }
    public long getPixelsProcessedCount() { return nPixelsProcessed.get(); }
    public boolean wasInterrupted() { return wasInterrupted.get(); }

    /**
     * Saves the normalized tile mat as a 32-bit TIFF image in {@link #saveDir} for
     * preprocessing inspection. Uses ImageJ's IJ.save(), which correctly handles
     * any channel count (including 2-channel images) and preserves float32 precision.
     * Errors are logged as warnings and do not abort the run.
     */
    private void saveTile(Mat mat, RegionRequest region) {
        int idx = saveTileIndex.getAndIncrement();
        try {
            var imp = OpenCVTools.matToImagePlus("tile", mat);
            String filename = String.format("tile_%04d_x%d_y%d_w%d_h%d.tif",
                    idx, region.getX(), region.getY(), region.getWidth(), region.getHeight());
            Path outPath = saveDir.resolve(filename);
            IJ.save(imp, outPath.toString());
            logger.debug("Saved preprocessed tile to: {}", outPath);
        } catch (Exception e) {
            logger.warn("Failed to save preprocessed tile {}: {}", idx, e.getMessage());
        }
    }

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
                roi -> CpSamNormalization.buildNormalizationOp(params.getImageData(), roi, channels,
                        lowPercentile, highPercentile,
                        normalizationDownsample, normalizationMaxDimension));

        mat = preprocessing.apply(mat);

        // Save the normalized tile image before inference if requested (for preprocessing diagnostics).
        if (saveDir != null) {
            saveTile(mat, params.getRegionRequest());
        }

        // Log baseline VRAM once, before the very first inference call of this run.
        if (verboseLogging && nTilesProcessed.get() == 0) {
            CpSamUtils.logVramUsage("before-first-tile");
        }

        Predictor<NDList, NDList> predictor = null;
        try {
            predictor = predictors.take();

            // Use a sub-manager scoped to this tile so all intermediate GPU tensors
            // (batchInput, backbone activations, etc.) are released immediately after
            // predict() returns — before the next tile starts.  Without this, the
            // per-run ndManager holds every tile's GPU allocations simultaneously,
            // causing VRAM to grow linearly with tile count.
            //
            // We copy the mask to a Java float[] inside the scope, then reconstruct
            // a fresh NDArray under ndManager.  This avoids relying on attach() to
            // rescue a tensor from a closing sub-manager, which is unreliable in this
            // DJL version.
            float[] maskData;
            long[] maskShape;
            try (NDManager tileManager = ndManager.newSubManager()) {
                NDArray batchInput = CpSamUtils.matToBatchInput(mat, tileManager);

                if (verboseLogging) {
                    logger.info("Tile model input BCHW: shape={}, dtype={}", batchInput.getShape(), batchInput.getDataType());
                    CpSamUtils.logPerChannelMinMax("Tile normalized BCHW", batchInput.squeeze(0));
                }

                if (verboseLogging) {
                    logger.info("Tile scalar params: diameter={}, cellprob={}, flow={}, niter={}, batchSize={}",
                            diameterTensor.getFloat(), cellprobTensor.getFloat(), flowTensor.getFloat(), niterTensor.getLong(), batchSizeTensor.getInt());
                }

                batchInput.setName("img");
                diameterTensor.setName("diameter");
                cellprobTensor.setName("cellprob_threshold");
                flowTensor.setName("flow_threshold");
                niterTensor.setName("niter");
                batchSizeTensor.setName("batch_size");
                
                NDList modelInput = new NDList(
                    batchInput,
                    diameterTensor,
                    cellprobTensor,
                    flowTensor,
                    niterTensor,
                    batchSizeTensor
                );

                NDList output = predictor.predict(modelInput);

                // Model returns a dict: {"masks": [B,H,W], "flows": [B,2,H,W], "cellprob": [B,H,W]}.
                // DJL names each NDArray in the output NDList with the dict key.
                // Extract only the masks tensor; flows and cellprob are not needed here.
                NDArray rawMasks = output.get("masks");
                if (rawMasks == null) rawMasks = output.get(0); // fallback if name not set
                try (NDArray maskGpu = rawMasks.squeeze(0).toType(DataType.FLOAT32, true)) {
                    output.close();
                    maskData  = maskGpu.toFloatArray();
                    maskShape = maskGpu.getShape().getShape();
                }
            } // tileManager.close(): frees batchInput and all GPU tensors for this tile

            // Log VRAM after each tile to verify GPU memory is freed between tiles.
            if (verboseLogging) {
                CpSamUtils.logVramUsage("after-tile-" + nTilesProcessed.get());
            }

            // Reconstruct the mask as a plain CPU NDArray owned by the run-level ndManager.
            // MaskToObjectConverter calls DjlTools.ndArrayToMat() which only needs the data.
            NDArray maskND = ndManager.create(maskData, new Shape(maskShape));

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


}
