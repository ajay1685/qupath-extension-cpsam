package qupath.ext.cpsam;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslateException;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.djl.DjlTools;
import qupath.lib.experimental.pixels.Parameters;
import qupath.lib.experimental.pixels.Processor;

import java.io.IOException;
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
    private final double lowPercentile;
    private final double highPercentile;
    private final int preferredTileDims;

    // Model scalar parameters — packaged into the NDList on every tile call
    private final float diameter;
    private final float cellprobThreshold;
    private final float flowThreshold;
    private final int niter;
    private final int batchSize;

    // NDManager for creating NDArrays — must outlive all tiles
    private final NDManager ndManager;

    private final AtomicLong nPixelsProcessed = new AtomicLong(0);
    private final AtomicInteger nTilesProcessed = new AtomicInteger(0);
    private final AtomicInteger nTilesFailed = new AtomicInteger(0);
    private final AtomicBoolean wasInterrupted = new AtomicBoolean(false);

    CpSamTileProcessor(BlockingQueue<Predictor<NDList, NDList>> predictors,
                       int preferredTileDims, NDManager ndManager,
                       double diameter, float cellprobThreshold, float flowThreshold,
                       int niter, int batchSize) {
        this.predictors = predictors;
        this.lowPercentile = 1.0;
        this.highPercentile = 99.0;
        this.preferredTileDims = preferredTileDims;
        this.ndManager = ndManager;
        this.diameter = (float) diameter;
        this.cellprobThreshold = cellprobThreshold;
        this.flowThreshold = flowThreshold;
        this.niter = niter;
        this.batchSize = batchSize;
    }

    public int getTilesProcessedCount() { return nTilesProcessed.get(); }
    public int getTilesFailedCount() { return nTilesFailed.get(); }
    public long getPixelsProcessedCount() { return nPixelsProcessed.get(); }
    public boolean wasInterrupted() { return wasInterrupted.get(); }

    @Override
    public NDArray[] process(Parameters<Mat, Mat> params) throws IOException {
        NDManager tileManager = ndManager;
        Mat mat = params.getImage();
        boolean verboseLogging = CpSamPreferences.verboseLoggingProperty().get();

        if (verboseLogging) {
            logger.info("Tile start: region={}, mat={}x{}, channels={}, type={}",
                    params.getRegionRequest(), mat.cols(), mat.rows(), mat.channels(), mat.type());
        }

        // Convert raw tile pixels (uint8 [0,255]) to NDArray [C,H,W] float32.
        // A single per-channel p1/p99 normalization is applied below — matching
        // the Groovy reference script exactly. No global pre-pass is needed.
        NDArray imgND = matToNDArrayCHW(mat, tileManager);

        if (verboseLogging) {
            logger.info("Tile NDArray CHW: shape={}, dtype={}", imgND.getShape(), imgND.getDataType());
            logPerChannelMinMax("Tile NDArray CHW", imgND);
        }

        NDArray normalized = perChannelNormalize(imgND, verboseLogging);

        // Step 2: Expand dims [C, H, W] → [1, C, H, W] for model input
        NDArray batchInput = normalized.expandDims(0);

        if (verboseLogging) {
            logger.info("Tile model input BCHW: shape={}, dtype={}", batchInput.getShape(), batchInput.getDataType());
            logPerChannelMinMax("Tile normalized CHW", normalized);
        }

        // Step 3: Build NDList (image + scalar params) and call model — mirrors the working Groovy script exactly.
        // Using raw NDList→NDList avoids the translator which caused TorchScript copy_ failures.
        NDArray diameterTensor     = tileManager.create(diameter);
        NDArray cellprobTensor     = tileManager.create(cellprobThreshold);
        NDArray flowTensor         = tileManager.create(flowThreshold);
        NDArray niterTensor        = tileManager.create((long) niter);   // int64
        NDArray batchSizeTensor    = tileManager.create(batchSize);       // int32
        NDList modelInput = new NDList(batchInput, diameterTensor, cellprobTensor, flowTensor, niterTensor, batchSizeTensor);

        if (verboseLogging) {
            logger.info("Tile model input BCHW: shape={}, dtype={}", batchInput.getShape(), batchInput.getDataType());
            logger.info("Tile scalar params: diameter={}, cellprob={}, flow={}, niter={}, batchSize={}",
                    diameter, cellprobThreshold, flowThreshold, niter, batchSize);
            logPerChannelMinMax("Tile normalized CHW", normalized);
        }

        Predictor<NDList, NDList> predictor = null;
        try {
            predictor = predictors.take();
            NDList output = predictor.predict(modelInput);
            NDArray maskND = output.get(0).squeeze(0).toType(DataType.FLOAT32, true); // [B,H,W] → [H,W] float32

            if (verboseLogging) {
                logger.info("Tile model output: shape={}, dtype={}, min={}, max={}",
                        maskND.getShape(), maskND.getDataType(), maskND.min().getFloat(), maskND.max().getFloat());
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
     * Convert OpenCV Mat (HWC format) to DJL NDArray [C, H, W] float32.
     * Values are preserved as-is (uint8 images stay in [0, 255] range).
     * perChannelNormalize() is responsible for p1/p99 scaling to [0, 1].
     *
     * NOTE: DjlTools.matToNDArray with layout "CHW" routes through opencv_dnn.blobFromImage()
     * which always outputs float32, but the NDArray is created with the original mat's datatype
     * (uint8) — causing a buffer type mismatch crash. Using "HWC" avoids that path and exactly
     * mirrors the Groovy script: ndImg.transpose(2,0,1).toType(FLOAT32, false).
     */
    private static NDArray matToNDArrayCHW(Mat mat, NDManager manager) {
        // "HWC" uses the simple raw-buffer path (no blobFromImage), preserving the original dtype.
        // Transpose (2,0,1): [H, W, C] → [C, H, W], then cast to float32.
        return DjlTools.matToNDArray(manager, mat, "HWC").transpose(2, 0, 1).toType(DataType.FLOAT32, false);
    }

    /**
     * Per-channel percentile normalization (1%-99% clip + scale to [0,1]).
     */
    private static NDArray perChannelNormalize(NDArray img, boolean verboseLogging) {
        int c = (int) img.getShape().get(0);
        NDArray[] channels = new NDArray[c];

        for (int ch = 0; ch < c; ch++) {
            NDArray channel = img.get(ch);
            float[] values = channel.toFloatArray();

            float[] sorted = values.clone();
            java.util.Arrays.sort(sorted);
            int n = sorted.length;
            float lo = sorted[Math.max(0, (int) (1.0 / 100.0 * n))];
            float hi = sorted[Math.min(n - 1, (int) (99.0 / 100.0 * n))];
            float range = hi - lo;

            NDArray normCh;
            if (range > 1e-3f) {
                // Normalize: (x - lo) / range  — matches Groovy: chan.sub(x01).div(range)
                normCh = channel.sub(lo).div(range);
            } else {
                // Flat/empty channel → zero out (matches Groovy's else branch)
                normCh = channel.zerosLike();
            }
            channels[ch] = normCh;

            if (verboseLogging) {
                float min = channels[ch].min().getFloat();
                float max = channels[ch].max().getFloat();
                logger.info("Tile channel {} normalization: p1={}, p99={}, range={}, outMin={}, outMax={}",
                        ch, lo, hi, range, min, max);
            }
        }

        return NDArrays.stack(new NDList(channels)).reshape(new Shape(c, img.getShape().get(1), img.getShape().get(2)));
    }

    private static void logPerChannelMinMax(String label, NDArray chw) {
        int channels = (int) chw.getShape().get(0);
        for (int ch = 0; ch < channels; ch++) {
            NDArray channel = chw.get(ch);
            logger.info("{} channel {}: min={}, max={}", label, ch, channel.min().getFloat(), channel.max().getFloat());
        }
    }
}
