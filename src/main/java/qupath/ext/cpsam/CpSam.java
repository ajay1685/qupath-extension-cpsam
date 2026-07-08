package qupath.ext.cpsam;

import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.experimental.pixels.OutputHandler;
import qupath.lib.experimental.pixels.PixelProcessor;
import qupath.lib.experimental.pixels.Processor;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ColorTransforms;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathCellObject;
import qupath.lib.objects.PathDetectionObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.utils.ObjectProcessor;
import qupath.lib.objects.utils.Tiler;
import qupath.lib.plugins.TaskRunner;
import qupath.lib.plugins.TaskRunnerUtils;
import qupath.opencv.ops.ImageOps;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * Core orchestrator for CPSAM detection.
 * Mirrors InstanSeg.java but simplified since the TorchScript model handles all internal CellposeSAM logic.
 */
public class CpSam {

    private static final Logger logger = LoggerFactory.getLogger(CpSam.class);

    // Cached model + predictor pool: reused across runs when path/device/numPredictors are unchanged.
    // This eliminates both the ~900ms model-reload cost and CUDA kernel JIT warmup on every run.
    private static ZooModel<NDList, NDList> cachedModel = null;
    private static BlockingQueue<Predictor<NDList, NDList>> cachedPredictors = null;
    private static String cachedModelPath = null;
    private static String cachedDevice = null;
    private static int cachedNumPredictors = 0;

    /**
     * Returns the cached predictor queue, reloading the model and/or recreating predictors only
     * when the model path, device, or requested predictor count has changed.
     */
    private static synchronized BlockingQueue<Predictor<NDList, NDList>> getOrLoadPredictors(
            Path modelPath, String device, int numPredictors) throws Exception {
        String pathStr = modelPath.toString();
        boolean modelSame = pathStr.equals(cachedModelPath) && device.equals(cachedDevice);
        if (modelSame && numPredictors == cachedNumPredictors && cachedPredictors != null) {
            logger.debug("CPSAM reusing {} cached predictor(s) for path={}, device={}", numPredictors, pathStr, device);
            return cachedPredictors;
        }
        // Close old predictors before recreating or closing the model
        closeCachedPredictors();
        if (!modelSame && cachedModel != null) {
            logger.debug("CPSAM model path or device changed — closing cached model");
            try { cachedModel.close(); } catch (Exception ex) { logger.warn("Error closing cached model", ex); }
            cachedModel = null;
        }
        if (cachedModel == null) {
            logger.info("CPSAM loading model: path={}, device={}", pathStr, device);
            var criteria = Criteria.builder()
                    .setTypes(NDList.class, NDList.class)
                    .optModelUrls(modelPath.toUri().toString())
                    .optProgress(new ProgressBar())
                    .optDevice(Device.fromName(device))
                    .build();
            cachedModel = criteria.loadModel();
            cachedModelPath = pathStr;
            cachedDevice = device;
        }
        logger.info("CPSAM creating {} predictor(s) for device={}", numPredictors, device);
        cachedPredictors = new ArrayBlockingQueue<>(numPredictors);
        for (int i = 0; i < numPredictors; i++) {
            cachedPredictors.add(cachedModel.newPredictor());
        }
        cachedNumPredictors = numPredictors;
        return cachedPredictors;
    }

    private static void closeCachedPredictors() {
        if (cachedPredictors != null) {
            List<Predictor<NDList, NDList>> toClose = new ArrayList<>();
            cachedPredictors.drainTo(toClose);
            for (var p : toClose) {
                try { p.close(); } catch (Exception ex) { logger.warn("Error closing predictor", ex); }
            }
            cachedPredictors = null;
            cachedNumPredictors = 0;
        }
    }

    /** Close the cached model and predictors, e.g. when the panel is closed. */
    public static synchronized void clearModelCache() {
        closeCachedPredictors();
        if (cachedModel != null) {
            try { cachedModel.close(); } catch (Exception ex) { logger.warn("Error closing model cache", ex); }
            cachedModel = null;
            cachedModelPath = null;
            cachedDevice = null;
        }
        //CpSamUtils.emptyCudaCache();
    }

    private final int tileDims;
    private final double downsample;
    private final int padding;
    private final double diameter;
    private final float cellprobThreshold;
    private final float flowThreshold;
    private final int niter;
    private final int batchSize;
    private final int numPredictors;
    private final String device;
    private final List<ColorTransforms.ColorTransform> channels;
    private final Class<? extends PathObject> preferredOutputType;
    private final TaskRunner taskRunner;
    private final CpSamModel model;
    private final double normalizationDownsample;
    private final int normalizationMaxDimension;
    private final double normalizationLowPercentile;
    private final double normalizationHighPercentile;
    private final boolean measureShape; 
    private final boolean measureIntensity; 

    private CpSam(Builder builder) {
        this.tileDims = builder.tileDims;
        this.downsample = builder.downsample;
        this.padding = builder.padding;
        this.diameter = builder.diameter;
        this.cellprobThreshold = builder.cellprobThreshold;
        this.flowThreshold = builder.flowThreshold;
        this.niter = builder.niter;
        this.batchSize = builder.batchSize;
        this.numPredictors = builder.numPredictors;
        this.device = builder.device;
        this.channels = builder.channels;
        this.preferredOutputType = builder.preferredOutputType;
        this.taskRunner = builder.taskRunner;
        this.model = builder.model;
        this.normalizationDownsample = builder.normalizationDownsample;
        this.normalizationMaxDimension = builder.normalizationMaxDimension;
        this.normalizationLowPercentile = builder.normalizationLowPercentile;
        this.normalizationHighPercentile = builder.normalizationHighPercentile;
        this.measureShape = builder.measureShape;
        this.measureIntensity = builder.measureIntensity;
    }

    /**
     * Run detection on the currently selected PathObjects in the current image.
     */
    public CpSamResults detectObjects() {
        return detectObjects(qupath.lib.scripting.QP.getCurrentImageData());
    }

    /**
     * Run detection on the currently selected PathObjects in the specified image.
     */
    public CpSamResults detectObjects(ImageData<BufferedImage> imageData) {
        Objects.requireNonNull(imageData, "No imageData available");
        return detectObjects(imageData, imageData.getHierarchy().getSelectionModel().getSelectedObjects());
    }

    /**
     * Run detection for a collection of PathObjects from the current image.
     */
    public CpSamResults detectObjects(Collection<? extends PathObject> pathObjects) {
        var imageData = qupath.lib.scripting.QP.getCurrentImageData();
        var results = runCpSam(imageData, pathObjects);
        return results;
    }

    /**
     * Run detection for a collection of PathObjects associated with the specified image.
     */
    public CpSamResults detectObjects(ImageData<BufferedImage> imageData,
                                       Collection<? extends PathObject> pathObjects) {
        Objects.requireNonNull(imageData, "No imageData available");
        Objects.requireNonNull(pathObjects, "No objects available");
        return runCpSam(imageData, pathObjects);
    }

    /**
     * Create a builder for CpSam.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for CpSam.
     */
    public static final class Builder {

        private static final int MIN_TILE_DIMS = 256;
        private static final int MAX_TILE_DIMS = 4096;

        private int tileDims = 512;
        private double downsample = -1;
        private int padding = 60;
        private double diameter = 30.0;
        private float cellprobThreshold = 0.0f;
        private float flowThreshold = 0.4f;
        private int niter = 200;
        private int batchSize = 1;
        private int numPredictors = Integer.getInteger("cpsam.numPredictors", 1);
        private String device = "cpu";
        private Class<? extends PathObject> preferredOutputType = PathDetectionObject.class;
        private TaskRunner taskRunner = TaskRunnerUtils.getDefaultInstance().createTaskRunner();
        private CpSamModel model;
        private List<ColorTransforms.ColorTransform> channels = null;
        private double normalizationDownsample = Double.NaN; // NaN → derived from normalizationMaxDimension
        private int normalizationMaxDimension = 2048;
        private double normalizationLowPercentile = 1.0;
        private double normalizationHighPercentile = 99.0;
        private boolean measureShape = false;
        private boolean measureIntensity = false;

        Builder() {}

        public Builder tileDims(int tileDims) {
            if (tileDims < MIN_TILE_DIMS) {
                logger.warn("Tile dimensions too small, setting to minimum value of {}", MIN_TILE_DIMS);
                this.tileDims = MIN_TILE_DIMS;
            } else if (tileDims > MAX_TILE_DIMS) {
                logger.warn("Tile dimensions too large, setting to maximum value of {}", MAX_TILE_DIMS);
                this.tileDims = MAX_TILE_DIMS;
            } else {
                this.tileDims = tileDims;
            }
            return this;
        }

        public Builder downsample(double downsample) {
            this.downsample = downsample;
            return this;
        }

        public Builder interTilePadding(int padding) {
            if (padding < 0) {
                logger.warn("Padding cannot be negative, setting to 0");
                this.padding = 0;
            } else {
                this.padding = padding;
            }
            return this;
        }

        public Builder diameter(double diameter) {
            this.diameter = diameter;
            return this;
        }

        public Builder cellprobThreshold(float cellprobThreshold) {
            this.cellprobThreshold = cellprobThreshold;
            return this;
        }

        public Builder flowThreshold(float flowThreshold) {
            this.flowThreshold = flowThreshold;
            return this;
        }

        public Builder niter(int niter) {
            this.niter = niter;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder numPredictors(int numPredictors) {
            this.numPredictors = Math.max(1, numPredictors);
            return this;
        }

        public Builder device(String device) {
            this.device = device;
            return this;
        }

        public Builder taskRunner(TaskRunner taskRunner) {
            this.taskRunner = taskRunner;
            return this;
        }

        public Builder model(CpSamModel model) {
            this.model = model;
            return this;
        }

        /**
         * Set the model path.
         */
        public Builder modelPath(Path path) throws IOException {
            return model(CpSamModel.fromPath(path));
        }

        /**
         * Set the model path from a string.
         */
        public Builder modelPath(String path) throws IOException {
            return model(CpSamModel.fromPath(java.nio.file.Path.of(path)));
        }

        public Builder preferredOutputType(Class<? extends PathObject> preferredOutputType) {
            this.preferredOutputType = preferredOutputType;
            return this;
        }

        public Builder outputCells() {
            return preferredOutputType(PathCellObject.class);
        }

        public Builder outputDetections() {
            return preferredOutputType(PathDetectionObject.class);
        }

        public Builder outputAnnotations() {
            return preferredOutputType(PathAnnotationObject.class);
        }

        public Builder nThreads(int nThreads) {
            this.taskRunner = TaskRunnerUtils.getDefaultInstance().createTaskRunner(nThreads);
            this.numPredictors = Math.max(1, nThreads);
            return this;
        }

        /**
         * Set which image channels to use as model inputs (in order).
         * If not set, the first available channels of the image are used automatically.
         * Fewer than 3 channels will be zero-padded; more than 3 will be ignored.
         */
        public Builder inputChannels(List<ColorTransforms.ColorTransform> channels) {
            this.channels = channels == null ? null : List.copyOf(channels);
            return this;
        }

        /**
         * Set an explicit downsample factor used when reading the annotation region for
         * computing global normalization percentiles. When set, {@link #normalizationMaxDimension}
         * is ignored. Use this when you want reproducible normalization regardless of annotation size.
         * E.g. {@code .normalizationDownsample(4.0)} reads at 1/4 full resolution.
         */
        public Builder normalizationDownsample(double downsample) {
            this.normalizationDownsample = downsample;
            return this;
        }

        /**
         * Set the maximum image dimension (pixels) used when auto-computing the downsample factor
         * for global normalization. The downsample is chosen so the larger dimension of the
         * annotation bounding box does not exceed this value. Default is 2048.
         */
        public Builder normalizationMaxDimension(int maxDimension) {
            this.normalizationMaxDimension = Math.max(4096, maxDimension);
            if (Double.isNaN(this.normalizationDownsample)) {
                // nothing — already NaN, maxDimension will be read in build()
            }
            return this;
        }

        /**
         * Set the low and high percentile values used for global per-channel normalization.
         * Defaults are 1.0 (low) and 99.0 (high).
         */
        public Builder normalizationPercentiles(double low, double high) {
            if (low < 0 || low >= high || high > 100)
                throw new IllegalArgumentException("Percentiles must satisfy 0 <= low < high <= 100, got " + low + ", " + high);
            this.normalizationLowPercentile = low;
            this.normalizationHighPercentile = high;
            return this;
        }

        /**
         * Enable or disable shape measurements after detection.
         * Shape features include area, length, circularity, solidity, min/max diameter.
         */
        public Builder measureShape(boolean measureShape) {
            this.measureShape = measureShape;
            return this;
        }

        /**
         * Enable or disable per-channel intensity measurements after detection.
         * Uses the batched API for efficiency on large object sets.
         */
        public Builder measureIntensity(boolean measureIntensity) {
            this.measureIntensity = measureIntensity;
            return this;
        }

        public CpSam build() {
            return new CpSam(this);
        }
    }


    /**
     * Run CPSAM detection on the specified image and PathObjects.
     * Returns a CpSamResults object containing counts and timing information.
     */
    private CpSamResults runCpSam(ImageData<BufferedImage> imageData,
                                        Collection<? extends PathObject> pathObjects) {
            long startTime = System.currentTimeMillis();
            boolean verboseLogging = CpSamPreferences.verboseLoggingProperty().get();

            if (!model.isValid()) {
                return new CpSamResults(0, 0, 0, 0,
                        System.currentTimeMillis() - startTime, false);
            }

            Path modelPath = model.getModelPath();

            if (verboseLogging) {
                logger.info("CPSAM model path: {}", modelPath);
                logger.info("CPSAM inputs: device={}, numPredictors={}, tileDims={}, padding={}, diameter={}, cellprobThreshold={}, flowThreshold={}, niter={}, batchSize={}, selectedObjects={}",
                    device, numPredictors, tileDims, padding, diameter, cellprobThreshold, flowThreshold, niter, batchSize, pathObjects.size());
            }

            // Get the downsample
            double effectiveDownsample;
            if (this.downsample > 0) {
                effectiveDownsample = this.downsample;
            } else {
                effectiveDownsample = 1.0;
                logger.debug("Defaulting to downsample 1.0 (diameter is in pixel units)");
            }

            if (verboseLogging) {
                logger.info("CPSAM effective downsample: {}", effectiveDownsample);
            }

            // Create NDManager for the device
            NDManager ndManager = NDManager.newBaseManager(Device.fromName(this.device));

            try {
                try (ndManager) {
                    var inputChannels = (this.channels != null && !this.channels.isEmpty())
                            ? this.channels
                            : CpSamPreProcessing.getInputChannels(imageData);

                    // Get (or load) the cached predictor pool — model + predictors survive across runs
                    BlockingQueue<Predictor<NDList, NDList>> predictors =
                            getOrLoadPredictors(modelPath, this.device, numPredictors);
                    
                    // inner scope — no predictor closing here; pool is reused next run
                    {   
                        try {
                            // Tiling strategy: controls step size and overlap between adjacent tiles.
                            CpSamTiling tilingConfig = new CpSamTiling(tileDims, padding);
                            Tiler tiler = tilingConfig.createTiler(effectiveDownsample);

                            // Create tile save directory if requested (preference enabled at run start).
                            Path saveDir = null;
                            if (CpSamPreferences.savePreprocessedTilesProperty().get()) {
                                saveDir = CpSamTileSaveDir.create(imageData);
                                CpSamTileSaveDir.clearTempFiles(saveDir);
                                CpSamTileSaveDir.resetTileIndex();
                            }

                            // Create processor
                            Processor<Mat, Mat, NDArray[]> processor = new CpSamTileProcessor(
                                    predictors, inputChannels, ndManager,
                                    diameter, (float) cellprobThreshold, (float) flowThreshold, niter, batchSize,
                                    normalizationDownsample, normalizationMaxDimension,
                                    normalizationLowPercentile, normalizationHighPercentile,
                                    saveDir);

                            // Post-processing strategy: output handler (per-tile) + merger (across all tiles).
                            CpSamPostProcessing postProcessingConfig = new CpSamPostProcessing(preferredOutputType);
                            OutputHandler<Mat, Mat, NDArray[]> outputHandler = postProcessingConfig.createOutputHandler();
                            ObjectProcessor postProcessor = postProcessingConfig.createPostProcessor();

                            // Extracts channels from the image data
                            var imageSupplier = (qupath.lib.experimental.pixels.ImageSupplier<Mat>) params ->
                                    ImageOps.buildImageDataOp(inputChannels)
                                            .apply(params.getImageData(), params.getRegionRequest());

                            // Build and run PixelProcessor
                            var pixelProcessor = PixelProcessor.<Mat, Mat, NDArray[]>builder()
                                    .processor(processor)
                                    .imageSupplier(imageSupplier)
                                    .tiler(tiler)
                                    .outputHandler(outputHandler)
                                    .padding(tilingConfig.getFullResPadding(effectiveDownsample))
                                    .postProcess(postProcessor)
                                    .downsample(effectiveDownsample)
                                    .build();

                            pixelProcessor.processObjects(taskRunner, imageData, pathObjects);

                            // Post-detection measurements
                            if (measureShape || measureIntensity) {
                                var allDetected = pathObjects.stream()
                                        .flatMap(p -> p.getChildObjects().stream())
                                        .toList();
                                if (!allDetected.isEmpty()) {
                                    if (measureShape) {
                                        CpSamMeasurements.addShapeMeasurements(imageData, allDetected);
                                    }
                                    if (measureIntensity) {
                                        CpSamMeasurements.addIntensityMeasurements(imageData, allDetected,
                                                effectiveDownsample, numPredictors);
                                    }
                                    imageData.getHierarchy().fireObjectMeasurementsChangedEvent(null, allDetected);
                                }
                            }

                            int nObjects = pathObjects.stream().mapToInt(PathObject::nChildObjects).sum();
                            int nTiles = ((CpSamTileProcessor) processor).getTilesProcessedCount();
                            int nFailed = ((CpSamTileProcessor) processor).getTilesFailedCount();
                            long nPixels = ((CpSamTileProcessor) processor).getPixelsProcessedCount();
                            boolean interrupted = ((CpSamTileProcessor) processor).wasInterrupted();

                            long totalElapsedMs = System.currentTimeMillis() - startTime;
                            if (verboseLogging) {
                                logger.info("CPSAM run finished: tilesProcessed={}, tilesFailed={}, pixelsProcessed={}, outputObjects={}, interrupted={}, elapsedMs={} ({} s)",
                                        nTiles, nFailed, nPixels, nObjects, interrupted, totalElapsedMs, String.format("%.2f", totalElapsedMs / 1000.0));
                            }

                            return new CpSamResults(nPixels, nTiles, nFailed, nObjects,
                                    totalElapsedMs, interrupted);
                        } catch (Exception ex) {
                            // If inference failed, evict the predictor cache — CUDA state may be corrupt
                            logger.warn("CPSAM run failed; evicting predictor cache to force fresh state on next run", ex);
                            //clearModelCache();
                            throw ex;
                        }
                    }   // end inner scope — predictors remain cached for next run
                }
            } catch (Exception e) {
                logger.error("Error running CPSAM detection", e);
                return new CpSamResults(0, 0, 0, 0,
                        System.currentTimeMillis() - startTime, e instanceof InterruptedException);
            } finally {
                // The ndManager is guaranteed to be closed at this point.
                // Safe to force PyTorch to clear its caching allocator.
                //CpSamUtils.emptyCudaCache();
                if (verboseLogging) {
                    CpSamUtils.logVramUsage("after-run");
                }
            }
        }
    
}
