package qupath.ext.cpsam;

import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
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
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ColorTransforms;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathCellObject;
import qupath.lib.objects.PathDetectionObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.utils.ObjectMerger;
import qupath.lib.objects.utils.ObjectProcessor;
import qupath.lib.objects.utils.OverlapFixer;
import qupath.lib.objects.utils.Tiler;
import qupath.lib.plugins.TaskRunner;
import qupath.lib.plugins.TaskRunnerUtils;
import qupath.opencv.ops.ImageOps;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
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
    private final Class<? extends PathObject> preferredOutputType;
    private final TaskRunner taskRunner;
    private final CpSamModel model;

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
        this.preferredOutputType = builder.preferredOutputType;
        this.taskRunner = builder.taskRunner;
        this.model = builder.model;
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
            logger.info("CPSAM run starting: modelPath={}, device={}, tileDims={}, padding={}, diameter={}, cellprobThreshold={}, flowThreshold={}, niter={}, batchSize={}, selectedObjects={}",
                modelPath, device, tileDims, padding, diameter, cellprobThreshold, flowThreshold, niter, batchSize, pathObjects.size());
        }

        // Get the downsample
        double effectiveDownsample;
        if (this.downsample > 0) {
            effectiveDownsample = this.downsample;
        } else {
            // Diameter is in full-resolution pixel units; the TorchScript model handles
            // internal scaling based on diameter. Always process at full resolution.
            effectiveDownsample = 1.0;
            logger.debug("Defaulting to downsample 1.0 (diameter is in pixel units)");
        }

        if (verboseLogging) {
            logger.info("CPSAM effective downsample: {}", effectiveDownsample);
        }

        // Create NDManager for the device
        NDManager ndManager = NDManager.newBaseManager(Device.fromName(this.device));

        try (ndManager) {
            var inputChannels = getInputChannels(imageData);

            // Create the model loading criteria
            var criteria = Criteria.builder()
                    .setTypes(NDList.class, NDList.class)
                    .optModelUrls(modelPath.toUri().toString())
                    .optProgress(new ProgressBar())
                    .optDevice(Device.fromName(this.device))
                    .build();

            try (var loadedModel = criteria.loadModel()) {
                BlockingQueue<Predictor<NDList, NDList>> predictors = new ArrayBlockingQueue<>(numPredictors);
                for (int i = 0; i < numPredictors; i++) {
                    predictors.add(loadedModel.newPredictor());
                }

                // Create tiler
                int sizeWithoutPadding = (int) Math.round(effectiveDownsample * (tileDims - (double) padding * 2));
                Tiler tiler = Tiler.builder(Math.max(256, sizeWithoutPadding))
                        .alignCenter()
                        .cropTiles(false)
                        .build();

                if (verboseLogging) {
                    logger.info("CPSAM tiler configured: tileSizeWithoutPadding={}, finalTileSize={}, fullResPadding={}",
                        sizeWithoutPadding, Math.max(256, sizeWithoutPadding), (int) Math.round(padding * effectiveDownsample));
                }

                // Create processor
                Processor<Mat, Mat, NDArray[]> processor = new CpSamTileProcessor(
                        predictors, inputChannels, ndManager,
                        diameter, (float) cellprobThreshold, (float) flowThreshold, niter, batchSize);

                // Create output handler
                OutputHandler<Mat, Mat, NDArray[]> outputHandler =
                        OutputHandler.createObjectOutputHandler(new MaskToObjectConverter(preferredOutputType));

                // Create post-processor (merge overlapping objects across tiles)
                ObjectProcessor postProcessor = ObjectMerger.createSharedTileBoundaryMerger(0.5)
                        .andThen(OverlapFixer.builder()
                                .clipOverlaps()
                                .keepFragments(false)
                                .sortBySolidity()
                                .build());

                // Create image supplier using ImageOps (same as InstanSeg)
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
                        .padding((int) Math.round(padding * effectiveDownsample))
                        .postProcess(postProcessor)
                        .downsample(effectiveDownsample)
                        .build();

                pixelProcessor.processObjects(taskRunner, imageData, pathObjects);

                int nObjects = pathObjects.stream().mapToInt(PathObject::nChildObjects).sum();
                int nTiles = ((CpSamTileProcessor) processor).getTilesProcessedCount();
                int nFailed = ((CpSamTileProcessor) processor).getTilesFailedCount();
                long nPixels = ((CpSamTileProcessor) processor).getPixelsProcessedCount();
                boolean interrupted = ((CpSamTileProcessor) processor).wasInterrupted();

                if (verboseLogging) {
                    logger.info("CPSAM run finished: tilesProcessed={}, tilesFailed={}, pixelsProcessed={}, outputObjects={}, interrupted={}, elapsedMs={}",
                            nTiles, nFailed, nPixels, nObjects, interrupted, System.currentTimeMillis() - startTime);
                }

                return new CpSamResults(nPixels, nTiles, nFailed, nObjects,
                        System.currentTimeMillis() - startTime, interrupted);
            }
        } catch (Exception e) {
            logger.error("Error running CPSAM detection", e);
            return new CpSamResults(0, 0, 0, 0,
                    System.currentTimeMillis() - startTime, e instanceof InterruptedException);
        }
    }

    private List<ColorTransforms.ColorTransform> getInputChannels(ImageData<BufferedImage> imageData) {
        List<ColorTransforms.ColorTransform> channels = new ArrayList<>();
        for (int i = 0; i < imageData.getServer().nChannels(); i++) {
            channels.add(ColorTransforms.createChannelExtractor(i));
        }
        return channels;
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
        private int padding = 80;
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

        public CpSam build() {
            return new CpSam(this);
        }
    }
}
