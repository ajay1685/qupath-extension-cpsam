package qupath.ext.cpsam;

import javafx.concurrent.Task;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.TaskRunnerFX;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.plugins.workflow.DefaultScriptableWorkflowStep;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.ResourceBundle;
import java.util.stream.Collectors;
import qupath.lib.images.servers.ColorTransforms;

public class CpSamTask extends Task<Void> {

    private static final Logger logger = LoggerFactory.getLogger(CpSamTask.class);
    private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.cpsam.ui.strings");

    private final ImageData<BufferedImage> imageData;
    private final String modelPathStr;
    private final String device;
    private final double diameter;
    private final float cellprobThreshold;
    private final float flowThreshold;
    private final int niter;
    private final int batchSize;
    private final int tileSize;
    private final int tilePadding;
    private final Class<? extends PathObject> preferredOutputType;
    private final List<CpSamChannelItem> channelItems;

    public CpSamTask(ImageData<BufferedImage> imageData, String modelPathStr,
              String device, double diameter, float cellprobThreshold, float flowThreshold,
              int niter, int batchSize, int tileSize, int tilePadding,
              Class<? extends PathObject> preferredOutputType, List<CpSamChannelItem> channelItems) {
        this.imageData = imageData;
        this.modelPathStr = modelPathStr;
        this.device = device;
        this.diameter = diameter;
        this.cellprobThreshold = cellprobThreshold;
        this.flowThreshold = flowThreshold;
        this.niter = niter;
        this.batchSize = batchSize;
        this.tileSize = tileSize;
        this.tilePadding = tilePadding;
        this.preferredOutputType = preferredOutputType;
        this.channelItems = List.copyOf(channelItems);
    }

    /**
     * Build ColorTransform objects from the selected channel items.
     * Handles both raw channels and color deconvolution channels.
     */
    private List<ColorTransforms.ColorTransform> buildChannelTransforms() {
        if (channelItems.isEmpty()) return List.of();
        var transforms = new ArrayList<ColorTransforms.ColorTransform>();
        for (CpSamChannelItem item : channelItems) {
            var transform = item.getTransform();
            if (transform != null) {
                transforms.add(transform);
            }
        }
        return transforms;
    }

    @Override
    protected Void call() {
        int nThreads = CpSamPreferences.numThreadsProperty().get();
        double normLow = CpSamPreferences.normLowProperty().get();
        double normHigh = CpSamPreferences.normHighProperty().get();
        boolean measureShape = CpSamPreferences.measureShapeProperty().get();
        boolean measureIntensity = CpSamPreferences.measureIntensityProperty().get();
        var taskRunner = new TaskRunnerFX(QuPathGUI.getInstance(), nThreads);

        var selectedObjects = imageData.getHierarchy().getSelectionModel().getSelectedObjects();

        CpSamResults results;
        try {
            Path modelPath = java.nio.file.Path.of(modelPathStr);
            if (!java.nio.file.Files.exists(modelPath)) {
                Dialogs.showErrorNotification("CPSAM", "Model path does not exist: " + modelPath);
                return null;
            }

            results = CpSam.builder()
                    .modelPath(modelPath)
                    .device(device)
                    .diameter(diameter)
                    .cellprobThreshold(cellprobThreshold)
                    .flowThreshold(flowThreshold)
                    .niter(niter)
                    .batchSize(batchSize)
                    .numPredictors(nThreads)
                    .tileDims(tileSize)
                    .interTilePadding(tilePadding)
                    .inputChannels(buildChannelTransforms())
                    .normalizationPercentiles(normLow, normHigh)
                    .measureShape(measureShape)
                    .measureIntensity(measureIntensity)
                    .taskRunner(taskRunner)
                    .preferredOutputType(preferredOutputType)
                    .build()
                    .detectObjects(imageData, selectedObjects);
        } catch (Exception e) {
            logger.error("Error running CPSAM", e);
            Dialogs.showErrorNotification("CPSAM", "Error running segmentation: " + e.getMessage());
            results = new CpSamResults(0, 0, 0, 0, 0, false);
        }

        imageData.getHierarchy().fireHierarchyChangedEvent(this);
        String channelLine;
        if (channelItems.isEmpty()) {
            channelLine = "";
        } else {
            channelLine = "\n    .inputChannels(java.util.List.of(" +
                    channelItems.stream()
                            .map(item -> {
                                if (item.getStainNumber() > 0) {
                                    return String.format("qupath.lib.images.servers.ColorTransforms.createColorDeconvolvedChannel(getCurrentImageData().getColorDeconvolutionStains(), %d)", item.getStainNumber());
                                } else {
                                    return String.format("qupath.lib.images.servers.ColorTransforms.createChannelExtractor(%d)", item.getIndex());
                                }
                            })
                            .collect(Collectors.joining(", ")) +
                    "))";
        }
        imageData.getHistoryWorkflow()
                .addStep(new DefaultScriptableWorkflowStep(
                        "Run CPSAM segmentation",
                        String.format("""
                                qupath.ext.cpsam.CpSam.builder()
                                    .modelPath("%s")
                                    .device("%s")
                                    .diameter(%.1f)
                                    .cellprobThreshold(%.2f)
                                    .flowThreshold(%.2f)
                                    .niter(%d)
                                    .batchSize(%d)
                                    .numPredictors(%d)
                                    .tileDims(%d)
                                    .interTilePadding(%d)%s
                                    .normalizationPercentiles(%.1f, %.1f)
                                    .measureShape(%b)
                                    .measureIntensity(%b)
                                    .outputDetections()
                                    .build()
                                    .detectObjects()
                                """.formatted(
                                modelPathStr.replace("\\", "/"),
                                device, diameter, cellprobThreshold, flowThreshold,
                                niter, batchSize, nThreads, tileSize, tilePadding,
                                channelLine, normLow, normHigh, measureShape, measureIntensity
                        ).strip()
                )));

        logger.info("Results: {}", results);
        int nFailed = results.getTilesFailed();
        if (nFailed > 0 && !results.wasInterrupted()) {
            logger.error("{} tiles failed during segmentation", nFailed);
            Dialogs.showErrorNotification("CPSAM", String.format("%d tiles failed during segmentation", nFailed));
        }
        return null;
    }
}
