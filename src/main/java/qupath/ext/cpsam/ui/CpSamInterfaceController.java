package qupath.ext.cpsam.ui;

import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.beans.binding.BooleanBinding;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.value.ObservableValue;
import javafx.collections.ListChangeListener;
import javafx.concurrent.Worker;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.controlsfx.control.CheckComboBox;
import org.controlsfx.control.SearchableComboBox;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.cpsam.CpSamPreferences;
import qupath.ext.djl.DjlTools;
import qupath.ext.cpsam.CpSamTask;
import qupath.ext.cpsam.CpSamModel;
import qupath.fx.dialogs.Dialogs;
import qupath.fx.dialogs.FileChoosers;
import qupath.fx.utils.FXUtils;
import qupath.lib.common.ThreadTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.images.ImageData;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.ResourceBundle;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.FutureTask;

public class CpSamInterfaceController extends VBox {

    private static final Logger logger = LoggerFactory.getLogger(CpSamInterfaceController.class);
    private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.cpsam.ui.strings");

    @FXML
    private Label modelPathLabel;
    @FXML
    private Button modelPathButton;
    @FXML
    private Spinner<Double> diameterSpinner;
    @FXML
    private Spinner<Double> flowThresholdSpinner;
    @FXML
    private Spinner<Double> cellProbThresholdSpinner;
    @FXML
    private Spinner<Integer> niterSpinner;
    @FXML
    private Spinner<Integer> batchSizeSpinner;
    @FXML
    private ChoiceBox<Integer> tileSizeChoiceBox;
    @FXML
    private ChoiceBox<Integer> tilePaddingChoiceBox;
    @FXML
    private ChoiceBox<String> deviceChoices;
    @FXML
    private Spinner<Integer> threadSpinner;
    @FXML
    private ComboBox<String> comboOutputType;
    @FXML
    private CheckBox randomColorsCheckBox;
    @FXML
    private CheckBox savePreprocessedTilesCheckBox;
    @FXML
    private Button runButton;
    @FXML
    private Label labelMessage;
    @FXML
    private ComboBox<String> channelCombo1;
    @FXML
    private ComboBox<String> channelCombo2;
    @FXML
    private ComboBox<String> channelCombo3;

    private static final String CHANNEL_NONE = "(None)";

    private final ExecutorService pool = Executors.newSingleThreadExecutor(ThreadTools.createThreadFactory("cpsam", true));
    private final QuPathGUI qupath;
    private final ObjectProperty<FutureTask<?>> pendingTask = new SimpleObjectProperty<>();
    private final BooleanProperty pytorchAvailable = new SimpleBooleanProperty(false);

    private static final ObjectProperty<Path> modelPathBinding = new SimpleObjectProperty<>();

    public static CpSamInterfaceController createInstance(QuPathGUI qupath) throws IOException {
        return new CpSamInterfaceController(qupath);
    }

    private CpSamInterfaceController(QuPathGUI qupath) throws IOException {
        this.qupath = qupath;
        FXMLLoader loader = new FXMLLoader(CpSamInterfaceController.class.getResource("CpsamInterface.fxml"));
        loader.setResources(resources);
        loader.setRoot(this);
        loader.setController(this);
        loader.load();

        // Restore persisted model path (only if not already set in this session)
        if (modelPathBinding.get() == null) {
            String savedPath = CpSamPreferences.modelDirectoryProperty().get();
            if (savedPath != null && !savedPath.isBlank()) {
                Path p = Path.of(savedPath);
                if (Files.exists(p)) {
                    modelPathBinding.set(p);
                }
            }
        }

        // Initialize controls
        initSpinners();
        initChoiceBoxes();
        initBindings();
        initPyTorchCheck();
        updateModelPathLabel();
        updateDeviceChoices();
        // Populate channel combos from the current image; repopulate when image changes
        qupath.imageDataProperty().addListener(this::handleImageDataChange);
        handleImageDataChange(qupath.imageDataProperty(), null, qupath.imageDataProperty().get());
    }

    private void initSpinners() {
        // Diameter
        SpinnerValueFactory.DoubleSpinnerValueFactory diamFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(1.0, 200.0, 30.0, 0.5);
        diameterSpinner.setValueFactory(diamFactory);
        diamFactory.valueProperty().addListener((v, o, n) -> {});

        // Flow threshold
        SpinnerValueFactory.DoubleSpinnerValueFactory flowFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0, 10.0, 0.4, 0.05);
        flowThresholdSpinner.setValueFactory(flowFactory);

        // Cell prob threshold
        SpinnerValueFactory.DoubleSpinnerValueFactory probFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(-2.0, 2.0, 0.0, 0.05);
        cellProbThresholdSpinner.setValueFactory(probFactory);

        // Niter
        SpinnerValueFactory.IntegerSpinnerValueFactory niterFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 5000, 200);
        niterSpinner.setValueFactory(niterFactory);

        // Batch size
        SpinnerValueFactory.IntegerSpinnerValueFactory batchFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 128, CpSamPreferences.batchSizeProperty().get());
        batchSizeSpinner.setValueFactory(batchFactory);
        batchFactory.valueProperty().addListener((v, o, n) -> CpSamPreferences.batchSizeProperty().set(n));

        // Threads — load persisted preference, cap at 8 (more concurrent GPU inferences than that
        // always causes VRAM exhaustion and massive slowdown; CPU users can set via script if needed)
        SpinnerValueFactory.IntegerSpinnerValueFactory threadFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 8,
            CpSamPreferences.numThreadsProperty().get());
        threadSpinner.setValueFactory(threadFactory);
        threadFactory.valueProperty().addListener((v, o, n) -> CpSamPreferences.numThreadsProperty().set(n));
    }

    private void initChoiceBoxes() {
        // Tile sizes
        tileSizeChoiceBox.getItems().addAll(256, 512, 1024, 2048, 4096);
        tileSizeChoiceBox.setValue(CpSamPreferences.tileSizeProperty().getValue());
        tileSizeChoiceBox.valueProperty().addListener((v, o, n) -> CpSamPreferences.tileSizeProperty().set(n));

        // Tile padding
        tilePaddingChoiceBox.getItems().addAll(0, 16, 32, 48, 64, 80, 96);
        tilePaddingChoiceBox.setValue(CpSamPreferences.tilePaddingProperty().getValue());
        tilePaddingChoiceBox.valueProperty().addListener((v, o, n) -> CpSamPreferences.tilePaddingProperty().set(n));

        // Output type
        comboOutputType.getItems().addAll("Detections", "Annotations", "Cells");
        comboOutputType.getSelectionModel().select(0);

        // Save device selection whenever it changes (population done in updateDeviceChoices)
        deviceChoices.valueProperty().addListener((v, o, n) -> {
            if (n != null) CpSamPreferences.preferredDeviceProperty().set(n);
        });
    }

    private void initBindings() {
        // Run button bindings
        runButton.disableProperty().bind(
            qupath.imageDataProperty().isNull()
                .or(pendingTask.isNotNull())
                .or(Bindings.createBooleanBinding(() -> modelPathBinding.get() == null, modelPathBinding))
        );

        // Save preprocessed tiles checkbox
        savePreprocessedTilesCheckBox.selectedProperty().bindBidirectional(
                CpSamPreferences.savePreprocessedTilesProperty());

        // Model path label tooltip
        modelPathLabel.setOnMouseClicked(e -> {
            if (e.getClickCount() == 2) {
                promptForModelDirectory();
            }
        });

        // Pending task execution
        pendingTask.addListener((observable, oldValue, newValue) -> {
            if (newValue != null) {
                pool.execute(newValue);
            }
        });
    }

    private void initPyTorchCheck() {
        try {
            Class.forName("ai.djl.pytorch.jni.PyTorchLibrary");
            pytorchAvailable.set(true);
        } catch (ClassNotFoundException e) {
            pytorchAvailable.set(false);
            // Try to download PyTorch
            try {
                ai.djl.engine.Engine.getInstance().getEngine("PyTorch");
                pytorchAvailable.set(true);
            } catch (Exception ex) {
                logger.warn("PyTorch engine not available");
            }
        }

        // Update devices when PyTorch becomes available
        pytorchAvailable.addListener((v, o, n) -> {
            if (n) updateDeviceChoices();
        });
    }

    private void updateDeviceChoices() {
        deviceChoices.getItems().clear();
        deviceChoices.getItems().add("cpu");

        try {
            ai.djl.engine.Engine engine = ai.djl.engine.Engine.getInstance().getEngine("PyTorch");
            if (engine != null) {
                for (var device : engine.getDevices()) {
                    String name = device.getDeviceType();
                    if (device.getDeviceId() != -1) {
                        name += device.getDeviceId();
                    }
                    if (!deviceChoices.getItems().contains(name)) {
                        deviceChoices.getItems().add(name);
                    }
                }
            }
        } catch (Exception e) {
            logger.debug("Could not query PyTorch devices", e);
        }

        // Select last preferred device
        String pref = CpSamPreferences.preferredDeviceProperty().getValue();
        if (deviceChoices.getItems().contains(pref)) {
            deviceChoices.getSelectionModel().select(pref);
        } else {
            deviceChoices.getSelectionModel().selectFirst();
        }
    }

    private void updateModelPathLabel() {
        Path modelPath = modelPathBinding.get();
        if (modelPath != null) {
            modelPathLabel.getStyleClass().removeAll("warning-message");
            modelPathLabel.getStyleClass().add("standard-message");
            modelPathLabel.setCursor(javafx.scene.Cursor.HAND);
            modelPathLabel.setText(modelPath.toString());
        } else {
            modelPathLabel.getStyleClass().removeAll("standard-message");
            modelPathLabel.getStyleClass().add("warning-message");
            modelPathLabel.setText("No model selected");
            modelPathLabel.setCursor(javafx.scene.Cursor.DEFAULT);
        }
    }

    private void handleImageDataChange(ObservableValue<? extends ImageData<BufferedImage>> obs,
                                       ImageData<BufferedImage> oldValue,
                                       ImageData<BufferedImage> newValue) {
        if (!Platform.isFxApplicationThread()) {
            Platform.runLater(() -> handleImageDataChange(obs, oldValue, newValue));
            return;
        }
        updateChannelCombos(newValue);
    }

    private void updateChannelCombos(ImageData<BufferedImage> imageData) {
        // Save current selections before clearing
        String prev1 = channelCombo1.getValue();
        String prev2 = channelCombo2.getValue();
        String prev3 = channelCombo3.getValue();

        channelCombo1.getItems().clear();
        channelCombo2.getItems().clear();
        channelCombo3.getItems().clear();

        if (imageData == null) {
            channelCombo1.setDisable(true);
            channelCombo2.setDisable(true);
            channelCombo3.setDisable(true);
            return;
        }

        List<String> names = new ArrayList<>();
        for (var ch : imageData.getServer().getMetadata().getChannels()) {
            names.add(ch.getName());
        }

        if (names.isEmpty()) {
            channelCombo1.setDisable(true);
            channelCombo2.setDisable(true);
            channelCombo3.setDisable(true);
            return;
        }

        channelCombo1.setDisable(false);
        channelCombo2.setDisable(false);
        channelCombo3.setDisable(false);

        // Channel 1: required — no None option
        channelCombo1.getItems().addAll(names);
        // Channels 2 & 3: optional — (None) means zero-pad that slot
        channelCombo2.getItems().add(CHANNEL_NONE);
        channelCombo2.getItems().addAll(names);
        channelCombo3.getItems().add(CHANNEL_NONE);
        channelCombo3.getItems().addAll(names);

        // Restore previous selection if still valid, else use a sensible default
        channelCombo1.setValue(names.contains(prev1) ? prev1 : names.get(0));
        channelCombo2.setValue(CHANNEL_NONE.equals(prev2) || names.contains(prev2)
                ? prev2 : (names.size() >= 2 ? names.get(1) : CHANNEL_NONE));
        channelCombo3.setValue(CHANNEL_NONE.equals(prev3) || names.contains(prev3)
                ? prev3 : (names.size() >= 3 ? names.get(2) : CHANNEL_NONE));
    }

    /**
     * Returns the ordered list of non-(None) channel names from the three channel combos.
     * Trailing (None) selections stop collection; enforceThreeChannels() will zero-pad those slots.
     */
    private List<String> getSelectedChannelNames() {
        List<String> result = new ArrayList<>(3);
        for (var combo : List.of(channelCombo1, channelCombo2, channelCombo3)) {
            String val = combo.getValue();
            if (val == null || CHANNEL_NONE.equals(val)) break;
            result.add(val);
        }
        return result;
    }

    @FXML
    void promptForModelDirectory() {
        Path currentPath = modelPathBinding.get();
        File initialDir = null;
        if (currentPath != null && Files.exists(currentPath)) {
            if (Files.isDirectory(currentPath)) {
                initialDir = currentPath.toFile();
            } else if (currentPath.getParent() != null) {
                initialDir = currentPath.getParent().toFile();
            }
        }

        FileChooser chooser = new FileChooser();
        chooser.setTitle("Select CPSAM TorchScript model");
        chooser.getExtensionFilters().add(
            new FileChooser.ExtensionFilter("TorchScript model", "*.pt")
        );
        if (initialDir != null) {
            chooser.setInitialDirectory(initialDir);
        }
        chooser.setInitialFileName("cpsam_torchscript.pt");

        File file = chooser.showOpenDialog(modelPathLabel.getScene().getWindow());
        if (file != null) {
            Path selected = file.toPath();
            modelPathBinding.set(selected);
            CpSamPreferences.modelDirectoryProperty().set(selected.toString());
            updateModelPathLabel();
        }
    }

    @FXML
    void runCpSam() {
        ImageData<BufferedImage> imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorNotification("CPSAM", "No image data available");
            return;
        }

        Path modelPath = modelPathBinding.get();
        if (modelPath == null) {
            Dialogs.showErrorNotification("CPSAM", "No model path specified");
            return;
        }

        // Validate model path
        if (!Files.exists(modelPath)) {
            Dialogs.showErrorNotification("CPSAM", "Model file not found: " + modelPath);
            return;
        }

        // Check PyTorch
        try {
            if (!ai.djl.engine.Engine.getInstance().getEngine("PyTorch").getEngineName().equals("PyTorch")) {
                // Try loading PyTorch
                try {
                    DjlTools.getEngine("PyTorch", true);
                } catch (Exception e) {
                    Dialogs.showWarningNotification("CPSAM", "PyTorch engine not available. Please install PyTorch first.");
                    return;
                }
            }
        } catch (Exception e) {
            Dialogs.showWarningNotification("CPSAM", "Could not verify PyTorch: " + e.getMessage());
            return;
        }

        // Get parameters
        double diameter = diameterSpinner.getValueFactory().getValue();
        float cellprobThreshold = cellProbThresholdSpinner.getValueFactory().getValue().floatValue();
        float flowThreshold = flowThresholdSpinner.getValueFactory().getValue().floatValue();
        int niter = niterSpinner.getValueFactory().getValue();
        int batchSize = batchSizeSpinner.getValueFactory().getValue();
        int tileSize = tileSizeChoiceBox.getValue();
        int tilePadding = tilePaddingChoiceBox.getValue();
        String device = deviceChoices.getSelectionModel().getSelectedItem();

        // Create output type mapping
        Class<? extends qupath.lib.objects.PathObject> outputClass = switch (comboOutputType.getSelectionModel().getSelectedItem()) {
            case "Annotations" -> qupath.lib.objects.PathAnnotationObject.class;
            case "Cells" -> qupath.lib.objects.PathCellObject.class;
            default -> qupath.lib.objects.PathDetectionObject.class;
        };

        // Create and schedule task
        List<String> channelNames = getSelectedChannelNames();
        var task = new CpSamTask(
                imageData,
                modelPath.toString(),
                device,
                diameter,
                cellprobThreshold,
                flowThreshold,
                niter,
                batchSize,
                tileSize,
                tilePadding,
                outputClass,
                channelNames
        );

        // Attach state listener
        task.stateProperty().addListener((observable, oldValue, newValue) -> {
            if (Set.of(Worker.State.CANCELLED, Worker.State.SUCCEEDED, Worker.State.FAILED).contains(newValue)) {
                if (pendingTask.get() == task) {
                    pendingTask.set(null);
                }
            }
        });

        pendingTask.set(task);
    }

}
