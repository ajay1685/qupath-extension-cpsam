package qupath.ext.cpsam.ui;

import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.value.ObservableValue;
import javafx.concurrent.Worker;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.cpsam.*;
import qupath.ext.djl.DjlTools;
import qupath.fx.dialogs.Dialogs;
import qupath.lib.common.ThreadTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.UserDirectoryManager;
import qupath.lib.images.ImageData;

import java.text.MessageFormat;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

public class CpSamInterfaceController extends VBox {

    private static final Logger logger = LoggerFactory.getLogger(CpSamInterfaceController.class);
    private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.cpsam.ui.strings");

    @FXML
    private ChoiceBox<String> modelFamilyChoiceBox;
    @FXML
    private ComboBox<CpSamModel> modelVariantCombo;
    @FXML
    private Button modelDownloadButton;
    @FXML
    private javafx.scene.control.ProgressBar modelDownloadProgress;
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
    private Spinner<Double> normLowSpinner;
    @FXML
    private Spinner<Double> normHighSpinner;
    @FXML
    private CheckBox measureShapeCheckBox;
    @FXML
    private CheckBox measureIntensityCheckBox;
    @FXML
    private Button runButton;
    @FXML
    private Label labelMessage;
    @FXML
    private Label labelActiveModel;
    @FXML
    private Label labelModelDescription;
    @FXML
    private Label labelDeviceWarning;

    @FXML
    private ComboBox<CpSamChannelItem> channelCombo1;
    @FXML
    private ComboBox<CpSamChannelItem> channelCombo2;
    @FXML
    private ComboBox<CpSamChannelItem> channelCombo3;

    private ExecutorService pool = Executors.newSingleThreadExecutor(ThreadTools.createThreadFactory("cpsam", true));
    private ExecutorService downloadPool = Executors.newSingleThreadExecutor(ThreadTools.createThreadFactory("cpsam-download", true));

    /** Lazily recreate the segmentation executor if it was shut down during close(). Thread-safe for FX thread only. */
    private ExecutorService getPool() {
        if (pool.isShutdown() || pool.isTerminated()) {
            pool = Executors.newSingleThreadExecutor(ThreadTools.createThreadFactory("cpsam", true));
            logger.debug("Recreated segmentation executor");
        }
        return pool;
    }

    /** Lazily recreate the download executor if it was shut down during close(). Thread-safe for FX thread only. */
    private ExecutorService getDownloadPool() {
        if (downloadPool.isShutdown() || downloadPool.isTerminated()) {
            downloadPool = Executors.newSingleThreadExecutor(ThreadTools.createThreadFactory("cpsam-download", true));
            logger.debug("Recreated download executor");
        }
        return downloadPool;
    }

    private final QuPathGUI qupath;
    private final ObjectProperty<Task<?>> pendingTask = new SimpleObjectProperty<>();

    private static final ObjectProperty<Path> modelPathBinding = new SimpleObjectProperty<>();

    /** Loaded remote model catalog entries. */
    private List<CpSamRemoteModel> remoteCatalog;

    /** Variant combo value change listener — stored as a field so it can be detached/reattached during auto-selection. */
    private javafx.beans.value.ChangeListener<CpSamModel> variantSelectionListener;

    /** Channel 2 → Channel 3 disable listener — stored as a field so it can be detached/reattached when image changes. */
    private javafx.beans.value.ChangeListener<CpSamChannelItem> channelCombo2Listener;

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

        // Initialize controls
        initSpinners();
        initChoiceBoxes();
        initPyTorchCheck();
        updateDeviceChoices();
        initModelSelection();
        initBindings();
        // Populate channel combos from the current image; repopulate when image changes
        qupath.imageDataProperty().addListener(this::handleImageDataChange);
        handleImageDataChange(qupath.imageDataProperty(), null, qupath.imageDataProperty().get());
    }

    private void initSpinners() {
        // Diameter
        SpinnerValueFactory.DoubleSpinnerValueFactory diamFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(10.0, 200.0, CpSamPreferences.diameterProperty().get(), 1);
        diameterSpinner.setValueFactory(diamFactory);
        diamFactory.valueProperty().addListener((v, o, n) -> CpSamPreferences.diameterProperty().set(n));

        // Flow threshold
        SpinnerValueFactory.DoubleSpinnerValueFactory flowFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(0.0, 10.0, CpSamPreferences.flowThresholdProperty().get(), 0.05);
        flowThresholdSpinner.setValueFactory(flowFactory);
        flowFactory.valueProperty().addListener((v, o, n) -> CpSamPreferences.flowThresholdProperty().set(n));

        // Cell prob threshold
        SpinnerValueFactory.DoubleSpinnerValueFactory probFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(-5.0, 5.0, CpSamPreferences.cellprobThresholdProperty().get(), 0.05);
        cellProbThresholdSpinner.setValueFactory(probFactory);
        probFactory.valueProperty().addListener((v, o, n) -> CpSamPreferences.cellprobThresholdProperty().set(n));

        // Niter
        SpinnerValueFactory.IntegerSpinnerValueFactory niterFactory = new SpinnerValueFactory.IntegerSpinnerValueFactory(100, 5000, CpSamPreferences.niterProperty().get(), 10);
        niterSpinner.setValueFactory(niterFactory);
        niterFactory.valueProperty().addListener((v, o, n) -> CpSamPreferences.niterProperty().set(n));

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

        // Norm low percentile
        SpinnerValueFactory.DoubleSpinnerValueFactory normLowFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(
                0.0, 100.0, CpSamPreferences.normLowProperty().get(), 0.5);
        normLowSpinner.setValueFactory(normLowFactory);
        normLowFactory.valueProperty().addListener((v, o, n) -> CpSamPreferences.normLowProperty().set(n));

        // Norm high percentile
        SpinnerValueFactory.DoubleSpinnerValueFactory normHighFactory = new SpinnerValueFactory.DoubleSpinnerValueFactory(
                0.0, 100.0, CpSamPreferences.normHighProperty().get(), 0.5);
        normHighSpinner.setValueFactory(normHighFactory);
        normHighFactory.valueProperty().addListener((v, o, n) -> CpSamPreferences.normHighProperty().set(n));
    }

    private void initChoiceBoxes() {
        // Tile sizes
        tileSizeChoiceBox.getItems().addAll(512, 1024, 2048, 4096);
        tileSizeChoiceBox.setValue(CpSamPreferences.tileSizeProperty().getValue());
        tileSizeChoiceBox.valueProperty().addListener((v, o, n) -> CpSamPreferences.tileSizeProperty().set(n));

        // Tile padding
        tilePaddingChoiceBox.getItems().addAll(0, 32, 64, 96, 128);
        tilePaddingChoiceBox.setValue(CpSamPreferences.tilePaddingProperty().getValue());
        tilePaddingChoiceBox.valueProperty().addListener((v, o, n) -> CpSamPreferences.tilePaddingProperty().set(n));

        // Output type
        comboOutputType.getItems().addAll("Detections", "Annotations");//, "Cells");
        comboOutputType.getSelectionModel().select(0);

        // Save device selection whenever it changes (population done in updateDeviceChoices)
        deviceChoices.valueProperty().addListener((v, o, n) -> {
            if (n != null) CpSamPreferences.preferredDeviceProperty().set(n);
            checkDeviceCompatibility();
        });
    }

    private void initModelSelection() {
        // Load remote model catalog from bundled JSON
        remoteCatalog = CpSamRemoteModel.loadCatalog();
        logger.info("Loaded {} remote model entries", remoteCatalog.size());

        // Populate family choice box with unique families from the catalog
        Set<String> families = remoteCatalog.stream()
                .map(CpSamRemoteModel::getFamily)
                .collect(Collectors.toCollection(() -> new TreeSet<>(String.CASE_INSENSITIVE_ORDER)));

        if (families.isEmpty()) {
            logger.warn("No model families found in catalog — model-index.json may be missing or failed to parse");
            // Leave family box empty so user knows something went wrong
            // Custom models can still be loaded via the scripting interface:
            //   CpSam.runCpSam(imageData, annotation, new File("/path/to/model.ts"), ...)
        } else {
            modelFamilyChoiceBox.getItems().addAll(families);
        }

        // Restore saved family, or default to CPSAM if available, otherwise first family alphabetically
        String savedFamily = CpSamPreferences.modelFamilyProperty().get();
        String defaultFamily;
        if (savedFamily != null && families.contains(savedFamily)) {
            defaultFamily = savedFamily;
        } else if (families.contains("CPSAM")) {
            defaultFamily = "CPSAM";
        } else {
            defaultFamily = families.isEmpty() ? null : families.iterator().next();
        }
        modelFamilyChoiceBox.setValue(defaultFamily);

        // On variant selection: notify device compatibility, update download button, auto-wire if downloaded
        variantSelectionListener = (javafx.beans.value.ChangeListener<CpSamModel>) (obs, oldVal, newVal) -> {
            if (newVal == null) return;

            // Notify user which device types this model works with (no filtering — just inform)
            notifyDeviceCompatibility(newVal);

            updateDownloadButtonState(newVal);

            // Auto-set the path if downloaded; clear it if not — prevents running inference
            // with a stale path from a different variant
            if (newVal.isValid()) {
                modelPathBinding.set(newVal.getPath());
            } else {
                modelPathBinding.set(null);
            }

            // Update active model label so user knows what's selected
            updateActiveModelLabel(newVal);

            // Save selection preference
            newVal.getName().ifPresent(name -> CpSamPreferences.modelCatalogKeyProperty().set(name));
        };
        modelVariantCombo.valueProperty().addListener(variantSelectionListener);

        // Set up cell factory for download status icons
        modelVariantCombo.setCellFactory(param -> new CpSamModelListCell());
        modelVariantCombo.setButtonCell(new CpSamModelListCell());

        // On family change, save preference, populate variants and filter devices for the auto-selected model
        modelFamilyChoiceBox.valueProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal == null) return;
            CpSamPreferences.modelFamilyProperty().set(newVal);
            populateVariantsAndFilterDevices(newVal);
        });

        // When the model download directory preference changes, rebuild the variant list so
        // CpSamModel instances resolve against the new directory — without this, isValid()
        // keeps checking the old directory and downloaded models are never detected.
        CpSamPreferences.modelDownloadDirectoryProperty().addListener((obs, oldDir, newDir) -> {
            String family = modelFamilyChoiceBox.getValue();
            if (family != null) {
                logger.info("Model download directory changed to: {} — refreshing variants", newDir);
                populateVariantsAndFilterDevices(family);
                refreshDownloadStatus();
            }
        });

        // Trigger initial population with device filtering
        if (defaultFamily != null) {
            populateVariantsAndFilterDevices(defaultFamily);
        }

        // Restore previously selected variant
        String savedKey = CpSamPreferences.modelCatalogKeyProperty().get();
        if (savedKey != null && modelVariantCombo.getItems() != null) {
            for (CpSamModel model : modelVariantCombo.getItems()) {
                if (savedKey.equals(model.getName().orElse(""))) {
                    modelVariantCombo.getSelectionModel().select(model);
                    break;
                }
            }
        }

        // Scan cache directory for locally downloaded models to update validity status
        refreshDownloadStatus();

        // Initialize active model label based on current model path
        if (modelPathBinding.get() == null) {
            labelActiveModel.setText(resources.getString("ui.active.model.not-set"));
        }
    }

    /**
     * Populate the variant combo box with models matching the selected family.
     * Models are created from remote catalog entries via CpSamModel.fromRemote().
     * Variants with no compatible device on this machine are filtered out.
     */
    private void populateVariantsForFamily(String family) {
        Path cacheDir = resolveCacheDir();

        modelVariantCombo.getItems().clear();

        List<CpSamModel> variants = remoteCatalog.stream()
                .filter(remote -> remote.getFamily().equalsIgnoreCase(family))
                .map(remote -> CpSamModel.fromRemote(remote, cacheDir))
                .collect(Collectors.toList());

        // Separate compatible vs incompatible based on available devices.
        // If no devices are known yet (shouldn't happen), treat all as compatible.
        List<String> availableDevices = new ArrayList<>(deviceChoices.getItems());
        List<CpSamModel> compatible = new ArrayList<>();
        List<CpSamModel> incompatible = new ArrayList<>();

        if (availableDevices.isEmpty()) {
            compatible.addAll(variants);
        } else {
            for (CpSamModel v : variants) {
                boolean anyCompatible = availableDevices.stream()
                    .anyMatch(d -> v.isDeviceCompatible(d));
                if (anyCompatible) compatible.add(v);
                else incompatible.add(v);
            }
        }

        modelVariantCombo.getItems().addAll(compatible);

        // Log filtered models
        if (!incompatible.isEmpty()) {
            logger.info("Filtered {} model(s) for family '{}' — no compatible devices available",
                incompatible.size(), family);
        }

        // Enable/disable the combo based on whether any compatible variants exist
        modelVariantCombo.setDisable(compatible.isEmpty());

        // Default: select first available variant
        if (!compatible.isEmpty()) {
            modelVariantCombo.getSelectionModel().selectFirst();
        }
    }

    /**
     * Populate variants for a family and update active model label for the auto-selected model.
     * Auto-selects the first already-downloaded variant if available, otherwise picks the first variant.
     * Only considers variants compatible with the available devices.
     */
    private void populateVariantsAndFilterDevices(String family) {
        modelVariantCombo.valueProperty().removeListener(variantSelectionListener);
        try {
            populateVariantsForFamily(family);

            // Prefer selecting an already-downloaded variant from the compatible list
            CpSamModel downloaded = null;
            for (CpSamModel variant : modelVariantCombo.getItems()) {
                if (variant.isValid()) {
                    downloaded = variant;
                    break;
                }
            }
            if (downloaded != null) {
                modelVariantCombo.getSelectionModel().select(downloaded);
                modelPathBinding.set(downloaded.getPath());
                updateActiveModelLabel(downloaded);
            } else if (!modelVariantCombo.getItems().isEmpty()) {
                // No models downloaded for this family — clear path so Run button disables
                modelPathBinding.set(null);
                CpSamModel first = modelVariantCombo.getValue();
                if (first != null) {
                    updateActiveModelLabel(first);
                }
            }
        } finally {
            modelVariantCombo.valueProperty().addListener(variantSelectionListener);
        }
    }

    /**
     * Scan the cache directory to refresh download status of combo box items.
     * This is called on startup and after successful downloads.
     */
    private void refreshDownloadStatus() {
        // Force re-evaluation by clearing and re-adding items so ListCells re-render
        // Must save selection first — clearing items resets combo value to null
        CpSamModel selected = modelVariantCombo.getValue();
        var items = modelVariantCombo.getItems();
        var snapshot = new ArrayList<>(items);
        items.clear();
        items.addAll(snapshot);

        // Restore selection and update download button state
        if (selected != null && items.contains(selected)) {
            modelVariantCombo.getSelectionModel().select(selected);
            updateDownloadButtonState(selected);
        } else if (!items.isEmpty()) {
            modelVariantCombo.getSelectionModel().selectFirst();
        }
    }

    @FXML
    void handleDownloadModel() {
        CpSamModel selectedModel = modelVariantCombo.getValue();
        if (selectedModel == null) {
            Dialogs.showWarningNotification("CPSAM", "No model variant selected");
            return;
        }

        // Check if already downloaded
        if (selectedModel.isValid()) {
            // Auto-set path if not already set
            if (!Files.exists(modelPathBinding.get())) {
                modelPathBinding.set(selectedModel.getPath());
            }
            return;
        }

        // Get remote URL for download
        var urlOpt = selectedModel.getRemoteUrl();
        if (!urlOpt.isPresent()) {
            Dialogs.showErrorNotification("CPSAM", "No download URL available for this model");
            return;
        }

        // Resolve cache directory
        Path cacheDir = resolveCacheDir();

        // Show progress bar
        modelDownloadProgress.setVisible(true);
        modelDownloadProgress.setProgress(0.0);

        // Disable download button during download
        modelDownloadButton.setDisable(true);

        // Download asynchronously
        CpSamModelDownloader downloader = new CpSamModelDownloader(cacheDir);
        String modelName = selectedModel.getName().orElse("model");

        CompletableFuture<CpSamModelDownloader.DownloadResult> future = CompletableFuture.supplyAsync(() ->
            downloader.downloadFromUrl(
                urlOpt.get().toString(),
                modelName + ".zip",
                null,  // SHA-256 checksum (null = skip verification for now)
                p -> Platform.runLater(() -> modelDownloadProgress.setProgress(p))
            ), getDownloadPool());

        future.whenComplete((result, ex) -> {
            Platform.runLater(() -> {
                modelDownloadProgress.setVisible(false);
                modelDownloadButton.setDisable(false);

                if (ex != null) {
                    Dialogs.showErrorNotification("CPSAM", "Download failed: " + ex.getMessage());
                    logger.error("Download failed", ex);
                    return;
                }

                if (result.isSuccess()) {
                    Path modelPath = result.getPath();
                    selectedModel.setPath(modelPath);
                    modelPathBinding.set(modelPath);

                    // Save the selection preference
                    selectedModel.getName().ifPresent(name -> CpSamPreferences.modelCatalogKeyProperty().set(name));

                    // Save the downloaded version for future update checks
                    String version = selectedModel.getVersion();
                    if (version != null && !version.isEmpty()) {
                        selectedModel.getName().ifPresent(name ->
                            CpSamPreferences.downloadedVersionProperty(name).set(version));
                    }

                    updateDownloadButtonState(selectedModel);
                    updateActiveModelLabel(selectedModel);
                    refreshDownloadStatus();
                    Dialogs.showInfoNotification("CPSAM", "Model downloaded successfully: " + modelPath.getFileName());
                } else {
                    Dialogs.showErrorNotification("CPSAM", "Download failed: " + result.getMessage());
                }
            });
        });
    }

    private Path resolveCacheDir() {
        // Try preference first — create the directory if it doesn't exist yet
        String prefDir = CpSamPreferences.modelDownloadDirectoryProperty().get();
        if (prefDir != null && !prefDir.isBlank()) {
            Path p = Path.of(prefDir);
            if (Files.exists(p) && Files.isDirectory(p)) {
                return p;
            }
            try {
                Files.createDirectories(p);
                logger.info("Created model download directory: {}", p);
                return p;
            } catch (IOException e) {
                logger.warn("Could not use preferred model directory '{}' — falling back to default: {}", p, e.getMessage());
            }
        }

        // Fall back to QuPath user directory / cpsam-models
        Path qupathUserDir = UserDirectoryManager.getInstance().getUserPath();
        if (qupathUserDir == null) {
            // Very unlikely fallback
            qupathUserDir = Path.of(System.getProperty("user.home"));
        }
        Path defaultCache = qupathUserDir.resolve("cpsam-models");

        try {
            Files.createDirectories(defaultCache);
        } catch (IOException e) {
            logger.warn("Could not create default cache directory: {}", defaultCache, e);
            try {
                defaultCache = Path.of(System.getProperty("java.io.tmpdir")).resolve("cpsam-models");
                Files.createDirectories(defaultCache);
            } catch (IOException e2) {
                throw new RuntimeException("Cannot create cache directory", e2);
            }
        }

        return defaultCache;
    }

    /** Update download button: disabled + "✓ Downloaded" if model valid; enabled + "Download" if not. */
    private void updateDownloadButtonState(CpSamModel model) {
        if (model == null) {
            modelDownloadButton.setDisable(true);
            return;
        }

        if (model.isValid()) {
            modelDownloadButton.setDisable(true);
            modelDownloadButton.setText(resources.getString("ui.model.downloaded.button"));
            // Auto-wire the model path if not already set to something else
            Path current = modelPathBinding.get();
            if (current == null || !Files.exists(current)) {
                modelPathBinding.set(model.getPath());
            }
        } else {
            modelDownloadButton.setDisable(false);
            modelDownloadButton.setText(resources.getString("ui.model.download.button"));
        }
    }

    private void initBindings() {
        // Run button bindings
        runButton.disableProperty().bind(
            qupath.imageDataProperty().isNull()
                .or(pendingTask.isNotNull())
                .or(Bindings.createBooleanBinding(() -> modelPathBinding.get() == null, modelPathBinding))
        );

        // Measure shape
        measureShapeCheckBox.selectedProperty().bindBidirectional(
                CpSamPreferences.measureShapeProperty());

        // Measure intensity
        measureIntensityCheckBox.selectedProperty().bindBidirectional(
                CpSamPreferences.measureIntensityProperty());

        // Pending task execution — use getPool() for lazy recreation after close()
        pendingTask.addListener((observable, oldValue, newValue) -> {
            if (newValue != null) {
                getPool().execute(newValue);
            }
        });
    }

    private void initPyTorchCheck() {
        // Ensure PyTorch is loaded before querying devices.
        // DJL is in offline mode (ai.djl.offline=true set by DjlExtension), so Engine.getEngine()
        // will NOT attempt a download — it either loads if already available or throws.
        // Device choices are populated immediately after this in the constructor via updateDeviceChoices().
        try {
            Class.forName("ai.djl.pytorch.jni.PyTorchLibrary");
        } catch (ClassNotFoundException e) {
            try {
                ai.djl.engine.Engine.getInstance().getEngine("PyTorch");
            } catch (Exception ex) {
                logger.warn("PyTorch engine not available");
            }
        }
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

        // Check compatibility now that devices are populated
        checkDeviceCompatibility();
    }

    /**
     * Notify the user which device types a model is compatible with.
     * This replaces the old filterDevicesForModel — no filtering, just log + active label update.
     */
    private void notifyDeviceCompatibility(CpSamModel model) {
        CpSamRemoteModel.DeviceConstraint constraint = model.getDeviceConstraint();
        if (constraint == CpSamRemoteModel.DeviceConstraint.ANY) {
            // No need to notify for universally-compatible models
            return;
        }

        String constraintDesc = switch (constraint) {
            case CPU -> "CPU";
            case CUDA -> "CUDA GPU";
            default -> "any device";
        };

        logger.info("Model '{}' requires {}", model.getName().orElse("unknown"), constraintDesc);
    }

    /**
     * Update the active model label to show which model will be used for inference.
     */
    private void updateActiveModelLabel(CpSamModel model) {
        if (model == null) {
            labelActiveModel.setText(resources.getString("ui.active.model.not-set"));
            labelModelDescription.setVisible(false);
            labelDeviceWarning.setVisible(false);
            return;
        }

        String displayName = model.getDisplayName() != null ? model.getDisplayName() : model.getName().orElse("?");
        String status = model.isValid() ? "" : " (not downloaded)";
        CpSamRemoteModel.DeviceConstraint constraint = model.getDeviceConstraint();
        String deviceHint = switch (constraint) {
            case ANY -> "";
            case CPU -> " [CPU]";
            case CUDA -> " [CUDA GPU]";
        };

        labelActiveModel.setText("Active model: " + displayName + status + deviceHint);

        // Set description
        String desc = model.getDescription();
        if (desc.isEmpty()) {
            labelModelDescription.setText("");
            labelModelDescription.setVisible(false);
        } else {
            labelModelDescription.setText(desc);
            labelModelDescription.setVisible(true);
        }

        // Check device compatibility
        checkDeviceCompatibility();
    }

    /**
     * Check if the currently selected device is compatible with the selected model.
     * Shows a warning label if incompatible, hides it if compatible.
     */
    private void checkDeviceCompatibility() {
        CpSamModel model = modelVariantCombo.getValue();
        if (model == null) {
            labelDeviceWarning.setVisible(false);
            return;
        }
        String device = deviceChoices.getValue();
        if (model.isDeviceCompatible(device)) {
            labelDeviceWarning.setVisible(false);
        } else {
            String constraintDesc = switch (model.getDeviceConstraint()) {
                case CPU -> resources.getString("ui.device.warning.cpu");
                case CUDA -> resources.getString("ui.device.warning.cuda");
                default -> resources.getString("ui.device.warning.any");
            };
            String pattern = resources.getString("ui.device.warning");
            labelDeviceWarning.setText(MessageFormat.format(pattern, constraintDesc, device));
            labelDeviceWarning.setVisible(true);
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
        CpSamChannelItem prev1 = channelCombo1.getValue();
        CpSamChannelItem prev2 = channelCombo2.getValue();
        CpSamChannelItem prev3 = channelCombo3.getValue();

        channelCombo1.getItems().clear();
        channelCombo2.getItems().clear();
        channelCombo3.getItems().clear();

        if (imageData == null) {
            channelCombo1.setDisable(true);
            channelCombo2.setDisable(true);
            channelCombo3.setDisable(true);
            return;
        }

        if (imageData.getServer().nChannels() == 0) {
            channelCombo1.setDisable(true);
            channelCombo2.setDisable(true);
            channelCombo3.setDisable(true);
            return;
        }

        channelCombo1.setDisable(false);
        channelCombo2.setDisable(false);
        channelCombo3.setDisable(false);

        // Get available channels (raw + deconvolution)
        List<CpSamChannelItem> allChannels = CpSamChannelItem.getAvailableChannels(imageData);
        List<CpSamChannelItem> noneItems = CpSamChannelItem.getNoneItems();

        // Channel 1: required — no None option
        channelCombo1.getItems().addAll(allChannels);
        // Channels 2 & 3: optional — (None) means zero-pad that slot
        channelCombo2.getItems().addAll(noneItems);
        channelCombo2.getItems().addAll(allChannels);
        channelCombo3.getItems().addAll(noneItems);
        channelCombo3.getItems().addAll(allChannels);

        // Restore previous selection if still valid, else use a sensible default
        channelCombo1.setValue(restoreOrDefault(prev1, allChannels, 0));
        channelCombo2.setValue(restoreOrDefaultOptional(prev2, allChannels, noneItems.get(0), 1));
        channelCombo3.setValue(restoreOrDefaultOptional(prev3, allChannels, noneItems.get(0), 2));

        // Disable channel 3 when channel 2 is (None) — no point selecting a third channel if the second is zero-padded
        if (channelCombo2Listener != null) {
            channelCombo2.valueProperty().removeListener(channelCombo2Listener);
        }
        channelCombo2Listener = (obs, oldVal, newVal) -> {
            channelCombo3.setDisable(newVal != null && newVal.isNone());
            if (newVal != null && newVal.isNone()) {
                channelCombo3.setValue(null);
            }
        };
        channelCombo2.valueProperty().addListener(channelCombo2Listener);
    }

    /**
     * Restore a previous selection if it's still in the available list, otherwise return the item at the given index.
     */
    private CpSamChannelItem restoreOrDefault(CpSamChannelItem prev, List<CpSamChannelItem> available, int defaultIndex) {
        if (prev != null && available.contains(prev)) {
            return prev;
        }
        return available.get(Math.min(defaultIndex, available.size() - 1));
    }

    /**
     * Restore a previous selection for optional channel combos.
     * Accepts the value if it's in either the available channels or is a None item.
     * @param defaultIndex index into allChannels to use as default (e.g. 1 for channel 2, 2 for channel 3)
     */
    private CpSamChannelItem restoreOrDefaultOptional(CpSamChannelItem prev, List<CpSamChannelItem> allChannels, CpSamChannelItem noneDefault, int defaultIndex) {
        if (prev != null) {
            // Check if it's a valid channel
            if (allChannels.contains(prev)) {
                return prev;
            }
            // Check if it's a None item (match by isNone flag since different instances won't be equal)
            if (prev.isNone()) {
                return noneDefault;
            }
        }
        // Default: use the channel at defaultIndex if available, otherwise None
        if (defaultIndex < allChannels.size()) {
            return allChannels.get(defaultIndex);
        }
        return noneDefault;
    }

    /**
     * Returns the ordered list of selected channel items from the three channel combos.
     * Trailing (None) selections stop collection; CpSamPreProcessing.enforceThreeChannels() will zero-pad those slots.
     */
    private List<CpSamChannelItem> getSelectedChannels() {
        List<CpSamChannelItem> result = new ArrayList<>(3);
        for (var combo : List.of(channelCombo1, channelCombo2, channelCombo3)) {
            CpSamChannelItem val = combo.getValue();
            if (val == null || val.isNone()) break;
            result.add(val);
        }
        return result;
    }

    @FXML
    void runCpSam() {
        ImageData<BufferedImage> imageData = qupath.getImageData();
        if (imageData == null) {
            Dialogs.showErrorNotification("CPSAM", "No image data available");
            return;
        }

        CpSamModel selectedModel = modelVariantCombo.getValue();
        if (selectedModel == null || !selectedModel.isValid()) {
            Dialogs.showErrorNotification("CPSAM", "No valid model selected — download a model from the catalog first");
            return;
        }
        Path modelPath = selectedModel.getModelPath();

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
        int numThreads = threadSpinner.getValueFactory().getValue();
        double normLow = normLowSpinner.getValueFactory().getValue();
        double normHigh = normHighSpinner.getValueFactory().getValue();
        boolean measureShape = measureShapeCheckBox.isSelected();
        boolean measureIntensity = measureIntensityCheckBox.isSelected();
        String device = deviceChoices.getSelectionModel().getSelectedItem();

        // Check device compatibility before running
        if (!selectedModel.isDeviceCompatible(device)) {
            String constraintDesc = switch (selectedModel.getDeviceConstraint()) {
                case CPU -> resources.getString("ui.device.warning.cpu");
                case CUDA -> resources.getString("ui.device.warning.cuda");
                default -> resources.getString("ui.device.warning.any");
            };
            String pattern = resources.getString("ui.device.incompatible.error");
            Dialogs.showErrorNotification("CPSAM", MessageFormat.format(pattern, constraintDesc, device));
            return;
        }

        // Create output type mapping
        Class<? extends qupath.lib.objects.PathObject> outputClass = switch (comboOutputType.getSelectionModel().getSelectedItem()) {
            case "Annotations" -> qupath.lib.objects.PathAnnotationObject.class;
            //case "Cells" -> qupath.lib.objects.PathCellObject.class;
            default -> qupath.lib.objects.PathDetectionObject.class;
        };

        // Create and schedule task
        List<CpSamChannelItem> selectedChannels = getSelectedChannels();
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
                numThreads,
                normLow,
                normHigh,
                measureShape,
                measureIntensity,
                outputClass,
                selectedChannels
        );

        // Show running status
        labelMessage.setStyle("");
        labelMessage.setText(resources.getString("ui.status.running"));

        // Attach state listener
        task.stateProperty().addListener((observable, oldValue, newValue) -> {
            if (Set.of(Worker.State.CANCELLED, Worker.State.SUCCEEDED, Worker.State.FAILED).contains(newValue)) {
                if (pendingTask.get() == task) {
                    pendingTask.set(null);
                }
                Platform.runLater(() -> {
                    if (newValue == Worker.State.SUCCEEDED) {
                        labelMessage.setStyle("");
                        labelMessage.setText(resources.getString("ui.status.completed"));
                    } else if (newValue == Worker.State.FAILED) {
                        labelMessage.setStyle("-fx-text-fill: red;");
                        labelMessage.setText(resources.getString("ui.status.error"));
                    } else {
                        labelMessage.setStyle("");
                        labelMessage.setText("Cancelled");
                    }
                });
            }
        });

        pendingTask.set(task);
    }

    /**
     * Clean up resources when the panel is closed.
     * Shuts down the segmentation executor and clears the model cache.
     * The download executor is NOT shut down — in-progress downloads are allowed
     * to complete, delivering their notification via the JavaFX thread.
     * It will be lazily recreated on next use via getDownloadPool().
     */
    public void close() {
        try {
            pool.shutdownNow();
            logger.debug("CPSAM segmentation executor shut down");
        } catch (Exception e) {
            logger.debug("Error shutting down segmentation executor: {}", e.getMessage());
        }
        CpSam.clearModelCache();
    }
}
