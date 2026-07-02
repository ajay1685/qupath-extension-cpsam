package qupath.ext.cpsam;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.StringProperty;
import qupath.lib.common.GeneralTools;
import qupath.lib.gui.prefs.PathPrefs;

/**
 * Persistent preferences for the CPSAM extension.
 */
public class CpSamPreferences {

    private CpSamPreferences() {
        throw new AssertionError("Cannot instantiate this class");
    }

    private static final StringProperty modelDirectoryProperty = PathPrefs.createPersistentPreference(
            "cpsam.model.dir", null);

    private static final StringProperty preferredDeviceProperty = PathPrefs.createPersistentPreference(
            "cpsam.pref.device", getDefaultDevice());

    private static final DoubleProperty diameterProperty = PathPrefs.createPersistentPreference(
            "cpsam.diameter", 30.0);

    private static final DoubleProperty cellprobThresholdProperty = PathPrefs.createPersistentPreference(
            "cpsam.cellprob_threshold", 0.0);

    private static final DoubleProperty flowThresholdProperty = PathPrefs.createPersistentPreference(
            "cpsam.flow_threshold", 0.4);

    private static final IntegerProperty niterProperty = PathPrefs.createPersistentPreference(
            "cpsam.niter", 200);

    private static final IntegerProperty batchSizeProperty = PathPrefs.createPersistentPreference(
            "cpsam.batch_size", 1);

    private static final IntegerProperty tileSizeProperty = PathPrefs.createPersistentPreference(
            "cpsam.tile_size", 2048);

    private static final IntegerProperty tilePaddingProperty = PathPrefs.createPersistentPreference(
            "cpsam.tile_padding", 64);

    private static final DoubleProperty normLowProperty = PathPrefs.createPersistentPreference(
            "cpsam.norm_low", 1.0);

    private static final DoubleProperty normHighProperty = PathPrefs.createPersistentPreference(
            "cpsam.norm_high", 99.0);

    private static final BooleanProperty makeMeasurementsProperty = PathPrefs.createPersistentPreference(
            "cpsam.measurements", true);

    private static final IntegerProperty numThreadsProperty = PathPrefs.createPersistentPreference(
            "cpsam.num_threads", GeneralTools.clipValue(Runtime.getRuntime().availableProcessors() / 2, 1, 4));

        private static final BooleanProperty verboseLoggingProperty = PathPrefs.createPersistentPreference(
            "cpsam.verbose_logging", false);

    private static final BooleanProperty savePreprocessedTilesProperty = PathPrefs.createPersistentPreference(
            "cpsam.save_preprocessed_tiles", false);

    /**
     * MPS should work reliably (and much faster) on Apple Silicon, so set as default.
     * Everywhere else, use CPU as we can't count on a GPU/CUDA being available.
     */
    private static String getDefaultDevice() {
        if (System.getProperty("os.name").toLowerCase().contains("mac") &&
            "aarch64".equals(System.getProperty("os.arch"))) {
            return "mps";
        }
        return "cpu";
    }

    public static StringProperty modelDirectoryProperty() {
        return modelDirectoryProperty;
    }

    public static StringProperty preferredDeviceProperty() {
        return preferredDeviceProperty;
    }

    public static DoubleProperty diameterProperty() {
        return diameterProperty;
    }

    public static DoubleProperty cellprobThresholdProperty() {
        return cellprobThresholdProperty;
    }

    public static DoubleProperty flowThresholdProperty() {
        return flowThresholdProperty;
    }

    public static IntegerProperty niterProperty() {
        return niterProperty;
    }

    public static IntegerProperty batchSizeProperty() {
        return batchSizeProperty;
    }

    public static IntegerProperty tileSizeProperty() {
        return tileSizeProperty;
    }

    public static IntegerProperty tilePaddingProperty() {
        return tilePaddingProperty;
    }

    public static DoubleProperty normLowProperty() {
        return normLowProperty;
    }

    public static DoubleProperty normHighProperty() {
        return normHighProperty;
    }

    public static BooleanProperty makeMeasurementsProperty() {
        return makeMeasurementsProperty;
    }

    public static IntegerProperty numThreadsProperty() {
        return numThreadsProperty;
    }

    public static BooleanProperty verboseLoggingProperty() {
        return verboseLoggingProperty;
    }

    public static BooleanProperty savePreprocessedTilesProperty() {
        return savePreprocessedTilesProperty;
    }
}
