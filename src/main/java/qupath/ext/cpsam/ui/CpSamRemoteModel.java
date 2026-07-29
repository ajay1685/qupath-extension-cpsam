package qupath.ext.cpsam.ui;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.stream.Collectors;

/**
 * DTO for remote CPSAM/CPDINO model entries loaded from a bundled JSON catalog.
 * <p>
 * Mirrors InstanSeg's {@code RemoteModel} class, extended with device constraint
 * information to enforce compatible device selection at runtime.
 */
public class CpSamRemoteModel {

    private static final Logger logger = LoggerFactory.getLogger(CpSamRemoteModel.class);

    /** Path to the bundled model index JSON file. */
    
    //private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.cpsam.ui.model-index.json");
    public static final String MODEL_INDEX_RESOURCE = "model-index.json";

    private final String name;
    private final String displayName;
    private final String family;
    private final String type;
    private final String version;
    private final URL url;
    private final DeviceConstraint deviceConstraint;
    private final String description;

    /**
     * What device capability a TorchScript model requires.
     */
    public enum DeviceConstraint {
        /** Universal model — works on CPU, CUDA, and MPS. */
        ANY,
        /** Runs on CPU (and MPS via CPU fallback). Not CUDA-accelerated. */
        CPU,
        /** Requires an actual CUDA GPU — will fail or crash on CPU/MPS. */
        CUDA;

        /**
         * Check if a device string is compatible with this constraint.
         * @param device a device name like "cpu", "cuda0", "mps"
         * @return true if the device can run models with this constraint
         */
        public boolean isCompatibleWithDevice(String device) {
            if (device == null) return false;
            String d = device.toLowerCase();
            return switch (this) {
                case ANY -> true;
                case CPU -> true;  // cpu/mps via CPU fallback
                case CUDA -> d.startsWith("cuda");
            };
        }
    }

    CpSamRemoteModel(String name, URL url, String version, String family, String displayName, DeviceConstraint deviceConstraint, String type, String description) {
        this.name = name;
        this.url = url;
        this.version = version;
        this.family = family != null ? family : "CPSAM";
        this.displayName = displayName;
        this.deviceConstraint = deviceConstraint != null ? deviceConstraint : DeviceConstraint.ANY;
        this.type = type;
        this.description = description;
    }

    public String getName() {
        return name;
    }

    public URL getUrl() {
        return url;
    }

    public String getVersion() {
        return version;
    }

    public String getFamily() {
        return family;
    }

    public DeviceConstraint getDeviceConstraint() {
        return deviceConstraint;
    }

    /**
     * Get the display name. Falls back to {@code name} if not set.
     */
    public String getDisplayName() {
        return displayName != null ? displayName : name;
    }

    /**
     * Get the model type (e.g., "vitl", "vitb" for CPDINO). Null for models without a type (CPSAM).
     */
    public String getType() {
        return type;
    }

    /**
     * Get the human-readable description. Returns empty string if not set.
     */
    public String getDescription() {
        return description != null ? description : "";
    }

    /**
     * Derive a folder name from the model name.
     * Must match the directory created by the downloader (which strips .zip from the archive filename).
     * Since downloads use name + ".zip", the extraction directory is simply "name".
     */
    public String getFolderName() {
        return name;
    }

    /**
     * List of folder name candidates to probe when checking validity.
     * Covers current convention (name only) and previous convention (name-version).
     */
    public List<String> getFolderNameCandidates() {
        if (version != null && !version.isEmpty()) {
            return List.of(name, name + "-" + version);
        }
        return List.of(name);
    }

    @Override
    public String toString() {
        return getDisplayName() + " (" + version + ")";
    }

    /**
     * Load the model catalog from the bundled JSON resource.
     * @return an unmodifiable list of remote model entries
     */
    public static List<CpSamRemoteModel> loadCatalog() {
        try (InputStream in = CpSamRemoteModel.class.getResourceAsStream(MODEL_INDEX_RESOURCE)) {
            if (in == null) {
                logger.warn("Model index resource not found: {}", MODEL_INDEX_RESOURCE);
                return List.of();
            }
            String json = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            Type type = TypeToken.getParameterized(List.class, CpSamRemoteModelEntry.class).getType();
            List<CpSamRemoteModelEntry> entries = new Gson().fromJson(json, type);
            if (entries == null) return List.of();

            var result = entries.stream()
                    .filter(e -> e.name != null && e.getUrl() != null)
                    .map(e -> {
                        try {
                            return fromEntry(e);
                        } catch (Exception ex) {
                            logger.warn("Skipping model entry '{}': {}", e.name, ex.getMessage());
                            return null;
                        }
                    })
                    .filter(model -> {
                        if (model == null) return false;
                        logger.info("Loaded remote model: {} ({})", model.getName(), model.getUrl());
                        return true;
                    })
                    .collect(Collectors.collectingAndThen(Collectors.toList(), List::copyOf));

            if (result.isEmpty() && entries.size() > 0) {
                logger.warn("All {} catalog entries failed to parse", entries.size());
            }
            return result;
        } catch (Exception e) {
            logger.error("Failed to load model catalog: {}", e.getMessage(), e);
            return List.of();
        }
    }

    private static CpSamRemoteModel fromEntry(CpSamRemoteModelEntry entry) {
        DeviceConstraint constraint = DeviceConstraint.ANY;
        if (entry.deviceConstraint != null) {
            try {
                constraint = DeviceConstraint.valueOf(entry.deviceConstraint.toUpperCase());
            } catch (IllegalArgumentException e) {
                logger.warn("Unknown device constraint '{}', defaulting to ANY for model {}", entry.deviceConstraint, entry.name);
            }
        }
        return new CpSamRemoteModel(
                entry.name,
                entry.getUrl(),
                entry.version,
                entry.family,
                entry.displayName,
                constraint,
                entry.type,
                entry.description
        );
    }

    /** Inner class for Gson deserialization — mirrors the JSON structure. */
    private static class CpSamRemoteModelEntry {
        String name;
        String displayName;
        String family;
        String type;
        String version;
        String url;  // Gson deserializes JSON string into String, then we convert to URL
        String deviceConstraint;
        String description;

        URL getUrl() {
            try {
                return url != null ? new java.net.URI(url).toURL() : null;
            } catch (Exception e) {
                logger.warn("Invalid URL '{}' for entry '{}'", url, name);
                return null;
            }
        }
    }
}
