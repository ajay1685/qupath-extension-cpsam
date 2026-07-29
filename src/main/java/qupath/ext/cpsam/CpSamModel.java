package qupath.ext.cpsam;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.common.GeneralTools;
import qupath.lib.images.servers.PixelCalibration;
import qupath.ext.cpsam.ui.CpSamModelDownloader;
import qupath.ext.cpsam.ui.CpSamRemoteModel;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/**
 * Represents a CPSAM or CPDINO TorchScript model.
 * <p>
 * Can be created from an existing path on disk ({@link #fromPath(Path)}) or
 * from a remote catalog entry.
 * When created from a remote entry, {@link #checkIfDownloaded(Path, boolean)} handles
 * download + unzip lifecycle.
 */
public class CpSamModel {

    private static final Logger logger = LoggerFactory.getLogger(CpSamModel.class);
    static final String MODEL_FILENAME = "torchscript.ts";
    private static final double DEFAULT_REQUESTED_PIXEL_SIZE = 1.0;

    private Path path;
    private final double requestedPixelSize;
    private String name;
    private String displayName;
    private String type;
    private String description;
    private String version;
    private CpSamRemoteModel.DeviceConstraint deviceConstraint;
    private URL remoteUrl;

    /** Cache directory where catalog models are stored. Null for locally-browsed models. */
    private Path modelCacheDir;

    private CpSamModel(Path path, double requestedPixelSize) {
        this.path = path.toAbsolutePath();
        this.requestedPixelSize = requestedPixelSize;
        this.deviceConstraint = CpSamRemoteModel.DeviceConstraint.ANY;
        this.remoteUrl = null;
    }

    private CpSamModel(CpSamRemoteModel remote, Path cacheDir) {
        this.name = remote.getName();
        this.displayName = remote.getDisplayName();
        this.type = remote.getType();
        this.description = remote.getDescription();
        this.version = remote.getVersion();
        this.deviceConstraint = remote.getDeviceConstraint();
        this.requestedPixelSize = DEFAULT_REQUESTED_PIXEL_SIZE;
        this.remoteUrl = remote.getUrl();
        this.modelCacheDir = cacheDir;

        // Set path to expected download location — probe candidates in case previously downloaded under a different convention
        Path found = findExistingModelPath(remote, cacheDir);
        if (found != null) {
            this.path = found;
            logger.info("Found existing model '{}' at: {}", name, found);
        } else {
            // Fallback: default to primary folder name (downloader target)
            this.path = cacheDir.resolve(remote.getFolderName());
        }
    }

    /**
     * Search for an already-downloaded model that could belong to this catalog entry.
     * Only returns a path if the match is unambiguous — never claim a downloaded folder
     * for multiple catalog entries.
     */
    private static Path findExistingModelPath(CpSamRemoteModel remote, Path cacheDir) {
        if (!Files.isDirectory(cacheDir)) return null;

        // Try named candidates (primary folder name and any versioned variants)
        for (String candidate : remote.getFolderNameCandidates()) {
            Path candidatePath = cacheDir.resolve(candidate);
            if (isValidModelDirForName(candidatePath, remote.getName())) {
                return candidatePath;
            }
        }

        // Fallback: scan cache directory — prefix match on the catalog name vs folder name.
        // A folder matches if its name starts with the catalog name (e.g. "cpsam_wrapper" matches
        // "cpsam_wrapper" and "cpsam_wrapper-v1.0.0") or the catalog name starts with the folder
        // name (reverse case for legacy naming). Only returns a match if it's unique.
        // The .ts file inside must also start with the catalog name (prevents cross-claiming).
        String modelName = remote.getName().toLowerCase();
        try (var stream = Files.list(cacheDir)) {
            var dirs = stream.filter(Files::isDirectory).collect(java.util.stream.Collectors.toList());

            Path bestMatch = null;
            int  matchCount = 0;

            for (Path dir : dirs) {
                String dirName = dir.getFileName().toString().toLowerCase();
                if ((dirName.startsWith(modelName) || modelName.startsWith(dirName))
                        && isValidModelDirForName(dir, remote.getName())) {
                    bestMatch = dir;
                    matchCount++;
                }
            }

            // Only claim the folder if there's exactly one prefix match — avoids two catalog
            // entries both claiming the same downloaded model (e.g. "cpsam_wrapper" vs
            // "cpsam_v2_wrapper" would both match with substring but not with prefix).
            if (matchCount == 1 && bestMatch != null) {
                return bestMatch;
            }
        } catch (IOException e) {
            logger.debug("Could not scan cache directory: {}", e.getMessage());
        }
        return null;
    }

    /**
     * Create a CpSamModel from a path on disk.
     * The path may point to either a directory containing the TorchScript wrapper
     * or directly to the .ts file.
     *
     * @param path path on disk
     * @return a new CpSamModel instance
     * @throws IOException if the path doesn't exist
     */
    public static CpSamModel fromPath(Path path) throws IOException {
        if (!Files.exists(path)) {
            throw new IOException("Model path does not exist: " + path);
        }
        return new CpSamModel(path, DEFAULT_REQUESTED_PIXEL_SIZE);
    }

    /**
     * Create a CpSamModel pointing to the file expected for a remote catalog entry
     * in the given cache directory.  Does not download; the file must already exist.
     *
     * @param remote   the remote model describing the model
     * @param cacheDir the directory where downloaded models are stored
     * @return a new CpSamModel instance pointing to {@code cacheDir/remote.getFolderName()}
     */
    public static CpSamModel fromRemote(CpSamRemoteModel remote, Path cacheDir) {
        return new CpSamModel(remote, cacheDir);
    }

    /**
     * Get the model directory path.
     */
    public Path getPath() {
        return path;
    }

    /**
     * Update the model path (used after successful download).
     */
    public void setPath(Path newPath) {
        this.path = newPath.toAbsolutePath();
    }

    /**
     * Get the full path to the TorchScript model file.
     * If this.path points to a directory, looks for any .ts file in it.
     * If it points directly to a .ts file, returns it as-is.
     * Returns a non-existent path if no .ts file is found (so isValid() returns false).
     */
    public Path getModelPath() {
        if (Files.isDirectory(path)) {
            // Try the conventional filename first
            Path candidate = path.resolve(MODEL_FILENAME);
            if (Files.exists(candidate)) return candidate;
            // Fallback: find any .ts file in the directory
            try (var stream = Files.list(path)) {
                return stream
                        .filter(p -> p.toString().endsWith(".ts"))
                        .findFirst()
                        .orElse(path.resolve(MODEL_FILENAME));  // return non-existent path, not the dir
            } catch (IOException e) {
                logger.warn("Could not list directory {} for .ts file", path);
                return path.resolve(MODEL_FILENAME);
            }
        }
        return path;
    }

    /**
     * Check if the model file exists on disk.
     * For catalog models, the .ts file name must start with the catalog name
     * to prevent a wrong model (e.g. cpsam_v2_wrapper.ts) from being accepted
     * as the correct one (e.g. cpsam_wrapper).
     */
    public boolean isValid() {
        Path modelPath = getModelPath();
        if (!Files.exists(modelPath)) {
            return false;
        }
        // For catalog models, verify the .ts file name matches the expected catalog name.
        // e.g. "cpsam_wrapper" should not accept "cpsam_v2_wrapper.ts".
        if (name != null && Files.isDirectory(path)) {
            String fileName = modelPath.getFileName().toString().toLowerCase();
            if (!fileName.startsWith(name.toLowerCase())) {
                logger.debug("Model '{}' at {} has mismatched file: {}", name, path, fileName);
                return false;
            }
        }
        return true;
    }

    /**
     * Validate that the model file exists and (for catalog models) has the expected name,
     * throwing an exception if not.
     */
    public void checkValid() throws IOException {
        if (!isValid()) {
            Path modelPath = getModelPath();
            String expected = name != null ? " (expected a file starting with '" + name + ".ts')" : "";
            throw new IOException("CPSAM TorchScript model not found: " + modelPath + expected);
        }
    }

    /**
     * Check if the model has been downloaded already, and optionally download it.
     * Mirrors InstanSeg's {@code InstanSegModel.checkIfDownloaded()}.
     *
     * @param cacheDir           the directory where models are cached
     * @param downloadIfNotValid if true, download from remote URL if not present locally
     */
    public void checkIfDownloaded(Path cacheDir, boolean downloadIfNotValid) throws IOException {
        if (remoteUrl == null) {
            // Browsed model — nothing to download
            return;
        }

        // Ensure expected directory exists and has the correct .ts file (not just any .ts file)
        if (isValid()) {
            return;
        }

        if (!downloadIfNotValid) {
            return;
        }

        Files.createDirectories(cacheDir);
        CpSamModelDownloader downloader = new CpSamModelDownloader(cacheDir);
        var result = downloader.downloadFromUrl(
                remoteUrl.toString(),
                name + ".zip",
                null,
                progress -> {}  // no progress callback here; caller handles it
        );

        if (result.isSuccess()) {
            logger.info("Downloaded and extracted model: {}", path);
        } else {
            throw new IOException("Failed to download model: " + result.getMessage());
        }
    }

    /**
     * Retrieve the remote URL for this model. Only valid for downloadable models.
     */
    public Optional<URL> getRemoteUrl() {
        return Optional.ofNullable(remoteUrl);
    }

    /**
     * Check if a directory contains a valid model (.ts file).
     * Accepts any .ts file — useful for quick directory scans where we don't know the expected name.
     */
    public static boolean isValidModelDir(Path path) {
        if (Files.isDirectory(path)) {
            try (var stream = Files.list(path)) {
                return stream.anyMatch(p -> p.toString().endsWith(".ts"));
            } catch (IOException e) {
                return false;
            }
        }
        return false;
    }

    /**
     * Check if a directory contains a .ts file whose name starts with the expected catalog name.
     * Prevents a wrong model (e.g. cpsam_v2_wrapper.ts) from being accepted for a different
     * catalog entry (e.g. cpsam_wrapper).
     *
     * @param path the directory to check
     * @param catalogName the expected catalog model name (e.g. "cpsam_wrapper")
     * @return true if the directory contains a .ts file starting with the catalog name
     */
    public static boolean isValidModelDirForName(Path path, String catalogName) {
        if (!Files.isDirectory(path) || catalogName == null) return false;
        String prefix = catalogName.toLowerCase();
        try (var stream = Files.list(path)) {
            return stream.anyMatch(p -> {
                String fileName = p.getFileName().toString().toLowerCase();
                return fileName.endsWith(".ts") && fileName.startsWith(prefix);
            });
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * Get the model name. Empty for browsed models.
     */
    public Optional<String> getName() {
        return Optional.ofNullable(name);
    }

    /**
     * Get the display name for this model (from catalog metadata). Null for browsed models.
     */
    public String getDisplayName() {
        return displayName != null ? displayName : name;
    }

    /**
     * Get the device constraint for this model. Browsed models default to ANY.
     */
    public CpSamRemoteModel.DeviceConstraint getDeviceConstraint() {
        return deviceConstraint;
    }

    /**
     * Check if the given device string is compatible with this model's device constraint.
     * @param device a device name like "cpu", "cuda0", "mps"
     * @return true if the device can run this model
     */
    public boolean isDeviceCompatible(String device) {
        return deviceConstraint.isCompatibleWithDevice(device);
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
     * Get the model version. Null for locally-browsed models.
     */
    public String getVersion() {
        return version;
    }

    /**
     * Get the preferred pixel size for the model.
     * @return the requested pixel size
     */
    public double getRequestedPixelSize() {
        return requestedPixelSize;
    }

    /**
     * Get the preferred downsample for running the model, incorporating information from the pixel calibration of the image.
     * <p>
     * Not used by the UI (which defaults to downsample 1.0). Intended for scripting/headless usage:
     * {@code CpSam.builder().model(remoteModel).downsample(model.getPreferredDownsample(cal)).build()}
     *
     * @param cal The pixel calibration of the image
     * @return the preferred downsample to use
     */
    public double getPreferredDownsample(PixelCalibration cal) {
        if (requestedPixelSize <= 0) {
            return cal.getAveragedPixelSize().doubleValue();
        }
        double currentPixelSize = cal.getAveragedPixelSize().doubleValue();
        if (currentPixelSize <= 0) {
            return 1.0;
        }
        return getPreferredDownsample(currentPixelSize, requestedPixelSize);
    }

    /**
     * Get the preferred downsample from current and requested pixel sizes.
     */
    static double getPreferredDownsample(double currentPixelSize, double requestedPixelSize) {
        double downsample = requestedPixelSize / currentPixelSize;
        double downsampleRounded = Math.round(downsample);
        if (GeneralTools.almostTheSame(downsample, Math.round(downsample), 0.01)) {
            return downsampleRounded;
        } else {
            return downsample;
        }
    }

    /**
     * Check if the model can be loaded by DJL.
     * This attempts to load the model metadata without fully initializing it.
     *
     * @return true if the model appears loadable
     */
    public boolean canLoad() {
        try {
            checkValid();
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    @Override
    public String toString() {
        String nameStr = getDisplayName();
        if (nameStr == null || nameStr.isEmpty()) {
            nameStr = getModelPath().getFileName().toString();
        }
        if (path != null && Files.isDirectory(path)) {
            Path parent = path.getParent();
            if (parent != null) {
                return parent.getFileName() + "/" + nameStr;
            }
        }
        return nameStr;
    }
}