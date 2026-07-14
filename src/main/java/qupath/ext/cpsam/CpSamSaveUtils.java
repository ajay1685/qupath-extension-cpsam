package qupath.ext.cpsam;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.gui.QuPathGUI;
import qupath.lib.regions.RegionRequest;
import qupath.opencv.tools.OpenCVTools;

import ij.IJ;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.atomic.AtomicInteger;

import org.bytedeco.opencv.opencv_core.Mat;


/**
 * Resolves the directory where preprocessed tiles are saved when
 * {@code savePreprocessedTiles} preference is enabled.
 * <p>
 * Resolution order:
 * <ol>
 *   <li>QuPath project directory (parent of the project file)</li>
 *   <li>Image file parent directory</li>
 *   <li>System temp directory</li>
 * </ol>
 * The actual path is {@code <base>/cpsam-temp/<timestamp>}.
 */
class CpSamSaveUtils {

    private static final Logger logger = LoggerFactory.getLogger(CpSamSaveUtils.class);
    private static final String FOLDER_NAME = "cpsam-temp";

    private static final AtomicInteger saveTileIndex = new AtomicInteger(0);

    private CpSamSaveUtils() {}

    /**
     * Create and return the tile save directory, or {@code null} if creation fails.
     *
     * @param imageData the current image (used to derive a fallback base directory)
     * @return the created directory, or {@code null} on failure
     */
    static Path create(qupath.lib.images.ImageData<?> imageData) {
        Path baseDir = resolveBaseDir(imageData);
        if (baseDir == null) {
            logger.warn("Could not resolve a base directory for tile saving");
            return null;
        }
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        Path saveDir = baseDir.resolve(FOLDER_NAME);//.resolve(timestamp);
        try {
            Files.createDirectories(saveDir);
            logger.info("Saving preprocessed tiles to: {}", saveDir);
            return saveDir;
        } catch (IOException e) {
            logger.warn("Could not create cpsam-temp directory; tile saving disabled: {}", e.getMessage());
            return null;
        }
    }

    private static Path resolveBaseDir(qupath.lib.images.ImageData<?> imageData) {
        // 1. QuPath project directory
        var guiInstance = QuPathGUI.getInstance();
        var project = guiInstance != null ? guiInstance.getProject() : null;
        if (project != null && project.getPath() != null) {
            return project.getPath().getParent();
        }
        // 2. Image file parent directory
        try {
            URI imageUri = imageData.getServer().getURIs().stream().findFirst().orElse(null);
            if (imageUri != null && "file".equals(imageUri.getScheme())) {
                return Path.of(imageUri).getParent();
            }
        } catch (Exception e) {
            logger.debug("Could not resolve image URI: {}", e.getMessage());
        }
        // 3. System temp
        return Path.of(System.getProperty("java.io.tmpdir"));
    }

    /**
     * Saves the normalized tile mat as a 32-bit TIFF image in {@link #saveDir} for
     * preprocessing inspection. Uses ImageJ's IJ.save(), which correctly handles
     * any channel count (including 2-channel images) and preserves float32 precision.
     */
    static void savePreprocessedTile(Mat mat, RegionRequest region, Path saveDir) {
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

    static void resetTileIndex() {
        saveTileIndex.set(0);
    }


    /**
     * Deletes all files in the temporary tile save directory.
     * <p>
     * This is called after inference completes, to avoid leaving large numbers of
     * temporary files on disk. If the directory does not exist, or deletion fails,
     * a warning is logged but no exception is thrown.
     *
     * @param saveDir the directory to clear
     */
    static void clearTempFiles(Path saveDir) {
        if (saveDir != null && Files.exists(saveDir)) {
            //try {
                //Files.walk(saveDir)
                        //.sorted((a, b) -> b.compareTo(a)) // delete children first
                        //.forEach(path -> {
                            //try {
                                //Files.delete(path);
                            //} catch (IOException e) {
                                //logger.warn("Failed to delete temp file {}: {}", path, e.getMessage());
                            //}
                        //});
                //logger.info("Cleared temporary tile save directory: {}", saveDir);
            //} catch (IOException e) {
                //logger.warn("Failed to clear temporary tile save directory {}: {}", saveDir, e.getMessage());
            //}
        
            try (var paths = Files.list(saveDir)) { // top-level only
                paths.filter(Files::isRegularFile)
                    .filter(p -> {
                        String name = p.getFileName().toString().toLowerCase();
                        return name.endsWith(".tif") || name.endsWith(".tiff");
                    })
                    .forEach(p -> {
                        try {
                            Files.deleteIfExists(p);
                        } catch (IOException e) {
                            //throw new RuntimeException("Failed to delete " + p, e);
                            logger.warn("Failed to delete temp file {}: {}", p, e.getMessage());
                        }
                    });
                logger.info("Cleared temporary tile save directory: {}", saveDir);
            }catch (IOException e) {
                logger.warn("Failed to clear temporary tile save directory {}: {}", saveDir, e.getMessage());
            }
    
        }
    }
}
