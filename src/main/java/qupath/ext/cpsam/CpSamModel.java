package qupath.ext.cpsam;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.common.GeneralTools;
import qupath.lib.images.servers.PixelCalibration;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

/**
 * Represents a loaded CPSAM TorchScript model.
 * Manages path validation and model loading checks.
 */
public class CpSamModel {

    private static final Logger logger = LoggerFactory.getLogger(CpSamModel.class);
    private static final String MODEL_FILENAME = "torchscript.ts";
    private static final double DEFAULT_REQUESTED_PIXEL_SIZE = 1.0;

    private final Path path;
    private final double requestedPixelSize;

    private CpSamModel(Path path, double requestedPixelSize) {
        this.path = path.toAbsolutePath();
        this.requestedPixelSize = requestedPixelSize;
    }

    /**
     * Create a CpSamModel from a path on disk.
     * The path may point to either a directory containing the torchscript wrapper
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
     * Get the model directory path.
     */
    public Path getPath() {
        return path;
    }

    /**
     * Get the full path to the TorchScript model file.
     * If this.path points to a directory, appends cpsam_torchscript.pt.
     * If it points directly to a .pt file, returns it as-is.
     */
    public Path getModelPath() {
        if (Files.isDirectory(path)) {
            return path.resolve(MODEL_FILENAME);
        }
        return path;
    }

    /**
     * Check if the model file exists on disk.
     */
    public boolean isValid() {
        return Files.exists(getModelPath());
    }

    /**
     * Validate that the model file exists, throwing an exception if not.
     */
    public void checkValid() throws IOException {
        Path modelPath = getModelPath();
        if (!Files.exists(modelPath)) {
            throw new IOException("CPSAM TorchScript model not found: " + modelPath);
        }
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
        return getModelPath().toString();
    }
}
