package qupath.ext.cpsam;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.analysis.features.ObjectMeasurements;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.TransformedServerBuilder;
import qupath.lib.objects.PathObject;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Executors;

/**
 * Post-detection measurement utilities for CPSAM.
 * <p>
 * Uses the per-object {@link ObjectMeasurements} API for QuPath 0.7.x compatibility.
 */
class CpSamMeasurementsOld {

    private static final Logger logger = LoggerFactory.getLogger(CpSamMeasurementsOld.class);

    // Explicit lists instead of ALL_MEASUREMENTS / ALL_COMPARTMENTS (QuPath ≥ 0.8.0 only)
    private static final List<ObjectMeasurements.Measurements> ALL_MEASUREMENTS =
            Arrays.asList(ObjectMeasurements.Measurements.values());
    private static final List<ObjectMeasurements.Compartments> ALL_COMPARTMENTS =
            Arrays.asList(ObjectMeasurements.Compartments.values());

    private CpSamMeasurementsOld() {}

    /**
     * Add shape measurements (area, length, circularity, etc.) to all objects.
     * This is purely geometry-based and requires no pixel reads.
     *
     * @param imageData the image whose pixel calibration is used for physical-unit measurements
     * @param objects   the detected objects to measure
     */
    static void addShapeMeasurements(ImageData<BufferedImage> imageData,
                                     Collection<? extends PathObject> objects) {
        if (objects.isEmpty()) return;
        var cal = imageData.getServer().getPixelCalibration();
        ObjectMeasurements.addShapeMeasurements(objects, cal);
        logger.debug("Added shape measurements to {} objects", objects.size());
    }

    /**
     * Add per-channel intensity measurements to all objects.
     * <p>
     * Uses the single-object API in a parallel stream for QuPath 0.7.x compatibility.
     * Compartments are silently ignored for non-cell objects.
     *
     * @param imageData  the image to read pixel values from
     * @param objects    the detected objects to measure
     * @param downsample the resolution at which to read pixels (should match inference downsample)
     * @throws IOException if a tile read fails
     */
    static void addIntensityMeasurements(ImageData<BufferedImage> imageData,
                                         Collection<? extends PathObject> objects,
                                         double downsample) throws IOException {
        if (objects.isEmpty()) return;
        var server = imageData.getServer();
        logger.debug("Adding intensity measurements to {} objects at downsample={}", objects.size(), downsample);
        try {
            objects.parallelStream().forEach(obj -> {
                try {
                    ObjectMeasurements.addIntensityMeasurements(
                            server, obj, downsample, ALL_MEASUREMENTS, ALL_COMPARTMENTS);
                } catch (IOException e) {
                    throw new RuntimeException("Failed to measure object: " + e.getMessage(), e);
                }
            });
        } catch (RuntimeException e) {
            if (e.getCause() instanceof IOException ioe) throw ioe;
            throw e;
        }
        logger.debug("Intensity measurements complete for {} objects", objects.size());
    }
}

/**
 * Object measurement utilities for CPSAM.
 * <p>
 * Uses the new {@link ObjectMeasurements} API introduced in QuPath v0.8.0
 * (PR #2113) which partitions objects into spatially-coherent batches to minimise
 * image reads, and optionally parallelises across a thread pool.
 */
class CpSamMeasurements {

    private static final Logger logger = LoggerFactory.getLogger(CpSamMeasurements.class);

    private CpSamMeasurements() {}

    /**
     * Add shape measurements (area, length, circularity, etc.) to all objects.
     * This is purely geometry-based and requires no pixel reads.
     *
     * @param imageData the image whose pixel calibration is used for physical-unit measurements
     * @param objects   the detected objects to measure
     */
    static void addShapeMeasurements(ImageData<BufferedImage> imageData,
                                     Collection<? extends PathObject> objects) {
        if (objects.isEmpty()) return;
        var cal = imageData.getServer().getPixelCalibration();
        // Passing no ShapeFeatures varargs defaults to ALL_SHAPE_FEATURES inside ObjectMeasurements
        ObjectMeasurements.addShapeMeasurements(objects, cal);
        logger.debug("Added shape measurements to {} objects", objects.size());
    }

    /**
     * Add per-channel intensity measurements to all objects using the batched API.
     * <p>
     * Objects are partitioned into spatially-coherent batches to minimise image tile reads.
     * A fixed thread pool of {@code nThreads} workers is used so batches are processed
     * in parallel. The method blocks until all batches complete.
     * <p>
     * All {@link ObjectMeasurements.Measurements} (mean, median, min, max, std-dev) and all
     * {@link ObjectMeasurements.Compartments} (nucleus, cytoplasm, cell, membrane) are measured.
     * Compartments are silently ignored for non-cell objects.
     * <p>
     * If color deconvolution stains are available in the image, the measurement server is
     * transformed to use all non-residual deconvolution channels.
     *
     * @param imageData  the image to read pixel values from
     * @param objects    the detected objects to measure
     * @param downsample the resolution at which to read pixels (should match inference downsample)
     * @param nThreads   number of parallel worker threads; clamped to at least 1
     * @throws IOException if a tile read fails
     */
    static void addIntensityMeasurements(ImageData<BufferedImage> imageData,
                                         Collection<? extends PathObject> objects,
                                         double downsample,
                                         int nThreads) throws IOException {
        if (objects.isEmpty()) return;
        var server = imageData.getServer();
        int threads = Math.max(1, nThreads);
        logger.debug("Adding intensity measurements to {} objects using {} thread(s) at downsample={}",
                objects.size(), threads, downsample);

        // Apply color deconvolution to measurement server if stains are available
        var stains = imageData.getColorDeconvolutionStains();
        var builder = new TransformedServerBuilder(server);
        if (stains != null) {
            List<Integer> stainNumbers = new ArrayList<>();
            for (int s = 1; s <= 3; s++) {
                if (!stains.getStain(s).isResidual())
                    stainNumbers.add(s);
            }
            if (!stainNumbers.isEmpty()) {
                builder.deconvolveStains(stains, stainNumbers.stream().mapToInt(i -> i).toArray());
            }
        }

        try {
            var server2 = builder.build();
            try (var pool = Executors.newFixedThreadPool(threads)) {
                ObjectMeasurements.addIntensityMeasurements(
                        server2,
                        objects,
                        downsample,
                        ObjectMeasurements.ALL_MEASUREMENTS,
                        ObjectMeasurements.ALL_COMPARTMENTS,
                        pool);
            }
            // server2 is closed here if it's different from server
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Failed to measure intensity: " + e.getMessage(), e);
        }
        logger.debug("Intensity measurements complete for {} objects", objects.size());
    }
}
