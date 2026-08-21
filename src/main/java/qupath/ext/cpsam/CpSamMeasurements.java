package qupath.ext.cpsam;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.analysis.features.ObjectMeasurements;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.TransformedServerBuilder;
import qupath.lib.objects.PathObject;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Post-detection measurement utilities for CPSAM — fallback for QuPath &lt; 0.8.0.
 * <p>
 * Uses the per-object {@link ObjectMeasurements} API with a parallel stream,
 * which works on QuPath 0.7.x but is slower for large object counts than the
 * batched API in {@link CpSamMeasurements}. Automatically invoked by
 * {@link CpSamMeasurements} when the new API is unavailable.
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
     * If color deconvolution stains are available, the measurement server is transformed
     * to use all non-residual deconvolution channels.
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
        var server2 = builder.build();

        logger.debug("Adding intensity measurements to {} objects at downsample={}", objects.size(), downsample);
        try {
            objects.parallelStream().forEach(obj -> {
                try {
                    ObjectMeasurements.addIntensityMeasurements(
                            server2, obj, downsample, ALL_MEASUREMENTS, ALL_COMPARTMENTS);
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
 * which partitions objects into spatially-coherent batches to minimise
 * image reads, and optionally parallelises across a thread pool.
 * Automatically falls back to per-object measurements ({@link CpSamMeasurementsOld})
 * if running on QuPath &lt; 0.8.0.
 */
class CpSamMeasurements {

    private static final Logger logger = LoggerFactory.getLogger(CpSamMeasurements.class);

    // Local equivalents of ObjectMeasurements.ALL_MEASUREMENTS / ALL_COMPARTMENTS
    // (only available as static fields in QuPath 0.8.0+, so we create our own for portability)
    private static final List<ObjectMeasurements.Measurements> ALL_MEASUREMENTS =
            Arrays.asList(ObjectMeasurements.Measurements.values());
    private static final List<ObjectMeasurements.Compartments> ALL_COMPARTMENTS =
            Arrays.asList(ObjectMeasurements.Compartments.values());

    // Version detection: probe for the batched addIntensityMeasurements method (QuPath 0.8.0+).
    // Cached after first check. null = not yet checked; UNAVAILABLE sentinel = checked but not found.
    private static volatile Method batchedMeasureMethod = null;
    @SuppressWarnings("rawtypes")
    private static final Method UNAVAILABLE;
    static {
        try {
            UNAVAILABLE = String.class.getMethod("toString");
        } catch (NoSuchMethodException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

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
     * Add per-channel intensity measurements to all objects.
     * <p>
     * On QuPath &ge; 0.8.0, uses the batched {@link ObjectMeasurements} API which partitions
     * objects into spatially-coherent batches to minimise image tile reads. A fixed thread
     * pool of {@code nThreads} workers processes batches in parallel.
     * <p>
     * On QuPath &lt; 0.8.0, automatically falls back to per-object measurements via a
     * parallel stream ({@link CpSamMeasurementsOld}). The fallback is detected on first
     * call and cached for subsequent invocations.
     * <p>
     * All {@link ObjectMeasurements.Measurements} (mean, median, min, max, std-dev) and all
     * {@link ObjectMeasurements.Compartments} (nucleus, cytoplasm, cell, membrane) are measured.
     * Compartments are silently ignored for non-cell objects.
     * <p>
     * If color deconvolution stains are available in the image, the measurement server is
     * transformed to use all non-residual deconvolution channels (e.g. Hematoxylin/Eosin).
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

        // On first call, probe for the batched addIntensityMeasurements method (QuPath 0.8.0+)
        if (batchedMeasureMethod == null) {
            synchronized (CpSamMeasurements.class) {
                if (batchedMeasureMethod == null) {
                    try {
                        batchedMeasureMethod = ObjectMeasurements.class.getMethod(
                                "addIntensityMeasurements",
                                qupath.lib.images.servers.ImageServer.class,
                                Collection.class, double.class,
                                Collection.class, Collection.class,
                                Executor.class);
                        logger.debug("QuPath batched intensity measurement API available");
                    } catch (NoSuchMethodException e) {
                        batchedMeasureMethod = UNAVAILABLE; // sentinel: checked but not available
                        logger.info("QuPath batched intensity measurement API not found — falling back to per-object measurements");
                    }
                }
            }
        }

        // Fall back to per-object API if batched method unavailable
        if (batchedMeasureMethod == UNAVAILABLE) {
            CpSamMeasurementsOld.addIntensityMeasurements(imageData, objects, downsample);
            return;
        }

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
                batchedMeasureMethod.invoke(null,
                        server2,
                        objects,
                        downsample,
                        ALL_MEASUREMENTS,
                        ALL_COMPARTMENTS,
                        pool);
            }
            // server2 is closed here if it's different from server
        } catch (InvocationTargetException e) {
            Throwable cause = e.getCause();
            if (cause instanceof IOException ioe) throw ioe;
            throw new IOException("Failed to measure intensity: " + cause.getMessage(), cause);
        } catch (Exception e) {
            throw new IOException("Failed to measure intensity: " + e.getMessage(), e);
        }
        logger.debug("Intensity measurements complete for {} objects", objects.size());
    }
}
