package qupath.ext.cpsam;

import ai.djl.ndarray.NDArray;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.experimental.pixels.OutputHandler;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathCellObject;
import qupath.lib.objects.PathDetectionObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.utils.ObjectMerger;
import qupath.lib.objects.utils.ObjectProcessor;
import qupath.lib.objects.utils.OverlapFixer;

/**
 * Post-processing strategy for CpSam inference.
 * <p>
 * Encapsulates two concerns that together ensure clean tile-boundary handling:
 * <ol>
 *   <li><b>Output handler</b> — {@link PruneObjectOutputHandler}: discards detections whose
 *       bounding box touches the padded tile fetch boundary (within {@code pruneThreshold} pixels),
 *       then clips surviving objects to the parent annotation ROI.  This prevents objects from
 *       appearing outside the annotation and eliminates cells that were only partially sampled at
 *       the very edge of the fetched tile.</li>
 *   <li><b>Post-processor (merger + overlap fixer)</b>:
 *       <ul>
 *         <li>{@code IoMinMerger}: deduplicates cross-tile detections of the same cell.  A partial
 *             detection B that is geometrically contained within a full detection A has
 *             {@code IoMin = intersection / min(area_A, area_B) = 1.0}, so it is always merged
 *             regardless of the area ratio.  {@code IoUMerger} fails for containment because
 *             {@code IoU = area(B) / area(A)} can be well below 0.5 for small partial detections.
 *             Additionally, JTS {@code overlaps()} returns {@code false} for containment, so
 *             {@link OverlapFixer} would silently skip such pairs, leaving the small fragment
 *             visible inside the larger cell.</li>
 *         <li>{@link OverlapFixer}: clips any remaining true overlaps that survive the merger.</li>
 *       </ul>
 *   </li>
 * </ol>
 * <p>
 * <b>Why NOT {@code createSharedTileBoundaryMerger}:</b>
 * There are no background pixels between adjacent touching cells.
 * Two touching cells therefore share an exact pixel boundary, which the boundary
 * merger scores as IoU ≈ 1.0 and incorrectly merges.  
 * <p>
 * Swap this class to experiment with different merging strategies without touching the rest of the
 * pipeline.
 */
class CpSamPostProcessing {

    private static final Logger logger = LoggerFactory.getLogger(CpSamPostProcessing.class);

    private final Class<? extends PathObject> preferredOutputType;
    /** Pixels from the tile fetch boundary within which detections are discarded. */
    private final int pruneThreshold;
    /** IoMin threshold for merging near-duplicate cross-tile detections. */
    private final double ioMinThreshold;

    /**
     * Default configuration: {@code pruneThreshold=1}, {@code ioMinThreshold=0.5}.
     *
     * @param preferredOutputType the QuPath object class to create (detection, annotation, or cell)
     */
    CpSamPostProcessing(Class<? extends PathObject> preferredOutputType) {
        this(preferredOutputType, 1, 0.5);
    }

    /**
     * @param preferredOutputType the QuPath object class to create
     * @param pruneThreshold      pixels from the padded tile boundary within which objects are pruned
     * @param ioMinThreshold      minimum IoMin score for two objects to be merged
     */
    CpSamPostProcessing(Class<? extends PathObject> preferredOutputType,
                        int pruneThreshold,
                        double ioMinThreshold) {
        this.preferredOutputType = preferredOutputType;
        this.pruneThreshold      = pruneThreshold;
        this.ioMinThreshold      = ioMinThreshold;

        if (CpSamPreferences.verboseLoggingProperty().get()) {
            logger.info("CpSamPostProcessing: outputType={}, pruneThreshold={} px, ioMinThreshold={}",
                    outputTypeName(), pruneThreshold, ioMinThreshold);
        }
    }

    /**
     * Creates the {@link OutputHandler} used per tile.
     * <p>
     * Uses {@link PruneObjectOutputHandler} which drops objects touching the padded tile
     * boundary, then clips survivors to the annotation ROI.
     */
    OutputHandler<Mat, Mat, NDArray[]> createOutputHandler() {
        return new PruneObjectOutputHandler<>(
                new MaskToObjectConverter(preferredOutputType),
                pruneThreshold);
    }

    /**
     * Creates the post-processor applied after all tiles complete.
     * <p>
     * Pipeline: {@code IoMinMerger(ioMinThreshold)} → {@link OverlapFixer#clipOverlaps()}.
     */
    ObjectProcessor createPostProcessor() {
        return ObjectMerger.createIoMinMerger(ioMinThreshold)
                .andThen(OverlapFixer.builder()
                        .clipOverlaps()
                        .keepFragments(false)
                        .sortBySolidity()
                        .build());
    }

    Class<? extends PathObject> getPreferredOutputType() { return preferredOutputType; }
    int    getPruneThreshold() { return pruneThreshold; }
    double getIoMinThreshold() { return ioMinThreshold; }

    private String outputTypeName() {
        if (preferredOutputType == null || preferredOutputType == PathDetectionObject.class) return "detection";
        if (preferredOutputType == PathAnnotationObject.class) return "annotation";
        if (preferredOutputType == PathCellObject.class) return "cell";
        return preferredOutputType.getSimpleName();
    }
}
