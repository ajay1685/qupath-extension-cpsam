package qupath.ext.cpsam;

import org.locationtech.jts.geom.Envelope;
import qupath.lib.experimental.pixels.OutputHandler;
import qupath.lib.experimental.pixels.Parameters;
import qupath.lib.experimental.pixels.PixelProcessorUtils;
import qupath.lib.objects.PathObject;
import qupath.lib.roi.GeometryTools;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * An output handler that prunes detections touching tile step boundaries, then masks
 * surviving objects to the parent annotation ROI.
 * <p>
 * Objects whose bounding box is within {@code boundaryThreshold} pixels of the tile/region
 * boundary are discarded — they will be fully captured by the adjacent overlapping tile.
 * Objects at the true image boundary are kept even though they touch the image edge.
 * <p>
 * This prevents two known artefacts that arise with {@code createUnmaskedObjectOutputHandler}:
 * <ol>
 *   <li>Objects outside the annotation ROI (from the padded tile area).</li>
 *   <li>Small fragments encapsulated inside larger objects at tile boundaries (caused by
 *       near-duplicate cross-tile detections being clipped rather than removed by
 *       {@code OverlapFixer.clipOverlaps()}).</li>
 * </ol>
 * Ported from {@code qupath-extension-instanseg} {@code PruneObjectOutputHandler}.
 */
class PruneObjectOutputHandler<S, T, U> implements OutputHandler<S, T, U> {

    private final OutputToObjectConverter<S, T, U> converter;
    private final int boundaryThreshold;
    /** Optional shared counter incremented with object count after each tile. May be null. */
    private final AtomicInteger detectedObjectCount;

    PruneObjectOutputHandler(OutputToObjectConverter<S, T, U> converter, int boundaryThreshold,
                             AtomicInteger detectedObjectCount) {
        this.converter = converter;
        this.boundaryThreshold = boundaryThreshold;
        this.detectedObjectCount = detectedObjectCount;
    }

    @Override
    public boolean handleOutput(Parameters<S, T> params, U output) {
        if (output == null)
            return false;

        List<PathObject> newObjects = converter.convertToObjects(params, output);
        if (newObjects == null)
            return false;

        var parentOrProxy = params.getParentOrProxy();
        parentOrProxy.removeAllChildObjects();

        // Remove objects within boundaryThreshold pixels of the tile step boundary,
        // except at the image boundary itself.
        var bounds = GeometryTools.regionToEnvelope(params.getRegionRequest());
        int width = params.getServer().getWidth();
        int height = params.getServer().getHeight();

        newObjects = newObjects.parallelStream()
                .filter(p -> doesntTouchBoundaries(GeometryTools.roiToEnvelope(p.getROI()), bounds, boundaryThreshold, width, height))
                .toList();

        if (!newObjects.isEmpty()) {
            // Clip surviving objects to the parent annotation ROI — prevents objects
            // from protruding outside the annotation even at the annotation boundary.
            var parent = params.getParent().getROI();
            newObjects = newObjects.parallelStream()
                    .flatMap(p -> PixelProcessorUtils.maskObject(parent, p).stream())
                    .toList();
        }

        parentOrProxy.addChildObjects(newObjects);
        parentOrProxy.setLocked(true);

        // Update shared counter for real-time progress reporting
        if (detectedObjectCount != null) {
            detectedObjectCount.addAndGet(newObjects.size());
        }

        return true;
    }

    /**
     * Returns true if the detection should be kept (i.e. does NOT touch any relevant boundary).
     * Objects at the image edge are kept; objects at any tile/region step boundary are dropped.
     */
    private boolean doesntTouchBoundaries(Envelope det, Envelope region, int boundaryPixels,
                                          int imageWidth, int imageHeight) {
        if (touchesLeftOfImage(det, boundaryPixels)) {
            if (touchesTopOfImage(det, boundaryPixels) || touchesBottomOfImage(det, imageHeight, boundaryPixels))
                return true;
            if (!(touchesBottomOfRegion(det, region, boundaryPixels) || touchesTopOfRegion(det, region, boundaryPixels)))
                return true;
        }
        if (touchesTopOfImage(det, boundaryPixels)) {
            if (touchesLeftOfImage(det, boundaryPixels) || touchesRightOfImage(det, imageWidth, boundaryPixels))
                return true;
            if (!(touchesLeftOfRegion(det, region, boundaryPixels) || touchesRightOfRegion(det, region, boundaryPixels)))
                return true;
        }
        if (touchesRightOfImage(det, imageWidth, boundaryPixels)) {
            if (touchesTopOfImage(det, boundaryPixels) || touchesBottomOfImage(det, imageHeight, boundaryPixels))
                return true;
            if (!(touchesBottomOfRegion(det, region, boundaryPixels) || touchesTopOfRegion(det, region, boundaryPixels)))
                return true;
        }
        if (touchesBottomOfImage(det, imageHeight, boundaryPixels)) {
            if (touchesLeftOfImage(det, boundaryPixels) || touchesRightOfImage(det, imageWidth, boundaryPixels))
                return true;
            if (!(touchesLeftOfRegion(det, region, boundaryPixels) || touchesRightOfRegion(det, region, boundaryPixels)))
                return true;
        }
        return !(touchesLeftOfRegion(det, region, boundaryPixels)
                || touchesRightOfRegion(det, region, boundaryPixels)
                || touchesBottomOfRegion(det, region, boundaryPixels)
                || touchesTopOfRegion(det, region, boundaryPixels));
    }

    private static boolean touchesLeftOfImage(Envelope det, int boundary) {
        return det.getMinX() < boundary;
    }

    private static boolean touchesRightOfImage(Envelope det, int width, int boundary) {
        return width - det.getMaxX() < boundary;
    }

    private static boolean touchesTopOfImage(Envelope det, int boundary) {
        return det.getMinY() < boundary;
    }

    private static boolean touchesBottomOfImage(Envelope det, int height, int boundary) {
        return height - det.getMaxY() < boundary;
    }

    private static boolean touchesLeftOfRegion(Envelope det, Envelope region, int boundary) {
        return det.getMinX() - region.getMinX() < boundary;
    }

    private static boolean touchesRightOfRegion(Envelope det, Envelope region, int boundary) {
        return region.getMaxX() - det.getMaxX() < boundary;
    }

    private static boolean touchesTopOfRegion(Envelope det, Envelope region, int boundary) {
        return det.getMinY() - region.getMinY() < boundary;
    }

    private static boolean touchesBottomOfRegion(Envelope det, Envelope region, int boundary) {
        return region.getMaxY() - det.getMaxY() < boundary;
    }
}
