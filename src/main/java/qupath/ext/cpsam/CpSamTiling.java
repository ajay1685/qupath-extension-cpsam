package qupath.ext.cpsam;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.objects.utils.Tiler;
import qupath.ext.cpsam.ui.CpSamPreferences;

/**
 * Tiling strategy for CpSam inference.
 * <p>
 * Encapsulates how a large annotation region is divided into tiles for the PixelProcessor.
 * Follows the same convention as the cellpose QuPath extension:
 * {@code tileDims} is the <em>non-overlapping step</em> between tile centres (the content each
 * tile uniquely covers). Each tile is then fetched with {@code padding} extra pixels of context
 * on each side, so the model receives images of size {@code tileDims + 2*padding}. Adjacent tiles
 * therefore overlap by {@code 2*padding} full-resolution pixels.
 * <p>
 * With this convention a 2000×2000 ROI, tileDims=1024 and padding=64 produces a 2×2 grid
 * (4 tiles), matching the cellpose extension. The old convention (step = tileDims − 2×padding)
 * produced a 3×3 grid (9 tiles) for the same parameters.
 * <p>
 * Swap this class to experiment with different tiling approaches without touching the rest of the
 * pipeline.
 */
class CpSamTiling {

    private static final Logger logger = LoggerFactory.getLogger(CpSamTiling.class);

    private final int tileDims;
    private final int padding;

    /**
     * @param tileDims  non-overlapping step size in pixels; each tile covers this many unique pixels
     *                  (matches the cellpose extension's definition of "tile size")
     * @param padding   context added to each side of the step region in pixels; the model therefore
     *                  receives tiles of size {@code tileDims + 2*padding}
     */
    CpSamTiling(int tileDims, int padding) {
        this.tileDims = tileDims;
        this.padding  = padding;
    }

    /**
     * Build a {@link Tiler} configured with centre alignment and no cropping.
     *
     * @param effectiveDownsample downsample factor applied when reading image tiles
     * @return a ready-to-use {@link Tiler}
     */
    Tiler createTiler(double effectiveDownsample) {
        int sizeWithoutPadding = computeStepSize(effectiveDownsample);
        if (sizeWithoutPadding <= 0) {
            logger.warn("CpSamTiling: padding ({}) is too large for tileDims ({}); "
                    + "step will be clamped to 1 — reduce padding or increase tile size",
                    padding, tileDims);
        }

        int step = Math.max(1, sizeWithoutPadding);

        if (CpSamPreferences.verboseLoggingProperty().get()) {
            int fullResPad = getFullResPadding(effectiveDownsample);
            logger.info("CpSamTiling: tileStep={} px (non-overlapping), modelInputSize={} px, "
                    + "fullResPadding={} px, fullResOverlap={} px",
                    step, tileDims + padding * 2,
                    fullResPad, fullResPad * 2);
        }

        return Tiler.builder(step)
                .alignCenter()
                .cropTiles(false)
                .build();
    }

    /**
     * Returns the padding to pass to {@link qupath.lib.experimental.pixels.PixelProcessor.Builder#padding(int)}.
     * This is the per-side overlap in <em>full-resolution</em> pixels.
     *
     * @param effectiveDownsample downsample factor applied when reading image tiles
     * @return padding in full-resolution pixels
     */
    int getFullResPadding(double effectiveDownsample) {
        return (int) Math.round(padding * effectiveDownsample);
    }

    /**
     * Step size between tile centres in full-resolution pixels.
     * Equals {@code tileDims * effectiveDownsample} — the non-overlapping content per tile.
     */
    private int computeStepSize(double effectiveDownsample) {
        return (int) Math.round(effectiveDownsample * tileDims);
    }

    int getTileDims() { return tileDims; }
    int getPadding()  { return padding;  }
}
