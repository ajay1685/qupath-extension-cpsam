package qupath.ext.cpsam;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.awt.common.BufferedImageTools;
import qupath.lib.experimental.pixels.MeasurementProcessor;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ColorTransforms;
import qupath.lib.images.servers.PixelType;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;
import qupath.opencv.ops.ImageOp;
import qupath.opencv.ops.ImageOps;

import java.awt.image.BufferedImage;
import java.util.Collection;

/**
 * Builds per-ROI normalization {@link ImageOp} pipelines for CpSam tiles.
 * <p>
 * Computes per-channel percentile normalization factors from the full ROI at a reduced
 * resolution and returns an {@link ImageOp} that can be applied to each tile Mat.
 */
class CpSamNormalization {

    private static final Logger logger = LoggerFactory.getLogger(CpSamNormalization.class);
    static final double EPSILON = 1e-6;

    private CpSamNormalization() {}

    /**
     * Computes per-channel percentile normalization factors from the full ROI region
     * and returns an {@link ImageOp} that can be applied to each tile Mat.
     * Falls back to a simple global percentile op if the region read fails.
     */
    static ImageOp buildNormalizationOp(
            ImageData<BufferedImage> imageData,
            ROI roi,
            Collection<ColorTransforms.ColorTransform> channels,
            double lowPerc,
            double highPerc,
            double normDownsample,
            int normMaxDimension) {

        try {
            // Resolve downsample: explicit value wins; otherwise auto from max dimension
            double downsample;
            if (!Double.isNaN(normDownsample) && normDownsample > 0) {
                downsample = normDownsample;
            } else {
                downsample = Math.max(1.0,
                        Math.max(roi.getBoundsWidth(), roi.getBoundsHeight()) / (double) normMaxDimension);
            }
            var request = RegionRequest.createInstance(imageData.getServerPath(), downsample, roi);
            BufferedImage image = imageData.getServer().readRegion(request);

            var params = channels.stream().map(colorTransform -> {
                var mask = BufferedImageTools.createROIMask(image.getWidth(), image.getHeight(), roi, request);
                float[] maskPix = ColorTransforms.createChannelExtractor(0).extractChannel(null, mask, null);
                float[] fpix = colorTransform.extractChannel(imageData.getServer(), image, null);

                int ind = 0;
                for (int i = 0; i < maskPix.length; i++) {
                    if (maskPix[i] == 255) {
                        fpix[ind] = fpix[i];
                        ind++;
                    }
                }

                double[] usePixels = new double[ind];
                for (int i = 0; i < ind; i++) {
                    usePixels[i] = fpix[i];
                }

                double lo = MeasurementProcessor.Functions.percentile(lowPerc).apply(usePixels);
                double hi = MeasurementProcessor.Functions.percentile(highPerc).apply(usePixels);
                double scale = hi > lo ? 1.0 / (hi - lo + EPSILON) : 0.0;
                double offset = -lo * scale;
                return new double[]{offset, scale, lo, hi};
            }).toList();

            return ImageOps.Core.sequential(
                    ImageOps.Core.ensureType(PixelType.FLOAT32),
                    ImageOps.Core.multiply(params.stream().mapToDouble(e -> e[1]).toArray()),
                    ImageOps.Core.add(params.stream().mapToDouble(e -> e[0]).toArray())
            );
        } catch (Exception e) {
            logger.error("Error preparing cached CPSAM normalization", e);
        }

        return ImageOps.Core.sequential(
                ImageOps.Core.ensureType(PixelType.FLOAT32),
                ImageOps.Normalize.percentile(lowPerc, highPerc, true, EPSILON)
        );
    }

}
