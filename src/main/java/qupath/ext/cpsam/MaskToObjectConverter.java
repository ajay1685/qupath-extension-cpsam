package qupath.ext.cpsam;

import ai.djl.ndarray.NDArray;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.djl.DjlTools;
import qupath.lib.analysis.images.ContourTracing;
import qupath.lib.experimental.pixels.OutputHandler;
import qupath.lib.experimental.pixels.Parameters;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathCellObject;
import qupath.lib.objects.PathDetectionObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.roi.interfaces.ROI;
import qupath.opencv.tools.OpenCVTools;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Objects;

/**
 * Converts CPSAM integer mask output (float32 NDArray [H,W]) to PathObjects.
 * Uses ContourTracing.createObjects — processes all labels in a single pass.
 */
class MaskToObjectConverter implements OutputHandler.OutputToObjectConverter<Mat, Mat, NDArray[]> {

    private static final Logger logger = LoggerFactory.getLogger(MaskToObjectConverter.class);

    private final Class<? extends PathObject> preferredObjectClass;

    MaskToObjectConverter(Class<? extends PathObject> preferredObjectClass) {
        this.preferredObjectClass = preferredObjectClass;
    }

    @Override
    public List<PathObject> convertToObjects(Parameters<Mat, Mat> params, NDArray[] output) {
        if (output == null || output[0] == null) {
            return List.of();
        }

        NDArray maskND = output[0]; // [H, W] float32 with integer labels (0 = background)

        // Convert NDArray to OpenCV Mat (CV_32F, single channel), then to SimpleImage.
        // ContourTracing handles all labels in one pass — no per-label threshold loops.
        try (Mat floatMat = DjlTools.ndArrayToMat(maskND, "HW")) {
            var image = OpenCVTools.matToSimpleImage(floatMat, 0);
            return ContourTracing.createObjects(
                    image,
                    params.getRegionRequest(),
                    1, -1,
                    (roi, label) -> createObject(roi, preferredObjectClass)
            );
        } catch (Exception e) {
            logger.error("Error converting mask to objects", e);
            return List.of();
        }
    }

    private static PathObject createObject(ROI roi, Class<? extends PathObject> preferredType) {
        if (preferredType == null || Objects.equals(PathDetectionObject.class, preferredType)) {
            return PathObjects.createDetectionObject(roi);
        } else if (Objects.equals(PathAnnotationObject.class, preferredType)) {
            var obj = PathObjects.createAnnotationObject(roi);
            if (obj instanceof PathAnnotationObject ann) {
                ann.setLocked(true);
            }
            return obj;
        } else if (Objects.equals(PathCellObject.class, preferredType)) {
            return PathObjects.createCellObject(roi, roi);
        } else {
            return PathObjects.createDetectionObject(roi);
        }
    }
}

