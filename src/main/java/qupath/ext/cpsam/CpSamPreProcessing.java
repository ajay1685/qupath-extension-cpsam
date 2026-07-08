package qupath.ext.cpsam;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ColorTransforms;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.ArrayList;
import qupath.ext.djl.DjlTools;

/**
 * Pre-processing utilities that convert a normalised tile {@link Mat} into the
 * model's expected {@code [1, 3, H, W]} BCHW input tensor.
 */
class CpSamPreProcessing {

    private static final Logger logger = LoggerFactory.getLogger(CpSamPreProcessing.class);

    private CpSamPreProcessing() {}

    /**
     * Convert a normalised float32 {@link Mat} to the model's expected [1, 3, H, W] BCHW tensor.
     *
     * <p>The TorchScript wrapper requires exactly 3 channels (SAM image encoder).
     * <ul>
     *   <li>C == 3: passed through unchanged</li>
     *   <li>C &lt; 3: channels 0..C-1 copied, remaining channels zero-padded</li>
     *   <li>C &gt; 3: only the first 3 channels are used (extra channels discarded)</li>
     * </ul>
     *
     * <p>NOTE: {@code DjlTools.matToNDArray} with layout {@code "CHW"} routes through
     * {@code opencv_dnn.blobFromImage()} and is safe here because preprocessing has already
     * converted the tile to float32.
     */
    static NDArray matToBatchInput(Mat mat, NDManager manager) {
        NDArray chw = DjlTools.matToNDArray(manager, mat, "CHW").toType(DataType.FLOAT32, false);
        chw = enforceThreeChannels(chw, manager);
        return chw.expandDims(0);
    }

    /**
     * Ensures a CHW {@link NDArray} has exactly 3 channels.
     * Extra channels beyond 3 are dropped; missing channels are zero-padded.
     */
    static NDArray enforceThreeChannels(NDArray chw, NDManager manager) {
        int c = (int) chw.getShape().get(0);
        if (c == 3) return chw;

        long h = chw.getShape().get(1);
        long w = chw.getShape().get(2);

        if (c > 3) {
            logger.debug("Image has {} channels — only the first 3 will be sent to the model", c);
        } else {
            logger.debug("Image has {} channel(s) — zero-padding to 3 channels for the model", c);
        }

        NDList channelList = new NDList(3);
        for (int i = 0; i < 3; i++) {
            channelList.add(i < c
                    ? chw.get(i).expandDims(0)
                    : manager.zeros(new Shape(1, h, w), DataType.FLOAT32));
        }
        return NDArrays.concat(channelList, 0);
    }

    /**
     * Convert a normalised float32 {@link Mat} to the model's expected [1, 3, H, W] BCHW tensor.
     * <p>
     * This is a convenience method that creates a temporary {@link NDManager} for the conversion.
     * The caller is responsible for closing the returned {@link NDArray}.
     */
    static List<ColorTransforms.ColorTransform> getInputChannels(ImageData<BufferedImage> imageData) {
        int total = imageData.getServer().nChannels();
        int nUsed = Math.min(3, total);
        if (total > 3) {
            logger.warn("Image has {} channels — only the first 3 will be used for predictions", total);
        }
        List<ColorTransforms.ColorTransform> channels = new ArrayList<>(nUsed);
        for (int i = 0; i < nUsed; i++) {
            channels.add(ColorTransforms.createChannelExtractor(i));
        }
        return channels;
    }
}
