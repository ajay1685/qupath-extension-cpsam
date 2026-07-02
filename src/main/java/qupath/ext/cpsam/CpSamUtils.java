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
import qupath.ext.djl.DjlTools;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Method;
import java.util.concurrent.TimeUnit;

/**
 * General-purpose utilities for the CpSam extension.
 */
class CpSamUtils {

    private static final Logger logger = LoggerFactory.getLogger(CpSamUtils.class);

    private CpSamUtils() {}

    /**
     * Calls the PyTorch equivalent of {@code torch.cuda.empty_cache()} via DJL's internal JNI.
     * <p>
     * PyTorch's CUDA allocator keeps freed memory blocks in a cache for fast reuse.  After a run,
     * nvidia-smi will still show high VRAM usage even though the tensors are logically freed, because
     * the allocator hasn't returned the blocks to CUDA.  Calling this method forces the release so
     * the next run starts with clean VRAM.
     * <p>
     * Uses reflection so the extension compiles without a hard dependency on the PyTorch engine module.
     * Silently skips if PyTorch JNI is not on the class path (CPU / non-PyTorch backend).
     */
    static void emptyCudaCache() {
        try {
            Class<?> jniUtils = Class.forName("ai.djl.pytorch.jni.JniUtils");
            Method m = jniUtils.getDeclaredMethod("emptyCudaCache");
            m.setAccessible(true);
            m.invoke(null);
            logger.info("CUDA allocator cache cleared (torch.cuda.empty_cache())");
        } catch (ClassNotFoundException e) {
            logger.debug("PyTorch JNI not on classpath — skipping CUDA cache clear");
        } catch (NoSuchMethodException e) {
            logger.debug("JniUtils.emptyCache() not found in this DJL version — skipping");
        } catch (Exception e) {
            logger.debug("Could not clear CUDA allocator cache: {}", e.getMessage());
        }
    }

    /**
     * Queries GPU VRAM usage via nvidia-smi and logs total/used/free for each GPU at INFO level.
     * Silently skips if nvidia-smi is not available (e.g. CPU-only system or not in PATH).
     *
     * @param context a short label included in the log message to identify the call site
     */
    static void logVramUsage(String context) {
        try {
            Process proc = new ProcessBuilder(
                    "nvidia-smi",
                    "--query-gpu=index,memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits")
                    .redirectErrorStream(true)
                    .start();

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(proc.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;
                    String[] parts = line.split(",");
                    if (parts.length >= 4) {
                        int gpu    = Integer.parseInt(parts[0].trim());
                        long total = Long.parseLong(parts[1].trim());
                        long used  = Long.parseLong(parts[2].trim());
                        long free  = Long.parseLong(parts[3].trim());
                        logger.info("VRAM [{}] GPU {}: total={} MiB, used={} MiB, free={} MiB",
                                context, gpu, total, used, free);
                    } else {
                        logger.info("VRAM [{}]: {}", context, line);
                    }
                }
            }
            proc.waitFor(3, TimeUnit.SECONDS);
        } catch (Exception e) {
            logger.debug("VRAM query unavailable: {}", e.getMessage());
        }
    }

    /**
     * Convert a normalized float32 Mat to the model's expected [1, 3, H, W] BCHW tensor.
     *
     * The TorchScript wrapper always requires exactly 3 channels (SAM image encoder).
     * Channel mapping matches Python's {@code transforms.convert_image()}:
     * <ul>
     *   <li>C == 3: passed through unchanged</li>
     *   <li>C &lt; 3: channels 0..C-1 copied, remaining channels zero-padded</li>
     *   <li>C &gt; 3: only the first 3 channels are used (extra channels discarded)</li>
     * </ul>
     *
     * NOTE: DjlTools.matToNDArray with layout "CHW" routes through opencv_dnn.blobFromImage()
     * and is safe here because preprocessing has already converted the tile to float32.
     */
    static NDArray matToBatchInput(Mat mat, NDManager manager) {
        NDArray chw = DjlTools.matToNDArray(manager, mat, "CHW").toType(DataType.FLOAT32, false);
        chw = enforceThreeChannels(chw, manager);
        return chw.expandDims(0);
    }

    /**
     * Ensures a CHW NDArray has exactly 3 channels.
     * Extra channels beyond 3 are dropped; missing channels are zero-padded.
     */
    static NDArray enforceThreeChannels(NDArray chw, NDManager manager) {
        int c = (int) chw.getShape().get(0);
        if (c == 3) return chw;

        long h = chw.getShape().get(1);
        long w = chw.getShape().get(2);

        if (c > 3) {
            logger.warn("Image has {} channels \u2014 only the first 3 will be sent to the model", c);
        } else {
            logger.warn("Image has {} channel(s) \u2014 zero-padding to 3 channels for the model", c);
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
     * Logs per-channel min/max values of a CHW NDArray at INFO level.
     */
    static void logPerChannelMinMax(String label, NDArray chw) {
        int channels = (int) chw.getShape().get(0);
        for (int ch = 0; ch < channels; ch++) {
            NDArray channel = chw.get(ch);
            logger.info("{} channel {}: min={}, max={}", label, ch, channel.min().getFloat(), channel.max().getFloat());
        }
    }

}
