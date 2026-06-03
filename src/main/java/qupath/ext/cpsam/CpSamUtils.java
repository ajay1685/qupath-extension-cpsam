package qupath.ext.cpsam;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
}
