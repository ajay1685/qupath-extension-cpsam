package qupath.ext.cpsam;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.TaskRunnerFX;
import qupath.lib.plugins.CommandLineTaskRunner;
import qupath.lib.plugins.SimpleProgressMonitor;
import qupath.lib.regions.ImageRegion;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * A {@link TaskRunnerFX} that wraps the progress monitor to inject a real-time
 * object detection count into the progress message.
 * <p>
 * Instead of the default "Completed annotation [x-y] N%", the progress label shows
 * "47 objects detected (45%)".
 */
class CpSamTaskRunner extends TaskRunnerFX {

    private static final Logger logger = LoggerFactory.getLogger(CpSamTaskRunner.class);

    /** Shared counter incremented by {@link PruneObjectOutputHandler} after each tile. */
    private final AtomicInteger detectedObjectCount;

    /**
     * @param qupath                the QuPath instance
     * @param nThreads              number of parallel threads
     * @param detectedObjectCount   shared counter for real-time object count (may be null)
     */
    CpSamTaskRunner(QuPathGUI qupath, int nThreads, AtomicInteger detectedObjectCount) {
        super(qupath, nThreads);
        this.detectedObjectCount = detectedObjectCount;
    }

    @Override
    public SimpleProgressMonitor makeProgressMonitor() {
        SimpleProgressMonitor parent = super.makeProgressMonitor();

        // If no counter, or we got a command-line monitor (headless), use as-is
        if (detectedObjectCount == null || parent instanceof CommandLineTaskRunner.CommandLineProgressMonitor) {
            return parent;
        }

        // Wrap to inject object count into progress messages
        return new CountingProgressMonitor(parent, detectedObjectCount);
    }

    /**
     * Wraps a {@link SimpleProgressMonitor} and replaces the progress message with
     * a real-time object detection count.
     */
    private static class CountingProgressMonitor implements SimpleProgressMonitor {

        private final SimpleProgressMonitor delegate;
        private final AtomicInteger count;

        CountingProgressMonitor(SimpleProgressMonitor delegate, AtomicInteger count) {
            this.delegate = delegate;
            this.count = count;
        }

        @Override
        public void startMonitoring(String message, int maxProgress, boolean mayCancel) {
            delegate.startMonitoring(message, maxProgress, mayCancel);
        }

        @Override
        public void updateProgress(int increment, String message, ImageRegion region) {
            int objects = count.get();
            String countMessage = objects + " objects detected";
            delegate.updateProgress(increment, countMessage, region);
        }

        @Override
        public void pluginCompleted(String message) {
            delegate.pluginCompleted(message);
        }

        @Override
        public boolean cancelled() {
            return delegate.cancelled();
        }
    }
}
