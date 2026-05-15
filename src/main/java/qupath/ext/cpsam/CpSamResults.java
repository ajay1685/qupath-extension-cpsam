package qupath.ext.cpsam;

/**
 * Results from running CPSAM detection.
 */
public class CpSamResults {

    private final long pixelsProcessed;
    private final int tilesProcessed;
    private final int tilesFailed;
    private final int nObjects;
    private final long elapsedMs;
    private final boolean wasInterrupted;

    public CpSamResults(long pixelsProcessed, int tilesProcessed, int tilesFailed,
                        int nObjects, long elapsedMs, boolean wasInterrupted) {
        this.pixelsProcessed = pixelsProcessed;
        this.tilesProcessed = tilesProcessed;
        this.tilesFailed = tilesFailed;
        this.nObjects = nObjects;
        this.elapsedMs = elapsedMs;
        this.wasInterrupted = wasInterrupted;
    }

    public long getPixelsProcessed() { return pixelsProcessed; }
    public int getTilesProcessed() { return tilesProcessed; }
    public int getTilesFailed() { return tilesFailed; }
    public int getNObjects() { return nObjects; }
    public long getElapsedMs() { return elapsedMs; }
    public boolean wasInterrupted() { return wasInterrupted; }

    @Override
    public String toString() {
        return String.format("CPSAM Results: %d tiles (%d failed), %d objects in %d ms",
                tilesProcessed, tilesFailed, nObjects, elapsedMs);
    }
}
