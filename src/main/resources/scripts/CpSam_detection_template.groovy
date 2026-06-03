/**
 * CPSAM Detection Template
 * @author Ajay Zalavadia
 *
 * Runs the CPSAM TorchScript model via DJL (without Python).
 *
 * Prerequisites:
 *   - qupath-extension-cpsam installed
 *   - cpsam_wrapper.pt model file downloaded
 *   - qupath-extension-djl and PyTorch backend
 *   - system with gpu highly recommended
 *
 * For all builder options see:
 *   https://github.com/ajay1685/qupath-extension-cpsam
 */

import qupath.ext.cpsam.CpSam
import qupath.lib.images.servers.ColorTransforms
import qupath.fx.dialogs.Dialogs

// Path to the cpsam_wrapper.pt file (or a directory containing it)
def modelPath = "C:/Users/zalavaa/cellpose/notebooks/cpsam_wrapper.pt" 

// Use "gpu0" or "gpu1" for cuda, "mps" for Apple Silicon, "cpu" for CPU
def device = "gpu0"

// ── Build the CpSam detector ──────────────────────────────────────────────────
def cpsam = CpSam.builder()
    .modelPath(modelPath)
    .device(device)
    .diameter(30.0)               // Expected cell diameter in pixels at run downsample
    .cellprobThreshold(0.0f)      // Cell probability threshold; lower = more cells (range ~-6 to +6)
    .flowThreshold(0.4f)          // Flow quality threshold; higher = stricter cell shape
    .niter(200)                   // Dynamics solver iterations
    .batchSize(16)                 // VRAM ( i.e. 11GB VRAM = batchSize 16–32)
    .tileDims(1024)               // Tile size in pixels (256–4096)
    .interTilePadding(0)         // Overlap context between adjacent tiles
    .build()

// ── Run detection ─────────────────────────────────────────────────────────────
def imageData = getCurrentImageData()
// Detect on selected annotations
def pathObjects = getSelectedObjects()   
if (pathObjects.isEmpty()) {
    Dialogs.showErrorMessage("CpSam", "Please select a parent annotation first!")
    return
}

def results = cpsam.detectObjects(imageData, pathObjects)

println "CpSam done — tiles: ${results.tilesProcessed}, failed: ${results.tilesFailed}, objects: ${results.nObjects}"
