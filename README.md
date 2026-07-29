# CPSAM Extension for QuPath

A QuPath extension for running TorchScript-based Cellpose-SAM models (CPSAM and CPDINO) for cell segmentation in brightfiled and fluorescence images including whole slide images.

## Screenshot

![Preview](assets/Screenshot-0.PNG)

## Torchscript Models

| Source Model Weights | Backbone | Devices | Torchscript variants | Data type
|-------|----------|---------|---------|---------
| **CPSAM** | SAM ViT-L | Device agnostic | cpsam_wrapper.ts | Float32
| **CPSAM_V2** | SAM ViT-L | Device agnostic | cpsam_v2_wrapper.ts | Float32
| **CPDINO (VIT-L)** | DINOv3 (vitl) | CUDA | cpdino_vitl_wrapper_cuda.ts | Float32
|                    |               | CPU |  cpdino_vitl_wrapper_cpu.ts | Float32
| **CPDINO (VIT-B)** | DINOv3 (vitb) | CUDA | cpdino_vitb_wrapper_cuda.ts | Float32
|                    |               | CPU | cpdino_vitb_wrapper_cpu.ts | Float32

> [!IMPORTANT]
> - Inference is only available for 2D image data, no support for 3D datasets is available or planned.



## Prerequisites

- **QuPath 0.8.0+** (depends on experimental PixelProcessor and ObjectMeasurements api)
- **PyTorch engine** Please check: _Extensions → Deep Java Library → Manage DJL Engines_
- **GPU acceleration**: CUDA and cuDNN are required (tested with CUDA 12.8 and cuDNN 9.10). Refer to qupath documentation [QuPath docs for gpu](https://qupath.readthedocs.io/en/stable/docs/deep/gpu.html)
- While not required it is highly recommended to use these models with GPU, transformer based models can be very slow on CPU.

## Installation

### Installation via QuPath's extension manager.
Simple install method coming soon...

### Manual install

1. Build the extension:
   ```bash
   cd qupath-extension-cpsam
   gradlew build
   ```
2. Copy `build/libs/qupath-extension-cpsam-*.jar` to your QuPath extension directory, drag it on the QuPath UI.


## Usage
### QuPath
1. Open an image in QuPath (brightfield or fluorescence)
2. Create and/or select one or more _Annotation_ to process
3. Launch the extension: _Extensions → CPSAM → Run CPSAM_

### CPSAM Parameters
Configure parameters in the dialog:
   - **Model path**: Select a `.ts` TorchScript model file
   - **Device**: Select cpu, cuda (for nvidia GPUs) or mps (Apple Silicon). Apple silicon support pending...
   - **Diameter**: Expected cell diameter in pixel units
   - **Cell Probability**: Cell probability threshold between -5.0 and 5.0, defaults to 0.0 (refer to cellpose documentation for explaination).
   - **Flow Threshold**: Flow error threshold (defaults to 0.4), when set to 0 no cells are discarded. Increase flow error threshold to detect more cells (refer to cellpose documentation for explaination). Set to 0 to disable quality check and return all possible detections.
   - **Tile size / padding**: Controls tiling for large (i.e. whole slide) images.
   - **Batch size**: Increase or descrese batch size based on available GPU VRAM. 
   - **Threads**: Number of parallel threads. Increase number of threads when you have enough cpu, gpu and ram available.

### Channel Selection

The extension supports selecting 1–3 input channels from:

- **Raw image channels**: direct channel extraction by name
- **Color deconvolution channels**: Hematoxylin, Eosin, DAB, or other stains for brightfield images. Refer to QuPath's documentation for color deconvolution and seperating stains: https://qupath.readthedocs.io/en/stable/docs/tutorials/separating_stains.html#separating-stains 
- **(None)**: Channel 1 is required. Channels 2 and 3 are optional. When only one or two channels are selected, we zero-pads the other channels to force 3 channel input expected by the model.

### Post-Processing

Shape and intensity feature can be calculated on the fly:

- **Measurements**: Perform shape and intensity measurements on detected objects. New ObjectMeasurements api introduced in QuPath version 0.8.0 makes the intensity measurements faster.
- On the fly measurements will not work on QuPath version 0.7.0 or below, use _Analyze → Calculate Features → Add shape features and Add intensity features_ for measurements.

> [!WARNING]
> - On the fly measurements only works on QuPath v0.8.0 (when released) or higher. See Post-processing section above for measurements in QuPath v0.7.0 or below.

## Parameters Reference


| Parameter | Default | Description |
|-----------|---------|-------------|
| Diameter | 30.0 px | Expected cell diameter; used for image scaling before inference |
| Cell prob threshold | 0.0 | Probability logit threshold (post-sigmoid = 0.5); lower = more detections, higher = less detections and tighter bounderies |
| Flow threshold | 0.4 | Flow integration threshold; lower = stricter quality filter, zero = return all cells |
| Tile size | 1024 | Non-overlapping tile step size in pixels |
| Tile padding | 64 | Extra context pixels per tile (model receives tile + 2×padding) |
| Batch size | 4 | Models's internal batch size for CPSAM and CPDINO backbone; larger = more VRAM required |
| Threads | 2 | Parallel threads for tile processing and measurements |
| Normalization percentile (low/high) | 1 and 99 | Per-channel global percentile normalization range |

Please refer to Cellpose documentation for additional information regarding normalization, cell probability threshold, flow error threshold, niter and channels. https://cellpose.readthedocs.io/en/latest/settings.html

## Troubleshooting

### Slow performance on CUDA
- Check if the GPU memory usage is exceeding VRAM, and consuming shared memory. Reduce **batch size**, **threads** first, you may also have to reduce the **tile size**.
- Check VRAM usage: the extension logs `nvidia-smi` output when debug is enabled in the preference pane.
- Try running on a small annotation, and/or CPU for debugging (slower but eliminates VRAM overflow issues)

### Model

- Ensure the `.ts` file is a valid TorchScript model (not a raw `.pt` checkpoint). Instructions to obtain these models are coming soon...
- CPSAM torchscript wrapper is device agnostic (same model works for cpu or cuda) use cpsam_wrapper or cpsam_v2_wrapper.
- CPDINO models are device dependent; use the correct model type and device selection (i.e. cpdino_vitl_wrapper_cuda, cpdino_vitl_wrapper_cpu, cpdino_vitb_wrapper_cuda or cpdino_vitb_wrapper_cpu).
- Verify PyTorch engine is installed: _Extensions → Deep Java Library → Manage DJL Engines_
- Verify that the PyTorch engine can use CUDA if applicable, refer to https://qupath.readthedocs.io/en/stable/docs/deep/gpu.html#checking-everything-works 

### No Detections

- Adjust **diameter** to match your cell size
- Lower **cell prob threshold** (e.g., -2.0) for more sensitive detection.
- set **Flow threshold** to zero (return as much as possible) or higher value to return more detections.
- Check the selected input channels, and test your images with **cellpose v4.x** python install. 

> [!TIP]
> Refer to Cellpose documentation, verify everything work with Cellpose python install first.

## Licensing Terms for the Torchscript Models

> [!CAUTION]
> Please consider doing your own research regarding applicable licenses.

The TorchScript models loaded by this extension are derived from the [Cellpose v4.x](https://github.com/mouseland/cellpose-sam) pre-trained weights. Those weights are from models trained on datasets with varying licensing terms. Users are responsible for ensuring their use complies with the original licenses and terms of the cellpose model weights (cpsam, cpsam_v2, cpdino, cpdino-vitb) from cellpose.

For license around the cellpose models (and derived work) please refer to Cellpose repository: [MouseLand/Cellpose](https://github.com/mouseland/cellpose)

> [!TIP]
> A discussion regarding this topic: https://forum.image.sc/t/question-about-cellpose-sam-pretrained-model-licensing-in-commercial-bioimage-software/121128

## GPL v3 License for QuPath-Extension-CPSAM (excluding torchscript models)

The extension code in this repository is licensed under the same terms as QuPath (GPL Version 3).

## Getting Help

For questions about using this extension, or to report issues please use the [image.sc forum](https://forum.image.sc/tag/qupath).


## Citations to consider

Pachitariu, M., Rariden, M., & Stringer, C. (2025). Cellpose-SAM: superhuman generalization for cellular segmentation. bioRxiv.

Stringer, C., Wang, T., Michaelos, M. et al Cellpose: a generalist algorithm for cellular segmentation. Nat Methods 18, 100–106 (2021). https://doi.org/10.1038/s41592-020-01018-x

Pachitariu, M. & Stringer, C. (2022). Cellpose 2.0: how to train your own model. Nature methods, 1-8.

Bankhead, P. et al. QuPath: Open source software for digital pathology image analysis. Scientific Reports (2017). https://doi.org/10.1038/s41598-017-17204-5

### More information regarding citation
- https://github.com/MouseLand/cellpose#citation
- https://qupath.readthedocs.io/en/stable/docs/intro/citing.html#how-to-cite-qupath

## Credits
This extension was inspired by and builds upon the work of several open-source projects from the bioimage analysis community.

- Cellpose-SAM
- QuPath
- QuPath-Extension-Cellpose
- QuPath-Extension-StarDist
- QuPath-Extension-InstanSeg
- QuPath-Extension-DJL
- QuPath-Extension-WSIInfer
- QuPath-Extension-SpotiFlow
- Several interesting discussions in the image.sc forum.

### Disclaimer
- Several large language models (both open and proprietary weights) were used in the development and implementation of various features of this extension, with human in the loop.

---

> [!NOTE]
> - Eventhough scripting works (kind of), I have not tested this extension in headless QuPath.


## Important Notes:
- This is not a full Cellpose-SAM implementation, nor is it endorsed by Dr. Pachitariu, Dr. Stringer, Dr. Rariden and their team at HHMI Janelia.  This extension offers fast and easy to use method for segmenting cells in QuPath (based on cellpose-SAM pretrained models cpsam and cpdino variants).
- This is not a replacement for _BIOP/QuPath-Extension-Cellpose_ which offers more flexibility and features. Link: 
- While the pre-processing steps for normalization and post-processing mimics cellpose and it's dynamics, it can not guarantee exact match with the cellpose output at pixel level. 
### Why?
- While limited this extension may offer a few advantages (it doesn't depend on python, offers faster inference, easy to use GUI, and it doen't save tiles/mask/temp files which may help extend SSD/HDD life). 
- My internal testing indicates higher than 99% match with cellpose output. Instructions to compare the output with cellpose output are coming soon...
- Instruction to export your own model as torchscript model will be made available depending on the interest from the community.
---


