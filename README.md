# UELer
Usability Enhanced Linked Viewer: a Jupyter Notebook integrated viewer for MIBI images with linked interactive plots and enhanced usability.

## Try it on Binder
You can try UELer without installation by launching it on [Binder](https://mybinder.org/v2/gh/HartmannLab/UELer/main?urlpath=%2Fdoc%2Ftree%2Fscript%2Frun_ueler_binder.ipynb):
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HartmannLab/UELer/main?urlpath=%2Fdoc%2Ftree%2Fscript%2Frun_ueler_binder.ipynb)

## Installation

### 1. Set up the environment

You can create a compatible environment using the `env/environment.yml` file provided in this repository.

1. Download the `environment.yml` file to your preferred folder.
2. Change your current directory to that folder.
3. Create the environment by running:

   ```shell
   micromamba env create --name ark-analysis-ueler --file environment.yml
   ```

### 2. Install UELer

1. Navigate to the directory where you want to install the tool, then clone the repository:

   ```shell
   git clone https://github.com/HartmannLab/UELer.git
   ```
2. Activate your environment:

   ```shell
   micromamba activate ark-analysis-ueler
   ```
3. Change into the cloned UELer directory:

   ```shell
   cd <path-to-UELer-folder>
   ```
4. Install the package in editable mode:

   ```shell
   pip install -e .
   ```

## Updating Your Environment for v0.1.7-alpha (or earlier) users
If you're using UELer v0.1.7-alpha or earlier, you'll need to update your environment by following these steps:
1. Activate your environment:
```shell
micromamba activate <your-environment-directory>
```
2. Install Dask:
```
micromamba install dask
```
3. Install Dask-Image:
```
micromamba install dask-image
```
After completing these steps, your environment should be ready to go!

### Upgrade UELer
To update UELer, navigate to your UELer directory and run:
```shell
git pull
```

## Getting started
1. Open your favorite editor that supports Jupyter notebook.
2. Navigate to the cloned UELer repository, then open the notebook `/script/run_ueler.ipynb`.
3. Select the kernel for an ark-analysis compatible conda/micromamba env.
4. Change the lines according to the instructions in the notebook: when configuring the `/script/run_ueler.ipynb`, ensure that you specify the following directory paths:
  - **`base_folder`**: The directory containing the FOV (Field of View) folders with image data (e.g., `.../image_data`).
  - **`masks_folder`** (optional): The directory containing the segmentation `.tif` files for cell segmentation (e.g., `.../segmentation/cellpose_output`).
  - **`annotations_folder`** (optional): The directory containing annotation files for marking regions of interest (e.g., `.../annotations`).
  - **`cell_table_path`** (optional): The path to the file containing the cell table data (e.g., `.../segmentation/cell_table/cell_table_size_normalized.csv`).
Make sure these paths are correctly set in the notebook for the viewer to access the data correctly.

5. Run the code and you will see the viewer displayed.

### Streaming from the BioImage Archive (BIA)
You can explore a public BioImage Archive study (an `S-BIAD*` accession) without downloading the
whole dataset first:
```python
from ueler.runner import run_viewer_bia

viewer = run_viewer_bia(
    "S-BIAD2557",                      # accession id (or a direct HTTPS base URL)
    descriptor={                        # optional; auto-detection is attempted if omitted
        "mode": "folder",
        "base": "Files/spatial_murine_iCCAvsHCC/image_data",
        "mask_dir": "Files/spatial_murine_iCCAvsHCC/segmentation/cleaned_mask",
        "mask_glob": "{fov}_*.tiff",
    },
)
```
Because BIA studies have no standard folder layout, a small JSON **descriptor** (a dict or a path
to a `.json` file) maps the study files onto FOVs / channels / masks; when omitted, UELer attempts
to auto-detect the folder-per-FOV, OME-TIFF-per-FOV, or zip-container layouts. The descriptor is
flexible enough for the variation seen across real studies:
- **Masks** accept either a single `mask_dir`/`mask_glob`, or a `masks` list of sources — each with
  an optional `name` (renames masks named `<fov>.tiff` to a clean label) or `per_fov: true` (masks
  stored in a per-FOV subfolder `<dir>/<fov>/*.tiff`). `annotations` uses the same shape.
- **Zipped FOVs**: set `"fov_container": "zip"` when each FOV is a `<FOV>.zip` of channel TIFFs —
  UELer reads a single channel straight out of the remote zip via an HTTP byte-range request rather
  than downloading the whole archive.

Pyramidal OME-TIFFs and single zip members are streamed via HTTP byte-range requests; other files
(e.g. single-resolution MIBI TIFFs) are downloaded once into a local cache. A per-study
**workspace** at `~/.ueler/bia/<accession>/` (override with `local_dir=`) holds your persistent
`.UELer` work (ROIs, checkpoints, palettes) plus a disposable `cache/` of downloaded images.

Examples for three real studies — `S-BIAD2557` (single-dir masks), `S-BIAD2864` (two named mask
folders), and `S-BIAD2708` (zipped FOVs + per-FOV masks) — are in `script/run_ueler_BIA.ipynb`.

## User interface
![GUI_preview](/doc/GUI_preview.png)
The GUI can be split into four main regions (wide plugins toggle the optional footer automatically):
- left: overall settings (channel, annotation, and mask accordions)
- middle: main viewer with overlay controls and image navigation
- right: plugin tools (Mask Painter, ROI Manager, palette editors, statistics panels)
- bottom (optional): wide plugin tabs (e.g., horizontal heatmap or gallery extensions)

### Overall Settings
- **Cache Size**: Defines the number of images that can be loaded into memory at one time.  
- **Select Image**: Choose an image to display in the main viewer.  
- **Channel Selection**: Select the channels you want to display. You can select multiple channels by holding down the **Shift** key and clicking.  
- **No image (masks only)**: When masks are available, enable this checkbox to skip rendering the image layer and show masks plus annotations on a black background. This also reduces image-compositing work when you only need spatial mask context.
- **Marker Set**: Load a pre-defined marker set, which includes channels, colors, and color ranges.
- **Control sections**: Channel, annotation, and mask controls now live in a collapsible accordion so you can jump straight to the section you need. When annotations are available, their controls appear ahead of masks, and each pane scrolls independently to keep the palette tools in reach even with dozens of channels.
- **Annotations**: When `<base_folder>/annotations` contains rasters named `<fov>_<annotation>.tif(f)`, enable the overlay toggle to color pixels by class. Choose between mask outlines, annotation fills, or a combined view, adjust fill opacity, and launch the palette editor to customize class colors and display labels. Annotation names can include spaces (for example, `Simple Segmentation`)—they remain selectable and the **Edit palette…** button now activates as soon as such an entry loads.
- **Masks**: Load segmentation rasters, edit per-class colours, and save or recall `.maskcolors.json` sets—default colours are tracked automatically, and optional `ipyfilechooser` dialogs speed up import/export.

### Tools & Plugins
- **Mask Painter**: Focus on edited classes, reuse colour sets, let inactive classes follow the global default fill mode, and restore saved per-class opacity, border, and filtered-list state without leaving the plugin.
- **ROI Manager**: Capture, centre, and tag regions of interest with persistent storage in `<base_folder>/.UELer/roi_manager.csv`; combo-box tagging keeps new labels available for future sessions.
- **Wide Plugins**: Enable "Horizontal layout" (for example, in the heatmap plugin) to undock the tool into the footer while keeping the accordion available for other controls.

## New Update  
### **UELer v0.4.1 Summary**
- Made channel selection consistent across plugins (#113): the **Histogram** and **Scatter plot** plugins now use the same channel picker as the left panel — a tag-style selector with a **Load set** button that loads your saved marker sets (defining new sets still happens in the left panel, and loading a set inside a plugin no longer changes what the main image shows). In the **Histogram** plugin the picker sits above the **Plot** button. In the **Scatter plot** plugin the multi-pair picker is now the main control on top (the single X/Y selector moved into a **Single-pair** tab), and plotting several channels lays the scatters out as a triangular pairwise matrix (each pair once, no diagonal); a single-pair plot added afterwards appears on a new row below.
- Fixed the subset overlay in the **Histogram** plugin (#112 reply): when you brush/select a subset of cells, the "Selected" distribution overlaid on each histogram now uses the **same bins as the full histogram**, so the two are directly comparable. Previously the subset was re-binned over its own (narrower) range, so a narrow selection was squeezed into its own bins and looked misaligned with the full distribution.
- Separated histograms from the scatter-plot plugin (#112): the old **Chart** plugin drew histograms and scatter plots into the same area, so opening one hid the other. There are now two plugins — **Scatter plot** and **Histogram** — that can be open at the same time. The **Scatter plot** plugin adds a **Multi-pair** tab (pick several channels → plot every pairwise scatter at once). The **Histogram** plugin shows several channels at once and has two interaction modes: **Cutoff** (click to set an above/below threshold that highlights cells in the viewer, same as before) and **Brush** (drag a range to select cells — the selection is reflected in the viewer/cell gallery and overlaid as a distribution on every histogram so you can compare across channels).
- Moved the **Lasso Select** toggle to the top of the image viewer (#111): it previously sat in the second row of the left control panel; it now lives at the top of the middle (viewer) panel, so it's clearer that it selects cell masks in the image. Behavior is unchanged.
- Stream/cache-load images from the BioImage Archive (#110): a new `run_viewer_bia("S-BIAD….", descriptor=…)` entry point lets you explore a public BIA study without downloading the whole dataset. Point it at an accession id (resolved via the BioStudies REST API) or a direct base URL; a small JSON descriptor maps the study's files onto FOVs/channels/masks (with best-effort auto-detection of folder-per-FOV and OME-TIFF-per-FOV layouts). Pyramidal OME-TIFFs stream over HTTP byte-range requests; other files are downloaded once into a local cache. Your ROIs, checkpoints, and palettes persist in a per-study workspace under `~/.ueler/bia/<accession>/`. See "Streaming from the BioImage Archive (BIA)" above.

_Earlier changes (v0.4.0 and before) are in the [update log](/doc/log.md)._

## Earlier Updates  

You can find previous update logs [here](/doc/log.md).
