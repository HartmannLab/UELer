# UELer
Unified Exploratory Linked Viewer: a Jupyter-based framework for interactive exploration of multiplexed imaging datasets.

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
### **UELer v0.4.4 Summary**
- **Fill and border are now separate switches in the Mask Painter, and the two coloring modes match (#132).** The discrete and continuous modes used to disagree: `Identifier` sat at the top as if it applied to both, though only discrete coloring uses it; discrete had `Global fill` plus `Borders on filled`, while continuous had a single `Fill (unchecked = outline)` box and no border control at all. Now each mode has a **Fill** checkbox with its own opacity — `Global fill` is renamed **Fill all classes** — and both share one **Border** checkbox with its own opacity and color. All four combinations do what they say: fill only, border only (the old "unchecked = outline"), both, or neither. The `Identifier` dropdown moved into the discrete section. The border color can be the **Painted color** (default — the class color, or the colormap color in continuous mode), the **Mask color**, or a **Custom…** color you pick. Every opacity field is now the same width in both modes. ⚠️ Palettes and ROIs saved before this update stored the old **Mask color** border setting; since the border now also colors the outline of *unfilled* cells, those will draw mask-colored outlines until you switch **Border color** to **Painted color**.
- **Outline mode in the Mask Painter is no longer slow (#131).** With **Fill (unchecked = outline)** switched off, redrawing took a moment per cell rather than a moment per image, so the wait grew with the number of painted cells — around 5 s for 200 cells and nearly two minutes for 5000, while the same view in fill mode took a fraction of a second. Cell outlines were being traced one cell at a time; they are now traced for the whole image in a single pass. Rendering no longer depends on how many cells are painted: the 5000-cell case went from ~117 s to ~65 ms, and outline mode is now the faster of the two. What you see is unchanged, with one exception at **outline thickness 2 or more** — where a thick outline overlaps a filled cell's border, the filled cell's border is now drawn on top instead of whichever cell had the larger id.
- **The ROI manager's advanced filter understands tags with spaces (#130).** Tags such as `tumour core` could be created but not filtered on: typing one into the advanced expression field — or clicking its helper button — gave `Expression did not reduce to a single value.`, because the space was read as a separator between two tags. A space inside a name is now part of the name, so `tumour core & !necrotic edge` and `good roi & (figure 1 | figure 2)` work as typed. Spaces around a name are ignored, and quotes are still available for tags containing `&`, `|`, `!` or brackets.
- **Switching the histogram between Cutoff and Brush no longer resets the plots (#127 reply).** Flipping the **Interaction** toggle rebuilt every histogram: the gate you had built up was remembered, but the plots flashed and any zoom or pan you had set was thrown away. The toggle now only changes what a click-drag does — the bars, the gate bands and cutoff lines, the "Selected" overlay and your zoom all stay put. A stray click while brushing no longer sets a cutoff; conversely, if you pick the box-select tool from a plot's toolbar while in Cutoff mode, the range you drag is used.
- **The histogram's "Main viewer" link is respected in both interaction modes (#129).** With the **Main viewer** box in the histogram's *Linked plugins* tab unchecked, setting a **cutoff** still outlined the matching cells in the main viewer (brushing a range correctly did not), and it re-appeared after every FOV switch. The checkbox now decides *every* highlight the histogram pushes. Unchecking it also **removes** the outlines it had drawn, instead of just freezing them on screen, and re-checking puts them back for the FOV you are on. The cell-gallery link and the cell-gallery/scatter/heatmap plugins are unaffected.
- **Cell gallery images are all at the same scale now (#128).** Cells close to the border of a field of view could not fill their square cutout, and the missing part was made up by scaling the tile: the cell was drawn up to ~1.3× larger than a cell from the middle of the image, and the tile came out taller than a square. Comparing cell sizes across the gallery was therefore misleading. The part of a cutout that falls outside the image is now filled with **white padding** instead, so every tile is the same size and every cell is drawn at the same scale, wherever it sits in the FOV. Nothing else about the tiles changes — the cell stays in the position it occupies in the image rather than being re-centred, and mask outlines and annotations are unaffected. (Per-cell **batch export** still writes the smaller crop for edge cells; that path is unchanged for now.)

_Earlier changes (v0.4.3 and before) are in the [update log](/doc/log.md)._

## Earlier Updates  

You can find previous update logs [here](/doc/log.md).
