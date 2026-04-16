# UELer
Usability Enhanced Linked Viewer: a Jupyter Notebook integrated viewer for MIBI images with linked interactive plots and enhanced usability.

## Try it on Binder
You can try UELer without installation by launching it on [Binder](https://mybinder.org/v2/gh/HartmannLab/UELer/binder-app?urlpath=%2Fdoc%2Ftree%2Fscript%2Frun_ueler.ipynb):
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HartmannLab/UELer/binder-app?urlpath=%2Fdoc%2Ftree%2Fscript%2Frun_ueler.ipynb)

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
- **Marker Set**: Load a pre-defined marker set, which includes channels, colors, and color ranges.
- **Control sections**: Channel, annotation, and mask controls now live in a collapsible accordion so you can jump straight to the section you need. When annotations are available, their controls appear ahead of masks, and each pane scrolls independently to keep the palette tools in reach even with dozens of channels.
- **Annotations**: When `<base_folder>/annotations` contains rasters named `<fov>_<annotation>.tif(f)`, enable the overlay toggle to color pixels by class. Choose between mask outlines, annotation fills, or a combined view, adjust fill opacity, and launch the palette editor to customize class colors and display labels. Annotation names can include spaces (for example, `Simple Segmentation`)—they remain selectable and the **Edit palette…** button now activates as soon as such an entry loads.
- **Masks**: Load segmentation rasters, edit per-class colours, and save or recall `.maskcolors.json` sets—default colours are tracked automatically, and optional `ipyfilechooser` dialogs speed up import/export.

### Tools & Plugins
- **Mask Painter**: Focus on edited classes, reuse colour sets, and jump between identifiers without leaving the plugin.
- **ROI Manager**: Capture, centre, and tag regions of interest with persistent storage in `<base_folder>/.UELer/roi_manager.csv`; combo-box tagging keeps new labels available for future sessions.
- **Wide Plugins**: Enable "Horizontal layout" (for example, in the heatmap plugin) to undock the tool into the footer while keeping the accordion available for other controls.

## New Update  
### **UELer v0.3.1 Summary**
- Published a [documentation site](https://hartmannlab.github.io/UELer/) built with Material for MkDocs, covering installation, getting started, tutorials, FAQ, and developer notes. The site is auto-deployed to GitHub Pages on every push to `main`.
- Restored `ueler.image_utils` as a real packaged module and corrected the legacy utility shims so packaging cleanup no longer breaks imports through `ueler.image_utils` or the old root-level utility module names.
- Added per-channel grid display mode: a new "Channel grid view" checkbox in the Channels panel renders each visible channel as a separate labelled pane in a synchronised grid layout.
- Linked the histogram cutoff in the Chart plugin to the cell gallery: enabling the "Cell gallery" checkbox in "Linked plugins" now sends all cells above/below the cutoff (across all FOVs) to the gallery; toggling above/below updates the gallery immediately.
- Enabled batch export in simple viewer mode (images-only): the Batch Export plugin now loads when no cell table is present; the Single Cells tab is replaced with an informational notice and restored automatically once a cell table is added.
- Added full map mode support to ROI Manager and Batch Export plugins: FOV lookups now use `get_active_fov()` to handle the disabled/stale `image_selector` in map mode; plugins automatically disable FOV-scope filters when map mode activates and restore them on deactivation; `center_on_roi()` translates ROI pixel coordinates to stitched-canvas space in map mode. ROI thumbnails in map mode are rendered from the stitched map layer; the gallery navigates to page 1 after each capture so new ROIs are immediately visible; ROI labels show `[MAP:<id>]` for map-mode entries.
- Fixed batch export of map-mode ROIs: previously `_build_roi_items()` silently skipped all map-mode ROIs (`fov=""`) and the raw canvas-pixel coordinates were misinterpreted as single-FOV coordinates. A new `_export_map_roi_worker()` renders map-mode ROIs via the stitched `VirtualMapLayer` (same approach used for thumbnails), converting canvas pixels to physical µm before calling `set_viewport` / `render`. `_update_selected_roi()` now also preserves `map_id` so ROI labels remain correct after an update in map mode.
- Consolidated dev notes into topic-oriented summaries under `dev_note/`.
- Added a dev note index mapping source notes to the new topic summaries for quicker navigation.
- Removed `dev_note/issue_tracking/` after distilling its contents into the topic summaries and related issue links.

## Earlier Updates  

You can find previous update logs [here](/doc/log.md).
