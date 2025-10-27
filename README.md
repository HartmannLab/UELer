# UELer
Usability Enhanced Linked Viewer: a Jupyter Notebook integrated viewer for MIBI images with linked interactive plots and enhanced usability.

## Try it on Binder
You can try UELer without installation by launching it on [Binder](https://mybinder.org/v2/gh/HartmannLab/UELer/binder-app?urlpath=%2Fdoc%2Ftree%2Fscript%2Frun_ueler.ipynb):
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HartmannLab/UELer/binder-app?urlpath=%2Fdoc%2Ftree%2Fscript%2Frun_ueler.ipynb)

## Installation

### 1. Set up the environment

You can create a compatible environment using the `environment.yml` file provided in this repository.

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
  - **`masks_folder`**: The directory containing the segmentation `.tif` files for cell segmentation (e.g., `.../segmentation/cellpose_output`).
  - **`cell_table_path`**: The path to the file containing the cell table data (e.g., `.../segmentation/cell_table/cell_table_size_normalized.csv`).
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
### v0.2.0-rc2
The first release candidate delivers automatic scale bars across the viewer and batch export workflows while retaining the Batch Export UI, overlay plumbing, and job runner improvements from earlier `v0.2.0` milestones.

**Phase 4b cell export fixes**
- Marker profiles now fall back to the viewer's active channel selection when stored marker sets lack entries, preventing blank single-cell exports and enforcing per-channel render settings before jobs run.
- The single-cell preview reuses captured overlays, passes keyword arguments into `_finalise_array`, and draws optional scale bars so the UI preview mirrors batch outputs without triggering the earlier signature error.
- Extended `tests/test_export_fovs_batch.py` with regression cases covering the fallback logic and preview workflow.

**Scale bar automation**
- Added `ueler.viewer.scale_bar` with an engineering-style rounding helper that picks tidy physical lengths (≤10 % of the frame) and formats labels in µm/mm as needed.
- Refreshed the main viewer so the scale bar updates whenever pixel size, downsample, or view extents change, ensuring on-screen measurements stay accurate without manual tweaks.
- Batch exports now draw the same scale bar into PNG/JPEG/TIFF outputs and embed it ahead of PDF saves, capturing the computed length in per-item metadata.
- Phase 4a hotfix: the interactive viewer now scales the rendered bar using the active downsample factor, preventing undersized annotations when zoomed out.

**Batch export experience**
- The plugin continues to provide mode-aware exports (Full FOV, Single Cells, ROIs), overlay snapshots, and cancellation-ready jobs, now seeded with the viewer's pixel size to keep in-app and exported measurements in sync.
- Raster/PDF writers share a Matplotlib-based overlay helper, guaranteeing consistent styling and placement across formats while preserving previous mask/annotation options.

**Linked plugin reliability**
- Scatter chart plugins now re-run observer setup after dynamic plugin loading, ensuring the cell gallery reflects scatter selections as soon as the link toggle is enabled and avoiding duplicate callbacks when plugins refresh.
- Added regression coverage in `tests/test_chart_footer_behavior.py` for both linked and unlinked flows, alongside lightweight imaging stubs that keep the fast suite dependency-light.

**Rendering & tests**
- Extended `_finalise_array` to return scale bar specifications, introduced `_render_with_scale_bar`/`_write_pdf_with_scale_bar`, and added fallbacks for environments lacking full Matplotlib bindings.
- Added `tests/test_scale_bar_helper.py` to lock in rounding behaviour and effective pixel sizing, alongside updates to the existing batch export suite (with fresh ipywidgets/matplotlib stubs) to cover the new pipeline.
- Extra regression coverage ensures non-unity downsample factors stay accurate in both helper calculations and viewer rendering.

**FOV detection filtering**
- Enhanced FOV detection to only recognize directories containing .tif or .tiff files as valid FOVs, preventing misclassification of folders like '.ueler' that lack image data.
- Added comprehensive unit tests to validate the filtering logic and ensure no regressions in existing functionality.

**Verification**
- `python -m unittest tests.test_scale_bar_helper tests.test_export_fovs_batch`
- `python -m unittest tests.test_export_fovs_batch`
- `python -m unittest tests.test_fov_detection`

## Earlier Updates  

You can find previous update logs [here](/doc/log.md).
