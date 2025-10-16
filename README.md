# UELer
Usability Enhanced Linked Viewer: a Jupyter Notebook integrated viewer for MIBI images with linked interactive plots and enhanced usability.

## Installation
### Set up the environment (env)
You can create a compatible env using the environment.yml file in this repository.
Download the yml file to a preferred folder, change the current directory to it, and then run:
```shell
micromamba env create --name ark-analysis-ueler --file environment.yml
```
### Install UELer
Go to the directory where you want to install the tool, and then git clone this repository:
```
git clone https://github.com/HartmannLab/UELer.git
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
2. Navigate to the cloned UELer repository, then open the notebook `/script/run_viewer.ipynb`.
3. Select the kernel for an ark-analysis compatible conda/micromamba env.
4. Change the lines according to the instructions in the notebook: when configuring the `/script/run_viewer.ipynb`, ensure that you specify the following directory paths:
  - **`viewer_dir`**: The directory path where the UELer folder is located (e.g., `.../UELer`).
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
### v0.1.10-rc3
**Assign tab activation**
- Wired `HeatmapDisplay` to observe `current_clusters["index"]` so the Assign tab's `Meta-cluster ID` field and Apply button enable as soon as a cluster selection exists and disable when it clears ([issue #9](https://github.com/HartmannLab/UELer/issues/9)).
- Initialized the observer during widget restore to keep the controls accurate after notebook reloads, preventing stale disabled states when selections persist across sessions ([issue #9](https://github.com/HartmannLab/UELer/issues/9)).

**Automatic cutoff locking**
- Re-engaged the Lock Cutoff control automatically after meta-cluster assignments and dendrogram edits so patches immediately regain protection from accidental reclassification ([issue #10](https://github.com/HartmannLab/UELer/issues/10)).
- Added a guarded “Unlock once” workflow that forces users to request a one-time override before editing the cutoff, restoring the lock after changes and keeping manual toggles from bypassing the safeguard ([issue #10](https://github.com/HartmannLab/UELer/issues/10)).

**Chart footer placement**
- Updated `ChartDisplay` so the plugin automatically refreshes the footer when multiple scatter plots are active, keeping controls in the bottom tab while leaving a clear notice in the sidebar ([issue #5](https://github.com/HartmannLab/UELer/issues/5)).
- Added `tests/test_chart_footer_behavior.py` with lightweight widget and data stubs to guard the layout toggle and footer refresh workflow ([issue #5](https://github.com/HartmannLab/UELer/issues/5)).

**Heatmap footer stability**
- Ensured the heatmap plugin reattaches its canvas during footer rebuilds so horizontal heatmaps persist after scatter plots are added or removed in the chart plugin ([issue #6](https://github.com/HartmannLab/UELer/issues/6)).
- Expanded `tests/test_chart_footer_behavior.py` to simulate combined chart and heatmap footer refreshes, guaranteeing the heatmap remains registered in `BottomPlots` following scatter changes ([issue #6](https://github.com/HartmannLab/UELer/issues/6)).

**Footer pane caching**
- Reworked `viewer/ui_components.update_wide_plugin_panel` to cache wide plugin panes, preserving live matplotlib canvases when the chart toggles between sidebar and footer layouts ([issue #6](https://github.com/HartmannLab/UELer/issues/6)).
- Extended `tests/test_wide_plugin_panel.py` with cache-aware stubs so chart relocations reuse the existing heatmap pane instead of instantiating a fresh widget tree ([issue #6](https://github.com/HartmannLab/UELer/issues/6)).

**Heatmap redraw automation**
- Triggered the heatmap footer pane to replay its Plot routine whenever a cached wide layout is reused so horizontal canvases repopulate immediately after chart-driven footer rebuilds ([issue #6](https://github.com/HartmannLab/UELer/issues/6)).
- Added a regression to `tests/test_wide_plugin_panel.py` asserting cached panes invoke the redraw helper every time the footer recomposes, preventing future regressions ([issue #6](https://github.com/HartmannLab/UELer/issues/6)).

**Wide heatmap tick alignment**
- Centered tick locations and histogram bins in horizontal heatmaps so column and row labels sit on the underlying grid instead of drifting half a cell off to the left/top ([issue #8](https://github.com/HartmannLab/UELer/issues/8)).
- Added regression coverage around the tick-label helper to guard against future misalignment regressions in wide mode ([issue #8](https://github.com/HartmannLab/UELer/issues/8)).

**Cluster ID persistence**
- Cached `meta_cluster_revised` overrides before rebuilding heatmaps and restored them afterward so manual assignments survive horizontal/vertical layout toggles ([issue #11](https://github.com/HartmannLab/UELer/issues/11)).
- Added focused regression tests that exercise the caching helper to prevent future regressions in manual assignment retention ([issue #11](https://github.com/HartmannLab/UELer/issues/11)).

## Earlier Updates  

You can find previous update logs [here](/doc/log.md).
