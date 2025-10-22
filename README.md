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
### v0.2.0-rc1
The release candidate combines the new Batch Export UI with the rendering and job runner refactors introduced across the `v0.2.0-alpha` and `v0.2.0-beta` milestones.

**Mask & annotation exports**
- Snapshot the viewer's current mask/annotation state and replay it during batch jobs so exported images honour in-app overlay visibility, colours, and modes.
- Added overlay toggles with availability hints to the Batch Export plugin, letting users opt into masks/annotations per run without surfacing invalid options when datasets lack overlays.
- Extended the renderer with alpha and outline blending plus new tests that cover translucent and outline masks alongside overlay snapshot reconstruction.
- Upgraded mask outlines to respect per-cell label maps with an adjustable thickness slider; the renderer now falls back to a pure NumPy boundary finder when `skimage` is absent, and tests cover both baseline and dilated contours.

**Mask outline controls & plugin independence**
- Added a main-viewer mask-outline slider that updates the live render and notifies plugins without forcing feedback loops.
- Seeded the Batch Export plugin's slider from the viewer while preserving a local override that flows through overlay snapshots, cache keys, and export workers.
- Hardened synchronisation so viewer-driven changes are adopted only when the plugin has not diverged, keeping exports and on-screen previews aligned without clobbering local adjustments.
- Expanded `tests/test_rendering.py` and `tests/test_export_fovs_batch.py` with label-preserving outline assertions and slider sync regressions.

**Batch export UI & UX**
- Replaced the placeholder plugin with `BatchExportPlugin`, offering mode selection (Full FOV, Single Cells, ROIs), marker profiles, output configuration, and asynchronous Start/Cancel controls with progress feedback.
- Added per-mode panels with cell filtering, ROI selectors, crop sizing, and a single-cell preview workflow that uses the shared rendering helpers before launching full jobs.
- Surfaced scale-bar toggles and PDF/bitmap format handling; scale-bar sizing hooks are in place ahead of the Phase 4 implementation.

**Rendering & job orchestration**
- Reuse the pure compositing helpers (`render_fov_to_array`, `render_crop_to_array`, `render_roi_to_array`) for all export modes, ensuring consistent colour/overlay output without UI state coupling.
- Drive exports through `ueler.export.job.Job`, capturing structured per-item results, cancellation, and observable progress that feeds the new UI components.

**Developer experience**
- Expanded the lightweight matplotlib bootstrap to stub text, backend, patches, widgets, and `mpl_toolkits.axes_grid1` helpers so export suites run without heavy graphics dependencies.
- Kept the namespace migration, compatibility shims, packaging metadata, and Makefile utilities introduced earlier in the `v0.2.0` cycle.

**Verification**
- `python -m unittest tests.test_rendering tests.test_export_fovs_batch`
- `python -m unittest tests.test_export_job`

## Earlier Updates  

You can find previous update logs [here](/doc/log.md).
