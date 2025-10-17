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
### v0.2.0-alpha
**Heatmap selection safeguards**
- Wrapped `InteractionLayer._apply_cluster_highlights` so scatter highlights respect the chart link toggle, eliminating false redraws during Task 2 heatmap tests.
- Preloaded the chart plugin before test modules import so downstream suites reuse the real `ChartDisplay` implementation instead of minimal stubs.

**Bootstrap dependency coverage**
- Normalized ad-hoc pandas stubs by grafting the shared test DataFrame/Series helpers whenever modules downgrade `pandas` to `object`, restoring ROI and heatmap helpers.
- Expanded the ipywidgets shim to surface `allowed_tags`, `allow_new`, and other TagsInput traits that the ROI manager exercises during tag merge scenarios.

**Test suite reliability**
- Pre-imported key plugins and reran `python -m unittest discover tests`, confirming all 44 tests pass under the shared bootstrap.

## Earlier Updates  

You can find previous update logs [here](/doc/log.md).
