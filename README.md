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
**Fast-test dependency isolation**
- Forced the shared bootstrap to install in-process seaborn/scipy stubs whenever pandas is stubbed so heatmap imports no longer reach for the real libraries, and wired the annotation palette suite to load the bootstrap before importing viewer modules to guarantee the lightweight shims take effect.
- Added explicit stub markers for the installed modules so subsequent imports keep reusing the lightweight implementations during `unittest` discovery.

**Chart widget layout compatibility**
- Updated the chart plugin to build every `VBox`/`HBox` with the `children=` keyword, ensuring older ipywidgets stubs capture the controls and plot panes so footer layout assertions reflect production behavior.
- Retained the legacy `color_points` signature while documenting the unused `selected_colors` parameter to satisfy lint without altering callers.

**Compatibility import shims**
- Registered lazy module aliases bridging the new `ueler.*` namespace to the legacy `viewer` modules via `_compat.register_module_aliases`, letting notebooks adopt the packaged layout without eager imports.
- Hardened the alias finder to fall back to stubbed modules lacking `ModuleSpec` metadata so fast-stub tests keep loading lightweight plugin placeholders without errors.
- Expanded `tests/test_shims_imports.py` to assert the alias matrix resolves to the legacy modules and to skip gracefully when optional dependencies such as `cv2` are unavailable.

**Incremental module moves**
- Migrated `viewer/ui_components.py` into `ueler.viewer.ui_components` while retaining the legacy import path through a lightweight compatibility wrapper and alias bridge.
- Updated shim tests to tolerate downstream stubs and confirmed the fast suite stays green after the relocation.
- Relocated `viewer/color_palettes.py` into `ueler.viewer.color_palettes`, added a legacy wrapper plus reverse alias in `_compat.py`, and reran the fast test suite to verify shim coverage remains intact.
- Moved `viewer/decorators.py` into `ueler.viewer.decorators`, introduced helpers to simplify the status-bar decorator, and left a compatibility wrapper for legacy imports.
- Transitioned `viewer/observable.py` to `ueler.viewer.observable`, tightened typing for the observable helper, and replaced the legacy module with a thin forwarding shim.
- Relocated `viewer/annotation_palette_editor.py` into `ueler.viewer.annotation_palette_editor`, updated its color helper imports, and retained a legacy shim for backward compatibility.
- Shifted `viewer/annotation_display.py` into `ueler.viewer.annotation_display`, refreshed imports to use the packaged namespace, added a lazy widget loader so test stubs initialize before instantiation, and provided a compatibility wrapper plus alias updates so existing code keeps working.
- Repositioned `viewer/roi_manager.py` into `ueler.viewer.roi_manager`, switched timestamps to `datetime.now(timezone.utc)` for lint compliance, and left a compatibility wrapper plus reverse alias so legacy imports remain operational.
- Ported `viewer/plugin/plugin_base.py` into `ueler.viewer.plugin.plugin_base`, introduced a packaged plugin scaffold, and replaced the legacy module with a forwarding shim plus reverse alias coverage.
- Moved `viewer/plugin/export_fovs.py` into `ueler.viewer.plugin.export_fovs`, preserved the placeholder UI, and wrapped the legacy module with a forwarding shim plus alias updates.
- Shifted `viewer/plugin/go_to.py` into `ueler.viewer.plugin.go_to`, refreshed imports to use the packaged helpers, and added a legacy wrapper plus alias updates for continuity.
- Migrated `viewer/plugin/cell_gallery.py` into `ueler.viewer.plugin.cell_gallery`, tightened deterministic sampling when downscaling selections, and replaced the legacy module with a forwarding shim plus alias updates to preserve backwards compatibility.
- Relocated `viewer/plugin/chart.py` into `ueler.viewer.plugin.chart`, re-pointed intra-project imports to the packaged namespace, and slimmed the legacy module to a forwarding shim while pruning redundant alias table entries.
- Migrated `viewer/plugin/run_flowsom.py` into `ueler.viewer.plugin.run_flowsom`, added a graceful fallback when `pyFlowSOM` is missing, deduplicated repeated layout strings, and left the legacy module as a thin wrapper.
- Ported `viewer/main_viewer.py` into `ueler.viewer.main_viewer`, updated its internal imports to favor the packaged namespace, pointed dynamic plugin loading at `ueler.viewer.plugin.*`, and replaced the legacy file with a compatibility wrapper.

**Fast-stub pandas parity**
- Extended the shared pandas shim with `Series.loc`/`.iloc` indexers, `map`, `astype`, boolean helpers, and dictionary-aware constructors so chart and scatter tests align categories correctly without the real library installed.
- Patched fallback pandas modules discovered during test imports to graft the shared `api.types` helpers, keeping `is_numeric_dtype` and `is_object_dtype` available even when ad-hoc stubs surface.
- Hardened reindex and assignment support (`Series.reindex`, `Series.loc[...] = value`) to preserve ROI, heatmap, and scatter workflows in the fast-test environment.

**Matplotlib stub coverage**
- Registered a lightweight `matplotlib.pyplot` stub with canvas and axis helpers so histogram code paths execute during fast tests without pulling in the full plotting stack.
- Updated the plugin preloader to replace minimalist viewer stubs with the real chart/ROI modules before tests run, ensuring footer layout assertions exercise production logic.

**Heatmap footer redraw preference**
- Taught `HeatmapDisplay.restore_footer_canvas` to attempt cached redraws before scheduling a canvas repaint, satisfying the footer regression tests and avoiding redundant `draw_idle` calls when the cache already holds a canvas snapshot.

**Packaging skeleton groundwork**
- Added `pyproject.toml` with minimal project metadata, setuptools configuration, and developer extras to unblock incremental packaging work.
- Created a lightweight `Makefile` offering virtualenv creation, editable installs, and fast/integration test targets to align local workflows with the mitigation strategy.
- Introduced `ueler.__init__` and `ueler.viewer.__init__` compatibility shims that lazily forward to the legacy `viewer` module so consumers can begin migrating import paths without runtime changes.

**Root helper packaging**
- Listed `constants.py`, `data_loader.py`, and `image_utils.py` under `tool.setuptools.py-modules` so wheel builds include the legacy helpers relied upon by the compatibility shims.
- Flagged follow-up release validation to build wheels/sdists and confirm the helpers remain available after installation.

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
