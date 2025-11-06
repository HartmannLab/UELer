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
### v0.2.0-rc3
**Cell gallery mask painter synchronization**
- Cell gallery now synchronizes with the mask painter for adaptive thickness and painted color display (fixes [#54](https://github.com/HartmannLab/UELer/issues/54)).
- Added "Use uniform color" checkbox to the cell gallery: when disabled (default), painted mask colors from the mask painter are displayed in gallery thumbnails; when enabled, all masks use the uniform color from the "Mask colour" picker.
- The cell gallery's outline thickness slider automatically syncs with the main viewer's mask outline thickness setting, ensuring consistent rendering across all viewing contexts.
- Implemented persistent color mapping in the mask painter that stores which color was applied to each cell, allowing the cell gallery to retrieve and display exact painted colors.
- The gallery auto-refreshes when the mask painter applies colors or when the main viewer's thickness slider changes, keeping all views synchronized without manual intervention.

**ROI gallery width stabilization**
- Switched ROI gallery to static narrow figure sizing (4.8 inches at 72 DPI ≈ 346px) to eliminate thumbnail clipping at narrow widths (addresses [#39](https://github.com/HartmannLab/UELer/issues/39)).
- Removed ResizeObserver-based responsive width code after investigation revealed that JavaScript can only resize DOM wrapper elements, not Matplotlib's pre-rendered raster content—when the container shrinks below the original render width, the fixed-size raster overflows and clips.
- Gallery now renders conservatively narrow and relies on CSS `width: 100%` to stretch when space is available, trading slight blur at wider widths for guaranteed no-clip behavior in narrow panels.
- Fixed vertical clipping by removing redundant VBox scroll container—`browser_output` now provides the only scrolling container, allowing full canvas height to be visible when scrolling.

**Cache configuration**
- Moved the Cache Size control into Advanced Settings and seeded new viewers with a default of 100 so fresh sessions match notebook expectations while keeping the top panel focused (fixes [#53](https://github.com/HartmannLab/UELer/issues/53)).
- Added `tests/test_cache_settings.py` to assert the widget placement and default cache size, preventing regressions as UI layouts evolve.

**Cell tooltip precision**
- Main viewer tooltips now switch to scientific notation when fixed-point rounding would otherwise render tiny non-zero marker intensities as 0.00, keeping near-zero signals visible (fixes [#51](https://github.com/HartmannLab/UELer/issues/51)).
- Light regression coverage in `tests/test_image_display_tooltip.py` locks in the formatter for regular, tiny, and negative values.

**ROI browser refresh throttling**
- The ROI Manager caches its browser signature and only rebuilds thumbnails when ROI data or presets change, so routine FOV navigation keeps pagination and cached previews intact (fixes [#50](https://github.com/HartmannLab/UELer/issues/50)).
- Thumbnail rendering now honours saved mask painter presets by querying the active colour set before assembling previews, keeping per-ROI styling consistent between the browser and viewer.

**Cell gallery FOV debounce**
- Cell gallery clicks mark the next viewer-driven FOV change as internal, skipping the immediate `on_fov_change` redraw that previously regenerated the gallery unnecessarily (addresses [#52](https://github.com/HartmannLab/UELer/issues/52)).
- Scatter-to-gallery workflows still refresh correctly because external plugin broadcasts clear the guard after their single-point navigation completes.

**Chart histogram tuning**
- The histogram bin slider now updates plots immediately with continuous slider feedback, keeping cutoff markers visible after each redraw so tweaking bins delivers instant insight (addresses [#47](https://github.com/HartmannLab/UELer/issues/47)).
- Cutoff-based highlights persist as you switch FOVs because the chart plugin reapplies its selection after the viewer clears overlays, ensuring qualifying cells stay emphasized during navigation.

**Scatter gallery guard**
- Chart scatter clicks and trace commands now set a transient `single_point_click_state`, suppress single-point forwarding to the gallery, and let the gallery's `on_fov_change` clear the flag so single-cell jumps stop collapsing the gallery while multi-cell selections keep syncing (fixes [#48](https://github.com/HartmannLab/UELer/issues/48)).

**Mask outline scaling**
- Shared downsample-aware outline helpers so viewer overlays, ROI highlights, and mask painter recolouring all apply `max(1, t/f)` outline thickness (fixes [#46](https://github.com/HartmannLab/UELer/issues/46)) while keeping outlines visible at native zoom.
- Regression coverage in `tests/test_rendering.py` locks in the new scaling behaviour alongside the legacy outline dilation path.

The first release candidate delivers automatic scale bars across the viewer and batch export workflows while retaining the Batch Export UI, overlay plumbing, and job runner improvements from earlier `v0.2.0` milestones.

**ROI browser pagination**
- Replaced the lazy scroll loader with Previous/Next buttons, a page indicator, and deterministic 3x4 slices so gallery navigation stays predictable even as filters or expressions change (addresses [#44](https://github.com/HartmannLab/UELer/issues/44)).
- Added a cursor-aware expression helper that tracks the selection/focus inside the filter field, letting operator/tag buttons splice tokens at the caret, keep the widget focused, and preserve the caret location even after helper buttons momentarily steal focus.
- Hardened the JavaScript bridge that reports caret updates so it finds the Text widget across modern JupyterLab builds as well as classic Notebook/Voila layouts, fixing stacks where the previous selectors missed the input.
- When expressions reload from notebook state while the field is unfocused, the caret now defaults to the tail so the next helper insertion appends instead of jumping to the front.
- Refined the helper insertion routine to honour the cached start/end indices (including highlighted ranges), so repeated button presses keep chaining at the cursor instead of resetting to index 0.
- Restored the selection resolver and focus-aware caching after a regression so helper buttons keep updating the field even when caret telemetry drops blur events.
- Shifted snippet insertion to happen entirely in the browser: helper buttons now send an `insert-snippet` event that updates the field client-side before syncing back to Python, eliminating focus-race edge cases.
- Added a readiness check that temporarily falls back to the Python insertion path until the browser bridge reports a live caret snapshot, keeping helper buttons responsive immediately after load.
- Replaced the inline `<script>` injection with `IPython.display.Javascript` so the caret bridge runs under JupyterLab’s sanitized output policy and keeps reporting selection updates reliably.
- Simplified the caret bridge to mirror the standalone ipywidgets DOM-binding demo: it now resolves the widget input via stable selectors, runs the helper JavaScript through a shared `ipywidgets.Output` in the same frame, performs the splice entirely in the browser, and then syncs the new value/caret back to Python across Notebook, Voila, and JupyterLab 4 hosts.
- Constrained the ROI browser output panel to a 400px viewport with internal scrolling so the sidebar stays compact while the tile gallery grows (reply 7 to [#44](https://github.com/HartmannLab/UELer/issues/44)).
- The Matplotlib gallery now fixes its three-column grid, pads empty slots, and clamps figure width to 98% of the plugin pane so thumbnails stay fully visible without horizontal clipping.
- Follow-up: forcing the output container (and parent flex boxes) to `min_width=0` with `flex=1 1 auto` plus a scoped CSS rule keeps rendered figures within the box, letting the scrollbar apply solely to thumbnails and keeping pagination buttons unobscured (reply 8 to [#44](https://github.com/HartmannLab/UELer/issues/44)).
- Latest tweak: the gallery now wraps the Matplotlib canvas in a fixed-height `VBox`, applies explicit pixel sizing to the canvas layout, and displays the widget directly so the 400px viewport and vertical scrollbar take effect; environments without the widget backend still fall back to the traditional `plt.show` path.

**ROI filter tabset**
- ROI manager browser filters now live behind `simple` and `advanced` tabs, separating the AND/OR widgets from expression entry for clearer navigation (addresses [#49](https://github.com/HartmannLab/UELer/issues/49)).
- Only the active tab's inputs drive ROI filtering and the gallery refreshes automatically when you switch tabs, keeping pagination and status messaging in sync with the selected mode.

**Cell gallery tile padding**
- Gallery slots now expand to the widest rendered tile and center narrower crops so selecting multiple cells never triggers NumPy broadcasting errors when their cutouts differ slightly in width.
- Added a `tests/test_cell_gallery.py` regression to lock in the padding behaviour and keep the gallery, batch export, and main viewer selections in sync.

**Gallery rendering unification**
- Rebuilt the cell gallery plugin on top of the shared `ueler.rendering` engine, funnelling overlay snapshots, mask outline tinting, and downsampling controls through the same pipeline used by the main viewer and batch export so per-cell previews stay visually consistent.
- Restored legacy rendering helpers (`find_boundaries`, `_label_boundaries`, `_binary_dilation_4`) via `ueler.viewer.rendering` to keep historical imports and renderer tests green while the new engine powers gallery tiles.

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

**Plugin layout refinements**
- Introduced shared layout helpers for ipywidgets containers and applied them to the ROI Manager, Batch Export, and Go To plugins so control rows wrap cleanly without producing horizontal scrollbars.
- Tightened button groups and selectors to flex within their parents, eliminating the ~5 % overflow scroll bars called out in issue #39.

**Pixel annotation clarity**
- Removed the Overlay mode toggle from the Pixel annotations pane so mask visibility is governed solely by the mask section; mask checkboxes now always apply even when pixel annotations are hidden (addresses [#41](https://github.com/HartmannLab/UELer/issues/41)).
- Renamed the accordion tab to `Pixel annotations` and reordered the controls to `Channels`, `Masks`, `Pixel annotations`, keeping mask toggles front-and-centre while separating pixel overlays from mask visibility.

**ROI browser presets**
- The ROI Manager now opens with `ROI browser` and `ROI editor` tabs: browse ROIs via a Matplotlib gallery with tag/FOV filters, click to centre in the main viewer, or centre with preset to apply the ROI's stored rendering state.
- ROI captures now persist the active annotation palette and mask colour set alongside marker presets; the viewer exposes helpers so plugins can record the active palette and replay saved presets when restoring ROIs.
- The browser adds a boolean expression builder for tag filtering (`() & | !` with quoted tags), eager validation with inline error hints, HUD-free Matplotlib tiles sized to 98 % of the scroll container, and a repaired lazy-loading script that fetches four more previews whenever you scroll near the end.
- Tests cover both the parser (`tests/test_tag_expression.py`) and plugin wiring (`tests/test_roi_manager_tags.py`) so future tweaks keep validation and preset restoration intact.

**Pixel annotation palette management**
- The Pixel annotations accordion now mirrors the mask painter workflow with Save/Load/Manage tabs, optional `ipyfilechooser` dialogs, and shared registry handling so class palettes can be saved and restored consistently (addresses [#42](https://github.com/HartmannLab/UELer/issues/42)).
- Introduced `ueler.viewer.palette_store` for common palette persistence helpers and aligned the mask painter plugin plus new unit tests (`tests/test_annotation_palettes.py`) on the shared format.

**Documentation updates**
- Documented the viewer's automatic FOV downsampling flow in `dev_note/main_viewer.md`, covering factor selection, caching, and scale bar adjustments for large scenes.

**ROI browser thumbnails**
- Unified thumbnail downsampling with the main viewer helper so ROI previews auto-scale the longest edge to 256 px regardless of saved zoom metadata, improving navigation performance on oversized FOVs.
- Follow-up: thumbnails now measure each ROI viewport, apply `factor = 2^ceil(log2(ceil(longest/256)))`, and rely on ceiling division when deriving downsampled bounds so non-divisible widths/heights still render complete tiles.

**ROI preview reliability**
- Normalised ROI metadata parsing so cropped entries with string or NaN viewport bounds still render, falling back to safe defaults when coordinates are missing and eliminating the “preview unavailable” tiles called out in Reply 6 to [#44](https://github.com/HartmannLab/UELer/issues/44).
- Aligned downsampled region math with the sampled pixel grid and added regression coverage for string-based ROI coordinates, keeping thumbnails consistent across full-FOV and cropped selections.

**Channels accordion consolidation**
- Moved the channel tag chips plus marker-set dropdown, name entry, and action buttons into the Channels accordion pane so channel selection, presets, and per-channel sliders stay in one cohesive space.
- Rebuilt the pane with nested containers that maintain spacing and focus order, removing duplicate widgets from the header while keeping keyboard navigation unchanged.
- Let the accordion body stay static while the slider list gains its own scroll region, preventing double scrollbars and keeping long channel lists usable.

**Verification**
- `python -m unittest tests.test_rendering tests.test_export_fovs_batch`
- `python -m unittest tests.test_scale_bar_helper tests.test_export_fovs_batch`
- `python -m unittest tests.test_export_fovs_batch`
- `python -m unittest tests.test_fov_detection`
- `python -m unittest tests.test_roi_manager_tags`

## Earlier Updates  

You can find previous update logs [here](/doc/log.md).
