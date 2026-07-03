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
### **UELer v0.3.1 Summary**
- Heatmap remembers its scale (figure size) after updating the tree cut (#109): previously, dragging the dendrogram cutoff rebuilt the heatmap at the default size, discarding the size the user had set with the ipympl resize handle (the triangle at the bottom-right corner). The plugin now captures the current `fig.get_size_inches()` before a cutoff-triggered rebuild and rebuilds the clustermap at that size, so the enlarged plot stays enlarged across re-clustering. A fresh **Plot** still uses the default size.
- Fixed the heatmap not appearing (#108): the Heatmap plugin computed the plot (the log even said "render complete") but nothing showed — even with the interactive `ipympl` (`%matplotlib widget`) backend, which renders the Chart histogram and galleries fine. Root cause: the heatmap built its `sns.clustermap` figure **inside** the `with <output>:` display context and then called `plt.show()`, so under interactive mode ipympl emitted the canvas twice (once on creation, once on show) → a blank/duplicate canvas. The Chart histogram is reliable because it builds the figure **outside** its Output and emits it once. The heatmap now does the same: it builds with `plt.ioff()` outside the Output, then emits the interactive canvas exactly once with `display(fig.canvas)` into a fresh `Output` swapped into the panel — preserving all interactivity (cell click, dendrogram-cutoff drag, color-axis select) and the footer docking for the wide layout. The layout-switch flash-then-blank was also fixed (removed a double render on toggle; the reparented footer canvas is now force-repainted after it becomes visible). No static fallback — the live matplotlib canvas is kept.
- Replaced the Matplotlib galleries with an interactive `anywidget` tile grid (#107): the ROI Manager browser and the Cell Gallery no longer render into a Matplotlib figure wired through the interactive `ipympl` backend. Each thumbnail is now a pre-rendered PNG shown in a responsive, clickable CSS grid; clicking a tile activates the ROI / focuses the cell as before, and the hover label is an in-tile CSS tooltip. This removes the `ipympl`-backend fragility (unreliable click/hover in VSCode/Voila) while reusing the existing tile-rendering pipeline unchanged. No new dependency is introduced (`anywidget` was already required). A new shared widget lives in `ueler/viewer/plugin/tile_gallery_widget.py`. Also fixed a pre-existing bug where all FOV-based ROI thumbnails showed "Preview unavailable" (a `snapshot`-before-assignment `UnboundLocalError` in `_render_roi_tile`, swallowed by a broad `except`, dating back to the #91 "No image" mode).
- Added a UELer log console + routed the whole viewer's messages through `logging` (#105 reply 2): when launched with `debug=True` (e.g. `run_viewer(folder, debug=True)`), the viewer shows a dedicated **Log Console** docked at the bottom of the UI — scrollable, selectable/copyable, with a **Clear** button. It is driven by the standard `logging` module (a handler on the `ueler` logger renders records into an `ipywidgets.Output`), so messages no longer land in the notebook cell or a plugin's plot area. A package-wide sweep converted ~140 `print()` calls across the viewer core and all plugins to module loggers (debug traces, info confirmations, warnings, errors at their proper levels), and the plugins' status-label/log helpers now mirror their text into the console as well.
- Fixed heatmap not displayed after Cell Annotation plugin load (#105 reply): `CellAnnotationPlugin.after_all_plugins_loaded()` no longer calls `super()` (preventing `AttributeError` from `PluginBase.load_widget_states()` on `ui_component`); each plugin call in `main_viewer.after_all_plugins_loaded()` is now isolated with `try/except` so a crash in one plugin cannot block later ones from initializing.
- Added Cell Annotation plugin with checkpoint save/load (#105): users can now save the full heatmap annotation state (z-scored median matrix, meta-cluster palette, dendrogram, FlowSOM params, UI settings) as `.h5ad` files and reload them at any time via a parent-child checkpoint browser. Requires new dependency `anndata>=0.10`. The checkpoint tree records parent–child relationships between iteration levels for full workflow provenance.
- Added "Merge same color" option to batch export plugin (#103 reply): when "Export channels separately" is active, channels sharing the same RGB color in the viewer's render settings are grouped into a single composite image (e.g. `FOV1_merged_DNA_CD8.png`) instead of separate files. The checkbox is disabled unless "Export channels separately" is checked.
- Added custom ROI name support (#103): users can now assign a meaningful name to any ROI via the ROI Manager editor. The name is shown in the ROI dropdown label in place of the UUID suffix and used as the filename stem in batch exports (e.g. `FOV1_roi_tumor_core.png`).
- Added separate channels export option to batch export plugin (#102): a new "Export channels separately" checkbox exports each selected channel as an individual image file (e.g. `FOV1_DNA.png`) instead of a merged composite. Works across all three export modes: Full FOV, Single Cells, and ROIs (including map-mode ROIs).
- Added mask opacity control to batch export plugin (#101 follow-up): a new opacity slider (0–100%) lets users control how transparent the mask overlay is in exported images.
- Added explicit mask layer selector and color picker to batch export plugin (#101 reply): a new dropdown lists all available mask layers; a color picker sets the outline color. Mask export is now fully independent of the live viewer's overlay state — `Include Mask` always works regardless of which viewer panel checkboxes are ticked or whether the Mask Painter is active.
- Fixed batch export mask handling (#101): `Include Mask` in the batch export plugin now works independently of the live viewer state. When `Override Mask Palette` is not checked, live Mask Painter per-cell colours are no longer applied to the export (only simple mask outlines appear). When the Mask Painter is disabled and no mask-layer panel checkboxes are ticked, a fallback mask outline is automatically included so exported images always contain mask content when `Include Mask` is checked.
- Fixed export config paths stored as absolute (#99): the registry now stores only the filename and the `output_path` field inside each template is stored relative to `base_folder`; both are expanded back to absolute at load time, so saved configs survive project moves and sharing across machines.
- Fixed batch export partial images in map mode (#98): tile load failures in `_render_map_region_direct` now trigger a retry (50 ms) and then raise `RuntimeError` rather than continuing with a black patch. The job runner records the export as failed and surfaces the error message — no partial image is written to disk.
- Unified ROI display name across all plugins (#96): ROI Manager dropdown, Batch Export dropdown, and ROI Gallery hover tooltip now all use the format `{location} · {marker_set}[{tags}] · {id[:8]}`. Added a hover-tooltip label below the ROI Gallery thumbnail grid that updates as the cursor moves over thumbnails.
- Optimized Mask Painter rendering performance (#95): pan, zoom, and FOV-change renders now call `build_painter_state_maps_for_fov` once per FOV instead of four times, and results are cached per FOV until painter UI state changes. Map-mode tile rendering similarly consolidates four per-tile painter queries into one. Combined with a per-FOV result cache, repeated renders of the same FOV with unchanged settings are now O(1).
- Fixed OOM kernel crash in VSCode when masks are enabled: all masks are now stored in the cache as lazy Dask arrays (matching channel images and annotations) so cache memory is near zero between renders. Previously, `load_masks_for_fov` materialized every mask as a full-resolution NumPy array (~16 MB each); with 200+ FOVs this accumulated to ≥3.2 GB. Pre-labeled masks keep their Dask graph; binary/boolean masks use `dask.delayed(measure.label)` so connected-components labeling is deferred to render time.
- Fixed map-mode mask-load skip when overlay controls are empty: the guard condition in `load_fov` was `if _mask_controls and not any(...)` which fired the load path when `_mask_controls = {}` (falsy); changed to `if not _mask_controls or not any(...)`.
- Added Cell Table Editor plugin: a side-panel plugin that writes a string value to any column (new or existing) in the loaded cell table for all currently selected cells. Works in single-FOV and map mode.
- Fixed Cell Table Editor Apply button and row-matching: `ImageDisplay` now broadcasts `on_selection_change` to all plugins after each click or lasso selection so the Apply button enables immediately; fixed apply logic to match cell table rows by `label_key` instead of `mask_key`, so "Apply to selected cells" now correctly updates the corresponding rows.
- Added freehand lasso selection to the main viewer: a "Lasso Select" toggle button activates one-shot polygon selection; any cell whose mask pixels touch the drawn lasso is added to the selection. Works in both single-FOV and map mode.
- Fixed map-mode lasso selection: the map canvas is viewport-sized and `dest_x0/dest_y0` are viewport-relative downsampled indices, while lasso vertices are global full-res data coordinates. The fix converts mask pixel coordinates to data space via `xmin_px + (dest_x0 + col) * downsample` before the point-in-polygon test.
- Fixed batch export mask outline color (#92 reply): `_snapshot_from_palette_payload` now receives the live viewer's left-panel overlay color and uses it as `mask_type_color` in the palette snapshot, so `Show borders on filled masks` with `border_color_mode="mask_type_color"` renders the correct class overlay color instead of the palette's fallback fill color.
- Added `Output folder` to batch export config templates (#92 reply): the output path field is now saved and restored when loading a named config template.
- Replaced batch export mode `ToggleButtons` with `Tab` widget (#92 reply): the existing mode-tabs widget is now the sole mode selector; the redundant toggle-buttons row and its sync handler have been removed. Added horizontal separator lines below the DPI field and Scale bar % width slider for clearer visual grouping.
- Fixed Mask Painter reply-6 follow-up behavior (#91): global fill opacity once again updates classes whose current opacity still matches the previous global value, inactive classes now follow the `Global fill` default mode without altering active class modes, `Only specified` restores the full list with customized classes first, and the global fill row now has clearer spacing plus a narrower opacity input.
- Fixed Mask Painter reply-5 state handling (#91): globally linked fill opacity and the new global fill toggle now update only classes still inheriting the global/default behavior, `Only specified` keeps customized classes surfaced first, and saved mask-color sets now round-trip the full painter display state with backward-compatible defaults for older palettes.
- Added `No image (masks only)` mode (#91 reply 4): a new left-panel checkbox skips image-layer rendering entirely while keeping masks and annotations visible on a black background. The same state now propagates through single-FOV rendering, map mode, export, ROI thumbnails, and the cell gallery.
- Fixed Mask Painter filled-border dimming (#91 follow-up): thickened filled-mask borders are now clipped to the owning cell and painted after all fill blending, so enabling borders no longer alters neighboring fill colors or makes adjacent filled cells look dimmer.
- Fixed Mask Painter map-mode parity (#91 reply 3): the live stitched-map overlay now resolves the same effective color, fill, opacity, and filled-border state from the current Mask Painter UI that single-FOV rendering uses, so map mode no longer depends on stale cached painter registry values.
- Fixed Mask Painter current-FOV recoloring for NumPy-backed masks (#91 reply 2): the immediate highlight path in `ImageDisplay.set_mask_colors_current_fov()` now materializes either eager or lazy arrays before edge generation, so `apply_colors_to_masks()` no longer crashes when the label slice is already a NumPy array.
- Added a follow-up pass for Mask Painter ROI replay and filled-border colors (#91): map-mode ROI thumbnails and map ROI export now replay the saved painter snapshot, and filled-mask borders can either use the left-panel mask color or the fill color while preserving the captured mask-type color in ROI/gallery/export replay.
- Added Mask Painter opacity and fill-border controls (#91): each active class can now store its own fill opacity, the global opacity control updates classes still linked to the previous global value, and a new border toggle keeps outlines visible on filled masks. The same painter state now propagates through the live viewer, cell gallery, ROI snapshots, and batch export.
- Fixed Mask Painter redraw visibility (#90): single-FOV zoom/pan redraws now consume the current painter UI state directly during `_compose_fov_image()`, so shown classes stay visible, hidden active classes stay hidden, and the plugin starts disabled by default.
- Added "Only specified" toggle to Mask Painter (#89 follow-up): a new `only_specified_checkbox` placed inline with the default color picker; when ON, the active class list is filtered to classes whose assigned color differs from the default color; when OFF, all `current_classes` are restored to active. Classes removed from the active list now remain visible with the default color until they are re-activated.
- Added Add/Remove class feature to Mask Painter (#89 follow-up): users can now start with the first ≤6 classes active and dynamically add or remove classes via inline JS controls (a `<select>` + `+ Add` button in a footer row; `×` per-row buttons to remove). Inactive classes are stored in `available_classes`, keep their remembered class-specific color, and render with the default color while inactive. Implemented via three new traitlets (`available_classes`, `add_requested`, `remove_requested`) and corresponding Python observers (`_on_add_requested`, `_on_remove_requested`).
- Redesigned Mask Painter plugin side panel (#89): replaced horizontal-overflow `HBox` split with a full-width vertical layout, swapped per-class `ToggleButtons` for compact `Checkbox`, collapsed the palette manager into an `Accordion`, replaced `Output` feedback with inline styled `HTML`, added reactive save-button disable, and restructured the top bar into two clean rows. Class list further replaced with a drag-sortable `anywidget` row list (#89 follow-up): each class row shows a drag handle, visibility checkbox, native color picker, class name, and fill checkbox in a single compact row with HTML5 drag-and-drop reordering.
- Replaced ROI Manager advanced tag-filter expression editor with a self-contained `anywidget`-based widget (#88 follow-up 5): the entire expression field, Apply button, operator row, and tag row are now rendered in ESM JavaScript inside a single `ROIExpressionEditorWidget`; `mousedown` + `preventDefault()` on buttons keeps the text input focused so `selectionStart`/`End` are always correct at click time, eliminating all previous async JS→Python reliability issues.
- Redesigned ROI Manager advanced tag-filter expression editor (#88 follow-up 4): helper buttons now edit the expression field purely in JavaScript (reading the live DOM caret, applying spacing rules, updating the field without a Python round-trip); a new "Apply" button triggers gallery refresh on demand, eliminating all async JS→Python caret-sync race conditions.
- Fixed ROI Manager advanced tag-filter helper buttons (#88): operator and tag buttons in the advanced filter now insert syntax reliably, stale cached positions are no longer reused after manual typing, and caret-only reposition is flushed on button pre-click so mid-string insertion follows the visible cursor position.
- Fixed histogram threshold highlight refresh (#87): repeated cutoff changes in the Chart plugin now replace previous viewer highlights cleanly instead of leaving stale cell outlines behind, so small threshold adjustments update the highlighted cells reliably.
- Fixed Chart multi-scatter selection synchronization (#86): selections made in one scatter plot now propagate to all active scatter plots in the same widget while preserving the existing linked mask-highlighting and single-point navigation behavior.
- Added a seventh #85 refinement pass to fix repeated channel widget rows on marker-set load: marker-set channel lists are now de-duplicated in first-seen order during save/update/apply, and redundant channel-control rebuild calls in the marker-set apply path were removed.
- Added a sixth #85 refinement pass to enforce parent channel-panel vertical scrolling when channel content exceeds container height by preventing grouped channel rows from shrink-to-fit compression.
- Added a fifth #85 refinement pass to restore shared scrolling in channel controls (removing per-channel internal scrollbars) and shift the compact color dropdown 5px left in the header row.
- Added a fourth #85 refinement pass that reorganizes each channel control into three rows: row 1 shows checkbox + marker name + compact color dropdown, row 2 shows Min slider, and row 3 shows Max slider; marker name now appears once (header only) and Min/Max slider labels no longer repeat channel names.
- Added a third #85 refinement pass for remaining channel-panel regressions: marker-set action buttons now use a wrap-capable compact row, the confirm-deletion row is width-bounded, channel legend rendering wraps long names safely, per-marker checkbox-to-dropdown spacing is tighter, and color dropdowns are narrower to reduce overflow pressure.
- Finalized issue #85 follow-up (Reply 2) with a container/content split layout policy: container wrappers remain at `max_width: 99%`, while overflow-prone content controls (sliders/text/dropdowns/checkboxes) now use `width/max_width: calc(100% - 5px)` so controls shrink slightly before triggering horizontal scrollbars.
- Reduced unnecessary horizontal scrollbars in control and plugin panels (#85): overflow-prone wrappers now use shrink-safe constraints (`max_width: 99%`, `min_width: 0`, `box_sizing: border-box`) across the main control UI, wide footer panel, chart/heatmap/ROI/export plugin sections, and annotation palette editor, with follow-up tuning for marker-set fields and channel slider controls.
- Fixed map mode black/square view on initial viewer launch (#84 follow-up, five root causes): (1) canvas flushed synchronously before the widget is sent to the browser; (2) toolbar Home view re-synced after display; (3) scroll-wheel zoom correctly triggers tile reload; (4) `load_cell_table` no longer overwrites the map canvas axis limits with single-FOV dimensions; (5) a `_map_needs_initial_render` flag ensures the first `on_draw` after the widget becomes visible always renders real tiles, bypassing the short-circuit once so the browser never shows a blank placeholder.
- Improved mask painter performance on large datasets (#82): restructured the cell colour registry to a nested dict for O(1) per-FOV access, replaced the `iterrows()` global-registration loop with a vectorised bulk write, and added per-class dirty tracking so unchanged classes skip re-registration entirely.
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
