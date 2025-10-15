### v0.1.10-rc3
**Chart footer placement**
- Updated `ChartDisplay` so the plugin automatically refreshes the footer when multiple scatter plots are active, keeping controls in the bottom tab while leaving a clear notice in the sidebar ([issue #5](https://github.com/HartmannLab/UELer/issues/5)).
- Added `tests/test_chart_footer_behavior.py` with lightweight widget and data stubs to guard the layout toggle and footer refresh workflow ([issue #5](https://github.com/HartmannLab/UELer/issues/5)).

### v0.1.10-rc2
**Annotation overlays & control layout**
- Added `load_annotations_for_fov` plus rich overlay controls (mode toggle, opacity slider, palette editor launcher) so pixel annotations render as fills, outlines, or both directly in the main viewer; reshaped the left column into a scrollable accordion that keeps annotations ahead of masks and anchors the palette editor for easy access ([issue #21](https://github.com/HartmannLab/UELer/issues/21)).
- Normalised annotation discovery for both Dask and NumPy rasters, enabled palette editing for names containing spaces, and prevented startup crashes when restoring widget states with already-materialised composites ([issue #21](https://github.com/HartmannLab/UELer/issues/21)).

**Footer-wide plugin layouts**
- Introduced the bottom-tab host (`BottomPlots`) and helper utilities so plugins can opt into a full-width footer without vacating the accordion; the heatmap plugin now moves into the footer when “Horizontal layout” is enabled and returns gracefully when toggled off ([issue #24](https://github.com/HartmannLab/UELer_alpha/issues/24)).
- Hardened plugin observer setup to cope with `SimpleNamespace` registries, eliminating the footer-related startup regression ([issue #24](https://github.com/HartmannLab/UELer_alpha/issues/24)).

**ROI management & tagging**
- Shipped a persistent `ROIManager` backend backing onto `<base_folder>/.UELer/roi_manager.csv`, complete with import/export helpers, timestamping, and a dedicated accordion plugin for capture, centring, and metadata edits ([issue #16](https://github.com/HartmannLab/UELer_alpha/issues/16)).
- Rebuilt the tag workflow with a ComboBox + TagsInput hybrid that normalises and preserves new labels even under restrictive widget front-ends, covering multiple regression scenarios in unit tests ([issue #23](https://github.com/HartmannLab/UELer_alpha/issues/23)).

**Mask painter & channel workflows**
- Added mask colour set persistence (`.maskcolors.json`), default-colour management, optional `ipyfilechooser` support, and identifier-aware palette switching, alongside UI affordances to focus on edited classes ([issue #18](https://github.com/HartmannLab/UELer_alpha/issues/18)).
- Centralised channel intensity caching via `merge_channel_max`, updated contrast slider formatting, and clarified mask loading so binary rasters are promoted to labelled images with consistent naming ([issue #15](https://github.com/HartmannLab/UELer_alpha/issues/15)).

**Test & developer support**
- Added targeted suites covering palettes, mask colour persistence, ROI tagging, and footer layout assembly (`tests/test_annotation_palettes.py`, `tests/test_mask_color_sets.py`, `tests/test_roi_manager_tags.py`, `tests/test_wide_plugin_panel.py`) to guard against future regressions ([issues #21](https://github.com/HartmannLab/UELer_alpha/issues/21), [#23](https://github.com/HartmannLab/UELer_alpha/issues/23), [#24](https://github.com/HartmannLab/UELer_alpha/issues/24)).

**Chart gallery upgrades**
- Replaced both chart accordions with `jupyter-scatter` for multi-plot scatter views, including the heatmap variant; selections now sync across all active plots, the footer automatically hosts multi-plot layouts, and histogram fallbacks remain available ([issue #22](https://github.com/HartmannLab/UELer_alpha/issues/22)).
- Added `anywidget>=0.9` as a dependency for the new scatter widgets—make sure the environment that launches JupyterLab has `pip install anywidget` applied (and, if you use a separate Lab environment, install `anywidget` there as well so the `@anywidget/jupyterlab` federated extension is available).
- Streamlined the heatmap plugin's selection handling so scatter clicks and lasso selections update row highlights in-place instead of rebuilding the entire dendrogram, dramatically improving responsiveness during linked exploration ([issue #25](https://github.com/HartmannLab/UELer_alpha/issues/25)).

**Heatmap enhancement**
- Refreshed the heatmap's meta-cluster patches so horizontal (wide) layouts redraw their column highlights immediately after reassignment, keeping footer views synced with cluster edits ([issue #26](https://github.com/HartmannLab/UELer_alpha/issues/26)).
- Restored the heatmap → chart linkage so clicking a heatmap cell recolors and highlights the linked scatter plots via the shared `ScatterPlotWidget` API—no more stale Matplotlib handles when `Chart` is linked ([issue #27](https://github.com/HartmannLab/UELer_alpha/issues/27)).
- Promoted the lightweight `HeatmapModeAdapter` scaffold into the primary layout engine so both vertical and wide heatmaps share the same histogram, cutoff, and palette logic with far less branching inside `viewer/plugin/heatmap.py`.
- Continued the refactor by breaking `HeatmapDisplay` into dedicated data, interaction, and display mixins, leaving the orchestrator class focused on wiring while keeping the public plugin API intact (refactor plan step 5-6).
- Fixed the “Horizontal layout” toggle so it once again flips the plugin into wide mode after the mixin refactor.
- Reduced orientation drift by funnelling heatmap drawing and click handling through shared helpers, so vertical and horizontal layouts highlight, recolor, and log selections identically.
- Eliminated the disappearing heatmap regression when scatter plots trigger a footer refresh—the vertical layout now reattaches its canvas after bottom-panel rebuilds so plotting order no longer matters ([issue #28](https://github.com/HartmannLab/UELer_alpha/issues/28)).


### v0.1.9-alpha  

**Performance Improvements**
- **Enhanced efficiency**: UELer now uses the `dask` package as the backend for lazy loading, significantly improving speed and reducing memory usage. See [issue #7](https://github.com/HartmannLab/UELer/issues/7).

**UI Enhancements**
- **Automatic settings saving**: Some plugins now support automatic saving of settings. See [issue #9](https://github.com/HartmannLab/UELer/issues/9).
- **Status bar**: A new status bar now indicates when an intensive computation is running. See [issue #13](https://github.com/HartmannLab/UELer/issues/13).
- **Mask painter palettes**: Save, load, and manage reusable mask color sets directly from the plugin—palettes live under `<base_folder>/.UELer/` by default, the UI focuses on the actively selected classes (with an optional “show all” toggle), `ipyfilechooser` support provides file dialogs when available, the default color can be changed on the fly, classes that stick with the default are automatically deselected when loading a palette, and each saved set remembers the identifier it applies to—all exported as `.maskcolors.json` files.
- **Annotation overlays**: Pixel annotation images can now be loaded alongside masks. Per-annotation controls in the main viewer let you switch overlay modes (mask outlines, annotation fill, or both), tune opacity, and open the class palette editor to assign colors and friendly labels to annotation IDs.

**Improved Interactivity**
- **Cluster tracing**: In the main viewer, you can now trace cells belonging to the same cluster, which are highlighted in the heatmap.
- **Cell gallery navigation**: Clicking on a cell in the cell gallery brings you to its location in the main viewer.

**New Plugins**
- **Mask Painter**: Enables coloring of mask outlines.
- **FlowSOM**: Supports FlowSOM clustering as part of the FlowSOM workflow integration. See [issue #12](https://github.com/HartmannLab/UELer/issues/12).

**Bug Fixes**
- **Corrected UI settings keys**: Settings now align properly with the UI. See [issue #4](https://github.com/HartmannLab/UELer/issues/4).
- **Fixed cell selection error**: The issue with cell selection has been resolved. See [issue #2](https://github.com/HartmannLab/UELer/issues/2).
- **ROI manager tags**: A ComboBox + TagsInput hybrid now lets you type or pick tags freely; new entries are added to both the active tag list and future suggestions even when the frontend enforces the allowed-tag list. See [issue #23](https://github.com/HartmannLab/UELer/issues/23).
- **ROI manager panel deduplication**: The ROI Manager now lives exclusively in its plugin accordion, eliminating the duplicate block that previously appeared in the left control panel.
- **Annotation palette visibility**: Annotation rasters that load as NumPy arrays now populate the palette editor correctly, and the default overlay mode shows annotation fills alongside mask outlines for immediate feedback.
- **Saved widget state crash**: Restoring widget presets no longer raises an error when the base image buffer is already a NumPy array (common when annotations are enabled).

### v0.1.7-alpha
**Allowing user specified column keys**
- Users can now specify custom column keys in the `Advanced Settings` under `Data mapping`.
- To use this feature, navigate to `Advanced Settings` > `Data mapping`, and enter the desired column keys for each data field.

**Heatmap on subsetted cell table**
- Users can generate a heatmap based on a subset of the cell table data.
- To use this feature, select the desired values in the selected column in the cell table and then generate the heatmap.
- This is a key step in a FlowSOM workflow.

### v0.1.6-alpha
**Automatic settings saving/loading**
- Automatic Saving: Overall settings are now saved automatically whenever changes are made.
- Automatic Loading: The latest settings are loaded when images are opened in the viewer.

### v0.1.5-alpha
**Interactive Histogram**
- When using the chart tool, selecting only the **x-axis** will display a histogram.  
- By clicking on the histogram, you can set a cutoff value to filter cells.  
- Cells above or below the cutoff (as specified) will be highlighted in the currently displayed image.  