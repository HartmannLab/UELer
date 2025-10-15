### 2025-10-15
- Split the heatmap plugin into `DataLayer`, `InteractionLayer`, and `DisplayLayer` mixins (`viewer/plugin/heatmap_layers.py`) so `HeatmapDisplay` now focuses on wiring while reusing the adapter helpers created earlier.
- Trimmed `viewer/plugin/heatmap.py` to initialize the adapter, UI, and data containers, delegating previous helper methods to the new mixins and keeping observable registrations intact.
- Noted the new layering work in the README to document refactor plan steps 5 and 6 for downstream contributors.
- Marked the heatmap plugin as initialized after UI wiring so the horizontal layout toggle flips the adapter back into wide mode and refreshes the footer tabs correctly.
- Unified drawing and event handling between vertical and horizontal layouts by routing histogram ticks, heatmap validation, and click handling through shared helpers, reducing orientation-specific branching.
- Ensured the vertical heatmap keeps its canvas after `refresh_bottom_panel()` rebuilds the footer by reattaching the plot output and asking the heatmap plugin to restore it post-refresh.

### 2025-10-14
- Added `HeatmapModeAdapter` in `viewer/plugin/heatmap_adapter.py` and updated the heatmap plugin to consume the adapter mode for orientation checks as the first incremental step of the documented refactor.
- Replaced the vertical/horizontal layout branches in `viewer/plugin/heatmap.py` with adapter-driven helpers so histogram placement, dendrogram cutoff lines, and cluster palettes now flow through shared code regardless of orientation.

### v0.1.10-alpha
**Annotation overlays & control layout**
- Added `load_annotations_for_fov` plus rich overlay controls (mode toggle, opacity slider, palette editor launcher) so pixel annotations render as fills, outlines, or both directly in the main viewer; reshaped the left column into a scrollable accordion that keeps annotations ahead of masks and anchors the palette editor for easy access ([issue #21](https://github.com/HartmannLab/UELer/issues/21)).
- Normalised annotation discovery for both Dask and NumPy rasters, enabled palette editing for names containing spaces, and prevented startup crashes when restoring widget states with already-materialised composites ([issue #21](https://github.com/HartmannLab/UELer/issues/21)).

**Footer-wide plugin layouts**
- Introduced the bottom-tab host (`BottomPlots`) and helper utilities so plugins can opt into a full-width footer without vacating the accordion; the heatmap plugin now moves into the footer when “Horizontal layout” is enabled and returns gracefully when toggled off ([issue #24](https://github.com/HartmannLab/UELer/issues/24)).
- Hardened plugin observer setup to cope with `SimpleNamespace` registries, eliminating the footer-related startup regression ([issue #24](https://github.com/HartmannLab/UELer/issues/24)).

**ROI management & tagging**
- Shipped a persistent `ROIManager` backend backing onto `<base_folder>/.UELer/roi_manager.csv`, complete with import/export helpers, timestamping, and a dedicated accordion plugin for capture, centring, and metadata edits ([issue #16](https://github.com/HartmannLab/UELer/issues/16)).
- Rebuilt the tag workflow with a ComboBox + TagsInput hybrid that normalises and preserves new labels even under restrictive widget front-ends, covering multiple regression scenarios in unit tests ([issue #23](https://github.com/HartmannLab/UELer/issues/23)).

**Mask painter & channel workflows**
- Added mask colour set persistence (`.maskcolors.json`), default-colour management, optional `ipyfilechooser` support, and identifier-aware palette switching, alongside UI affordances to focus on edited classes ([issue #18](https://github.com/HartmannLab/UELer/issues/18)).
- Centralised channel intensity caching via `merge_channel_max`, updated contrast slider formatting, and clarified mask loading so binary rasters are promoted to labelled images with consistent naming ([issue #15](https://github.com/HartmannLab/UELer/issues/15)).

**Test & developer support**
- Added targeted suites covering palettes, mask colour persistence, ROI tagging, and footer layout assembly (`tests/test_annotation_palettes.py`, `tests/test_mask_color_sets.py`, `tests/test_roi_manager_tags.py`, `tests/test_wide_plugin_panel.py`) to guard against future regressions ([issues #21](https://github.com/HartmannLab/UELer/issues/21), [#23](https://github.com/HartmannLab/UELer/issues/23), [#24](https://github.com/HartmannLab/UELer/issues/24)).

**Chart gallery upgrades**
- Swapped the chart and heatmap scatter panels to `jupyter-scatter`, preserving histogram tools while unlocking multi-plot grids that relocate to the footer automatically and keep selections linked across widgets ([issue #22](https://github.com/HartmannLab/UELer/issues/22)).
- Refreshed the heatmap footer patches so meta-cluster edits repaint in both vertical and horizontal layouts without requiring a full dendrogram rebuild ([issue #26](https://github.com/HartmannLab/UELer/issues/26)).
- Updated the heatmap plugin’s selection observer to reuse the existing dendrogram render and simply refresh highlights, avoiding costly redraws when scatter selections change ([issue #25](https://github.com/HartmannLab/UELer/issues/25)).
- Reinstated heatmap-driven scatter coloring by routing cluster palettes through the new `ScatterPlotWidget` API so linked charts update immediately after heatmap clicks ([issue #27](https://github.com/HartmannLab/UELer/issues/27)).

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