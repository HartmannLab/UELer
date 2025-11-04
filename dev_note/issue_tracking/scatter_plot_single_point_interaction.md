# Scatter Plot Single-Point Interaction Investigation

## Action Plan
- Review chart plugin selection flow to document how scatter events propagate to other plugins.
- Trace communication between chart selections and the cell gallery to confirm when the gallery narrows to a single cell.
- Identify hooks tied to `on_fov_change()` to verify whether FOV updates amplify the issue.
- Summarize the root cause and highlight touchpoints for potential fixes per issue #48 expectations.

## Investigation Notes

### Scatter selection wiring
- `ChartDisplay.selected_indices` is an `Observable` that the chart updates on every selection change (`_on_scatter_selection`) and it normalizes the selected IDs before forwarding (`ueler/viewer/plugin/chart.py` around lines 551-609).
- During `setup_observe`, the chart registers `forward_to_cell_gallery` so any selection change pushes the indices into `self.main_viewer.SidePlots.cell_gallery_output.set_selected_cells(...)` whenever the "Cell gallery" link checkbox is active (`ueler/viewer/plugin/chart.py` lines 604-625).
- The test harness in `tests/test_chart_footer_behavior.py` confirms that the chart always forwards the selection payload verbatim when the link is enabled, regardless of how many cells are selected, which matches the behavior users report (see lines 448-473).

### Cell gallery response
- `CellGalleryDisplay.set_selected_cells` simply stores the indices on its own `Observable`, which immediately retriggers `plot_gellery()` and rebuilds the gallery using only the incoming indices (`ueler/viewer/plugin/cell_gallery.py` lines 109-120 and 243-347).
- There is no guard or caching logic to fall back to prior selections; whatever subset the chart provides (including a single-element set) becomes the gallery's definitive list until another plugin overwrites it.

### FOV linkage side-effects
- When the chart's "Main viewer" link is enabled, a single-point scatter selection also jumps the viewer to that cell's FOV (`ChartDisplay._focus_main_viewer` in `ueler/viewer/plugin/chart.py` lines 566-603). This path updates the image selector and triggers `ImageMaskViewer.on_image_change`, which ends with a broadcast to every plugin's `on_fov_change()` (`ueler/viewer/main_viewer.py` lines 510-616).
- The gallery plugin currently inherits the no-op base implementation of `on_fov_change`, so the follow-up event does not undo the single-cell selection. That means the gallery remains in the narrowed state until another bulk selection arrives.

### Trace command parity
- The "Trace" button reuses `_apply_external_selection`, so it funnels the traced cell IDs through the exact same observable pipeline. Hitting "Trace" with a single mask highlighted therefore produces the same single-cell gallery state.

## Root Cause Summary
- The chart plugin's linkage forwards every selection payload directly to the gallery without distinguishing between single-cell and multi-cell contexts. Once the gallery receives a one-element list, it redraws itself around that single index and never widens back out on its own. Because the gallery does not react to `on_fov_change`, the subsequent viewer navigation event has no chance to restore the broader gallery, leaving the interface stuck on a lone cell until the user clears or re-selects a multi-cell subset.

## Opportunities for Fix
- Introduce a transient "single-point selection" flag within the chart plugin so the gallery can decide whether to honor the next `on_fov_change` broadcast, as outlined in issue #48. Clearing that flag after the check would let multi-point selections continue to synchronize normally.
- Extend the gallery plugin's `on_fov_change` to consult the flag and skip re-rendering when the previous action was a single-point scatter click or trace, preventing the gallery from collapsing to one tile.
- Consider adding regression coverage that exercises the single-point selection path with the link enabled to avoid future regressions.
