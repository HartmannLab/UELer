# Issue #114 — Broken links to the Scatter plot & Histogram plugins from other plugins

## Problem

After #112 split the old combined **Chart** plugin into two independent plugins —
**Scatter plot** (`chart.py` → `chart_output`) and **Histogram** (`histogram.py` →
`histogram_output`) — the heatmap plugin's "Linked plugins" tab still used the
*old combined-plugin* logic to drive them.

`HeatmapDisplay.update_linked()` (in `heatmap_layers.py`) had a single **Chart**
checkbox and branched on the scatter plugin's y-axis:

```python
if self.ui_component.chart_checkbox.value:
    if self.main_viewer.SidePlots.chart_output.ui_component.y_axis_selector.value == "None":
        _logger.debug("The response of a histogram is not implemented yet.")
    else:
        self.color_points_by_meta_cluster()
        self.highlight_scatter_plot()
```

This was a leftover of the era when `chart_output` rendered *both* a histogram
(when `y == "None"`) and a scatter (when `y` was set). Consequences after the split:

1. **Histogram link never worked.** The `y == "None"` branch only logged
   "not implemented yet" — the real, now-separate `histogram_output` plugin was
   never contacted, so clicking a heatmap cluster produced no response in the
   histogram plugin.
2. **Scatter link was gated by a meaningless guard.** `chart_output` is now
   *always* a scatter plugin, so branching on `y == "None"` no longer makes sense
   and could suppress the scatter response.

## Fix

### 1. Histogram plugin — accept an external selection (`histogram.py`)

Added `HistogramDisplay.show_external_selection(row_indices)`. It is the entry
point other plugins use to push a set of cell-table row indices into the
histogram. It publishes the selection on `selected_indices` (so cell-gallery
forwarding and the viewer-highlight link still work when enabled) and calls the
existing `_refresh_overlays()`, which draws the subset as the **"Selected"**
overlay distribution on every plotted channel — reusing the same machinery that
brush selections already use. Indices outside the currently plotted data are
ignored by `_refresh_overlays`, so the overlay reflects whatever channels/subset
the histogram is showing.

### 2. Heatmap plugin — link to each plugin independently

- `heatmap.py`: renamed the **Chart** checkbox to **Scatter plot** and added a
  new **Histogram** checkbox.
- `heatmap_layers.py`:
  - `update_linked()` now routes to the two plugins independently, each gated by
    its own checkbox, with the dead `y == "None"` guard removed:
    ```python
    if self.ui_component.chart_checkbox.value:
        self.color_points_by_meta_cluster()
        self.highlight_scatter_plot()
    if self.ui_component.histogram_checkbox.value:
        self.update_histogram_distribution()
    ```
  - Added `update_histogram_distribution()`: resolves the active cluster's row
    indices (all cells in the cluster) and calls
    `histogram_output.show_external_selection(...)`. Guards gracefully when the
    histogram plugin is unavailable.
  - Added both checkboxes to the "Linked plugins" tab layout.

## Tests

- `tests/test_histogram_plugin.py` — `show_external_selection` publishes indices,
  forwards to the cell gallery when linked, highlights the viewer when linked, and
  drives the "Selected" overlay (bokeh-only).
- `tests/test_heatmap_selection.py` — new `HeatmapHistogramLinkTests`:
  `update_histogram_distribution` pushes the cluster's indices; `update_linked`
  dispatches to scatter-only / histogram-only / both; no crash when the histogram
  plugin is absent.
- Fixed pre-existing staleness in `tests/test_heatmap_selection.py` left over from
  the `viewer` → `ueler.viewer` package rename (the module could not import at all,
  so its tests never ran): corrected the `spec_from_file_location` paths, the
  `sys.modules` lookup key, the `patch("…heatmap_layers.display")` targets, added
  the required `adapter` stub to the z-score test, and refreshed a stale expected
  z-score value to match the implementation (median aggregation + ddof=1).

Full suite: no new failures/errors vs. the `develop` baseline; +29 net passing
tests (28 new tests plus the now-collectable heatmap-selection file).
