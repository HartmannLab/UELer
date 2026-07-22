# Issue #118 — Scatter-plot data points cropped in multi-pair mode

## Problem

In multi-pair mode (**Plot all pairs**), the Scatter plot plugin clips data points that
sit near the edge of the plot. Two independent causes were confirmed:

1. **Not enough width (wide-footer layout).** When ≥2 scatters are active the plugin moves
   into the "wide" footer panel. `build_wide_plugin_pane()`
   (`ueler/viewer/ui_components.py`) laid the plugin's **controls** and **plot content**
   out **side-by-side** in an `HBox`, with the controls pinned to a fixed **6-inch** left
   column. The triangular scatter matrix (`_triangular_grid`, `chart.py`) therefore only
   got `viewer_width − 6in`; each matrix cell (`flex 1 1 0%`) became narrow and jscatter
   clipped markers.
2. **Axis framing had no padding.** `ScatterPlotWidget` (`scatter_widget.py`) never set an
   x/y scale domain, so jscatter framed the view on the raw data **min/max**. Points at
   the extremes were drawn on the very canvas edge and their marker radius spilled past it
   — the "origin at the data minimum + offset" the issue describes.

## Fix

### 1. Stack controls above content in the wide footer (`ueler/viewer/ui_components.py`)

`build_wide_plugin_pane()` now returns a **`VBox`** stacking the controls **on top** of a
**full-width** content box, instead of the side-by-side `HBox`. The control box keeps its
~6in width but is left-aligned inside a full-width parent row, so the plot content spans the
entire viewer width below it. This is the shared helper used by the Scatter plot,
Chart+heatmap, Histogram, and Heatmap wide panes, so all of them now stack. The function
signature is unchanged.

### 2. Pad the jscatter axis domain (`ueler/viewer/plugin/scatter_widget.py`)

Added `_padded_domain(values, fraction=0.05)` — a pure helper returning a `(lo, hi)` domain
padded by 5 % of the data range (with a small symmetric fallback for empty / all-equal /
non-finite data). `ScatterPlotWidget.__init__` now calls
`self._scatter.x(scale=_padded_domain(...))` / `.y(scale=_padded_domain(...))` right after
constructing the `Scatter`, giving every scatter (single-pair and each matrix cell) a small
margin so edge points always stay in view. The static Matplotlib fallback already autoscales
with a margin and was left unchanged.

### 3. Reply — plots not filling width + mis-scaled "double" grid (`ueler/viewer/plugin/chart.py`)

After the full-width layout landed, the developer reported that in the wide footer each
multi-pair scatter (a) left a blank strip to the right of the y-axis label instead of
filling its cell, and (b) showed **two grids at different scales**; manually resizing the
plot corrected both. **Root cause:** the triangular matrix (`_triangular_grid`) was built
with **flexbox** cells (`flex: 1 1 0%`) that have no *definite* size. When the full-width
layout gave the footer lots of width, jscatter's canvas (`width='auto'`, driven by a
`ResizeObserver`) measured a stale/narrow width and drew its WebGL grid at that width, while
the axis SVG overlay used the real cell width — so the two grids disagreed and blank space
was left over. The manual resize fired the observer and reconciled them.

**Fix:** `_triangular_grid` now lays the cells out in a CSS-grid **`GridBox`** — the exact
layout `jscatter.compose` uses — with `N-1` equal **`1fr`** columns and **fixed-height**
rows (`grid_auto_rows`). Grid cells have a definite width/height on first layout, so each
jscatter canvas measures correctly and stretches to fill its cell (no blank strip, single
aligned grid). Row height = canvas height (320px) + jscatter's ~36px axis reserve
(`AXES_PADDING_Y` 20 + `AXES_LABEL_SIZE` 16), so axes never clip. Plot cells are the scatter
widget placed directly as a grid item (CSS grid stretches it); blank lower-triangle cells are
empty `Box`es. Cells are emitted row-major and auto-placed by the grid.

## Tests

- `tests/test_scatter_axis_padding.py` (new) — unit tests for `_padded_domain`: normal range
  → 5 % pad each side; all-equal column → symmetric non-zero pad; empty / all-NaN → `(-1, 1)`;
  non-finite values ignored.
- `tests/test_wide_plugin_panel.py` — `test_build_wide_plugin_pane_stacks_control_above_content`
  asserts the new stacked orientation (control-row on top, content below; the 6in control box
  stays 6in inside a full-width row; `max_width == '99%'` preserved).
- `tests/bootstrap.py` — the shared jscatter `Scatter` stub gained `x()` / `y()` methods
  mirroring the real jscatter API so widget construction exercises the new `scale=` calls;
  its `show()` now returns a container widget with children (like real jscatter) so tests can
  tell a plot cell from a blank cell.
- `tests/test_chart_footer_behavior.py` — the triangular-matrix tests now read the flat,
  row-major `GridBox` children (chunked into rows via a `_grid_rows` helper) and assert the
  `1fr` column count, instead of the old nested `VBox`/`HBox` rows.

Full suite: failure/error set identical to the `develop` baseline (31 pre-existing, no new
failures/errors).
