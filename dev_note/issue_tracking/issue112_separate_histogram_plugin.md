# Issue #112 — Separate histograms from the scatter-plot plugin

GitHub issue: [#112](https://github.com/HartmannLab/UELer/issues/112)

## Problem
The combined **Chart** plugin (`ueler/viewer/plugin/chart.py`, `ChartDisplay`,
registered as `SidePlots.chart_output`) rendered **both** histograms and scatter
plots into one shared render host (`self._plot_host`). Because plotting either
one reassigned `self._plot_host.children`, a histogram and a scatter plot could
never be visible at the same time — creating one wiped the other.

The issue asked to split them, then (full scope, approved by the developer) to
add the "further considerations":

* **Scatter:** multi-pair scatter — pick several channels and auto-generate every
  pairwise scatter in one plugin.
* **Histogram:** multiple histograms at once; brush a range in one histogram and
  (a) reflect that selection across the others and (b) overlay the selected
  subset's distribution on every other histogram.

## Approach
Two independent plugins sharing a private helper module:

* `ueler/viewer/plugin/_chart_common.py` (new, `_`-prefixed so the plugin
  auto-loader ignores it) — the shared, de-duplicated logic: `prepare_dataframe`,
  `build_subset_controls`, `build_link_checkboxes`, `subset_options_for`,
  `sync_mask_highlights_from_selection`, `normalize_indices`.
* `ueler/viewer/plugin/chart.py` — kept the class name `ChartDisplay` and the id
  `chart_output` (so the un-guarded references in `heatmap_layers.py`,
  `cell_gallery.py`, and `ui_components.py` keep working); only the
  `displayed_name` changed to **"Scatter plot"**. Removed all histogram code;
  `plot_chart` now requires both X and Y. Added the **Multi-pair** tab:
  `multipair_channels` (`SelectMultiple`) + **Plot all pairs** → one scatter per
  `itertools.combinations(channels, 2)`, reusing `_scatter_views` /
  `_render_scatter_area` and the existing cross-plot selection sync.
* `ueler/viewer/plugin/histogram.py` (new) — `HistogramDisplay`, id
  `histogram_output`, title **"Histogram"**, its own render host, subset controls,
  link checkboxes, and `selected_indices`. Two interaction modes:
  * **Cutoff** (feature parity): click a subplot → set the cutoff for that
    channel → `highlight_cells()` selects cells above/below the cutoff in the
    viewer (single-FOV + map mode) and forwards to the cell gallery. Reproduces
    the old behaviour, including the FOV-change re-highlight.
  * **Brush** (further considerations): a `matplotlib.widgets.SpanSelector` on
    each subplot; brushing a range selects the cells in that range, publishes
    them via `selected_indices` (→ cell gallery + viewer highlight), and redraws
    every subplot with the selected subset overlaid — so a selection on one
    channel is reflected as a distribution across all channels.

## Cross-plugin wiring
* `main_viewer.py` FOV-change block now re-applies the cutoff highlight via
  `SidePlots.histogram_output.highlight_cells()` (guarded), instead of
  `chart_output.highlight_cells()` (which no longer exists on the scatter plugin).
* Scatter/shared references stay on `chart_output`: `selected_indices`,
  `y_axis_selector`, `impose_fov_checkbox`, `color_points`
  (`heatmap_layers.py`), and `single_point_click_state` (`cell_gallery.py`).
* Histogram selection is intentionally scoped to viewer highlight + cell gallery
  + its own overlays; it does not drive the scatter plugin's selection.

## Rendering note
`histogram._render` emits the figure canvas once via `display(fig.canvas)` inside
a fresh `Output` (the cross-backend idiom adopted for the heatmap in issue #108),
which works under both the interactive widget backend and the headless `agg`
backend used in tests.

## Tests
* `tests/test_histogram_plugin.py` (new) — cutoff → gallery/viewer parity,
  brush-range selection + gallery/viewer linking + overlay re-render, multi-channel
  plot state, and a real-Matplotlib render smoke test.
* `tests/test_chart_cell_gallery_link.py` — histogram classes removed; retains the
  scatter selection → mask-highlight / gallery-forwarding tests.
* `tests/test_chart_footer_behavior.py` — added `MultiPairScatterTests`
  (N channels → C(N,2) scatter views).

Full suite: identical failure/error set vs. the `develop` HEAD baseline (no new
failures or errors); the split adds 9 net passing tests.

## Reply — subset-overlay bin edges (fixed)
The initial brush-mode overlay drew the base and subset histograms by passing `bins`
as an integer to each `ax.hist` call, so Matplotlib recomputed edges from each call's
own data range: the full histogram spanned the full range while a narrow subset was
squeezed into its own bins, so the overlay was not comparable. Fixed with
`HistogramDisplay._histogram_bin_edges(channel, bins)` (= `np.histogram_bin_edges`
over the full plotted column), whose edges are passed to **both** the base and the
overlay `ax.hist` calls. Regression tests assert the edges span the full range and are
independent of the current selection.

## Follow-on — re-implemented on Bokeh (done)
The matplotlib render path rebuilt the whole figure on every brush and its
interactivity depended on `ipympl` (the fragility #107 moved away from). The Histogram
plugin now renders on **Bokeh** + **jupyter_bokeh** (added to `pyproject.toml`):

- Per channel a Bokeh `figure` with `quad` bars for the full counts + an overlaid `quad`
  (own `ColumnDataSource`) for the selected subset, both on the **same** bin edges.
  N figures laid out in a `column`, hosted via `BokehModel`.
- **Brush mode:** `BoxSelectTool` → `SelectionGeometry` event → kernel-side
  `handle_range(channel, lo, hi)` → computes the cell selection, drives
  `selected_indices` (→ cell gallery + viewer masks via the existing wiring), and
  recomputes the "selected" overlay source for **every** channel in place (no full
  re-render).
- **Cutoff mode:** `Tap` event → existing `highlight_cells`; threshold drawn as a `Span`
  on the active channel only.
- Binning stays in Python and is unit-tested (`bin_counts`, `_build_figures`, pure
  `handle_range`; `_on_brush` kept as an alias). Imports are guarded so the plugin still
  loads headlessly; when the Bokeh stack is absent it shows an install notice instead of
  rendering.

Tests (`tests/test_histogram_plugin.py`): logic tests + a bokeh-only `_build_figures` /
cutoff-span layout test + a full-stack `BokehModel` render smoke test (each skips if its
dependency is missing). Full suite: failure/error set identical to baseline. The live
brush/tap interaction is verified manually in the notebook (not headlessly testable).

### Brush-activation fix
Brush mode initially only *added* a `BoxSelectTool`; Bokeh's default active drag (pan)
still handled click-drag, so no range could be selected. Fixed by setting
`p.toolbar.active_drag = <BoxSelectTool>` in the brush branch of `_build_figures`, and
guarding `_make_range_handler` on `event.final` so the selection is computed once per
gesture (mouse-up). Regression tests assert `toolbar.active_drag` is a `BoxSelectTool` in
Brush mode and is not in Cutoff mode.

## Out of scope / notes
* `ueler/viewer/plugin/chart_heatmap.py` is a pre-existing stale near-duplicate
  `ChartDisplay` titled "Chart (heatmap)" that also auto-loads. It was left
  untouched here; worth removing in a separate cleanup.
* Renaming `displayed_name` to "Scatter plot" changes the widget-state filename
  (`Chart_widget_states.json` → `Scatter plot_widget_states.json`); old saved
  states are orphaned (harmless).
