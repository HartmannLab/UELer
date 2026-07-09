# Issue #113 — Align Histogram & Scatter channel selection with the left panel

GitHub issue: [#113](https://github.com/HartmannLab/UELer/issues/113)

## Problem
After #112 split the histogram out of the combined chart plugin, the two plugins'
channel pickers drifted away from the **left-panel channel selector** (the
reference UX):

- **Left panel** (`ui_components.py`, `uicomponents.channel_selector`) uses an
  ipywidgets `TagsInput` plus a marker-set dropdown + save/load/update/delete
  buttons. Marker sets live in `viewer.marker_sets`
  (`{name: {'selected_channels': [...], 'channel_settings': {...}}}`).
- **Histogram** used a plain `SelectMultiple` inside a tab (below the Plot
  button), with no marker-set support.
- **Scatter** showed the single-pair X/Y/Color dropdowns on top and hid the
  multi-pair `SelectMultiple` in a "Multi-pair" tab; multi-pair plots rendered as
  a 2-column `jscatter.compose` grid.

The issue asked to make both plugins consistent with the left panel, add
marker-set **loading** (but not *defining*), swap the scatter selectors so
multi-pair is on top, and lay the pairwise scatters out as an upper-triangular
matrix.

## Approach

### Shared builder (`ueler/viewer/plugin/_chart_common.py`)
Both plugins already share this module, so the channel-picker widgetry is
extracted here once so it can't drift again:

- `numeric_columns(viewer)` — the plottable (numeric) cell-table columns.
- `ChannelSelector` (dataclass) — bundles `tags` (a `TagsInput`, or
  `SelectMultiple` when `TagsInput` is unavailable), `marker_set_dropdown`,
  `load_button`, a composed `box`, and `available` (the numeric columns).
- `build_channel_selector(viewer)` — mirrors the left-panel `TagsInput` and adds
  a marker-set *load* control (no save/update/delete).
- `refresh_marker_set_options(selector, viewer)` — repopulate the dropdown from
  `sorted(viewer.marker_sets.keys())`, keeping a valid current selection.
- `apply_marker_set_to_selector(selector, viewer)` — **local read**: set
  `selector.tags.value` from `viewer.marker_sets[name]['selected_channels']`,
  filtered to `selector.available`.

### Marker-set load semantics — local, not global
Loading a set in a plot plugin fills only that plugin's picker; it deliberately
does **not** call `viewer.apply_marker_set_by_name` (which would repaint the main
image viewer). This mirrors the local-read pattern already used by
`export_fovs._resolve_marker_profile`. A regression test asserts the left-panel
selector is untouched after a plugin load.

Each plugin implements `on_marker_sets_changed` (the viewer broadcasts it on
save/update/delete/load of marker sets) and calls it from
`after_all_plugins_loaded` (marker sets are restored from `widget_states.json`
after plugin `__init__`).

### Histogram (`histogram.py`)
- `UiComponent` builds the shared bundle; `channel_selector` is kept as an alias
  to `channel_selector_bundle.tags` so `plot_histograms` is unchanged.
- Layout: the bundle now sits **above** the Plot button (previously the selector
  lived inside a tab, i.e. below Plot); tab 0 keeps only the bin/mode + selection
  controls.

### Scatter (`chart.py`)
- The multi-pair picker (shared bundle + "Plot all pairs") is now the
  always-visible control on top; the single-pair X/Y/Color selector moved into a
  "Single-pair" tab (tabs: Scatter plot / Single-pair / Subset / Trace / Linked
  plugins).
- `multipair_channels` is an alias to `channel_selector_bundle.tags` so
  `plot_all_pairs` is unchanged.

### Triangular layout — `GridBox`, not `jscatter.compose`
`jscatter.compose` packs a flat list row-major into `cols` columns with no
empty/placeholder-cell support, so it cannot express a triangular matrix. The
multi-pair case is instead laid out with an ipywidgets `GridBox` using CSS grid
line placement (`grid_row`/`grid_column`). Selection sync is already handled at
the plugin level (`_commit_scatter_selection`; compose was called with
`sync_selection=False`) and hover sync is unused, so nothing is lost.

- `_scatter_pairs[id] = (x, y)` tracks each view's channels (`ScatterViewState`
  does not expose x/y).
- `_multipair_channels_last` anchors the matrix on the most recent "Plot all
  pairs"; it is **not** reset by `plot_chart`, so a single-pair plot added after
  a matrix is **appended on a new row below it** (flowing left→right), per the
  developer's follow-on request.
- `_triangular_grid()`: for channels indexed `0..N-1`, pair `(i, j)` with `i<j`
  → `grid_row = i+1`, `grid_column = j`, `grid_template_columns = repeat(N-1,
  1fr)`. Extra (non-matrix) views → `grid_row = N + e // (N-1)`,
  `grid_column = 1 + e % (N-1)`. Returns `None` (→ `compose` fallback) when the
  matrix is not fully present (e.g. after a manual removal).

Verified placement for C1..C4 matches the staircase in the issue.

## Tests
- `tests/test_histogram_plugin.py` — `TestHistogramChannelSelector`: shared
  bundle identity, marker-set load populates channels **without** mutating the
  left panel, unknown/non-numeric channels filtered.
- `tests/test_chart_footer_behavior.py` — `MultiPairScatterTests`: multi-pair
  uses the shared selector; 3 channels → a `VBox` of 2 `HBox` rows (row 0: two
  plot cells; row 1: a blank then a plot); single-pair after a matrix lands on a
  new row; removing a matrix view falls back to `compose`. (See "Reply 1" for the
  switch away from `GridBox`.)
- Full suite: failure/error set **identical to the `develop` baseline** (60 both).

## Deferred / out of scope
- The static matplotlib scatter fallback still renders only the first pair; a
  triangular matplotlib subplot grid is a possible follow-up.
- Live `TagsInput` + marker-set loading and the triangular grid rendering are
  verified manually in the notebook (jscatter/Bokeh are interactive-only).

## Reply 1 — Fix triangular rendering + axis clipping

The developer tested the first cut and found (1) the multi-pair layout did not
render as the upper-triangular matrix, and (2) every scatter (single-pair too)
was clipped — the left (y) and bottom (x) axes/labels were cut off.

**Root cause (verified against jscatter 0.22.2 source):** `ScatterPlotWidget`
pinned the jscatter widget's DOM box to exactly the plot height
(`Layout(height="320px")`). jscatter draws axes/tick-labels/axis-labels
**outside** the plot canvas: its own `compose()` computes
`y_padding = AXES_PADDING_Y(20) + AXES_LABEL_SIZE(16) [+ TITLE_HEIGHT(28)]` and
shrinks the plot to `row_height - y_padding`, and it never pins `layout.height`
(a fresh widget has `layout.height = None`, self-sizing from its `height` trait).
Pinning the DOM box to the 320px plot height therefore clipped the axis
furniture. The matrix was compounded by `grid_auto_rows="320px"` squeezing each
~360px widget into a 320px track.

**Fixes:**
- `scatter_widget.py`: the widget layout is now `Layout(width="100%")` (no fixed
  height) so jscatter self-sizes and reserves axis space — fixes single-pair and
  every scatter.
- `chart.py`: `_triangular_grid` was rebuilt from a sparse CSS-grid `GridBox`
  (whose per-child placement could not be verified without a live frontend) into
  a **`VBox` of `HBox` rows** (plain flexbox). Each row has exactly `N-1`
  equal-flex cells (`Box(flex="1 1 0%", min_width="0")`); blank cells fill the
  lower triangle. Row `i` (`0..N-2`) = `i` leading blanks + plots for
  `(channels[i], channels[j])`, `j>i`. Extra (non-matrix) views append as new
  full rows below, padded to `N-1` with blanks. Same "matrix must be complete"
  guard → `compose` fallback otherwise. `GridBox` import removed.

Structure verified headlessly for 3 and 4 channels (`row0: PPP / row1: .PP /
row2: ..P` for 4 channels). Full suite failure/error set unchanged vs. baseline.
The rendered result (jscatter needs a live frontend) must be confirmed in the
notebook.
