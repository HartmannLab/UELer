# Issue #135 — map the main viewer's cell selection into the other plots

> GitHub issue: [#135](https://github.com/HartmannLab/UELer/issues/135)
> Status: implemented
> Type: Feature
> Scope: `_chart_common`, Scatter plot, Histogram, Heatmap, `PluginBase`

## Problem

Selections flow **one way** today. A scatter lasso, a histogram gate or a heatmap cluster can be pushed *into* the main viewer (the "Main viewer" checkbox in each plugin's *Linked plugins* tab, #119/#129), and a cell picked in the image can be written back to the cell table by the Cell table editor — but nothing takes the cells the user picked **in the image** and shows them **in the plots**. The only two paths that go that direction are pull-based, single-shot buttons: the scatter's **Trace** button and the heatmap's **Trace cluster** button, both of which the user has to press after every selection, and both of which only look at the *current* FOV / the *first* selected cell.

## Expected behaviour (from the issue)

* Cells selected in the main viewer are highlighted in the other plots that are linked to the same cell table.
* Only in the plots where the feature is switched on — a per-plugin checkbox in the *Linked plugins* tab.

## Naming

The issue asks for a better name than `Receive Selection`. The existing checkboxes in that tab are named after the *target* of the push (**Main viewer**, **Cell gallery**, **Scatter plot**, **Histogram**), so a name that reads as a direction is the one that disambiguates:

| Candidate | Verdict |
| --- | --- |
| `Receive selection` | Accurate but jargon; says nothing about *from where*. |
| `From main viewer` | Clear, but sits oddly next to the checkbox named `Main viewer`, which means the opposite direction. |
| **`Follow main viewer`** | **Chosen.** Verb-first, names the source, and reads as continuous ("follow") rather than one-shot, which is exactly what distinguishes it from the **Trace** button next to it. |

Description: `Follow main viewer`. Tooltip: *"Highlight the cells selected in the main viewer in this plot"*.

## Mechanism that already exists

`ImageDisplay` already broadcasts every image-side selection change through the plugin bus:

```python
self.main_viewer.inform_plugins("on_selection_change")
```

fired from `on_mouse_click` (click / ctrl-click), `_on_lasso_selected` (lasso, single-FOV *and* map mode) and `clear_patches`. Only `cell_table_editor` implements the hook today (it enables/disables its Apply button). So the feature needs **no new event** — only receivers.

The selection itself lives in `image_display.selected_masks_label` as `MaskSelection(fov, mask, mask_id)` triples, i.e. it is already multi-FOV in map mode.

## Solution

Three receivers, each gated by its own `Follow main viewer` checkbox, all fed by one shared translation helper.

```
image click / ctrl-click / lasso / clear
        │  inform_plugins("on_selection_change")
        ▼
_chart_common.viewer_selection_indices(viewer)      (fov, label) → cell-table row indices
        │
        ├─▶ Scatter    _commit_scatter_selection(idx, push_highlight=False)   → points selected in every scatter view
        ├─▶ Histogram  show_external_selection(idx, push_highlight=False)     → "Selected" overlay on every channel
        └─▶ Heatmap    _apply_cluster_highlights(_map_indices_to_cluster_positions(idx))  → the clusters those cells belong to
```

Design decisions:

1. **The receive path never pushes back.** Each plugin's existing selection entry point ends by pushing mask highlights into the viewer when its "Main viewer" box is ticked. Reached from `on_selection_change` that would call `set_mask_ids`, which *replaces* `selected_masks_label` with the current-FOV projection of what it just received — silently dropping the other FOVs' cells of a map-mode lasso and re-deriving the very selection the user made by hand. Both entry points therefore take a `push_highlight` keyword (default `True`, so every existing caller is unchanged) and the receive path passes `False`. This also means there is no echo: `set_mask_ids` does not broadcast `on_selection_change`, but not calling it at all keeps the user's selection authoritative.
2. **One shared translator, `viewer_selection_indices`**, in `_chart_common`. It matches on `(fov, label)` — the same pair `cell_table_editor` uses — and ignores the mask *name*, because `selected_masks_label` records the mask the pixel was hit in while the cell table only keys on the label id. Matching is done per FOV (`fov == f` & `label.isin(ids)`) so a map-mode selection spanning many FOVs resolves correctly, and label ids are offered as both `int` and `str` so a string-typed label column still matches. It returns an empty set — never raises — on any missing piece (no image display, no cell table, unknown keys), because `inform_plugins` swallows `AttributeError` and would hide a real failure inside a hook.
3. **Ticking the box applies the current selection immediately.** Each plugin observes its own checkbox and runs the receive path on the transition to `True`, so the user does not have to re-click a cell for the link to take effect. Unticking deliberately leaves the plot as it is: unlike the outline the main viewer draws (#129), a selection *in a plot* is a normal, user-clearable plot state ("Clear selection"), and wiping it on untick would also wipe a selection the user made in that plot itself.
4. **The heatmap reuses its scatter-link machinery.** `_map_indices_to_cluster_positions` + `_apply_cluster_highlights` already turn a set of cell-table row indices into highlighted cluster bands (that is how the scatter→heatmap link works); the hook only has to call them. Notably it does *not* reuse `trace_cluster`, which handles a single cell, needs the heatmap grid geometry and ends by calling `highlight_cells()` — a push back into the main viewer, which decision 1 rules out.
5. **`PluginBase` gains an `on_selection_change` no-op**, joining the other lifecycle hooks it documents. `inform_plugins` swallows `AttributeError` for plugins that lack a hook, so this is behaviour-neutral; it makes the hook discoverable next to `on_fov_change` and friends.

### Out of scope (deliberately)

* **Chart (heatmap)** (`chart_heatmap.py`) — it plots `heatmap_plugin.heatmap_data`, whose rows are *clusters*, not cells. A cell-row-index selection has no meaning in its index, so the checkbox would do nothing there.
* **Cell gallery** — not a plot, and it is already a push *target* of every other plugin (`set_selected_cells`). Showing the image selection as a gallery is a reasonable follow-up but is a different link (it renders crops, it does not highlight).
* **Hover** linking, and highlighting *within* a plot with a distinct colour rather than the plot's normal selection state.

## Implementation

* `ueler/viewer/plugin/_chart_common.py`
  * new `build_follow_selection_checkbox()` — the shared `Follow main viewer` widget.
  * new `viewer_selection_indices(viewer)` — `selected_masks_label` → cell-table row indices.
* `ueler/viewer/plugin/plugin_base.py` — new `on_selection_change()` no-op hook.
* `ueler/viewer/plugin/chart.py` — `follow_mv_checkbox` in the *Linked plugins* tab; `_commit_scatter_selection(..., push_highlight=True)`; new `on_selection_change()` and `_on_follow_mv_change()`.
* `ueler/viewer/plugin/histogram.py` — same checkbox; `show_external_selection(..., push_highlight=True)`; new `on_selection_change()` and `_on_follow_mv_change()`.
* `ueler/viewer/plugin/heatmap.py` — `follow_mv_checkbox` in the `UiComponent`, added to the *Linked plugins* tab and observed.
* `ueler/viewer/plugin/heatmap_layers.py` — new `on_selection_change()` / `_on_follow_mv_change()` on `InteractionLayer`.

## Tests

`tests/test_issue135_follow_main_viewer_selection.py`

* **`ViewerSelectionIndicesTestCase`** — the translator: single FOV, multi-FOV (map mode), unknown label ids dropped, string-typed label column, empty selection, and the defensive paths (no image display / no cell table) returning `set()`.
* **`ScatterFollowTestCase`** — unchecked → no change; checked → every scatter view gets the selection and `selected_indices` is published; the main viewer is *not* written back even with "Main viewer" also ticked; clearing the image selection clears the plot; ticking the box applies the selection already on screen.
* **`HistogramFollowTestCase`** — the same matrix through `show_external_selection`, plus: the local gates are dropped (an external selection replaces them) and `push_highlight=True` still works for the heatmap→histogram link.
* **`HeatmapFollowTestCase`** — unchecked → no cluster highlight; checked → the clusters of the selected cells are highlighted and no `set_mask_ids` is issued.
* **`PluginBaseHookTestCase`** — the base hook exists and is a no-op, and `inform_plugins("on_selection_change")` reaches every plugin.

```bash
python -m unittest tests.test_issue135_follow_main_viewer_selection tests.test_chart_footer_behavior \
    tests.test_histogram_plugin tests.test_heatmap_selection tests.test_issue119_selection_across_fov
python tools/run_test_suite.py --max-skips 0
```

---

# Reply 1 — a small selection is invisible in the histogram

## Problem

`Follow main viewer` works in the histogram, but what it draws is easy to miss. The selection is rendered as its own distribution: a second `quad` on the *same* y-axis as the "All" bars (`_build_figures`, `sources[channel]["selected"]`). That encoding is honest, and it is the right one when the selection is a gate covering a good fraction of the cells — which is what it was built for (#112 linked brushing, #127 gating).

It is the wrong encoding for the case #135 introduced. Clicking cells in the image selects a handful of them: 5 cells out of 80 000 puts a 0.006 %-tall orange bar under a full-height blue one. The bar is drawn correctly and is a fraction of a pixel high, so the user sees nothing change and reads the feature as broken.

## Suggested behaviour (from the reply)

> The histogram should highlight the selected cells in a more visible way, such as by changing the color of the bins that contain the selected cells when selected cells are less than 5% of the total.

## Solution

Keep the proportional overlay, and add a second, *categorical* encoding that switches on when the proportional one stops being readable: **tint the whole bin** wherever at least one selected cell landed. The tinted bar answers "where are my cells", which is the only question a five-cell selection can answer anyway — the shape of a five-cell distribution is noise.

Three properties made this the shape to build:

* **The two encodings do not fight.** The tint is drawn between the base bars and the "Selected" overlay: the full bar goes orange, and the true (tiny) count still sits on top of it in the stronger colour. Nothing about the "All" bars or the y-axis changes, so no number on screen becomes a lie — the tint means *this bin contains selected cells*, and the legend says so.
* **It is immune to zoom.** The alternative — scaling the overlay onto a secondary y-axis so the selected distribution fills the frame — was rejected: the two ranges desynchronise the moment the user wheel-zooms in y (the main range is a `DataRange1d` the kernel does not track), and it silently changes what the bar heights mean.
* **It is a data write, not a rebuild.** The tint is a third `ColumnDataSource` written by `_refresh_overlays`, so it costs one more `.data` assignment per channel and cannot disturb a zoom, a pan or another channel's gate marker — the invariant #127 established.

### When it turns on

Per figure, not per plugin: the tint is applied to channel *c* when

```
max(selected counts on c)  <  _FAINT_FRACTION * max(all counts on c)      (_FAINT_FRACTION = 0.05)
```

The reply says "less than 5% of the total [cells]". The rule above keeps the 5 % but measures it on the **peak bar heights of that channel** rather than on the cell count, because that ratio *is* the visibility: it is literally the fraction of the plot height the tallest orange bar gets. Counting cells instead would misfire in both directions — 3 % of the cells concentrated in one bin is perfectly visible and would be tinted for nothing, while 8 % of the cells spread thinly over 200 bins is invisible and would not be tinted. Deciding per channel matters for the same reason: one selection can be sharp on a marker it is defined by and diffuse on every other.

### Control

A `Mark faint selections` checkbox in the *Histogram* tab, on by default, so the automatic behaviour can be switched off by anyone who wants the strictly proportional reading. Toggling it re-runs `_refresh_overlays()` — it never replots. It persists with every other widget state.

## Implementation

* `ueler/viewer/plugin/histogram.py`
  * `_FAINT_FRACTION` and `_HIT_ALPHA` constants next to the existing colours.
  * `_build_figures` — a third `quad` per figure fed by `sources[channel]["hits"]`, drawn between the base and the overlay, legend `Bins with selection`; the per-channel full counts cached in `sources[channel]["full"]` so the refresh does not have to re-bin the column; the new renderer joins the `selection_glyph`/`nonselection_glyph` pinning loop (#127).
  * `_faint_selection_tops(counts, full)` — pure helper returning the tint tops, so the threshold rule is unit-testable without Bokeh.
  * `_refresh_overlays` — writes the `hits` source alongside `selected`.
  * `UiComponent.faint_highlight_checkbox` + `_on_faint_highlight_change` wired in `_wire_events` and placed in `_build_layout`.

## Tests

`tests/test_issue135_faint_selection_highlight.py`

* **`FaintSelectionRuleTestCase`** — the pure helper: below the threshold tints exactly the occupied bins, at/above it tints nothing, an empty selection tints nothing, an all-zero base is safe, and a bin with a single selected cell is tinted to the *full* bar height.
* **`FaintSelectionOverlayTestCase`** — through real Bokeh figures: a one-cell selection fills the `hits` source and leaves `selected` at its true counts; a large selection leaves `hits` empty; the decision is taken per channel; unticking the checkbox clears the tint without replotting; the tint follows a `show_external_selection` push (the `Follow main viewer` path).
