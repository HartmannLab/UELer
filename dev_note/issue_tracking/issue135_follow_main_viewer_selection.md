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
