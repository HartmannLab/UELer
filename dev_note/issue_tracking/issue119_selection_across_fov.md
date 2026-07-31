# Issue #119 — selected cells are not highlighted after switching FOV

> GitHub issue: [#119](https://github.com/HartmannLab/UELer/issues/119)
> Status: implemented (see *Implementation* below)

## Problem

> The selected cells (through histograms or scatter plots) in the main viewer were not highlighted
> after switching to the next FOV.
>
> **Expected behavior:** The selected cells should remain highlighted in the main viewer even after
> switching to the next FOV for a seamless experience.
>
> **Suggested mechanism:** I hypothesize that this issue is caused by the mechanism to improve
> performance by caching the selected cells only for the current FOV. This is fine but the cache has
> to be updated when switching to the next FOV.

### Root cause

The hypothesis in the issue is correct, and this is the whole of it. The highlight is stored in a
form that only describes the FOV that was active when the selection was made.

A plot selection starts life as a set of **cell-table row indices** — FOV-independent, and held by
the plugin that owns it (`ChartDisplay.selected_indices`, `HistogramDisplay.selected_indices`, …).
[`sync_mask_highlights_from_selection`](../../ueler/viewer/plugin/_chart_common.py#L126) then
*projects* those rows onto the active FOV and throws the rest away:

```python
active_fov = viewer.get_active_fov()
if active_fov:
    rows = cell_table.loc[valid_indices, [fov_col, lbl_col]]
    mask_ids = rows.loc[rows[fov_col] == active_fov, lbl_col].astype(int).tolist()   # <- other FOVs dropped
    image_display.set_mask_ids(mask_name=mask_key, mask_ids=mask_ids)
```

[`ImageDisplay.set_mask_ids`](../../ueler/viewer/image_display.py#L514) narrows it further — it keeps
only ids that exist in the *current* FOV's label mask, tagged with the current FOV:

```python
self.selected_masks_label.add(MaskSelection(fov=str(current_fov), mask=..., mask_id=...))
```

and [`update_patches`](../../ueler/viewer/image_display.py#L428) draws only the triples whose `fov`
matches the active one:

```python
selections = [sel for sel in self.selected_masks_label if sel.fov == current_fov]
if not selections:
    ...  # nothing to outline
```

So after a FOV switch `selected_masks_label` holds triples for the **previous** FOV, the filter
matches nothing, and no outlines are drawn. The plugin still holds the correct, FOV-independent
selection — nothing ever asks it to re-project.

On top of that, [`on_image_change`](../../ueler/viewer/main_viewer.py#L2323) actively discarded what
was left and re-applied only two narrow special cases:

```python
if not heatmap_linked:
    self.image_display.clear_patches()          # drop the stale triples
if histogram_plugin is not None and hasattr(histogram_plugin, "highlight_cells"):
    histogram_plugin.highlight_cells()          # cutoff highlight only — needs a cutoff set
if heatmap_linked and heatmap_plugin is not None:
    heatmap_plugin.highlight_cells()            # cluster highlight, only while linked
```

Both of those recompute from scratch for the new FOV, which is why the histogram **cutoff** and the
linked heatmap **cluster** highlights already survived a FOV switch. Nothing covered a row-index
selection, so a scatter-plot lasso/box selection and a histogram **brush** were lost — which is
exactly the pair named in the issue.

## Solution

Keep the plot selection in its FOV-independent form on the viewer, and re-project it onto the new
FOV when the FOV changes. This is the "update the cache on switch" the issue asks for; the per-FOV
cache itself is left alone, so nothing else that reads `selected_masks_label` changes.

```
plugin.selected_indices          {12, 87, 341, …}   FOV-independent, already existed
        │
        ├─▶ viewer.linked_selection_indices          NEW: the viewer remembers it
        │
        ▼  project onto the active FOV
image_display.selected_masks_label  {(FOV2, whole_cell, 7), …}   per-FOV, unchanged
        ▼
update_patches()  → outlines
```

Design decisions:

1. **One record on the viewer, `linked_selection_indices`**, rather than a new `on_fov_change` hook
   in each plugin. Three plugins can drive the highlight and `set_mask_ids` *replaces* the whole
   highlight set, so only one selection is ever displayed — last writer wins. Re-projecting from
   three independent hooks would have them clobber each other in `dir(SidePlots)` order. A single
   record reproduces the displayed state exactly.
2. **Written in `sync_mask_highlights_from_selection`**, the one function every row-index selection
   already funnels through (scatter selection, histogram brush, `show_external_selection`, and the
   heatmap→scatter/histogram links that go through them). No plugin needed changing.
3. **Invalidated in `set_mask_ids`**, so the three `highlight_cells()` methods
   ([histogram](../../ueler/viewer/plugin/histogram.py#L512),
   [chart_heatmap](../../ueler/viewer/plugin/chart_heatmap.py#L334),
   [heatmap_layers](../../ueler/viewer/plugin/heatmap_layers.py#L1158)) — which compute FOV-filtered
   mask ids and call it directly — take the highlight over cleanly. `sync_...` re-establishes the
   record immediately after its own `set_mask_ids` call, which is why the order matters.
   Direct manipulation in the image (click, ctrl-click, lasso, `clear_patches`) invalidates it too:
   those selections are inherently spatial and per-FOV, and must not resurrect a stale plot
   selection on the next switch.
4. **The legacy path is kept verbatim as the fallback.** `_reapply_selection_highlights()` only
   takes the new branch when the record is a non-empty set; otherwise the previous
   `clear_patches()` + cutoff + cluster block runs unchanged. The fix is therefore additive — no
   existing highlight behaviour is altered, only the case that was previously dropped is covered.
   Clearing a plot selection empties it through `set_mask_ids([])`, which drops the record, so
   "clear, then switch FOV" keeps behaving as it does today instead of restoring a dismissed
   selection.
5. **Re-projection happens at the same point in `on_image_change`** as the block it replaces (after
   `update_controls`, i.e. after the new FOV has been composited). `update_display` overwrites the
   canvas data from `self.combined`, so a highlight applied before it would be wiped.

### Out of scope (deliberately)

* Storing the highlight for **all** FOVs at once (i.e. dropping the per-FOV filter in `set_mask_ids`
  so no re-projection is needed). It would make `selected_masks_label` multi-FOV in single-FOV mode,
  which changes what every other consumer sees — `ChartDisplay.trace_cells`,
  `InteractionLayer.trace_cluster`, the cell-table editor, the exporters — and grows the set by the
  number of FOVs for a whole-dataset selection. Rejected as disproportionate risk for the same
  visible result.
* Click / lasso selections persisting across FOVs — they are spatial selections in one image.
* The `update_patches()` call inside `update_display` ([main_viewer.py:4632](../../ueler/viewer/main_viewer.py#L4632))
  which runs *before* `img_display.set_data(combined)` and so paints outlines onto the array that is
  about to be replaced. Unrelated to this issue's path (the re-projection runs after
  `update_display`), but noted here as a real latent bug worth its own issue.

## Implementation

* `ueler/viewer/main_viewer.py`
  * `__init__` — `self.linked_selection_indices = None`.
  * new `_reapply_selection_highlights()` — re-projects a non-empty record via
    `_chart_common.sync_mask_highlights_from_selection`, else runs the previous
    `clear_patches` + histogram-cutoff + heatmap-cluster fallback.
  * `on_image_change` — the inline block is replaced by the single call.
* `ueler/viewer/plugin/_chart_common.py` — `sync_mask_highlights_from_selection` records
  `viewer.linked_selection_indices` after projecting.
* `ueler/viewer/image_display.py` — new `_forget_linked_selection()` helper, called from
  `set_mask_ids`, `clear_patches`, `on_mouse_click` and `_on_lasso_selected`.

## Tests

`tests/test_issue119_selection_across_fov.py` (17 tests, 4 classes):

* **`LinkedSelectionRecordTestCase`** — `sync_mask_highlights_from_selection` records the selection
  (filtered to rows that exist in the table) while only the active FOV's ids reach the per-FOV
  cache; a direct `set_mask_ids` clears the record, as do `clear_patches`, a ctrl-click and a lasso.
* **`ReapplyHighlightsTestCase`** — `_reapply_selection_highlights` re-projects a non-empty record
  and leaves the legacy fallback untouched when there is none (asserts `clear_patches` and the
  histogram/heatmap `highlight_cells` calls, including the `main_viewer_checkbox` gate).
* **`FovSwitchRegressionTestCase`** — the issue itself, on a real `ImageDisplay` and a real
  two-FOV cell table: select cells in both FOVs while FOV1 is active, confirm only FOV1's ids are
  cached, switch the selector to FOV2, re-project, and confirm FOV2's ids are now cached. The same
  test asserts the pre-fix behaviour is what the issue describes by checking that the record — not
  `selected_masks_label` — is what carries the selection across.
* **`MapModeTestCase`** — with map mode active `get_active_fov()` returns `None`, so the record is
  still written and the all-FOV `fov_mask_pairs` path is used; a FOV switch out of map mode
  re-projects correctly.

```bash
python -m unittest tests.test_issue119_selection_across_fov tests.test_lasso_selection \
    tests.test_heatmap_selection tests.test_histogram_plugin tests.test_chart_cell_gallery_link
python -m unittest discover -s tests -t .
```

**Not covered by tests, to confirm in a notebook:** that the outlines are actually *drawn* after the
switch. The tests assert the selection cache is correct for the new FOV and that `update_patches` is
invoked; whether Matplotlib paints them depends on the canvas state, which needs a live check.
Steps: load a cell table spanning ≥2 FOVs, tick **Main viewer** in the Scatter plot, lasso a group
of cells that spans both FOVs, then switch FOV in **Select Image** — the cells belonging to the new
FOV should be outlined immediately. Repeat with a histogram brush.
