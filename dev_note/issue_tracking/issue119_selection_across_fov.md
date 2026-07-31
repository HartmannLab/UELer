# Issue #119 — selected cells are not highlighted after switching FOV

> GitHub issue: [#119](https://github.com/HartmannLab/UELer/issues/119)
> Status: implemented (see *Implementation* below), plus a reported follow-up — the highlight was also
> lost on zoom/pan — fixed in [*Follow-up*](#follow-up--the-highlight-disappears-after-zooming-in-or-out)

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
* ~~The `update_patches()` call inside `update_display` which runs *before*
  `img_display.set_data(combined)` and so paints outlines onto the array that is about to be
  replaced.~~ Unrelated to the FOV-switch path (the re-projection runs after `update_display`), so it
  was left alone in the first pass — but it is what the reported zoom regression turned out to be, and
  it is fixed in the follow-up below.

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

---

# Follow-up — the highlight disappears after zooming in or out

> Reported on #119 after the fix above: *"the selected cells are now highlighted after switching to
> the next FOV, but they are not persistent after zooming in and out. When the user zooms in and out,
> the selected cells are first highlighted, but then they are cleared after a few seconds. Either they
> are really cleared or they are overwritten by a mask update."*

The reporter's second guess is the right one: **they are overwritten.** The selection cache is never
touched by a zoom — `selected_masks_label` still holds the correct triples for the (unchanged) FOV.

### Root cause

The outline is not a Matplotlib artist. `update_patches` paints white pixels *into* a copy of the RGB
array and pushes it with `set_data`:

```python
combined = self._materialize_combined()      # a copy of image_display.combined
combined[mapped_rows, mapped_cols] = [1, 1, 1]
self.img_display.set_data(combined)
```

So any later `set_data` with a freshly rendered array erases it — and that is precisely the order
`update_display` used:

```python
            if self.masks_available:
                ...
                if hasattr(self.image_display, "update_patches"):
                    self.image_display.update_patches()   # (1) outline the OLD array

        self.image_display.img_display.set_data(combined)  # (2) …then replace it
        self.image_display.combined = combined
```

A zoom or a pan changes the axis limits, which fires a `draw_event` →
[`ImageDisplay.on_draw`](../../ueler/viewer/image_display.py#L236) → `update_display(...)`. Step (1)
repainted the *pre-zoom* array (visible for one frame — the "first highlighted" the reporter saw) and
step (2) then installed the new render without outlines. This is the latent bug listed under *Out of
scope* above; it did not affect the FOV switch only because `_reapply_selection_highlights()` runs
*after* `update_display` returns.

Two consequences, both visible in the report:

1. **The highlight is lost on every zoom, pan and resize** — anything that reaches `update_display`.
2. Even in the frame where it *was* drawn it could be misplaced: `update_patches` compares its expected
   region size against `combined.shape`, which at that point was still the previous viewport's array,
   so it could fall back to the absolute-offset mapping (`rows + ymin_ds`) and land the outline outside
   the visible slice.

### Solution

Repaint the highlight **after** the new array is installed, so it is drawn onto what is actually on
screen:

```python
        repaint_selection = hasattr(self.image_display, "update_patches")   # in the mask branch

        self.image_display.img_display.set_data(combined)
        self.image_display.combined = combined
        self.image_display.img_display.set_extent(xym_r)
        if repaint_selection:
            self.image_display.update_patches()
```

Notes:

* `image_display.combined` is assigned the **clean** render before the repaint, so the outlines are
  never baked into the base image and cannot accumulate across redraws — `update_patches` re-derives
  them from `combined` every time.
* The flag is only set in the single-FOV mask branch, so map mode is unaffected: there `update_patches`
  just delegates to `_update_map_mask_highlights()`, which `update_display` already calls itself.
* Setting the data twice (clean render, then outlined) is deliberate and cheap: `set_data` only swaps
  the array reference, and the single `draw_idle()` at the end coalesces the render.

### Out of scope (deliberately)

* `MaskPainter.on_mv_update_display` re-applies painted colours through `set_mask_colors_current_fov`,
  which also rebuilds from `combined` and would wipe the outline. It fires only when the FOV, the
  identifier or the continuous spec changed — never on a plain zoom — and each of those paths
  re-highlights afterwards, so it is not part of this report. Worth revisiting if a painted-colour
  refresh is ever seen to clear a selection.
* The dead `if getattr(self, '_in_on_draw', False): return` guard at the top of `update_patches`:
  nothing ever sets `_in_on_draw`, so it is a no-op. Left alone rather than removed or wired up, since
  making it live would suppress exactly the repaint this fix relies on.

### Tests

`tests/test_issue119_zoom_highlight.py` (12 tests, 2 classes) runs the real
`ImageMaskViewer.update_display` *and* the real `ImageDisplay.update_patches` against a stub viewer and
asserts on the **pixels of the array left on screen** — the ordering bug is invisible to a test that
only inspects `selected_masks_label`.

* **`ZoomKeepsHighlightTestCase`** — the array installed last carries the outline; the outline traces
  the selected cell and not the unselected one; `image_display.combined` stays outline-free; a
  zoom-in → zoom-out sequence keeps the highlight at every step without leaking into the base image;
  after a zoom the outline is placed in the new array's own coordinates (the mis-mapping above); a
  coarser downsample factor still works. Plus one negative control: a viewport entirely *inside* a cell
  correctly shows nothing, so the assertions are not read as "an outline at any zoom level".
* **`UnchangedBehaviourTestCase`** — without a selection the render is shown as-is, the label-mask
  caches are still populated before the repaint, `on_mv_update_display` is still broadcast, map mode
  still uses `_update_map_mask_highlights` (and never `update_patches`), and
  `_suspend_display_updates` still short-circuits.

Verified as a genuine regression test: with the old ordering restored, 8 of the 12 fail.

```bash
python -m unittest tests.test_issue119_zoom_highlight tests.test_issue119_selection_across_fov \
    tests.test_lasso_selection tests.test_heatmap_selection tests.test_map_mode_activation
python -m unittest discover -s tests -t .    # 833 tests, OK
```

**Not covered by tests, to confirm in a notebook:** the timing the reporter describes ("cleared after a
few seconds"). Steps: link **Main viewer** in the Scatter plot, lasso some cells, then zoom in with the
toolbar, zoom out again, and pan — the outlines should stay put throughout, including while a coarser
downsample level is being loaded.
