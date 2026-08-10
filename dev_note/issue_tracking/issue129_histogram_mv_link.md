# Issue #129 — Unexpected highlighting of cells in the main viewer

**Status:** implemented
**Type:** Bug
**Scope:** Histogram plugin linked-plugin selection (`ueler/viewer/plugin/histogram.py`)

---

## Problem

The Histogram plugin's *Linked plugins* tab has a **Main viewer** checkbox that is supposed
to decide whether a gate made in a histogram outlines the corresponding cells in the main
viewer. With the box **unchecked**, the main viewer still gets highlighted.

### Mechanism

Three separate paths push mask highlights, and only one of them consults the checkbox.

1. **Cutoff mode (the main offender).** A `Tap` on a histogram sets the cutoff and calls
   `highlight_cells(push_to_gallery=True)`
   ([histogram.py:483](../../ueler/viewer/plugin/histogram.py#L483)), which ended with

   ```python
   self._apply_gate(publish=push_to_gallery, highlight=True)
   ```

   — `highlight=True` hard-coded, so `_apply_gate` called
   `sync_mask_highlights_from_selection()` regardless of the checkbox. Brush mode was fine:
   `handle_range` → `set_gate` → `_apply_gate(highlight=mv_linked_checkbox.value)`. So the
   bug reproduced in "Cutoff" interaction mode and not in "Brush" mode, which is why it
   looked intermittent.

   The same unconditional path is reached by `_on_above_below_change` (flipping
   above/below re-applies the gate) and by
   `ImageMaskViewer._reapply_selection_highlights()`, which calls
   `histogram_plugin.highlight_cells()` after every FOV change
   ([main_viewer.py:2409](../../ueler/viewer/main_viewer.py#L2409)) — so an unlinked
   histogram with a live cutoff re-drew its highlight on each FOV switch too.

2. **Unchecking the box did not withdraw the highlight.** Nothing observes
   `mv_linked_checkbox`, so outlines drawn while linked stayed on the canvas after
   unlinking. From the user's side that is indistinguishable from the link still being
   active — the histogram's selection is still "affecting the highlighting of the cells in
   the main viewer".

`viewer.linked_selection_indices` (the FOV-independent record from #119) is *not* a third
leak: `ImageDisplay.set_mask_ids()` calls `_forget_linked_selection()` first, so an empty
`sync_mask_highlights_from_selection(viewer, set())` clears the record as well as the
patches.

---

## Solution

Make the checkbox the single gate for *every* highlight push out of this plugin, and make
toggling it act immediately.

1. `highlight_cells()` passes `highlight=self.ui_component.mv_linked_checkbox.value`
   instead of `True`. Cutoff mode, the above/below toggle and the FOV-change re-apply then
   behave exactly like brush mode. The cell-gallery link is untouched — it has its own
   checkbox and its own `push_to_gallery` flag.
2. New `sync_main_viewer_link()` + `_on_mv_link_change` observer wired in `_wire_events()`:
   - unchecking → `sync_mask_highlights_from_selection(viewer, set())`, which clears the
     patches *and* the `linked_selection_indices` record so the next FOV switch cannot
     resurrect the highlight;
   - re-checking → re-projects the current `selected_indices` onto the active FOV.
   It is a no-op when this plugin has no published selection, so toggling an idle histogram
   cannot wipe a highlight another plugin (scatter / heatmap) owns.

The observer reads `mv_linked_checkbox.value` rather than the `change` payload, so it works
with both an ipywidgets `Bunch` and the lighter change objects used in tests.

### Not changed

The scatter (`chart.py`) and heatmap (`heatmap_layers.py`) plugins already gate their
highlight pushes on their own checkbox; they share symptom 2 (stale outlines after
unlinking), but that is outside this issue and would need the same observer per plugin.

---

## Implementation steps

1. `histogram.py` — `highlight_cells()`: honour `mv_linked_checkbox`.
2. `histogram.py` — add `sync_main_viewer_link()` / `_on_mv_link_change()` and observe the
   checkbox in `_wire_events()`.
3. `tests/test_histogram_plugin.py` — the two cutoff tests that assert on
   `image_display` now opt into the link explicitly; new `TestHistogramMainViewerLink`
   covering: cutoff tap does not highlight when unlinked, does when linked, above/below
   flip stays unlinked, unchecking withdraws the highlight, re-checking restores it, and
   toggling with no selection is a no-op.
4. Docs: `doc/log.md`, `README.md` (new-update section), `dev_note/github_issues.md`.

---

## Tests

```bash
python -m unittest tests.test_histogram_plugin
python -m unittest discover -s tests -t .
```
