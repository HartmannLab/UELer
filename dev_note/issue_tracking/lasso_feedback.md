# Lasso Select: feedback after a loop, and the reason when it selects nothing

## Problem

**Lasso Select** is a one-shot toggle at the top of the viewer. Once armed, a loop drawn on the canvas adds every cell with a mask pixel inside it to `ImageDisplay.selected_masks_label`. The toggle then releases itself and the loop disappears. Two failures looked the same to the user, an empty result with nothing on screen to explain it:

- **The lasso selected nothing.** `_find_masks_in_lasso_single_fov` searches `main_viewer.full_resolution_label_masks`, which `update_display` fills only with the masks ticked under **Masks**, and `_find_masks_in_lasso_map_mode` uses `_selected_mask_names()`, the same set. The mask checkboxes are created with `value=False`, so on a freshly opened viewer every loop returns an empty set. A viewer without a masks folder behaves the same way. In a notebook without `%matplotlib widget`, the figure gets a plain Agg canvas that renders once as a PNG and receives no mouse events, so the loop cannot be drawn at all.
- **A successful loop left no guidance either.** The selected cells were outlined, but nothing said how many were selected, how to change the selection, or which panels act on it.

The manual added to the confusion: `docs/tutorials/user-interface.md` and `docs/tutorials/cell-table.md` said cells are selected when their "mask centroids" fall inside the loop. That is not what the code does.

## Approach

A single line of text beside the toggle (`ui_component.lasso_hint`, an `HTML` widget in the lasso toolbar row). It is the one place a user looks after pressing the button, and it costs no layout space when empty.

| State | Text | Toggle |
| --- | --- | --- |
| Armed, a mask shown | "Draw a loop around the cells and release to select them." | armed |
| Armed, no mask shown | warning: no mask is shown, tick one under **Masks** | armed (a mask can still be ticked before drawing) |
| No masks loaded | warning: no segmentation masks, start with a mask folder | released at once |
| Static canvas | warning: run `%matplotlib widget`, start the viewer again | released at once |
| Loop finished, cells found | "**N cells selected** (T in total)", then how to add, remove or clear cells, and where the selection is used | released |
| Loop finished, nothing found | warning: the reason (no mask shown, or no cells inside the loop) | released |
| Selection changed another way | empty | — |

**Ruled out:**

- **Ticking a mask automatically when the lasso is armed.** It would change what is on screen as a side effect of pressing a different button, and with several masks the choice of which one to tick is arbitrary.
- **Keeping the drawn loop on screen after selection.** The outlines of the selected cells already show the result. A persistent polygon would need its own lifecycle: removal on FOV switch, on map-mode viewport changes and on the next selection.
- **Detecting an ipympl front end that is installed but broken.** The kernel cannot see this, unlike the canvas type. The anywidget check in `widget_frontend_check.py` covers the comparable case for anywidget, and the FAQ points to `%matplotlib widget`.

## Implementation

1. `image_display.canvas_is_interactive(canvas)`: `False` only for the exact `FigureCanvasAgg` and `FigureCanvasBase` types, which is what `%matplotlib inline` (and no `%matplotlib` line) gives. ipympl's `Canvas` and the desktop canvases subclass them.
2. `ImageMaskViewer.on_lasso_select_toggle` consults `_lasso_blocked_reason()` before arming. A blocked arm releases the toggle and shows the reason. An unblocked arm shows the armed hint, or the no-mask warning. Releasing the toggle before drawing clears the armed hint; `_lasso_hint_armed` records whether the current text is the armed hint.
3. `ImageDisplay._on_lasso_selected` calls `main_viewer.report_lasso_result(added, total)` after `on_complete()`. `on_complete()` releases the toggle, and the toggle's handler would otherwise clear the result. A degenerate loop (fewer than three vertices) reports `added = 0`. The hooks are looked up with `getattr`, so lightweight viewer stubs in tests keep working.
4. `ImageDisplay._clear_lasso_hint()` runs from `clear_patches`, the click path of `on_mouse_click` and `set_mask_ids`, so the line never shows a stale count. It does nothing while the lasso is armed.
5. Manual: corrected the selection rule in the two tutorials, and added the FAQ entry "The lasso selects nothing".

## **Update Summary: Lasso feedback and empty-result reasons (uncommitted)**

The lasso gave no feedback after a loop and no reason when it selected nothing. The common cause was that no mask is shown on a fresh viewer.

---

### **Actions Taken**
- **Hint line beside Lasso Select:**
  added an armed hint, a result with counts and next steps, and a reason for an empty result. It is cleared when the selection changes another way.
- **Refuse to arm when no loop could select anything:**
  this covers a viewer with no masks loaded and a static (non-widget) canvas. The toggle releases and the hint says why.
- **Manual corrected:**
  the docs now state the selection rule (any mask pixel inside the loop, masks shown only), and a FAQ entry covers the empty lasso.

---

### **Files Changed**
- `ueler/viewer/image_display.py` — `canvas_is_interactive`; `_on_lasso_selected` reports the result; `_clear_lasso_hint` called from `clear_patches`, `on_mouse_click` and `set_mask_ids`.
- `ueler/viewer/main_viewer.py` — `on_lasso_select_toggle` checks before arming; `_lasso_blocked_reason`, `set_lasso_hint`, `report_lasso_result`.
- `ueler/viewer/ui_components.py` — `lasso_hint` widget in the lasso toolbar row.
- `tests/test_lasso_hint.py` — new, 15 tests: canvas detection (plain Agg, subclass, real ipympl canvas), arm/cancel/blocked states, result counts and empty reasons, and `ImageDisplay` reporting and clearing.
- `tests/test_fullscreen_toggle.py` — the stub UI gains `lasso_hint`.
- `docs/tutorials/user-interface.md`, `docs/tutorials/cell-table.md`, `docs/faq.md`.

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1537 tests, OK
- `mkdocs build --strict` passes.
- Real viewer (benchmark harness, synthetic FOV with masks): arm with no mask shown → warning; cancel → empty; tick a mask and arm → armed hint; a loop → "35 cells selected", toggle released; clear → empty.

## **Update Summary: Scatter plot status line (uncommitted)**

Follow-up. The scatter plot's lasso could look broken in the same silent ways as the image lasso, so it gets the same kind of status line. A second aim was that jupyter-scatter works in every front end, with no reason ever to fall back to the static plot.

---

### **Actions Taken**
- **Why the scatter lasso looked broken:**
  - The plots open in `mouse_mode = "panZoom"`, so a drag pans. A lasso needs the toolbar lasso or Shift+drag (regl-scatterplot's key map), and the panel said neither.
  - With **Main viewer** linked, `sync_mask_highlights_from_selection` outlines only the cells of the FOV shown. A lasso over cells of other FOVs therefore outlines nothing.
  - `trace_cells` reported every failure through `_logger.warning`, which a notebook user does not see.
  - When the anywidget front end is missing, the plots render as empty boxes. The cause was named only by the start-up warning from `warn_about_widget_frontend`.
- **Status line above the plots:**
  `ChartDisplay._selection_status` (an `HTML` widget in `plot_section`) is written by `_update_selection_status(indices, origin)`. `_commit_scatter_selection` calls it with an `origin`: `"scatter"` for a lasso or click in a plot, `"image"` from Follow main viewer, `"external"` for Trace, the heatmap and linked plots. `_render_scatter_area` calls it too, so the lasso tip appears with the first plot. The FOV count (`_cells_in_shown_fov`) matches the cell table's FOV column against the image selector, and is skipped in map mode, where every visible tile is outlined.
- **Front-end problem shown in the panel:**
  `ImageMaskViewer` keeps the result of `warn_about_widget_frontend()` as `widget_frontend_status`. While plots exist and the status is a problem, its message (HTML-escaped) replaces the tip.
- **Static backend stays opt-in only:**
  no code path selects it; only `UELER_SCATTER_BACKEND=static` does. Inspecting the installed wheels shows that jupyter-scatter 1.0.1 ships no `share/jupyter` assets, because its ESM is part of the widget model. anywidget 0.11 ships both a labextension and an nbextension, and a standard install puts them under `sys.prefix`. The start-up check reports `ok` for such an environment. So the scatter works wherever anywidget's front end loads, and the cases where it does not (`--user` install into a read-only environment, version skew between overlapping installs) are exactly the ones `widget_frontend_check` detects.
- **Ruled out:**
  - **Detecting from the kernel whether the plot actually rendered,** with a probe widget that reports back. It would need a timer and a round trip per plot. The start-up check already identifies the conditions under which the plot cannot render.
  - **A fresh-environment install in this session.** It failed on the local SSL trust store, as the release rehearsal script did; the wheel contents were inspected in the existing environment instead.

---

### **Files Changed**
- `ueler/viewer/plugin/chart.py` — `_selection_status`, `_set_selection_status`, `_update_selection_status`, `_cells_in_shown_fov`, `_frontend_problem`; `origin` threaded through the selection paths; Trace failures shown in the panel.
- `ueler/viewer/main_viewer.py` — `widget_frontend_status` kept from the start-up check.
- `tests/test_scatter_selection_status.py` — new, 11 tests.
- `docs/tutorials/scatter-histogram.md`, `docs/faq.md`.

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1548 tests, OK
- `mkdocs build --strict` passes.
