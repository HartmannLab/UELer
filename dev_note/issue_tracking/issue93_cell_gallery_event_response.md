# Issue #93 — Cell Gallery Does Not Always Respond to Events

## Problem

When a user selects a **set of points** in the Chart plugin's scatter plot (with Cell Gallery linking enabled), the status bar briefly shows "busy" but the Cell Gallery does not update to show the selected cells. The regression was introduced by commits that added the "No image (masks only)" mode and Mask Painter opacity/fill controls.

## Root Cause

### `@update_status_bar` silently swallows `AttributeError`

**File:** `ueler/viewer/decorators.py`

```python
# Before fix — AttributeError was caught and discarded
except AttributeError as error:
    print(f"Error: {error}")
```

The `@update_status_bar` decorator, which wraps `plot_gellery()`, was catching `AttributeError` and only printing it to stdout. This produced the observed symptom exactly:
- Status bar flashes "processing" → "ready" (the `finally` block fires in both cases)
- Gallery content is unchanged (the wrapped function never completes normally)
- No visible error to the user

Any `AttributeError` introduced by newly added code paths in `_collect_ui_values`, `_draw_gallery`, or `create_gallery` would be silently swallowed, preventing the gallery from updating while showing no diagnostic feedback.

### Note: single-cell click guard is intentional

The `single_point_click_state == 1` guard in `forward_to_cell_gallery` (`chart.py`) is intentional. Clicking a single point in the scatter plot navigates the main viewer to that cell; the main viewer already shows it in context, so updating the gallery for a single cell is redundant. This guard is preserved.

## Fix

**File:** `ueler/viewer/decorators.py` — removed the `except AttributeError` clause.

The `finally` block still resets the status bar unconditionally. Errors now propagate as visible tracebacks in the notebook, enabling immediate diagnosis.

Per-operation error handling remains in the correct places:
- `_capture_overlay_snapshot` (cell_gallery.py) — try/except for overlay snapshot capture
- `_render_tile_for_index` (cell_gallery.py) — try/except for per-tile rendering

## Tests Added (Cycle 1)

- `tests/test_cell_gallery.py` — `TestUpdateStatusBarDecorator`: verifies `AttributeError` propagates, status bar resets in both success and error paths.
- `tests/test_chart_cell_gallery_link.py` — `TestScatterToGalleryForwarding`: verifies multi-cell selections reach the gallery, single-cell clicks remain blocked, checkbox gating works, and successive selections overwrite.

---

## Follow-up: Gallery Rarely Displays Matplotlib Figure (Cycle 2)

After the decorator fix surfaced errors, three rendering issues in `cell_gallery.py` were identified and fixed.

### RC-A: `clear_output()` inside `with output:` without `wait=True`

**File:** `cell_gallery.py` — `_draw_gallery` and `_show_empty_message`

Calling `clear_output()` inside the `with self.plot_output:` context without `wait=True` races with new content in VS Code's asynchronous Jupyter rendering, producing a blank output. The fix is to call `clear_output(wait=True)` **before** the `with` block, matching the pattern in `chart.py:_render_histogram`.

### RC-B: `plt.show()` without explicit figure argument

**File:** `cell_gallery.py:_draw_gallery`

`plt.show()` without an argument is unreliable in VS Code. `plt.show(fig)` with an explicit figure ensures the correct ipympl canvas widget is sent to the Output widget.

### RC-C: `fig.canvas.new_timer()` raises `AttributeError` on non-ipympl backends

**File:** `cell_gallery.py` — `_draw_gallery` (timer init) and `on_mouse_move` (timer restart)

`new_timer()` is ipympl-specific and raises `AttributeError` when the matplotlib backend is not the widget backend. This aborts the render or raises on hover. Both call sites are now wrapped in `try/except AttributeError`; the gallery renders without hover tooltips when timers are unavailable.

### Fix

All three issues were fixed in `ueler/viewer/plugin/cell_gallery.py`:
- `_draw_gallery` restructured to use `clear_output(wait=True)` before `with`, `plt.show(fig)` with explicit figure, and guarded `new_timer`.
- `on_mouse_move` timer wrapped in `try/except AttributeError`.
- `_show_empty_message` updated to use `clear_output(wait=True)` before `with`.

### Tests Added (Cycle 2)

- `tests/test_cell_gallery.py` — `TestDrawGalleryRendering`:
  - `test_draw_gallery_calls_clear_output_with_wait`
  - `test_draw_gallery_calls_plt_show_with_figure`
  - `test_draw_gallery_handles_missing_new_timer`
  - `test_show_empty_message_clears_with_wait`
