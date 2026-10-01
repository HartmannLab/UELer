# ipympl `AttributeError: 'NoneType' object has no attribute 'handle_json'`

A user reported this traceback in the notebook. It carries no UELer frames — it surfaces from the kernel's comm handler — which is what made it hard to place:

```
File .../site-packages/ipympl/backend_nbagg.py:285, in Canvas._handle_message(self, object, content, buffers)
    282     self.manager.handle_json(content)
    284 else:
--> 285     self.manager.handle_json(content)

AttributeError: 'NoneType' object has no attribute 'handle_json'
```

## Problem

`FigureCanvasBase.manager` is `None` in exactly two windows, and ipympl's `_handle_message` dereferences it unguarded:

- **during `savefig`** — `backend_bases.py:2126` wraps the whole of `print_figure` in `cbook._setattr_cm(self, manager=None)`, commented "Remove the figure manager, if any, to avoid resizing the GUI widget";
- **during canvas construction** — `FigureCanvasBase.__init__` sets `self.manager = None` and ipympl assigns the real manager only on the next line of `new_figure_manager_given_figure` (`Canvas(figure)` → `FigureManager(canvas, num)`), by which point the widget's comm is already open.

Neither window is normally reachable, because pyplot and the kernel's comm handler both run on the main thread, so no message can be processed *inside* one. **It becomes reachable as soon as pyplot is used off the main thread**, which is what the export plugin does. `BatchExportPlugin` runs every job on a `ThreadPoolExecutor` (`export_fovs.py:118`), and the worker path reaches pyplot twice:

- `_render_with_scale_bar` (`export_fovs.py:2726`) — `plt.figure()`, then `fig.canvas.draw()` and `fig.canvas.buffer_rgba()`
- `_write_pdf_with_scale_bar` (`export_fovs.py:2752`) — `plt.figure()` + `fig.savefig()`

Under `%matplotlib widget` each `plt.figure()` builds a real ipympl `Canvas` **ipywidget** — opening a comm to the browser — from a background thread, and the `savefig` on it nulls that canvas's manager while the main thread is free to service the comm. The arrow in the user's traceback is on the `else:` branch, i.e. an ordinary interaction message (mouse move, resize, toolbar) rather than `initialized`, which fits the `savefig` window.

A third site, `_preview_single_cell` (`export_fovs.py:2885`), uses `plt.subplots()` + `display(fig)` + `plt.close(fig)` on the **main** thread, so it cannot trigger the crash, but it shares the underlying defect: it goes through the global pyplot state machine to build a throwaway figure, which under the widget backend means a live `Canvas` widget per preview and a `display(fig)` whose result depends on which backend is active.

Two further consequences of routing these through pyplot, independent of the crash:

- every throwaway export figure is registered in the global `Gcf` and becomes the **active figure**, so an export silently changes what a later bare `plt.*` call in the user's own notebook cell targets;
- `_render_with_scale_bar` guards its pixel readback with `if hasattr(fig.canvas, "buffer_rgba")` and silently returns the **unannotated** array when the attribute is missing — which is what happens under any non-Agg-derived backend (`svg`, `pdf`, `template`). The scale bar is dropped with no error.

## Reproduction

In the notebook: `%matplotlib widget`, open the viewer, start a **PDF or scale-bar export** (the only paths through those call sites), and pan or move the mouse over any figure while the job runs. Large FOV sets widen the window.

Headless, no browser, hits it in about three seconds — a worker thread in `savefig` against a main thread delivering a comm message:

```python
import io, threading, time
import matplotlib; matplotlib.use('module://ipympl.backend_nbagg')
import matplotlib.pyplot as plt

MSG = {'type': 'motion_notify', 'x': 1, 'y': 1, 'button': 0, 'buttons': 0, 'modifiers': [], 'guiEvent': None}
fig, ax = plt.subplots(); canvas = fig.canvas
stop, errors = threading.Event(), []

def worker():                      # stands in for the export plugin's ThreadPoolExecutor
    while not stop.is_set():
        fig.savefig(io.BytesIO(), format='png')

t = threading.Thread(target=worker, daemon=True); t.start()
deadline = time.time() + 3
while time.time() < deadline and not errors:
    try:                           # stands in for the kernel's comm handler
        canvas._handle_message(None, MSG, [])
    except AttributeError as exc:
        errors.append(exc)
stop.set(); t.join()
print(errors[0] if errors else 'no hit — try more iterations')
```

## Approach

Two layers, because they address different things: the first removes the defect UELer owns, the second contains an upstream wart UELer does not.

### 1. Keep pyplot off the worker thread (the fix)

The repo already has the correct pattern at `mask_painter.py:1803` — `Figure()` + `FigureCanvasAgg(fig)`, no pyplot, no widget, no `Gcf`. Giving `export_fovs.py` a module-level `_agg_figure(figsize, dpi)` helper and routing all three sites through it removes widget creation *and* the manager-nulling from background threads entirely. The `plt.close()` calls then go away as well: a figure that was never registered in `Gcf` has nothing to close, and dropping the reference is enough.

Deleting the `from matplotlib import pyplot as plt` import outright is part of the fix, not cosmetic — it is what stops a future edit from quietly reintroducing a pyplot call on the worker thread.

Pinning the canvas to Agg also repairs the silent-scale-bar-drop above: `buffer_rgba` is now always present, so the `hasattr` branch stops depending on the user's backend.

`_preview_single_cell` moves to the same helper and renders through `IPython.display.Image` with a PNG buffer, as `mask_painter.py` does. This makes the preview identical under every backend instead of depending on `display(fig)` semantics, and removes a `Canvas` widget per preview click.

### 2. Guard `Canvas._handle_message` (defence in depth)

Worth having even after step 1, because UELer does not own every path into that line — a notebook reconnecting to a stale widget model after a kernel restart reaches it too, as does any third-party library a user runs in the same kernel. A new `ueler/viewer/ipympl_guard.py` installs an idempotent wrapper that drops comm messages arriving while `manager is None` and logs at debug level. Dropping is the correct response rather than a fallback: a message that arrives in that window cannot be dispatched at all, so the choice is between discarding it and crashing the handler.

The patch is applied from `ImageMaskViewer.__init__`, so it is in place before any UELer figure exists, and is a no-op when ipympl is not installed or not the active backend.

## Implementation steps

1. Add `ueler/viewer/ipympl_guard.py` with `install_canvas_message_guard()` — idempotent, import-safe without ipympl, returns whether it patched.
2. Call it at the top of `ImageMaskViewer.__init__`.
3. Add `_agg_figure()` to `export_fovs.py`; convert `_render_with_scale_bar`, `_write_pdf_with_scale_bar` and `_preview_single_cell`; drop the `plt` import and the three `plt.close()` calls.
4. Tests: a new `tests/test_ipympl_none_manager.py` covering the guard and asserting the export module is pyplot-free; update the existing `_preview_single_cell` test in `tests/test_export_fovs_batch.py`, which currently patches `export_fovs.plt.subplots` / `plt.close`.
5. Docs: `doc/log.md`, `README.md` "New Update", and the issue entry in `dev_note/github_issues.md`.

## Out of scope

`chart.py:648` calls `plt.show(fig)` with no `plt.close(fig)`, leaking a live `Canvas` widget per render. Main-thread, unrelated to this crash, and the scatter rendering path has its own ownership rules — tracked separately rather than folded in here.
