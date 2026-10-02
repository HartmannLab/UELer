# Floating plugin windows — move, resize, collapse

Status: **feasibility assessment and plan, not yet implemented.** Written as the follow-up to the item left out of scope in [issue142_setup_dialog.md](issue142_setup_dialog.md).

## Why this is being reopened

The out-of-scope note at the end of the #142 plan reads: *"dragging is not reachable from `ipywidgets` at all … it needs an `anywidget` `_esm` module and a real window manager."* The first half is true of stock ipywidgets and stays true. The second half was written as a reason to stop, and that reading does not survive contact with the repo:

- `anywidget>=0.9` is a **hard dependency** in `pyproject.toml`, not an optional extra.
- Five modules under `ueler/viewer/plugin/` already subclass `anywidget.AnyWidget` (channel picker, ROI expression editor, Mask Painter class list, tile gallery, cell-annotation checkpoint tree), plus `jupyter-scatter` — see [anywidget_frontend_missing.md](anywidget_frontend_missing.md).
- `channel_picker_widget.py` already carries ~420 lines of hand-written ESM and ~180 lines of CSS as plain Python string constants, with **no npm, no bundler and no build step**, and that ESM already implements HTML5 drag-and-drop with a drop indicator and a keyboard equivalent (#126).

So the prerequisite that the note treated as a blocker is already paid for and already in production. What remains is a design question — *where the window chrome lives relative to the widgets it frames* — and that question has one answer that works and several that do not.

## Problem

Plugin panels are confined to two fixed regions: the side column, and the bottom wide panel. `update_wide_plugin_panel` (`ueler/viewer/ui_components.py`) renders the wide-panel plugins into a `Tab`, so **one of them is visible at a time**. Three consequences:

- Two plugins cannot be compared side by side — switching tabs is the only way to see the second, and the first is gone.
- A plugin cannot be kept in view while the user works elsewhere; watching the histogram update while painting a mask is not possible.
- Every panel that is open competes with the image canvas for vertical space, on a page that is already tall.

A floating window — one the user can move out of the way, shrink to a title bar, or park over a region of the canvas they are not looking at — addresses all three without changing any plugin's own code.

## Design

### 1. The window frames widgets; it does not contain them

The intuitive design is an `AnyWidget` whose ESM renders the panel inside itself. It does not work and should not be attempted: anywidget offers no supported way to render child widget views, and re-parenting DOM that holds an ipympl canvas or a bqplot figure blanks it. `_restore_heatmap` and the `restore_footer_canvas` / `restore_vertical_canvas` hooks exist precisely because these canvases already need nudging when their pane is merely *reused*.

Instead the window is an ordinary `VBox` carrying a marker CSS class, holding the real plugin children, **plus a small sibling `AnyWidget` acting as chrome driver**. The driver's ESM resolves its host with `el.closest('.ueler-window')` — it is rendered inside that box, so the lookup is reliable — and from there sets `transform`, width, height and the collapsed state on the host's style. Nothing is ever re-parented; only CSS changes. This also behaves correctly under ipywidgets' multiple-views property: each view's `closest()` resolves to its own host node.

The title bar, the collapse caret and the resize grip are ordinary ipywidgets inside the same box, so they keep working — as dead chrome — when the ESM does not load.

### 2. Decisions in Python, transport in JavaScript

Geometry (`x`, `y`, `width`, `height`, `collapsed`, `z`) lives in synced traitlets. Viewport clamping, off-screen recovery, z-order and focus are computed **in Python**. The ESM reads those traits, writes them back on `pointerup`, and holds no state of its own.

This is a testability argument before it is anything else. There is no node tooling in the repo and no JS test runner; ESM is currently verified by substring assertions over the Python string (`tests/test_issue126_chip_reorder.py`). Logic placed in JavaScript is logic the suite cannot reach, and a window manager is mostly logic. Keeping the JS to a thin transport keeps the interesting parts under `unittest`.

The cost is one kernel round trip per gesture — on release, not per `mousemove`. During the drag the ESM updates the host's CSS locally, so the motion is smooth regardless of kernel latency.

### 3. Degradation is a starting constraint, not a fallback

The frontend may not resolve at all. This is not hypothetical: it is the open report in [anywidget_frontend_missing.md](anywidget_frontend_missing.md), where a `pip install --user` into a shared HPC environment leaves JupyterLab working and VS Code rendering blank boxes, with nothing in the kernel log.

So the window module follows the established `ANYWIDGET_AVAILABLE` / `ANYWIDGET_STUBBED` pattern: with no real anywidget, the driver is omitted and the box is a plain stacked `VBox` — today's layout exactly. The consequence for the design is that **the content must be correctly placed without any JavaScript**, and the ESM may only add floating on top of a layout that is already right. A window whose contents are only positioned by script is not acceptable here.

### 4. Persistence

`save_widget_states` walks `vars(self.ui_component)` and keeps only `Widget` instances, so a plain holder object escapes it — the same reason `SetupDialog` records its own state. Window geometry therefore goes to its own file in the settings folder, `window_layout.json`, next to `setup_dialog.json`, read and written through `settings_paths.py` and never raising on failure.

Restoring geometry re-applies the clamp before use: a layout saved on a 32-inch monitor must not place a window off-screen on a laptop.

## Risks, worst first

**JupyterLab's windowed notebook.** `position: fixed` is resolved against any ancestor carrying a `transform`, `filter` or `contain`, and a cell scrolled out of view can be unmounted entirely. `ConfirmDialog` already takes this bet, but a scrim that exists for three seconds is a different wager from a panel the user parks and then scrolls past. The textbook fix — portal the node to `document.body` — is re-parenting, i.e. straight back into the canvas hazard. Mitigation: keep windows anchored within the viewer's own root container rather than the document, accept that they are confined to it, and test the scrolled-away case in Lab and VS Code before adopting the module anywhere load-bearing.

**Canvases do not reflow on container resize.** A resized window containing the heatmap or histogram must call back into the existing restore hooks on resize-end. This is why resize syncs to Python rather than being handled entirely in CSS.

**Pointer conflicts with the ipympl canvas.** `setPointerCapture` on the title bar keeps drag moves away from the figure beneath. Separately, a window floating over the canvas will swallow scroll-zoom unless `pointer-events` is managed deliberately — the collapsed title bar in particular should not block a figure it is resting on.

**Untestable chrome.** Mitigated by design decision 2, but not eliminated: the drag itself can only be verified by hand, in each of Lab, VS Code and Binder.

## Implementation steps

Staged so that each tier is independently useful and the expensive tier is optional.

**Tier 1 — float, drag, collapse** (~300 lines ESM/CSS, ~150 lines Python)

1. `ueler/viewer/window_frame.py` — `WindowGeometry` (clamping, off-screen recovery, z-order as pure functions), `WindowFrame` (the `VBox` + driver pair), `_ESM` / `_CSS`, and the `ANYWIDGET_AVAILABLE` fallback.
2. `tests/test_window_frame.py` — the geometry model exhaustively; the fallback path; substring assertions over `_ESM` for the wired events, matching the house style.

**Tier 2 — resize, persistence, focus**

3. Resize grip, resize-end callback into the canvas restore hooks.
4. `window_layout.json` in the settings folder; clamp on restore.
5. Focus/z-order on click.

**Tier 3 — adoption** (deliberately deferred)

6. Dock/undock for wide-panel plugins, which is where this meets the `update_wide_plugin_panel` pane cache and the re-parenting hazard, and where most of the risk lives.

**First adopter:** the setup dialog from #142. It contains no canvas, it is already a scrim-and-card built on the same mount convention, and the worst failure is a dialog that does not drag. Plugin panels stay docked until the module has survived several real sessions in both Lab and VS Code.

## Out of scope

- **Tiling, snapping or saved workspace layouts.** A window manager that only moves, resizes and collapses is a known quantity; one that arranges is a product.
- **Detaching a window into a separate browser window.** Needs the widget manager to render into another document; not reachable from anywidget.
- **Making the main image canvas floatable.** The ipympl canvas owns its pointer handling, and the interaction between its capture and a drag handle is the one combination with no cheap answer.
