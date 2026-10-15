# Issue #104 — Full-Screen Mode for the Viewer

**GitHub:** https://github.com/HartmannLab/UELer/issues/104

> **Status (2026-10-13):** the sections from *Solution* through *Follow-up Fix 5* describe a first attempt that was **never committed** — `ueler/viewer/fullscreen_widget.py` and its tests do not exist in the repo. The feature was rebuilt from scratch; what is in the code is described under [Restart: host by class, native first, maximize fallback](#restart-host-by-class-native-first-maximize-fallback) at the end of this document.

---

## Problem Description

The UELer viewer is rendered inside a Jupyter notebook cell output area, constraining it to the cell's height and width. Users working with high-resolution MIBI images or wanting a more immersive analysis view have no way to expand the viewer to fill the entire screen.

---

## Solution

Added a full-screen toggle button in a thin toolbar row above the main viewer panels. The button uses a two-strategy approach:

1. **Browser Fullscreen API** (`element.requestFullscreen()`) — expands the output cell natively in browser-based JupyterLab.
2. **CSS overlay fallback** (`position: fixed; inset: 0; z-index: 9999`) — for VSCode's embedded Jupyter webview where the Fullscreen API is blocked.

A `Bool` traitlet (`is_fullscreen`) syncs state between Python and the browser button icon. Pressing Escape or clicking the button again exits full-screen in both modes.

---

## Implementation

### New file: `ueler/viewer/fullscreen_widget.py`

- `FullscreenWidget(anywidget.AnyWidget)` with `is_fullscreen: Bool` traitlet (`sync=True`).
- ESM JavaScript: creates a `<button>` element, walks up the DOM to find the Jupyter output cell, toggles native fullscreen or CSS overlay, and cleans up all document-level event listeners on unmount.
- CSS: `.ueler-fs-btn` (button styling with JupyterLab theme variables) and `.ueler-fs-overlay` (overlay fullscreen fallback class).
- `HasTraits` fallback class when anywidget is absent (test environments).

### Modified: `ueler/viewer/ui_components.py`

- Added `from .fullscreen_widget import FullscreenWidget` import.
- In `display_ui()`: instantiated `viewer.fullscreen_widget = FullscreenWidget()`, wrapped in a right-aligned `HBox(toolbar)`, and inserted as the first child of the root `VBox` (above the main `HBox(ui)`).

---

## Files Changed

- `ueler/viewer/fullscreen_widget.py` — new file: `FullscreenWidget` anywidget + `HasTraits` fallback
- `ueler/viewer/ui_components.py` — import + toolbar row in `display_ui()`
- `tests/test_fullscreen_widget.py` — new test file (13 tests)

---

---

## Follow-up Fix: CSS Overlay Targets Wrong Container (VSCode Bug)

**Bug reported after initial commit:** In VSCode, clicking the button showed only the `×` exit icon; all UELer panels (left panel, image canvas, side plots) disappeared.

**Root cause:** `findContainer()` used `el.closest('.jp-OutputArea-output')` which returns `null` in VSCode (different DOM), falling back to `el.parentElement` — a tiny div holding only the anywidget button. Applying `position: fixed` to that element made it fill the screen while the rest of the viewer remained behind it, hidden.

**Fix:** Replaced `findContainer()` with `findRootContainer()` that walks UP from `el` tracking the **outermost** `.jupyter-widgets` ancestor. That element is the root VBox's DOM node, which contains all panels. Secondary fallback is unchanged. Also:
- Renamed CSS class `.ueler-fs-overlay` → `.ueler-fullscreen-root` (applied to root element)
- Added `savedContainer` closure variable so enter/exit reference the exact same DOM node
- Added `!important` to all overlay CSS properties for stronger specificity
- Cleanup function now also removes the CSS class if the widget is destroyed while fullscreen

---

## Tests

```bash
python -m unittest tests.test_fullscreen_widget -v
```

- ✅ All 13 tests passed (initial implementation)
- ✅ All 13 tests passed (after follow-up fix 1 — DOM traversal)
- ✅ All 14 tests passed (after follow-up fix 2 — Python-side class toggle)

---

## Follow-up Fix 2: Python-side `add_class` / `remove_class` (Reply 2 VSCode bug)

**Bug reported (Reply 2):** After the DOM-traversal fix, the `[x]` button disappeared entirely on click. User clarification: the original bug was code cells overlaid on top of the viewer canvas (not all panels gone), and the button was always visible but the code-cell overlay remained.

**Root cause of Reply 2:** `findRootContainer()` walked up from `el` using `.jupyter-widgets` / `.widget-vbox` class names. In VSCode, ipywidgets may not emit these standard class names. When no matching ancestor was found, the code fell back to `el.parentElement` — the anywidget's own tiny container — applied `position: fixed` to it, and hidden `el` (the button) along with it.

**Fix (Reply 2):**
- Removed ALL JavaScript DOM traversal.
- The ESM now only toggles `model.set('is_fullscreen', ...)` and syncs the button icon.
- In `display_ui()`, a Python closure `_toggle_fullscreen_class` is registered via `viewer.fullscreen_widget.observe(...)`. It calls `root.add_class('ueler-fullscreen-root')` or `root.remove_class(...)` on the actual root VBox object — no DOM guesswork needed.
- Added `add_class` / `remove_class` stubs to `tests/bootstrap.py`.
- Added `test_observer_adds_removes_class_on_root` integration test (14 total).

---

## Follow-up Fix 3: Walk-up native fullscreen (Reply 2 clarification)

**Clarification:** The code cells (notebook input area) were still visible on top of the viewer canvas. This is because CSS `position: fixed` is contained by ancestor elements with `transform` (used by VSCode's notebook scroll container for virtualization). Native `requestFullscreen()` is NOT affected by `transform` ancestors.

**Why `el.requestFullscreen()` alone doesn't work for UELer:** `el` contains only the toolbar button. All viewer panels are in sibling widgets outside `el`.

**Fix:** `tryNativeFullscreen(node)` — a recursive Promise chain that calls `requestFullscreen()` on `node`, catches rejection, and retries on `node.parentElement`. Stops at `document.body`. The first permissive ancestor will contain all viewer panels. CSS fallback (Python-side `add_class`) is kept for when all native attempts are rejected.

Unicode escapes (`⛶`, `×`) replace ambiguous in-string characters to avoid HTML-entity rendering issues.

---

## Follow-up Fix 4: Python-side traitlet-only ESM (Fix 3 introduced blank fullscreen)

**Bug:** Fix 3's walk-up `requestFullscreen()` found the button's immediate container first (VSCode permits `requestFullscreen()` on any element). That tiny element went native fullscreen — covers the output area, but blank.

**Fix:** Reverted to traitlet-toggle-only ESM. Button click sets `model.set('is_fullscreen', ...)`. Python observer `_toggle_fullscreen_class` calls `root.add_class('ueler-fullscreen-root')` / `remove_class`. Added `test_observer_adds_removes_class_on_root` (14 tests total).

**Outcome:** Something expands (the root VBox element gets `position:fixed`) but all sub-widgets are invisible — because the root VBox's DOM element is empty in VSCode.

---

## Follow-up Fix 5: VSCode `.cell_container` approach (DOM inspection confirmed root cause)

**Root cause confirmed via VSCode DevTools:** The DOM screenshot revealed that VSCode renders each ipywidgets child view as a separate `output_container` sibling under `.cell_container` — they are **not** nested inside the VBox's DOM element. Specifically:

- `output_container` 1 (`height: 0px`, `top: 215px`): root VBox DOM element — has the toolbar HBox but NOT the canvas/panels
- `output_container` 2 (`height: 246px`, `top: 243px`): matplotlib canvas and viewer panels
- `output_container` 4 (`height: 41px`, `top: 489px`): wide plugin panel / footer

`position: fixed` on the root VBox expanded an effectively-empty element (only toolbar button visible; the matplotlib canvas and panels are siblings in a different `output_container`).

**Fix:** Removed the Python `_toggle_fullscreen_class` observer entirely. The ESM now:
1. Calls `el.closest('.cell_container')` to find the shared ancestor of all sibling widget views in the cell.
2. Applies `ueler-fs-active` CSS class to that container.
3. Companion CSS overrides VSCode's `position: absolute; height: 0px; max-height: 0px; overflow: hidden` on every `output_container` child, making all sibling views visible and stacked normally inside the fullscreen overlay.
4. Falls back to size-based ancestor walk (`rect.width > 400, rect.height > 100`) + `ueler-fullscreen-root` class for JupyterLab (where VBox children ARE properly DOM-nested).

Native `requestFullscreen()` is also tried on the found container before falling back to CSS.

**Tests:** Removed `test_observer_adds_removes_class_on_root` (observer gone). 13 tests pass.

---

## Restart: host by class, native first, maximize fallback

### What the first attempt got wrong

Every fix in the first attempt changed *how* the host element was found, and every one of them found the wrong node in VS Code: `.jp-OutputArea-output` (absent there), the outermost `.jupyter-widgets` ancestor (class names differ there), the first ancestor that accepted `requestFullscreen()` (the button's own wrapper, which went fullscreen blank), and a size-based ancestor walk. Fix 3 is the useful data point: **`requestFullscreen()` does resolve inside the VS Code output webview.** Native fullscreen was never the problem; the target was. Fix 5's reading of the DevTools tree, that VS Code renders a `VBox`'s children as sibling `output_container`s, was never confirmed and is inconsistent with how the ipywidgets HTML manager builds child views, so the restart does not design around it.

### Design

- **The host is named, not searched for.** `display_ui` calls `root.add_class('ueler-root')` on the root `VBox`, and the button is mounted *inside* that box, as the first child. The ESM resolves its host with `el.closest('.ueler-root')`. `closest()` cannot return a node that is not an ancestor carrying that exact class, so the "fullscreened the button's wrapper" failure cannot recur, and the test suite asserts that the ESM contains no `parentElement` walk.
- **Native fullscreen first.** VS Code renders outputs in a webview drawn beneath the editor's own layer, so no CSS inside an output can paint over the code cells; that was the "code cells on top of the canvas" report. Only the Fullscreen API escapes that layer, so it is always tried first.
- **Maximize as the fallback.** When `document.fullscreenEnabled` is false, or `requestFullscreen()` rejects (an iframe without `allow="fullscreen"`, a locked-down webview), the host gets `ueler-maximized`: `position: fixed; inset: 0` with an opaque background. In JupyterLab and Notebook this fills the browser tab. In a VS Code webview it can only fill the output layer, which is the limit of what that layer allows.
- **Python sees the outcome.** `FullscreenToggle.mode` is a synced trait (`off` / `native` / `maximized`) written by the ESM. Setting it to `"off"` from Python tears down whichever mode is active. All teardown runs from the single `change:mode` handler, whether the button, Esc, the browser's own fullscreen exit or Python asked for it, so `exitFullscreen()` is never called twice.
- **Multiple views stay independent.** Each view remembers the node *it* enlarged (`active`) and ignores fullscreen changes and Esc presses that do not concern it, so displaying the viewer twice cannot make one view switch the other off.
- **Canvases are told to reflow.** After every enter and exit the ESM dispatches a `resize` event on the next frame. jupyter-scatter and bqplot re-measure on it; the ipympl canvas keeps its Python-side 6×6 in figure size, so full screen gives the panels around it more room but does not enlarge the image itself.
- **No anywidget, no button.** `build_fullscreen_toggle()` returns `None` without a real anywidget, and the toolbar row is skipped, so the layout is then exactly what it was before.

### Ruled out

- **Python-side `add_class` as the fullscreen mechanism** (first attempt, Fix 2/4). A class toggled from Python can only produce the CSS overlay, which cannot cover VS Code's code cells. Python now adds a class only to *name* the host.
- **Portalling the root to `document.body`.** Re-parenting DOM that holds an ipympl canvas blanks it (see [window_frame_floating_panels.md](window_frame_floating_panels.md)).
- **Enlarging the ipympl figure in full screen.** It would need the viewer to recompute the figure size and re-render from Python on every mode change. That is worth doing, but it is a separate change with its own render-path risks.

### Open: verify by hand in VS Code

The fake-DOM tests prove the ESM's logic, not VS Code's webview. Before closing #104, check in VS Code: (1) the button enters native fullscreen with the whole viewer visible; (2) Esc and the button both exit; (3) the setup and confirm dialogs still show while in fullscreen (they are inside the root, so they should). If (1) falls back to maximize instead, the code cells will show over the overlay; more CSS cannot fix that, so record which VS Code version and setup refused the request before trying anything else.

---

## **Update Summary: full-screen toggle, rebuilt (uncommitted)**

Adds a full-screen button at the top-left of the viewer that tries native browser fullscreen and falls back to a maximize overlay, replacing the uncommitted first attempt.

---

### **Actions Taken**
- **Added `FullscreenToggle`:**
  an anywidget icon button whose ESM resolves its host with `el.closest('.ueler-root')`, tries `requestFullscreen()` on it, falls back to the `ueler-maximized` overlay, handles Esc in both modes, and syncs the outcome to the `mode` trait. Includes a `HasTraits` stand-in for the headless test bootstrap.
- **Mounted it in `display_ui`:**
  the root `VBox` is tagged `ueler-root`, and the button sits in a left-aligned toolbar row as the root's first child. It is omitted when anywidget cannot render.
- **Tested the ESM by running it:**
  as in #126, the ESM is executed under `node`, here against a minimal fake DOM and model, so the assertions cover what the browser runs.

---

### **Files Changed**
- `ueler/viewer/fullscreen_toggle.py` — new: `FullscreenToggle`, `build_fullscreen_toggle`, ESM and CSS
- `ueler/viewer/ui_components.py` — `display_ui` tags the root and mounts the toolbar row
- `tests/test_fullscreen_toggle.py` — new: 10 tests (Python contract, `display_ui` mount, four ESM scenarios under `node`)
- `dev_note/topic_viewer_runtime_ui.md`, `doc/log.md`, `README.md` — documentation

---

### **Tests**
```bash
python -m unittest tests.test_fullscreen_toggle
python -m unittest discover -s tests
```
- ✅ All tests passed (10 new; full suite 1404, OK)

---

## **Update Summary: VS Code grows in place, canvas fills the screen (follow-up, uncommitted)**

User report on the rebuilt version: in VS Code, clicking the button changed the icon and expanded the viewer, but **code cells were drawn on top of it while markdown cells were not**. So VS Code refused `requestFullscreen()`, and the maximize overlay ran instead.

That split pins down the cause. VS Code renders markdown cells inside the same output webview as widget outputs, so the overlay covers them. Code-cell editors are Monaco instances in VS Code's own workbench DOM, drawn on a layer above the webview, and nothing inside the webview can paint over them. The overlay also filled exactly the visible editor area, which confirms that the webview's `window.innerHeight` is the visible editor height. No CSS fix exists for this; the webview simply cannot cover the code cells.

---

### **Actions Taken**
- **Added an `expanded` mode for VS Code:**
  when full screen is refused and the page is a VS Code webview (its `<body>` carries a `vscode-*` theme class), the root gets `ueler-expanded` instead of the overlay: in-flow, `height: var(--ueler-expanded-height)` set to the webview's `window.innerHeight`, internal scroll. VS Code resizes the output to match, so the cells below are pushed down instead of drawn over the viewer. The user chose this over keeping the overlay. Other pages that refuse full screen still get the fixed overlay.
- **The image canvas now fills the space:**
  the ESM reports the enlarged size as a `viewport` trait (`[]` when off, re-measured on window resize with a 200 ms debounce). `attach_canvas_fit` in Python computes the canvas size from it, subtracting the 350 px left panel, the 576 px side accordions when present and a fixed allowance for chrome. It calls `fig.set_size_inches(..., forward=True)`, re-renders at the current downsample factor, and restores the saved size on exit. The size calculation is in Python so the unit tests can reach it. The figure's `_original_dpi` is used as the CSS-pixel scale because ipympl raises `fig.dpi` by the device pixel ratio.
- **Superseded:** the earlier *Known limit* that the image does not grow in full screen no longer applies.

---

### **Files Changed**
- `ueler/viewer/fullscreen_toggle.py` — `expanded` mode, `viewport` trait, `fit_canvas_inches`, `attach_canvas_fit`
- `ueler/viewer/ui_components.py` — attaches the canvas fitter in `display_ui`
- `tests/test_fullscreen_toggle.py` — 16 tests (adds a VS Code webview scenario under `node`, canvas-fit math, enlarge/restore)
- `doc/log.md`, `README.md`, `dev_note/topic_viewer_runtime_ui.md` — documentation

---

### **Tests**
```bash
python -m unittest tests.test_fullscreen_toggle
python -m unittest discover -s tests
```
- ✅ All tests passed (16 in the module; full suite 1410, OK)

---

## **Update Summary: expanded mode grew without end (follow-up 2, uncommitted)**

User report: in grow-in-place mode the canvas kept getting taller, and the image turned into a broken-image icon.

The previous follow-up's inference was wrong: the overlay looking right did *not* show that the webview's `window.innerHeight` is the visible editor height. VS Code sizes the output webview to the notebook *content*. Expanded mode set the viewer's height from `innerHeight` and re-measured on every window `resize`, which formed a loop: the canvas grew, the output grew, the webview grew, `innerHeight` grew, `resize` fired, and the canvas grew again. The figure eventually got too large for ipympl to send as an image, hence the broken-image icon.

---

### **Actions Taken**
- **Height fixed on entry, bounded by the screen:**
  `computeExpandedHeight()` takes `min(innerHeight, screen.availHeight − 160)` once, with a 400 px floor. The 160 px allows for VS Code's title bar, tabs, breadcrumbs, notebook toolbar and status bar. `screen.availHeight` does not depend on layout, so it cannot feed back.
- **No re-measuring in expanded mode:**
  the window `resize` handler returns early in `expanded`, because there the window grows *because* the viewer did. Native and maximize still re-measure: their host is the screen or a fixed overlay, whose size does not depend on the content.
- **Python-side cap:**
  `fit_canvas_inches` clamps each viewport side to `MAX_VIEWPORT_PX` (4096 CSS px), so no frontend bug can again request a figure too large to render.

---

### **Files Changed**
- `ueler/viewer/fullscreen_toggle.py` — `computeExpandedHeight`, no resize re-measure in `expanded`, `MAX_VIEWPORT_PX`
- `tests/test_fullscreen_toggle.py` — 18 tests: adds a `vscode-runaway` scenario (20000 px webview that grows further after entry) and the cap
- `doc/log.md`, `README.md`, `dev_note/topic_viewer_runtime_ui.md` — corrected the height description

---

### **Tests**
```bash
python -m unittest tests.test_fullscreen_toggle
python -m unittest discover -s tests
```
- ✅ All tests passed (18 in the module; full suite 1412, OK)

---

## **Update Summary: keep the expanded viewer anchored (follow-up 3, uncommitted)**

User request: when scrolling up past the top of the anchored viewer, the viewer should return to its anchor point.

A literal snap-back is not reachable. A widget runs inside VS Code's output webview and has no API to scroll the notebook, and the webview cannot tell where the editor has scrolled it. What *is* reachable is the cause. VS Code's webview preload listens for `wheel` on `window` and forwards an event to the notebook scroller when no element on its path can still scroll in that direction, unless the event is `defaultPrevented`. Reaching the top of the viewer's internal scroll is exactly that case, so the notebook, and the viewer with it, moved.

---

### **Actions Taken**
- **Wheel guard in expanded mode:**
  while expanded, a non-passive `wheel` listener on the root walks from the event target up to the root. If any element on that path has `overflow-y: auto|scroll` and room to scroll in the wheel's direction, the event is left alone; otherwise it is `preventDefault()`-ed. Inner panels, the viewer's own scroll and the ipympl scroll-zoom therefore behave as before, and only the hand-off to the notebook is suppressed, in both directions. The listener is added on entering `expanded` and removed in `teardown()`.
- **Not applied to native or maximize:** in those modes the viewer covers the notebook, so there is nothing to drift away from.
- **Consequence:** with the pointer over the expanded viewer, the wheel cannot scroll the notebook at all. Leaving the mode (button or Esc), or scrolling with the pointer outside the viewer, still works.

---

### **Files Changed**
- `ueler/viewer/fullscreen_toggle.py` — `canScroll`, `onWheel`, wired in `fallback()` / `teardown()`
- `tests/test_fullscreen_toggle.py` — 19 tests: the VS Code scenario now checks wheel at the viewer top (cancelled), over a scrollable panel and with room to scroll (passed through), and that the listener is removed
- `doc/log.md`, `README.md`, `dev_note/topic_viewer_runtime_ui.md`

---

### **Tests**
```bash
python -m unittest tests.test_fullscreen_toggle
python -m unittest discover -s tests
```
- ✅ All tests passed (19 in the module; full suite 1413, OK)

---

## **Update Summary: latch wheel gestures instead of blocking them (follow-up 4, uncommitted)**

User report on follow-up 3: in grow-in-place mode, nothing could be scrolled at all.

Follow-up 3 cancelled every wheel event that nothing inside the viewer could absorb. The expanded viewer spans the full editor width, so the pointer is always over it, and every attempt to scroll the notebook was cancelled. If the viewer had no internal overflow, nothing scrolled at all. The flaw is in the rule, not in its implementation: a widget cannot see where VS Code has scrolled its output, so it cannot tell "past the anchor" from "elsewhere in the notebook", and any rule based on position blocks too much.

---

### **Actions Taken**
- **Scroll latching:**
  a wheel event that scrolls something between its target and the root sets `latchUntil = now + 400 ms`. A later event that would leave the viewer (nothing on its path can scroll) is `preventDefault()`-ed only while the latch is live, and each swallowed event extends it, so trackpad momentum stays latched until the gesture ends. After a pause, a fresh gesture reaches the notebook. This is the scroll-chaining behaviour of maps and mobile scroll views: a flick that reaches the top of the viewer stops there, while the notebook stays scrollable.
- **What it gives up:** a deliberate new gesture over the viewer scrolls the notebook, so the viewer *can* still be scrolled away from its anchor. This is intended; the alternative is the unscrollable notebook just reported.

---

### **Files Changed**
- `ueler/viewer/fullscreen_toggle.py` — `onWheel` latches by gesture (`LATCH_MS`, `scrollsInsideViewer`)
- `tests/test_fullscreen_toggle.py` — the VS Code scenario drives a fake clock: a fresh gesture and a post-pause gesture reach the notebook, inside scrolling is untouched, momentum after hitting the top is swallowed
- `doc/log.md`, `README.md`, `dev_note/topic_viewer_runtime_ui.md` — corrected the follow-up 3 description

---

### **Tests**
```bash
python -m unittest tests.test_fullscreen_toggle
python -m unittest discover -s tests
```
- ✅ All tests passed (19 in the module; full suite 1413, OK)

---

## **Update Summary: anchor by visibility, not by gesture (follow-up 5, uncommitted)**

User report on follow-up 4: the gesture was still passed to the notebook.

Together, the two reports settle how VS Code behaves. Follow-up 3 (cancel every boundary event) froze the notebook, so `defaultPrevented` **is** honoured. Follow-up 4's latch never engaged, so the scrolls that move the notebook are not continuations of a scroll inside the viewer. A rule based on timing cannot express "stop at the anchor"; what is needed is the viewer's position in the editor, which follow-up 4 had wrongly called unobservable.

---

### **Actions Taken**
- **Position from `IntersectionObserver`:**
  with the implicit root, Chromium (and therefore Electron/VS Code) reports a target's intersection with the *top-level* viewport, clipped by the ancestor frames and their scroll containers. Inside the webview, `getBoundingClientRect` and `innerHeight` cannot give this, because the webview spans the whole notebook. The observer watches the toggle button's element (`el`), which sits at the top of the viewer, and keeps `topVisible` up to date.
- **Rule:**
  an upward wheel event that nothing between its target and the root can scroll is `preventDefault()`-ed while `topVisible`. Downward events are never touched. So with the viewer's top above the editor, scrolling up brings it back; once the top is on screen, scrolling up stops. The overshoot is at most the one wheel step that brought the button into view.
- **Removed the latch** (`LATCH_MS`); the observer is created on entering `expanded` and disconnected in `teardown()`.
- **Risk:** if the observer does not clip by the editor's scroll area in some VS Code build, `topVisible` stays `true`, and scrolling *up* over the viewer is blocked while scrolling down still works. Leaving the mode clears it.

---

### **Files Changed**
- `ueler/viewer/fullscreen_toggle.py` — `observeTop` / `unobserveTop`, up-only `onWheel` gated on `topVisible`
- `tests/test_fullscreen_toggle.py` — fake `IntersectionObserver` driven by the test: up passes while the top is hidden and is cancelled once it is visible; down and inside-viewer scrolling pass; the observer is disconnected on unmount
- `doc/log.md`, `README.md`, `dev_note/topic_viewer_runtime_ui.md`

---

### **Tests**
```bash
python -m unittest tests.test_fullscreen_toggle
python -m unittest discover -s tests
```
- ✅ All tests passed (19 in the module; full suite 1413, OK)

---

## **Update Summary: confine the notebook scroll to the viewer (follow-up 6, uncommitted)**

User clarification: the notebook should not scroll past the viewer's top *or* its bottom. While grow-in-place is on, the notebook's scroller may only move within the range where the viewer covers the editor.

---

### **Actions Taken**
- **Bottom marker:**
  on entering `expanded`, a 1-px `.ueler-fs-bottom-mark` (`flex: 0 0 1px; margin-top: -1px`, so it takes no space) is appended to the root and observed with the button by the same `IntersectionObserver`. It is removed in `teardown()`. Both markers sit in the viewer's scrolled content rather than at its box edges. This is correct, because the rule applies only once the viewer's own scroll is exhausted in that direction, and at that point the content edge is the box edge.
- **Symmetric rule:**
  a wheel event that nothing between its target and the root can scroll is `preventDefault()`-ed when the edge in its direction is on screen: the top for up, the bottom for down.

### **Limits**
- Only wheel events that start over the viewer can be governed. Dragging VS Code's scrollbar, keyboard scrolling with focus outside the output, and wheel events over the editor's gutter never reach the webview.
- The expanded height is `min(innerHeight, screen.availHeight − 160)`, not the editor's exact height. If the viewer ends up shorter than the visible editor area, both edges are on screen together and the range is zero: the wheel over the viewer does not scroll the notebook at all, as specified. If it is taller, the range is the difference.

---

### **Files Changed**
- `ueler/viewer/fullscreen_toggle.py` — `observeEdges` / `unobserveEdges`, the bottom marker and its CSS, a symmetric `onWheel`
- `tests/test_fullscreen_toggle.py` — the VS Code scenario covers the bottom edge: down passes while the bottom is hidden and is cancelled once it is visible, and the marker is removed on unmount
- `doc/log.md`, `README.md`, `dev_note/topic_viewer_runtime_ui.md`

---

### **Tests**
```bash
python -m unittest tests.test_fullscreen_toggle
python -m unittest discover -s tests
```
- ✅ All tests passed (19 in the module; full suite 1413, OK)

---

## **Update Summary: exact anchoring by clamping each step (follow-up 7, uncommitted)**

User report on follow-up 6: "every time it stops at a different place". It should feel anchored.

The cause was in the rule itself. Follow-up 6 cancelled a wheel event only once the edge was *already* visible, so the step that brought the edge into view went through in full. With a trackpad that step can be anywhere from a few pixels to a few hundred, so the viewer stopped wherever that step happened to leave it.

---

### **Actions Taken**
- **Measure the distance, not just visibility:**
  the root itself is observed with an implicit-root `IntersectionObserver` with 201 thresholds, a callback every few pixels. `boundingClientRect` and `intersectionRect` (the latter mapped into the target frame for a cross-origin root) give `hidden.top` and `hidden.bottom`: how much of the viewer is cut off by the editor. This replaces the button and bottom-marker booleans; the marker element is gone.
- **Clamp, then forward:**
  a wheel event that nothing inside the viewer can scroll is always cancelled. A synthetic `WheelEvent` (pixel mode, `deltaY` clamped to the hidden distance in that direction) is dispatched on the root and bubbles to VS Code's `window` listener, which forwards it to the notebook. The synthetic event is tagged in a `WeakSet` so our own listener skips it. The notebook therefore moves exactly up to the edge. The hidden distances are updated locally after each forwarded step, so a burst of momentum events between two observer callbacks cannot overshoot. Line- and page-mode deltas are converted to pixels first.
- **Fit the viewer to the editor:**
  the expanded height now starts at `screen.availHeight`, which is never shorter than the editor. The first time both edges are cut off, the visible height *is* the editor's height, so the viewer shrinks to it (shrink only, so it cannot feed back), and the canvas refits through `measure()`. From then on the range is zero: once flush, the viewer cannot be scrolled off the editor's top or bottom by the wheel over it.
- **Unknown position means hands off:** until the observer has reported, or with no `IntersectionObserver`, wheel events are left alone.

### **Risk**
- This relies on VS Code forwarding *synthetic* (`isTrusted: false`) wheel events. If it ignores them, scrolling the notebook over the viewer stops entirely in this mode. Leaving the mode clears it, and the fix is to fall back to plain cancellation.

---

### **Files Changed**
- `ueler/viewer/fullscreen_toggle.py` — `onEdges` (distance and editor-height fit), clamped synthetic forwarding in `onWheel`, `setExpandedHeight`; bottom marker and its CSS removed
- `tests/test_fullscreen_toggle.py` — the VS Code scenario simulates a 900 px editor: steps pass while an edge is cut off, the viewer shrinks to 900 px once both are, a 250 px flick is clamped to the remaining 100 px and lands exactly flush (`y = 0`), and every later step, including line-mode deltas, moves 0; the start height is now the screen's 1080 px
- `doc/log.md`, `README.md`, `dev_note/topic_viewer_runtime_ui.md`

---

### **Tests**
```bash
python -m unittest tests.test_fullscreen_toggle
python -m unittest discover -s tests
```
- ✅ All tests passed (19 in the module; full suite 1413, OK)

---

## **Update Summary: snap to the anchor through VS Code's in-page link scrolling (follow-up 8, uncommitted)**

User report on follow-up 7: still not anchored, and it "sometimes even bounced to the opposite direction". The user's model: in notebook coordinates the viewer spans [A, B] and the editor shows [C, D]; keep C ≥ A and D ≤ B.

That model is right, and it shows what all four wheel-based attempts lacked. None of them could **set** the notebook's scroll position; they could only veto or rewrite wheel events and guess. The bounce is most likely VS Code's forwarding reading fields of the synthetic `WheelEvent` other than `deltaY` (Chromium derives `wheelDeltaY` with the opposite sign, and VS Code's `StandardWheelEvent` prefers it). Rather than reverse-engineer that, the wheel path is abandoned.

---

### **Actions Taken**
- **Set the position instead of steering the wheel:**
  VS Code's output preload handles clicks on in-page links (`<a href="#id">`) by asking the editor to scroll the notebook so that element sits at the top. The root gets an `id` while expanded (`ueler-anchor-…`, removed on exit if it was ours). `snap()` appends a hidden `<a href="#id">` to the button's element, `click()`s it and removes it. That sets C = A directly.
- **When to snap:**
  on entry, and whenever the viewer is visible but not aligned (`intersectionRatio` < 0.995) and visibility reports have been still for 150 ms. The user can scroll freely; the viewer snaps back when they stop. A viewer scrolled entirely off-screen (ratio 0, e.g. via the scrollbar or the outline) is left alone.
- **D = B by measurement:**
  with the viewer's height equal to the editor's, C = A also gives D = B. The report produced by a snap measures the editor: the top is aligned, so `ratio × height` is the editor's height. Only that one report measures (a `measureNext` flag), because a first version that measured for 600 ms after a snap shrank the viewer to whatever was visible after a quick follow-up scroll, which the tests caught. The viewer only ever shrinks.
- **Only `intersectionRatio` is used.** It is the value least likely to be distorted across the cross-origin webview boundary; follow-up 7's use of `intersectionRect` coordinates may have contributed to its odd stops.
- **Removed:** the wheel listener, synthetic `WheelEvent`s, the clamping and the inside-scroll detection. Scrolling inside the viewer is now never involved at all.
- **Content taller than the editor** (user point): the frame stays at the editor's height and the content scrolls inside it (`overflow: auto`). `.ueler-root.ueler-expanded > * { flex-shrink: 0 }` stops the root's flex column from squeezing its rows to fit, so the whole app scrolls as one page instead of nesting scrollbars. Inner scrolling moves only the content, not the frame, so it never triggers a snap.

### **Risk**
- This relies on VS Code honouring a scripted `click()` (`isTrusted: false`) on an in-page link. If it does not, the viewer is not anchored, but nothing else breaks: no wheel event is touched any more.

---

### **Files Changed**
- `ueler/viewer/fullscreen_toggle.py` — `snap`, `onAnchor`, `startAnchoring` / `stopAnchoring`; the wheel machinery removed
- `tests/test_fullscreen_toggle.py` — the VS Code scenario simulates an editor that scrolls a link target to its top: on entry the viewer snaps to `y = 0` and shrinks from 1080 to 900 px; after scrolling down or up it settles back to exactly `y = 0`; once aligned there are no further snaps; fully off-screen it is left alone; a scroll soon after a snap does not shrink it; the id is removed on exit
- `doc/log.md`, `README.md`, `dev_note/topic_viewer_runtime_ui.md`

---

### **Tests**
```bash
python -m unittest tests.test_fullscreen_toggle
python -m unittest discover -s tests
```
- ✅ All tests passed (19 in the module; full suite 1413, OK)

---

## **Update Summary: snap-to-anchor removed; it blanked the notebook (follow-up 9, uncommitted)**

User report on follow-up 8: on entering the mode, the markdown cells went blank and the app disappeared.

The scripted `click()` on `<a href="#id">` did not reach VS Code's in-page-link handling as hoped. The browser performed the fragment navigation itself, inside the output webview, and scrolled the webview's own document to the viewer. VS Code positions every output (rendered markdown included) at fixed offsets in that document and never expects it to scroll, so all outputs slid out of the slots VS Code draws them in. The webview stays broken until the window is reloaded.

That the webview document *could* scroll at all is a new fact. It sits uneasily with the runaway-height observation (follow-up 2), which suggested the webview is sized to the content. VS Code's webview geometry is not understood well enough here to build on.

---

### **Actions Taken**
- **Removed snapping entirely:** no `id` on the root, no link, no `IntersectionObserver`. A test now asserts that the ESM contains no `link.click`, `IntersectionObserver` or `WheelEvent`, and that entering expanded mode clicks nothing and leaves the root's `id` alone.
- **Height back to one estimate:** `min(innerHeight, screen.availHeight − 160)`, taken once on entry. The "measure the editor after a snap" step depended on snapping and is gone with it.
- **Kept:** in-flow growth, internal scrolling with `flex-shrink: 0`, and canvas fitting.

### **Where anchoring stands**
Four routes have now failed: boundary veto (froze the notebook), gesture latch (never engaged), visibility veto (uneven stops), clamped synthetic wheel events (bounced), and in-page link (blanked the outputs). Each was designed from assumptions about VS Code's webview preload that could not be checked here. Another attempt should start from facts gathered in VS Code's webview DevTools (**Developer: Open Webview Developer Tools**): the webview's `innerHeight` against the editor height, whether `document.scrollingElement` scrolls, and which wheel fields the preload forwards. It should not start from another guess.

---

### **Files Changed**
- `ueler/viewer/fullscreen_toggle.py` — anchoring removed; fixed one-screen height restored
- `tests/test_fullscreen_toggle.py` — snap scenario replaced by a "never clicks links" guard; expected height back to 920 px
- `doc/log.md`, `README.md`, `dev_note/topic_viewer_runtime_ui.md` — no longer claim anchoring

---

### **Tests**
```bash
python -m unittest tests.test_fullscreen_toggle
python -m unittest discover -s tests
```
- ✅ All tests passed (19 in the module; full suite 1413, OK)

---

## **Update Summary: read-only geometry probe (follow-up 10, uncommitted)**

Webview DevTools proved hard to use for the facts asked for in follow-up 9: the frame holding the outputs is difficult to select (the user found only `.editor-instance`). The widget now collects those facts itself.

---

### **Actions Taken**
- **Probe in expanded mode:**
  on entering `expanded`, the ESM samples every 250 ms for 15 s, or until the mode is left. Each sample records `innerHeight`, `screen.availHeight`, the webview document's `scrollHeight` and `scrollTop`, the root's `top` and `height` from `getBoundingClientRect`, and the root's `intersectionRatio` from an implicit-root `IntersectionObserver`. The samples are sent to Python **once**, as the `diagnostics` trait, when the probe ends or the mode is left. The probe only reads: a test asserts that the ESM has no wheel listener, no `WheelEvent`, no `link.click` and no `scrollTo(` / `scrollBy(` / `scrollIntoView(` / `scrollTop =`.
- **`diagnostics_report(toggle)`** formats the samples as a table and drops rows where nothing changed, so it shows what scrolling the notebook actually moved.

### **How to run it**
Reinstall (`pip install .` from the repo; the environment's copy is not editable), restart the kernel and re-run the viewer. Enter grow-in-place, scroll the notebook up and down for a few seconds, leave the mode, then run:

```python
from ueler.viewer.fullscreen_toggle import diagnostics_report
print(diagnostics_report(viewer.fullscreen_toggle))
```

### **What it decides**
- `rootTop` changes while the notebook scrolls → the webview can see the viewer's position directly.
- `rootTop` constant but `visibleRatio` changes → the position is visible only through the observer.
- `docScrollTop` ≠ 0 at any point → the webview document itself scrolls, which is what the link click disturbed.
- `innerHeight` against the visible editor height → whether the webview is sized to the editor or to the content.

---

### **Files Changed**
- `ueler/viewer/fullscreen_toggle.py` — `startProbe` in the ESM, `diagnostics` trait, `DIAGNOSTIC_COLUMNS`, `diagnostics_report`
- `tests/test_fullscreen_toggle.py` — 22 tests: the probe records the expected geometry and reports only on exit, the report skips unchanged rows, the read-only guard; the node harness now exits once it has printed, so a running probe cannot hold it open

---

### **Tests**
```bash
python -m unittest tests.test_fullscreen_toggle
python -m unittest discover -s tests
```
- ✅ All tests passed (22 in the module; full suite 1416, OK)

---

## **Probe results (2026-10-15): exact anchoring is not reachable from an output**

The user ran the follow-up 10 probe in VS Code (macOS, `screen.availHeight` 994) and scrolled the notebook over the expanded viewer for about 4 s. Distinct samples:

```text
  ms  innerHeight  screenAvail  docScrollHeight  docScrollTop  rootTop  rootHeight  visibleRatio
   5        12650          994            12650             0     5830         834            -1
 272        12659          994            12659             0     5830         834          0.92
1005        12659          994            12659             0     5830         834         0.789
1506        12659          994            12659             0     5830         834         0.488
2006        12659          994            12659             0     5830         834         0.202
3505        12659          994            12659             0     5830         834         0.951
4005        12659          994            12659             0     5830         834         0.859
```

### Findings
- **The output webview is as tall as the whole notebook** (`innerHeight` = `docScrollHeight` ≈ 12,659 px). This confirms the follow-up 2 diagnosis of the runaway height. It also confirms that the follow-up 1 assumption ("`innerHeight` is the editor height") was wrong.
- **The webview document never scrolls** (`docScrollTop` = 0 throughout). VS Code scrolls the notebook by moving the webview element in the workbench. The in-page link of follow-up 8 scrolled this document itself, which is exactly the state VS Code never expects; hence the blank outputs.
- **The viewer's position is invisible from inside** (`rootTop` = 5830 throughout). Nothing in the webview's own coordinates changes when the notebook scrolls.
- **Only the visible fraction crosses the boundary** (`visibleRatio` tracks scrolling, 0.20 to 0.95). An implicit-root `IntersectionObserver` reports clipping by the editor, but a ratio does not say which edge is cut.
- **The frame was taller than the editor:** the ratio never reached 1. The editor's visible height is about 0.951 × 834 ≈ 793 px, against the estimate `screen.availHeight − 160` = 834.

### Conclusion
An output can neither **set** nor **read** the notebook's scroll position. It can only observe how much of itself is visible and veto or pass wheel events. Exact anchoring needs one of the first two, so it is not reachable. Any wheel-based scheme overshoots by up to one wheel step, which is the "stops at a different place every time" behaviour the user rejected in follow-up 6. Anchoring work stops here unless VS Code gains an API for it.

### What the data does allow
Fitting the frame to the editor's real height: the largest `visibleRatio` seen while the frame is taller than the editor, times the frame height, approximates the editor height to within one observer threshold (1 % of the frame, about 8 px). Built in follow-up 11 below.

---

## **Update Summary: fit the frame to the editor; probe removed (follow-up 11, uncommitted)**

The user chose to fit the expanded frame to the editor's real height and remove the probe.

---

### **Actions Taken**
- **Measure only while the viewer covers the editor:**
  `visibleRatio × frameHeight` equals the editor's height only when both of the viewer's edges are cut off. With one edge on screen the visible part is smaller, and fitting to it would shrink the frame too far, for example right after entering with the viewer half off the bottom. "Both cut" is judged from `intersectionRect` against `boundingClientRect` (top and bottom each more than 1 px in); the height itself comes from `intersectionRatio`, the value the probe showed crossing the boundary. If the rects do not cross the boundary, no fit ever happens and the frame keeps its estimate, so it fails safe.
- **Apply on settle, shrink only:**
  the largest measured height is applied 400 ms after the visibility reports stop, through `measure()`, so the canvas refits once. The frame never grows, so the fit cannot feed back. Re-entering the mode resets to the estimate `min(innerHeight, screen.availHeight − 160)`.
- **Probe removed:** `startProbe`, the `diagnostics` trait, `DIAGNOSTIC_COLUMNS` and `diagnostics_report` are gone; the results are kept above.

### **Limit**
If the editor is *taller* than the estimate (a large screen with little VS Code chrome), the viewer can never cover it, nothing is measured, and the frame stays at the estimate, a little short of the editor.

---

### **Files Changed**
- `ueler/viewer/fullscreen_toggle.py` — `startFit` / `onFit` / `applyFit` / `stopFit`; probe and diagnostics removed
- `tests/test_fullscreen_toggle.py` — 20 tests: in a simulated 790 px editor, a partial view leaves the 920 px frame alone; covering the editor shrinks it to 790 px and refits the canvas (`viewport` [1414, 790]); lining up afterwards keeps 790; re-entering resets to 920
- `doc/log.md`, `README.md`, `dev_note/topic_viewer_runtime_ui.md`

---

### **Tests**
```bash
python -m unittest tests.test_fullscreen_toggle
python -m unittest discover -s tests
```
- ✅ All tests passed (20 in the module; full suite 1414, OK)

---

## **Update Summary: square image filling the width by default; frame fit removed (follow-up 12, uncommitted)**

The frame fit of follow-up 11 "does not do much" (user): at most about 40 px, and unverified whether it triggered at all. It is removed; anchoring and the editor-height fit are closed (see "Probe results" above).

The user then pointed out a side effect of the enlarged-mode canvas fit: the image filled the width, so the side column landed at the right edge. They asked for that as the **default layout**, with the image kept **square**.

---

### **Actions Taken**
- **Width reported in every mode:**
  the ESM watches the root with a `ResizeObserver` and reports `host_width` (debounced 150 ms; changes under 2 px and widths under 200 px, hidden or still laying out, are ignored). `render` can run before the button is attached, so the watcher retries `closest()` for up to 60 frames. Width is safe to follow where height was not: the probe showed the webview width is the editor's (1414 px), and the root is 100 % of the output's width, which does not depend on the canvas, so refitting cannot feed back.
- **Square default:**
  `fit_square_inches` takes the width less the 350 px left panel, the 576 px side column when present and 64 px of chrome, and returns that as both sides (minimum 3 in). `attach_canvas_fit` now recomputes one target from all three traits: the `viewport` fill while enlarged, otherwise the square from `host_width`, otherwise the figure's own size. It skips changes under 0.02 in. Leaving full screen therefore returns to the square, not to 6 × 6 in.
- **Enlarged modes still fill the screen** rather than staying square: limited by height, a square would leave a gap on the right, which is what this change set out to remove.

---

### **Files Changed**
- `ueler/viewer/fullscreen_toggle.py` — `host_width` trait, `watchWidth` / `reportWidth` in the ESM, `fit_square_inches`, `attach_canvas_fit` rewritten around one target; frame fit removed
- `ueler/viewer/ui_components.py` — comment at the `attach_canvas_fit` call
- `tests/test_fullscreen_toggle.py` — 24 tests: square math, default square follows the width, a 1 px change does not refit, leaving full screen returns to the square, width reported under `node` with no mode entered and the observer disconnected on unmount; frame-fit scenario removed
- `doc/log.md`, `README.md`, `dev_note/topic_viewer_runtime_ui.md`

---

### **Tests**
```bash
python -m unittest tests.test_fullscreen_toggle
python -m unittest discover -s tests
```
- ✅ All tests passed (24 in the module; full suite 1418, OK)

---

## **Update Summary: VS Code notice before grow-in-place (follow-up 13)**

User request: in VS Code, clicking the button should say that full screen is not supported there and that it needs JupyterLab or JupyterHub.

---

### **Actions Taken**
- **Notice instead of an attempt:**
  in a VS Code webview the click no longer calls `requestFullscreen()` (always refused there). It opens a notice under the button: "Full screen is not supported in VS Code", explaining that VS Code does not let notebook outputs go full screen and to open the notebook in JupyterLab or JupyterHub if needed. **Enlarge in place instead** enters the existing `expanded` mode, so grow-in-place stays reachable; **Close**, Esc and the button itself dismiss the notice. Focus moves to **Close**, and the notice carries `role="dialog"`.
- **Inline, not a modal:**
  `ConfirmDialog`-style `position: fixed` centring is wrong in VS Code. The probe showed the output webview is as tall as the whole notebook, so "the centre of the viewport" can be thousands of pixels off screen, and it would be drawn under the code cells. The notice is appended to the button's own element, in the page flow, where the click happened.
- **Other environments unchanged:** JupyterLab, Notebook and any other page still go straight to native full screen or the overlay.

---

### **Files Changed**
- `ueler/viewer/fullscreen_toggle.py` — `showNotice` / `closeNotice` / `noticeButton`, click routing, Esc, unmount; notice CSS using JupyterLab and VS Code theme variables
- `tests/test_fullscreen_toggle.py` — 25 tests: the click shows the notice without attempting full screen; its text names VS Code, JupyterLab and JupyterHub; Esc, Close and the button dismiss it; unmount removes it; the VS Code scenarios reach grow-in-place through **Enlarge in place instead**
- `doc/log.md`, `README.md`, `dev_note/topic_viewer_runtime_ui.md`

---

### **Tests**
```bash
python -m unittest tests.test_fullscreen_toggle
python -m unittest discover -s tests
```
- ✅ All tests passed (25 in the module; full suite 1419, OK)
