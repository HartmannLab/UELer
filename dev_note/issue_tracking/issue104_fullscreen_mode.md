# Issue #104 — Full-Screen Mode for the Viewer

**GitHub:** https://github.com/HartmannLab/UELer/issues/104

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
