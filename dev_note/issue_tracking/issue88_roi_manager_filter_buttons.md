# Issue #88: ROI manager advanced tag-filter helper buttons

## Problem
The helper buttons in the ROI manager's advanced tag-filter UI do not reliably insert operators or tag tokens into the expression field when clicked in notebook front-ends such as VS Code.

## Root cause
- `_insert_browser_expression_snippet(...)` treated the JS helper path as authoritative.
- `_insert_browser_expression_snippet_js(...)` returned success after emitting JavaScript to a hidden output widget, but that did not guarantee that the browser-side field was actually found or mutated.
- When the JS path reported success without a real field update, the already-correct backend insertion path was skipped, making button clicks appear unresponsive.

## Chosen approach
- Make backend insertion into `browser_expression_input.value` authoritative.
- Keep browser-side JavaScript strictly best-effort for focus/selection synchronization only.
- Add focused regression coverage for operator-button clicks, dynamic tag-button clicks, and the JS-success-without-mutation case.

## Implementation steps
1. Add focused tests in `tests/test_roi_manager_tags.py` covering operator and tag helper buttons.
2. Refactor `ROIManagerPlugin._insert_browser_expression_snippet(...)` so it always performs backend insertion.
3. Keep `_insert_browser_expression_snippet_js(...)` as a best-effort DOM sync after backend insertion rather than as the source of truth for the text mutation.
4. Normalize helper-button construction through one local helper so operator and tag buttons share the same callback wiring.
5. Validate with the full ROI manager tag test module.

## Validation
- `python -m unittest tests.test_roi_manager_tags -v`

## Risks
- Browser-side caret placement may still vary slightly across front-ends because the DOM sync remains best-effort.
- The functional path is still safe because Python now owns the text mutation even if the DOM sync does nothing.

## Follow-up: live caret synchronization

### Problem
After the initial button-responsiveness fix, helper-button clicks could still insert at a stale caret position. Python owned the expression text mutation, but `_browser_expression_selection` only reflected backend-side updates and did not change when the user simply clicked into a different location in the browser field.

### Chosen approach
- Keep Python authoritative for expression text mutation.
- Add a minimal browser-to-Python caret bridge that updates the existing `_browser_expression_selection` cache.
- Use a hidden widget state channel for the browser to push `selectionStart` / `selectionEnd` back into the plugin.

### Follow-up implementation
1. Added a hidden `browser_expression_selection_state` widget to carry caret updates from the browser into Python.
2. Added `_install_browser_expression_selection_bridge()` to attach browser listeners (`focus`, `click`, `keyup`, `mouseup`, `select`, `input`) to the advanced expression field and mirror the live selection into that hidden widget.
3. Added `_on_browser_expression_selection_state_change()` to parse the pushed selection payload and refresh `_browser_expression_selection`.
4. Added regression coverage for direct insertion and helper-button click paths when the live tracked caret differs from the stale cached selection.

### Follow-up validation
- `python -m unittest tests.test_roi_manager_tags -v`

## Follow-up 2: stale selection after manual typing

### Problem
Even after the helper buttons became responsive and live-caret sync was added, a stale insertion point could still survive across manual text edits. Typical failure: helper clicks build a prefix, the user types more text, and the next helper click inserts back at the older prefix position instead of after the newly typed text.

### Root cause
- `_browser_expression_selection` survived across expression-text revisions.
- `_on_browser_expression_change()` only reset the cached selection when it was `None`, so manual typing after a helper insertion preserved the older insertion point.
- The live-caret bridge improved precision when fresh selection events arrived, but it was not a sufficient safety net when the cached selection belonged to an older expression value.

### Chosen approach
- Track which expression string the cached selection belongs to.
- Collapse stale cached selections to the end of the new expression text when the text changes and no fresher selection is associated with that new text revision.
- Keep the live-caret bridge in place so it can still override the fallback with a precise mid-string caret when available.

### Follow-up 2 implementation
1. Added `_browser_expression_selection_text` to record the expression value associated with the current cached selection.
2. `_insert_browser_expression_snippet_backend()` now updates both the cached selection and the associated expression text.
3. `_on_browser_expression_selection_state_change()` now refreshes the cached selection text alongside the observed caret range.
4. `_on_browser_expression_change()` now collapses the selection to the end of the new text whenever the cached selection belongs to an older expression revision.
5. Added a focused regression covering the exact reported sequence: helper insertion, manual typing, then another helper insertion.

### Follow-up 2 validation
- `python -m unittest tests.test_roi_manager_tags -v`

## Follow-up 3: caret reposition before helper click

### Problem
After the stale-selection-after-typing fix, helper-button insertion still failed when the user only moved the caret inside the existing expression without changing the text. The next operator or tag click still used the previous tail selection because the background selection bridge did not reliably update Python before the button callback ran.

### Root cause
- Caret-only moves depended on asynchronous background bridge events from the expression field.
- Those updates could arrive too late relative to the helper-button click callback.
- As a result, `_insert_browser_expression_snippet_backend()` still read the older cached selection even though the user had visibly moved the caret in the browser.

### Chosen approach
- Keep the background selection bridge for general freshness.
- Add an explicit pre-click selection flush on helper buttons so the live DOM selection is pushed into Python on `pointerdown` / `mousedown`, before the Python click callback runs.
- Preserve backend-owned expression mutation and formatting rules.

### Follow-up 3 implementation
1. Added a shared CSS class for ROI expression helper buttons.
2. Extended `_install_browser_expression_selection_bridge()` to attach `pointerdown` / `mousedown` listeners to helper buttons and flush the live DOM selection into the hidden selection-state widget before click handling.
3. Reinstalled the bridge after regenerating dynamic tag buttons so newly rendered buttons also receive the pre-click listeners.
4. Added a pre-click hook on helper buttons for focused regression coverage in headless tests.

### Follow-up 3 validation
- `python -m unittest tests.test_roi_manager_tags -v`

## Follow-up 4: JS-only expression editing + Apply button

### Root cause (definitive)
All three previous follow-ups attempted to push caret state from JS to Python before a helper-button click. This relies on the Jupyter widget comm bridge, whose message delivery order is not guaranteed in VS Code. No amount of tuning the existing design can eliminate this race condition because the fundamental constraint — synchronous caret data arriving in Python before an async click callback — cannot be guaranteed across all Jupyter front-ends.

### Chosen approach
Eliminate the Python caret dependency entirely:
- Helper buttons emit a **self-contained JS snippet** per click (via the existing `browser_expression_js_output` Output widget). The snippet reads `field.selectionStart`/`End` from the live DOM, applies the same spacing rules as `_format_expression_insertion()` (replicated in JS), sets `field.value` and `field.setSelectionRange()`, then dispatches `input`/`change` events so ipywidgets auto-syncs the new value to Python.
- A new **"Apply" button** (`browser_expression_apply_button`) reads `browser_expression_input.value` (auto-synced by ipywidgets) and calls `_apply_browser_expression()`, which compiles and refreshes the gallery.
- No hidden state widget, no caret cache, no real-time gallery refresh on typing.

### Follow-up 4 implementation
1. Rewrote `_insert_browser_expression_snippet()` and `_insert_browser_expression_snippet_js()` in `roi_manager_plugin.py`.
2. Removed: `browser_expression_selection_state` widget, `_install_browser_expression_selection_bridge()`, `_on_browser_expression_selection_state_change()`, `_resolve_browser_expression_selection()`, `_flush_browser_expression_selection_before_click()`, `_insert_browser_expression_snippet_backend()`, `_on_browser_expression_change()`, and the `_browser_expression_selection`/`_browser_expression_selection_text` caches.
3. Added: `browser_expression_apply_button`, `_on_apply_expression_click()`, `_apply_browser_expression()`, `_insert_browser_expression_snippet_test()` (test-mode tail-insert fallback).
4. Updated `_connect_events()` to remove old observers and add the Apply button handler.
5. Updated `_on_browser_filter_tab_change()` to compile and show feedback on tab switch instead of reinstalling the bridge.
6. Updated `tests/test_roi_manager_tags.py`: removed five caret-tracking tests, updated five insertion tests for tail-append test-mode behaviour, added five Apply button tests.

### Follow-up 4 validation
- `python -m unittest tests.test_roi_manager_tags -v` — 39/39 passed
## Follow-up 5 Option B: anywidget self-contained expression editor

### Root cause (definitive for Option A failure)
Follow-up 5 Option A installed the insertion logic once via a JS bridge. Even this approach remained unreliable in VS Code Jupyter because VS Code's Jupyter front-end does not guarantee the bridge install message is processed before the user clicks a helper button — or that the document-level delegated listener fires in the expected order relative to the anywidget DOM. The fundamental constraint — JS listening for clicks in foreign widget DOMs via Python comm — cannot be reliably satisfied.

### Chosen approach (Option B)
Render the entire expression editor UI (text input, Apply button, operator buttons, tag buttons) inside a single `anywidget.AnyWidget` widget. Because all buttons live in the same `el` as the `<input>`, `mousedown` + `preventDefault()` on buttons within the same shadow context guarantees focus cannot leave the input before `click` fires. `selectionStart`/`End` are therefore always accurate. No Python comm round-trip is needed for insertion — the JS handler reads the caret, applies spacing rules, updates `input.value`, and pushes `model.set("expression", ...)` + `model.save_changes()`. Apply click increments `apply_requested`; Python observes it.

### Option B implementation
1. Created `ueler/viewer/plugin/roi_expression_editor.py`:
   - `ROIExpressionEditorWidget(anywidget.AnyWidget)` with traitlets `expression`, `tags`, `apply_requested` (all `tag(sync=True)`).
   - `_esm` ESM module: `render({ model, el })` builds the DOM, wires `mousedown` + `click` on all buttons, `input` event on the text field, `model.on("change:tags", ...)` to rebuild tag row, `model.on("change:expression", ...)` to sync Python→DOM.
   - `_css` string for widget styling.
   - `HasTraits` fallback (no `anywidget`, same traitlets without `tag(sync=True)`) for test environments.
2. Updated `ueler/viewer/plugin/roi_manager_plugin.py`:
   - Replaced five expression filter widgets with `browser_expression_editor = ROIExpressionEditorWidget()`.
   - Updated `advanced_filter_box` to `[browser_expression_editor, browser_expression_feedback]`.
   - Updated `_connect_events()`: replaced `apply_button.on_click` with `browser_expression_editor.observe(_on_apply_requested_change, names="apply_requested")`.
   - Added `_on_apply_requested_change(change)`.
   - Updated `_on_apply_expression_click` (kept as alias for test compatibility).
   - Updated `_refresh_expression_tag_buttons` to `editor.tags = list(tags)`.
   - Updated all expression-text reading to `browser_expression_editor.expression`.
   - Updated `_on_browser_filter_tab_change` to use `editor.expression`.
   - Removed: `_make_expression_insert_button`, `_insert_browser_expression_snippet*`, `_use_browser_expression_js`, `_browser_operator_buttons`, `json` import, `IPythonJS` import.
3. Updated `tests/test_roi_manager_tags.py`:
   - Removed 6 obsolete insertion tests.
   - Updated 4 Apply button tests to use `browser_expression_editor.expression`.
   - Added `test_refresh_tag_buttons_updates_editor_tags`.
   - Repurposed `test_expression_insertion_at_end_of_expression` to test `_format_expression_insertion` directly.
   - Removed `_use_browser_expression_js = False` from both `make_plugin()` helpers.

### Option B validation
- `python -m unittest tests.test_roi_manager_tags -v`
- ✅ All 34 tests passed
