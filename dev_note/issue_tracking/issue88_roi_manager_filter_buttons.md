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