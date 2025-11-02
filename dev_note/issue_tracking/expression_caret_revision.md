# ROI Expression Caret Revision Plan

## Context
- The ROI browser expression builder still inserts helper symbols at incorrect positions when the text box loses or regains focus.
- Existing caret caching and blur guards have reduced but not eliminated the misalignment, indicating that the insertion routine does not honor the most recent selection state.

## Goals
- Always insert helper symbols at the caret index that was active when the user clicked a helper button.
- Preserve selection ranges so replacements work when the user highlights text rather than inserting at a single caret position.
- Keep the expression field focused after insertion to support chained edits.

## Constraints & Considerations
- The widget stack spans Python (traitlets) and JavaScript (front-end caret telemetry); updating either side risks triggering unexpected `value` syncs.
- Notebook environments differ (classic Notebook, JupyterLab, Voila); selectors and focus logic must remain compatible across hosts.
- Regression tests currently stub focus events; new logic should be testable without browser automation.

## Action Plan
1. **Audit caret telemetry contract**
   - Confirm which widget DOM events update the stored selection snapshot and ensure blur/ focus guards do not drop legitimate updates.
   - Log the raw `start` and `end` indices received in `_on_browser_expression_cursor_change` for manual runs to verify accuracy.
2. **Read current caret index on insertion**
   - Before mutating the expression string, read the cached `start`/`end` indices; fall back to the string tail when no selection is known.
   - Handle selection ranges (`start != end`) by replacing the highlighted segment with the helper symbol payload.
3. **Apply insertion at computed slice**
   - Split the existing expression into prefix (`expression[:start]`) and suffix (`expression[end:]`), insert the new symbol between them, and rebuild the string.
   - Update the cached caret to land immediately after the inserted snippet so repeated helper clicks chain correctly.
4. **Sync updated value to the field**
   - Push the rebuilt expression back to the widget value without triggering recursive resets (respect `_browser_expression_skip_reset`).
   - Issue a post-update message to reposition the caret on the front end using the new `start`/`end`.
5. **Extend regression coverage**
   - Add unit tests that simulate cached selection indices, helper insertions, and selection ranges to ensure the computed value matches expectations.
   - Verify the caret cache updates to the new index in the tests to mirror real user flows.
6. **Manual verification**
   - Run the viewer notebook, use the helper buttons after moving the caret to multiple positions, and confirm insertions respect the index in different browsers.
7. **Documentation & rollout**
   - Update `dev_note/github_issues.md`, `doc/log.md`, and `README.md` once the implementation lands to describe the revised insertion logic and coverage.

## Acceptance Criteria
- Helper buttons always insert symbols at the cached caret position, including when text is selected.
- Caret position after insertion advances to the end of the inserted snippet.
- Unit tests cover insertion at the head, middle, tail, and with highlighted ranges.
- Manual notebook testing confirms behavior across supported environments.

## Status
- ✅ Implemented index-based insertion in `_insert_browser_expression_snippet`, updating the cached start/end after every helper click.
- ✅ Added regression tests in `tests/test_roi_manager_tags.py` covering head/middle/tail insertions and highlighted replacements.
- ✅ Ran `python -m unittest tests.test_roi_manager_tags` to confirm the new coverage passes.
- ✅ Restored the selection resolver and focus-aware caching to handle blur-driven telemetry drops without breaking helper insertions.
- ✅ Shifted helper insertion to the browser so snippet events mutate the value client-side before syncing back to Python, eliminating caret drift races.
- ✅ Added a readiness fallback so helper buttons reuse the Python insertion path until the browser bridge reports a live selection snapshot, keeping the workflow responsive immediately after loading.
- ✅ Delivered the caret bridge via `IPython.display.Javascript` instead of inline `<script>` tags so sanitized notebook outputs still execute the binding logic.
- ✅ Simplified the browser script to the ipywidgets DOM-binding pattern so it locates the Text input via stable selectors, runs through a shared `ipywidgets.Output`, performs the splice client-side, and keeps the cached selection aligned for Python fallbacks across Notebook, Voila, and JupyterLab 4.
