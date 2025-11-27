# Issue #61 — Cell Mask Tooltip Coverage

## Problem Statement
When hovering masked cells in the main viewer we currently filter the cell table using hard-coded column names (`fov` and `label`). Datasets that rename these identifiers (for example `FOV_ID`, `CellID`, or per-mask keys) therefore yield no matching rows, and the tooltip falls back to the default "mask ID only" text. The bug breaks user expectations that mean channel expressions and any fields picked in "Cell tooltip label" appear for every hover.

## Proposed Approach
1. Respect the viewer’s configurable keys.
   * `ImageMaskViewer` already exposes `fov_key`, `label_key`, and (optionally) `mask_key`. Tooltips should consult those attributes instead of assuming the legacy column names.
   * If a mask column exists, match both the ID and the mask name so multi-mask datasets stay disambiguated.
2. Centralise tooltip cell lookup.
   * Refactor the lookup into a dedicated helper so we can unit-test custom key handling without instantiating the full viewer.
   * Cache lookups by `(fov, mask_name, mask_id)` so repeated hover events remain fast.
3. Re-use the helper when composing tooltip text, gracefully skipping missing columns (channel or custom labels) but never failing the hover.

## Implementation Steps
1. Introduce a helper in `ueler/viewer/tooltip_utils.py` (e.g. `resolve_cell_record`) that accepts the cell table plus key metadata and returns the matching row or `None`. Incorporate mask matching when a mask column is configured and stabilise cache keys around `(fov, mask_name, mask_id)`.
2. Update `ImageDisplay.on_mouse_move` to leverage the helper, swap the hard-coded constants for viewer attributes, and adjust tooltip construction to iterate visible channels and selected labels only when the row is present.
3. Extend `tests/test_image_display_tooltip.py` with parametrised coverage that exercises different key names (default `fov`/`label`, custom `FOV_ID`/`CellID`, with and without a mask column) to ensure the helper and on-mouse-move path return the expected values.
4. Verify the existing tooltip formatting tests still pass, run the full fast test suite, and then document the change in `doc/log.md` and `README.md` after implementation.

## Validation Plan
- Unit tests for the new helper cover default and custom keys plus missing-row scenarios.
- A focused integration-style test confirms tooltip text includes channel means and selected labels when the row is found.
- Manual smoke test: run the viewer notebook against a dataset with renamed cell table columns and confirm hover tooltips list the expected details.

## 2025-11-10 Update
- Implemented the `resolve_cell_record` helper in `ueler/viewer/tooltip_utils.py` and refactored `ImageDisplay` hover handling to use viewer-configured keys, covering optional mask columns without regressing legacy datasets.
- Added lightweight `SimpleCellTable` fixtures and integration coverage in `tests/test_image_display_tooltip.py`, validating default/custom key lookups and caching behaviour; ran `python -m unittest tests.test_image_display_tooltip` to confirm all tooltip cases pass.
- Updated `doc/log.md` and the README “New Update” section with the tooltip fix summary; remaining work is optional manual smoke testing with real datasets.
