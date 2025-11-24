# Map Mode Kernel Crash (Issue #58) — Implementation Plan

## Problem Summary
- Reproducing the bug shows the viewer kernel dies immediately after enabling map mode and selecting any map descriptor.
- When `_activate_map_mode` calls `_set_map_canvas_dimensions`, the helper constructs a blank `np.zeros((height_px, width_px, 3), dtype=np.float32)` array to reset the artist.
- Large stitched maps (e.g., slide `402` with 171 tiles) report canvas sizes around `36 706 × 115 751` pixels, so the blank allocation exceeds 50 GB and the Python process is OOM-killed before any warning can surface.
- Even when the blank allocation succeeds, the first `update_display` render still uses the previous FOV-derived downsample factor, which can be orders of magnitude too small for slide-sized canvases and risks further memory spikes.

## Proposed Fix
- Teach `_set_map_canvas_dimensions` to reuse a constant 1×1 placeholder when retargeting the Matplotlib artist so map activation never attempts to allocate a canvas with the full stitched resolution.
- When a map is activated, recompute `current_downsample_factor` with `select_downsample_factor` against the stitched canvas size so the first render starts from the coarsest allowed scale that keeps the longest edge ≤512 px.
- Add regression coverage that simulates map activation with a very large descriptor and asserts we only allocate the placeholder canvas and that the downsample factor snaps to the expected coarse value.

## Implementation Steps
1. Update `_set_map_canvas_dimensions` in `ueler/viewer/main_viewer.py` to:
   - Guard against missing `image_display` early.
   - Avoid creating a full-resolution blank; instead set the extent with a cached 1×1 float32 placeholder and keep `display.combined` pointing at that placeholder.
2. Inside `_activate_map_mode`, after computing `width_px`/`height_px`, call `select_downsample_factor` with `self.downsample_factors` (minimum 1) and assign the result to `self.current_downsample_factor` before triggering the first render.
3. Extend the test suite (new case in `tests/test_virtual_map_layer.py` or a dedicated test module) to patch `np.zeros` during `_set_map_canvas_dimensions`, confirm only the placeholder shape `(1, 1, 3)` is requested for a huge map, and assert the downsample factor jumps to the coarse expected value.
4. Run `python -m unittest tests.test_virtual_map_layer` (plus the new test module if separate) to verify the regression guard and ensure no collateral failures.
5. Document the fix in `dev_note/github_issue.md` (issue #58 entry), `doc/log.md`, and update the README “New Update” section accordingly after implementation and tests pass.

### Follow-up — 2025-11-24
- Addressed a post-patch regression where map activation triggered `ValueError: Viewport must have positive width and height`. `image_utils.get_axis_limits_with_padding` now aligns bounds using floor/ceil arithmetic and enforces at least one downsampled cell per axis, preventing zero-width viewports when the active downsample factor exceeds the visible span.
- Added `tests/test_map_mode_activation.py::test_update_display_preserves_positive_viewport` to assert that map-mode renders always receive strictly positive viewport dimensions, even at coarse downsample factors.
- Verified coverage with `python -m unittest tests.test_map_mode_activation`.
- Resolved the zoom-triggered `ValueError: operands could not be broadcast together` by recalculating downsampled tile bounds in `VirtualMapLayer._compute_tile_region` with ceil-based dimensions, ensuring viewport sampling stays aligned for non-divisible tile intersections. Guarded with `tests/test_virtual_map_layer.py::test_render_handles_partial_downsample_tiles` and revalidated via `python -m unittest tests.test_virtual_map_layer`.
