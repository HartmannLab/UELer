﻿# Issue #59 — Map canvas renders black after switching maps

## Summary
- **Observed behaviour:** Switching from the initial map to any other map leaves the stitched canvas black even though the canvas dimensions and zoom interactions work. Switching back to the first map restores normal rendering.
- **Suspected root cause:** The stitched renderer receives viewport coordinates that implicitly start at `(0, 0)` for every map. Descriptors whose bounds do not originate at `(min_x = 0, min_y = 0)` therefore produce intersection windows that fall outside each tile, resulting in empty renders.

## Investigation Checklist
1. Confirm descriptor bounds for the failing maps and capture their minimum X/Y offsets.
2. Instrument `ImageMaskViewer._render_map_view` and `VirtualMapLayer.set_viewport` to log viewport coordinates when switching between maps.
3. Reproduce the failure by simulating map activation using a descriptor whose bounds start at non-zero coordinates.
4. Inspect `VirtualMapLayer._collect_visible_tiles` to verify it rejects tiles whenever viewport coordinates are shifted.

## Proposed Fix
- Offset the viewport passed to `VirtualMapLayer.set_viewport` by the descriptor’s minimum `(x, y)` bounds so viewport comparisons operate in the same coordinate frame as the tiles.
- Add a regression unit test that activates map mode with a descriptor offset from the origin and asserts the stitched canvas is non-empty.

## Validation Plan
- Unit: extend `tests/test_virtual_map_layer.py` with an offset viewport scenario.
- Integration: run an interactive smoke test switching between multiple descriptors to confirm tiles render correctly and contrast/zoom continue to function.
- Documentation: update `dev_note/github_issues.md` entry for issue #59 with a link to this plan and record changes in `doc/log.md`.

## 2025-11-25 Follow-up — reset-to-original view regression
- **Issue:** Clicking Matplotlib's reset button after entering map mode restored the original square FOV viewport, wiping the stitched slide and reintroducing the black canvas/zoom errors.
- **Root cause:** The toolbar's navigation stack kept its home view from the initial single-FOV canvas because map activation never refreshed the stored axis limits.
- **Resolution:** Added `_sync_navigation_home_view()` to update (or seed) the toolbar's home entry whenever `_set_map_canvas_dimensions()` adjusts the stitched canvas, keeping reset aligned with the active slide.
- **Validation:** New regression tests in `tests/test_map_mode_activation.py` assert that existing navigation entries retain metadata while picking up the stitched bounds and that empty stacks are seeded with the correct limits.