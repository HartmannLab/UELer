# Issue #90: Mask painter visibility lost on redraw

## Problem
When the mask painter plugin is enabled, cell masks stop behaving correctly during normal viewer redraws. Users can set a class to shown in the mask painter UI, but after zooming or panning the FOV the masks disappear and do not come back. The current behavior also conflicts with the issue requirements in two other ways: the painter currently starts enabled, and inactive classes are treated like hidden classes rather than remaining visible with the default color.

## Scope
- Single-FOV viewer rendering path while the mask painter plugin is enabled
- Mask painter enable/disable lifecycle
- Active vs inactive class visibility semantics
- Focused regression coverage for redraw persistence and default state
- Required documentation updates for the implementation record

## Root cause hypothesis
The main viewer applies the mask painter overlay during `_compose_fov_image()`, but plugin callbacks such as `on_mv_update_display()` run only after the frame has already been pushed into the canvas. In single-FOV mode the painter mostly updates the live image display and only writes the shared registry in selected cases, so a redraw can occur before the painter state has been materialized in the form consumed by `_compose_fov_image()`. As a result, zoom/pan redraws can skip the intended shown/hidden state even though the UI still reflects it.

## Required behavior
- The mask painter is disabled by default.
- When the painter is enabled, all active classes are shown by default.
- Active classes can be hidden explicitly through the visibility checkbox.
- Inactive classes are not hidden; they remain visible with the default color.
- A class-specific custom color must be remembered when that class becomes inactive and restored if it is re-activated.
- Zoom and pan redraws must preserve the current shown/hidden state without requiring another manual "Update Colors" click.

## Approach
Use the current mask painter UI state directly in the single-FOV render path instead of relying on a post-render plugin callback or on the shared registry timing.

1. Add focused issue #90 tests that prove the redraw failure and the required class semantics.
2. Add a helper in `ueler/viewer/plugin/mask_painter.py` that computes the effective current-FOV painter state:
   - active visible classes -> chosen class color
   - active hidden classes -> hidden sentinel `""`
   - inactive classes -> default color
   - per-cell mode map for active classes
3. Update `ueler/viewer/main_viewer.py` so `_compose_fov_image()` passes that explicit current-FOV color map and mode map into `apply_registry_colors(...)` when the painter is enabled.
4. Keep the shared registry for gallery/export/map-mode consumers, but remove the single-FOV correctness dependency on post-render synchronization.
5. Add an explicit enable/disable observer so toggling the painter triggers the minimal state sync and redraw.
6. Change hidden-class computation so only active unchecked classes are hidden.

## Implementation steps
1. Add issue #90 regression tests in `tests/test_mask_painter_mode_visibility.py`.
2. Change `UiComponent.enabled_checkbox` in `ueler/viewer/plugin/mask_painter.py` to default to `False`.
3. Add helper methods in `ueler/viewer/plugin/mask_painter.py` to build the current-FOV color/mode state from the UI.
4. Update `_compose_fov_image()` in `ueler/viewer/main_viewer.py` to consume that helper output via the existing `color_map` and `mode_map` arguments to `apply_registry_colors()`.
5. Add enable/disable handling in `ueler/viewer/plugin/mask_painter.py` so the viewer redraws immediately when the painter state changes.
6. Adjust inactive/hidden semantics and preserve saved class colors across activation changes.
7. Run the focused mask painter test suite and perform a manual notebook validation.

## Validation
- Automated:
  - `python -m unittest tests.test_mask_painter_mode_visibility tests.test_mask_color_overlay tests.test_mask_color_sets`
- Manual:
  1. Launch `script/run_ueler.ipynb`.
  2. Confirm masks are visible before enabling the painter.
  3. Enable the painter and confirm active classes remain shown.
  4. Hide one active class, zoom and pan, and confirm the hidden/shown state persists.
  5. Remove a class from the active list and confirm it still displays with the default color.

## Risks
- If current-FOV direct rendering and cross-context registry synchronization diverge, painter state logic could fork. Mitigation: centralize both outputs behind one state-builder in `mask_painter.py`.
- Changing inactive-class semantics can affect tests added for issue #89. Mitigation: update those tests only where they encoded behavior that issue #90 explicitly rejects.