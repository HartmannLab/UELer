# Issue #91: Mask painter opacity and borders on filled masks

## Problem
The mask painter can currently switch a class between outline and fill, but filled masks always render with a fixed alpha and cannot also draw a border. That leaves three gaps in the current behavior:

- adjacent cells of the same class blend together when fill mode is used
- users cannot make filled masks fully opaque or fully transparent
- downstream consumers such as the cell gallery, batch export, and ROI playback do not have a state model that can preserve richer per-class painter settings

## Scope
- Mask painter UI and saved palette format
- Single-FOV live viewer rendering path
- Cell gallery, batch export, and ROI playback snapshot propagation
- ROI persistence for captured painter state
- Focused regression coverage and required documentation updates

## Root cause hypothesis
The live single-FOV viewer already uses direct painter-derived color and mode maps, but opacity is still hardcoded inside `apply_registry_colors(...)` in `ueler/viewer/mask_color_overlay.py`. Separately, export and ROI/gallery consumers rebuild overlays from `OverlaySnapshot`, whose current mask snapshot model only represents uniform per-mask `alpha` and `mode`, and the current snapshot capture path hardcodes `alpha=1.0` and `mode="outline"`.

## Required behavior
- Each class can set fill opacity from 0% to 100%.
- Filled masks can optionally render a border on top of the fill.
- A global opacity control can update classes that are still linked to the previous global value.
- The main viewer, cell gallery, exports, and ROI thumbnails must agree on the same painter settings.
- ROI thumbnails should preserve the painter settings captured with the ROI instead of always resolving the latest live state.

## Approach
Extend the existing mask painter state model once, then reuse it everywhere.

1. Add per-class opacity state and global border/global opacity controls to `ueler/viewer/plugin/mask_painter.py` and `ueler/viewer/plugin/mask_class_list_widget.py`.
2. Extend the painter helper surface so it can build effective per-FOV color, mode, and opacity maps from the current UI state.
3. Update `ueler/viewer/mask_color_overlay.py` so live painter overlays can apply per-cell opacity and fill-plus-border rendering.
4. Keep uniform mask snapshots and painter snapshots separate: add a painter-specific snapshot payload for non-live consumers instead of overloading `MaskOverlaySnapshot`.
5. Propagate that painter snapshot through the cell gallery, batch export, and ROI manager, and persist it in ROI records.

## Implementation steps
1. Add focused tests for painter opacity state building and overlay blending in `tests/test_mask_painter_mode_visibility.py` and `tests/test_mask_color_overlay.py`.
2. Add per-class opacity state, global opacity state, and a border-on-filled toggle in `ueler/viewer/plugin/mask_painter.py`.
3. Update `ueler/viewer/plugin/mask_class_list_widget.py` if inline per-class opacity controls are needed for the compact row layout.
4. Extend `apply_registry_colors(...)` in `ueler/viewer/mask_color_overlay.py` to accept per-cell opacity and border-on-filled behavior.
5. Update `_compose_fov_image()` in `ueler/viewer/main_viewer.py` to request effective painter opacity state alongside the existing painter color and mode maps.
6. Add a painter-specific snapshot payload to `ueler/rendering/engine.py` and route it through `capture_overlay_snapshot()` / `build_overlay_settings_from_snapshot()` in `ueler/viewer/main_viewer.py`.
7. Update `ueler/viewer/plugin/cell_gallery.py`, `ueler/viewer/plugin/export_fovs.py`, `ueler/viewer/plugin/roi_manager_plugin.py`, and `ueler/viewer/roi_manager.py` to preserve and replay captured painter settings.
8. Run focused tests first, then cross-plugin tests, then a notebook smoke test.

## Validation
- Automated:
  - `python -m unittest tests.test_mask_painter_mode_visibility tests.test_mask_color_overlay`
  - `python -m unittest tests.test_cell_gallery tests.test_export_fovs_batch tests.test_roi_manager_tags`
- Manual:
  1. Launch `script/run_ueler.ipynb`.
  2. Set one class to fill at 100%, one to an intermediate opacity, and one to 0%.
  3. Toggle borders on filled masks and confirm adjacent same-class cells remain distinguishable.
  4. Change the global opacity and confirm only linked classes update.
  5. Confirm the cell gallery, exported images, and ROI thumbnails match the viewer.

## Risks
- If live painter rendering and snapshot replay derive their state differently, viewer and export behavior can fork. Mitigation: centralize painter state building in `mask_painter.py`.
- Extending the anywidget row layout with another per-class control can widen the UI again. Mitigation: keep the first pass narrow and validate panel width before widening the control surface further.
- ROI schema changes can regress older CSV payloads. Mitigation: add backward-compatible defaults in `ueler/viewer/roi_manager.py`.