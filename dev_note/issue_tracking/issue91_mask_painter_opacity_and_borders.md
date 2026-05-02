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

## Follow-up: reply-to-#91 scope

### Problem
The first implementation preserved painter state for single-FOV ROI thumbnails and downstream replay, but two follow-up gaps remained:

- map-mode ROI thumbnails still returned the raw stitched map render without replaying the saved painter snapshot
- filled-mask borders always reused the fill color, so users could not keep borders aligned with the left-panel mask color when fill colors changed

### Required behavior
- ROI thumbnails and center-on-ROI playback should continue to reflect the painter snapshot captured with the ROI, including in map mode.
- Filled-mask borders should support two modes:
  - `mask_type_color`: use the left-panel mask color captured with the snapshot
  - `same_as_fill`: reuse the class fill color

### Follow-up approach
1. Extend the painter snapshot model with border-color mode plus the resolved mask-type color.
2. Teach `apply_registry_colors(...)` to accept a distinct `border_color_map` for fill-border rendering.
3. Forward that border-color map through live viewer rendering and snapshot replay.
4. Reuse a shared stitched-map replay helper for map-mode ROI thumbnails and map ROI export.
5. Add focused regressions for overlay rendering, painter snapshot capture/restore, map-mode ROI thumbnail replay, cell gallery replay, and map ROI export.

### Follow-up validation
- `python -m unittest tests.test_mask_color_overlay tests.test_mask_painter_mode_visibility tests.test_roi_manager_tags tests.test_cell_gallery`
- `python -m unittest tests.test_export_fovs_batch.BatchExportMapROIItemsTests.test_export_map_roi_worker_calls_render_map_region_direct tests.test_export_fovs_batch.BatchExportMapROIItemsTests.test_export_map_roi_worker_applies_map_bounds_offset tests.test_export_fovs_batch.BatchExportMapROIItemsTests.test_export_map_roi_worker_raises_on_empty_roi`

## Follow-up: reply 2 scope

### Problem
`MaskPainterDisplay.apply_colors_to_masks()` still uses `ImageDisplay.set_mask_colors_current_fov()` for the immediate current-FOV redraw path. That code assumed the sliced label mask and generated edge mask always provided `.compute()`, but the current masking path can already return a NumPy array. The result is an `AttributeError` during recoloring even though the data is already materialized.

### Required behavior
- Applying mask painter colors in the current FOV must work for both NumPy-backed and lazy label-mask inputs.
- The immediate highlight path should keep using the same edge-generation logic without forcing callers to know the backing array type.

### Follow-up approach
1. Materialize array-like inputs inside `ImageDisplay.set_mask_colors_current_fov()` right before `np.where(...)` and right after `generate_edges(...)`.
2. Keep the fix local to the owning image-display method rather than branching in `MaskPainterDisplay.apply_colors_to_masks()`.
3. Add a regression that exercises `set_mask_colors_current_fov()` with a NumPy mask and a NumPy edge mask.

### Follow-up validation
- `python -m unittest tests.test_image_display_tooltip tests.test_mask_painter_mode_visibility`

## Follow-up: reply 3 scope

### Problem
Single-FOV live rendering already resolves mask painter state directly from the current UI via `get_effective_*_for_fov(...)`, but the live map-mode overlay still painted from the last globally registered registry/mode/opacity state. That creates a parity gap: map mode can lag behind the current Mask Painter settings and diverge from the single-FOV display.

### Required behavior
- Live map-mode rendering must use the same effective painter state as single-FOV rendering.
- Color, fill/outline mode, opacity, and filled-border color should all come from the current UI-derived painter state during map-mode redraw.

### Follow-up approach
1. Update `_apply_map_painter_overlay()` to resolve per-FOV painter color/mode/opacity/border maps via the same `get_effective_*_for_fov(...)` helpers used by `_compose_fov_image()`.
2. Keep the old registry lookup only as a fallback when the effective helper is unavailable.
3. Add a regression that fails if map-mode live rendering uses stale registry values instead of the current effective painter state.

### Follow-up validation
- `python -m unittest tests.test_mask_painter_mode_visibility`

## Follow-up: filled-border dimming regression

### Problem
With `show_borders_on_filled=True`, thickened filled-mask borders could alter neighboring filled cells and make their fill colors look dimmer or less saturated. The issue is in the shared helper that every live/snapshot painter path uses: it allowed thickened border masks to spill outside the owning cell and interleaved per-cell fill and border compositing.

### Required behavior
- Filled-mask borders and their thickened pixels should overwrite only the owning cell's fill pixels.
- Enabling filled borders must not change adjacent cells' fill values.
- The fix must apply everywhere that reuses `apply_registry_colors(...)`: live FOV rendering, live map mode, ROI replay, and export replay.

### Follow-up approach
1. Add an adjacent-cell regression that simulates a thickened filled border reaching into a neighbor cell and asserts the neighbor's interior fill value remains unchanged.
2. Refactor `ueler/viewer/mask_color_overlay.py` so fill blending happens before border painting.
3. Clip thickened filled-border masks back to `mask_bool` before painting, preserving overwrite-on-top semantics inside the same cell while preventing spill into adjacent cells.

### Follow-up validation
- `python -m unittest tests.test_mask_color_overlay`
- `python -m unittest tests.test_mask_painter_mode_visibility`

## Follow-up: reply 4 scope

### Problem
The viewer can already suppress channel selection entirely, but that path returns a plain black canvas and bypasses mask/annotation rendering. Reply 4 needs a true `No image (masks only)` mode that skips image compositing work while still rendering overlays, and it must stay consistent beyond the live viewer.

### Required behavior
- Add a left-panel checkbox labeled `No image (masks only)` above the channel settings.
- Disable the checkbox when masks are unavailable.
- When enabled, do not render the image layer at all; instead render masks and annotations over a black background.
- Keep the same behavior in single-FOV mode, map mode, export, ROI thumbnails, and the cell gallery.

### Follow-up approach
1. Add a shared `skip_image_layer` render flag to `ueler/rendering/engine.py` so `render_fov_to_array(...)`, `render_crop_to_array(...)`, and `render_roi_to_array(...)` can start from a black base while still applying annotation and mask overlays.
2. Add a viewer-level checkbox and callback in `ueler/viewer/ui_components.py` and `ueler/viewer/main_viewer.py`, and include the flag in `_map_state_signature(...)` so stitched-map tiles redraw when the mode changes.
3. Capture the flag in `OverlaySnapshot` for downstream consumers that already rely on snapshot replay.
4. Forward the flag through export workers and gallery/ROI render helpers so downstream renders honor the same no-image state.
5. Add focused regressions for shared rendering, viewer forwarding, export propagation, gallery propagation, and ROI snapshot propagation.

### Follow-up validation
- `python -m unittest tests.test_rendering.RenderingHelpersTests.test_render_fov_to_array_skips_image_layer_but_preserves_masks tests.test_rendering.RenderingHelpersTests.test_render_fov_to_array_without_channels_can_render_overlays tests.test_mask_painter_mode_visibility.TestMaskPainterRenderPath`
- `python -m unittest tests.test_export_fovs_batch.ExportFOVsBatchTests.test_capture_overlay_snapshot_and_rebuild tests.test_export_fovs_batch.BatchExportMapROIItemsTests.test_export_map_roi_worker_calls_render_map_region_direct tests.test_export_fovs_batch.BatchExportMapROIItemsTests.test_render_map_region_direct_uses_render_fov_to_array_per_tile tests.test_export_fovs_batch.BatchExportMapROIItemsTests.test_export_map_roi_worker_applies_map_bounds_offset tests.test_cell_gallery.TestCellGalleryColors.test_gallery_forwards_skip_image_layer_from_snapshot tests.test_roi_manager_tags.ROIManagerMapModeTests.test_build_overlay_snapshot_carries_no_image_mode`

## Follow-up: reply 5 scope

### Problem
The Mask Painter still had a few state-model gaps after reply 4:

- changing the global fill opacity only matched classes by raw value, so inherited classes could drift when per-class widgets were rebuilt or coincidentally shared the same value
- there was no global fill toggle for classes still inheriting the default fill behavior
- saved palettes did not preserve the active class list, `Only specified` state, or the difference between globally linked classes and classes that merely matched the same value

### Required behavior
- Global fill opacity changes should update only classes that are still linked to the inherited global opacity behavior.
- Add a global fill toggle beside the global opacity control, scoped to classes still inheriting the global/default fill mode.
- Keep customized classes surfaced first when `Only specified` is enabled.
- Save and reload the full Mask Painter display state, while older palettes continue to load with sensible defaults.

### Follow-up approach
1. Track linked fill classes and linked opacity classes explicitly inside `MaskPainterDisplay`, with a value-based fallback only when old/manual control replacements leave no tracked linkage for the current controls.
2. Add a `Global fill` checkbox to the Mask Painter control row and route it through the same linked-class update logic used by the global opacity control.
3. Persist reply-5 state in `.maskcolors.json`: `active_classes`, `only_specified`, `global_fill`, `linked_fill_classes`, and `linked_opacity_classes`, alongside the already-saved per-class color/mode/visibility/opacity and border settings.
4. Restore missing reply-5 fields from current defaults when loading older palettes, and guard palette restore with `_syncing` so partial widget updates do not trigger intermediate handlers.
5. Add focused regressions for linked global fill propagation, `Only specified` ordering, full palette round-trip, and older-palette fallback behavior.

### Follow-up validation
- `python -m unittest tests.test_mask_painter_mode_visibility tests.test_mask_color_sets`

## Follow-up: reply 6 implementation

### Implemented behavior
- Global fill opacity now uses live value-based linkage again: any class whose current opacity still matches the previous global opacity value is updated when the global value changes.
- `Global fill` now governs the effective mode of inactive classes only. Inactive classes still use the default color, but when `Global fill` is enabled they resolve as filled with the global opacity; when it is disabled they resolve as outlines.
- Active class mode controls are no longer batch-mutated by the `Global fill` toggle, which preserves the original outline/fill choice for customized active classes.
- `Only specified` now restores the full list in a customized-first order after a toggle cycle instead of reverting to the original unsorted order.
- The Global fill control row now has explicit spacing and a narrower opacity input.

### Implementation notes
1. Updated `build_painter_state_maps_for_fov(...)` and `apply_colors_to_masks(...)` together so single-FOV effective rendering, map-mode replay, and globally registered painter state all resolve inactive/default-state classes the same way.
2. Added `global_fill` to `MaskPainterSnapshot` so ROI preset capture/apply and other snapshot-based consumers retain the inactive-class default-mode semantics.
3. Left reply-5 palette fields backward-compatible while changing runtime behavior to rely on live widget/default-state values instead of the saved linked-opacity metadata.

### Validation
- `python -m unittest tests.test_mask_painter_mode_visibility tests.test_mask_color_sets`
- `python -m unittest tests.test_roi_manager_tags.ROIManagerTagsTests.test_capture_and_apply_roi_preserves_mask_painter_payload`

## Follow-up: reply 7 — Global fill toggle-off for inactive classes

### Problem
Toggling `Global fill` OFF caused inactive classes to display as a "fused blob of fill with shared outline" instead of reverting to separate per-cell outlines. Two bugs: (1) the fill was not removed, and (2) the outlines of adjacent same-class inactive cells merged.

### Root cause
Inside `apply_colors_to_masks`, inactive classes were handled by calling `_apply_color_to_current_fov` (which calls `ImageDisplay.set_mask_colors_current_fov`). That function:
1. Added outlines on top of the existing canvas via `cummulative=True` without clearing prior fills.
2. Passed a combined multi-label integer array to `generate_edges` → `find_boundaries(mode="inner")`, which only detects the outer boundary of the combined foreground, not per-cell boundaries, merging adjacent same-class cells into one outline.

While the subsequent `update_display → _compose_fov_image → apply_registry_colors` call at the end of `apply_colors_to_masks` would have produced the correct per-cell rendering, the intermediate wrong state could persist in Jupyter widget environments where the first `draw_idle()` fired before the second.

### Fix
Removed the `_apply_color_to_current_fov(cls_value, self.default_color)` call from the inactive-class loop in `apply_colors_to_masks`. Inactive classes are now rendered exclusively via `update_display → _compose_fov_image → apply_registry_colors`, which resolves fresh per-cell outline/fill state from `get_effective_mode_map_for_fov()` on every redraw.

### Implementation notes
1. Change is in `apply_colors_to_masks` in `ueler/viewer/plugin/mask_painter.py` (~lines 1269–1273): removed two lines from the inactive-class loop.
2. Active classes are unaffected; their `_apply_color_to_current_fov` calls remain.

### Validation
- `python -m unittest tests.test_mask_painter_mode_visibility tests.test_mask_color_sets`