# Issue #62 — Map Mode Cell Localization Plan

## Context
- Map mode renders stitched FOVs using descriptor-provided offsets, but plugins still assume cell coordinates are relative to the original standalone FOV images.
- When scatter plots, cell gallery rows, or Go To actions try to center on a cell while map mode is active, they pan to the wrong location because they ignore the stitched-map offset.
- Raw XY values from the cell table must remain unchanged for data access and lazy loading workflows; only interactive display routines should apply stitched offsets.

## Proposed Approach
- Reuse the existing `VirtualMapLayer` geometry to expose each tile's map coordinates and pixel scale so the viewer can translate FOV-local XY values into stitched-map positions.
- Add a viewer-level helper that resolves a cell's stitched position (in map pixels) and a higher-level helper that recenters the viewport while preserving toolbar history.
- Update plugins that focus the main viewer on a cell (scatter, heatmap scatter, cell gallery, Go To) to route through the new helper. Continue falling back to legacy FOV behavior when map mode is disabled or when the FOV is missing from the active descriptor.

## Implementation Steps
1. **Tile geometry exposure**
   - Extend `VirtualMapLayer` with a lookup table keyed by FOV name and a method that returns geometry (pixel size, map offsets, dimensions) for quick retrieval.
2. **Map coordinate resolver**
   - Add a dataclass for stitched map positions plus a `ImageMaskViewer.resolve_cell_map_position(...)` helper that converts FOV-local pixels into stitched-map pixel coordinates using descriptor bounds.
3. **Viewport focusing helper**
   - Implement `ImageMaskViewer.focus_on_cell(...)` that handles both stitched and single-FOV workflows, keeps toolbar navigation history intact, and avoids toggling out of map mode.
4. **Plugin integrations**
   - Update `chart`, `chart_heatmap`, `cell_gallery`, and `go_to` plugins to call the new helper (passing their existing padding / crop widths). Ensure legacy fallbacks remain when the helper returns `None`.
5. **Tests**
   - Extend `tests/test_map_mode_activation.py` (or a new suite) with coverage for the resolver and focusing helper in both stitched and non-stitched contexts using lightweight stubs.

## Edge Considerations
- Guard against descriptors that omit the requested FOV or report inconsistent pixel sizes by returning `None` and preserving existing behavior.
- Preserve existing cell-table coordinates for exports and analytics by never mutating the underlying dataframe or default key settings.
- Maintain axis orientation when map mode inverts the Y axis by checking current limits before recentering.

## Status
- 2025-11-28: Implemented stitched coordinate helper, plugin updates, and regression tests; `python -m unittest tests.test_map_mode_activation` passes.
- 2025-11-28: Follow-up reported `ValueError` from `jscatter.compose` when selecting cells with multiple scatter plots; root cause traced to duplicate selection syncing. Plan to rely on chart plugin’s internal selection propagation and disable `compose(..., sync_selection=True)` to prevent unobserve failures. Add targeted regression to ensure compose is invoked without sync selection.
- 2025-11-29: Additional regression report highlights three gaps: (1) stitched centering still falls outside the expected region for some FOVs, (2) map-to-map switching does not occur when navigating to cells outside the active slide, and (3) mask hover/click interactions lack stitched-coordinate awareness. Work items below address each gap.

## Follow-up Plan (2025-11-29)

1. **Map coordinate reconciliation**
   - Allow `ImageMaskViewer.focus_on_cell` to identify the owning map for a target FOV, switch `self._active_map_id` when necessary, and remove canvas-based clamping that corrupts stitched coordinates for larger slides.
   - Extend `VirtualMapLayer` with a reverse lookup so map-pixel coordinates can be resolved back to FOV-local pixels for hover/click interactions.

2. **Viewport + descriptor switching safeguards**
   - Maintain a FOV→map lookup generated alongside descriptor ingestion to drive the automatic stitched-map switch.
   - Refresh the stitched viewport immediately after switching maps so downstream helpers operate with the correct layer and bounds information.

3. **Mask interactivity in map mode**
   - Update `ImageDisplay` hover/click handlers to translate map coordinates into FOV-local pixels, fetch mask IDs from cached label masks, and surface the correct tooltip metadata.
   - Introduce lightweight stitched-map highlight helpers (per-mask bounding overlays) so clicks still outline the selected cell while map mode remains active.

4. **Regression coverage**
   - Add tests covering the FOV→map lookup, stitched coordinate inversion, and composed chart behaviour to guard map switching and mask interactivity.

## Status — 2025-11-30

- Implemented a persistent FOV→map index during descriptor loading and taught `focus_on_cell` to activate the owning stitched map before centering, eliminating cross-slide navigation slips.
- Extended `VirtualMapLayer` with per-render viewport metadata and a `localize_map_pixel` helper; `ImageMaskViewer` now resolves map pixels back to FOV-local coordinates and surfaces them via `resolve_map_pixel_to_fov` / `resolve_mask_hit_at_viewport`.
- Reworked `ImageDisplay` hover and click pipelines to rely on the viewer’s stitched resolvers, introducing the `MaskSelection` structure so tooltips, chart tracing, and mask highlighting stay consistent across map and single-FOV modes.
- Added stitched highlight rendering that maps mask outlines onto the combined canvas when map mode is active, keeping selection visuals intact after the coordinate translation.
- Expanded `tests/test_map_mode_activation.py` with coverage for map switching, pixel reverse lookups, and map-aware mask hits; updated tooltip integration tests to exercise the new resolver API; ran `python -m unittest tests.test_map_mode_activation tests.test_image_display_tooltip`.
