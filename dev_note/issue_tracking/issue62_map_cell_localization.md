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
