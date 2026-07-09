# Issue #94 — Map Mode Edge Hidden + Batch Export Subregion

## Problem

Two distinct map-mode bugs:

1. **Map edge hidden in viewer** — when fully zoomed out (ds=8 or ds=16), tiles at the edge of a large map (>80 FOVs) are not rendered; they appear black. Zooming in reveals them.
2. **Batch export exports only a subregion** — when exporting a large multi-FOV ROI at full resolution, the output contains only part of the expected region (blank patches where tile loading silently failed).

---

## Bug 1: Map Edge Hidden in Viewer

### Root Cause

**File:** `ueler/viewer/virtual_map_layer.py` — `render()`, lines 125–139

```python
_RENDER_TILE_LIMIT: int = getattr(self._viewer, '_map_render_tile_limit', 80)
if len(visible_tiles) > _RENDER_TILE_LIMIT:
    # sort by distance from viewport centre, keep closest 80
    visible_tiles = sorted(...)[:_RENDER_TILE_LIMIT]
```

For a map with more than 80 FOVs at a fully-zoomed-out viewport, all tiles are visible but the limit drops the ones farthest from the viewport centre — always the edge tiles.

The limit was designed for `ds=1` (high resolution, slow TIFF I/O). At `ds=8` each tile is 64× cheaper to render, and a MIBI experiment with 100–300 FOVs at coarse zoom needs all tiles for a complete overview.

### Fix

**File:** `ueler/viewer/virtual_map_layer.py`

Two changes:

1. Moved `channels_tuple` and `state_signature` computation **before** the tile cap so they can be used for cache key lookup in the partition step.
2. Replaced the flat 80-tile cap with a cache-aware, ds-scaled cap:
   - Cached tiles always render regardless of count (zero I/O cost).
   - Uncached tile budget = `base_limit × ds_factor` (so ds=8 → 640, covering virtually all practical maps).

---

## Bug 2: Batch Export Exports Only a Subregion

### Root Cause (RC-A, primary)

**File:** `ueler/viewer/plugin/export_fovs.py` — `_render_map_region_direct`

```python
viewer.load_fov(tile.name, channels)
fov_arrays = viewer.image_cache.get(tile.name)
if not fov_arrays:
    continue  # silently skips the tile → black patch in output
```

When a tile cannot be loaded (network FS issue, cache eviction race), the tile is skipped silently. For a large ROI spanning many FOVs the result is an image with correct overall dimensions but black tiles — described as "only a subregion".

### Fix (partial — see issue #98 for follow-up)

**File:** `ueler/viewer/plugin/export_fovs.py`

Replaced the silent `continue` with a `warnings.warn` that names the failed tile:

```python
if not fov_arrays:
    import warnings
    warnings.warn(
        f"Map export: tile '{tile.name}' could not be loaded; it will appear blank in the output.",
        stacklevel=2,
    )
    continue
```

> **Note:** This fix was insufficient — the partial canvas was still written to disk, producing partial images. Issue #98 addresses this with a retry + `RuntimeError` to abort the export instead of silently accepting an incomplete result. See [issue98_partial_map_export.md](issue98_partial_map_export.md).

---

## Tests Added

**File:** `tests/test_virtual_map_layer.py` — new class `TileCapTests`:

| Test | Assertion |
|------|-----------|
| `test_render_all_cached_tiles_beyond_limit` | After cache warms, all tiles appear in `_last_visible_fovs` and viewer is not called again |
| `test_render_limits_uncached_tiles_scales_with_ds` | ds=1 → 2 uncached tiles rendered; ds=4 → all 5 tiles rendered (8 ≥ 5) |
| `test_allocate_canvas_shape_never_under_allocated` | Canvas dimensions ≥ `ceil(extent / pixel_size)` for various float extents |

**File:** `tests/test_export_fovs_batch.py` — added to `BatchExportMapROIItemsTests`:

| Test | Assertion |
|------|-----------|
| `test_render_map_region_direct_warns_when_tile_load_fails` | `UserWarning` naming the tile emitted when `image_cache.get` returns `None` |
| `test_render_map_region_direct_canvas_size_correct_for_multi_tile_roi` | Canvas shape ≥ `ceil(300/ds) × ceil(300/ds)` for a 3×3 tile 300µm ROI |
