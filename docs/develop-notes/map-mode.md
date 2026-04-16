# Map Mode Internals

> Source: [`dev_note/topic_map_mode_spatial.md`](https://github.com/HartmannLab/UELer/blob/main/dev_note/topic_map_mode_spatial.md)

---

## Context

Map mode stitches multiple FOVs into a single composite view and introduces spatial navigation challenges for plugins that assume per-FOV coordinates.

---

## Key Decisions

- **`VirtualMapLayer`** renders stitched tiles by reusing existing single-FOV rendering paths.
- Map mode is gated behind `ENABLE_MAP_MODE`; single-FOV behavior is fully preserved when disabled.
- FOV-local coordinates are resolved into stitched-map pixels for selection and navigation without mutating underlying data tables.
- A dedicated stitched-tile cache (keyed by channel and overlay state) avoids stale renders.

---

## Coordinate Systems

| System | Description |
|---|---|
| FOV-local pixels | Pixel coordinates within a single FOV image |
| Stitched-canvas pixels | Pixel coordinates in the full stitched map canvas |
| Physical µm | Real-world coordinates from the map descriptor |

Conversion helpers:

- `resolve_cell_map_position()` — FOV-local → stitched-canvas
- `map_bounds()` → physical µm bounds of the full map
- `base_pixel_size_um()` → µm per pixel at base resolution

---

## Tile Rendering

1. `VirtualMapLayer.render()` is called with the current viewport.
2. `_collect_visible_tiles()` returns tiles whose bounding boxes intersect the viewport.
3. Each tile is rendered via `render_fov_to_array()` using `channel_settings` from the marker profile (no UI widget reads).
4. Tiles are stitched onto the canvas via `_allocate_canvas()` + `_blit_tile()`.
5. If the number of visible tiles exceeds `_RENDER_TILE_LIMIT`, only the nearest tiles to the viewport center are rendered.

---

## Plugin Map-Mode Hooks

All plugins inherit from `PluginBase`, which provides two lifecycle hooks:

- `on_map_mode_activate()` — Called when map mode is enabled. Plugins should disable FOV-scope filters.
- `on_map_mode_deactivate()` — Called when map mode is disabled. Plugins restore normal behavior.

The active FOV is retrieved via `get_active_fov()`, which returns `None` in map mode to prevent stale widget reads.

---

## Known Fixes

- **Missing bounds offset (v0.3.1):** `_export_map_roi_worker` now applies `map_bounds()` origin offset to canvas-pixel → µm conversion.
- **Render suppression at startup:** `_suspend_display_updates` prevents burst renders during `load_widget_states`.

---

## Related Issues

- [#3](https://github.com/HartmannLab/UELer/issues/3)
- [#58](https://github.com/HartmannLab/UELer/issues/58)
- [#59](https://github.com/HartmannLab/UELer/issues/59)
- [#62](https://github.com/HartmannLab/UELer/issues/62)
