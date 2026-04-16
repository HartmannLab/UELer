# Export Pipeline

> Source: [`dev_note/topic_export_pipeline.md`](https://github.com/HartmannLab/UELer/blob/main/dev_note/topic_export_pipeline.md)

---

## Context

The export pipeline handles batch exports for FOVs, ROIs (including map-mode ROIs), and single-cell crops. It also provides a shared scale bar computation utility.

---

## Key Decisions

- A **job runner abstraction** structures exports as per-item jobs with result tracking.
- Export rendering is independent of UI widget state — `channel_settings` are read directly from the marker profile.
- Overlay snapshots (masks, annotations) and configurable outline thickness are supported.
- Scale bar helpers compute a rounded physical length capped to 10% of image width.

---

## Job Runner

Each export run produces a list of `JobItem` objects. The runner processes them sequentially, writing the output file and recording success/failure per item. Results are displayed in the plugin UI.

---

## Rendering Pipeline

```
_build_roi_items()
    ↓
for each item:
    _export_fov_worker()          ← single-FOV exports
    _export_map_roi_worker()      ← map-mode ROI exports
        ↓
        _render_map_region_direct()
            ↓
            VirtualMapLayer._collect_visible_tiles()
            render_fov_to_array(tile, ..., channel_settings)
            _blit_tile()
    ↓
    _finalise_array()
    _write_image()
```

---

## Scale Bar

- Computed in `scale_bar_helper.py`.
- Physical length is rounded to a "nice" value and capped to 10% of the image width.
- Shared between the live viewer display and all export paths.
- `pixel_size_nm` for map-mode exports: `base_pixel_size_um × 1000 × downsample`.

---

## Map-Mode ROI Export

Map-mode ROIs (`fov=""`, `map_id` non-empty) are routed to `_export_map_roi_worker()`:

1. Retrieve the `VirtualMapLayer` via `_get_map_layer(map_id)`.
2. Convert canvas pixels to physical µm using `layer.map_bounds()` + `base_pixel_size_um()`.
3. Call `_render_map_region_direct()` to render without UI dependency.
4. Pass through `_finalise_array()` + `_write_image()`.

---

## NaN Sanitization

`_ensure_dataframe()` sanitizes the `fov` column: empty CSV cells read back as `float('nan')` by pandas. Without sanitization, `NaN` is truthy and `_build_roi_items()` would route map-mode ROIs to the single-FOV path, producing black output with a filename of `nan_...`.

---

## Related Issues

- [#1](https://github.com/HartmannLab/UELer/issues/1)
- [#78](https://github.com/HartmannLab/UELer/issues/78)
