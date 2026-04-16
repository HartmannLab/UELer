# Map Mode

Map mode stitches multiple FOVs into a single spatial overview, allowing you to navigate the full tissue section as one continuous canvas.

---

## What Is Map Mode?

In standard view, UELer loads one FOV at a time. In **map mode**, all FOVs are stitched together using a spatial layout descriptor. This lets you:

- See the spatial relationship between FOVs at a glance.
- Pan and zoom across the entire tissue without switching FOVs manually.
- Capture and export ROIs that span multiple FOVs.

---

## Enabling Map Mode

Map mode requires a **map descriptor** file that defines the physical positions and sizes of each FOV. To activate it:

1. In the notebook, pass the path to your map descriptor when creating the viewer.
2. Click the **Map mode** toggle in the viewer controls.

Once activated, the image canvas switches to the stitched composite view and the FOV selector is disabled (navigation happens by panning).

---

## Navigating in Map Mode

- **Pan** — Click and drag the canvas to move around the stitched map.
- **Zoom** — Scroll to zoom in or out.
- **Tile rendering** — Only the tiles visible in the current viewport are loaded; the viewer caps the number of tiles rendered per frame to maintain responsiveness.

!!! note "Large datasets"
    For maps with many FOVs (> 80 tiles in view), only the nearest tiles to the viewport center are rendered. Adjust the `map_render_tile_limit` setting in the viewer if needed.

---

## Capturing ROIs in Map Mode

You can capture ROIs in map mode just like in standard mode. The ROI Manager stores the stitched-map coordinates alongside a `[MAP:<id>]` label so you can distinguish map-mode ROIs from single-FOV ROIs.

ROI thumbnails in map mode are rendered from the stitched tile layer.

---

## Exporting ROIs in Map Mode

Batch export fully supports map-mode ROIs. When a ROI has `fov=""` and a non-empty `map_id`, the batch exporter renders the stitched region using the `VirtualMapLayer` and saves it with the filename prefix `map_<id>_roi_<roi_id>.<format>`.

See the [Batch Export](export.md) tutorial for more details.

---

## Deactivating Map Mode

Click the **Map mode** toggle again to return to single-FOV mode. The FOV selector and all per-FOV plugin behaviors are restored automatically.
