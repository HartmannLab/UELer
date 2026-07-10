# Map Mode

Map mode stitches multiple FOVs into a single spatial overview, letting you navigate a full tissue
section as one continuous canvas. It works on images alone (no cell table required).

!!! note "Opt-in feature"
    Map mode is off by default. It only appears when **both** conditions are met:

    1. The `ENABLE_MAP_MODE` environment variable is set to a truthy value (`1`, `true`, `yes`,
       `on`) before launching the viewer, and
    2. At least one map descriptor is found under `<base_folder>/.UELer/maps/` (UELer scans that
       folder for `*.json` descriptors).

    When either is missing, the map controls stay hidden.

---

## What Is Map Mode?

In standard view, UELer loads one FOV at a time. In map mode, FOVs are stitched together using a
spatial layout descriptor. This lets you:

- See the spatial relationship between FOVs at a glance.
- Pan and zoom across the whole tissue without switching FOVs manually.
- Capture and export ROIs that span the stitched map.

---

## Enabling Map Mode

1. Launch the viewer with `ENABLE_MAP_MODE=1` set in the environment, with map descriptor JSON files
   present under `<base_folder>/.UELer/maps/`.
2. In the left panel, tick the **Map mode** checkbox.
3. Choose a map from the **Select map:** dropdown (options are labelled with the map id and its tile
   count).

Once active, the canvas switches to the stitched composite and the **Select Image:** FOV selector is
disabled — you navigate by panning. The channel grid view is also disabled while map mode is on.

---

## Navigating in Map Mode

- **Pan** — click and drag the canvas.
- **Zoom** — scroll to zoom in and out.
- **Tile rendering** — only tiles near the viewport are drawn. Already-cached tiles always render;
  when too many *uncached* tiles would be needed at once, the viewer keeps the ones nearest the
  viewport center to stay responsive (the budget scales with the downsample factor).

---

## Capturing ROIs in Map Mode

Capture ROIs exactly as in standard mode (see [Regions of Interest](roi-manager.md)). Map-mode ROIs
are stored with an empty `fov` and a populated `map_id`, and display a `[MAP:<id>]` location label so
you can tell them apart from single-FOV ROIs. The FOV-scope filters in the ROI manager are disabled
while map mode is active.

---

## Exporting ROIs in Map Mode

[Batch Export](export.md) fully supports map-mode ROIs. A map ROI (empty `fov`, non-empty `map_id`)
is rendered from the stitched map layer and saved with the filename pattern:

```
map_<map_id>_roi_<name-or-id>.<format>
```

---

## Deactivating Map Mode

Untick the **Map mode** checkbox to return to single-FOV mode. The FOV selector and per-FOV behaviors
are restored automatically.
