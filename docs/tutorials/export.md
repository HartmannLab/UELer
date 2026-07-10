# Batch Export

The **Batch export** plugin exports full FOV images, ROI crops, and single-cell crops to image files.
It works in both standard and simple (images-only) viewer mode.

---

## Export Modes

The plugin's **Mode** tabs select what you export:

### Full FOV

Exports whole FOV renderings.

- **Export all FOVs** (on by default) or pick specific FOVs from **FOVs:**.
- Optional **Figure width (px):** / **Figure height (px):** overrides (0 = auto).

### Single Cells

Exports cropped thumbnails centered on individual cells. Requires a cell table.

- **Filter (query):** — a pandas-style query (e.g. `marker > 0`); click **Apply**.
- Pick cells from **Cells:**, set **Crop size (px):** (default 128), and use **Preview** to check.

!!! note "Simple viewer mode"
    Without a cell table, the Single Cells tab shows a notice and is unavailable. Full FOV and ROI
    export still work.

### ROIs

Exports the crops for ROIs saved in the [ROI manager](roi-manager.md).

- Pick ROIs from **ROIs:**, optionally limited to the current FOV via **Current FOV only**.

---

## Shared Options

These apply to whichever mode is active:

- **Marker set:** — the channels/colors/contrast used for the render.
- **Output folder:** (defaults to `<base_folder>/exports`) and **Browse**.
- **Format:** — **PNG**, **JPG**, **TIF**, **TIFF**, or **PDF**.
- **Downsample:** and **DPI:** (default 300).
- **Include scale bar** and **Scale bar % width:** (1–10%, default 10).
- **Include annotations** and **Include masks** (both on by default).
- **Mask outline px:** — outline thickness.

### Mask appearance

- **Masks only (black background)** — render masks on black, without the channel image.
- **Mask layer:**, **Mask color:**, and **Mask opacity:** (0–1) — set the color and opacity of a
  specific mask layer.
- **Override mask palette** + **Palette:** — apply a saved palette instead of the current settings.

### Exporting channels separately

- **Export channels separately** — write one file per channel instead of a single composite.
- **Merge same color** (enabled only when the above is on) — group channels that share a color into a
  single merged file.

---

## Export Config Templates

Save a full set of export options for reuse. In the **Export config templates** accordion: type a
**Name:** and click **Save config**; reload later from the **Saved:** dropdown with **Load config**
(or **Delete**). Templates are stored under `<base_folder>/.UELer/export_configs/`.

---

## Running an Export

Click **Start** to begin (or **Cancel** to stop). A progress bar tracks the run, and an **Open output
folder** link appears when it finishes.

---

## Output Filenames

| Export | Filename pattern |
|---|---|
| Full FOV | `<fov>.<format>` |
| Single cell | `<fov>_cell_<label>.<format>` |
| ROI (single FOV) | `<fov>_roi_<name-or-id>.<format>` |
| ROI (map mode) | `map_<map_id>_roi_<name-or-id>.<format>` |
| Separate channels | `<stem>_<channel>.<format>` |

Map-mode ROIs are rendered from the stitched map layer — see [Map Mode](map-mode.md).
