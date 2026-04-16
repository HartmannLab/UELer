# Batch Export

The Batch Export plugin lets you export full FOV images, ROI crops, and single-cell crops to PNG or PDF files.

---

## Opening the Batch Export Plugin

The **Batch Export** plugin appears in the right panel. It is available in both standard viewer mode and simple viewer mode (images only, no cell table required).

---

## Export Modes

### Full FOV Export

Exports the entire image of one or more FOVs at a selected resolution.

1. Select the FOVs you want to export (or choose **All FOVs**).
2. Choose the output format (PNG or PDF).
3. Select a **Marker Set** to define the channels and settings for the export.
4. Click **Export**.

### ROI Export

Exports the image crops corresponding to ROIs saved in the ROI Manager.

1. Select the ROIs you want to export (or choose **All ROIs**).
2. The Marker Set and format options apply here too.
3. Click **Export**.

### Single Cell Export

Exports cropped thumbnails centered on individual cells. Requires a cell table to be loaded.

1. Filter cells using the chart or heatmap plugin.
2. In the Batch Export plugin, switch to the **Single Cells** tab.
3. Configure crop size and output format.
4. Click **Export**.

---

## Scale Bar

When pixel size metadata is available, exported images include a scale bar in the lower-right corner. The bar length is automatically rounded to a round physical value and capped to 10% of the image width.

---

## Overlays in Exports

You can include mask or annotation overlays in the exported images. Use the overlay toggles in the Batch Export plugin to enable them. The rendered overlays match what is shown in the viewer.

---

## Map Mode ROI Export

ROIs captured in [map mode](map-mode.md) are exported differently:

- The exporter detects that the ROI has `fov=""` and a `map_id`.
- It renders the corresponding region of the stitched map canvas using physical µm coordinates.
- Output filenames use the prefix `map_<id>_roi_<roi_id>.<format>`.

---

## Marker Set Dropdown

The **Marker Set** dropdown lists all sets currently defined in the channel controls. It updates automatically when you save, overwrite, or delete a marker set — no viewer restart required.

---

## Tips

!!! tip "Simple viewer mode"
    Even without a cell table, you can still use Full FOV Export and ROI Export. The Single Cells tab shows an informational notice instead.

!!! tip "Refreshing the ROI list"
    The ROI list in Batch Export updates automatically whenever a new ROI is captured. If you do not see a recently added ROI, try toggling the **Limit to current FOV** filter.
