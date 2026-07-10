# User Interface

This page is a reference map of the UELer interface — the four main regions and the controls in each.
For a hands-on first run, see [Basic Usage](basic-usage.md).

![GUI preview](../GUI_preview.png)

---

## Layout Overview

| Region | Location | Contents |
|---|---|---|
| **Left panel** | Left column | FOV selector, channel / mask / annotation controls, marker sets, advanced settings |
| **Main viewer** | Center | Image canvas, lasso toggle, zoom/pan, scale bar |
| **Right panel** | Right column | Plugin tools (accordion) |
| **Footer** (optional) | Bottom | Wide plugin tabs (e.g. heatmap, multi-scatter) |

Some plugins move themselves into the footer automatically when they need the extra width.

---

## Left Panel

### Top controls

- **Select Image:** — choose the active FOV.
- **Cache Size:** — number of FOVs kept in memory at once (default 100).

### Channels

- **Channels:** — a tag input for the visible channels (add/remove one chip at a time; no
  Shift/Ctrl range select).
- Per-channel **color** dropdown, visibility checkbox, and **Min** / **Max** contrast sliders.
- **Show channel legend** — display a color key for the visible channels.
- **Channel grid view** — render each visible channel as its own labelled pane in a synchronized grid.
- **Mask &lt;name&gt;** dropdowns + enable checkboxes, and **Mask outline px:** — control segmentation
  mask overlays. (For per-class fill/outline, opacity, and saved palettes, use the **Mask painter**
  plugin below.)

### Marker Sets

Save and restore named channel/color/contrast combinations: **Marker Set:** dropdown, **Set Name:**
input, and **Load / Save / Update / Delete Marker Set** buttons (deletion is gated by a **Confirm
Deletion** checkbox).

### Pixel Annotations

Visible when `annotations_folder` contains valid rasters:

- **Show annotation** — toggle the annotation overlay.
- **Annotation:** — choose which annotation to display.
- **Fill alpha:** — overlay transparency.
- **Legend labels:** — show class IDs or text labels.
- **Edit palette…** — customize per-class colors (with save/load of `.pixelannotations.json` sets).

### Advanced Settings

Data-mapping keys (**X key:**, **Y key:**, **Label key:**, **Mask key:**, **Fov key:**), the
**Pixel Size (nm):** input that drives the [scale bar](#main-viewer), and a **Downsample** toggle.

---

## Main Viewer

- **Image canvas** — the composited multi-channel image with any active overlays.
- **Lasso Select** (top of the viewer) — a one-shot toggle: draw a freehand lasso to select the cells
  whose mask centroids fall inside it. The toggle resets itself after each lasso. (Selection drives
  the linked plots and gallery — see [Working with a Cell Table](cell-table.md).)
- **Zoom and pan** — scroll to zoom, drag to pan.
- **Scale bar** — computed from the **Pixel Size (nm):** value when available (there is no separate
  on/off toggle; set the pixel size to 0 to omit it).
- **No image (masks only)** — hide the channel image to inspect overlays on a blank background.

---

## Right Panel (Plugins)

Plugins appear as an accordion. **Which plugins load depends on whether a cell table is present:**

- **Without a cell table**, only these two load — both work on images alone:
    - **ROI manager** — capture, tag, and browse regions of interest. See [Regions of Interest](roi-manager.md).
    - **Batch export** — export FOVs, ROIs, and single-cell crops. See [Batch Export](export.md).
- **With a cell table loaded**, the analytical plugins also appear:
    - **Scatter plot** and **Histogram** — linked distribution plots. See [Scatter & Histogram](scatter-histogram.md).
    - **Gallery** — thumbnails of the currently selected cells.
    - **Heatmap**, **FlowSOM**, **Cell Annotation** — clustering and annotation. See [Clustering & Annotation](clustering-annotation.md).
    - **Mask painter** — per-class mask colors, fill/outline modes, opacity, and saved palettes.
    - **Go to** — jump and zoom to a specific cell.
    - **Cell Table Editor** — write a value onto the selected cells.
    - **Cell tooltip label** — choose which cell-table columns appear in the hover tooltip.

!!! note "Panel order"
    The right-panel accordion order is not curated — locate a plugin by its name rather than its
    position.

---

## Footer (Wide Plugins)

Some plugins expand into a horizontal footer panel so the main viewer stays visible alongside them:

- **Heatmap** — enable **Horizontal layout** in the plugin to move it to the footer.
- **Scatter plot** — moves to the footer automatically when more than one scatter is active.
