# Working with a Cell Table

A **cell table** is a per-cell feature table (a CSV, e.g. from `ark-analysis`) with one row per
segmented cell. Loading one turns UELer from an image viewer into a linked single-cell explorer. This
page explains how to load a table, the idea of **linked selection** that ties everything together,
and the tools that operate on the current selection.

---

## Loading a Cell Table

Attach a table to an existing viewer with `load_cell_table`:

```python
import pandas as pd
from ueler import load_cell_table

cell_table = pd.read_csv(cell_table_path)
load_cell_table(viewer, cell_table=cell_table, auto_display=True, after_plugins=True)
```

See [Get Started](../getting-started.md) for the full launch flow. Once the table is loaded, the
analytical plugins appear in the right panel (**Scatter plot**, **Histogram**, **Gallery**,
**Heatmap**, **FlowSOM**, **Cell Annotation**, **Mask painter**, **Go to**, **Cell Table Editor**,
**Cell tooltip label**).

---

## The Idea: Linked Selection

Most single-cell features in UELer revolve around one shared concept — a **current selection** of
cells. Whenever you select cells in one place, every linked view updates:

```
Scatter / Histogram / Lasso  ─┐
                              ├──▶  selected cells  ──▶  highlighted in the image
                              │                     └─▶  shown in the Cell Gallery
```

Plots such as the scatter and histogram publish their selection; when their **Main viewer** and
**Cell gallery** link checkboxes are enabled, that selection highlights the corresponding masks in
the image and populates the gallery. This is what makes "gate on a marker, see those cells light up
in the tissue" work.

---

## Selecting Cells

You can create a selection several ways:

- **Lasso Select** — the one-shot toggle at the top of the viewer; draw a freehand loop to select the
  cells whose mask centroids fall inside it.
- **Scatter plot / Histogram** — brush or gate on marker values. See
  [Scatter & Histogram](scatter-histogram.md).
- **Heatmap / FlowSOM** — trace a cluster or meta-cluster. See
  [Clustering & Annotation](clustering-annotation.md).

---

## The Cell Gallery

The **Gallery** plugin shows cropped thumbnails of the currently selected cells (5 per row, with
internal scrolling). Each tile is labelled `<fov>: <mask id>`; click a tile to jump the viewer to
that cell.

Useful controls:

- **Cutout Size (px):** — crop size around each cell (default 150).
- **Max Displayed Cells:** — cap the number of thumbnails (default 20; warns above 100).
- **Downsample**, **Refresh**, **Mask colour** / **Outline px**, and **Use uniform color** (when
  off, painted colors from the Mask painter are shown).

---

## Acting on the Selection

Three plugins consume the current selection directly:

- **Go to** — jump and zoom to a specific cell. Pick the **FOV:** and **Cell ID:**, set **Width
  (pixel):** (the crop width, default 150), and click **Go to**.
- **Cell Table Editor** — write a value onto the selected cells. Enter a **Column:** (new or
  existing) and a **Value:**, then click **Apply to selected cells** (the button is enabled only when
  a selection exists). This is handy for manual gating/labelling.
- **Cell tooltip label** — tick the cell-table columns you want to appear in the image hover tooltip.
  It reacts live to each checkbox.

---

## Next Steps

- [Scatter & Histogram](scatter-histogram.md) — explore marker distributions with linked brushing.
- [Clustering & Annotation](clustering-annotation.md) — heatmaps, FlowSOM, and annotation checkpoints.
