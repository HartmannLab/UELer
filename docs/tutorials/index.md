# Tutorials

These tutorials build up from simply viewing images to full single-cell analysis. You don't need to
read them in order, but each section assumes you're comfortable with the one before it.

---

## A quick note on cell tables

UELer works in two modes, and it's worth knowing which one you're in:

- **Without a cell table**, UELer is an image viewer. You can browse channels, overlay masks and
  annotations, capture regions of interest, and batch-export images.
- **With a cell table loaded**, UELer becomes a linked single-cell explorer — scatter plots,
  histograms, heatmaps, clustering, and the cell gallery all light up and stay synchronized with the
  image.

This isn't just a suggestion: the viewer only loads the analytical plugins once a cell table is
present. With no cell table, the right panel shows just the **ROI manager** and **Batch export**
plugins. See [Working with a Cell Table](cell-table.md) for how to load one.

---

## Part 1 · Essentials

Everything here works on images alone — no cell table required.

| Tutorial | What you'll learn |
|---|---|
| [Basic Usage](basic-usage.md) | Launch the viewer, pick an FOV, select channels, and navigate |
| [User Interface](user-interface.md) | A reference map of the four regions and every left-panel control |
| [Regions of Interest](roi-manager.md) | Capture, tag, and browse ROIs that persist across sessions |
| [Map Mode](map-mode.md) | Stitch multiple FOVs into one spatial canvas *(opt-in feature)* |
| [Batch Export](export.md) | Export full FOVs, ROIs, and single-cell crops to image files |

---

## Part 2 · Single-Cell Analysis

These features require a **cell table** (a per-cell feature CSV, e.g. from `ark-analysis`).

| Tutorial | What you'll learn |
|---|---|
| [Working with a Cell Table](cell-table.md) | Load a table, understand linked selection, and use the cell gallery |
| [Scatter & Histogram](scatter-histogram.md) | Explore marker distributions with linked brushing |
| [Clustering & Annotation](clustering-annotation.md) | Heatmaps, FlowSOM clustering, and saving annotation checkpoints |

---

!!! tip "New here?"
    Start with [Basic Usage](basic-usage.md), then skim the [User Interface](user-interface.md)
    reference. Once you have a cell table, move on to
    [Working with a Cell Table](cell-table.md).
