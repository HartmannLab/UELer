# Scatter & Histogram

The **Scatter plot** and **Histogram** plugins are UELer's linked distribution tools. Both require a
[cell table](cell-table.md), share the same channel picker, and publish a selection that can drive the
image and the [cell gallery](cell-table.md#the-cell-gallery).

Both plugins have a **Linked plugins** tab with two checkboxes — **Main viewer** and **Cell gallery** —
that control whether their selection highlights masks in the image and fills the gallery.

---

## Histogram

The **Histogram** plugin draws one distribution per selected channel, rendered with Bokeh.

1. Pick channels in the **Channels:** field (the same tag input as the left panel; you can also load
   a marker set's channels via the **Marker set:** dropdown + **Load set**).
2. Click **Plot**. Adjust **Bins:** (default 50) as needed.

### Interaction modes

The **Interaction:** toggle switches between two ways of selecting cells:

- **Cutoff** (default) — click a histogram to set an above/below threshold. A red dashed line marks
  the cutoff; the **Highlight:** toggle chooses whether cells **below** or **above** it are selected.
- **Brush** — drag a range on a histogram to select the cells in it. The selected subset's
  distribution is overlaid (in orange) on *every* channel's histogram for comparison.

Use **Clear selection** to reset. The legend entries (**All** / **Selected**) can be clicked to hide
either series.

!!! note "Bokeh in VS Code"
    The histogram loads BokehJS automatically on first plot, so it renders in VS Code without a
    priming cell. If you see a "requires Bokeh" notice, install `bokeh` and `jupyter_bokeh` (both are
    UELer dependencies) and restart the kernel.

---

## Scatter Plot

The **Scatter plot** plugin plots cell features against each other using an interactive
[jscatter](https://github.com/flekschas/jupyter-scatter) widget.

### A single scatter

On the **Single-pair** tab, choose **X:**, **Y:**, and optionally a **Color:** column, then click
**Plot**. Adjust **Point Size:** on the **Scatter plot** tab.

### All pairwise scatters (multi-pair)

To compare many markers at once, select several channels in the top **Channels:** picker and click
**Plot all pairs**. UELer generates a scatter for every pair and lays them out as an
**upper-triangular matrix**. When more than one scatter is active, the plots move into the footer
panel so the viewer stays visible.

### Brushing

Select cells with the **lasso** tool in the scatter toolbar (pan/zoom and box tools are available
too). The selection mirrors across all scatter views and, with the **Linked plugins** checkboxes on,
highlights the cells in the image and gallery. **Clear selection** resets it.

!!! note "Scatter in VS Code"
    In VS Code the scatter defaults to a static Matplotlib fallback. To force the interactive widget,
    set the environment variable `UELER_SCATTER_BACKEND=widget` before launching.

---

## Linked Brushing in Practice

Because both plugins publish to the same shared selection, you can gate in one and see the result
everywhere: brush a marker range in the histogram, and — with **Main viewer** and **Cell gallery**
linked — those cells light up in the tissue and populate the gallery. See
[Working with a Cell Table](cell-table.md#the-idea-linked-selection) for the full picture.
