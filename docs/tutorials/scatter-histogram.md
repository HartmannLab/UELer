# Scatter & Histogram

The **Scatter plot** and **Histogram** plugins are UELer's linked distribution tools. Both require a
[cell table](cell-table.md), share the same channel picker, and publish a selection that can drive the
image and the [cell gallery](cell-table.md#the-cell-gallery).

They sit in different places: the **Histogram** is in the right-panel accordion, while the **Scatter plot** is allocated permanently to the wide footer beneath the viewer, since a scatter matrix is unreadable at side-panel width.

## Linking a plot to the rest of the UI

Both plugins have a **Linked plugins** tab, and the three checkboxes on it point in two different
directions. Getting them the wrong way round is the usual reason a link "doesn't work":

| Control | Direction | Effect |
|---|---|---|
| **Main viewer** | plot → image | This plot's selection highlights the matching masks in the image |
| **Cell gallery** | plot → gallery | This plot's selection fills the [Gallery](cell-table.md#the-cell-gallery) with thumbnails |
| **Follow main viewer** | image → plot | Cells you select *in the image* (click, ctrl-click, lasso) are highlighted in this plot |

All three are **off** by default. **Follow main viewer** is the exact counterpart of **Main viewer**,
not a duplicate of it: one pushes the plot's selection outward, the other pulls the image's selection
in. It is continuous — for a one-shot version, the Scatter plot's **Trace** tab does the same pull a
single time.

---

## Histogram

The **Histogram** plugin draws one distribution per selected channel, rendered with Bokeh.

1. Pick channels in the **Channels:** field — the same searchable picker as the left panel, with the
   same filter box, **Select all shown** / **Clear** buttons and keyboard navigation. You can also
   load a marker set's channels via the **Marker set:** dropdown + **Load set**; that fills this
   plugin's picker only and does not repaint the main image.
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
**upper-triangular matrix**.

### Brushing

Select cells with the **lasso** tool in the scatter toolbar (pan/zoom and box tools are available
too). The selection mirrors across all scatter views and, with the **Linked plugins** checkboxes on,
highlights the cells in the image and gallery. **Clear selection** resets it.

The selection is a set of cells, not a set of pixels, so it **survives a FOV change**: switch FOV in
**Select Image** and the selected cells belonging to the newly loaded FOV are outlined right away
(#119). A selection with no cells in the new FOV simply shows nothing until you switch to one that
has some. Zooming and panning keep the outlines too — including while a finer resolution level is
being loaded.

!!! note "Static scatter fallback"
    The interactive `jupyter-scatter` widget is the default everywhere, VS Code included — no environment variable needed. If you want the static Matplotlib scatter instead, set `UELER_SCATTER_BACKEND=static` before launching; the plot then carries an inline notice saying so.

---

## Linked Brushing in Practice

Because both plugins publish to the same shared selection, you can gate in one and see the result
everywhere: brush a marker range in the histogram, and — with **Main viewer** and **Cell gallery**
linked — those cells light up in the tissue and populate the gallery. See
[Working with a Cell Table](cell-table.md#the-idea-linked-selection) for the full picture.
