# Working with a Cell Table

A **cell table** is a per-cell feature table (a CSV or an AnnData object, e.g. from `ark-analysis` or
`scanpy`) with one row per segmented cell. Loading one turns UELer from an image viewer into a linked
single-cell explorer. This page explains how to load a table, the idea of **linked selection** that
ties everything together, and the tools that operate on the current selection.

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
analytical plugins appear — most in the right-panel accordion (**Histogram**, **Gallery**,
**FlowSOM**, **Cell Annotation**, **Mask painter**, **Go to**, **Cell Table Editor**, **Cell tooltip
label**), and two in the wide footer below the viewer (**Scatter plot** and **Heatmap**, which need
the full window width). See the [User Interface](user-interface.md#right-panel-plugins) reference.

---

## AnnData Input

An `AnnData` object works just as well as a DataFrame — pass the object, or point
`cell_table_path` at an `.h5ad` file:

```python
import anndata as ad
from ueler import load_cell_table

load_cell_table(viewer, cell_table=ad.read_h5ad("cells.h5ad"), auto_display=True)

# equivalent, straight from disk:
load_cell_table(viewer, cell_table_path="cells.h5ad", auto_display=True)
```

**What becomes selectable.** AnnData keeps metadata and measurements apart; UELer offers both:

| From the AnnData | Appears as | Where you'll use it |
|---|---|---|
| `obs` columns | metadata columns | **Class:**, **Subset on:**, mask-painter identifier, tooltip labels |
| `X` (or a `layers` entry), named by `var_names` | one column per marker | marker/channel pickers, heatmap, FlowSOM features |
| `obsm` arrays up to 3 columns wide, e.g. `X_umap` | `X_umap1`, `X_umap2` | scatter **X:** / **Y:** axes |
| `obs_names` | an `obs_names` column | cell identity, tooltips |

Markers are listed **first** in the marker pickers, so they aren't buried under `label`, `area`,
`X` and `Y`.

Your `obs` must still contain the FOV and mask-label columns that link cells to the images — by
default `fov` and `label`. If yours are named differently, set the **Fov key:** / **Label key:**
fields in the left panel (or rename the columns in `obs` beforehand).

Two optional arguments:

- `layer="counts"` — read `adata.layers["counts"]` instead of `adata.X`.
- `obsm_keys=["X_pca"]` — also expose an `obsm` entry that is too wide to be included by default.

### Getting your annotations back out

The object you passed in is kept, and anything the plugins add to the table — FlowSOM clusters,
heatmap meta-clusters, manual labels from the **Cell Table Editor** — is written back into its `obs`:

```python
adata = viewer.get_cell_table_adata()
adata.obs.columns          # now includes e.g. 'FlowSOM_cluster'
adata.write_h5ad("cells_annotated.h5ad")
```

This also works for a CSV/DataFrame table: `get_cell_table_adata()` then builds an AnnData for you,
using the numeric non-key columns as `X`.

---

## The Idea: Linked Selection

Most single-cell features in UELer revolve around one shared concept — a **current selection** of
cells. Whenever you select cells in one place, every linked view updates:

```
Scatter / Histogram / Heatmap ──▶  selected cells  ──▶  highlighted in the image
       ▲                      └────────────────────▶  shown in the Gallery
       │
       └── "Follow main viewer"  ◀──  click / ctrl-click / lasso in the image
```

The link is opt-in in both directions, and each direction has its own checkbox on the plot's
**Linked plugins** tab:

- **Outward** (plot → rest of the UI): tick **Main viewer** to highlight the selected cells' masks in
  the image, and **Cell gallery** to populate the gallery. This is what makes "gate on a marker, see
  those cells light up in the tissue" work.
- **Inward** (image → plot): tick **Follow main viewer** so that cells you pick in the image appear
  highlighted in the plot instead. Useful in reverse — click an odd-looking cell in the tissue and see
  where it falls in the marker distribution.

All of these start **off**, so a freshly loaded viewer shows no linking until you ask for it. See
[Scatter & Histogram](scatter-histogram.md#linking-a-plot-to-the-rest-of-the-ui) for the full table.

The selection spans the whole cell table, not just the FOV on screen, so **switching FOV keeps it**:
the image re-highlights the selected cells belonging to the FOV you moved to. Zooming and panning
keep the highlight as well. Selections made directly in the image — a click or a **lasso** — are
spatial and stay with the FOV they were drawn in.

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
