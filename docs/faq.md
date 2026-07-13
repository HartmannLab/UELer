# FAQ

---

## General

### What imaging platforms does UELer support?

UELer is designed primarily for **MIBI** (Multiplexed Ion Beam Imaging) data but works with any
multiplexed imaging platform (e.g. IMC) that produces per-channel TIFF files organized in per-FOV
subdirectories. It also loads **OME-TIFF** files and is compatible with data produced by
[ark-analysis](https://github.com/angelolab/ark-analysis).

### Do I need a cell table to use UELer?

No — the cell table is optional. Without one, UELer runs as an image viewer: multi-channel rendering,
mask/annotation overlays, ROI capture, and batch export all work, and the right panel shows just the
**ROI manager** and **Batch export** plugins.

Loading a cell table unlocks the single-cell features — scatter plots, histograms, the cell gallery,
heatmaps, clustering, and annotation. See [Working with a Cell Table](tutorials/cell-table.md).

### Does UELer use a GPU?

No. UELer is **CPU-only** — it uses NumPy and Dask for image processing and does not use or require a
GPU. A GPU is not supported.

---

## Getting Started & Data

### Can I try UELer without my own data?

Yes. The [Binder demo](getting-started.md#try-it-instantly-no-installation) streams a public
[BioImage Archive](https://www.ebi.ac.uk/bioimage-archive/) study over the network — no installation
and no local data required.

### How do I load a cell table?

UELer builds the viewer first and attaches the cell table in a second step:

```python
import ueler
import pandas as pd
from ueler import load_cell_table
%matplotlib widget

viewer = ueler.run_viewer(base_folder, masks_folder=masks_folder, auto_display=False)

cell_table = pd.read_csv(cell_table_path)
load_cell_table(viewer, cell_table=cell_table, auto_display=True, after_plugins=True)
```

`run_viewer` itself has no `cell_table_path` argument — the table is always attached with
`load_cell_table`. See [Get Started](getting-started.md) and
[Working with a Cell Table](tutorials/cell-table.md).

### How do I stream a study from the BioImage Archive?

Use `run_viewer_bia` with an accession id (or a direct HTTPS URL) and an optional layout descriptor:

```python
viewer = ueler.run_viewer_bia(
    "S-BIAD2557",
    descriptor={
        "mode": "folder",
        "base": "Files/spatial_murine_iCCAvsHCC/image_data",
        "mask_dir": "Files/spatial_murine_iCCAvsHCC/segmentation/cleaned_mask",
        "mask_glob": "{fov}_*.tiff",
    },
)
```

Only the channels you open are streamed/cached. See `script/run_ueler_BIA.ipynb` for more examples.

---

## Installation

### I get `ModuleNotFoundError: No module named 'ueler'`

Activate your environment and run `pip install .` (or `pip install -e .`) from the root of the cloned
repository.

### `pip install` fails with a dependency error

Make sure the correct `micromamba`/conda environment is active first. Create it from the repository's
`environment.yml`, then install UELer. For documentation or test extras, use `pip install ".[docs]"`
or `pip install ".[dev]"`.

---

## Viewer & Widgets

### Nothing renders / the widgets don't show up

Run `%matplotlib widget` once per kernel session before launching the viewer — it enables the
interactive backend UELer relies on. If widgets still don't appear, restart the kernel and re-run the
cells.

### The scatter plot is not shown in VS Code

In VS Code the scatter plugin defaults to a **static Matplotlib fallback** (interactive
`jupyter-scatter` widgets don't always render there). To force the interactive widget, set the
environment variable before launching:

```python
import os
os.environ["UELER_SCATTER_BACKEND"] = "widget"
```

JupyterLab and the classic Notebook use the interactive backend by default.

### The histogram says it requires Bokeh

The Histogram plugin renders with Bokeh. Install `bokeh` and `jupyter_bokeh` (both are UELer
dependencies) and restart the kernel. UELer auto-loads BokehJS on first plot, so it renders in VS Code
without a priming cell.

### The viewer is blank or does not update after launching

1. Run cells one at a time to isolate the issue.
2. Restart the kernel and run all cells again.
3. Check that your data paths are correct.

### Channel images appear all-white or all-black

This is usually a contrast issue. Expand the **Channels** accordion and adjust the **Min** / **Max**
contrast sliders for the affected channel.

### How do I enable Map Mode?

Map mode is opt-in. Set the `ENABLE_MAP_MODE` environment variable to a truthy value (`1`, `true`,
`yes`, `on`) before launching, and place map descriptor JSON files under `<base_folder>/.UELer/maps/`.
When both are present, the **Map mode** toggle and **Select map:** dropdown appear. See
[Map Mode](tutorials/map-mode.md).

### Map mode renders slowly for large datasets

Map mode limits how many *uncached* tiles it draws per frame (default 80), keeping the ones nearest
the viewport. You can lower this on the viewer object:

```python
viewer._map_render_tile_limit = 40  # lower for slower systems
```

---

## Single-Cell Analysis

### FlowSOM won't run

The FlowSOM plugin needs the optional `pyFlowSOM` package. If it isn't installed, the rest of UELer
still works — only clicking **Run** in the FlowSOM plugin raises `pyFlowSOM is required to run the
FlowSOM plugin`. Install `pyFlowSOM` and restart the kernel.

### Which features need a cell table?

Scatter plot, Histogram, Gallery, Heatmap, FlowSOM, Cell Annotation, Mask painter, Go to, Cell Table
Editor, and the cell tooltip labels all require a cell table. ROI manager and Batch export work
without one. See [Working with a Cell Table](tutorials/cell-table.md).

---

## ROI, Export & Files

### Where does UELer store my work?

Per-dataset state lives in a `.UELer/` folder inside your `base_folder`:

| Path | Contents |
|---|---|
| `.UELer/roi_manager.csv` | Saved ROIs |
| `.UELer/widget_states.json` | Remembered UI/widget state |
| `.UELer/export_configs/` | Saved batch-export config templates |
| `.UELer/dataset_<id>/checkpoints/` | Cell Annotation checkpoints (`.h5ad`) |
| `.UELer/maps/` | Map descriptors (for [map mode](tutorials/map-mode.md)) |

Mask and annotation palettes are also saved under `.UELer/`.

### The scale bar is missing in my exports (or the viewer)

The scale bar is driven by pixel size. Set the **Pixel Size (nm):** value in the left panel's Advanced
Settings (see the [User Interface](tutorials/user-interface.md#left-panel) reference). If no pixel size
is available, the scale bar is omitted.
