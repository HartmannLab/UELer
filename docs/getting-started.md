# Get Started

This page walks you through launching UELer for the first time — either instantly in your browser,
or locally on your own data.

---

## Try It Instantly (no installation)

The fastest way to see UELer is on **Binder**. The demo notebook **streams a public
[BioImage Archive](https://www.ebi.ac.uk/bioimage-archive/) study directly over the network**, so
there is nothing to download and no data to configure — the viewer opens on a real multiplexed
imaging dataset.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HartmannLab/UELer/main?urlpath=%2Fdoc%2Ftree%2Fscript%2Frun_ueler_binder.ipynb)

Click the badge, wait for the environment to build, then run all cells. The first field of view is
fetched on demand — only the channels you open are streamed.

!!! note "First launch can be slow"
    Binder builds a fresh environment on first use, which may take a few minutes. Subsequent
    launches are faster while the image is cached.

---

## Run Locally

To work with your own data, run UELer inside a Jupyter notebook.

### Prerequisites

Make sure you have completed [installation](installation.md) and activated the environment:

```shell
micromamba activate ark-analysis-ueler
```

### 1. Open the Notebook

Open your notebook environment (JupyterLab, Jupyter Notebook, or VS Code), navigate to the cloned
UELer repository, and open:

```
script/run_ueler.ipynb
```

Select the **ark-analysis-ueler** kernel for the notebook.

### 2. Set Your Data Paths

Edit the paths cell near the top of the notebook. Only `base_folder` is required — masks,
annotations, and the cell table are all optional, and the viewer simply skips those features when a
path is not provided.

```python
base_folder = "/path/to/your/image_data"                     # Required
masks_folder = "/path/to/segmentation/output"                # Optional
annotations_folder = "/path/to/annotations"                  # Optional
cell_table_path = "/path/to/cell_table.csv"                  # Optional
```

| Variable | Purpose | Required |
|---|---|---|
| `base_folder` | FOV folders containing per-channel TIFF images | ✅ Yes |
| `masks_folder` | Segmentation `.tif` rasters | ❌ Optional |
| `annotations_folder` | Annotation raster `.tif` files | ❌ Optional |
| `cell_table_path` | Per-cell feature table (CSV) | ❌ Optional |

### 3. Launch the Viewer

UELer is launched from Python. Start with the minimal call and add data sources as you need them.

**Minimal — images only:**

```python
import ueler
%matplotlib widget

viewer = ueler.run_viewer(base_folder)
```

`run_viewer` also accepts `masks_folder=...` and `annotations_folder=...` if you want segmentation
and annotation overlays without a cell table.

**Full — masks, annotations, and a cell table:**

```python
import ueler
import pandas as pd
from ueler import load_cell_table
%matplotlib widget

# 1. Build the viewer with images + overlays (do not display yet)
viewer = ueler.run_viewer(
    base_folder,
    masks_folder=masks_folder,
    annotations_folder=annotations_folder,
    auto_display=False,
)

# 2. Attach the cell table, then display
cell_table = pd.read_csv(cell_table_path)
load_cell_table(viewer, cell_table=cell_table, auto_display=True, after_plugins=True)
```

!!! note "Why two steps?"
    `run_viewer` builds and displays the image and mask viewer. The cell table is attached
    **separately** with `load_cell_table`, which re-renders the UI with the linked scatter and
    histogram plots and the cell gallery enabled. If you have no cell table, skip step 2 and let
    `run_viewer` display on its own (`auto_display` defaults to `True`).

!!! tip "`%matplotlib widget`"
    The `%matplotlib widget` magic enables the interactive backend used by the viewer. Run it once
    per kernel session before launching.

---

## Stream Any BioImage Archive Study

You are not limited to local folders. `run_viewer_bia` streams a public BioImage Archive study
(`S-BIAD*`) without downloading the whole dataset — pass an accession id (or a direct HTTPS URL) and
an optional layout descriptor:

```python
import ueler
%matplotlib widget

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

See `script/run_ueler_BIA.ipynb` for more worked examples across several studies.

---

## Expected Folder Structure

For local data, UELer expects one subdirectory per FOV, each containing single-channel TIFF images
named by the channel/marker:

```
base_folder/
├── fov1/
│   ├── CD3.tiff
│   ├── CD8.tiff
│   └── DAPI.tiff
├── fov2/
│   ├── CD3.tiff
│   └── ...
└── ...
```

---

## Next Steps

- Learn the [User Interface](tutorials/user-interface.md) layout.
- Explore the [Tutorials](tutorials/index.md) for individual features.
- Check the [FAQ](faq.md) if you run into issues.
