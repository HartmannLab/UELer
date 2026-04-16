# Get Started

This page walks you through launching UELer for the first time.

---

## Prerequisites

Make sure you have completed [installation](installation.md) before proceeding.

---

## 1. Open the Notebook

1. Activate your environment:

    ```shell
    micromamba activate ark-analysis-ueler
    ```

2. Open your notebook environment (JupyterLab, Jupyter Notebook, or VS Code).
3. Navigate to the cloned UELer repository and open:

    ```
    script/run_ueler.ipynb
    ```

4. Select the **ark-analysis-ueler** kernel for the notebook.

---

## 2. Configure Your Data Paths

The notebook requires you to set a few paths at the top. Edit the configuration cell:

```python
base_folder = "/path/to/your/image_data"          # Required
masks_folder = "/path/to/segmentation/output"     # Optional
annotations_folder = "/path/to/annotations"        # Optional
cell_table_path = "/path/to/cell_table.csv"       # Optional
```

| Variable | Description | Required |
|---|---|---|
| `base_folder` | Directory containing the FOV folders with channel images | ✅ Yes |
| `masks_folder` | Directory containing segmentation `.tif` files | ❌ Optional |
| `annotations_folder` | Directory containing annotation raster `.tif` files | ❌ Optional |
| `cell_table_path` | Path to the CSV cell table | ❌ Optional |

!!! note "Minimal setup"
    You can run UELer with only `base_folder` set. Masks, annotations, and the cell table are all optional — the viewer will simply skip those features when the paths are not provided.

---

## 3. Run the Notebook

Run all cells in `run_ueler.ipynb`. The viewer will appear inline in the notebook output.

---

## 4. Expected Folder Structure

UELer expects your image data to follow this layout:

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

Each subdirectory is treated as one FOV (Field of View). Channel files must be single-channel TIFF images named by the channel/marker.

---

## 5. Try It on Binder

No local setup? Try UELer directly in your browser:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HartmannLab/UELer/binder-app?urlpath=%2Fdoc%2Ftree%2Fscript%2Frun_ueler.ipynb)

---

## Next Steps

- Learn the [User Interface](tutorials/user-interface.md) layout.
- Explore [Tutorials](tutorials/index.md) for more advanced features.
- Check the [FAQ](faq.md) if you run into issues.
