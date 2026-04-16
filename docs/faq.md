# FAQ

---

## General

### What imaging platforms does UELer support?

UELer is designed primarily for **MIBI** (Multiplexed Ion Beam Imaging) data but works with any multiplexed imaging platform that produces per-channel TIFF files organized in per-FOV subdirectories. It also supports **OME-TIFF** files and is compatible with data produced by [ark-analysis](https://github.com/angelolab/ark-analysis).

---

### Do I need a cell table to use UELer?

No. The cell table is optional. Without it, UELer runs in **simple viewer mode**, which still provides full multi-channel rendering, mask/annotation overlays, ROI capture, and batch export. Features that require per-cell data (scatter plot, heatmap, single-cell gallery) are hidden in this mode.

---

### Can I use UELer without GPU?

Yes. UELer uses NumPy and Dask for image processing and does not require a GPU.

---

## Installation

### `pip install -e .` fails with a dependency error

Make sure you have activated the correct `micromamba`/conda environment before running `pip install`. The `environment.yml` in the repository pins all required dependencies.

If a specific package is missing, install it manually:

```shell
micromamba install <package-name>
```

### I get `ModuleNotFoundError: No module named 'ueler'`

Run `pip install -e .` from the root of the cloned repository while your environment is active.

---

## Viewer

### The scatter plot is not shown in VS Code

VS Code does not always render `jupyter-scatter` widgets. UELer automatically falls back to a static Matplotlib figure in this case. The plot is still interactive via the chart plugin controls.

If you want the full interactive scatter, use JupyterLab or a standard Jupyter Notebook front-end.

---

### The viewer is blank or does not update after launching

This can happen if the notebook kernel is slow to initialize. Try:

1. Running cells one at a time to isolate the issue.
2. Restarting the kernel and running all cells again.
3. Checking that your data paths are correct.

---

### Channel images appear as all-white or all-black

This is usually a contrast issue. Expand the **Channel** accordion and adjust the minimum/maximum contrast sliders for the affected channel.

---

### Map mode renders slowly for large datasets

Map mode caps the number of tiles rendered per frame (default: 80). You can adjust this in the viewer:

```python
viewer.map_render_tile_limit = 40  # lower for slower systems
```

Renders during widget state restoration at startup are suppressed to avoid kernel timeouts.

---

## ROI Manager

### ROI names show "nan" after reloading

This is a known issue that was fixed in v0.3.1. Update UELer with `git pull` and reinstall (`pip install -e .`). The fix sanitizes empty `fov` values that pandas reads back as `NaN` from CSV.

---

### The batch export ROI list does not update after capturing a new ROI

This was fixed in v0.3.1. The batch export plugin now observes the ROI table and refreshes automatically. Update UELer and restart the viewer.

---

## Export

### Exported images are all-black or all-white

For single-FOV exports: check that the selected marker set has valid contrast ranges (not all-zero).

For map-mode ROI exports: this was fixed in v0.3.1. The exporter now uses `channel_settings` from the marker profile directly, bypassing UI widget state.

---

### The scale bar is missing in my exports

Scale bars require pixel size metadata. Make sure your data provides this information (e.g., via OME-TIFF metadata or ark-analysis output). If not available, the scale bar is omitted silently.
