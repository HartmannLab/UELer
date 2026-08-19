# UELer
Unified Exploratory Linked Viewer: a Jupyter-based framework for interactive exploration of multiplexed imaging datasets.

## Try it on Binder
You can try UELer without installation by launching it on [Binder](https://mybinder.org/v2/gh/HartmannLab/UELer/main?urlpath=%2Fdoc%2Ftree%2Fscript%2Frun_ueler_binder.ipynb):
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HartmannLab/UELer/main?urlpath=%2Fdoc%2Ftree%2Fscript%2Frun_ueler_binder.ipynb)

## Installation

### Option A — install from TestPyPI (recommended)

**The install name is `ueler-viewer`; the import name is `ueler`.** PyPI administratively prohibits the name `ueler`, so the distribution ships as `ueler-viewer` — but nothing about using it changes, and `import ueler` stays exactly as it is. The same split as `scikit-image`/`skimage` and `opencv-python`/`cv2`.

Releases currently live on **TestPyPI** while UELer is in pre-release. Install from there, on one line — `--extra-index-url` is required, because TestPyPI does not mirror UELer's runtime dependencies:

```shell
pip install --pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ueler-viewer
```

`--pre` is needed while UELer is on a pre-release version. This pulls in every runtime dependency. Two optional extras are available:

```shell
# adds ark-analysis (pinned) for ark-based workflows
pip install --pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "ueler-viewer[ark]"
# adds the mkdocs toolchain for building the docs
pip install --pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "ueler-viewer[docs]"
```

Requires Python 3.10 or 3.11. Then, in Python:

```python
import ueler
```

#### If the install fails

- **`No matching distribution found for scikit-image>=0.19`** (or for any other dependency) — the resolver is only seeing TestPyPI, which hosts an empty `scikit-image` project. Keep the command on **one line**: the `--extra-index-url https://pypi.org/simple/` part is what lets the dependencies come from real PyPI, and it is easy to lose when a multi-line command is pasted.
- **Installing with `uv`** — uv's default `--index-strategy first-index` stops at the first index that lists a package at all, so it never falls back to PyPI for the dependencies. It needs an extra flag:

  ```shell
  uv pip install --prerelease=allow --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match ueler-viewer
  ```
- **You installed an earlier release under the old `ueler` distribution name** — run `pip uninstall ueler` first. Both distributions install the same `ueler/` package, and pip does not know they are the same project, so having both leaves two installs fighting over the same files.

### Option B — install from source (for development)

Use this if you want to modify UELer or track the `develop` branch.

1. Create a compatible environment from the `env/environment.yml` file in this repository:

   ```shell
   micromamba env create --name ark-analysis-ueler --file environment.yml
   ```
2. Clone the repository and activate the environment:

   ```shell
   git clone https://github.com/HartmannLab/UELer.git
   micromamba activate ark-analysis-ueler
   ```
3. Install in editable mode from the cloned directory:

   ```shell
   cd UELer
   pip install -e .
   ```

### Upgrade UELer
If you installed from TestPyPI:
```shell
pip install --upgrade --pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ueler-viewer
```
Coming from a release installed as `ueler` rather than `ueler-viewer`? Run `pip uninstall ueler` first — see [If the install fails](#if-the-install-fails).
If you installed from source, pull the latest commits in your UELer directory. Re-run the install only when the dependencies changed — an editable install picks up code changes on its own:
```shell
git pull
pip install -e .   # only needed if env/environment.yml or pyproject.toml changed
```

## Getting started
1. Open your favorite editor that supports Jupyter notebook.
2. Open the starter notebook `script/run_ueler.ipynb`. If you installed with pip rather than cloning, download it from [the repository](https://github.com/HartmannLab/UELer/blob/main/script/run_ueler.ipynb).
3. Select the kernel for an ark-analysis compatible conda/micromamba env.
4. Change the lines according to the instructions in the notebook: when configuring the `/script/run_ueler.ipynb`, ensure that you specify the following directory paths:
  - **`base_folder`**: The directory containing the FOV (Field of View) folders with image data (e.g., `.../image_data`).
  - **`masks_folder`** (optional): The directory containing the segmentation `.tif` files for cell segmentation (e.g., `.../segmentation/cellpose_output`).
  - **`annotations_folder`** (optional): The directory containing annotation files for marking regions of interest (e.g., `.../annotations`).
  - **`cell_table_path`** (optional): The path to the file containing the cell table data (e.g., `.../segmentation/cell_table/cell_table_size_normalized.csv`).
Make sure these paths are correctly set in the notebook for the viewer to access the data correctly.

5. Run the code and you will see the viewer displayed.

### Streaming from the BioImage Archive (BIA)
You can explore a public BioImage Archive study (an `S-BIAD*` accession) without downloading the
whole dataset first:
```python
from ueler.runner import run_viewer_bia

viewer = run_viewer_bia(
    "S-BIAD2557",                      # accession id (or a direct HTTPS base URL)
    descriptor={                        # optional; auto-detection is attempted if omitted
        "mode": "folder",
        "base": "Files/spatial_murine_iCCAvsHCC/image_data",
        "mask_dir": "Files/spatial_murine_iCCAvsHCC/segmentation/cleaned_mask",
        "mask_glob": "{fov}_*.tiff",
    },
)
```
Because BIA studies have no standard folder layout, a small JSON **descriptor** (a dict or a path
to a `.json` file) maps the study files onto FOVs / channels / masks; when omitted, UELer attempts
to auto-detect the folder-per-FOV, OME-TIFF-per-FOV, or zip-container layouts. The descriptor is
flexible enough for the variation seen across real studies:
- **Masks** accept either a single `mask_dir`/`mask_glob`, or a `masks` list of sources — each with
  an optional `name` (renames masks named `<fov>.tiff` to a clean label) or `per_fov: true` (masks
  stored in a per-FOV subfolder `<dir>/<fov>/*.tiff`). `annotations` uses the same shape.
- **Zipped FOVs**: set `"fov_container": "zip"` when each FOV is a `<FOV>.zip` of channel TIFFs —
  UELer reads a single channel straight out of the remote zip via an HTTP byte-range request rather
  than downloading the whole archive.

Pyramidal OME-TIFFs and single zip members are streamed via HTTP byte-range requests; other files
(e.g. single-resolution MIBI TIFFs) are downloaded once into a local cache. A per-study
**workspace** at `~/.ueler/bia/<accession>/` (override with `local_dir=`) holds your persistent
`.UELer` work (ROIs, checkpoints, palettes) plus a disposable `cache/` of downloaded images.

Examples for three real studies — `S-BIAD2557` (single-dir masks), `S-BIAD2864` (two named mask
folders), and `S-BIAD2708` (zipped FOVs + per-FOV masks) — are in `script/run_ueler_BIA.ipynb`.

## User interface
![GUI_preview](https://raw.githubusercontent.com/HartmannLab/UELer/main/doc/GUI_preview.png)
The GUI can be split into four main regions (wide plugins toggle the optional footer automatically):
- left: overall settings (channel, annotation, and mask accordions)
- middle: main viewer with overlay controls and image navigation
- right: plugin tools (Mask Painter, ROI Manager, palette editors, statistics panels)
- bottom (optional): wide plugin tabs (e.g., horizontal heatmap or gallery extensions)

### Overall Settings
- **Cache Size**: Defines the number of images that can be loaded into memory at one time.  
- **Select Image**: Choose an image to display in the main viewer.  
- **Channel Selection**: Select the channels you want to display. You can select multiple channels by holding down the **Shift** key and clicking.  
- **No image (masks only)**: When masks are available, enable this checkbox to skip rendering the image layer and show masks plus annotations on a black background. This also reduces image-compositing work when you only need spatial mask context.
- **Marker Set**: Load a pre-defined marker set, which includes channels, colors, and color ranges.
- **Control sections**: Channel, annotation, and mask controls now live in a collapsible accordion so you can jump straight to the section you need. When annotations are available, their controls appear ahead of masks, and each pane scrolls independently to keep the palette tools in reach even with dozens of channels.
- **Annotations**: When `<base_folder>/annotations` contains rasters named `<fov>_<annotation>.tif(f)`, enable the overlay toggle to color pixels by class. Choose between mask outlines, annotation fills, or a combined view, adjust fill opacity, and launch the palette editor to customize class colors and display labels. Annotation names can include spaces (for example, `Simple Segmentation`)—they remain selectable and the **Edit palette…** button now activates as soon as such an entry loads.
- **Masks**: Load segmentation rasters, edit per-class colours, and save or recall `.maskcolors.json` sets—default colours are tracked automatically, and optional `ipyfilechooser` dialogs speed up import/export.

### Tools & Plugins
- **Mask Painter**: Focus on edited classes, reuse colour sets, let inactive classes follow the global default fill mode, and restore saved per-class opacity, border, and filtered-list state without leaving the plugin.
- **ROI Manager**: Capture, centre, and tag regions of interest with persistent storage in `<base_folder>/.UELer/roi_manager.csv`; combo-box tagging keeps new labels available for future sessions.
- **Wide Plugins**: Enable "Horizontal layout" (for example, in the heatmap plugin) to undock the tool into the footer while keeping the accordion available for other controls.

## New Update  
### **UELer v0.5.0-alpha2 Summary**
- **Changed: pre-releases go to TestPyPI, stable releases will go to PyPI.** Which index a release lands on is now decided by the version itself: an `alpha` or `rc` is published to TestPyPI, and a plain `vX.Y.Z` is published to PyPI. Until the first stable release, [Installation](#installation) is still the TestPyPI command; after it, `pip install ueler-viewer` will be all you need. A stable release is also required to be the promotion of a release candidate you could already install from TestPyPI, unchanged apart from its version number — so nothing reaches PyPI that was not installable first.
- **Changed: install UELer as `ueler-viewer` — but keep importing it as `ueler`.** PyPI does not allow the project name `ueler` (most likely because it is one letter-swap away from the existing `euler` package), so releases are published as **`ueler-viewer`**. Only the `pip install` line changes: `import ueler` and every API, notebook and script stay exactly as they are, the same way you install `scikit-image` but import `skimage`. If you already installed an earlier release, run `pip uninstall ueler` before installing `ueler-viewer` — both provide the same `ueler` package, and pip cannot tell they are the same project. See [Installation](#installation) for the full command, including a fix for the `No matching distribution found for scikit-image>=0.19` error some installs hit.
- **New: a [Display Settings](https://github.com/HartmannLab/UELer/blob/main/docs/tutorials/display-settings.md) page in the documentation.** The viewer opens with defaults that suit a MIBI dataset, and one of them is worth checking before you export anything: **Pixel Size (nm)** defaults to 390, the MIBI pixel pitch, and it drives the scale bar in the viewer *and* in every exported image — so on IMC or any other platform, every figure you export carries a wrong scale bar unless you change it. The new page is a setup walkthrough rather than another control reference: what to set first and why, where the automatic contrast range comes from (the 99.9th percentile, which is why a channel with one bright artefact looks all-black), what **Downsample** actually does, how to size the cache against your memory, and how to reset a session that came back in a strange state.
- **Changed: the interactive scatter plot is now the default in every environment, VS Code included.** The Scatter plot plugin used to fall back to a static Matplotlib figure whenever it detected VS Code, so VS Code users lost linked brushing unless they knew to set `UELER_SCATTER_BACKEND=widget` before launching. You no longer need that line — delete it from your notebooks. The static renderer is still there if you want it, now as an opt-out: set `UELER_SCATTER_BACKEND=static` before launching the viewer.
- **New: you can draw line and polygon ROIs again, and they live in the ROI manager.** Open the *ROI editor* tab and use the **Line / polygon ROI** block: **Draw**, then click on the image to place vertices, drag one to move it, right-click to delete it, and **Save shape** when you are done — **Finish** ends the drawing without saving if you want to look at it first, and the shape stays on the canvas either way. Tick **Closed shape (polygon)** for a closed outline, and read the vertex count and the length or perimeter — in pixels and µm — under the buttons. Shapes are ordinary ROIs, so they take names, tags and comments, appear in the ROI browser with their path drawn on the thumbnail, and travel through ROI CSV import and export. In batch export they render as their bounding-box region, and a new **Include line/polygon ROIs** checkbox leaves them out when you only want your saved views. **Show shapes on canvas** draws every saved shape of the current FOV or map over the image. This restores the old *Region annotation* plugin, which was dropped by the package restructure that shipped in v0.4.0.
- **New: the cells you select in the main viewer can now be shown in the plots.** Tick **Follow main viewer** in the *Linked plugins* tab of the Scatter plot, Histogram or Heatmap plugin, and the cells you click, ctrl-click or lasso in the image are selected in that plot as you go — points in the scatter, a "Selected" distribution in the histogram, the corresponding clusters in the heatmap. It works across FOVs in map mode, only in the plots you switch it on for, and it leaves your image selection untouched. This is the continuous version of the **Trace** button, which does the same thing once for the current FOV. In the histogram, a selection of just a few cells would otherwise draw a bar too short to see, so when that happens the bins holding your cells are tinted over their whole height instead — switch it off with **Mark faint selections** in the *Histogram* tab.

_Earlier changes (v0.5.0-alpha1 and before) are in the [update log](https://github.com/HartmannLab/UELer/blob/main/doc/log.md)._

## Earlier Updates  

You can find previous update logs [here](https://github.com/HartmannLab/UELer/blob/main/doc/log.md).

## License
UELer is released under the **BSD 3-Clause License** — see
[LICENSE.txt](https://github.com/HartmannLab/UELer/blob/main/LICENSE.txt).

You are free to use, modify and redistribute UELer, including in commercial and closed-source
work, provided you keep the copyright notice and do not use the authors' names to endorse a
derived product. This is the same license as `scikit-image`, `dask`, `bokeh`, `anndata` and
`napari`, so UELer imposes no constraints your existing scientific Python stack does not.

If you use UELer in published work, a citation is appreciated but not required.

## Issues and contact
Bug reports and feature requests: [GitHub Issues](https://github.com/HartmannLab/UELer/issues).
Maintained by Yu-Le Wu, Hartmann Lab, DKFZ Heidelberg.
