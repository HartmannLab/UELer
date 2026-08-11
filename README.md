# UELer
Unified Exploratory Linked Viewer: a Jupyter-based framework for interactive exploration of multiplexed imaging datasets.

## Try it on Binder
You can try UELer without installation by launching it on [Binder](https://mybinder.org/v2/gh/HartmannLab/UELer/main?urlpath=%2Fdoc%2Ftree%2Fscript%2Frun_ueler_binder.ipynb):
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HartmannLab/UELer/main?urlpath=%2Fdoc%2Ftree%2Fscript%2Frun_ueler_binder.ipynb)

## Installation

### Option A — install from PyPI (recommended)

```shell
pip install ueler
```

While UELer is still on a pre-release version, ask pip for it explicitly:

```shell
pip install --pre ueler
```

This pulls in every runtime dependency. Two optional extras are available:

```shell
pip install "ueler[ark]"    # adds ark-analysis (pinned) for ark-based workflows
pip install "ueler[docs]"   # adds the mkdocs toolchain for building the docs
```

Requires Python 3.10 or 3.11.

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

## Updating Your Environment for v0.1.7-alpha (or earlier) users
If you're using UELer v0.1.7-alpha or earlier, you'll need to update your environment by following these steps:
1. Activate your environment:
```shell
micromamba activate <your-environment-directory>
```
2. Install Dask:
```
micromamba install dask
```
3. Install Dask-Image:
```
micromamba install dask-image
```
After completing these steps, your environment should be ready to go!

### Upgrade UELer
If you installed from PyPI:
```shell
pip install --upgrade ueler
```
If you installed from source, pull the latest commits in your UELer directory. Re-run the install
only when the dependencies changed — an editable install picks up code changes on its own:
```shell
git pull
pip install -e .   # only needed if env/environment.yml or pyproject.toml changed
```

## Getting started
1. Open your favorite editor that supports Jupyter notebook.
2. Open the starter notebook `script/run_ueler.ipynb`. If you installed from PyPI rather than
   cloning, download it from
   [the repository](https://github.com/HartmannLab/UELer/blob/main/script/run_ueler.ipynb).
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
### **UELer v0.5.0-alpha Summary**
- **Fixed: the "Plot all pairs" scatter matrix now follows the standard axis convention.** All plots in the same **row** share a y-axis and all plots in the same **column** share an x-axis, so you can compare across a row or down a column the way you would in any scatter-plot matrix. The grid previously did the opposite, which made the layout hard to read.
- **Fixed: painting masks by a cluster or cell-type column could fail with `Cannot interpret ... as a data type`.** It affected any identifier column that pandas keeps in one of its own dtypes rather than as plain Python objects — most importantly a **`category` column, which is what you get when the cell table comes from an AnnData / `.h5ad` file**, but also nullable-integer columns and, on pandas 3, ordinary text columns. Colouring by such a column raised an error instead of painting. Categorical columns of numbers are now matched by their underlying value, so selecting class `1` finds the cells in cluster `1` rather than silently finding none.
- **UELer is now BSD 3-Clause licensed (was GPL-3.0).** Changed before the first PyPI upload, deliberately: UELer is a package you *import*, and under the GPL any analysis code that imported it and was then distributed would have had to be GPL too. BSD-3 removes that — use, modify and redistribute UELer freely, including in commercial and closed-source work, as long as the copyright notice stays. It is the same license as `scikit-image`, `dask`, `bokeh`, `anndata` and `napari`, so UELer no longer imposes anything your existing scientific Python stack does not. **Nothing changes for existing users**, who gain permissions rather than lose them.
- **Every change is now tested automatically on Python 3.10 and 3.11 before it can be released.** UELer previously had no test workflow at all, so each release was validated on a single developer machine. Continuous integration now runs the full 922-test suite against the real dependency stack on both supported Python versions, and also builds the package, installs it into a clean environment and imports it there — the check that catches a missing bundled file, which is invisible when you build from your own working copy. A test that is *skipped* because an optional dependency is missing now fails the build rather than being counted as a pass; that had previously hidden a whole untested code path. Releases are published through an automated workflow that can only upload the exact artifact CI tested.
- **UELer is being prepared for release on PyPI (`pip install ueler`).** Installation now leads with a pip install rather than a `git clone`, and "Upgrade UELer" covers both paths — on the [documentation site](https://hartmannlab.github.io/UELer/) too, which until now told everyone to clone. The PyPI page will carry proper classifiers and links to the issue tracker and changelog. The supported Python versions are stated explicitly (**3.10 and 3.11**) instead of implied — the previous open-ended range would have let pip install UELer on interpreters it has never been tested against. Packaging fixes behind the scenes: bundled image assets can no longer be dropped from a build by an over-broad `.gitignore` rule, the source distribution no longer carries unusable test files, and `make build` / `make publish` targets always start from a clean `dist/`. The tool's one-line description is now consistent across the README, the docs site and the package metadata.
- **For developers working from a clone:** the test dependency stubs installed by `sitecustomize.py` / `usercustomize.py` are now **opt-in** via `UELER_TEST_BOOTSTRAP=1` (which `make test-fast` sets for you). Previously they were on by default, so any interpreter that had the repo root on `PYTHONPATH` could silently get stubbed versions of `pandas`, `matplotlib` and `ipywidgets`.
- **⚠️ Legacy `viewer` / `constants` / `data_loader` / `image_utils` imports have been removed (packaging cleanup).** Until now, `import ueler` also made the *old* pre-v0.2 module names importable, so a notebook could still say `from viewer.main_viewer import ImageMaskViewer`. That shim worked by claiming those four names for UELer across your whole Python session, which is not safe once UELer is installed from PyPI — a file of your own called `constants.py` or `data_loader.py` could be quietly shadowed by UELer's. The shim is gone. **If you have an old notebook using those names, change the import to the `ueler.` version** — `from ueler.viewer.main_viewer import ImageMaskViewer`, `import ueler.constants`, and so on. The modules themselves are unchanged; only the old spelling is no longer accepted. The `ensure_aliases=` argument of `run_viewer()` / `run_viewer_bia()` and the `ueler.ensure_compat_aliases()` helper are removed too; passing `ensure_aliases=` now prints a warning and is ignored rather than failing.

_Earlier changes (v0.4.4 and before) are in the [update log](https://github.com/HartmannLab/UELer/blob/main/doc/log.md)._

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
