# UELer

**Usability Enhanced Linked Viewer** — a Jupyter Notebook-integrated viewer for MIBI images with linked interactive plots and enhanced usability.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HartmannLab/UELer/binder-app?urlpath=%2Fdoc%2Ftree%2Fscript%2Frun_ueler.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-HartmannLab%2FUELer-blue?logo=github)](https://github.com/HartmannLab/UELer)

---

## What is UELer?

UELer is an interactive image viewer designed for **multiplexed imaging** data (MIBI, IMC, and similar technologies). It runs directly inside Jupyter notebooks and provides:

- **Linked, interactive visualizations** — scatter plots, heatmaps, and gallery views are synchronized with the spatial image display.
- **Multi-channel rendering** — visualize and compare channels with per-channel color and contrast controls.
- **Segmentation overlays** — view and paint cell segmentation masks, annotation overlays, and custom color sets.
- **ROI Manager** — capture, label, and export regions of interest with persistent storage.
- **Batch export** — export full FOVs, ROIs, and single-cell crops to PNG or PDF, with optional scale bars and overlays.
- **Map mode** — stitch multiple FOVs into a single spatial overview with full interactive navigation.
- **OME-TIFF support** — load and render OME-TIFF files alongside standard TIFF directories.

---

## Try It Without Installation

You can launch UELer in your browser via Binder — no local setup required:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HartmannLab/UELer/binder-app?urlpath=%2Fdoc%2Ftree%2Fscript%2Frun_ueler.ipynb)

---

## Quick Navigation

<div class="grid cards" markdown>

-   :material-download: **[Installation](installation.md)**

    Set up UELer using `micromamba` and `pip`.

-   :material-rocket-launch: **[Get Started](getting-started.md)**

    Configure and launch your first viewer session.

-   :material-school: **[Tutorials](tutorials/index.md)**

    Step-by-step guides for core features.

-   :material-help-circle: **[FAQ](faq.md)**

    Answers to common questions.

-   :material-code-braces: **[Developer Notes](develop-notes/index.md)**

    Architecture notes and design decisions.

</div>

---

## Supported Data Formats

| Format | Description |
|---|---|
| TIFF directory | Standard per-channel TIFFs organized in `<base_folder>/<fov>/` |
| OME-TIFF | Multi-channel OME-TIFF files with embedded metadata |
| CSV cell table | Per-cell feature tables (e.g., from `ark-analysis`) |
| Segmentation masks | Single-channel TIFF rasters for cell segmentation |
| Annotation masks | Per-class TIFF rasters for region annotation |

---

## License

UELer is released under the [GPL-3.0 license](https://github.com/HartmannLab/UELer/blob/main/LICENSE.txt).
