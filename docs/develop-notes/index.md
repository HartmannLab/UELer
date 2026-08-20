# Developer Notes

These notes document the architecture, design decisions, and implementation details of UELer. They are intended for contributors and advanced users who want to understand how the viewer works internally.

---

## Topic Summaries

| Topic | Description |
|---|---|
| [Packaging & Project](packaging.md) | Package structure, release process, CI, PyPI |
| [Plugin Development](plugin-development.md) | Writing a plugin: discovery, lifecycle hooks, cross-plugin communication |
| [Viewer Runtime & UI](viewer-runtime.md) | FOV load cycle, plugin discovery, downsampling, channel controls |
| [Mask Rendering & Coloring](mask-rendering.md) | The three overlay layers, compositing order, the cell-colour registry |
| [Map Mode Internals](map-mode.md) | Stitched rendering, the tile budget, coordinate translation |
| [Export Pipeline](export-pipeline.md) | Batch export, scale bar, overlay snapshots |
| [ROI Workflows](roi-gallery.md) | ROI manager, the CSV schema, view vs. shape ROIs, gallery paging |
| [Heatmap & Cell Annotation](heatmap.md) | FlowSOM clustering, heatmap adapter, annotation checkpoints |
| [OME-TIFF Loading](ome-tiff.md) | OME-TIFF ingestion, level selection, rendering |

!!! tip "Start here to extend UELer"
    [Plugin Development](plugin-development.md) is the entry point for adding a feature. Almost every
    tool a user interacts with is a plugin, so it is usually the only page you need.

---

## Source Notes

Detailed source-level notes are kept in the `dev_note/` directory of the repository. The topic summaries above consolidate those notes for quicker navigation, and each page links its source note at the top.

These pages are checked against the code, not just spell-checked. `tools/check_docs_consistency.py` — run by `make check-docs`, by `tests/test_docs_consistency.py`, and in the docs workflow — fails the build when a page names a module, symbol, extra, Make target, environment variable or plugin label that no longer exists. Prose still has to be kept honest by hand; the factual scaffolding is enforced.
