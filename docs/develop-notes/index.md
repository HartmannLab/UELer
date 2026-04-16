# Developer Notes

These notes document the architecture, design decisions, and implementation details of UELer. They are intended for contributors and advanced users who want to understand how the viewer works internally.

---

## Topic Summaries

| Topic | Description |
|---|---|
| [Packaging & Project](packaging.md) | Package structure, shims, and release notes |
| [Viewer Runtime & UI](viewer-runtime.md) | FOV load cycle, channel controls, downsampling, tooltips |
| [Map Mode Internals](map-mode.md) | Stitched rendering, tile cache, coordinate translation |
| [Export Pipeline](export-pipeline.md) | Batch export, scale bar, overlay snapshots |
| [ROI Workflows](roi-gallery.md) | ROI manager, gallery sizing, pagination |
| [Heatmap & Cell Annotation](heatmap.md) | FlowSOM clustering, heatmap adapter, annotation palette |
| [OME-TIFF Loading](ome-tiff.md) | OME-TIFF ingestion, level selection, rendering |

---

## Source Notes

Detailed source-level notes are kept in the `dev_note/` directory of the repository. The topic summaries above consolidate those notes for quicker navigation.
