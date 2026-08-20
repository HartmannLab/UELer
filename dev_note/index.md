# Dev Note Index

This index maps all existing dev notes to the new topic-oriented summaries. The topic files are concise syntheses; the original notes remain the source of truth for detailed context. Issue-tracking notes were consolidated into the topic summaries and removed from the repository.

## Topic summaries
- [dev_note/topic_plugin_development.md](dev_note/topic_plugin_development.md)
- [dev_note/topic_packaging_and_project.md](dev_note/topic_packaging_and_project.md)
- [dev_note/topic_viewer_runtime_ui.md](dev_note/topic_viewer_runtime_ui.md)
- [dev_note/topic_map_mode_spatial.md](dev_note/topic_map_mode_spatial.md)
- [dev_note/topic_ome_tiff_loading.md](dev_note/topic_ome_tiff_loading.md)
- [dev_note/topic_heatmap_flowsom_cell_annotation.md](dev_note/topic_heatmap_flowsom_cell_annotation.md)
- [dev_note/topic_roi_gallery_expression.md](dev_note/topic_roi_gallery_expression.md)
- [dev_note/topic_export_pipeline.md](dev_note/topic_export_pipeline.md)
- [dev_note/topic_mask_rendering_highlighting_coloring.md](dev_note/topic_mask_rendering_highlighting_coloring.md)

## Source note mapping

### Packaging and project structure
- [dev_note/release_procedure.md](dev_note/release_procedure.md) — the runbook: how to publish a pre-release or a stable release, what each refusal means, and how to recover
- [dev_note/Packaging_plan.md](dev_note/Packaging_plan.md)
- [dev_note/Todos.md](dev_note/Todos.md)

### Viewer runtime and UI behavior
- [dev_note/FOV_load_cycle.md](dev_note/FOV_load_cycle.md)
- [dev_note/main_viewer.md](dev_note/main_viewer.md)


### Map mode and spatial navigation
See the topic summary for consolidated map-mode notes and issue references.

### OME-TIFF and data loading
- [dev_note/ome_tiff_loading.md](dev_note/ome_tiff_loading.md)


### Heatmap, FlowSOM, and cell annotation
- [dev_note/Cell_annotation.md](dev_note/Cell_annotation.md)


### ROI workflows and gallery behavior
- [dev_note/gallery_width.md](dev_note/gallery_width.md)


### Export pipeline and scale bar
See the topic summary for consolidated export notes and issue references.

### Mask rendering, highlighting and coloring
See the topic summary for the three overlay layers, their compositing order and the cell-colour registry.

## Documentation site

Every topic summary above has a published counterpart under `docs/develop-notes/`, and the two are kept in step by hand — the topic note is the source, the docs page is the distillation, and each page links its source at the top.

`tools/check_docs_consistency.py` (run by `make check-docs`, `tests/test_docs_consistency.py`, and the docs workflow) enforces the *checkable* half of that relationship: a docs page naming a module, symbol, extra, Make target, environment variable or UI label that no longer exists fails the build. It has already caught an error in a topic note itself, so run it after editing either side.

- [dev_note/issue_tracking/docs_consistency_audit.md](dev_note/issue_tracking/docs_consistency_audit.md) — the 2026-08-20 audit that introduced the checker: what had drifted, why the checks are scoped the way they are, and what was deliberately left alone
