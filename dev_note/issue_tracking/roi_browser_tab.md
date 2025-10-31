# ROI Browser & Editor Enhancements (Issue #44)

## Context
- Existing ROI manager plugin offers a single accordion pane focused on capture/update/export workflows.
- No gallery or filtering UX exists for browsing saved ROIs, and metadata retention is limited to marker sets, tags, and comments.
- Pixel annotation palettes and mask painter color sets are managed elsewhere (main viewer + dedicated plugin) with no linkage to ROI records.

## Goals
- Introduce an `ROI browser` tab with visual grid previews, tag/FOV filtering, and click-to-load handoff to the main viewer.
- Keep the current controls under an `ROI editor` tab and extend them with preset-aware centering plus palette/mask persistence.
- Ensure ROI records capture enough state (marker, annotation palette reference, mask painter set) to faithfully restore the saved view.
- Deliver regression-safe changes with automated coverage where practical.

## Progress & Next Steps
1. **Data model updates**
   - *Done:* `ROI_COLUMNS` now captures `annotation_palette` and `mask_color_set` identifiers and defaults them to empty strings for legacy CSV compatibility.
   - *Pending:* Persist the mask visibility state (per-mask on/off) with each ROI and thread it through capture/update/export.

2. **Viewer integration hooks**
   - *Done:* `ImageMaskViewer` exposes `get_active_annotation_palette_set` and `apply_annotation_palette_set`, updating active palette names when palettes are loaded or applied.
   - *Pending:* Mask painter still needs `get_active_color_set_name`/`apply_color_set_by_name` helpers; ROI preset restore should call into them once available.
   - *Pending:* Add explicit fallbacks/logging when preset application fails so centering continues gracefully.

3. **ROI editor tab rework**
   - *Done:* Plugin now renders `ROI browser` and `ROI editor` tabs, moves legacy controls under the editor tab, and adds `Center with preset` wiring for stored presets.
   - *Pending:* Surface palette/mask identifiers in the editor UI (badges or labels) to show what will be applied.

4. **ROI browser implementation**
   - *Done:* Browser tab builds tag and FOV filters, renders a Matplotlib gallery, and centers ROIs with presets on selection.
   - *Pending:* Implement throttling so the gallery does not rebuild on every click; honor user choice to use current preset vs saved preset; support AND/OR tag logic; paginate gallery (load next 4 on scroll request); cap gallery container height with scroll; scale figures to 98% width and remove figure/subplot titles.

5. **Testing & validation**
   - *Pending:* Add unit coverage for ROI serialization with new fields, palette apply helpers, tag filter logic, and pagination throttles.
   - *Pending:* Schedule manual notebook verification once remaining UI behaviours land.

6. **Documentation & release notes**
   - *Done:* README, log, and issue summary updated to describe the ROI browser/editor split and preset persistence.
   - *Pending:* Revise release notes once mask state persistence and browser UX refinements are complete.

## Open Questions / Follow-ups
- *Resolved:* Palette and mask sets use globally unique names and always persist under `.UELer`, so no per-project paths are stored.
- *Updated:* Gallery tiles will adopt dynamic sizing (98% parent width) per new request; confirm Matplotlib layout can accommodate without distorting axes.
- *Resolved:* Lazy rendering relies on the shared `ueler.rendering` engine, keeping the browser layer lightweight.

## New Requests (2025-10-31)
- Scale gallery subplots so each figure occupies 98% of the parent container width (no outer padding-induced shrinkage).
- Suppress figure-level titles and subplot titles in the gallery view.
- Add a checkbox that lets users choose between applying the ROI's saved preset or retaining the viewer's current preset when clicking a gallery tile.
- Avoid triggering a full browser refresh on every ROI click; updates should respond to filter or data changes only.
- Persist the full mask visibility state with each ROI record and restore it during preset application.
- Provide AND/OR logic for tag filtering to toggle between intersection and union semantics.
- Constrain the gallery container to a 500px max height with automatic vertical scrolling.
- Implement a "scroll down and show more" affordance that loads the next four ROI previews on demand.

## Validation Plan
- Run the ROI-related pytest modules (`tests/test_roi_manager_tags.py`, `tests/test_export_job.py`, plus any new tests).
- Exercise the plugin manually within `run_viewer.ipynb`, checking both tabs, filtering, and preset applications.
- Verify CSV import/export retains new metadata across sessions.
