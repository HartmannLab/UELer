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
   - *Done:* `ROI_COLUMNS` captures `annotation_palette`, `mask_color_set`, and `mask_visibility` payloads with CSV compatibility ensured via defaults.
   - *Done:* Capture/update flows serialize per-mask visibility (on/off) so ROI exports and imports retain mask toggles.

2. **Viewer integration hooks**
   - *Done:* `ImageMaskViewer` exposes `get_active_annotation_palette_set`/`apply_annotation_palette_set` plus new mask visibility getters/setters.
   - *Done:* Mask painter now surfaces `get_active_color_set_name`/`apply_color_set_by_name`, letting ROI restores pull registry-backed presets.
   - *Pending:* Broaden status logging around preset failures if we want more granular messaging beyond the existing missing-presets summary.

3. **ROI editor tab rework**
   - *Done:* Plugin renders `ROI browser`/`ROI editor` tabs, wires `Center with preset`, and surfaces palette plus mask visibility summaries in the editor metadata block.

4. **ROI browser implementation**
   - *Done:* Browser tab builds filters, throttles redraws via signature caching, offers OR/AND tag logic, and respects a new “Apply saved preset” checkbox.
   - *Done:* Gallery tiles render inside a 500 px scroll box at 98% width, hide titles, and expose a “Show more” control that loads four additional ROIs per request.

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
- ✅ Scale gallery subplots so each figure occupies 98% of the parent container width (no outer padding-induced shrinkage).
- ✅ Suppress figure-level titles and subplot titles in the gallery view.
- ✅ Add a checkbox that lets users choose between applying the ROI's saved preset or retaining the viewer's current preset when clicking a gallery tile.
- ✅ Avoid triggering a full browser refresh on every ROI click; updates now respond to filter or data changes only.
- ✅ Persist the full mask visibility state with each ROI record and restore it during preset application.
- ✅ Provide AND/OR logic for tag filtering to toggle between intersection and union semantics.
- ✅ Constrain the gallery container to a 500px max height with automatic vertical scrolling.
- ✅ Implement a "scroll down and show more" affordance that loads the next four ROI previews on demand.

## Current Action Plan (Cycle starting 2025-10-31)
1. ✅ **Mask state persistence**
   - ROI schema, capture/update flows, and preset restore now record mask visibility alongside palettes.
2. ✅ **Preset restore UX controls**
   - Browser checkbox toggles saved-vs-current preset usage; metadata summaries highlight palette and mask visibility context.
3. ✅ **Gallery rendering polish**
   - Gallery tiles scale within a 500 px scroll box, hide titles, and offer incremental loading in batches of four.
4. ✅ **Interaction throttling & filtering**
   - Signature caching avoids redraws on clicks, and tag logic toggles between AND/OR semantics.
5. 🔄 **Testing & documentation**
   - Add targeted unit tests (ROI CSV round trips, tag filter logic, pagination) and finalize README/release log updates.

## Validation Plan
- Run the ROI-related pytest modules (`tests/test_roi_manager_tags.py`, `tests/test_export_job.py`, plus any new tests).
- Exercise the plugin manually within `run_viewer.ipynb`, checking both tabs, filtering, and preset applications.
- Verify CSV import/export retains new metadata across sessions.
