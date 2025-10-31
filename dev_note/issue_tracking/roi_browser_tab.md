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

## Proposed Approach
1. **Data model updates**
   - Extend `ROI_COLUMNS` and related helpers to store optional `annotation_palette` and `mask_color_set` identifiers alongside the existing marker set; both map to globally unique names managed under `.UELer`.
   - Update capture/update flows to record the currently active annotation palette & mask color set (collect from viewer + mask painter plugin).
   - Propagate new fields through CSV import/export; default to empty strings for backward compatibility.

2. **Viewer integration hooks**
   - Expose helper methods on the main viewer to (a) resolve the active annotation palette selection and (b) apply a saved palette by registry name without mutating UI state unexpectedly; palette persistence will always use the shared `.UELer` path.
   - Add a similar programmatic entry point to the mask painter plugin to apply a saved color set by name and report whether it succeeded; color sets are likewise stored at the fixed `.UELer` location.
   - Define safe fallbacks when requested presets are missing (log status + proceed with centering only).

3. **ROI editor tab rework**
   - Wrap existing ROI manager content in a `Tab` widget with two pages: `ROI browser` (new) and `ROI editor` (modernized existing layout).
   - Insert a `Center with preset` button that triggers centering plus preset/palette/mask application.
   - Surface saved palette/mask identifiers inside the editor (read-only badges or dropdown sync) for transparency.

4. **ROI browser implementation**
   - Build filter controls (e.g., multi-select TagsInput for tags, dropdown/multi-select for FOVs, optional text search).
   - Render ROI previews via Matplotlib subplots using the same downsampling mechanics as the main viewer, leveraging stored marker set plus palette/mask info where available; gracefully degrade when data is incomplete.
   - Handle selection clicks by focusing the main viewer on the ROI and ensuring rendering presets mirror the gallery tile.
   - Keep gallery responsive to ROI table changes and filter tweaks; rely on `ueler.rendering` for lazy data fetching and add UI-level debouncing if needed.

5. **Testing & validation**
   - Add unit tests covering ROI manager data serialization with new columns and the preset application helpers.
   - Where feasible, add widget-level tests (e.g., verifying filter logic, preset application stub interactions) using existing pytest infrastructure.
   - Manually verify gallery rendering in a notebook due to UI dependencies.

6. **Documentation & release notes**
   - Update `README.md`, `doc/log.md`, and `dev_note/github_issues.md` summaries after implementation to reflect the new UX and persistence behaviour.

## Open Questions / Follow-ups
- *Resolved:* Palette and mask sets use globally unique names and always persist under `.UELer`, so no per-project paths are stored.
- *Resolved:* Gallery tiles use Matplotlib subplots with the viewer's downsampling presets; no additional size controls planned initially.
- *Resolved:* Lazy rendering relies on the shared `ueler.rendering` engine, keeping the browser layer lightweight.

## Validation Plan
- Run the ROI-related pytest modules (`tests/test_roi_manager_tags.py`, `tests/test_export_job.py`, plus any new tests).
- Exercise the plugin manually within `run_viewer.ipynb`, checking both tabs, filtering, and preset applications.
- Verify CSV import/export retains new metadata across sessions.
