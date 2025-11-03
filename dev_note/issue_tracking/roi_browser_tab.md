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
   - *Done:* Gallery tiles render inside a 500px scroll box at 98% width, hide titles, and page through 3x4 slices via Previous/Next controls instead of the old show-more loader.

5. **Testing & validation**
   - *Pending:* Add unit coverage for ROI serialization with new fields, palette apply helpers, tag filter logic, and pagination throttles.
   - *Pending:* Schedule manual notebook verification once remaining UI behaviours land.

6. **Documentation & release notes**
   - *Done:* README, log, and issue summary updated to describe the ROI browser/editor split and preset persistence.
   - *Done:* Release notes now cover pagination and cursor-aware expression helpers alongside earlier preset persistence notes.

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
- ✅ Replaced the "scroll down and show more" affordance with explicit pagination that still surfaces four ROI previews per step.

## Current Action Plan (Cycle starting 2025-11-01)
1. ✅ **Expression builder cursor placement**
   - Completed: Added a JS-backed selection bridge and caret-aware insertion helpers so operator/tag shortcuts splice at the cursor and refocus the field.
2. ✅ **ROI gallery pagination controls**
   - Completed: Replaced the scroll-triggered loader with Previous/Next navigation, a page label, and automatic page resets when filters or expressions change.
3. 🚧 **Testing & documentation**
   - Next: Extend unit coverage for pagination/cursor helpers, run the ROI widget suites once the harness catches up, and capture screenshots if we add walkthroughs.
4. ✅ **Caret alignment regression fix**
   - Investigated blur-triggered selection resets that pushed helper insertions back to index 0.
   - Patched the Python-side message handler to ignore unfocused updates while preserving the last focused caret.
   - Added unit coverage simulating focus/blur cycles to lock the behaviour down, hardened the DOM selectors so the caret bridge attaches in modern JupyterLab builds alongside classic Notebook/Voila, and reset the default caret to the expression tail after backend restores.
   - Reworked the insertion helper to honour the cached start/end indices (including highlighted ranges) and advanced the stored caret so repeated helper clicks keep chaining from the user’s cursor.
   - Restored the selection resolver and focus-aware caching after a regression so helper buttons continue updating the expression field even when blur events fire mid-click.
   - Moved snippet insertion into the browser via custom widget messages so the front end edits the value and caret before the update syncs back to Python, removing backend/front-end caret races.

## Validation Plan
- Run the ROI-related pytest modules (`tests/test_roi_manager_tags.py`, `tests/test_export_job.py`, plus any new tests).
- Exercise the plugin manually within `run_viewer.ipynb`, checking both tabs, filtering, and preset applications.
- Verify CSV import/export retains new metadata across sessions.

## 2025-11-03 Scroll Containment Follow-up

- **Observation:**  After landing the Reply 8 layout tweak, the ROI gallery still renders past the 400px viewport in live notebooks. The Matplotlib canvas pushes below the pagination bar and the outer accordion picks up the scrollbar instead of the intended thumbnail pane (see screenshot attached in the issue thread).
- **Root cause hypothesis:**  The `ipywidgets.Output` height/overflow styles are applied to the outer `div.widget-output`, but Matplotlib’s nbagg canvas injects an inner `<div class="jp-OutputArea-output">` with `overflow: visible`. Because we call `plt.show(fig)` (which hands ownership to nbagg), the generated widget uses its own sizing and ignores the outer scroll constraints, so the browser_output container simply stretches.
- **Additional factors:**  The calculated figure height is ~560px (3 columns × 4 rows at ~1.96 in per tile × 72 dpi). With nbagg still in control, the canvas also sets `style="height: 560px; width: 420px;"`, so the browser box never receives a scroll height shorter than the content.

### Potential Fixes
1. **Drive sizing through `fig.canvas.layout`:** Switch to `fig.canvas.layout = Layout(height=self.BROWSER_SCROLL_HEIGHT, width="100%", overflow="hidden")` before calling `plt.show`. When using the nbagg backend the canvas itself is a widget, so overriding its layout may let the outer `Output` respect the 400px cap. We will need to guard this for non-widget backends (tests use stubs) with `hasattr(fig, "canvas")` checks.
2. **Render to an explicit widget image:** Replace `plt.show` with a manual PNG render (`fig.canvas.print_png(BytesIO())`) and feed the bytes into `ipywidgets.Image(layout=Layout(width="100%", max_height=self.BROWSER_SCROLL_HEIGHT, overflow="auto"))`. This makes the gallery a standard widget tree that honours the parent layout at the cost of losing interactive zoom.
3. **Wrap the figure inside a fixed-size container:** Create a `VBox` with `Layout(height=self.BROWSER_SCROLL_HEIGHT, overflow_y="auto")` and place the canvas inside using `container.children = (canvas,)`. This avoids the Output widget entirely for the gallery surface and gives us full control over scrollbars.
4. **CSS override:** As a stop-gap, inject a stylesheet targeting `.roi-browser-output .jp-OutputArea-output` to force `max-height: 400px; overflow-y: auto;`. This keeps nbagg but may rely on Lab-specific class names, so we should treat it as a fallback once we confirm the DOM contract across Notebook and Voila.

### Next Steps
- Implemented option (1) by assigning a constrained `ipywidgets.Layout` to `fig.canvas` and displaying the canvas widget directly when ipympl is active; confirm in notebooks that the canvas now respects the 400px viewport. If the overflow persists, pivot to option (2) since it keeps dependency light and is easiest to test.
- Update screenshots and add regression captures once the preferred approach is verified across notebook front-ends (Lab, classic, Voila).
