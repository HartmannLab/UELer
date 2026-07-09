# Issue #85: UI improvements to prevent unnecessary scrollbars

## Problem
Some UI panels match their parent width exactly (`width: 100%`). Combined with gaps, borders, and padding, this can trigger unnecessary horizontal scrollbars in notebook layouts.

## Scope
- Main viewer control-side containers (channel controls, marker set controls, mask/annotation panels)
- Wide footer panel wrappers
- Plugin control wrappers that currently use exact full-width layouts (Chart heatmap, Heatmap layers, ROI manager, Batch export)
- Annotation palette editor host panel

## Approach
Use a mixed strategy:
- Preserve existing helper-driven constrained layouts already working well (for example `column_block_layout` with 98% width)
- For overflow-prone wrappers currently using exact full-width, add shrink-safe constraints:
  - `max_width: 99%`
  - `min_width: 0`
  - `box_sizing: border-box`

## Follow-up (Reply to #85)
- Global bounded wrapper default relaxed from 97% to 99% after reports of over-constrained controls.
- Added targeted inner-control normalization for:
  - Marker set fields (`Marker Set`, `Set Name`)
  - Channel min/max sliders
  - Channel legend and grid-view checkbox rows
- Kept `min_width: 0` and `box_sizing: border-box` safeguards to preserve scrollbar-prevention behavior.

## Follow-up 2 (Reply 2 to #85)
- Keep container wrappers unchanged (`max_width: 99%`) and move additional shrink behavior into content widgets.
- Add a content-widget layout policy for overflow-prone controls:
  - `width: calc(100% - 5px)`
  - `max_width: calc(100% - 5px)`
  - `min_width: 0`
  - `box_sizing: border-box`
- Apply the policy to core viewer controls first (sliders/text/dropdowns/checkboxes), then selected plugin content widgets.
- Preserve existing container helpers and wrapper placements to avoid regressions in footer/tab layout behavior.

## Follow-up 3 (Reply 3 to #85)
- Keep container wrappers unchanged at `max_width: 99%`; focus on remaining row-level offenders in channel controls.
- Compact marker-set action row layout and confirm-deletion row so they do not trigger horizontal overflow.
- Tighten per-channel header composition (checkbox + color dropdown):
  - narrower dropdown footprint,
  - compact checkbox spacing,
  - hidden row overflow to prevent content escape.
- Preserve slider labels/readout but increase usable slider track width through row layout tuning.
- Harden channel legend rendering for long unbroken names using wrapping styles and content-width constraints.

## Follow-up 4 (Reply 4 to #85)
- Reorganize each channel control into 3 rows:
  - row 1: visibility checkbox + marker name + color dropdown,
  - row 2: Min slider,
  - row 3: Max slider.
- Ensure marker name appears only once per channel (header row only).
- Remove marker names from Min/Max slider descriptions.
- Keep color dropdown compact (roughly half previous footprint) using content-driven width that still displays color names.

## Follow-up 5 (Reply 5 to #85)
- Remove per-channel internal overflow scrollbars introduced by the three-row grouping so only the shared channel-panel scroller remains.
- Shift color dropdown 5px to the left in the header row.

## Follow-up 6 (Reply 6 to #85)
- Ensure parent channel-panel vertical scrolling is triggered when summed channel-widget height exceeds container height.
- Prevent per-channel groups from shrinking to fit by using non-shrinking flex behavior in grouped channel rows.

## Follow-up 7 (Reply 7 to #85)
- Fix duplicate channel widget rows observed when loading saved marker sets.
- Remove redundant channel-control rebuilds during marker-set apply.
- Normalize saved and loaded marker channel lists to unique channel names while preserving first-seen order.

## Implementation steps
1. Add a shared bounded panel layout helper in `ueler/viewer/ui_components.py` and apply it to core control wrappers.
2. Update dynamic row wrappers in `ImageMaskViewer.update_controls` to avoid row overflow when checkboxes + dropdowns/sliders are combined.
3. Apply the same constrained-width policy to selected plugin wrappers in:
   - `ueler/viewer/plugin/chart_heatmap.py`
   - `ueler/viewer/plugin/heatmap_layers.py`
   - `ueler/viewer/plugin/roi_manager_plugin.py`
   - `ueler/viewer/plugin/export_fovs.py`
   - `ueler/viewer/annotation_palette_editor.py`
4. Extend tests for wide panel layout properties and run focused UI/plugin test suite.

### Reply 2 implementation addendum
1. Add a dedicated content-widget helper in `ueler/viewer/ui_components.py` for `calc(100% - 5px)` sizing.
2. Update dynamic controls in `ueler/viewer/main_viewer.py` (`update_controls`) to use the content helper policy for sliders/dropdowns.
3. Add a shared content helper in `ueler/viewer/layout_utils.py` and use it for plugin content widgets (`roi_manager_plugin.py`, `export_fovs.py`, `chart_heatmap.py`).
4. Add regression coverage for the content helper and run focused suites.

## Validation
- Automated:
  - `python -m unittest tests.test_wide_plugin_panel tests.test_chart_footer_behavior`
- Manual (notebook):
  1. Launch `script/run_ueler.ipynb`
  2. Confirm no unnecessary horizontal scrollbars in channel controls and marker set editor
  3. Confirm plugin panels remain usable and footer panels still switch correctly

## Risks
- Over-constraining plot container widths can clip content. Mitigation: constrain wrapper panels and keep content containers flex-capable (`min_width: 0`, `flex: 1 1 auto`).
- Layout assertions in tests may need updates if they depended on exact width defaults.

## Reply 7 implementation addendum
1. Add a small channel-list normalization helper in `main_viewer.py` that de-duplicates channel names in first-seen order.
2. Apply normalization in marker-set save/update/apply paths so marker sets cannot persist duplicate channels.
3. In `_apply_marker_set`, avoid unconditional `update_controls(None)` after assigning `channel_selector.value`; rely on selector observers when the value changes and only force a refresh when unchanged.
4. Remove the direct `channel_selector -> update_controls` observer in `ui_components.py` so `on_channel_selection_change` remains the single channel-change entry point.
5. Add regression tests for channel de-duplication helper behavior and run focused UI suites.
