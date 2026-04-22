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
