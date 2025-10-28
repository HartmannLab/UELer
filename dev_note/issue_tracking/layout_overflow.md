# Layout Overflow Issue #39

## Problem Description
Several UI components in the app have contents that overflow or are larger than their containers (ipywidgets VBOX/HBOX):
- Left control panel, including all accordions
- Plugins
  - Chart (heatmap) display area
  - Chart display area
  - Batch export section
  - ROI manager

This layout issue does not occur in other plugins. It may be helpful to investigate their implementation for a solution or reference.

## Suggested Actions
- Review container sizing and widget layout logic in affected areas
- Compare with plugins where the problem does not exist
- Apply fixes to ensure all content fits within the intended containers
- Consider allowing for a max width of 95% of the parental box

## Investigation Notes

### Current Layout Structure
- **Main UI**: `HBox([left_panel, image_output, side_plot])`
  - `left_panel`: `VBox` with `width='350px'`, `overflow_y='auto'`
    - Contains `control_sections`: `Accordion` with `width='100%'`, `max_height='640px'`
      - Children: `channel_controls_box`, `mask_controls_box`, `annotation_controls_box` with `width='100%'`, `overflow_y='auto'`
  - `side_plot`: `VBox` of plugin accordions
    - Each accordion: `Layout(width='6in')`
    - Plugin `ui`: `VBox` with `width='100%'`, various `max_height`

### Potential Causes
- Plugin `ui` layouts have `width='100%'` but no `max_width`, allowing content to exceed accordion width
- Left panel accordion children have `width='100%'` but widgets with fixed widths (e.g., `width='250px'`) may cause horizontal overflow if total width exceeds container
- No `overflow_x` handling in containers, leading to content spilling out

### Comparison with Non-Affected Plugins
- Need to identify plugins that don't have overflow issues
- Check their layout settings for differences

## Action Plan

1. **Investigate Current Layouts**
   - Examine the `ui` layout of each affected plugin (chart, heatmap, export_fovs, roi_manager)
   - Check widget layouts in left panel controls
   - Identify specific widgets causing overflow

2. **Compare with Working Plugins**
   - List all plugins and check which ones don't have overflow
   - Analyze their layout differences

3. **Implement Fixes**
   - Add `max_width='95%'` to plugin `ui` layouts
   - Add `max_width='95%'` to left panel control boxes
   - Consider adding `overflow='auto'` to accordions if needed

4. **Test Changes**
   - Run the app and check for overflow in affected areas
   - Verify functionality is not broken

5. **Update Documentation**
   - Update README.md and log.md as per instructions

## Implementation Details

### Files to Modify
- `ueler/viewer/ui_components.py`: Left panel control boxes
- `ueler/viewer/plugin/chart.py`: Chart plugin ui
- `ueler/viewer/plugin/heatmap_layers.py`: Heatmap plugin ui
- `ueler/viewer/plugin/export_fovs.py`: Batch export plugin ui
- `ueler/viewer/plugin/roi_manager_plugin.py`: ROI manager plugin ui

### Code Changes
- For each plugin `self.ui = VBox(..., layout=Layout(..., max_width='95%'))`
- For left panel boxes: `layout=Layout(..., max_width='95%')`

## Progress Tracking
- [ ] Investigate current layouts
- [ ] Compare with non-affected plugins
- [ ] Implement max_width fixes
- [ ] Test changes
- [ ] Update documentation</content>
<parameter name="filePath">/omics/groups/OE0622/internal/ywu/UELer_public/dev_note/issue_tracking/layout_overflow_issue.md