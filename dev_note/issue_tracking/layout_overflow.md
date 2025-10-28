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
- Apply fixes to ensure all content fits within the intended containers without triggering scrollers
- Carefully plan widget widths to fit perfectly within parent containers

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
- Adding `max_width='95%'` triggers horizontal scrollers instead of proper fitting

### Comparison with Non-Affected Plugins
- Need to identify plugins that don't have overflow issues
- Check their layout settings for differences

## Action Plan

1. **Investigate Current Layouts**
   - Examine the `ui` layout of each affected plugin (chart, heatmap, export_fovs, roi_manager)
   - Check widget layouts in left panel controls
   - Identify specific widgets causing overflow
   - Measure actual content widths vs container widths

2. **Compare with Working Plugins**
   - List all plugins and check which ones don't have overflow
   - Analyze their layout settings for differences
   - Identify best practices for fitting content

3. **Implement Precise Fixes**
   - Adjust widget widths to fit within containers without overflow
   - Use flex layouts where appropriate to distribute space
   - Ensure total width of HBox children doesn't exceed parent
   - Avoid max_width caps that trigger scrollers
   - Consider adjusting container widths if necessary for better fit

4. **Test Changes**
   - Run the app and check for overflow in affected areas
   - Verify no horizontal scrollers appear
   - Ensure functionality is not broken

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
- Adjust widget layouts to fit precisely within containers
- Use appropriate width settings instead of max_width caps
- Ensure HBox children widths sum appropriately

## Progress Tracking
- [ ] Investigate current layouts and measure widths
- [ ] Compare with non-affected plugins
- [ ] Implement precise width adjustments
- [ ] Test changes without scrollers
- [ ] Update documentation</content>
<parameter name="filePath">/omics/groups/OE0622/internal/ywu/UELer_public/dev_note/issue_tracking/layout_overflow_issue.md