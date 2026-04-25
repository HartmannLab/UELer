# Issue #86: Scatter widget cross-plot selection sync

## Problem
When multiple scatter plots are active in the Chart plugin, selecting points in one scatter only updates that originating plot. Other scatter plots in the same widget keep stale selection state, so coordinated exploration across plots breaks.

## Root cause
- `ChartDisplay._render_scatter_area()` composes multi-plot layouts with `sync_selection=False`.
- That setting is intentional: `doc/log.md` records a prior `jscatter.compose` `ValueError` when native compose-level selection syncing was enabled.
- The intended replacement path in `ChartDisplay` was incomplete. `_on_scatter_selection()` updated shared plugin state (`selected_indices`, linked mask highlights, single-point navigation) but never propagated the new selection back to sibling scatter views in `_scatter_views`.

## Chosen approach
Keep `compose(..., sync_selection=False)` and restore synchronization inside UELer's own observer pipeline.

This keeps the earlier compose workaround intact while still making all active scatter plots reflect the same selection state. `ScatterPlotWidget.apply_selection(..., announce=False)` already suppresses callback echo through its `_suspend_selection` guard, so it is the right primitive for safe fan-out.

## Implementation steps
1. Refactor `ChartDisplay` so widget-originated and external selection paths share one helper that:
   - normalizes row indices,
   - updates `single_point_click_state`,
   - applies the normalized selection to every active scatter view with `announce=False`,
   - updates `selected_indices`,
   - preserves linked mask-highlight behavior,
   - preserves single-point main-viewer focus only for widget-originated selections.
2. Keep `compose(..., sync_selection=False)` unchanged.
3. Add regression coverage for:
   - multi-plot selection fan-out,
   - empty-selection clearing across all active plots,
   - no regression to linked mask highlighting.

## Validation
- Automated:
  - `python -m unittest tests.test_chart_footer_behavior tests.test_chart_cell_gallery_link`
- Manual:
  1. Launch the viewer in an environment with the interactive widget backend available.
  2. Create two scatter plots in the Chart plugin.
  3. Select points in the first scatter plot and confirm the second plot reflects the same selection.
  4. Clear the selection and confirm both plots clear.
  5. Confirm linked mask highlighting still updates when the main-viewer link toggle is enabled.

## Risks
- Re-enabling compose-level selection syncing would risk reintroducing the previous `jscatter.compose` error, so it remains out of scope for this fix.
- The VS Code static scatter fallback cannot validate this behavior because it does not render interactive scatter widgets.