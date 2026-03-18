# Issue #73 — Heatmap meta-cluster management tab and assignment dropdown

## Summary
Issue #73 adds a dedicated meta-cluster management workflow to the Heatmap plugin:
- Add a new `Rename` tab next to `Assign`.
- Replace free-text meta-cluster assignment with a dropdown.
- Show a color indicator next to each meta-cluster entry.
- Allow adding/removing meta-clusters.
- Reassign removed IDs to a default unassigned meta-cluster.

## Implementation Plan
1. Extend Heatmap UI widgets
- Replace `Assign` tab ID field with a dropdown backed by a meta-cluster registry.
- Add `Rename` tab controls: selection dropdown, rename text box, add/remove buttons, registry preview list.

2. Add a meta-cluster registry model
- Track `meta_cluster_names`, `meta_cluster_colors`, and `next_meta_cluster_id` in Heatmap data state.
- Keep an always-available unassigned entry (`-1`, `Unassigned`, grey color).
- Preserve color mapping for existing clusters and assign deterministic colors for new ones.

3. Wire assignment and management actions
- Assign selected heatmap clusters via dropdown value.
- Rename updates the label map only.
- Add creates a new ID/label/color and exposes it in dropdowns.
- Remove reassigns existing rows from removed ID to unassigned ID, then removes the ID from registry.

4. Keep rendering and legends consistent
- Keep row-color rendering based on current revised/meta IDs.
- Keep registry synced on each heatmap generation.
- Include revised IDs in sync so custom IDs continue to display after replot.

5. Regression tests
- Add focused tests for dropdown assignment, add-meta-cluster, and remove+reassign behavior.

## Validation
```bash
python -m unittest tests.test_heatmap_selection
```

## Status
- Implemented in `ueler/viewer/plugin/heatmap.py` and `ueler/viewer/plugin/heatmap_layers.py`.
- Added tests in `tests/test_heatmap_selection.py`.
- Focused test suite passes.
