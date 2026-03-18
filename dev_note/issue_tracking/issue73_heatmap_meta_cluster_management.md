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

## Follow-up Fix (2026-03-18)
### Problem
- Heatmap regeneration called `_sync_meta_cluster_registry(np.unique(meta_cluster_labels), ...)`.
- `_sync_meta_cluster_registry` used `list(meta_cluster_ids or [])`, which boolean-evaluated numpy arrays and raised:
	`ValueError: The truth value of an array with more than one element is ambiguous`.

### Resolution
- Updated `_sync_meta_cluster_registry` to avoid truthiness checks on array-like inputs:
	- Use `incoming_ids = [] if meta_cluster_ids is None else list(meta_cluster_ids)`.
	- Normalize from `incoming_ids` directly.

### Regression Coverage
- Added `test_sync_registry_accepts_numpy_array_ids` in `tests/test_heatmap_selection.py`.

### Validation
```bash
python -m unittest tests.test_heatmap_selection
```
- ✅ All tests passed (`Ran 16 tests ... OK`)

## Follow-up Fix (2026-03-18, horizontal layout blank output)
### Problem
- In wide/horizontal layout, clicking `Plot` could generate the heatmap but leave the footer display area blank.
- Toggling back to vertical sometimes showed a brief flash, indicating the figure existed but was not reliably replayed in the output widget.

### Resolution
- Added cached footer artifact replay in `DisplayLayer`:
	- Cache figure/canvas artifacts after `generate_heatmap`.
	- Add `redraw_cached_footer_canvas()` to redraw/re-display cached figure when widget-view output is missing.
	- Call `restore_footer_canvas()` after plotting in wide mode to enforce footer rendering.
- Added robust widget-view detection to avoid unnecessary re-display when a widget output already exists.

### Validation
```bash
python -m unittest tests.test_heatmap_selection
```
- ✅ All tests passed (`Ran 16 tests ... OK`)
