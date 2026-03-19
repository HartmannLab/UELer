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

## Follow-up Fix (2026-03-19, horizontal layout width overflow)
### Problem
- In wide/horizontal layout, heatmap width was computed as `len(meta_cluster_labels) * 0.3` inches.
- Large cluster counts could produce figures wider than the Heatmap footer tab, causing overflow in the horizontal tab pane.

### Resolution
- Added a wide-mode width clamp in `HeatmapModeAdapter.build_clustermap_kwargs`:
	- Keep data-driven width for smaller heatmaps.
	- Cap width to the plugin width budget (`width * 0.9`, with safe fallbacks) so rendered figures stay within tab width.

### Regression Coverage
- Added `tests/test_heatmap_adapter.py` with focused assertions for:
	- wide-mode width clamping when cluster count is large,
	- unchanged wide-mode sizing when under the cap,
	- unchanged vertical-layout sizing behavior.

### Validation
```bash
python -m unittest tests.test_heatmap_adapter tests.test_heatmap_selection
```
- ✅ All tests passed (`Ran 19 tests ... OK`)

## Follow-up Fix (2026-03-19, save-to-cell-table label output)
### Problem
- Heatmap `Save to Cell Table` was storing numeric meta-cluster IDs in the newly requested output column.
- After users renamed meta-clusters in the `Rename` tab, exported columns still contained IDs instead of the renamed labels.

### Resolution
- Updated `save_to_cell_table` to map saved values through the meta-cluster registry display-name resolver.
- The requested output column now stores display labels (including renamed labels), sourced from revised assignments when available.
- Preserved revised numeric IDs in `<column_name>_revised` when the revised assignment column exists.

### Validation
```bash
python -m unittest tests.test_heatmap_selection tests.test_heatmap_adapter
```
- ✅ All tests passed (`Ran 20 tests ... OK`)

## Follow-up Fix (2026-03-19, revised export column label output)
### Problem
- The companion `<column_name>_revised` export column still contained numeric IDs after the first label-export fix.
- Users expected the revised export column to show renamed labels too.

### Resolution
- Updated `save_to_cell_table` to map `<column_name>_revised` through the same meta-cluster display-name resolver.
- Both exported columns now save labels instead of numeric IDs.

### Validation
```bash
python -m unittest tests.test_heatmap_selection tests.test_heatmap_adapter
```
- ✅ All tests passed (`Ran 20 tests ... OK`)

## Follow-up Fix (2026-03-19, z-score across markers option)
### Problem
- Heatmap normalization supported only marker-wise z-scoring across classes.
- Users needed an option to z-score across markers (within each class) for alternate pattern interpretation.

### Resolution
- Added a new setup checkbox, `Z-score across markers`, in the Heatmap controls.
- Updated `prepare_heatmap_data` to switch normalization axis:
	- unchecked: default per-marker z-score across classes,
	- checked: per-class z-score across selected markers.
- Added focused tests for both modes in `tests/test_heatmap_selection.py`.

### Validation
```bash
python -m unittest tests.test_heatmap_selection tests.test_heatmap_adapter
```
- ✅ All tests passed (`Ran 22 tests ... OK`, `skipped=2`)

## Follow-up Fix (2026-03-19, z-score/non-zscore color palettes)
### Problem
- Heatmap colormap did not clearly encode signed z-score values and looked similar across normalization contexts.
- Users requested a diverging signed palette for z-score mode and a red-ish palette when z-scoring is not active.

### Resolution
- Added mode-aware colormap selection in heatmap rendering:
	- z-score mode: `bwr` with `center=0` (red positive, blue negative, white near zero),
	- non-zscore mode: `Reds` sequential palette.
- Extended `HeatmapModeAdapter.build_clustermap_kwargs` to accept explicit `cmap` and `center` kwargs.
- Added focused tests for palette selection and adapter kwargs forwarding.

### Validation
```bash
python -m unittest tests.test_heatmap_selection tests.test_heatmap_adapter
```
- ✅ All tests passed (`Ran 25 tests ... OK`, `skipped=2`)
