# Issue #74 - Meta-cluster color mapping beyond cutoff

## Problem
- Heatmap plotting currently derives a cutoff palette from `cut_tree(...)` labels.
- When users add/reassign meta-clusters beyond that cutoff-derived label range, row-color rendering can fall back to colormap sampling and show incorrect colors.
- Expected behavior is that all meta-clusters use the registry mapping (`meta_cluster_colors`) regardless of how many IDs exist beyond the cutoff.

## Root Cause
- Cluster color resolution tried to use the rendered colormap first.
- For IDs outside the original cutoff range, the fallback colormap path produced unintended colors before registry colors were considered.

## Proposed Fix
1. In heatmap row-color resolution, prioritize explicit registry colors:
   - `meta_cluster_colors` by meta-cluster ID
   - then `cluster_colors` by cluster label
   - only then fallback to colormap sampling
2. Keep existing rendering structure unchanged to avoid layout/interaction regressions.
3. Add a regression test where:
   - a colormap exists,
   - a meta-cluster ID beyond cutoff is present,
   - the resolver must return the registered registry color.

## Implementation
- Updated `DataLayer._build_cluster_color_resolver` in `ueler/viewer/plugin/heatmap_layers.py`.
- Added `test_cluster_color_resolver_prefers_registered_meta_cluster_colors` in `tests/test_heatmap_selection.py`.

## Validation
```bash
python -m unittest tests.test_heatmap_selection tests.test_heatmap_adapter
```
- Result: `Ran 26 tests ... OK` (`skipped=2`)

## Acceptance Criteria Mapping
- Plot reflects correct colors for meta-clusters added beyond initial cutoff: ✅
- Color mapping logic accommodates arbitrary meta-cluster IDs: ✅
