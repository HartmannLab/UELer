# Heatmap, FlowSOM, and Cell Annotation

## Context
These notes document the heatmap plugin evolution, meta-cluster management, and the planned cell annotation workflow that coordinates heatmap and FlowSOM checkpoints.

## Key decisions
- Keep a dedicated meta-cluster registry with rename/add/remove controls.
- Ensure meta-cluster colors always use registry mappings, even beyond dendrogram cutoffs.
- Support z-score normalization modes and make the colormap reflect the active mode.
- Define a cell-annotation checkpoint format (AnnData-based) with DAG-style lineage and marker set semantics.

## Current status
- Heatmap meta-cluster management tab and assignment dropdown are implemented.
- Meta-cluster color mapping beyond cutoff is fixed and tested.
- Z-score across markers and mode-aware colormaps are implemented with tests.
- Single-point scatter interactions are guarded to avoid collapsing the cell gallery in linked workflows.
- Cell annotation workflow is specified but not yet implemented as a plugin.

## Open items
- Implement the cell annotation orchestrator and checkpoint browser.
- Add formal schema validation and storage helpers for checkpoint artifacts.

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/48
- https://github.com/HartmannLab/UELer/issues/73
- https://github.com/HartmannLab/UELer/issues/74

## Key source links
- [dev_note/Cell_annotation.md](dev_note/Cell_annotation.md)
