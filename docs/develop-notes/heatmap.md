# Heatmap & Cell Annotation

> Source: [`dev_note/topic_heatmap_flowsom_cell_annotation.md`](https://github.com/HartmannLab/UELer/blob/main/dev_note/topic_heatmap_flowsom_cell_annotation.md)

---

## Context

These notes document the heatmap plugin evolution, meta-cluster management, and the planned cell annotation workflow that coordinates heatmap and FlowSOM checkpoints.

---

## Key Decisions

- A **dedicated meta-cluster registry** provides rename/add/remove controls separate from the dendrogram view.
- Meta-cluster colors always use registry mappings, even beyond dendrogram cutoffs.
- **Z-score normalization** across markers is supported, with mode-aware colormap rendering.
- **Single-point scatter interactions** are guarded to avoid collapsing the linked cell gallery.

---

## Heatmap Plugin

The heatmap displays a cell-by-marker matrix, optionally grouped by meta-cluster. Key features:

- Interactive row (cell) selection linked to the scatter plot and cell gallery.
- FlowSOM meta-cluster assignment with a dedicated management tab.
- Z-score normalization toggled per-session.
- Wide layout mode for the footer panel.

---

## Meta-Cluster Management

Meta-clusters are stored in a registry with:

- Unique ID and display name.
- Color assignment (used in heatmap rows, scatter plot points, and gallery borders).
- Assignment dropdown for changing a cell's cluster.

Color mappings are applied at render time and extend beyond the visible dendrogram cutoff.

---

## Cell Annotation Workflow (Planned)

A cell annotation orchestrator is specified but not yet implemented as a plugin. The planned design:

- AnnData-based checkpoint format with DAG-style lineage.
- Marker set semantics for annotation snapshots.
- Checkpoint browser UI for navigating annotation history.

---

## Related Issues

- [#48](https://github.com/HartmannLab/UELer/issues/48)
- [#73](https://github.com/HartmannLab/UELer/issues/73)
- [#74](https://github.com/HartmannLab/UELer/issues/74)
