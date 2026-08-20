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

The heatmap displays a **cluster × marker** matrix, not a cell × marker one: `DataLayer` groups the
cell table by the chosen class column and reduces each group to a median per marker
(`df.groupby(cluster_column)[marker_columns].median()`), then z-scores the result. One row is one
cluster, however many cells it contains — which is what makes the plugin usable on a million-cell
table.

Key features:

- Interactive row (cluster) selection linked to the scatter plot and cell gallery.
- FlowSOM meta-cluster assignment with a dedicated management tab.
- Z-score normalization toggled per-session, across markers or across classes.
- **Permanently allocated to the wide-footer panel** (`footer_only = True`, #121). The plugin is
  skipped when the side accordion is assembled and always renders in the wide orientation; the earlier
  footer/side placement toggle and the separate horizontal-layout toggle are both gone. The adapter is
  constructed as `HeatmapModeAdapter(mode="wide")` and never switched.
- Remembers a user-resized figure size across a tree re-cut.

The plugin class composes four bases — `DataLayer`, `InteractionLayer`, `DisplayLayer`, `PluginBase` —
which is why the implementation is spread across `heatmap.py` and `heatmap_layers.py`.

---

## Meta-Cluster Management

Meta-clusters are stored in a registry with:

- Unique ID and display name.
- Color assignment (used in heatmap rows, scatter plot points, and gallery borders).
- Assignment dropdown for changing a cell's cluster.

Color mappings are applied at render time and extend beyond the visible dendrogram cutoff.

---

## Cell Annotation Workflow

Shipped as `CellAnnotationPlugin` (`ueler/viewer/plugin/cell_annotation.py`), displayed as **Cell
Annotation**. The design that was planned here is the design that landed:

- **AnnData checkpoint format.** `CheckpointStore` (`ueler/viewer/checkpoint_store.py`) writes each
  checkpoint as an `.h5ad` under `<dataset_root>/.UELer/dataset_<sha1>/checkpoints/`, with an
  atomically-rewritten `manifest.json` beside it. The dataset hash scopes the history to one dataset,
  so several under a shared base folder do not collide.
- **DAG-style lineage.** Each record carries a `parent_id` (empty string when absent) and an `op` —
  `initial`, `subset`, `recluster` or `finalize` — which is what lets the browser render the history as
  a tree rather than a list.
- **Checkpoint browser UI.** `CheckpointTreeWidget`, an anywidget tree renderer with an ipywidgets
  fallback for environments without anywidget.
- **Coupled to the Heatmap and FlowSOM plugins.** A checkpoint captures the heatmap state (z-scored
  medians, meta-cluster palette, UI settings) *and* the FlowSOM parameters, so loading one restores
  both at once.

Unlike the Heatmap it links to, this plugin lives in the side accordion, not the footer.

---

## Related Issues

- [#48](https://github.com/HartmannLab/UELer/issues/48)
- [#73](https://github.com/HartmannLab/UELer/issues/73)
- [#74](https://github.com/HartmannLab/UELer/issues/74)
