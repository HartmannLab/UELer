# Clustering & Annotation

These three plugins — **Heatmap**, **FlowSOM**, and **Cell Annotation** — turn per-cell marker values
into named cell populations. They form a pipeline:

```
FlowSOM  ──▶  cluster labels in the cell table
   │
   ▼
Heatmap  ──▶  group clusters into meta-clusters, assign colors & names
   │
   ▼
Cell Annotation  ──▶  save/restore the whole state as a checkpoint
```

All three require a [cell table](cell-table.md).

---

## FlowSOM: unsupervised clustering

The **FlowSOM** plugin runs self-organizing-map clustering on the channels you choose and writes the
resulting cluster label into a cell-table column.

1. Pick the clustering channels in **Channels:**.
2. Set **Save as:** — the output column name (default `FlowSOM_cluster`).
3. Optionally restrict the input with **Subset on:** / **Subset:**.
4. Tune the SOM parameters: **xdim:** / **ydim:** (grid size, default 10×10), **rlen:** (training
   length, default 10), **seed:** (default 42).
5. Click **Run**.

!!! note "Optional dependency"
    FlowSOM needs the `pyFlowSOM` package. If it isn't installed, the rest of UELer still works — only
    this plugin raises "pyFlowSOM is required to run the FlowSOM plugin" when you click Run.

---

## Heatmap: meta-clusters and colors

The **Heatmap** plugin shows a marker × cluster heatmap of z-scored median expression, with a
dendrogram, and lets you group clusters into **meta-clusters**.

### Plot it (Setup tab)

Choose **Channels:**, the **Class:** column (the cluster column to summarize — e.g. your FlowSOM
output), and the clustering **Linkage:** and **Metric:**. Toggle **Z-score across markers** to
normalize per class instead of per marker. Click **Plot**.

Enable **Horizontal layout** to move the heatmap into the footer panel. The heatmap remembers a
figure size you set by dragging its resize handle, even after re-cutting the tree.

### Assign meta-clusters (Assign tab)

Click the dendrogram to set the tree-cut, then assign branches to a **Meta-cluster:** and click
**Apply**. The cutoff auto-locks after edits; use **Unlock once** to adjust it again.

### Name and color them (Rename tab)

Give each meta-cluster a display **Label:** (**Rename**), or **Add meta-cluster** / **Remove
selected**. Each meta-cluster gets a color, shown in the registry list as `[#hex] name (id)`. These
meta-cluster colors are used in the heatmap's color strip and can color the scatter plot.

### Save to the cell table (Save tab)

Enter a **Column Name:** (default `new_cluster`) and click **Save to Cell Table** to write the
meta-cluster labels back to the table (**Overwrite** if the column exists).

---

## Cell Annotation: checkpoints

The **Cell Annotation** plugin saves and restores your clustering/annotation state as `.h5ad`
**checkpoints**, so you can pause and resume a multi-step gating workflow.

- **Save checkpoint** — record the current heatmap state (meta-clusters, palette, tree-cut) together
  with the FlowSOM parameters. Tag it with a **Step:**, **Desc:**, **Parent:**, and an **Op:**
  (`initial`, `subset`, `recluster`, `finalize`).
- The **Checkpoint browser** shows the checkpoints as a tree (color-coded by op). Select one and use
  **Load selected** to restore it, or **Delete selected** to remove it.

Checkpoints are stored under `<base_folder>/.UELer/` (per-dataset), so they travel with your data.

!!! note
    Cell Annotation wires itself to the Heatmap and FlowSOM plugins, so saving/loading a checkpoint
    restores both the heatmap annotation state and the FlowSOM settings together.
