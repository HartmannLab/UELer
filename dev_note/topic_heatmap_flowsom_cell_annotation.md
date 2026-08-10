# Heatmap, FlowSOM, and Cell Annotation

## Context
These notes document the heatmap plugin evolution, meta-cluster management, and the planned cell annotation workflow that coordinates heatmap and FlowSOM checkpoints.

## Key decisions
- Keep a dedicated meta-cluster registry with rename/add/remove controls.
- Ensure meta-cluster colors always use registry mappings, even beyond dendrogram cutoffs.
- Support z-score normalization modes and make the colormap reflect the active mode.
- Define a cell-annotation checkpoint format (AnnData-based) with DAG-style lineage and marker set semantics.

## Current status
- Heatmap marker selection now shares the scatter/histogram channel picker (#117): `heatmap.UiComponent` builds `_chart_common.build_channel_selector(main_viewer)` and aliases `channel_selector = bundle.tags`, so the heatmap gets the same **"Marker set:" dropdown + "Load set" button** UX (loading a predefined marker set into the picker, local-only). `HeatmapDisplay.on_marker_sets_changed` → `_chart_common.refresh_marker_set_options` keeps the dropdown in sync (broadcast + `after_all_plugins_loaded`). Channel options are numeric-only as a result. See [issue_tracking](../dev_note) and `dev_note/github_issues.md` (#117).
- Heatmap remembers its scale (figure size) across tree-cut updates (#109): `apply_new_cutoff` captures `fig.get_size_inches()` before the rebuild and `_refresh_plot(restore_size=...)` → `generate_heatmap(figsize_override=...)` rebuilds the clustermap at that size, so the size set via the ipympl resize triangle survives re-clustering (fresh Plot uses the default). See [issue_tracking/issue109_heatmap_remember_zoom.md](issue_tracking/issue109_heatmap_remember_zoom.md).
- Heatmap rendering reliability fixed (#108): the real cause was heatmap-specific — the `sns.clustermap` figure was built **inside** the `with <output>:` display context and then `plt.show()`n, so ipympl (interactive mode) emitted the canvas twice → blank. The plugin now builds the figure with `plt.ioff()` outside the Output and emits the interactive canvas exactly once via `display(fig.canvas)` into a fresh `Output` swapped into `plot_section.children` (mirroring the Chart histogram, which builds outside and emits once). Layout-toggle double-render removed and the reparented footer canvas is force-repainted after it becomes visible. Full canvas interactivity and footer docking preserved; no static fallback. See [issue_tracking/issue108_heatmap_not_showing.md](issue_tracking/issue108_heatmap_not_showing.md).
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
