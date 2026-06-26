# Issue #105 — Heatmap Checkpoint Save/Load

## Problem

The heatmap + FlowSOM annotation workflow is fully iterative: users cluster all
immune cells, subset CD4s, recluster, relabel, etc. Currently nothing is written
to disk — every restart forces a full redo, making the workflow non-reproducible.

## Scope (Mid-Range)

- **New plugin**: `CellAnnotationPlugin` — save form + parent-child checkpoint tree.
- **New data format**: `.h5ad` (AnnData) files under `<root>/.UELer/dataset_<hash>/checkpoints/`.
- **Out of scope**: DAG merge flows, imputation/projection, checksums.

## New Files

| File | Purpose |
|---|---|
| `ueler/viewer/interfaces.py` | `HeatmapStateProvider` + `FlowsomParamsProvider` Protocol stubs |
| `ueler/viewer/checkpoint_store.py` | Atomic `.h5ad` read/write + `manifest.json` management |
| `ueler/viewer/plugin/cell_annotation.py` | `CellAnnotationPlugin` — save form + tree browser |
| `tests/test_checkpoint_store.py` | 14 unit tests for CheckpointStore |
| `tests/test_cell_annotation.py` | 14 unit tests for CellAnnotationPlugin |

## Modified Files

| File | Changes |
|---|---|
| `pyproject.toml` | Added `anndata>=0.10` to required dependencies |
| `ueler/viewer/plugin/heatmap_layers.py` | Added `export_heatmap_state()` / `import_heatmap_state()` to `DataLayer` |
| `ueler/viewer/plugin/run_flowsom.py` | Added `export_flowsom_params()` / `import_flowsom_params()` |
| `tests/bootstrap.py` | Fixed `_ensure_dask_stub()` to prefer real dask (prevents anndata import crash) |

## AnnData `.h5ad` Format

```
adata.X                            # float32, shape (n_clusters, n_markers), z-scored medians
adata.obs_names                    # cluster label strings
adata.obs["meta_cluster"]          # int
adata.obs["meta_cluster_revised"]  # int (if present)
adata.var_names                    # marker names
adata.uns["checkpoint"]            # {id, parent_id, op, step_id, description, created_at, producer}
adata.uns["palette"]               # {colors, names, next_id} — all keys as str
adata.uns["ui"]                    # {selected_channels, cluster_method, distance_metric, ...}
adata.uns["row_linkage"]           # linkage matrix as nested list
adata.uns["dendrogram_cut"]        # float
adata.uns["flowsom"]               # {column_name, channels, xdim, ydim, rlen, seed, ...} (optional)
```

## Key Design Decisions

- **palette overwrite**: `generate_heatmap()` overwrites `meta_cluster_colors` via
  `_sync_meta_cluster_registry()`. Import re-applies saved palette AFTER calling
  `generate_heatmap()`.
- **HDF5 None**: `parent_id=None` is serialized as `""` in the `.h5ad` uns dict
  (HDF5 cannot store Python None in string fields).
- **Atomic writes**: `.h5ad.partial` → fsync → `os.replace()` for crash-safe writes;
  same pattern for `manifest.json.partial`.
- **Auto-discovery**: `CellAnnotationPlugin` is auto-loaded by `dynamically_load_plugins`
  (filename `cell_annotation.py`, subclasses `PluginBase`) — no `ui_components.py`
  changes needed.

## Tests

```bash
/omics/groups/OE0622/internal/shared_envs/ark-analysis-dask_yw/bin/python \
  -m unittest tests.test_checkpoint_store tests.test_cell_annotation -v
```

- 28 tests, all pass.

---

## Follow-Up Fix — Heatmap Not Displayed After Cell Annotation Plugin Load

**Reported:** After the issue #105 implementation, clicking the Plot button in the Heatmap plugin showed the status bar going busy → idle but no heatmap was rendered.

### Root Cause

Two interacting problems:

1. **`CellAnnotationPlugin` has no `ui_component`.**  
   `PluginBase.after_all_plugins_loaded()` calls `self.load_widget_states(path)`. If a `Cell Annotation_widget_states.json` exists on disk, `load_widget_states()` does `vars(self.ui_component)` — raising `AttributeError` because `CellAnnotationPlugin` manages its widgets directly on `self`.

2. **No error isolation in the `main_viewer.after_all_plugins_loaded()` loop.**  
   `dir(self.SidePlots)` is alphabetical, so `cell_annotation_output` is processed **before** `heatmap_output`. An uncaught exception in the cell annotation call stopped the loop, preventing `HeatmapDisplay.after_all_plugins_loaded()` from running (which calls `_sync_panel_location()` and `refresh_bottom_panel()` — missing these left the plot section in a broken layout state).

### Fixes

- **`ueler/viewer/plugin/cell_annotation.py`**: Override `after_all_plugins_loaded()` to NOT call `super()` — wires `_store`, `_heatmap_plugin`, and `_flowsom_plugin` directly without touching `PluginBase.load_widget_states()`.
- **`ueler/viewer/main_viewer.py`** — two changes:
  - `after_all_plugins_loaded()`: wrapped each plugin call in `try/except Exception`.
  - `dynamically_load_plugins()`: wrapped plugin instantiation in `try/except Exception`.
- **`tests/test_cell_annotation.py`**: Added `TestAfterAllPluginsLoaded` regression class (now 29 tests total).

### Tests

```bash
/omics/groups/OE0622/internal/shared_envs/ark-analysis-dask_yw/bin/python \
  -m unittest tests.test_checkpoint_store tests.test_cell_annotation -v
```

- ✅ 29 tests, all pass.
