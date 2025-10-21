# Heatmap data saving (FlowSOM workflow checkpoints)

**Goal:** capture every heatmap state (matrix + annotations) in a portable, self-sufficient artifact so the FlowSOM workflow (steps 1–11: initial clustering, revisions, mixed/lineage subsets, and finalizations) can be resumed, branched, diffed, or audited at any stage.

## Artifact format

* Primary: **`.h5ad`** written via `anndata.AnnData.write_h5ad` with compression (gzip/zstd, chunked).
* Optional (roadmap): **Zarr** for very large runs or incremental/delta checkpoints.

Use **atomic writes**: write to `*.tmp` then `os.replace`.

---

## Data to persist (schema)

Always store in **canonical orientation**: **rows = clusters**, **cols = markers**. If the UI is flipped, record that in `uns["orientation"]`.

### `X` (matrix)

* `X`: **z-scored medians** (cluster × marker), exactly what the user sees.

### `layers`

* `layers["median"]`: **raw medians** (pre-zscore) for re-scaling later.

### `obs` (per cluster row)

* Index: `cluster_id` (stable ID, e.g., `cluster_001`).
* Columns:

  * `metacluster_id` (str/int) – most recent FlowSOM meta-cluster.
  * `metacluster_revision_id` (nullable str/int) – revised label (step 2).
  * `subset_lineage` (category, optional).
  * `subset_mixed` (category, optional).
  * `subset_cohort` (category, optional).
  * User-selected provenance columns (FOV IDs, mask/channel keys, etc.). Prefer categorical dtype for repeated strings.

### `var` (per marker col)

* Index: `marker_id` (stable).
* Columns:

  * `display_label` (str)
  * `channel` (raw channel name)
  * `pct_positive` (float, optional)
  * `flowsom_level` (int, optional)

### `uns`

* `row_order`: list[int] — frozen cluster display order.
* `col_order`: list[int] — frozen marker display order.
* `meta_cluster_colors`: Dict[str, str] — hex colors for meta-clusters.
* `dendrogram_cut`: float|int|None
* `selected_channels`: list[str]
* `subset_filters`: JSON-serializable representation of filters/gates that produced the subset (keep the original expression string as `subset_filters_expr` if available).
* `orientation`: `"rows=clusters/cols=markers"` or `"rows=markers/cols=clusters"`.
* **Checkpoint payload:**

  ```json
  uns["checkpoint"] = {
    "id": "<uuid4>",
    "parent_id": "<uuid4 or null>",
    "step_id": "subset_lineage",        // one of: 1,2,3,6,7,8,10,5,9,11
    "description": "Subset (lineage)",
    "created_at": "2025-10-21T10:00:00Z",
    "params": { "channels":[...], "gates": {...}, "cut": ..., "flowsom_cutoffs": {...} },
    "producer": { "package": "yourviewer", "version": "<semver>" },
    "artifact_version": "1.0.0",
    "schema_hash": "<sha1 over normalized schema json>"
  }
  ```
* **Run lineage/DAG:**

  ```json
  uns["lineage"] = {
    "<checkpoint_id>": { "children": ["<child_id>", "..."] },
    "root_id": "<checkpoint_id>"
  }
  ```
* **FlowSOM model snapshot (lightweight):**

  ```json
  uns["flowsom"] = {
    "seed": 42, "grid": [10,10], "n_metaclusters": 20,
    "algorithm_ver": "x.y.z",
    "weights_hash": "<sha256>",         // hash of SOM weights if not stored
    "params": { ... }                   // training params
  }
  ```
* **Version & audit:**

  * `artifact_version: "1.0.0"`
  * `producer: {package, version}`
  * `timestamp` (redundant with `created_at` okay)

### `obsm` (optional)

* `umap`, `tsne`, or `scatter_<name>`: shape `(n_obs, 2)` cached embeddings used alongside the heatmap.

---

## Workflow checkpoints

Create save points after steps: **1, 2, 3, 6, 7, 8, 10** plus **finalizations 5, 9, 11**.

Each checkpoint stores:

* `step_id`, `description`
* parameter snapshot (selected channels, gating thresholds, clustering cutoffs, orientation)
* parent pointer (`parent_id`) to form a **DAG** (branched histories supported)

---

## Implementation plan

### 1) Serializer utility

`viewer/plugin/heatmap_saver.py`

```python
def serialize_heatmap_state(
    display_state,                   # HeatmapDisplay: matrix, orders, colors, config
    metadata: dict,                  # step_id, description, params, parent_id (uuid or None)
    extra_obs_cols: list[str] | None = None,
    include_embeddings: bool = True,
    include_raw_medians: bool = True,
) -> anndata.AnnData:
    """
    Build an AnnData artifact per schema above.
    - Uses canonical orientation (rows=clusters, cols=markers)
    - Adds layers["median"] if include_raw_medians
    - Adds obsm embeddings if include_embeddings
    - Freezes row/col order and saves colors/config in uns
    - Generates checkpoint UUID and updates uns['lineage']
    """
```

### 2) HeatmapDisplay hook

Extend `HeatmapDisplay`:

```python
def collect_heatmap_state(
    step_id: str,
    description: str,
    metadata_cols: list[str] | None = None,
    parent_id: str | None = None,
    include_embeddings: bool = True,
    include_raw_medians: bool = True,
) -> anndata.AnnData:
    # 1) get active oriented matrix (z-scored)
    # 2) gather obs/var + requested provenance columns from main_viewer.cell_table keyed by cluster
    # 3) attach uns: colors, row/col order, config snapshot, flowsom model digest
    # 4) attach obsm/layers as requested
```

### 3) User trigger (UI)

Add **“Save heatmap…”** action:

* Step selector (pre-filled, free text allowed).
* Multi-select **provenance columns** (searchable).
* Toggles: “Include embeddings”, “Include raw medians”.
* File picker (default: `.h5ad`, suggested name `flowsom_<stepid>_<timestamp>.h5ad`).
* Friendly install flow if `anndata` missing (show pip/conda command and retry).

### 4) Checkpoint browser (ipywidgets + traitlets)

* Tree/DAG viewer listing checkpoints with parent/child relationships.
* Row shows badges: `step_id`, n_clusters, n_markers, dendrogram cut, #channels, subset type(s).
* Thumbnail preview: downsampled heatmap of `X`.
* **Diff mode**: select two checkpoints → show parameter deltas, set differences in channels/filters, and basic matrix stats deltas.

### 5) Persistence layer

* Write `.h5ad` per checkpoint.
* Save checkpoint context in `uns['checkpoint']` and update `uns['lineage']`.
* (Roadmap) Run folder with base + delta checkpoints if size becomes an issue.

### 6) Reload hook

`load_heatmap_state(filepath) -> HeatmapDisplay state`:

* Restores matrix, colors, order, selected channels, filters, dendrogram cut, and (if present) embeddings.
* Rehydrates the checkpoint tree UI and selects the loaded node.
* Treat saved palettes as source of truth; don’t recompute unless missing.

### 7) Dependencies

* Add `anndata` to `env/environment.yml` (and pip extras).
* Imports wrapped with informative errors & one-click install help.
* Zarr support behind optional dependency.

---

## Robustness & edge cases

* **Orientation flips**: always save canonical layout; record view orientation in `uns["orientation"]`. Round-trip must not change the visual order.
* **Stable IDs**: `obs_names`/`var_names` must be stable across runs.
* **Large provenance**: cast repetitive strings to `category`.
* **Color overwrites**: never auto-recompute if `meta_cluster_colors` exists.
* **Subset provenance**: store both structured filters (`subset_filters`) and human-readable expression (`subset_filters_expr`).
* **Atomic write**: prevent partial artifacts on crash.

---

## Validation utilities

* `validate_artifact(adata) -> None`
  Checks:

  * shapes of `X`, `layers["median"]` (if present), `obsm` dims
  * presence and types of required fields
  * `row_order`/`col_order` in bounds
  * `schema_hash` matches current schema

---

## Tests (high impact)

1. **Save → load round-trip**: `X` checksum, `obs_names`, `var_names`, orders, colors identical.
2. **Orientation**: UI flip then save; reload must restore original view fidelity.
3. **Diff**: checkpoints 3 vs 6 report expected channel/subset param deltas.
4. **Missing dependency**: friendly error path.
5. **Zarr parity** (if enabled).
6. **Large provenance**: categorical downcast reduces file size (smoke test).

---

## Documentation

* Rationale: *Why AnnData?* (interop, auditability, single-file)
* Quickstart:

  1. Run FlowSOM step → **Save heatmap…**
  2. Browse checkpoints → **Diff** or **Reload**
  3. Branch from step 6 (lineage subset) and continue
* Full schema table (fields, dtypes, optionality).
* Notes on performance & file sizes; when to toggle raw medians/embeddings.

---

## Nice-to-have (later)

* Per-run folder with shared base + delta checkpoints.
* Export/import of **palette themes** across projects.
* CLI: `heatmap-checkpoint diff A.h5ad B.h5ad`.
