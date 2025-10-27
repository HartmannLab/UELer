# Cell Annotation Workflow — Orchestrated Checkpoints, Merging, and Flexible Marker Sets

**Goal:** capture every heatmap state (matrix + annotations) in a portable, self-sufficient artifact so the FlowSOM workflow (steps 1–11: initial clustering, revisions, mixed/lineage subsets, and finalizations) can be resumed, branched, diffed, merged, or audited at any stage—**with a dedicated _Cell Annotation_ plugin** coordinating Heatmap and FlowSOM.

---

## 1) Architecture Overview (Plugin Boundary–Aware)

### Roles
- **Cell Annotation (new, orchestrator)**
  - Owns the **checkpoint lifecycle**: save, load, diff, branch/rebase, **merge**.
  - Hosts the **Checkpoint Browser UI** (DAG/tree, thumbnails, diffs).
  - Coordinates Heatmap & FlowSOM via **small interfaces**.
  - Manages **selection semantics** (subset-only downstream; merge as the only union exception).
  - Persists artifacts in the dataset’s **`.UELer`** folder (co-located with images).
  - Maintains a lightweight **`manifest.json`** for fast DAG rendering.
  - Ensures **atomic writes**, checksums, and schema/version validation.

- **Heatmap (existing)**
  - Owns **visualization** and user-facing heatmap state.
  - **Exports** current display state (matrix, orders, palettes, filters, z-score params, embeddings, optional raw medians, marker sets & roles).
  - **Imports** a loaded checkpoint state to restore view.
  - Renders **training + display-extra** markers; row clustering uses **training set only**.

- **FlowSOM (existing)**
  - Owns **clustering** and metaclustering runs.
  - **Exports** a lightweight FlowSOM snapshot (params, seed, grid, distance, version/deps, hashes).
  - Optionally **imports** specific FlowSOM params when resuming (soft-fail if incompatible).
  - Trains on **declared training markers**; can compute medians for **training ∪ display-extra** markers.

### Cross-Plugin Interfaces
Create `ueler/viewer/interfaces.py`:

```python
from typing import Protocol, Mapping, Any
import numpy as np
import pandas as pd

class SelectionSpec(Protocol):
    """Opaque handle for a cell selection (predicate, materialized Parquet, or hybrid)."""
    ...

class HeatmapStateProvider(Protocol):
    def export_heatmap_state(
        self, *, include_embeddings: bool,
        include_raw_medians: bool,
        extra_obs_cols: list[str] | None
    ) -> dict: ...
    def import_heatmap_state(self, adata_path: str) -> None: ...

class FlowsomParamsProvider(Protocol):
    def export_flowsom_params(self) -> dict: ...
    def import_flowsom_params(self, params: Mapping[str, Any]) -> None: ...
    def set_selection_context(self, selection: SelectionSpec) -> None: ...
    def run_flowsom(
        self,
        selection: SelectionSpec,
        params: Mapping[str, Any] | None = None,
        training_markers: list[str] = [],
        extra_markers: list[str] | None = None,
        imputation: dict | None = None,     # {"strategy": "median|panel_median|knn", "params": {...}, "seed": 42}
        projection: bool = False            # train on complete cases + project the rest
    ) -> Mapping[str, Any]: ...
```

**Notes**
- `Cell Annotation` calls `run_flowsom(...)` for **Recluster** and **Merge & Recluster**.
- Heatmap and FlowSOM register themselves with Cell Annotation on init.

---

## 2) Filesystem Layout (per Dataset)

All checkpoints live **next to the images** in a hidden folder:

```
<dataset_root>/
  images/...
  .UELer/
    dataset_<hash-or-stable-id>/
      manifest.json
      checkpoints/
        <uuidv7>__op-merge__20251024T0830Z.h5ad
        <uuidv7>__op-subset__20251024T0900Z.h5ad
      thumbnails/
        <uuidv7>.png
      selections/
        <uuid>.parquet   # materialized (fov_id, cell_id) for selection handles
```

- `dataset_<hash>` derives from a stable identifier (path + image set signature).
- If root is read-only, fall back to a user cache and mark `"storage_fallback": true` in the manifest.
- **Atomic writes** for artifacts and `manifest.json` (`*.partial` → fsync → `os.replace`).

---

## 3) Workflow Semantics: Subset-Only, with Merge Exception

- **Downstream nodes** must be **subsets** of their parent selection (cellwise).  
- **Merges** are the sole union exception: a merged node may have **multiple parents** and represents the **OR** of parent selections.
- The FlowSOM plugin enforces **subset-only controls** given a node context; widening attempts route to **Merge** via Cell Annotation.

### SelectionSpec
Represent selections as:
- **Predicate AST** (channels/gates/metadata); and/or
- **Materialized** cell IDs (Parquet: columns `[dataset_id, fov_id, cell_id]`); and/or
- **Hybrid** (predicate + exclusions).

Cell Annotation validates subset relations using materialized sets (exact) or counts (fast pre-checks).

---

## 4) Merge in the Checkpoint Browser

### User Flow
- Multi-select ≥2 **compatible** nodes (same dataset); click **Merge…**.
- Dialog options:
  - **Merge only** (create union node, no clustering).
  - **Merge & Recluster** (immediately run FlowSOM on the union).
  - Choose **training** and **display-extra** markers (defaults below).

### Node Semantics & Schema
- New node has **multiple parents** and `op="merge"`.
- All downstream nodes must be **subsets** of this merged selection.

Artifact (`uns` excerpts):

```json
uns["checkpoint"] = {
  "id": "<uuidv7>",
  "parents": ["<uuidA>", "<uuidB>"],
  "parent_id": "<uuidA>",           // kept for backward compat (first parent)
  "op": "merge|subset|recluster|finalize",
  "step_id": "6",
  "description": "Merged unknown+ambiguous",
  "created_at": "2025-10-24T08:30:00Z",
  "params": {...},
  "producer": {"package":"ueler","version":"<semver>","deps": {...}},
  "id_namespace": "<project/run>"
}

uns["selection"] = {
  "spec": { "type":"merge", "of": ["<uuidA>", "<uuidB>"] },
  "n_cells": 123456,
  "dataset_id": "dataset_<hash>",
  "handle": "selection://<uuid>",
  "hash": "<sha256-over-sorted-(dataset_id,cell_id)>"
}
```

Manifest entry:

```json
{"id":"<child>","parents":["<A>","<B>"],"op":"merge","path":"checkpoints/<child>.h5ad", "...": "..."}
```

---

## 5) Per-Node Marker Sets & Heatmap Extras

Each node records separate marker sets:

```json
uns["marker_sets"] = {
  "training": ["CD3", "CD8", "CD4", "..."],    // used for FlowSOM & row linkage by default
  "display_extra": ["PD1", "Ki67"],            // shown; excluded from FlowSOM & linkage
  "available": ["CD3","CD8","CD4","PD1","Ki67","..."],  // present in this node
  "linkage": ["CD3","CD8","CD4"],              // markers used to compute row dendrogram
  "expanded_training": false,                  // set true if user expands training beyond intersection
  "panel": { "id":"<panel_id>", "version":"<ver>" }
}
```

- `var["role"] ∈ {"training","display_extra","both"}`
- `var["available"] : bool` (per marker)
- `var["missing_rate"] : float` (per marker in current selection)

**Heatmap behavior**
- Render **training + display-extra** columns; visually group sections.
- Row dendrogram uses **`marker_sets.linkage`** (by default = **training ∩ fully-available** or plain **intersection**).  
- Z-score all **displayed** markers; NaNs ignored; show a subtle “(missing)” badge where applicable.

**Merges across panels**
- `available` = **union**; default `training` = **intersection** (user can curate).
- `display_extra` = any subset of `available` (missing values become NaN in medians/X).

---

## 6) Expanded Training Markers (Opt-In)

Default training set = **intersection**. Users may **include more markers** for FlowSOM using one of two strategies:

1. **Union + Impute (recommended default when expanded)**
   - Train on **expanded training** (union subset chosen by user).
   - Impute missing per-cell values (deterministic):
     - **`median`** within current selection (default), or
     - **`panel_median`** (median per acquisition panel), or
     - **`knn`** (kNN within selection; e.g., `k=10`).
   - Record imputation details in `uns["flowsom"].imputation`.

2. **Train on Complete Cases + Project (advanced)**
   - Train FlowSOM on cells with **all** expanded markers.
   - **Project** remaining cells using only the **intersection** basis.
   - Record training subset size and projection metrics.

Schema additions:

```json
uns["flowsom"] = {
  "training_markers": ["..."],            // == marker_sets.training
  "imputation": {"enabled": true, "strategy":"median|panel_median|knn", "params":{"k":10}, "seed":42},
  "projection": {"enabled": true, "train_cells":123456, "basis":"intersection", "metrics":{"oob_rate":0.02}},
  "availability": {"CD3":1.0, "PD1":0.62, "...":0.85},
  "seed": 42, "grid":[10,10], "n_metaclusters":20, "distance":"euclidean",
  "pre_scaling":"zscore_standard",
  "weights_hash":"<sha256>", "codebook_hash":"<sha256>",
  "params": {...}, "deps":{"anndata":"...", "scanpy":"...", "numpy":"..."}
}

uns["row_linkage_basis"] = {"marker_ids": ["..."], "distance":"correlation"}
```

**Best practice:** keep `marker_sets.linkage` = **intersection** to stabilize row order even when training expands. Allow override with a warning.

---

## 7) Artifact Format (`.h5ad` primary; Zarr optional)

- **Primary:** `.h5ad` via `AnnData.write_h5ad` with **gzip** compression and sensible **chunking** (e.g., `X` chunks ≈ `(min(n_obs,256), min(n_var,32))`).  
- **Optional:** **Zarr** for very large runs, backed I/O, or deltas (use `blosc/zstd`).  
- **Atomic writes:** same-dir temp (`*.partial`) → `flush()` + `os.fsync(file)` → `os.fsync(dir)` → `os.replace`.  
- `X`: **z-scored medians** (cluster × marker), **float32**.  
- `layers["median"]`: **raw medians** (float32).  
- `obs`/`var`: stable IDs; repeated strings → **category**; `string[pyarrow]` optional for large uniques.  
- `obsm`: `(n_obs, 2)` embeddings (`umap`, `tsne`, etc.).

`uns` key highlights:
```json
uns["artifact"] = {"version":"1.0.0","schema_hash":"<canonical>",
                   "checksums":{"X_sha256":"...","layers.median_sha256":"..."}}
uns["ui"] = {"orientation":"rows=clusters/cols=markers","row_sort":"frozen","col_sort":"frozen",
             "selected_channels_ordered":[...]}
uns["palette"] = {"meta_cluster_colors_present": {...}, "meta_cluster_colors_all": {...}}
uns["zscore_params"] = {"method":"standard|robust","per_marker":{"CD3":{"mean":..,"std":..}}, "clipped":false}
uns["filters"] = {"expr":"...", "structured": {...}, "source":"viewer|api|script"}
uns["row_linkage"], uns["col_linkage"], "dendrogram_cut"
uns["marker_sets"], uns["row_linkage_basis"]
uns["flowsom"] (as above)
uns["checkpoint"] (as above)
```

---

## 8) Manifest (`.UELer/.../manifest.json`)

```json
{
  "dataset_id": "dataset_<hash>",
  "updated_at": "2025-10-24T08:00:00Z",
  "storage_fallback": false,
  "checkpoints": [
    {"id":"<uuidv7>","parents":[], "op":"subset","step_id":"1",
     "description":"Initial clustering",
     "path":"checkpoints/<uuid>.h5ad","n_clusters":120,"n_markers":35}
  ],
  "thumbnails": {"<uuidv7>":"thumbnails/<uuidv7>.png"}
}
```

- Updated atomically. If missing/corrupt, rebuild by scanning `checkpoints/` (ignore `*.partial`).

---

## 9) Save / Load Flows

### Save (from either plugin)
1. Cell Annotation queries Heatmap: `export_heatmap_state(...)`.
2. Optionally queries FlowSOM: `export_flowsom_params()`.
3. Builds `AnnData` via `serialize_heatmap_state(display, flowsom, meta)`.
4. Writes under `.UELer/.../checkpoints` atomically; updates `manifest.json`; creates/updates thumbnail.
5. Emits event: `checkpoint:saved` with `{id, path}`.

### Load (from Browser)
1. Read `.h5ad`, validate schema & checksums.
2. Heatmap: `import_heatmap_state(path)` (source of truth for view).
3. If FlowSOM exists and artifact includes `uns["flowsom"]`, call `import_flowsom_params(...)` (soft-fail if incompatible).
4. Emit `checkpoint:loaded` with `{id}`; Browser selects node.

---

## 10) Checkpoint Browser UI

- **Left:** DAG/tree (from manifest) with search/filter (by step, text, op).
- **Right:** details (badges: `op`, `n_clusters`, `n_markers`, `cutoffs`, **marker sets**; palette diffs; channel lists).
- **Actions:**
  - **Save checkpoint…** (dialog: step, provenance allowlist, include embeddings/raw medians, suggested filename, live **size estimate**).
  - **Load**, **Diff** (param deltas; top‑K marker shifts by L2 on `X`).
  - **Merge** (multi-select parents) with **Merge only** or **Merge & Recluster**.
  - **Branch/Rebase from here** (pre-fills `parents` or `parent_id`).
- **Marker Manager** (in Cell Annotation):
  - Training markers: **Intersection (recommended)**, checkbox **Include more markers** (select from union with availability bars).
  - **Imputation** strategy or **Train on complete cases + Project** (advanced).
  - Display-extra markers (searchable multi-select).
  - Linkage markers default to **intersection**; optional “Use training set” (warn if expanded+imputed).

---

## 11) Validation & Invariants

- **Selection semantics**
  - Non-merge child nodes: `cells(child) ⊆ cells(parent)` (validated via SelectionSpec; hard-fail on violation; propose Merge).
  - Merge nodes: `cells(merge) = ⋃ cells(parents)`; parents must be same dataset & compatible.

- **Marker set semantics**
  - `training ⊆ available`
  - `display_extra ⊆ available`
  - `uns["flowsom"]["training_markers"] == marker_sets.training`
  - `uns["row_linkage_basis"]["marker_ids"] == marker_sets.linkage`
  - When expanded training is used without imputation/projection, training must equal the **intersection** (no missing allowed).

- **Heatmap fidelity**
  - Row order persists (linkage computed on `marker_sets.linkage` only).
  - Z-score params recorded for **all displayed markers**.

- **Artifact integrity**
  - Unique, non-empty `obs_names`/`var_names`.
  - Colors are valid hex; orders are proper permutations.
  - Checksums for `X` and `layers["median"]` match.

---

## 12) Implementation Notes

- **Chunking (`.h5ad`)**: start with `(min(n_obs,256), min(n_var,32))`; benchmark against row-wise access.
- **Dtypes**: `float32` for matrices; `category` for repeated strings; consider `string[pyarrow]` for large uniques.
- **Atomicity**: write temp in same dir; fsync file + dir; `os.replace`.
- **UUIDs & Time**: use **UUIDv7** for checkpoint IDs; timestamps in **UTC** ISO‑8601 with trailing `Z`.
- **Z-scores**: store `uns["zscore_params"]` with either `mean/std` or `median/mad` (robust), and `clipped` flag.
- **Row linkage**: store `row_linkage` matrix + `row_linkage_basis` (distance + markers).

---

## 13) Testing Matrix

1. **Round-trip fidelity**: orders, palettes, orientation, checksums identical after load.
2. **Subset enforcement**: FlowSOM cannot widen selection; UI offers Merge instead.
3. **Merge DAG**: multi-parent node persisted; manifest reflects edges.
4. **Merge & Recluster**: end‑to‑end run; medians for training ∪ extra; snapshot recorded.
5. **Different marker sets downstream**: training subset + display-extra superset work; linkage stable.
6. **Expanded training + Impute**: FlowSOM runs; `imputation.strategy` recorded; linkage remains intersection-based.
7. **Projection path**: train on complete cases; project rest; metrics captured.
8. **Row order stability**: toggling extra markers doesn’t alter linkage rows.
9. **Schema guards**: role/availability/missing_rate align; flowsom.training_markers and linkage basis match marker_sets.
10. **Large lineage**: 1k nodes stay responsive via manifest.
11. **Windows atomicity**: no orphan `*.partial`, browser ignores partials.
12. **Category round-trip**: category dtypes preserved (categories + order).

---

## 14) Minimal Service & Stubs (Sketch)

```python
# ueler/viewer/plugins/cell_annotation/plugin.py
class CellAnnotationPlugin:
    def register_heatmap(self, provider: HeatmapStateProvider): ...
    def register_flowsom(self, provider: FlowsomParamsProvider): ...

    def save_checkpoint(self, *, step_id, description, parents,  # parents: list[str]
                        include_embeddings=True, include_raw_medians=True,
                        extra_obs_cols=None, suggested_name=None) -> str: ...

    def merge_nodes(self, parent_ids: list[str], *, recluster: bool, marker_sets: dict, flowsom_params: dict): ...

    def load_checkpoint(self, checkpoint_id: str) -> None: ...
    def diff_checkpoints(self, a_id: str, b_id: str) -> dict: ...
```

```python
# FlowSOM API usage (Cell Annotation)
result = flowsom.run_flowsom(
    selection=current_selection_spec,
    params=flowsom_params,
    training_markers=marker_sets["training"],
    extra_markers=marker_sets["display_extra"],
    imputation={"strategy":"median"},
    projection=False
)
clusters_df, medians, snapshot = result["clusters_df"], result["medians"], result["snapshot"]
```

```python
# Heatmap import (from Cell Annotation on load)
heatmap.import_heatmap_state(path_to_checkpoint_h5ad)
```

---

## 15) Rationale (TL;DR)

- **Separation of concerns**: visualization (Heatmap), modeling (FlowSOM), provenance/workflow (Cell Annotation).
- **Determinism & auditability**: z-score provenance, linkage basis, checksums, explicit marker-set roles.
- **Flexibility**: per-node marker sets, opt‑in expanded training with imputation/projection, multi-parent merges.
- **Portability**: artifacts co-located under `.UELer`, single-file `.h5ad` with manifest acceleration.
