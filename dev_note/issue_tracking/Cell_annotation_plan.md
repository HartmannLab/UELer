Here’s a copy-pasteable **Markdown checklist** you can hand to your coding agent.

# Cell Annotation Workflow — Implementation Checklist

## 0) Project setup & guardrails

* [ ] Create feature branch `feature/cell-annotation-workflow`
* [ ] Add feature flag `ENABLE_CELL_ANNOTATION=true` (env/config)
* [ ] Wire CI for unit + integration tests for new plugin package
* [ ] Define perf budgets (targets):

  * [ ] DAG load ≤ **150 ms** @ 500 nodes
  * [ ] Save checkpoint ≤ **2 s** @ 2k clusters
  * [ ] Manifest rebuild ≤ **1 s** @ 1k files
* [ ] ✅ **Acceptance:** CI green; flag toggles plugin registration

## 1) Core plugin & storage scaffolding

* [ ] Create `plugins/cell_annotation/`:

  * [ ] `plugin.py` (orchestrator, event hooks: dataset opened/closed)
  * [ ] `store.py` (`.UELer` resolver, atomic write helpers)
  * [ ] `manifest.py` (read/update/rebuild)
  * [ ] `selection_spec.py` (predicate/materialized/hybrid; subset/union ops)
* [ ] Hook into `MainViewer` lifecycle to init store on dataset open
* [ ] ✅ **Acceptance:** `.UELer/dataset_<id>/{checkpoints,thumbnails,selections}` exist

## 2) Cross-plugin interfaces

* [ ] Add `viewer/interfaces.py` with:

  * [ ] `SelectionSpec` protocol
  * [ ] `HeatmapStateProvider` (export/import)
  * [ ] `FlowsomParamsProvider` (export/import, `set_selection_context`, `run_flowsom`)
* [ ] Register providers from Heatmap & FlowSOM on init
* [ ] ✅ **Acceptance:** orchestrator detects Heatmap; FlowSOM optional

## 3) Serializer, schema & validator

* [ ] Implement `serialize_heatmap_state(display, flowsom, meta) -> AnnData`
* [ ] Persist:

  * [ ] Canonical orientation; `X` float32; `layers["median"]`
  * [ ] `uns["artifact"]` (version, schema_hash, checksums)
  * [ ] `uns["ui"]`, `row_order`, `col_order`, palettes
  * [ ] `uns["zscore_params"]` (standard/robust, per-marker stats, clipped)
  * [ ] `uns["filters"]`
  * [ ] `uns["row_linkage"]` + `uns["row_linkage_basis"]`
  * [ ] **Marker sets:** `training`, `display_extra`, `available`, `linkage`, `expanded_training`
  * [ ] **Checkpoint:** `id` (UUIDv7), `parents` (multi), `op`, `created_at`, `producer`
  * [ ] **FlowSOM snapshot** (training_markers, imputation/projection, availability, params)
* [ ] Atomic writer for `.h5ad` (same-dir temp → fsync → `os.replace`)
* [ ] Validator enforcing invariants & checksums
* [ ] ✅ **Acceptance:** round-trip save/load identical on a synthetic dataset

## 4) Manifest & thumbnails

* [ ] Implement atomic `manifest.json` update & rebuild (ignore `*.partial`)
* [ ] Generate thumbnails (mini heatmap or sampled UMAP) on save
* [ ] ✅ **Acceptance:** DAG renders from manifest; rebuild works; thumbnails present

## 5) Checkpoint Browser UI

* [ ] DAG/tree view + search/filter (step/op/text)
* [ ] Details pane: badges (`op`, `n_clusters`, `n_markers`, cuts, **marker sets**)
* [ ] Actions:

  * [ ] **Save checkpoint…** (size estimate, toggles)
  * [ ] **Load**
  * [ ] **Diff** (param deltas; top-K marker L2 shifts on `X`)
  * [ ] **Branch/Rebase**
  * [ ] **Merge** (multi-select parents)
* [ ] ✅ **Acceptance:** end-to-end save → manifest update → UI refresh; diff renders

## 6) Heatmap upgrades

* [ ] `export_heatmap_state`: include marker sets, `var.role`, `var.available`, `var.missing_rate`, linkages, palettes, filters, zscore params
* [ ] `import_heatmap_state`: restore order/palette/orientation/filters/embeddings
* [ ] Render **training + display-extra**; group columns by section
* [ ] Row dendrogram strictly uses `marker_sets.linkage`
* [ ] ✅ **Acceptance:** loading a checkpoint restores identical view; extras don’t affect row order

## 7) FlowSOM upgrades

* [ ] Extend API:

  * [ ] `run_flowsom(selection, params, training_markers, extra_markers, imputation, projection)`
  * [ ] Compute medians for `training ∪ extra`; train only on `training`
* [ ] Subset-only mode under `set_selection_context` (disallow widening; prompt “Merge”)
* [ ] Record availability; export snapshot
* [ ] ✅ **Acceptance:** runs with intersection; supports expanded training (impute/projection); blocks supersets

## 8) Marker Manager (in Cell Annotation)

* [ ] Training markers:

  * [ ] **Intersection (default)**
  * [ ] **Include more markers** (select from union with availability bars)
  * [ ] Imputation strategy: median / panel_median / kNN
  * [ ] **Advanced:** complete-cases + projection
* [ ] Display-extra markers (searchable multi-select)
* [ ] Linkage set: default **intersection**; optional “use training set” (warn if expanded)
* [ ] ✅ **Acceptance:** FlowSOM uses declared sets; heatmap shows extras; linkage stable

## 9) Merge workflow

* [ ] Multi-select parents → **Merge**
* [ ] Validate dataset/panel compatibility
* [ ] Build union `SelectionSpec` (materialize Parquet handle as needed)
* [ ] Create node `op="merge"`, `parents=[...]`, `uns["selection"]`
* [ ] Modes:

  * [ ] **Merge only** (no clustering)
  * [ ] **Merge & Recluster** (invoke FlowSOM with chosen marker sets)
* [ ] ✅ **Acceptance:** merged node saved; recluster path yields valid heatmap; downstream subset rules enforced

## 10) Subset-only invariants (enforcement)

* [ ] Validate child ⊆ parent via `SelectionSpec` (exact or counted pre-check)
* [ ] FlowSOM UI disables widening under context; offer “Merge” handoff
* [ ] Clear errors (which predicate widened; % outside parent)
* [ ] ✅ **Acceptance:** any superset attempt is blocked with actionable guidance

## 11) Save/Load wiring

* [ ] Save: Heatmap export → optional FlowSOM snapshot → serialize → atomic write → manifest/thumbnail update → event emit
* [ ] Load: read/validate → Heatmap import → optional FlowSOM import → event emit
* [ ] ✅ **Acceptance:** both flows stable across restarts & across feature-flag toggles

## 12) Testing & QA

* **Unit**

  * [ ] Serializer/validator (checksums, invariants, linkage/zscore capture)
  * [ ] `SelectionSpec` ops (subset/union); parquet handles
  * [ ] Atomic write helpers
* **Golden**

  * [ ] Known checkpoints; checksum assertions
* **Integration**

  * [ ] Save→Load
  * [ ] Merge→Recluster (with different panels)
  * [ ] Expanded training (impute + projection)
  * [ ] Back-compat (old `parent_id` only)
* **Scale/Perf**

  * [ ] 1k-node manifest perf within budget
  * [ ] Large selections materialized; streaming to FlowSOM
  * [ ] Windows path + atomicity
* [ ] ✅ **Acceptance:** test suite green; budgets met

## 13) Documentation & UX polish

* [ ] Docs:

  * [ ] User guide (Browser, Merge, Marker Manager, expanded training)
  * [ ] Schema tables (`uns`, `obs`, `var`)
  * [ ] Dev guide (interfaces, selection spec, serializer)
* [ ] In-UI tooltips: missing markers, imputation warnings, op badges
* [ ] ✅ **Acceptance:** a new user can reproduce a workflow from docs alone

## 14) Release & migration

* [ ] Bump artifact `version` → `1.0.0`
* [ ] Migration script: scan existing checkpoints → build manifest; map `parent_id` → `parents=[...]`
* [ ] Flip feature flag default **on**
* [ ] Changelog; deprecate duplicate legacy save buttons
* [ ] ✅ **Acceptance:** legacy checkpoints appear in Browser; no regressions reported
