# Issue #140 — the Binder example should load S-BIAD2557's cell table

## Problem

`S-BIAD2557` now ships a cell table at `Files/spatial_murine_iCCAvsHCC/cell_table/pCSL005_cell_table.csv`, but the Binder notebook (`script/run_ueler_binder.ipynb`) opens the study with images and masks only. Everything that makes UELer interesting — the heatmap, the scatter plot, the cell gallery, expression-driven mask colouring — stays empty for the visitor who clicks the Binder badge.

Two findings from inspecting the live study turned this from a one-line notebook edit into a small feature:

1. **The study's image layout changed.** `Files/spatial_murine_iCCAvsHCC/image_data/` no longer holds one directory per FOV; each FOV is now a `<FOV>.zip` of channel TIFFs (~100 MB each, 562 of them). The descriptor baked into the Binder notebook, the README, `docs/getting-started.md`, `docs/faq.md` and `script/run_ueler_BIA.ipynb` (`"mode": "folder"` with no `fov_container`) therefore resolves **0 FOVs** and `list_fovs` raises `No FOVs found in the BIA study for the resolved layout.` The zip-container support added for `S-BIAD2708` (`"fov_container": "zip"`) covers the new layout exactly, so this is a descriptor fix, not new loader code.
2. **The cell table is 361 MB** (378,082,471 bytes, ~445 k rows × 50 columns: `cell_id, cell_area, sample_id, X, Y, fov, fov_X, fov_Y, distance_to_border, tissue_zone, in_cyst, sample_type`, 32 marker columns, `UMAP0/1`, `lineage_level1..4`). Streaming it end to end takes roughly 5 minutes at ~1.2 MB/s, and a full `pd.read_csv` is far too heavy for a 2 GB Binder container next to the image cache. Whatever we add has to be able to load **only the rows of the FOVs the visitor will actually open**.

There is also no way to reach a study's cell table through the BIA loader at all: the descriptor describes images, masks and annotations only, and `ueler.load_cell_table` takes a **local** file path. The Binder visitor has no local path, and the study's HTTPS base must be resolved through the BioStudies API rather than hardcoded (issue #110 established that the `fire/` vs `pub/databases/` prefix differs per study — and indeed `S-BIAD2557` has since moved to the `fire/` path, which is exactly why hardcoding it was rejected).

## Proposed solution

Teach the BIA layer about a study's cell table, with a row filter so the download can be trimmed to the FOVs in play, then use it from the Binder notebook.

### 1. Descriptor: a `cell_table` key (`ueler/bia_loader.py`)

```jsonc
"cell_table": "Files/spatial_murine_iCCAvsHCC/cell_table/pCSL005_cell_table.csv"
// or, to name the FOV column explicitly:
"cell_table": {"path": "Files/.../pCSL005_cell_table.csv", "fov_column": "fov"}
```

Normalised by `_normalise_cell_table` into `{"path": str, "fov_column": str}` and carried on `BIALayout.cell_table` alongside the existing mask/annotation sources. Absent key → `None`, and everything below becomes a no-op, so existing descriptors are untouched.

### 2. `BIADataSource.fetch_cell_table(fovs=None, force=False) -> Optional[str]`

Returns the path of a locally cached copy under `<cache>/tables/`, or `None` when the descriptor declares no cell table.

* `fovs=None` → download the whole file once via the existing `_download` (atomic, reused on a second call), cached as `tables/<original name>`.
* `fovs=[...]` (CSV only) → **stream** the remote file line by line and write only the rows whose FOV column value is in the set, into `tables/<stem>__fovs-<n>-<digest>.csv`. Peak memory is one chunk, not the table; the cached subset for the first 12 FOVs of `S-BIAD2557` is 12 MB instead of 361 MB. The header is always preserved, and the filter is applied by column *name* (looked up in the header), not position.
* A non-CSV table (`.h5ad`) ignores `fovs` with a warning and falls back to the whole-file download — row filtering an HDF5 file over HTTP is out of scope.
* `has_cell_table` mirrors the existing `has_masks` / `has_annotations` properties.

Filtering still reads the whole remote file (a CSV has no index), so the network cost is the same as a full download; what it buys is the **memory and disk** ceiling, which is what actually breaks a Binder container.

### 3. Entry points (`ueler/runner.py`)

* `run_viewer_bia(..., cell_table=False, cell_table_fovs=None)` — when `cell_table=True`, fetch and attach the table before the display/`after_all_plugins_loaded` tail, so plugins see a viewer that already has cell data (one pass, no re-render).
* `load_bia_cell_table(viewer, *, fovs=None, force=False, auto_display=True, after_plugins=True)` — the post-hoc path the notebooks use, since the FOV names needed for `fovs=` are only known once the viewer is open (`viewer.available_fovs`). It pulls the data source off the viewer, fetches, and delegates to the existing `load_cell_table(viewer, cell_table_path=...)`, so the refresh/redisplay behaviour is shared rather than duplicated.
* `ImageMaskViewer.data_source` — a read-only property over the existing `_data_source`, so the helper (and a curious user) does not have to reach into a private attribute.

### 4. Notebooks and docs

* `script/run_ueler_binder.ipynb`: descriptor updated to `"fov_container": "zip"` + the `cell_table` key; a new cell after the viewer attaches the table for the first 12 FOVs, with the download size/time stated plainly and the one-line change for "all 562 FOVs" spelled out.
* `script/run_ueler_BIA.ipynb`: example 1 (S-BIAD2557) gets the same descriptor fix, and the "(Optional) Load a cell table" section gains the BIA-native variant next to the existing local-file one.
* `README.md`, `docs/getting-started.md`, `docs/faq.md`: the S-BIAD2557 snippet is corrected (it is currently broken against the live study) and the `cell_table` descriptor key documented.

## Implementation steps

1. `_normalise_cell_table` + `BIALayout.cell_table` + `_layout_from_descriptor` wiring.
2. `BIADataSource`: `_tables_dir`, `has_cell_table`, `cell_table_url`, `fetch_cell_table` (whole-file and streamed FOV-filtered paths).
3. `ImageMaskViewer.data_source` property.
4. `runner`: `run_viewer_bia(cell_table=…, cell_table_fovs=…)`, `load_bia_cell_table`, export from `ueler/__init__.py`.
5. Tests in `tests/test_issue140_bia_cell_table.py` (network mocked), plus the existing `tests/test_issue110_bia_loader.py` suite for regressions.
6. Notebook + documentation updates.

## Verification

Live checks against the real study (network, not mocked):

* zip descriptor → 562 FOVs, 30 channels for FOV 0, one channel read out of the 100 MB zip in ~5 s, mask prefetched.
* cell table header/size read with a range request; row grouping sampled at five offsets (rows are grouped per FOV in contiguous blocks but **not** globally sorted, so no early-exit shortcut is safe — the filter must scan the whole file).
* the whole table downloaded once and parsed, to size the problem honestly: 439,339 rows × 50 columns, 402 MB in pandas, 455 of the 562 FOVs represented.
* `fetch_cell_table(fovs=viewer.available_fovs[:12])` end to end: 12.4 MB / 14,344 rows written in 298 s, 13.1 MB once parsed, and the second call returned the cached subset instantly.
