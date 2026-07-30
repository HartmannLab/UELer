# Issue #123 — Support AnnData as a cell-table input

> GitHub issue: [#123](https://github.com/HartmannLab/UELer/issues/123)
> Status: implemented (see *Implementation* below)

## Problem

> The current pipeline only supports cell tables in the form of a Pandas DataFrame. This is fine so
> far, and probably the user can convert an annData object to a Pandas DataFrame. However, it would
> be more convenient if the pipeline could directly accept an annData object as a cell table input.
> This would allow users to seamlessly integrate UELer into their existing workflows that use annData
> for single-cell data analysis. Please implement support for annData as a cell table input in the
> pipeline, ensuring that the relevant data (e.g., cell metadata) is correctly extracted and used
> within UELer.
>
> **Recommendations**
> - Keep the annData as the object type for the cell table input, when the user provides an annData
>   object. This would allow for more efficient data handling and avoid unnecessary conversions.
> - The columns of the variables (var) and the observations (obs) in the annData object should be
>   exposed as selectable columns in the cell table UI. The main difference between the Pandas
>   DataFrame and annData object is that the annData has to this separation of variables and
>   observations.
> - The front-end UI and UX should stay consistent with the current implementation.

### Starting point

Two entry points existed, neither of which validated anything:

- [`ImageMaskViewer.load_cell_table_from_path`](../../ueler/viewer/main_viewer.py) — `pd.read_csv`
  plus a float→`Int64` downcast for columns whose values are all integral.
- [`ImageMaskViewer.set_cell_table`](../../ueler/viewer/main_viewer.py) — literally
  `self.cell_table = cell_table`.

Passing an AnnData through `set_cell_table` therefore "worked" up to the first plugin that touched
the table, then failed with `AttributeError`. Users flattened by hand instead;
`script/run_ueler_MCB.ipynb` does exactly that (`read_h5ad` → rename `cell_id`→`label` →
`pd.merge` into the CSV table).

### The constraint that shapes the design

`viewer.cell_table` is read at roughly **108 sites across 15 plugin modules**, all of which assume
pandas: `.loc` / `.iloc` / `.index` / `.columns` / `select_dtypes` / `groupby().median()` /
`pd.merge(cell_table, …)` / `df.query(…)` / `.iterrows()` / in-place `drop` / column assignment. Two
of those are pandas *free functions* operating on the object
([`heatmap_layers.save_to_cell_table`](../../ueler/viewer/plugin/heatmap_layers.py),
[`export_fovs.refresh_cell_options`](../../ueler/viewer/plugin/export_fovs.py)), so an
adapter object emulating the pandas API could not satisfy them without becoming a `DataFrame`
subclass.

## Solution

Keep the AnnData **and** derive a DataFrame view of it:

```
adata (the user's object, retained)      viewer.cell_table (pandas, RangeIndex)
├── obs      ──────────────────────▶     obs columns          (metadata / clusters)
├── X / layers[layer] + var_names ──▶     marker columns       (features)
├── obsm (width ≤ 3)  ─────────────▶     X_umap1, X_umap2 …
└── obs_names         ─────────────▶     obs_names column
        ▲                                        │
        └──── sync new/edited columns ◀──────────┘  (on `on_cell_table_change`)
```

Design decisions:

1. **DataFrame view rather than an AnnData-backed adapter.** Every existing plugin keeps working
   untouched, and `obs` + `var` columns are both selectable with no UI change — the issue's second
   and third recommendations. The first recommendation ("keep the annData as the object type") is
   honoured in the sense that matters: the user's object is retained on
   `viewer.cell_table_adata`, is kept in sync, and is what `get_cell_table_adata()` hands back for
   `write_h5ad`. Marker columns keep their `float32` dtype, so the view costs one copy of `X`, not a
   float64 doubling.
2. **`RangeIndex`, with `obs_names` kept as a column.** The plugins mix label-based access
   (`idx in cell_table.index` in [`_chart_common`](../../ueler/viewer/plugin/_chart_common.py)) with
   *positional* access (`df.iloc[cell_index]` in
   [`cell_gallery`](../../ueler/viewer/plugin/cell_gallery.py)); those agree only on a `RangeIndex`,
   which is exactly what `pd.read_csv` has always produced. Keeping AnnData's string `obs_names` as
   the index would have made lasso→gallery selection silently select nothing.
3. **`category` obs columns are widened.** See *Root causes found while implementing* below.
4. **Reordering, not filtering, for the marker pickers.** `var` columns are listed before `obs`
   columns so real markers are not buried under `label`/`area`/`X`/`Y` in the picker's scrollable
   list (#125). Membership is unchanged, and it is a no-op for a DataFrame table.
5. **Write-back at a single choke point.** Every cell-table write path already ends with
   `inform_plugins('on_cell_table_change')` ([`run_flowsom`](../../ueler/viewer/plugin/run_flowsom.py),
   [`heatmap_layers`](../../ueler/viewer/plugin/heatmap_layers.py),
   [`cell_table_editor`](../../ueler/viewer/plugin/cell_table_editor.py)), so the sync hooks in there
   rather than in each plugin.

### Root causes found while implementing

Both were found by exercising the real `anndata` (0.11.4) / `pandas` (2.3.3) in the project env, and
both only bite after an `.h5ad` round-trip — which is how most users will arrive.

- **`category` dtype hid every class column.** `write_h5ad` calls `strings_to_categoricals`, so a
  re-read `obs` returns `category` for every string column. The eight
  `select_dtypes(include=['int', 'int64', 'object'])` call sites that populate the cluster /
  subset-on / identifier / tooltip-label dropdowns return **only** `['label']` for such a table —
  `fov` and `cell_type` disappear from the UI entirely.
- **`category` dtype broke manual labelling.** `Categorical` rejects assignment of a value outside
  its categories:

  ```
  TypeError: Cannot setitem on a Categorical with a new category (Z), set the categories first
  ```

  which is precisely `ct.loc[mask, column_name] = value` in the Cell Table Editor. So preserving the
  dtype faithfully would have made the editor crash on any AnnData table.

  **Fix:** `flatten_anndata` widens `category` columns — numeric categories become a real numeric
  column, everything else becomes `object` — so the derived frame behaves exactly like a CSV-loaded
  one. `write_h5ad` re-categorises strings on the way out, so the round-trip is not lossy. The
  `select_dtypes` sites were *also* consolidated behind `categorical_columns()` (which accepts
  `category`/`string`), because a user may legitimately pass `adata.obs` straight in as a DataFrame.

### Out of scope (deliberately)

- Making `viewer.cell_table` itself an AnnData-backed adapter object (see the constraint above).
- Persisting automatically — write-back is in-memory; the user calls
  `viewer.get_cell_table_adata().write_h5ad(...)`.
- `MuData` / multi-modal inputs, `adata.raw`, and reading more than one `layer` at a time.

## Implementation

- **New** [`ueler/cell_table.py`](../../ueler/cell_table.py) — all AnnData logic lives here; no other
  module imports `anndata` for the cell table.
  - `is_anndata(obj)` — duck-typed on `obs`/`var`/`X`/`n_obs`, so `anndata` stays out of the viewer's
    import path and AnnData-like objects take the same route.
  - `flatten_anndata(adata, *, layer=None, obsm_keys=None, max_obsm_width=3)` → `(frame, provenance)`.
    Positional re-indexing (not concat-by-`obs_names`, which duplicate names would scramble),
    `_decategorise` for the `category` widening, `adata.to_df(layer=…)` for the markers with a warning
    when a sparse `X` has to be densified, numbered columns for `obsm` entries ≤ 3 wide (wider ones
    need an explicit `obsm_keys`), and `_unique_name` suffixing (`_var`/`_obsm`) when a marker name
    collides with an obs column — obs wins.
  - `sync_cell_table_to_obs(adata, frame, provenance)` — positional write-back of every non-marker,
    non-index column; skips columns whose values are unchanged (dtype-insensitively, so untouched
    `category` columns keep their dtype in the user's object), drops obs columns it added earlier and
    that have since been removed from the frame, and no-ops with a warning when the row count no
    longer matches `n_obs`.
  - `dataframe_to_anndata(frame, provenance, *, system_keys, source)` — the export direction; undoes
    the marker renames, restores `obs_names`, and carries `var`/`obsm`/`uns` over from the source.
    Without provenance (a CSV table) `X` is the numeric columns minus the five viewer key columns.
  - `categorical_columns(frame, *, include_bool=False)` — the single class/cluster/grouping-column
    rule, `['int', 'int64', 'object', 'category', 'string']` (+ `bool` on request).
- [`ueler/viewer/main_viewer.py`](../../ueler/viewer/main_viewer.py)
  - `__init__` gained `cell_table_adata` and `cell_table_columns` (both `None` for a DataFrame table,
    so every DataFrame code path is unchanged).
  - `set_cell_table(cell_table, *, layer=None, obsm_keys=None)` dispatches on `is_anndata`; the two
    new kwargs raise for a plain DataFrame instead of being silently ignored.
  - `load_cell_table_from_path` dispatches on the `.h5ad` suffix; the CSV branch is untouched and now
    routes through `set_cell_table` so the AnnData attributes are reset.
  - **New** `sync_cell_table_to_adata()` and `get_cell_table_adata()`.
  - `inform_plugins` syncs before the plugin loop when the broadcast is `on_cell_table_change`, in its
    own `try/except` — the loop's bare `except AttributeError` must not swallow a sync failure.
- [`ueler/viewer/plugin/_chart_common.py`](../../ueler/viewer/plugin/_chart_common.py) — **new**
  `marker_first(viewer, columns)`, applied in `numeric_columns()`; `build_subset_controls` also
  accepts `categorical_columns`.
- `categorical_columns` replaced the eight `select_dtypes` call sites in
  [`annotation_display.py`](../../ueler/viewer/annotation_display.py),
  [`heatmap.py`](../../ueler/viewer/plugin/heatmap.py),
  [`heatmap_layers.py`](../../ueler/viewer/plugin/heatmap_layers.py) (×2),
  [`run_flowsom.py`](../../ueler/viewer/plugin/run_flowsom.py) (×2) and
  [`mask_painter.py`](../../ueler/viewer/plugin/mask_painter.py) (`include_bool=True`). The float-only
  continuous-colour selector in `mask_painter` was already correct and is unchanged.
- [`ueler/viewer/plugin/run_flowsom.py`](../../ueler/viewer/plugin/run_flowsom.py) — **new**
  `_feature_columns(main_viewer)` puts markers first in the feature picker; FlowSOM still offers *all*
  columns, so this is ordering only.
- [`ueler/runner.py`](../../ueler/runner.py) — `load_cell_table` gained `layer` and `obsm_keys` and
  forwards them; `_normalise_file` already accepted any existing file, so `.h5ad` needed no change.

## Tests

**New** [`tests/test_cell_table_anndata.py`](../../tests/test_cell_table_anndata.py) — 41 tests in
five groups: `flatten_anndata` (column order/provenance, `RangeIndex` + retained `obs_names`,
`float32` markers, `layer=`, narrow vs wide `obsm`, name-clash suffixing, sparse `X`, obs-only
AnnData, duplicate `obs_names`), categorical handling (the `.h5ad` round-trip premise, the widening,
assigning an unseen class the way the editor does, numeric categories, **and a guard that a
CSV-style DataFrame yields exactly the pre-refactor `select_dtypes(['int','int64','object'])`
result**), write-back (new column, edited column, markers never written, unchanged columns not
rewritten, drop propagation, row-count-mismatch no-op), the export direction, viewer integration
(`set_cell_table` both ways and the reset between them, `.h5ad` and CSV paths, the
`on_cell_table_change` broadcast vs. other broadcasts, `get_cell_table_adata`), and marker-first
ordering.

It carries the anndata-before-`tests.bootstrap` prologue used by `test_cell_annotation.py` /
`test_checkpoint_store.py` (a dask stub with `__spec__ is None` makes anndata's `find_spec("dask")`
raise).

```bash
python -m unittest tests.test_cell_table_anndata tests.test_heatmap_marker_selection \
    tests.test_cell_table_editor tests.test_heatmap_footer tests.test_histogram_plugin \
    tests.test_cell_gallery tests.test_annotation_palettes tests.test_runner
python -m unittest discover -s tests -t .
```

- ✅ 41 new tests pass; the targeted regression set (100 tests) passes
- ✅ Full suite **778 tests, OK** (737 before, +41), green on two consecutive runs
- ✅ Verified against the real `anndata` 0.11.4 / `pandas` 2.3.3: `.h5ad` round-trip → flatten →
  edit → sync → `dataframe_to_anndata`, and that the legacy `viewer.*` import shim still resolves to
  identical module objects
- ⚠️ To confirm live in a notebook: markers appearing first in the Scatter/Histogram/Heatmap pickers,
  `cell_type` selectable as a Mask-painter identifier and a heatmap cluster column, and a FlowSOM
  cluster column showing up in `viewer.get_cell_table_adata().obs`
