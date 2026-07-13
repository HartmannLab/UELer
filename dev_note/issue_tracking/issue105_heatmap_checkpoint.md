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
python \
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
python \
  -m unittest tests.test_checkpoint_store tests.test_cell_annotation -v
```

- ✅ 29 tests, all pass.

---

## Follow-Up Fix 2 — Debug Logging (Reply 2)

**Reported:** After Reply 1 fix, the heatmap is still not displayed. Jupyter shows `"Preparing heatmap data...\nUsing cluster:..."` as `"Unhandled kernel message from a widget: stream"` — execution reaches data preparation but the heatmap never renders.

### Root Cause of "Unhandled kernel message"

`plot_heatmap()` called `prepare_heatmap_data()` and `generate_dendrogram()` BEFORE the `with self.plot_output:` block, so their `print()` output went to kernel stdout instead of the Output widget.

`generate_heatmap()` has 5 silent early-return points (no markers, empty view, dendrogram None, cluster index None, no valid markers) and no try/except around `sns.clustermap()` or `_setup_layout()` — failures were invisible.

### Changes

- **`heatmap_layers.py` — `plot_heatmap()`**: Moved all pre-render calls (`prepare_heatmap_data`, `generate_dendrogram`, `restore_vertical_canvas`) inside `with self.plot_output:` so ALL output is captured in the widget. Added `[DEBUG][heatmap]` prints at each step.
- **`heatmap_layers.py` — `generate_heatmap()`**: Added `[DEBUG][heatmap]` prints at entry, each early-return branch (with reason), before `sns.clustermap()` (with data shape), after successful clustermap, and on render completion. Wrapped `sns.clustermap()` and `_setup_layout()` in `try/except` with `traceback.print_exc()`.
- **`cell_annotation.py` — `save_checkpoint()` / `load_checkpoint()`**: Added `[DEBUG][cell_annotation]` prints at key operations (entry, export/read, FlowSOM params, write/import completion).

> **Note:** Follow-Up Fix 3 (below) later moved these traces from `print()` to the
> `logging` module and reverted the `plot_heatmap()` pre-render calls back outside
> the `with self.plot_output:` block, so log routing is controlled centrally.

### Tests

```bash
python \
  -m unittest tests.test_checkpoint_store tests.test_cell_annotation -v
```

- ✅ 29 tests, all pass.

---

## Follow-Up Fix 3 — Route logs to the kernel console (fd-level handler) — ABANDONED

> **Superseded by Follow-Up Fix 4.** This kernel-fd approach was reverted: the
> handler in `ueler/__init__.py` (with `propagate=False`) broke the previously
> working root-propagated loggers, and the kernel console proved to be the wrong
> home for Python log messages. Kept below for the diagnosis of *why* ordinary
> sinks don't escape ipykernel.

**Reported:** The Reply-2 debug messages (switched from `print` to
`logging.getLogger(__name__)`) kept appearing in the notebook **output cell** or
the **plugin GUI**, even after `logging.basicConfig(level=DEBUG, force=True)`.
The user wants them in the VSCode kernel output channel ("the console").

### Root Cause (three stacked capture layers — `sys.stderr` is not the bottom)

`logging.basicConfig(level=DEBUG, force=True)` still failed because ipykernel
6.x (`watchfd=True`) captures output at three levels:

1. It replaces `sys.stdout`/`sys.stderr` with iopub `OutStream` objects →
   frontend attributes them to a **cell**.
2. It `dup2`'s a pipe **over the OS fd 1/2**, so even `os.write(2, …)` and
   `sys.__stderr__` (whose `fileno()` is still 2) funnel into iopub → cell.
   *(This is why the earlier `os.dup(2)` handler and `sys.__stdout__` attempts
   failed.)*
3. The ipywidgets **`Output` widget** (`with self.plot_output:`) grabs the iopub
   stream inside its block → **plugin GUI**.

The only escape: the `OutStream` keeps a dup of the **pre-redirect fd** in
`_original_stdstream_copy`. Writing there reaches the kernel's real stderr —
VSCode's "Jupyter Kernel" output channel — seen by neither iopub nor the
`Output` widget.

### Fix

- **`ueler/__init__.py`** — `_KernelConsoleStream` + `_install_ueler_console_handler()`
  (run once at import): the `ueler` logger gets a `StreamHandler` whose stream
  writes via `os.write(fd, …)` to `sys.stderr._original_stdstream_copy`
  (resolved dynamically per write; falls back to fd 2 outside a kernel).
  `propagate=False`; default level `WARNING`. Idempotent via a
  `_ueler_console_installed` sentinel.
- **`ueler/viewer/main_viewer.py`** — `ImageMaskViewer.__init__` raises
  `logging.getLogger("ueler")` to `DEBUG` when `debug=True` (reachable through
  `run_viewer(..., debug=True)`, which forwards `**viewer_kwargs`).

Result: `run_viewer(folder, debug=True)` → `[heatmap]` / `[cell_annotation]`
traces appear in the VSCode kernel output channel, not the cell or the plugin
panel. `logging.basicConfig(...)` is no longer required.

### Tests

```bash
python \
  -m unittest tests.test_checkpoint_store tests.test_cell_annotation -v
```

- ✅ 29 tests, all pass. Also verified `_KernelConsoleStream` resolves a
  simulated `_original_stdstream_copy` fd and writes there even while
  `sys.stderr` is redirected (and falls back to fd 2 with no kernel present).

---

## Follow-Up Fix 4 — UELer log console widget (FINAL)

**Decision:** Stop trying to route into the kernel console (ipykernel captures
stdout/stderr *and* the OS fds; the console isn't meant for Python log
messages). Instead, give UELer its **own** log console — a `logging.Handler`
that renders records into an `ipywidgets.Output` docked at the bottom of the UI.

### Investigation — how UELer surfaces messages (5 channels)

1. **`logging`** (`getLogger(__name__)`, propagating to `ueler`) — the working
   mechanism, now feeding the console widget.
2. **raw `print()`** — went to the cell / plugin widget; the source of the
   lingering symptom. Converted to `logging` in the heatmap files.
3. **`Output` widgets** (`plot_output`) — render figures and trap `print()`.
4. **HTML status labels** (`_set_status`) — per-plugin user-facing text.
5. **global status bar** (`@update_status_bar`) — busy/ready icon.

### Changes

- **Reverted** the `ueler/__init__.py` handler from Fix 3 (restores normal
  logger propagation; that `propagate=False` + invisible handler had silenced
  the loggers).
- **New `ueler/viewer/log_console.py`**: `OutputWidgetHandler(logging.Handler)`
  renders records into a scrollable `Output` (cap 1000, newest on top);
  `enable_log_console()` attaches it to `logging.getLogger("ueler")` with
  `propagate=False`; `disable_log_console()` detaches and restores propagation;
  `build_log_console_panel()` returns a titled, bordered panel with a **Clear**
  button.
- **`ui_components.display_ui()`**: when `viewer._debug`, dock the console as the
  last root child and enable it; otherwise disable it. (`ImageMaskViewer(debug=
  True)` already raises the `ueler` logger to DEBUG.)
- **`heatmap_layers.py` / `heatmap.py`**: all ~50 `print()` converted to
  `_logger.<level>` (debug traces, info confirmations, warnings, errors).

### Tests

```bash
python \
  -m unittest tests.test_log_console tests.test_checkpoint_store tests.test_cell_annotation -v
```

- ✅ `tests/test_log_console.py` (8 tests) + existing suites pass; full suite
  unchanged at the pre-existing 21F/41E baseline.

### Usage

`run_viewer(folder, debug=True)` → a "UELer Log Console" appears at the bottom of
the UI; heatmap/cell-annotation messages stream in at their levels; it scrolls,
the text is copyable, and **Clear** empties it. Without `debug=True`, no console
is shown and `ueler` logs propagate normally.

---

## Follow-Up Fix 5 — Package-wide logging sweep

**Goal:** Make the *entire* package feed the log console, not just the heatmap.

**Inventory:** ~140 `print()` across the viewer core (`main_viewer.py` ~95,
`image_display.py`, `plugin_base.py`, `ui_components.py`, `decorators.py`,
`runner.py`, `data_loader.py`) and plugins (`run_flowsom`, `chart`,
`chart_heatmap`, `cell_gallery`, `go_to`, `mask_painter`).

**Changes:**
- Every diagnostic/debug `print()` → `logging.getLogger(__name__)` at the
  appropriate level (debug for traces — dropping `if self._debug:` guards that
  only gated a print; info for confirmations; warning for user-actionable
  problems; `error(exc_info=True)` for failures). Each module that lacked one
  gained `_logger = logging.getLogger(__name__)`.
- Per the user's choice, plugin **UI text is mirrored** to the console: the
  funnel helpers `mask_painter._log()`, `roi_manager_plugin.set_status()`,
  `cell_annotation._set_status()`, `export_fovs._log()`,
  `main_viewer._log_annotation_palette()` emit a `ueler` record *and* keep their
  inline widget update; widget-rendered prints (cell_gallery messages, chart
  cutoff) keep the `with Output:` render and add a log mirror.
- `main_viewer.py` was converted with a one-off script (kept `if self._debug:`
  guards intact; chose level by guard-context + message text), then reviewed —
  the one widget-rendered print (`_log_annotation_palette`) was restored + mirrored.

**Tests:** `tests/test_logging_sweep.py` (4 tests, `assertLogs("ueler")` on the
helpers). Full suite unchanged at the 21F/41E baseline (535 tests).
