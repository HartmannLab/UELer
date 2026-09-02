# Issue #137 — configurable `.UELer` settings folder location

> GitHub issue: [#137](https://github.com/HartmannLab/UELer/issues/137)
> Status: implemented
> Type: Feature

## Problem

`.UELer` — UELer's hidden settings/cache folder (widget state, ROI table, annotation palettes, heatmap checkpoints, export-config templates, map descriptors) — was always created inside `base_folder`, the directory holding the multiplexed images. That breaks when `base_folder` is read-only, or when a downstream pipeline treats any extra file under its dataset root as an error.

## Investigation

Every production call site that builds a `.UELer` path turned out to derive it from the same value — `ImageMaskViewer.base_folder` — with no shared helper: nine independent `os.path.join(base_folder, ".UELer")` / `Path(base_folder) / ".UELer"` expressions across `main_viewer.py`, `roi_manager.py`, `checkpoint_store.py`, `plugin_base.py`, `mask_painter.py` and `export_fovs.py`. `run_viewer_bia` (BIA streaming mode) already had an equivalent relocation mechanism — `local_dir` moves its whole workspace (`.UELer` + `cache/`) — so this feature only needed to reach `run_viewer` (local-folder mode).

## Solution

A new `settings_path` argument on `run_viewer` / `ImageMaskViewer`. When given, `.UELer` moves from `<base_folder>/.UELer` to `<settings_path>/<base_folder name>/.UELer`; when omitted, behavior is unchanged.

The computation is centralized in a new `ueler/viewer/settings_paths.py` module:

- `resolve_settings_root(base_folder, settings_path=None)` — the directory that should contain `.UELer`.
- `resolve_settings_folder(base_folder, settings_path=None)` — `resolve_settings_root(...) / ".UELer"`.
- `viewer_settings_folder(viewer)` — reads `viewer.settings_folder` if present, else derives it from `viewer.base_folder`; the fallback keeps lightweight test doubles (e.g. `_ViewerStub` in `tests/test_export_fovs_mask_customization.py`) that only set `base_folder` working unchanged.

`ImageMaskViewer.__init__` computes `self.settings_path`, `self.settings_root` and `self.settings_folder` once; every downstream site (ROI manager, widget-state save/load, annotation-palette folder, map descriptors, mask-painter palette storage, export-config templates) now reads `self.settings_folder` (directly or via `viewer_settings_folder`) instead of re-deriving it from `base_folder`.

`ROIManager` gained an optional `settings_dir` parameter (defaults to the old `base_folder/.UELer` behavior). `CheckpointStore` gained an optional `storage_root` parameter — the checkpoint files move to `storage_root` when given, but `dataset_id` keeps hashing `dataset_root` itself, so a dataset's checkpoint identity doesn't change when its storage location does.

`run_viewer_bia` was intentionally left unchanged: it already relocates the whole `.UELer` + `cache/` workspace via `local_dir`, so exposing `settings_path` there too would create two competing ways to do the same thing.

## Verification

New `tests/test_settings_path.py` (12 tests) covers the path-resolution helpers, `ROIManager(settings_dir=...)`, `CheckpointStore(storage_root=...)` (including dataset-identity stability), and a full `ImageMaskViewer(base_folder, settings_path=...)` construction confirming the settings folder — and nothing else — lands under the redirected location. `tests/test_runner.py` was extended to assert `run_viewer` forwards `settings_path` to the viewer factory. Full suite: `python tools/run_test_suite.py --max-skips 0` → 1121 tests, 0 skips.
