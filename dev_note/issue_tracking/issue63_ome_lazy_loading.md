# Issue #63 — OME-TIFF lazy loading plan

## Context
- Users report stacked OME-TIFFs load very slowly (appearing to hang) and metadata omits expected frames.
- Current `OMEFovWrapper` slices only the first T/Z frame and caches by `(channel, ds_factor)`, so later frames are unreachable and cache mixes planes.
- `_select_level` sometimes picks finer levels even when a coarser pyramid level would satisfy the requested downsample, leaving more data to read.
- Unit test `tests.test_ome_tiff_loading::test_residual_stride_applies_to_selected_level` is failing, confirming the level picker regression.

## Goals
- Keep loading strictly lazy via Dask (no eager reads of full stacks).
- Respect stacked frames: expose frame count/axes metadata and allow selecting frames without mixing cache entries.
- Fix pyramid level selection so it prefers the coarsest level that still covers the requested downsample, restoring the test expectation.
- Avoid regressions to folder-per-FOV loading.

## Plan
1) **Level selection fix**
   - Rework `_select_level` to iterate levels from coarsest to finest, computing residual stride as `ceil(ds / scale)` and picking the first level whose downsampled shape covers the expected bounds.
   - Keep fallback to base level.

2) **Frame-aware lazy slices**
   - Detect frame axis (`T` or `Z`, fallback `S`) during `_init_levels`; record `frame_axis`, `frame_count`, and `current_frame_index` (default 0).
   - Include `frame_index` in cache keys and slice the chosen frame instead of hard-coding frame 0.
   - Add `set_frame_index()` to change planes lazily and clear caches when the frame changes.
   - Add lightweight metadata helpers (axes, frame count, channel names) for the viewer to surface.

3) **Viewer plumbing**
   - Allow `load_fov(..., frame_index=None)` and store per-FOV frame metadata so we can report available frames; default to frame 0 for rendering.
   - Ensure height/width and channel max sampling use the active frame and stay lazy.

4) **Tests**
   - Fix existing failing pyramid test; add coverage for frame-aware slicing (cache key includes frame, shapes honor frame changes).

5) **Docs**
   - Update `dev_note/github_issues.md` entry to link this plan; document the change in `doc/log.md` once implemented.

## 2025-12-23 Follow-up — incompatible keyframe fallback

- Added a tolerant `OMEFovWrapper` open path that retries series discovery without OME parsing when `tifffile` raises `RuntimeError: incompatible keyframe`, enabling stacked TIFFs with mismatched keyframes to load instead of crashing.
- Added regression coverage in `tests/test_ome_tiff_loading.py::test_incompatible_keyframe_retries_without_ome_series` and recorded the update in `dev_note/github_issues.md` and `doc/log.md`.

## 2025-12-23 Follow-up — suffix-less OME detection

- Added metadata-based detection (`find_ome_tiff_files`) so OME-TIFFs without the `.ome.tif(f)` suffix are discovered and listed as FOVs.
- Broadened `ImageMaskViewer.load_fov` file lookup to include `.tif/.tiff` names when instantiating `OMEFovWrapper`.
- Added regression coverage in `tests/test_ome_tiff_loading.py::test_find_ome_tiff_files_detects_suffixless`.
