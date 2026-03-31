# OME-TIFF and Data Loading

## Context
OME-TIFF support adds a parallel data-loading path while preserving the existing folder-per-FOV layout. The notes capture detection, lazy loading, pyramid level selection, and frame-aware access.

## Key decisions
- Automatically detect OME-TIFF datasets and keep the legacy folder layout unchanged.
- Use `OMEFovWrapper` to keep channel access lazy and downsample-aware.
- Prefer coarse pyramid levels that meet the requested downsample to avoid over-fetching.
- Include frame-aware access for stacked OME files and cache slices by frame index.

## Current status
- OME-TIFF detection is supported for `.ome.tif(f)` and suffix-less OME TIFFs.
- Lazy loading is frame-aware and respects downsample factors across pyramid levels.
- Keyframe compatibility and memory-usage regressions are addressed with fallbacks and lazy max computations.
- OME-specific rendering and viewport alignment fixes are covered by regression tests.

## Open items
- Continue monitoring large-stack performance and metadata edge cases.
- Document any UI exposure for frame selection once it is surfaced.

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/60
- https://github.com/HartmannLab/UELer/issues/63

## Key source links
- [dev_note/ome_tiff_loading.md](dev_note/ome_tiff_loading.md)
