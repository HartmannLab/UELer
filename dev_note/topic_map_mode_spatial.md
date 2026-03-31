# Map Mode and Spatial Navigation

## Context
Map mode stitches multiple FOVs into a single composite view and introduces spatial navigation challenges for plugins that assume per-FOV coordinates. The notes cover map descriptor ingestion, viewport stitching, cache behavior, and map-aware selection/hover behavior.

## Key decisions
- Use a `VirtualMapLayer` to render stitched tiles, reusing existing single-FOV rendering paths.
- Keep map mode gated behind `ENABLE_MAP_MODE` and preserve single-FOV behavior when disabled.
- Resolve FOV-local coordinates into stitched-map pixels for selection and navigation without mutating underlying data tables.
- Keep a dedicated stitched-tile cache keyed by channel and overlay state to avoid stale renders.

## Current status
- Descriptor parsing, map-mode activation, and stitched rendering are implemented with regression coverage.
- Map mode handles large canvas activation without allocating full-size buffers.
- Coordinate translation and reverse lookup are wired through `ImageMaskViewer` and `ImageDisplay` so tooltips, highlights, and plugin navigation remain map-aware.
- Navigation stack reset and viewport offset fixes address black-canvas and reset regressions.

## Open items
- Keep map-mode UI and export behavior aligned with the stitched render pipeline.
- Validate any new plugin interactions against stitched coordinate helpers.

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/3
- https://github.com/HartmannLab/UELer/issues/58
- https://github.com/HartmannLab/UELer/issues/59
- https://github.com/HartmannLab/UELer/issues/62

## Key source links
No remaining standalone notes; map mode details are consolidated in this topic summary.
