# Export Pipeline and Scale Bar

## Context
The export pipeline covers batch exports for FOVs, cells, and ROIs, plus shared scale bar behavior in the viewer and export paths.

## Key decisions
- Build a job runner abstraction for exports with structured per-item results.
- Keep export rendering logic reusable and independent of UI state.
- Provide overlay snapshots (masks, annotations) and configurable outline thickness for exports.
- Use shared scale bar helpers that respect downsample factors and image size constraints.

## Current status
- Export job runner, plugin UI, overlay snapshots, and mask outline rendering are implemented.
- Scale bar helpers compute a rounded physical length capped to 10% of image width and are shared across viewer and export paths.
- Export preview and cell-export fixes are implemented with targeted tests.

## Open items
- Add parallelism and reliability improvements (Phase 5).
- Add broader integration tests and example notebooks (Phase 6).

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/1

## Key source links
No remaining standalone notes; export details are consolidated in this topic summary.
