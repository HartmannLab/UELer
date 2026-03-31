# Viewer Runtime and UI Behavior

## Context
These notes cover the main viewer runtime, downsampling behavior, channel controls, tooltips, and notebook-specific behavior. The key focus is how the viewer loads FOVs, updates its display, and synchronizes UI controls with cached image and overlay data.

## Key decisions
- Maintain a lazy load pipeline for FOVs and overlays, with LRU-based caches for image, mask, and annotation data.
- Use downsampling factors derived from viewport and FOV size to keep navigation responsive.
- Respect dataset-specific keys for tooltips instead of hard-coded column names.
- Offer per-channel visibility toggles and a channel legend without altering the underlying selection list.
- Provide a VS Code scatter fallback when widget rendering fails.

## Current status
- FOV load cycle is documented end-to-end, including cache and plugin update steps.
- Downsampling flow and `select_downsample_factor` behavior are captured for main viewer and ROI thumbnails.
- Tooltip lookup uses viewer-configured keys and caches resolved rows for performance.
- VS Code scatter fallback uses a static Matplotlib plot when widget front-ends fail.
- Per-channel visibility toggles and channel color legend features are implemented and tested.

## Open items
- Keep UI/UX changes consistent with map mode and plugin rendering updates.
- Validate any future changes to channel controls or tooltip semantics against existing tests.

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/61
- https://github.com/HartmannLab/UELer/issues/64
- https://github.com/HartmannLab/UELer/issues/66
- https://github.com/HartmannLab/UELer/issues/75

## Key source links
- [dev_note/FOV_load_cycle.md](dev_note/FOV_load_cycle.md)
- [dev_note/main_viewer.md](dev_note/main_viewer.md)
