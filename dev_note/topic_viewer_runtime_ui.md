# Viewer Runtime and UI Behavior

## Context
These notes cover the main viewer runtime, downsampling behavior, channel controls, tooltips, and notebook-specific behavior. The key focus is how the viewer loads FOVs, updates its display, and synchronizes UI controls with cached image and overlay data.

## Key decisions
- Maintain a lazy load pipeline for FOVs and overlays, with LRU-based caches for image, mask, and annotation data.
- Use downsampling factors derived from viewport and FOV size to keep navigation responsive.
- Respect dataset-specific keys for tooltips instead of hard-coded column names.
- Offer per-channel visibility toggles, a channel legend, and a channel grid display mode without altering the underlying selection list.
- Provide a VS Code scatter fallback when widget rendering fails.

## Current status
- FOV load cycle is documented end-to-end, including cache and plugin update steps.
- Downsampling flow and `select_downsample_factor` behavior are captured for main viewer and ROI thumbnails.
- Tooltip lookup uses viewer-configured keys and caches resolved rows for performance.
- VS Code scatter fallback uses a static Matplotlib plot when widget front-ends fail.
- Per-channel visibility toggles and channel color legend features are implemented and tested.
- Channel grid display mode (Issue #76) is implemented: a "Channel grid view" checkbox renders each visible channel as a separate labelled pane in a synchronised matplotlib subplot grid.
- UI layout hardening for Issue #85 is in place: overflow-prone wrappers now use shrink-safe constraints (`max_width: 99%`, `min_width: 0`, `box_sizing: border-box`) in core viewer panels and selected plugin panels to reduce unnecessary horizontal scrollbars, with follow-up tuning for marker-set and channel slider controls.
- Reply 2 to Issue #85 adds a content-widget policy (`width/max_width: calc(100% - 5px)`) so controls shrink inside the unchanged 99%-width containers.
- Reply 3 to Issue #85 tightens remaining channel-panel row composition (marker-set buttons, confirm-deletion row, legend wrapping, compact checkbox/dropdown spacing, longer usable slider tracks) without changing the 99% container policy.
- Reply 4 to Issue #85 reorganizes each channel into a three-row block (header with checkbox+marker name+color, then Min and Max rows), removing channel-name repetition in slider labels.
- Reply 5 to Issue #85 restores shared scrolling behavior for channel controls (no per-channel internal scrollers) and offsets the color dropdown 5px left in the header row.
- Reply 6 to Issue #85 enforces parent vertical-scroll triggering for long channel lists by preventing grouped channel rows from shrinking to fit the container height.
- Reply 7 to Issue #85 removes marker-set channel duplication at load time by de-duplicating marker channels (save/update/apply) and eliminating redundant channel-control rebuilds in the marker-set apply flow.

## Open items
- Keep UI/UX changes consistent with map mode and plugin rendering updates.
- Validate any future changes to channel controls or tooltip semantics against existing tests.

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/61
- https://github.com/HartmannLab/UELer/issues/64
- https://github.com/HartmannLab/UELer/issues/66
- https://github.com/HartmannLab/UELer/issues/75
- https://github.com/HartmannLab/UELer/issues/76
- https://github.com/HartmannLab/UELer/issues/85

## Key source links
- [dev_note/FOV_load_cycle.md](dev_note/FOV_load_cycle.md)
- [dev_note/main_viewer.md](dev_note/main_viewer.md)
