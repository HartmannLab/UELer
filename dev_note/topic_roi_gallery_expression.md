# ROI Workflows and Gallery Behavior

## Context
ROI-related notes cover the ROI manager browser/editor UI, gallery sizing behavior, pagination, and expression-caret handling for the filter builder.

## Key decisions
- Keep ROI browser and editor tabs separate, with a scrollable gallery and pagination controls.
- Favor fixed, conservative Matplotlib sizing to avoid width clipping across notebook layouts.
- Use a caret-aware insertion pipeline for expression helpers, with browser-side integration to reduce focus drift.

## Current status
- ROI gallery uses a static narrow figure width to prevent clipping and keeps a fixed-height scroll container.
- Pagination and gallery refresh throttling are implemented to limit unnecessary redraws.
- Expression helper insertion uses cached caret indices and browser-side hooks to preserve cursor placement.
- ROI metadata now captures palette and mask-visibility settings for preset playback.

## Open items
- Continue validating ROI gallery sizing across notebook front-ends.
- Keep caret bridge behavior aligned with ipywidgets versions and notebook changes.

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/44

## Key source links
- [dev_note/gallery_width.md](dev_note/gallery_width.md)
