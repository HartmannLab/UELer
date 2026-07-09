# ROI Workflows and Gallery Behavior

## Context
ROI-related notes cover the ROI manager browser/editor UI, gallery sizing behavior, pagination, and expression-caret handling for the filter builder.

## Key decisions
- Keep ROI browser and editor tabs separate, with a scrollable gallery and pagination controls.
- Render the gallery as an `anywidget` CSS grid of pre-encoded PNG tiles (issue #107), not a Matplotlib figure, to avoid the interactive ipympl backend and its cross-front-end fragility. CSS handles responsive sizing.
- Use a caret-aware insertion pipeline for expression helpers, with browser-side integration to reduce focus drift.

## Current status
- ROI gallery uses the shared `TileGalleryWidget` (`ueler/viewer/plugin/tile_gallery_widget.py`): a responsive CSS grid of `<img>` tiles inside a fixed-height scroll container. Clicks route through a synced `clicked` traitlet to `_activate_roi_from_browser`; hover labels are in-tile CSS tooltips.
- Pagination and gallery refresh throttling (signature-based) are implemented to limit unnecessary redraws.
- Expression helper insertion uses cached caret indices and browser-side hooks to preserve cursor placement.
- ROI metadata now captures palette and mask-visibility settings for preset playback.

## Open items
- Manually validate click/hover across JupyterLab, VSCode, and Voila now that the gallery no longer uses ipympl.
- Keep caret bridge behavior aligned with ipywidgets versions and notebook changes.

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/44

## Key source links
- [dev_note/gallery_width.md](dev_note/gallery_width.md)
