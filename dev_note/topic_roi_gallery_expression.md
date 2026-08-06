# ROI Workflows and Gallery Behavior

## Context
ROI-related notes cover the ROI manager browser/editor UI, gallery sizing behavior, pagination, and expression-caret handling for the filter builder.

## Key decisions
- Keep ROI browser and editor tabs separate, with a scrollable gallery and pagination controls.
- Render the gallery as an `anywidget` CSS grid of pre-encoded PNG tiles (issue #107), not a Matplotlib figure, to avoid the interactive ipympl backend and its cross-front-end fragility. CSS handles responsive sizing.
- Host the whole advanced-expression editor (field, Apply, operator and tag buttons) in one `anywidget` so button clicks cannot move focus out of the input — this replaced the caret-bridge pipeline, which could not win the comm ordering race in VS Code (issue #88, follow-up 5 Option B).
- Treat a space between name characters as part of the tag name in the filter grammar (issue #130): tags may contain spaces, and two names can never be adjacent without an operator, so whitespace only has to separate a name from an operator, a parenthesis or a quote.

## Current status
- ROI gallery uses the shared `TileGalleryWidget` (`ueler/viewer/plugin/tile_gallery_widget.py`): a responsive CSS grid of `<img>` tiles inside a fixed-height scroll container. Clicks route through a synced `clicked` traitlet to `_activate_roi_from_browser`; hover labels are in-tile CSS tooltips.
- Pagination and gallery refresh throttling (signature-based) are implemented to limit unnecessary redraws.
- Expression editing lives in `ROIExpressionEditorWidget` (`ueler/viewer/plugin/roi_expression_editor.py`); its JS reads the live caret and applies the same spacing rules as `_format_expression_insertion`. Python observes `apply_requested` and reads `expression`; a `traitlets.HasTraits` fallback keeps it headless for tests.
- `ueler/viewer/tag_expression.py` parses the expression: unquoted names keep their internal spaces (collapsed to single spaces and stripped at the ends, on both the name and the ROI's tags); quotes are for names containing operator characters.
- ROI metadata now captures palette and mask-visibility settings for preset playback.

## Open items
- Manually validate click/hover across JupyterLab, VSCode, and Voila now that the gallery no longer uses ipympl.
- Confirm in a notebook that a spaced tag inserted from a helper button filters correctly end to end (#130 was verified by unit tests only).

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/44

## Key source links
- [dev_note/gallery_width.md](dev_note/gallery_width.md)
