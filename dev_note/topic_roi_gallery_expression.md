# ROI Workflows and Gallery Behavior

## Context
ROI-related notes cover the ROI manager browser/editor UI, gallery sizing behavior, pagination, and expression-caret handling for the filter builder.

## Key decisions
- Keep ROI browser and editor tabs separate, with a scrollable gallery and pagination controls.
- Render the gallery as an `anywidget` CSS grid of pre-encoded PNG tiles (issue #107), not a Matplotlib figure, to avoid the interactive ipympl backend and its cross-front-end fragility. CSS handles responsive sizing.
- Host the whole advanced-expression editor (field, Apply, operator and tag buttons) in one `anywidget` so button clicks cannot move focus out of the input — this replaced the caret-bridge pipeline, which could not win the comm ordering race in VS Code (issue #88, follow-up 5 Option B).
- Treat a space between name characters as part of the tag name in the filter grammar (issue #130): tags may contain spaces, and two names can never be adjacent without an operator, so whitespace only has to separate a name from an operator, a parenthesis or a quote.
- Store drawn shapes as a **row kind in the same ROI table** rather than as a separate plugin with its own file. The predecessor (`viewer/plugin/region_annotation.py`) owned a private CSV, which is why its loss in the package restructure was invisible and why it never gained tags, thumbnails, filtering or export. Shapes therefore reuse every consumer of an ROI record instead of duplicating them.
- Give every shape a **padded bounding box** in the existing `x_min`…`y_max` columns. That single decision is what lets `center_on_roi`, the thumbnail renderers and the batch-export job builder handle a shape without knowing it is one; the padding exists because a straight horizontal or vertical line has a zero-extent box that the map thumbnail renderer rejects.
- Draw shapes on the canvas as **Matplotlib artists**, never by painting into the RGB array the way `update_patches` draws mask highlights — baked pixels are erased by the next `set_data`, artists are not.

## Current status
- ROI gallery uses the shared `TileGalleryWidget` (`ueler/viewer/plugin/tile_gallery_widget.py`): a responsive CSS grid of `<img>` tiles inside a fixed-height scroll container. Clicks route through a synced `clicked` traitlet to `_activate_roi_from_browser`; hover labels are in-tile CSS tooltips.
- Pagination and gallery refresh throttling (signature-based) are implemented to limit unnecessary redraws.
- Expression editing lives in `ROIExpressionEditorWidget` (`ueler/viewer/plugin/roi_expression_editor.py`); its JS reads the live caret and applies the same spacing rules as `_format_expression_insertion`. Python observes `apply_requested` and reads `expression`; a `traitlets.HasTraits` fallback keeps it headless for tests.
- `ueler/viewer/tag_expression.py` parses the expression: unquoted names keep their internal spaces (collapsed to single spaces and stripped at the ends, on both the name and the ROI's tags); quotes are for names containing operator characters.
- ROI metadata now captures palette and mask-visibility settings for preset playback.
- ROI records carry `roi_kind` and `geometry` (`ueler/viewer/roi_manager.py`). `roi_kind` is `""`/`"view"` for a viewport bookmark and `"line"` for a drawn shape; `geometry` is `{"type", "closed", "points"}` as JSON, in FOV-local pixels or stitched-canvas pixels depending on whether the record names a `fov` or a `map_id`. Open polylines and closed polygons share the one kind and differ only by `closed`, so `is_shape_record()` is the single predicate every filter uses.
- Shapes are drawn and edited through `ImageDisplay.enable_polyline_editor` / `draw_shape_rois`, and the UI is a block inside the *ROI editor* tab (`_build_shape_widgets`), not a separate plugin or tab.
- Batch export lists shape ROIs by default and renders them as their bounding box; `Include line/polygon ROIs` removes them from the list.

## Open items
- Manually validate click/hover across JupyterLab, VSCode, and Voila now that the gallery no longer uses ipympl.
- Confirm in a notebook that a spaced tag inserted from a helper button filters correctly end to end (#130 was verified by unit tests only).
- Confirm in a notebook that the shape editor's pointer gestures place and drag vertices in a live ipympl canvas, and that the shape overlay repaints after a zoom — the tests drive the event handlers directly, not the frontend.
- Per-cell distance to a shape is deliberately not implemented; it is the natural follow-up to line ROIs.

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/44

## Key source links
- [dev_note/gallery_width.md](dev_note/gallery_width.md)
- [dev_note/issue_tracking/line_roi_support.md](dev_note/issue_tracking/line_roi_support.md)
