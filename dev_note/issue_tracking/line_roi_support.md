# Line and polygon ROIs in the ROI manager

## Problem

UELer once shipped a `Region annotation` plugin (`viewer/plugin/region_annotation.py`, class `RunFlowsom` — a copy-paste leftover name) that let the user click a chain of points onto the main canvas, drag them around, lasso-select them, nudge them with the arrow keys, and auto-save them to `<folder>/<fov>.csv`. It was added in `7df488f` (2025-10-15) with the initial transfer from the `UELer_alpha` repo and disappeared in `9985621` (2026-04-09), the `viewer/` → `ueler/viewer/` package migration: every other plugin was copied into `ueler/viewer/plugin/`, this one was not, and the old tree was then deleted. Because plugins are auto-discovered by `ImageMaskViewer.dynamically_load_plugins()` scanning the plugin directory, its absence produced no import error — the tab simply stopped existing.

Nothing replaced it. The current ROI manager stores viewport bookmarks and cell-selection metadata; there is no vertex, path, or shape geometry anywhere in the codebase.

The request is to bring the capability back, but **inside the ROI manager as a new ROI kind** rather than as a standalone plugin.

## Why not a standalone plugin again

The old plugin owned a private CSV that nothing else read. That is exactly why losing it was invisible and why it never gained tags, thumbnails, filtering, or export. Storing shapes as rows in the existing ROI table means the tag editor, the browser gallery, the tag-expression filter, CSV import/export, and batch export all apply to them with no new machinery.

## Design

### Storage: two new columns on the existing ROI table

`ROI_COLUMNS` gains:

- **`roi_kind`** — `""` or `"view"` for the existing viewport bookmarks, `"line"` for a drawn shape. Empty means view, so every ROI CSV written before this change loads unchanged.
- **`geometry`** — a JSON payload: `{"type": "polyline"|"polygon", "closed": false|true, "points": [[x, y], ...]}`.

Both are registered as string columns in `_ensure_dataframe`, which otherwise back-fills missing columns with `0.0`.

JSON in a single cell is the established convention in this table — `mask_visibility` and `mask_painter_state` are already stored that way — so the CSV stays one file and one round trip. `pandas` quotes the embedded commas correctly on write and read.

Open polylines and closed polygons share the single `roi_kind` value `"line"`, and are distinguished by `geometry.closed`. Filtering (in the browser and in batch export) therefore needs only one predicate, `is_shape_record()`, while the label and the UI still say "line" or "polygon" via `shape_display_kind()`.

### The bounding box is what keeps everything else working

A shape ROI still writes `x_min`/`x_max`/`y_min`/`y_max` (plus `x`/`y`/`width`/`height`) derived from its vertex bounds. `center_on_roi`, `render_roi_to_array`, `_render_roi_tile`, `_render_map_roi_tile` and the batch-export job builder then treat a shape exactly like any other ROI — no changes required in any of them. Exporting a line ROI exports its bounding-box region, which is the requested behaviour and comes for free.

A perfectly horizontal or vertical line has a zero-extent bounding box, and `_render_map_roi_tile` bails out on `x_max <= x_min`. `geometry_bounds()` therefore pads every box to `SHAPE_BBOX_MIN_EXTENT` (8 px) around its centre, plus `SHAPE_BBOX_PADDING` (4 px) of breathing room, so the shape is never flush against the tile edge.

Coordinates follow the convention the bbox columns already use: FOV-local pixels when `fov` is set, stitched-canvas pixels when `map_id` is set.

### Drawing: Matplotlib artists, not baked pixels

`ImageDisplay` already carries a one-shot selector pattern — `enable_roi_selector` / `disable_roi_selector` (a `RectangleSelector`, currently dead code with no callers) and `enable_lasso_selector` / `disable_lasso_selector`. The polyline editor follows it: `enable_polyline_editor(on_change, ...)` / `disable_polyline_editor()`, including the toolbar-mode release the lasso needed, because an active zoom or pan tool holds `canvas.widgetlock` and silently swallows every event.

Vertices and the connecting path are drawn as real `Line2D` artists. This matters: `update_patches` paints mask highlights *into* the RGB array and any later `set_data` erases them (see `doc/log.md` on #119). Artists are a separate layer, no code path anywhere clears the axes, and a shape drawn as an artist therefore survives zoom, pan, FOV switches, and every mask repaint.

`on_mouse_click` gains a `_polyline_active` guard next to the existing `_lasso_active` one, so drawing a shape does not also select cells underneath.

Interactions: left-click appends a vertex, left-drag on an existing vertex moves it, right-click deletes the nearest vertex, `ctrl+z` / `ctrl+y` undo and redo within the drawing session, `enter` finishes, `escape` cancels.

### UI

A **Line / polygon ROI** block inside the existing *ROI editor* tab — not a new tab and not a new plugin. It carries `Draw` / `Finish` / `Cancel`, a `Closed shape (polygon)` checkbox, `Save shape`, a `Show shapes on canvas` checkbox, and a live readout of vertex count, length (or perimeter) in pixels and µm via `effective_pixel_size_nm`. `Capture view` and the rest of the editor are untouched.

Saving is explicit, unlike the old plugin's save-on-every-edit, which matches how every other ROI in this manager is created.

### Browser and export

- The browser gallery draws the shape onto its rendered thumbnail after rendering, so a shape ROI is visually distinct from a viewport bookmark.
- Selecting a shape ROI in the browser centres on its padded bbox and re-draws it on the canvas.
- `export_fovs` gains an `Include line/polygon ROIs` checkbox (default on) beside the existing `Current FOV only` filter. When it is off, shape ROIs are dropped from the export list. When on, they export as their bounding-box region.

## Out of scope for this round

- **Per-cell distance to the shape** (the likely original scientific purpose) — deliberately deferred, confirmed with the developer.
- **Spline fitting** — the old `fit_spline()` was an empty `pass`, so nothing is lost.

## Follow-up: the canvas has three shape sources, not two

The first implementation drew the overlay from the ROI table alone, which left a shape invisible between `Finish` (which removes the editor's artists) and `Save shape` (which puts it in the table). The canvas therefore has to reconcile three things, not two:

- **saved records** in the ROI table, scoped to the active FOV or map;
- **the working copy** in `_shape_points`, drawn only when no drawing session is active — during a session the editor's own artists show it, and drawing both would double it;
- **the record being edited**, which is skipped from the saved set while a working copy exists, so an edited shape is not drawn once live and once stale.

After a save the working copy is released so the table is the single source, and `Save shape` finishes an active drawing session itself — requiring `Finish` first was the trap that produced the report.

## Known limitation

`_ensure_dataframe` ends with `df[ROI_COLUMNS]`, so an *older* UELer reading a CSV written by this version silently drops `roi_kind` and `geometry`. Forward-compatible, not backward.

## Implementation steps

1. `ueler/viewer/roi_manager.py` — new columns, string-column registration, geometry parse/serialise/bounds/length helpers, `is_shape_record`, `shape_display_kind`, kind marker in `format_roi_label`.
2. `ueler/viewer/image_display.py` — `enable_polyline_editor` / `disable_polyline_editor` with vertex add/move/delete/undo, `draw_shape_rois` / `clear_shape_rois` for persistent display artists, and the `_polyline_active` guard in `on_mouse_click`.
3. `ueler/viewer/plugin/roi_manager_plugin.py` — the shape UI block and its handlers, shape redraw on FOV change and map-mode transitions, shape-aware `_activate_roi_from_browser` and `_populate_fields`.
4. `ueler/viewer/plugin/roi_manager_plugin.py` — polyline drawn onto browser thumbnails.
5. `ueler/viewer/plugin/export_fovs.py` — the include/exclude checkbox.
6. `tests/test_line_roi.py`.
7. Documentation: `doc/log.md`, `README.md`, `dev_note/topic_roi_gallery_expression.md`.
