# ROI Workflows & Gallery Behavior

> Source: [`dev_note/topic_roi_gallery_expression.md`](https://github.com/HartmannLab/UELer/blob/main/dev_note/topic_roi_gallery_expression.md)

---

## Context

ROI-related notes cover the ROI Manager browser/editor UI, gallery sizing behavior, pagination, and expression-caret handling for the filter builder.

---

## Key Decisions

- **Browser and editor tabs** are separate within the ROI Manager plugin, with a scrollable gallery and pagination controls.
- **Static narrow figure width** avoids width clipping across different notebook layouts.
- **Caret-aware insertion pipeline** for expression helpers reduces focus drift when building cell filters.
- **Drawn shapes reuse the ROI record rather than getting their own store.** A polyline or polygon is an ROI with a `geometry`, so it inherits tagging, comments, filtering, thumbnails and export unchanged — and an older CSV stays readable because an absent `roi_kind` already means "view".

---

## ROI Manager

ROIs are stored persistently in `<base_folder>/.UELer/roi_manager.csv`. The schema is the
`ROI_COLUMNS` list in `ueler/viewer/roi_manager.py`, and the CSV is always written with exactly those
columns in that order — `_ensure_dataframe()` adds any that are missing (`""` for the string columns,
`0.0` otherwise) and reorders, so an older CSV is read without a migration step.

| Column | Description |
|---|---|
| `roi_id` | Stable identifier |
| `name` | User-assigned label |
| `fov` | FOV name (empty for map-mode ROIs) |
| `map_id` | Map layer ID (non-empty for map-mode ROIs) |
| `x`, `y`, `width`, `height`, `zoom` | The saved viewport |
| `x_min`, `x_max`, `y_min`, `y_max` | Bounding box in canvas pixels |
| `marker_set` | Marker set to re-apply on recall |
| `tags` | Comma-separated tag list |
| `annotation_palette`, `mask_color_set`, `mask_visibility`, `mask_painter_state` | Overlay presets captured with the ROI |
| `comment` | Free-text note |
| `roi_kind`, `geometry` | Shape support — see below |
| `created_at`, `updated_at` | Timestamps |

### View ROIs and shape ROIs

`roi_kind` discriminates the two record types:

- `ROI_KIND_VIEW` (`"view"`) — the viewport bookmark the manager has always stored. **An empty
  `roi_kind` means the same thing**, which is what lets pre-shape CSVs keep their meaning without
  migration.
- `ROI_KIND_SHAPE` (`"line"`) — a drawn shape, with its vertices in `geometry`. Open polylines and
  closed polygons share the one kind and are told apart by `geometry["closed"]`: a single kind keeps
  filtering to one predicate while the UI still names them separately. `is_shape_record()` also treats
  a record with parseable geometry but no `roi_kind` as a shape.

A straight horizontal or vertical line has a zero-extent bounding box, and the thumbnail renderers
reject `x_max <= x_min` — so every shape's box is grown to at least `SHAPE_BBOX_MIN_EXTENT` (8 px) per
axis, plus `SHAPE_BBOX_PADDING` (4 px) of breathing room so the shape is not flush against the tile
edge.

### Labels

`format_roi_label()` builds the display label:

- Single-FOV ROIs: `<fov> — <name>`
- Map-mode ROIs: `[MAP:<map_id>] — <name>`

---

## Gallery Behavior

- The gallery renders a paginated grid of ROI thumbnails using Matplotlib.
- After a new capture, the gallery navigates to page 1 so the new ROI is immediately visible.
- **Pagination throttling** limits unnecessary redraws during rapid page changes.

---

## Map-Mode Thumbnails

Thumbnails for map-mode ROIs are rendered via `_render_map_roi_tile()`, which calls `VirtualMapLayer.set_viewport()` + `VirtualMapLayer.render()` and restores the layer viewport in a `try/finally` block to avoid disturbing the live display.

---

## Expression Helper

- Caret position is cached on focus events.
- Insertion uses browser-side hooks to preserve cursor placement when typing filter expressions.

---

## Related Issues

- [#44](https://github.com/HartmannLab/UELer/issues/44)
