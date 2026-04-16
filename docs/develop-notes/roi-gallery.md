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

---

## ROI Manager

ROIs are stored persistently in `<base_folder>/.UELer/roi_manager.csv`. Each ROI record contains:

| Column | Description |
|---|---|
| `fov` | FOV name (empty for map-mode ROIs) |
| `map_id` | Map layer ID (non-empty for map-mode ROIs) |
| `x_min`, `x_max`, `y_min`, `y_max` | Bounding box in canvas pixels |
| `name` | User-assigned label |
| `tags` | Comma-separated tag list |
| `palette` | Serialized color palette snapshot |

### Labels

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
