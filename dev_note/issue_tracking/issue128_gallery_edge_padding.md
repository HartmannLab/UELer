# Issue #128 — Cell gallery does not maintain the original scale of individual images

**Status:** implemented
**Type:** Bug
**Scope:** cell gallery tiles (`ueler/viewer/plugin/cell_gallery.py`, `ueler/rendering/engine.py`)

---

## Problem

The cell gallery renders one square cutout per selected cell. For cells whose
`crop_width`-sized box would extend past the field of view, the rendered tile comes back
**smaller than requested** and the browser then scales it up to the grid column width, so
the tile is displayed at a different pixels-per-micron than every other tile. Comparing
cell sizes across the gallery is therefore misleading.

### Mechanism

1. `_render_tile_for_index()` calls `render_crop_to_array(..., size_px=context.crop_width)`
   ([cell_gallery.py:809](../../ueler/viewer/plugin/cell_gallery.py#L809)).
2. `render_crop_to_array()` builds the requested box around the centre and then **clamps** it
   with `_ensure_region_within_bounds()`
   ([engine.py:550](../../ueler/rendering/engine.py#L550)). A cell 20 px from the left edge of a
   150 px cutout yields a region of `95 × 150` instead of `150 × 150`.
3. The tile widget's CSS is `.tg-tile img { width: 100%; height: auto; }`
   ([tile_gallery_widget.py:138](../../ueler/viewer/plugin/tile_gallery_widget.py#L138)). A `95 × 150`
   array is stretched so its **95 columns** fill the full column width — a 1.58× horizontal
   zoom relative to a full tile — and `height: auto` then makes the tile taller than a square.
   That is the reported "width is extended … the displayed image longer".

Note the aspect ratio itself is preserved by the CSS; what is lost is the *scale*, because
each tile is independently fitted to the column width. `_compose_canvas()` (the legacy
Matplotlib composer) already centre-pads variable tile sizes, so the canvas path shows the
same cell at the wrong scale too, only without the stretch.

---

## Solution

Keep the crop region clamped (never read outside the FOV) but **pad the rendered array back
to the requested square** so every tile leaves the renderer at exactly the requested size.
This is what the issue proposes, and it is preferable to un-clamping the region because the
mask/annotation overlays are sliced with the same clamped region and would otherwise
mis-register.

- `render_crop_to_array()` gains **opt-in** `pad_to_size` / `pad_color` parameters. Opt-in
  rather than default because the batch-export path in `export_fovs.py` re-derives the
  clamped region with `compute_crop_regions()` and passes the array to
  `apply_overlay_snapshot_to_array()`; silently changing the array shape there would break
  overlay registration. Padding is enabled only by the cell gallery for now.
- Padding is placed at the **offset the missing data would have occupied**, computed on the
  downsampled grid, so the cell stays where it belongs inside the square instead of being
  re-centred.
- Pad colour is **white** (`(1.0, 1.0, 1.0)`), per the issue. On the black composite
  background this reads unambiguously as "outside the field of view" rather than as signal,
  and it is exposed as a parameter so it can be changed without touching the renderer.

### Implementation steps

1. `ueler/rendering/engine.py`
   - Add `_pad_region_to_requested_size(array, requested_xy, region_ds, downsample_factor, pad_color)`
     returning a canvas of the requested downsampled size with `array` blitted at the correct
     offset.
   - Add `pad_to_size: bool = False` and `pad_color: ColorTuple = (1.0, 1.0, 1.0)` to
     `render_crop_to_array()`; apply the helper to the result when `pad_to_size` is set.
2. `ueler/viewer/plugin/cell_gallery.py`
   - Add `GALLERY_PAD_COLOR = (1.0, 1.0, 1.0)`.
   - Pass `pad_to_size=True, pad_color=GALLERY_PAD_COLOR` from `_render_tile_for_index()`.
3. Tests (`tests/test_rendering.py`, `tests/test_cell_gallery.py`).

---

## Tests

- Engine: an edge crop with `pad_to_size=True` is exactly `size_px × size_px`; the padded
  band carries `pad_color`; the real pixels are unchanged and sit at the correct offset
  (left/top/right/bottom edges, and a corner where two sides are padded).
- Engine: `pad_to_size` is a no-op for an interior crop, and the default (`False`) keeps the
  old clamped shape so the export path is untouched.
- Engine: padding works with `downsample_factor > 1` (square of `ceil(size / factor)`).
- Gallery: `_render_tile_for_index()` produces a square tile for a cell at the FOV edge and
  the tile matches the size of an interior cell's tile.

---

## Notes / follow-ups

- The batch-export single-cell paths (`_preview_single_cell`, `_export_cell_worker`) still
  emit short crops for edge cells. Making those square needs the overlay re-registration in
  `apply_overlay_snapshot_to_array()` to be padding-aware; deliberately left out of #128.
- `_create_error_placeholder()` builds a `crop_width`-sized square regardless of the
  downsample factor, so an error tile can differ in pixel size from a rendered tile. It is
  still square, so gallery scale is unaffected.
