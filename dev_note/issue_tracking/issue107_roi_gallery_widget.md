# Issue #107 — Replace the Matplotlib galleries with an anywidget tile grid

## Problem / motivation
Both image galleries — the **ROI Manager browser** (`_refresh_browser_gallery` in
`ueler/viewer/plugin/roi_manager_plugin.py`) and the **Cell Gallery** (`_draw_gallery`
in `ueler/viewer/plugin/cell_gallery.py`) — rendered thumbnails into a Matplotlib
`plt.subplots` figure and wired click/hover through `fig.canvas.mpl_connect`. That path
depends on the interactive **ipympl** backend, which is a recurring source of fragility
in this repo (ipympl workarounds in `main_viewer.py`; `chart.py` forces a static scatter
backend under VSCode; `cell_gallery.py` already degraded hover gracefully when the ipympl
timer was unavailable). Click/hover reliability therefore varied across JupyterLab,
VSCode, and Voila.

The tile *pixels* are produced by front-end agnostic numpy code (`_render_roi_tile` →
`render_roi_to_array`; `_render_tile_for_index` in the cell gallery). Only the **display +
event layer** was coupled to Matplotlib.

## Feasibility assessment (original question: "replace the gallery with Bokeh?")
- **Bokeh**: feasible (native `TapTool`/`HoverTool`, BokehJS sidesteps ipympl) **but**
  adds a genuinely new runtime dependency (`bokeh`/`jupyter_bokeh`) with BokehJS
  asset-loading quirks, and is over-fit for a static thumbnail grid (`image_rgba` needs
  fiddly RGBA-uint32 packing). **Not adopted.**
- **anywidget image grid (chosen)**: `anywidget` is already a dependency, used by 5
  plugins, with an established `_css`/`_esm`/`sync=True` traitlet pattern and a headless
  `traitlets.HasTraits` fallback for CI (see `mask_class_list_widget.py`). Adds **no new
  dependency**, matches in-repo conventions, and removes the ipympl backend entirely.

**Decisions (confirmed with user):** migrate *both* galleries now; hover label is an
**in-tile CSS tooltip** (no Python round-trip).

## Implementation

### 1. Shared widget — `ueler/viewer/plugin/tile_gallery_widget.py` (new)
- `TileGalleryWidget(anywidget.AnyWidget)` with synced traits:
  - `tiles` — list of `{"id": str, "src": str, "label": str}` (`src` = base64 PNG data-URI).
  - `columns` — grid column count.
  - `clicked` — JS→Python signal, set to `"<id>|<nonce>"` on tile click.
  - `_esm` renders a `<div class="tg-grid">` of `<img>` tiles; click →
    `model.set('clicked', id + '|' + (nonce++)); model.save_changes()`; hover tooltip is
    pure CSS. Headless `traitlets.HasTraits` fallback for tests/CI.
- Module helpers:
  - `array_to_data_uri(arr)` — float `[0,1]`/uint8/grayscale → PNG data-URI (Pillow →
    imageio → skimage fallback chain, in-memory `BytesIO`).
  - `text_placeholder_uri(msg)` — light-gray tile with centered message (Pillow), used
    for "No channels" / "Preview unavailable".
  - `parse_clicked_id(value)` — strips the `|<nonce>` suffix via `rsplit("|", 1)` so ids
    that themselves contain `|` are preserved.

### 2. ROI Manager — `ueler/viewer/plugin/roi_manager_plugin.py`
- `_build_browser_widgets`: `browser_gallery = TileGalleryWidget(columns=BROWSER_COLUMNS)`
  inside the scrollable `browser_output` `VBox`; removed `browser_output_inner` (`Output`)
  and `browser_hover_label`.
- `_refresh_browser_gallery`: unchanged filtering/pagination/signature-throttle; builds
  `tiles = [{"id", "src", "label"}]` for the page (reusing `_render_roi_tile`,
  `_build_marker_profile`, `format_roi_label`; `text_placeholder_uri` for message tiles)
  and assigns `browser_gallery.tiles`/`.columns`.
- `_connect_events`: `browser_gallery.observe(self._on_gallery_clicked, names="clicked")`.
- New `_on_gallery_clicked` → `parse_clicked_id` → existing `_activate_roi_from_browser`.
- Removed: `_determine_gallery_layout`, `_resolve_browser_dpi`, `_ensure_browser_css`,
  `_on_browser_click`, `_on_browser_motion`, `_clear_browser_hover`,
  `_disconnect_browser_events`, related state, and the `matplotlib.pyplot` import.

### 3. Cell Gallery — `ueler/viewer/plugin/cell_gallery.py`
- Split `create_gallery()` → `create_gallery_tiles()` (returns `(images, color_range,
  displayed_indices)`) + a backwards-compatible `create_gallery()` wrapper that composes
  the canvas (keeps `_compose_canvas` and its test working).
- `__init__`: `self.gallery = TileGalleryWidget(columns=GRID_COLUMNS)` + `observe(...clicked)`;
  removed `plot_output` (`Output`) and hover-timer state.
- `plot_gellery` → `create_gallery_tiles` → rewritten `_draw_gallery(images,
  displayed_indices)` builds `{id=str(row_index), src, label="<fov>: <mask id>"}`.
- New `_on_gallery_clicked` → `focus_on_cell(...)` preserving `_skip_next_fov_refresh`.
- Removed: figure/`plt.show`/`mpl_connect` path, `on_mouse_move`, `process_hover_event`,
  `_create_annotation`, `_update_tile_metadata`, `plot_output`, `matplotlib.pyplot` import.

## Tests
- New `tests/test_tile_gallery_widget.py` (9): PNG encoding for float/uint8/grayscale +
  empty; placeholder URI; `parse_clicked_id` (nonce strip, pipe-preserving, empty); trait
  round-trip and nonce re-click firing.
- Updated `tests/test_roi_manager_tags.py`: gallery-widget structure/columns assertions
  replacing `_determine_gallery_layout` and `browser_output_inner`.
- Updated `tests/test_cell_gallery.py`: `TestTileGalleryRendering` (tile population,
  empty-clear, click→`focus_on_cell`, empty-payload ignore) replacing the obsolete
  `TestDrawGalleryRendering`; logger-based `_show_warning` test.
- Run (conda `ark-analysis-dask_yw`):
  `python -m unittest tests.test_tile_gallery_widget tests.test_roi_manager_tags tests.test_cell_gallery`
  → 69 passed. Full suite: **no new failures** vs. baseline (2 pre-existing failures
  incidentally fixed).

## Follow-up fix — "Preview unavailable" for all FOV-based ROIs
After the migration, all FOV-based ROI thumbnails showed "Preview unavailable". Root cause was
a **pre-existing** variable-ordering bug in `_render_roi_tile` (introduced in `f9c9996`, issue
#91's "No image (masks only)" mode), not the gallery migration: `snapshot` was referenced in the
`render_roi_to_array(...)` call (`skip_image_layer=bool(getattr(snapshot, ...))`) before it was
assigned a few lines below, raising `UnboundLocalError` that the surrounding
`except Exception: return None` swallowed — so the renderer returned `None` for every FOV-based
ROI. Map-mode ROIs were unaffected (different code path). Fix: move
`snapshot = self._build_overlay_snapshot(record, fov_name)` above the render call, simplify to
`skip_image_layer=bool(getattr(snapshot, "skip_image_layer", False))`, and log
`_logger.warning(..., exc_info=True)` in the `except` so future render failures are visible.
Regression coverage: `tests/test_roi_manager_tags.py::ROIManagerThumbnailRenderTests`.

## Manual verification (recommended before release)
Open `script/run_ueler_CRC_cohort.ipynb` in both JupyterLab/browser and VSCode:
- ROI Manager: clickable thumbnail grid; click centers/activates ROI + applies preset;
  re-clicking the selected ROI re-fires; hover shows the ROI label; pagination + filters
  behave as before.
- Cell Gallery: clickable thumbnails; click focuses the cell; hover shows `<fov>: <mask
  id>`; no ipympl backend involved.
