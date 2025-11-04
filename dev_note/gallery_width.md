## Gallery Width Investigation (2025-11-04)

### Cell Gallery
- Width comes directly from the plugin constructor (`CellGalleryDisplay.__init__` in `ueler/viewer/plugin/cell_gallery.py`), which receives the fixed `width=6` inches from `ImageMaskViewer.dynamically_load_plugins`.
- When tiles are rendered (`_draw_gallery`), the figure is instantiated with `plt.subplots(figsize=(self.width * 0.9, fig_height))`, so the canvas width is locked to `6 * 0.9 = 5.4"` (≈389 px at Matplotlib's 72 dpi default).
- No additional layout overrides are applied to the Matplotlib canvas. Because ipympl respects the requested figure size, the gallery width remains stable regardless of notebook or accordion resizing.

### ROI Gallery
- The ROI manager plugin is also constructed with `width=6`, but `_determine_gallery_layout` in `ueler/viewer/plugin/roi_manager_plugin.py` performs extra bookkeeping: it clamps column counts, applies an internal `GALLERY_WIDTH_RATIO` (default 0.98), subtracts a hard-coded `horizontal_padding = 0.4`, and derives a `fig_width` in inches.
- With defaults this yields `min(6 * 0.98 - 0.4, 6 - 0.4) ≈ 5.48"`, nearly matching the cell gallery width before padding.
- After plotting, `_configure_browser_canvas` wraps the ipympl canvas in a scroll box and explicitly sets `canvas.layout.width = "99%"` (and `max_width = "100%"`). The surrounding VBox also stretches to `width="100%"`.
- Because of the `99%` width override, the actual rendered width depends on whatever space the parent accordion panel or footer pane decides to allocate. Any change in container size (other plugins expanding, browser window width, sidebar toggles) reflows the canvas and produces the "unstable" width impression even though the Matplotlib figure was sized more tightly.

#### Scroll Container and W Derivation
- The ROI browser lives inside `browser_output` (an `ipywidgets.Output` created in `ROIManagerPlugin._build_browser_widgets`). Its layout fixes the vertical viewport to `BROWSER_SCROLL_HEIGHT = "400px"` and sets `overflow_y="auto"` / `overflow_x="hidden"`. This widget is the element that currently shows the scrollbar.
- `_configure_browser_canvas` (same module) wraps the Matplotlib canvas with another `VBox` whose layout duplicates the 400 px height limit and also enables `overflow_y="auto"`. In practice the outer `Output` owns the scrollbars; the inner `VBox` only ensures the canvas height is clamped before scrolling.
- Both the outer `browser_output` and the inner scroll box advertise `width="100%"`, but they inherit a `column_block_layout` parent that constrains the plugin column to `width="98%"` to avoid horizontal scroll on the accordion. As a result the widest attainable DOM width for the gallery container is whatever the accordion grants minus that 2% safety margin; call that pixel width `W`.
- The Matplotlib figure still uses the inch-based calculation from `_determine_gallery_layout` (about 5.48 in at 72 dpi) and is then stretched or squeezed by ipympl to match the widget width. There is no feedback loop from the actual DOM width to the figure sizing, so the gallery cannot enforce `canvas_width = W * 0.98` or maintain aspect until we measure `W`.
- Meeting the requested behavior means capturing the live width of `browser_output` (for example via a `ResizeObserver` injected into the widget subtree) and pushing that value back into Python so `_determine_gallery_layout` can recompute `fig_width` and `fig_height = (fig_width / columns) * rows`. Without that measurement hook the plugin only knows the static `width=6` inches supplied at construction time.

### ROI Gallery Stabilization (2025-11-04)
- Added a per-render `ResizeObserver` hook in `ROIManagerPlugin._install_gallery_resize_hook` that measures the scroll container width (`W`), assigns the outer canvas wrapper and `<canvas>` nodes to `W * 0.98`, and sets the height to `W * 0.98 * aspect_ratio` so the gallery preserves its Matplotlib tile ratio.
- The observer re-runs whenever the accordion resizes (parent width changes, sidebar toggles, browser zoom) and on window resize. The script is registered once per gallery render using a unique class token, preventing duplicate observers while keeping legacy Matplotlib fallbacks intact.
- The initial `ipympl` canvas layout now leaves width at `100%` (instead of hard-coding `99%`) so the resize hook can apply pixel-precise dimensions without fighting `Layout` constraints; the scroll container keeps its 400 px viewport and hidden horizontal overflow, ensuring only vertical scrolling engages when needed.

