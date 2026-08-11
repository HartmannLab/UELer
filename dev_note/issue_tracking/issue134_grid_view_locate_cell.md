# Issue #134 — Locating a single cell in the channel grid view fails

## Problem

With the **channel grid view** active, asking the viewer to locate a single cell (Go-To plugin, cell gallery double-click, scatter/heatmap point click) silently does nothing: no error is raised, but the panes stay exactly where they were instead of centring on the requested cell.

## Investigation

Programmatic navigation lives in two `ImageMaskViewer` methods:

- `focus_on_cell()` ([main_viewer.py:3875](../../ueler/viewer/main_viewer.py#L3875)) — used by `go_to.py`, `cell_gallery.py`, `chart.py`, `chart_heatmap.py`.
- `center_on_roi()` ([main_viewer.py:3829](../../ueler/viewer/main_viewer.py#L3829)) — used by the ROI manager.

Both compute a target window and write it **only** into `self.image_display.ax`, then call `image_display.fig.canvas.draw_idle()`.

Grid mode, however, does not reuse that figure. `on_grid_view_toggle()` builds a separate `GridChannelDisplay` with its own `matplotlib` figure and its own shared-x/y axes, hides `image_output`, and shows `grid_output` ([main_viewer.py:2754](../../ueler/viewer/main_viewer.py#L2754)). The only coupling between the two is one-directional: `GridChannelDisplay._on_draw()` copies the *grid* limits back into `image_display.ax` after every pan/zoom, so that `get_axis_limits_with_padding()` keeps working.

So in grid mode, `focus_on_cell()` moves a hidden axes. The grid axes are never touched, no draw event fires on the grid canvas, and the panes never re-render — hence "no error, nothing happens".

The issue description guessed at a coordinate-transform problem between panes. That is not the cause: all panes are `sharex/sharey` and use full-resolution pixel coordinates via `set_extent`, identical to the single-pane display. The coordinates are right; they are simply written to the wrong axes.

## Solution

Make the programmatic viewport writes propagate into the grid, rather than duplicating the navigation logic per display mode.

1. **`GridChannelDisplay.set_viewport(xlim, ylim)`** — new public counterpart to the existing `get_viewport()`. Applies the limits to the primary axes (they propagate to every pane through `sharex`/`sharey`), mirrors them into `image_display.ax` so `get_axis_limits_with_padding()` sees the new window, recomputes the downsample factor for the new zoom level, re-renders the panes, and schedules a redraw.

   To avoid re-entering the render path, `set_viewport` records the new centre in `_prev_cx`/`_prev_cy` *before* the redraw, so the `draw_event` it triggers is recognised by `_on_draw()` as already handled.

2. **Shared downsample helper** — the factor computation is lifted out of `_on_draw()` into `_sync_downsample_factor(xlim, ylim)` and reused by `set_viewport`, so both paths honour `DOWNSAMPLE_MAX_DIMENSION` and the "disable downsampling" checkbox identically.

3. **`ImageMaskViewer._sync_grid_viewport()`** — small bridge that pushes the current `image_display.ax` limits into the active grid display, and is a no-op when grid mode is off. Called from `center_on_roi()` and from both exit paths of `focus_on_cell()` (the map-mode branch and the per-FOV branch), immediately after the limits are set and before the canvas redraw.

This keeps the fix on the display side: the navigation methods stay the single source of truth for *where* to look, including the stitched-map coordinate resolution, and the grid simply follows.

## Implementation steps

1. Add `_sync_downsample_factor()` and `set_viewport()` to `ueler/viewer/channel_grid_view.py`; refactor `_on_draw()` to use the helper.
2. Add `_sync_grid_viewport()` to `ueler/viewer/main_viewer.py` and call it from `center_on_roi()` and `focus_on_cell()`.
3. Extend `tests/test_channel_grid_view.py` with coverage for `set_viewport` (limits applied, `image_display.ax` mirrored, panes re-rendered, downsample recomputed, no `_on_draw` re-entry) and for the viewer bridge (grid follows `focus_on_cell` / `center_on_roi`, no-op when grid mode is off).
4. Update `doc/log.md`, `README.md`, and `dev_note/github_issues.md`.
