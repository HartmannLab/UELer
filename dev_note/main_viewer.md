# Main Viewer

## Overview

## FOV downsampling
The main viewer auto-selects a downsample factor so large FOVs stay responsive while preserving overlays. Allowed factors are the powers of two in `constants.DOWNSAMPLE_FACTORS`; when a FOV loads the viewer samples its dimensions, calls `image_utils.calculate_downsample_factor`, and snaps the result to the nearest permitted value so the longest edge rendered in the notebook remains ≤`constants.DOWNSAMPLE_MAX_DIMENSION`, i.e. 2048 px.

While you pan or zoom, `ImageDisplay.on_draw` recomputes the factor from the current viewport. The `Advanced Settings → Downsample` checkbox toggles this behaviour: when enabled, the factor scales with zoom; when disabled, the viewer sticks to native resolution (`factor = 1`).

The **initial** factor honours the same checkbox (reply to #116). The size-based `select_downsample_factor` call runs during `__init__` *before* the widgets exist, so after `load_widget_states` restores the saved Downsample preference the viewer reconciles the initial factor: when Downsample is off it starts at native resolution (`factor = 1`), otherwise it uses the size-based factor — matching the `on_draw` path so initiation and zoom agree. The threshold is sourced from `constants.DOWNSAMPLE_MAX_DIMENSION` explicitly at every call site (`main_viewer`, `ImageDisplay.on_draw`, `channel_grid_view`) so the init and zoom thresholds cannot diverge.

Each FOV caches downsampled masks and annotations the first time it loads by slicing the rasters with `array[::factor, ::factor]`. `render_image` reuses those cached arrays and only materialises the visible window, so overlays, plugins, and batch exports share the same subsampled data without additional recomputation.

`get_axis_limits_with_padding` keeps the Matplotlib viewport aligned to factor-sized strides, and the scale bar adjusts via `effective_pixel_size_nm(pixel_size_nm, factor)` so physical measurements remain correct regardless of zoom level. Together this pipeline keeps navigation smooth and measurements trustworthy even on very large scenes.

The shared helper `image_utils.select_downsample_factor` now encapsulates the "snap to `DOWNSAMPLE_FACTORS`" logic, letting plugins such as the ROI browser request the same power-of-two factors with a different max-edge target (256 px for thumbnails).

ROI thumbnails re-compute their factor from each ROI's viewport rather than the parent FOV: the plugin measures the ROI bounds, evaluates `ratio = ceil(longest_edge / 256)`, and emits `factor = 2**ceil(log2(ratio))` (falling back to `1` when the viewport already fits). Rendering helpers also ceil-divide region bounds when deriving downsampled indices so non-divisible widths/heights still cover the entire ROI.