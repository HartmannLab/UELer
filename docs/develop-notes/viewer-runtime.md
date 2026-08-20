# Viewer Runtime & UI

> Source: [`dev_note/topic_viewer_runtime_ui.md`](https://github.com/HartmannLab/UELer/blob/main/dev_note/topic_viewer_runtime_ui.md)

---

## Context

These notes cover the main viewer runtime, downsampling behavior, channel controls, tooltips, and notebook-specific behavior.

---

## FOV Load Cycle

1. User selects an FOV via the **Select Image** dropdown.
2. The viewer checks the LRU cache for the image data; if missing, it reads the TIFF files from disk.
3. The channel compositor applies per-channel color and contrast settings.
4. Overlays (masks, annotations) are composited if enabled.
5. All registered plugins receive an `on_fov_change` notification and update their views.

Note that a cached FOV holds only the channels that have actually been opened, and unchecking a
channel's visibility skips its compositing without releasing its data — the pixels stay with the
cached FOV until that FOV is evicted.

---

## Plugin Discovery

`ImageMaskViewer.dynamically_load_plugins()` scans `ueler/viewer/plugin/` for modules not prefixed
with `_`, imports each one, and instantiates every `PluginBase` subclass it finds. There is no
registry to edit: dropping a module in the directory is the registration.

The optional `allow_plugins` argument is how **simple mode** works. `display_ui()` passes
`{"roi_manager_plugin", "export_fovs"}` when `viewer.cell_table is None`, so a viewer opened without a
cell table loads only those two plugins rather than loading the analytical ones and disabling them.
That is why the right panel is short in image-only mode, and why the single-cell plugins cannot be
coaxed into appearing without a table.

`PluginBase` declares the lifecycle hooks the viewer broadcasts through `inform_plugins()`:
`after_all_plugins_loaded`, `on_fov_change`, `on_cell_table_change`, `on_mv_update_display`,
`on_selection_change`, `on_map_mode_activate`, `on_map_mode_deactivate`, `on_no_image_toggle` and
`on_widget_value_change`. All are no-ops by default, so a plugin implements only what it needs.

---

## Downsampling

- Downsample factors are computed from the current viewport size and the FOV resolution.
- The target is `DOWNSAMPLE_MAX_DIMENSION = 2048` px on the longest drawn edge (`ueler/constants.py`, raised from a lower bound in #116). `calculate_downsample_factor()` doubles the factor until the largest dimension fits, so the factor is always a power of two.
- Because the calculation uses the *visible* region, zooming in shrinks it and the factor falls back toward 1 — full-resolution pixels return without a setting change. An image already at or below 2048 px gets factor 1, where the **Downsample** toggle changes nothing.
- `select_downsample_factor` clamps the factor to an allowed list to avoid blur artifacts.
- ROI thumbnails use a separate downsample path that respects thumbnail canvas size.

---

## Channel Controls

- **Channel picker** — `ChannelPickerWidget` (`plugin/channel_picker_widget.py`), an anywidget-based in-DOM picker that replaced the previous `TagsInput` in #125. It renders the full channel list with a filter box, a "*n* of *m* shown · *k* selected" counter, **Select all shown** / **Clear** actions and keyboard navigation. It falls back to a plain widget when anywidget is unavailable, and the same widget is reused by the Scatter plot, Histogram and Heatmap pickers so the selection UX is identical everywhere.
- **Chip reordering** — selected channels render as draggable chips (#126). The order drives the per-channel control rows, the legend, and the grid-view panes. It deliberately does **not** affect the composite: channels are blended additively, so the result is order-independent.
- **Visibility toggles** — Each loaded channel can be toggled independently without modifying the selection list.
- **Color legend** — A legend widget shows the current color assignments for all visible channels.
- **Channel grid view** — Renders each visible channel as a separate labelled pane in a synchronized Matplotlib subplot grid (#76). Cell location works while the grid is active (#134), which needed the locate path to resolve against the grid's axes rather than assuming the single composite canvas.
- **Contrast bounds grow, never reset.** A channel's stored maximum is the running maximum over every region computed so far, so the **Max** slider's upper bound can rise mid-session when a brighter FOV or map tile is first rendered. User-set values are preserved; only the headroom changes.

---

## Tooltips

- Tooltip column lookup uses viewer-configured keys rather than hard-coded column names.
- Resolved rows are cached to avoid repeated DataFrame lookups on hover.

---

## Static Scatter Fallback

`ChartDisplay.__init__` resolves `_scatter_backend` from `UELER_SCATTER_BACKEND`, defaulting to `widget` in every environment. An unrecognised value falls back to `widget` as well, and the comparison is case-insensitive.

Setting `UELER_SCATTER_BACKEND=static` replaces the `jupyter-scatter` widget with a static Matplotlib figure plus an inline notice; the chart plugin controls remain functional either way. This used to be selected automatically when `VSCODE_PID` was present, because `jupyter-scatter` did not render reliably in the VS Code webview. It does now, so the fallback is opt-in only (#122).

---

## Render Suppression at Startup

To prevent kernel timeouts on large maps, renders triggered by widget state restoration at startup (`load_widget_states`) are suppressed via a `_suspend_display_updates` flag. The first real render happens on the first user interaction.

---

## Related Issues

- [#61](https://github.com/HartmannLab/UELer/issues/61)
- [#64](https://github.com/HartmannLab/UELer/issues/64)
- [#66](https://github.com/HartmannLab/UELer/issues/66)
- [#75](https://github.com/HartmannLab/UELer/issues/75)
- [#76](https://github.com/HartmannLab/UELer/issues/76)
