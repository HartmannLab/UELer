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

---

## Downsampling

- Downsample factors are computed from the current viewport size and the FOV resolution.
- `select_downsample_factor` clamps the factor to an allowed list to avoid blur artifacts.
- ROI thumbnails use a separate downsample path that respects thumbnail canvas size.

---

## Channel Controls

- **Visibility toggles** — Each loaded channel can be toggled independently without modifying the selection list.
- **Color legend** — A legend widget shows the current color assignments for all visible channels.
- **Channel grid view** — Renders each visible channel as a separate labelled pane in a synchronized Matplotlib subplot grid (Issue #76).

---

## Tooltips

- Tooltip column lookup uses viewer-configured keys rather than hard-coded column names.
- Resolved rows are cached to avoid repeated DataFrame lookups on hover.

---

## VS Code Fallback

When `jupyter-scatter` widget rendering fails in VS Code, UELer automatically replaces the scatter widget with a static Matplotlib figure. The chart plugin controls remain functional.

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
