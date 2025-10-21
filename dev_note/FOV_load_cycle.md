## FOV (image) load cycle — concise overview

This document describes the runtime steps the viewer takes when loading and displaying a Field of View (FOV). It references the main functions and files involved so developers can quickly understand the call flow.

High-level sequence (call order)
- UI event: user changes FOV selection (widget) → `ui_components.image_selector` value change triggers `ImageMaskViewer.on_image_change` (`ueler/viewer/main_viewer.py`).
- `on_image_change(change)`
  - reads the newly selected FOV name from `self.ui_component.image_selector.value`.
  - calls `self.load_fov(fov_name)` to ensure image and mask data are cached.
  - updates image dimensions and `image_display` properties (height/width, axis limits).
  - refreshes channel/mask/annotation controls and preserves existing selections where possible.
  - synchronizes navigation stack / toolbar state (guarded to avoid AttributeError in notebook backends).
  - calls `self.inform_plugins('on_fov_change')` to notify plugins about the FOV change.

- `load_fov(fov_name, requested_channels=None)` (`ueler/viewer/main_viewer.py`)
  - loads channel metadata for the FOV (if not cached) via the channel loader helpers.
  - for each requested channel: loads pixel data on demand using `load_one_channel_fov(...)` when the channel image is missing.
  - updates caches (image_cache, mask_cache, label_masks_cache, edge_masks_cache, annotation_cache) and enforces cache size limits.
  - precomputes downsampled masks and annotation downsample variants for fast rendering.

- After `on_image_change` finishes updating metadata the UI calls the display refresh path:
  - `update_display(downsample_factor)` (`ueler/viewer/main_viewer.py`) computes the visible area and downsampled bounds.
  - `render_image(selected_channels, downsample_factor, xym, xym_ds)` produces the combined RGB array:
    - loads/merges channels, applies contrast/color mappings
    - overlays annotations and masks (edges and fills) using cached downsampled arrays
  - `update_display` sets the image data on the Matplotlib/ipywidget canvas, updates mask patches, and calls `inform_plugins('on_mv_update_display')` to let plugins re-render any linked views.

Plugin & runner interactions
- Plugins subscribe to viewer lifecycle events via `inform_plugins(...)` and provide handlers such as `on_fov_change`, `on_mv_update_display`, and `on_cell_table_change` (see `ueler/viewer/plugin/*`).
- The notebook runner (`ueler/runner.py`) may call `viewer.on_image_change(None)` as part of its refresh helpers; this call is guarded to avoid errors when side-plots are not yet initialised.

Important notes and guards
- The viewer uses lazy on-demand image loading so only requested channels are materialised from disk or storage.
- Many operations are defensive: toolbar/nav-stack accesses are guarded because notebook backends may omit interactive toolbar objects; plugin calls are wrapped to skip missing handlers gracefully.
- Masks and annotations are downsampled at multiple factors and cached to speed up repeated panning/zooming and to avoid recomputing heavy transforms during interactive updates.

Files to inspect for implementation details
- `ueler/viewer/main_viewer.py` — primary implementation of `load_fov`, `on_image_change`, `render_image`, and `update_display`.
- `ueler/viewer/image_display.py` — canvas/display helper and patch/update helpers for overlays.
- `ueler/runner.py` — notebook-friendly entrypoints and refresh helpers (`run_viewer`, `load_cell_table`) that call `on_image_change` and `update_display` where appropriate.
- `ueler/viewer/plugin/*` — plugin hooks that respond to `inform_plugins` calls.

This summary is intentionally concise; expand any section with code excerpts or a diagram on request.
