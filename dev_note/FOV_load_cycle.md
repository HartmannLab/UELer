# FOV (image) load cycle — concise overview

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

  Detailed rendering steps (what `render_image` does before the canvas update)
  - Determine requested channels and ensure their image arrays are available in `image_cache` for the requested downsample factor (the viewer may call `load_fov(...)` internally to materialise missing channel images).
  - Compute the pixel bounds to render (visible region) and corresponding downsampled indices `xym` / `xym_ds`. This reduces bandwidth and work for large images.
  - For each channel:
    - Read or slice the downsampled tile from cache (or compute the downsample on demand).
    - Apply contrast windowing: scale channel values to [0..1] using stored channel min/max or cached percentile-derived maxima (`channel_max_values`).
    - Map channel intensities to RGB via the configured color control (colour/colormap) to produce an RGB image for that channel.
  - Combine per-channel RGB layers into a single RGB image using additive or configured overlay rules (the viewer's code merges channels into `combined_np`).
  - If annotation overlays are enabled, fetch the downsampled annotation array for the FOV and blend a discrete colormap into the RGB image with configured alpha/transparency.
  - If mask edges or fills are enabled, use precomputed downsampled masks to draw edges (e.g., boundary detection) or filled color overlays; mask edge drawing typically writes colors into the combined RGB buffer at mask boundary pixels.
  - Clip and convert the combined RGB image to a float32 array in range [0..1] (or an 8-bit array, depending on the downstream display pipeline).
  - Return the combined array to `update_display` which sets it on the Matplotlib image artist and triggers a canvas redraw.

  Intermediates stored on the viewer object
  - `image_cache` (dict[fov_name -> dict[channel -> ndarray or None]])
    - Top-level per-FOV dictionary mapping channel names to either a loaded ndarray or a placeholder until the channel is materialised.
    - Used to avoid reloading pixel data from disk for recently accessed FOVs.
  - `mask_cache` (dict[fov_name -> dict[mask_name -> ndarray]])
    - Stores original mask arrays per FOV; downsampled variants are stored separately (see below).
  - `label_masks_cache` (dict[fov_name -> dict[mask_name -> dict[factor -> ndarray]]])
    - Per-FOV, per-mask, per-downsample-factor label masks used for fast overlay and boundary computation.
  - `edge_masks_cache` (dict[fov_name -> dict[mask_name -> dict[factor -> ndarray]]])
    - Precomputed edge/boundary masks at various downsample levels. Typically used to draw colored edges on `combined_np`.
  - `annotation_cache` / `annotation_label_cache` (dict-like)
    - Stores annotation rasters and their downsampled variants per FOV. Used when annotation overlays are blended into the RGB buffer.
  - `current_label_masks` (dict[mask_name -> ndarray])
    - Short-lived set of label mask slices (cropped to the visible region) used by `image_display.update_patches()`.
  - `full_resolution_label_masks` (dict[mask_name -> ndarray])
    - The unfactored label masks kept for sampling and high-resolution exports.
  - `image_display.combined` (ndarray)
    - The last composed RGB buffer that was passed to the Matplotlib image artist. Useful for inspection or incremental updates.
  - `channel_max_values` / `channel_*` metadata
    - Per-channel maxima / contrast metadata used when normalising intensities for display.

  Notes
  - Caches are bounded by `max_cache_size` and are trimmed as FOVs are evicted (LRU behaviour via OrderedDict semantics).
  - Downsample factors are precomputed for masks and annotations to avoid repeated full-resolution resampling during interactive pan/zoom.
  - Plugins may read or mutate these caches (for example, to add cluster overlays) — plugin authors should avoid large synchronous writes on the main thread.

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

## Future improvements
### **Enable loading multiple FOV tiles into the same view**
Related issue: [#3](https://github.com/HartmannLab/UELer/issues/3)

### Original post
Currently, users can only view individual FOV (Field of View) images separately. It would be valuable to allow loading multiple FOV tiles into the same view if the user requests this, enabling the display of all FOVs on the same slide for easier comparison and analysis.

Implement a feature that lets users choose to load all FOV tiles together in a single view. This will require adjustments to how FOV images are currently handled, including:
- Changing image loading logic to support multiple FOVs displayed simultaneously.
- Ensuring correct placement and alignment of tiles on the slide.
- Updating the UI to provide an option for loading all FOVs into the same view.
- Handling potential performance impacts and memory management when rendering many tiles.

This enhancement will improve usability for users needing to visualize the complete set of FOVs together.

### Implementation notes

Below are implementation notes for supporting map-based multi-FOV views while preserving the current per-FOV storage format and keeping the rest of the application unchanged.

1) Keep individual FOV files and current storage unchanged
- Continue storing each FOV image and its masks/annotations as before (single-FOV files, discovered by `load_fov`).
- The virtual map layer does not change the on-disk layout or per-FOV cache entries; instead it composes from them at render time.

2) Image selector UI: show map names / slides
- Add a JSON-based map descriptor format and a simple loader. The JSON schema (example):

```json
{
    "exportDateTime": "2024-11-04T12:45:08.914Z",
    "fovFormatVersion": "1.6",
    "fovs": [
      {
        "centerPointMicrons": {
          "x": 13880,
          "y": 47536
        },
        "fovSizeMicrons": 400,
        "focusSite": "NW",
        "focusOnly": 0,
        "timingChoice": 10,
        "frameSizePixels": {
          "width": 1024,
          "height": 1024
        },
        "imagingPreset": {
          "preset": "Coarse",
          "displayName": "Coarse"
        },
        "name": "CSL005mibi_1_run1_CSL069_Hep-PKf_A1_coarse_p5ms_fov400_frame1024_1",
        "standardTarget": null,
        "sectionId": 3707,
        "notes": null,
        "slideId": 402,
        "timingDescription": "0.5 ms",
        "scanCount": 1
      },
      {
        "centerPointMicrons": {
          "x": 14280,
          "y": 47536
        },
        "fovSizeMicrons": 400,
        "focusSite": "None",
        "focusOnly": 0,
        "timingChoice": 10,
        "frameSizePixels": {
          "width": 1024,
          "height": 1024
        },
        "imagingPreset": {
          "preset": "Coarse",
          "displayName": "Coarse"
        },
        "name": "CSL005mibi_1_run1_CSL069_Hep-PKf_A1_coarse_p5ms_fov400_frame1024_2",
        "standardTarget": null,
        "sectionId": 3707,
        "notes": null,
        "slideId": 402,
        "timingDescription": "0.5 ms",
        "scanCount": 1
      }
    ]
}
```
- The JSON lists all FOVs with their spatial coordinates (in microns) and metadata.

- When a map JSON is loaded the `ui_component.image_selector` should switch to a map-mode where top-level entries are `map_name` values and the selector exposes a control to pick the active map.
- Keep a toggle or selection mode: single-FOV mode (current behavior) vs map mode. In map mode, selecting a map activates the virtual image layer (see below).

3) Virtual image layer (responsibility and design)
- Purpose: act as a compositing layer that stitches the relevant FOV images into a single virtual image for the current viewport. It will own the process of deciding which FOVs intersect the current view and produce a stitched `combined` image for `image_display`.

- API surface (suggested):
  - class VirtualMapLayer:
    - __init__(viewer, map_descriptor, downsample_factors)
    - set_viewport(xmin, xmax, ymin, ymax, downsample_factor)
    - get_combined_image(selected_channels, xym, xym_ds) -> ndarray

- Responsibilities and behaviours:
  - Maintain a lightweight index of FOV positions (from the JSON) and their original pixel sizes.
  - Given a viewport and downsample factor, compute which FOVs intersect the viewport (bounding-box intersection) and the appropriate offsets to place each downsampled tile into the stitched canvas.
  - For each required FOV & channel, use the viewer's existing `image_cache` or `load_one_channel_fov` to obtain a downsampled tile (prefer using precomputed downsampled arrays if available). Do not reimplement disk loading logic — reuse `load_fov`.
  - Stitch downsampled tiles into a temporary buffer sized to the requested `xym_ds` region (not full-resolution). This keeps memory usage and compute bounded by the display scale.
  - Apply per-channel contrast/color mapping and overlays using the same `render_image` helpers where possible. The virtual layer can either:
    - request per-channel RGB tiles and composite them using the same additive logic, or
    - produce a stitched multi-channel tile and forward it to the existing `render_image` path for color mapping.
  - Produce and return a combined RGB buffer matching the `image_display` expected shape. The viewer then assigns it to `image_display.combined` and draws it. The virtual layer should not directly write to the Matplotlib artist; follow the same `update_display` handoff.

4) Cache & invalidation strategy
- Virtual layer should maintain a small in-memory cache of stitched tiles keyed by (viewport quantised to tiles, downsample_factor, selected_channels, map_name) to speed panning when the view doesn't change much.
- When a single FOV's underlying data changes (masks, annotations, channel maxima), the virtual layer should be notified (via `viewer.inform_plugins` or explicit `invalidate([fov])`) to drop affected cache entries.
- Respect the viewer's `max_cache_size` for per-FOV caches; the virtual layer cache should have its own small tunable limit.

5) Mask and annotation handling
- Masks and annotations remain per-FOV. The virtual layer should request downsampled masks/annotations for each FOV tile and composite them into the stitched RGB in the same order/opacity used by `render_image`.
- Update `current_label_masks` and `full_resolution_label_masks` semantics: when in map mode, `current_label_masks` should contain the stitched, visible-region mask slices (indexed by mask_name or by unique key combining map+mask) so UI patching logic can continue to work unchanged. The virtual layer can produce these derived masks before `image_display.update_patches()` runs.

6) UI & compatibility considerations
- Keep existing single-FOV behavior unchanged and opt into map-mode via a UI toggle. When map-mode is active, most of the viewer internals should behave as if the viewer had a single large virtual image; only the virtual layer composes from multiple FOVs.
- Keep plugin hook names unchanged (`on_fov_change`, `on_mv_update_display`). When map-mode is active, `on_fov_change` should be triggered when the selected map changes, and plugins that need per-FOV semantics may be passed the list of active FOVs in the viewport.

7) Performance and testing notes
- Stitching only at the downsampled scale reduces memory and CPU usage dramatically; implement and test for typical microscope slide scales (e.g., 4x/8x downsampled tiles).
- Add tests that simulate map JSONs with a few adjacent FOVs and assert correct stitching offsets, overlay behavior, and cache invalidation when underlying FOVs are updated.
- Measure latencies (panning, zooming) and tune the virtual layer cache size and downsample factor thresholds to maintain interactive frame rates.

8) Migration steps
- Add the JSON map loader and a minimal VisualMapLayer implementation behind a feature flag toggled in the UI.
- Once stable, expose the feature to users with documentation on how to author map JSON files (coordinate system, units, expected image sizes).

