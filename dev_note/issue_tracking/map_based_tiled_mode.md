## Map-based tiled mode
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
- Continue storing each FOV image and its masks/annotations as before (single-FOV files, discovered by `load_fov`, see `dev_note/FOV_load_cycle.md`).
- The virtual map layer does not change the on-disk layout or per-FOV cache entries; instead it composes from them at render time.

2) Image selector UI: show map names / slides
The FOVs' spatial arrangment on a slide is described by a new JSON-based map descriptor file that lists all FOVs with their coordinates. This file is created during the image acquisition/export process and follows this schema:

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
- FOVs sharing the same `slideId` belong to the same slide/map and should be treated as a single virtual image (see below).
- When a map JSON is loaded the `ui_component.image_selector` should switch to a map-mode where top-level entries are `map_name` values and the selector exposes a control to pick the active map.
- Keep a toggle or selection mode: single-FOV mode (current behavior) vs map mode. In map mode, selecting a map activates the virtual image layer.

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

