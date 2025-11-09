## Map-Based Tiled Mode Specification

**Related issue:** [#3](https://github.com/HartmannLab/UELer/issues/3)

This document captures the product and engineering specification for rendering multiple Fields of View (FOVs) inside a single composite view. It aligns with the established runtime described in `dev_note/FOV_load_cycle.md` and must be read alongside that reference.

### 1. Objectives
- Enable users to switch between single-FOV and slide-wide map views without altering on-disk assets.
- Preserve existing FOV load, cache, and render semantics by extending—not rewriting—the flow documented in `FOV_load_cycle.md`.
- Deliver responsive navigation for slides containing dozens of FOVs while supporting all overlays (masks, annotations, painter).

### 2. Functional Requirements
- **Map discovery:** Accept one or more JSON descriptors that group FOVs by slide and expose them as selectable map entries in the UI.
- **Viewport stitching:** Composite the subset of tiles within the active viewport into the buffer consumed by `ImageDisplay.update_display`.
- **Overlay parity:** Render masks, annotations, and painter colors exactly as in single-FOV mode; mask hit-testing must operate on the stitched region.
- **Plugin compatibility:** Maintain current plugin lifecycle calls (`on_fov_change`, `on_mv_update_display`) and provide slide-aware context when required.
- **State transitions:** Users can toggle map mode at runtime; the viewer must restore single-FOV behaviour instantly when map mode is disabled.
- **Batch exports:** Support exporting stitched map views through the existing batch export pipeline, writing tiles sequentially to honour descriptor order and keep memory usage predictable.

### 3. Non-functional Requirements
- **Performance:** Match single-FOV pan/zoom responsiveness (<150 ms render queue) for slides with up to 25 tiles at downsample factors ≥4.
- **Memory:** Use downsampled buffers sized to the viewport; stitched tiles must never exceed the pixel footprint requested by `update_display`.
- **Caching:** Respect the LRU policies outlined in `FOV_load_cycle.md`; map-mode caches may not cause premature eviction of per-FOV assets.

### 4. Inputs & Data Model
- **Map Descriptor (`*.json`):**
  - `slideId` (int|string) groups FOVs into a virtual slide.
  - `centerPointMicrons` (x, y) defines tile placement; initial scope supports translation-only positioning on a shared coordinate system.
  - `frameSizePixels.width|height` supplies per-tile pixel shapes.
  - `fovSizeMicrons` enables micron-to-pixel scaling when downsample factors differ between slides.
- **Viewer State:**
  - `ImageMaskViewer.image_cache`, `label_masks_cache`, `annotation_label_cache` (see `FOV_load_cycle.md`).
  - Channel, mask, and annotation selections held by UI widgets.
- **Runtime Flags:**
  - `map_mode_enabled` (bool): top-level toggle.
  - `active_map_id` (slideId) and optional `active_map_name`.

### 5. Architecture
- **VirtualMapLayer (new component):**
  - Created when map mode initializes; receives the viewer instance, resolved map descriptor, and allowed `DOWNSAMPLE_FACTORS`.
  - Exposes `set_viewport(xmin, xmax, ymin, ymax, downsample)` and `render(selected_channels, xym, xym_ds)`.
  - Internally selects tiles that fall within the current viewport, requests channel assets through the existing `load_fov` pipeline, and stitches RGB overlays.
- **Integration with FOV cycle:**
  - `ImageMaskViewer.on_image_change` detects map selections and calls `load_fov` for all tiles referenced by the map so caches warm before render.
  - `update_display` delegates to `VirtualMapLayer.render` when map mode is active; otherwise it executes the baseline single-FOV path.
  - `render_image` remains the source of truth for channel compositing; the virtual layer reuses it per tile to guarantee colour parity.

### 6. Stitching Workflow
1. **Viewport analysis:** Translate the current axes bounds into slide coordinates using the micron scale recorded in the descriptor.
2. **Tile selection:** Include tiles whose axis-aligned bounding boxes (AABB) fall within the viewport bounds; track per-tile pixel offsets relative to the viewport origin. No inter-tile overlap handling is required.
3. **Channel acquisition:** Ensure required channels are present via `load_fov` (idempotent); reuse cached downsampled arrays when available.
4. **Tile rendering:** Call `render_image` per tile using the tile-specific `xym` window; discard plugin notifications emitted during these calls.
5. **Canvas assembly:** Place each tile’s RGB output into the stitched buffer using computed offsets. Regions with no tile coverage are filled with zeros (black).
6. **Overlay merge:**
   - Masks: read from `label_masks_cache`, respecting outline thickness; aggregate into map-mode `current_label_masks` keyed by composite map identifiers.
   - Annotations: sample from `annotation_label_cache` and merge embeddings using palette metadata.
7. **Painter integration:** Run `collect_mask_regions` across the selected tile subset and call `apply_registry_colors` once on the final buffer.
8. **Publish:** Write the stitched buffer to `self.image_display.combined` and continue the standard `ImageDisplay` update path.

### 7. UI Specification
- **Selector Changes:**
  - Add a toggle labelled “Map mode” adjacent to the existing FOV selector.
  - When enabled, replace the selector options with map identifiers (e.g., `slideId`, human-readable `map_name`).
  - Provide a read-only summary of the number of tiles and coverage extents.
- **Status Indicators:**
  - Show a badge when per-tile loads are in flight (`Loading 12/32 tiles…`).
  - Display warning banners if the descriptor omits any FOV present in the dataset or contains malformed coordinates.
- **Tile metadata:** Surface per-tile FOV names via hover tooltips (mirroring single-FOV mode) while keeping the composite view uncluttered.
- **Plugin Notifications:**
  - Emit `on_fov_change(map_id, visible_fovs=[...])` so existing plugins can inspect which FOVs populate the viewport.
  - Preserve toolbar navigation history by recording map identifier entries alongside single-FOV states.

### 8. Cache & Invalidation
- Maintain a dedicated `map_tile_cache: OrderedDict[CacheKey, np.ndarray]` where `CacheKey = (map_id, fov_name, downsample, channel_signature)`.
- Invalidate cache entries when:
  - `load_fov` evicts a tile from `image_cache` (hook into the same eviction branch noted in `FOV_load_cycle.md`).
  - Viewer settings that affect render output change (channel colour, contrast, mask toggles, annotation palette).
- Limit stitched viewport caches to N entries (default 6) and drop least recently used when full.

### 9. Error Handling
- Gracefully disable map mode with a user-facing message if descriptor parsing fails.
- Fallback to blank regions with toast notifications when individual tile loads fail; log detailed tracebacks for diagnostics.
- Ensure plugin notifications still fire with `visible_fovs=[]` so listeners can reset state if a render fails.

### 10. Testing Strategy
- **Unit Tests:**
  - Descriptor parsing (valid/invalid JSON, mixed coordinate units).
  - Tile selection against viewport bounds (edge-touching, partially visible, fully outside).
  - Stitch assembly verifying pixel alignment and zero-fill gaps.
- **Functional Tests:**
  - Simulate map mode toggles inside `tests/test_rendering.py` or new spec dedicated suite.
  - Confirm mask hover hit-testing uses stitched `current_label_masks`.
  - Verify plugin notifications receive correct `visible_fovs` payloads.
  - Exercise batch export paths that target stitched map views and confirm tiles stream to disk following descriptor order.
- **Performance Harness:**
  - Benchmark render latency and memory usage for slides with 9, 16, and 25 tiles at downsample factors 4 and 8.

### 11. Rollout Plan
- Gate map mode behind `ENABLE_MAP_MODE` feature flag.
- Phase deployment:
  1. Land descriptor parser and virtual layer behind the flag, default off.
  2. Add UI toggle and smoke tests; collect feedback from notebook users.
  3. Document translation-only coordinate requirements and release sample descriptors.
- Update `README` and `doc/log` during rollout phases to communicate the feature and its flag status.

### 12. Implementation Plan
1. **Descriptor ingestion groundwork**
  - Build `MapDescriptorLoader` with validation against the translation-only schema; add fixtures covering mixed units and malformed entries.
  - Integrate loader into viewer bootstrap behind `ENABLE_MAP_MODE`, wiring errors to the new user-facing warnings.
  - Extend tests to assert parsed descriptors populate slide and FOV registries deterministically.
2. **VirtualMapLayer core**
  - Implement tile indexing, viewport intersection math, and stitched buffer assembly using existing `render_image` calls. _Status: done via `ueler/viewer/virtual_map_layer.py` with `_compose_fov_image` helper in `main_viewer`._
  - Thread cache hooks (`map_tile_cache`, invalidation triggers) through `ImageMaskViewer` in alignment with `FOV_load_cycle.md` semantics. _Status: done via viewer-managed tile cache, eviction hooks on FOV LRU pops, and render-state signatures shared with `VirtualMapLayer`._
  - Add unit coverage for offset alignment, zero-fill gaps, and cache eviction behaviour. _Status: done in `tests/test_virtual_map_layer.py`._
3. **UI integration and exports**
  - Add the map-mode toggle, selector wiring, and tooltip metadata to `ui_components`; ensure plugin notifications receive slide context.
  - Update batch export plumbing to recognise stitched renders and stream tiles in descriptor order; add corresponding functional tests.
  - Document user workflow and update smoke notebooks to exercise toggle, overlays, and export scenarios.

### 13. Dependencies & References
- Relies on `ImageMaskViewer.load_fov`, caches, and `render_image` entry points documented in `dev_note/FOV_load_cycle.md`.

### 14. Acceptance Criteria
- Map mode renders stitched tiles with masking and annotation parity relative to single-FOV mode.
- Toggling map mode on/off does not leave stale cache entries or break navigation history.
- Automated tests cover descriptor parsing, stitching, and overlay correctness; manual notebook validation confirms performance targets.
