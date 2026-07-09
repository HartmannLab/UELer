# Issue #92 — Batch Export Mask Customization Features

## Problem

The batch export plugin (`BatchExportPlugin`) always derives mask appearance from the viewer's live state at export time. Users needed three independent controls:

1. **Apply a saved mask painter palette** to exported masks without altering the viewer's live state.
2. **Export masks on a black background** ("Masks only") independently of the viewer's "No image" toggle.
3. **Save and reuse batch export UI settings** as named templates.

## Implementation

All changes are in `ueler/viewer/plugin/export_fovs.py`.

### Feature 1 — Palette override

- New state: `_palette_registry_folder`, `_palette_registry`
- New helpers: `_resolve_palette_registry_folder()`, `_load_palette_registry()`, `_refresh_palette_dropdown()`, `_snapshot_from_palette_payload(payload, outline_thickness)`
- `_snapshot_from_palette_payload` builds a `MaskPainterSnapshot` directly from a saved palette JSON without modifying `MaskPainterDisplay` state (safe for background threads).
- `_capture_overlay_snapshot` accepts a new `palette_name` kwarg; when set and found in the registry, it reads the palette file and replaces `snapshot.mask_painter` with the palette-derived snapshot.
- `_start_export` reads `mask_palette_enabled` / `mask_palette_dropdown` and passes `palette_name` to `_capture_overlay_snapshot`.
- `refresh_overlay_capabilities` calls `_refresh_palette_dropdown()` so the dropdown is always current.
- New widgets: `mask_palette_enabled` (Checkbox), `mask_palette_dropdown` (Dropdown), `mask_palette_box` (VBox).
- `_palette_registry` uses `palette_store.load_registry` with `mask_painter.REGISTRY_FILENAME`.

### Feature 2 — Masks-only mode

- New widget: `masks_only` (Checkbox) added to the include-checkboxes HBox.
- `_capture_overlay_snapshot` overrides `skip_image_layer` from the export-local `masks_only` widget, independent of the viewer's "No image" toggle.
- `_refresh_mask_controls` disables and resets `masks_only` when masks are unavailable.
- Cache invalidation wired via `_invalidate_overlay_cache` observer.

### Feature 3 — Export config templates

- New constants: `EXPORT_CONFIG_FILE_SUFFIX`, `EXPORT_CONFIG_REGISTRY_FILENAME`, `EXPORT_CONFIG_VERSION`
- Storage: `{base_folder}/.UELer/export_configs/`; reuses `palette_store` for all I/O.
- New state: `_export_config_folder`, `_export_config_registry`
- New helpers: `_resolve_export_config_folder()`, `_refresh_config_dropdown()`, `_collect_export_config(name)`, `_apply_export_config(payload)`, `_save_export_config()`, `_load_export_config()`, `_delete_export_config()`
- New UI: accordion section with `config_name_input`, `config_save_button`, `config_saved_dropdown`, `config_load_button`, `config_delete_button`, `config_status`.

### Refactor

- Extracted `_invalidate_overlay_cache()` helper (replaces two duplicate `_overlay_snapshot=None; _overlay_cache.clear()` pairs).

## Files changed

- `ueler/viewer/plugin/export_fovs.py` — all three features
- `tests/test_export_fovs_batch.py` — updated widget stubs; added tests for masks_only, palette override, and cache invalidation
- `tests/test_export_fovs_mask_customization.py` — new file for Feature 3 config template tests

## Tests

```bash
python -m unittest tests.test_export_fovs_batch tests.test_export_fovs_mask_customization
```

---

## Reply Follow-Up

### Bug Fix — Mask outline color ignores left-panel class colors

**Root cause:** `_snapshot_from_palette_payload` hardcoded `mask_type_color=default_color` (the palette's fallback fill color) instead of the actual mask overlay color from the viewer's left panel. When `border_color_mode="mask_type_color"` (the default), the render engine used the wrong color for filled-mask borders.

**Fix:** Added an optional `mask_type_color` parameter to `_snapshot_from_palette_payload`. In `_capture_overlay_snapshot`, the live `snapshot.mask_painter.mask_type_color` is now passed through when overriding with a palette, ensuring the correct left-panel overlay color is preserved.

### Feature Enhancement — Output folder in export config

Added `output_path` to `_collect_export_config` and `_apply_export_config` so the `Output folder` field is saved and restored with config templates.

### UI Enhancement — Tab-based mode selection and separator lines

- Removed the `mode_selector` `ToggleButtons` widget. The existing `mode_tabs` `Tab` widget now serves as the sole mode selector; `_start_export` derives the active mode from `mode_tabs.selected_index`.
- Removed `_on_mode_change` (no longer needed) and the `ToggleButtons` import.
- Added horizontal separator `HTML` lines in `_build_layout` after the DPI field and after the Scale bar % width slider.

### Files changed (reply)

- `ueler/viewer/plugin/export_fovs.py` — bug fix, output_path in config, Tab mode selection, separator lines
- `tests/test_export_fovs_batch.py` — removed ToggleButtons stub; added 3 new mask_type_color tests
- `tests/test_export_fovs_mask_customization.py` — added output_path stub; updated expected fields; added output_path restore test
