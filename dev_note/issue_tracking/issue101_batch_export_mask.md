# Issue #101 — Batch Export Does Not Handle Mask Properly

**GitHub:** https://github.com/HartmannLab/UELer/issues/101

---

## Problem Description

When the Batch Export plugin's `Include Mask` checkbox is checked, exported images do not include the mask overlay as expected. Two specific bugs were observed:

1. **Bug 1:** When the Mask Painter plugin is enabled, live per-cell annotation colours are applied to the export even when `Override Mask Palette` is **not** checked.
2. **Bug 2:** When the Mask Painter plugin is disabled, no mask overlay is included in the export at all, even when `Include Mask` is checked.

Both behaviours should be independent of the viewer's live state and the Mask Painter plugin status.

---

## Root Cause

Both bugs are in `_capture_overlay_snapshot` in `ueler/viewer/plugin/export_fovs.py` (lines 1350–1426).

### Bug 1

`capture_overlay_snapshot` (main_viewer.py) always captures the live `MaskPainterSnapshot` when the Mask Painter is enabled. The export plugin's `_capture_overlay_snapshot` retained this snapshot unchanged when `palette_name is None` (no Override checked). Later, `_export_fov_worker` calls `apply_overlay_snapshot_to_array`, which applies painter colours unconditionally whenever `snapshot.mask_painter is not None` — regardless of whether Override Mask Palette was requested.

### Bug 2

`capture_overlay_snapshot` only adds `MaskOverlaySnapshot` entries for mask layers whose checkbox is **ticked** in the viewer's left panel. When the Mask Painter is disabled and no mask-layer checkboxes are ticked, `snapshot.masks` is empty and `snapshot.mask_painter` is `None`. Neither rendering stage produces any mask output, leaving the exported image without any mask overlay even though `Include Mask` is checked.

---

## Fix

Both fixes are added to `_capture_overlay_snapshot` in `export_fovs.py`, immediately after the outline-thickness adjustment block.

### Fix 1 — Strip live painter when no palette override is requested

```python
if palette_name is None and getattr(snapshot, "mask_painter", None) is not None:
    snapshot = replace(snapshot, mask_painter=None)
```

This prevents per-cell Mask Painter colours from leaking into the export when the user has not explicitly requested a saved palette override.

### Fix 2 — Add fallback mask outline when nothing was captured

```python
if include_masks and palette_name is None and not snapshot.masks and getattr(snapshot, "mask_painter", None) is None:
    mask_key = str(getattr(self.main_viewer, "mask_key", "") or "")
    if mask_key:
        # Read colour from the viewer's mask colour controls
        ...
        snapshot = replace(snapshot, masks=(fallback,))
```

When `Include Mask` is on but no mask content was captured (painter disabled, no checkboxes ticked), a simple `MaskOverlaySnapshot` outline is injected for the primary mask key using the viewer's configured colour.

---

## Files Changed

- `ueler/viewer/plugin/export_fovs.py` — `_capture_overlay_snapshot` method
- `tests/test_export_fovs_batch.py` — updated 1 existing test + added 4 new tests

---

## Tests

```bash
/omics/groups/OE0622/internal/shared_envs/ark-analysis-dask_yw/bin/python -m unittest tests.test_export_fovs_batch
```

New/updated tests:
- `test_batch_export_snapshot_strips_painter_when_no_palette_override` (updated from old thickness test)
- `test_palette_override_preserves_outline_thickness` (new: thickness still works via palette path)
- `test_live_painter_stripped_when_no_palette_override` (new: Bug 1 regression)
- `test_fallback_mask_added_when_no_masks_and_painter_disabled` (new: Bug 2 regression)
- `test_no_fallback_mask_when_palette_override_is_set` (new: no double overlay with palette)

---

## Reply — Explicit Mask Layer Selector (Follow-Up Fix)

### Problem

After the Bug 1 + Bug 2 fix landed, `Include Mask` still did not work reliably in practice. Root cause: `_refresh_mask_controls()` called `refresh_overlay_capabilities()`, which checked whether any mask-layer checkboxes were ticked in the **viewer's live panel**. If none were ticked, it silently reset `include_masks.value = False` and disabled the checkbox, overriding the user's intent before the export even ran.

### Solution

Added an explicit **mask layer dropdown** and **color picker** directly inside the batch export plugin UI, making mask export fully independent of the viewer's live overlay state.

### Changes

- **`ueler/viewer/plugin/export_fovs.py`**
  - Added `ColorPicker` to the `ipywidgets` import block.
  - Added `mask_layer_dropdown` (Dropdown listing all `main_viewer.mask_names`), `mask_color_picker` (ColorPicker, default `#ffffff`), and `mask_layer_box` (VBox container) in `_build_widgets()`.
  - Inserted `mask_layer_box` between `mask_outline_thickness` and `overlay_hint` in the UI layout.
  - Added observers in `_connect_events()` for both new widgets to call `_invalidate_overlay_cache()` on change.
  - Added `_refresh_mask_layer_dropdown()`: populates the dropdown from `main_viewer.mask_names` (falls back to `mask_key`); preserves the current selection if still valid.
  - Updated `_refresh_mask_controls()`: removed `visible_masks` parameter and the block that disabled `include_masks` when no viewer panel checkboxes were ticked; added enable/disable of the new widgets; calls `_refresh_mask_layer_dropdown()` when masks are available.
  - Updated `refresh_overlay_capabilities()`: removed `visible_masks` calculation and removed it from the `_refresh_mask_controls` call.
  - Replaced the Bug 1 strip + Bug 2 fallback blocks in `_capture_overlay_snapshot()` with a single unified path: when `include_masks=True` and `palette_name is None`, always build a `MaskOverlaySnapshot` from the export-local dropdown value and color picker value, independent of live viewer state.
  - Updated `_collect_export_config()` and `_apply_export_config()` to serialize/restore `mask_layer` and `mask_color`.

- **`tests/test_export_fovs_batch.py`**
  - Added `mask_layer_dropdown` and `mask_color_picker` stubs to all `_build_widgets` helpers and manual `SimpleNamespace` fixtures.
  - Renamed `test_fallback_mask_added_when_no_masks_and_painter_disabled` → `test_export_local_layer_and_color_used_when_no_palette_override` (updated to exercise the widget-based path).
  - Added 5 new tests: `test_refresh_mask_layer_dropdown_populates_from_mask_names`, `test_refresh_mask_layer_dropdown_defaults_to_mask_key`, `test_capture_snapshot_uses_export_local_layer_dropdown`, `test_capture_snapshot_painter_is_none_without_palette_override`, `test_config_roundtrip_includes_mask_layer_and_color`.

### Tests

```bash
/omics/groups/OE0622/internal/shared_envs/ark-analysis-dask_yw/bin/python -m unittest tests.test_export_fovs_batch
```

Result: 66 tests — 65 passed, 1 pre-existing unrelated failure (`test_export_fovs_batch_writes_file`).
