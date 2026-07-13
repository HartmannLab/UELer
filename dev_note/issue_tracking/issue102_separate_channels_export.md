# Issue #102 — Separate Channels Export Option in Batch Export Plugin

**GitHub:** https://github.com/HartmannLab/UELer/issues/102

---

## Problem

The batch export plugin always exported a merged composite image (all selected channels blended into one RGB image). Users needing individual channels for downstream analysis had to manually split channels after export.

---

## Solution

Added an "Export channels separately" checkbox to the shared controls of the batch export plugin. When enabled, each selected channel is exported as an individual image per FOV/cell/ROI using its own configured `ChannelRenderSettings` (color and contrast as shown in the viewer).

**Applies to all three export modes:** Full FOV, Single Cells, ROIs (including map-mode ROIs).

---

## Naming Convention

- Full FOV: `{fov}_{channel}.{fmt}`
- Single Cell: `{fov}_cell_{label}__{channel}.{fmt}`
- FOV-mode ROI: `{fov}_roi_{roi_id[:12]}__{channel}.{fmt}`
- Map-mode ROI: `map_{map_id}_roi_{roi_id[:12]}__{channel}.{fmt}`

---

## Implementation Plan

1. **`separate_channels` checkbox** — added to `_build_widgets()` and inserted into the shared `controls` VBox (after `mask_palette_box`, before `config_accordion`).
2. **`_start_export()`** — reads `separate_channels` from widget and passes to `_build_job()`.
3. **`_build_job()`** — accepts and forwards `separate_channels` to all three builder methods.
4. **`_build_channel_items()` helper** — new private method that iterates over `marker_profile.selected_channels`, creates a single-channel `_MarkerProfile` per channel, and produces a `JobItem` for each.
5. **Builder methods** (`_build_full_fov_items`, `_build_single_cell_items`, `_build_roi_items`) — each accepts `separate_channels: bool = False`; when `True`, delegates to `_build_channel_items`.
6. **Config serialization** — `_collect_export_config()` and `_apply_export_config()` updated to persist/restore the `separate_channels` setting.
7. **Tests** — 4 new tests in `TestSeparateChannelsBuilders`.

---

## Files Changed

- `ueler/viewer/plugin/export_fovs.py` — checkbox widget, controls VBox, `_start_export`, `_build_job`, `_build_channel_items`, all three builders, config methods
- `tests/test_export_fovs_batch.py` — two `_build_widgets` stubs updated; 4 new tests added

---

## Tests

```bash
python -m unittest tests.test_export_fovs_batch
```

- ✅ All new tests pass (`TestSeparateChannelsBuilders` — 4 tests)
- ✅ All other existing tests pass (pre-existing failure in `test_export_fovs_batch_writes_file` is unrelated to this change)
