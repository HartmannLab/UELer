# Issue #103 — Custom ROI Names

**GitHub:** https://github.com/HartmannLab/UELer/issues/103

---

## Problem

ROIs were identified only by a UUID (`roi_id`). Display labels auto-generated from `format_roi_label()` always used `roi_id[:8]` as the suffix, and export filenames embedded `roi_id[:12]`. Users had no way to assign meaningful names to ROIs, making it hard to organize them or interpret exported files.

---

## Solution

Added a user-editable `name` field to the ROI data model. When set, the name:
- Appears in the ROI dropdown/table label in place of the `roi_id[:8]` suffix
- Is used as the filename stem in all batch export modes (FOV-mode and map-mode ROIs)

---

## Implementation

### Data Layer (`roi_manager.py`)
- Added `"name"` to `ROI_COLUMNS` (between `roi_id` and `fov`), with default `""`.
- Added `"name"` to the string-default set and the `fillna` cleanup in `_ensure_dataframe()` — backward-compatible with existing CSVs.
- Updated `format_roi_label()`: uses `name` when non-empty, falls back to `roi_id[:8]` otherwise.

### UI (`roi_manager_plugin.py`)
- Added `name_input` (`Text` widget) to the editor form (in `_build_editor_widgets()`), inserted into the `metadata_box` VBox before the tags section.
- Wires into `_populate_fields_from_record()`: populates `name_input.value` from the selected ROI record.
- Wires into `_clear_fields()`: resets `name_input.value = ""`.
- Wires into both `_capture_current_view()` (add_roi) and `_update_selected_roi()`: includes `"name": self.ui_component.name_input.value.strip()` in the updates dict.

### Export Filenames (`export_fovs.py`)
- Added `_roi_file_stem(self, record, roi_id)` private method: returns `_safe_filename(name)` when `record["name"]` is non-empty, else `_safe_filename(roi_id[:12])`.
- Both path branches in `_build_roi_items()` now call `_roi_file_stem()` instead of hard-coding `roi_id[:12]`:
  - FOV-mode: `{fov}_roi_{name}.{fmt}` (or fallback `{fov}_roi_{roi_id[:12]}.{fmt}`)
  - Map-mode: `map_{map_id}_roi_{name}.{fmt}` (or fallback)
- `separate_channels` interaction is automatic: `_build_channel_items()` derives paths via `os.path.splitext`, so `{fov}_roi_{name}__{channel}.{fmt}` is produced without extra changes.

---

## Files Changed

- `ueler/viewer/roi_manager.py` — `name` in `ROI_COLUMNS`, `_ensure_dataframe`, `format_roi_label`
- `ueler/viewer/plugin/roi_manager_plugin.py` — `name_input` widget, VBox layout, populate/clear/capture/update wiring
- `ueler/viewer/plugin/export_fovs.py` — `_roi_file_stem()` method, `_build_roi_items()` updated
- `tests/test_export_fovs_batch.py` — `TestCustomROINameFilenames` class (3 tests)
- `tests/test_roi_manager_tags.py` — `FormatROILabelTests` class (3 tests)

---

## Tests

```bash
/omics/groups/OE0622/internal/shared_envs/ark-analysis-dask_yw/bin/python -m unittest tests.test_export_fovs_batch tests.test_roi_manager_tags
```

- ✅ All 6 new tests pass
- ✅ All other existing tests pass (2 pre-existing unrelated failures excluded)

---

## Reply — Merge Same Color

### Problem

When "Export channels separately" is active, each channel is always exported individually. Users who assign the same color to multiple markers (e.g. two markers on the same dye) want them merged into a single composite image rather than split.

### Solution

Added a "Merge same color" checkbox (disabled unless `separate_channels` is checked). When both are active, channels sharing the same `ChannelRenderSettings.color` tuple are grouped and exported as one image.

### Implementation

#### Export Filenames
- Single-channel group (unique color): `{base}_{ch}.{fmt}` — same as `separate_channels` alone.
- Multi-channel group (shared color): `{base}_merged_{ch1}_{ch2}.{fmt}`.

#### `export_fovs.py`
- Added `merge_same_color` Checkbox (disabled=True) placed in an HBox next to `separate_channels` in the controls VBox.
- Observer in `_connect_events()`: toggles `merge_same_color.disabled` when `separate_channels` changes.
- `_start_export()` reads both values; `_build_job()` receives `merge_same_color: bool = False` and forwards to all three builders.
- Added `_build_grouped_channel_items()`: groups channels by `ChannelRenderSettings.color` (hashable tuple), creates per-group `_MarkerProfile`, returns `JobItem` list.
- All three builders (`_build_full_fov_items`, `_build_single_cell_items`, `_build_roi_items` — both FOV-mode and map-mode branches) dispatch to `_build_grouped_channel_items` when `merge_same_color=True`.
- Config: `_collect_export_config()` saves `merge_same_color`; `_apply_export_config()` restores it.

#### Tests (`tests/test_export_fovs_batch.py`)
- Both `_build_widgets` stubs updated with `merge_same_color = _StubWidget(False)`.
- `TestMergeSameColorBuilders` class (3 tests):
  - `test_merge_same_color_groups_channels` — 3 channels, 2 share color → 2 items (1 merged, 1 solo).
  - `test_merge_same_color_all_unique_same_as_separate` — all unique colors → 3 solo items.
  - `test_merge_same_color_false_uses_channel_items` — regression: `merge_same_color=False` → 2 per-channel items, no merged.

### Tests
```bash
/omics/groups/OE0622/internal/shared_envs/ark-analysis-dask_yw/bin/python -m unittest tests.test_export_fovs_batch tests.test_roi_manager_tags
```
- ✅ All 3 new tests pass
- ✅ All other existing tests pass (2 pre-existing unrelated failures excluded)
