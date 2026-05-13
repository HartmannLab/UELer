# Issue #99 — Batch export plugin uses absolute path for saved settings

## Problem

`_save_export_config` in `ueler/viewer/plugin/export_fovs.py` calls `.resolve()` on the config
file path before storing it in the registry JSON:

```python
path = (folder / f"{slug}{EXPORT_CONFIG_FILE_SUFFIX}").resolve()   # absolute
registry[name] = {"path": str(path), ...}
```

Because the registry stores an absolute path, loading or deleting a saved config breaks
whenever the project is moved to a different directory or shared with another user whose
filesystem layout differs.

## Root cause

`.resolve()` expands the path to a fully qualified absolute path.  That string is then
serialised directly into `export_configs_index.json`.  `_load_export_config` and
`_delete_export_config` reconstruct the `Path` from this string verbatim, so the file
is unreachable after any folder rename or machine transfer.

## Proposed fix

Store only the **filename** (e.g. `my-config.export_config.json`) in the registry instead
of the absolute path.  At read time, reconstruct the full path by joining the stored
filename with the current `_export_config_folder`.

A small helper `_resolve_config_path(folder, stored_path)` handles both:
- New entries (filename only) — joined with `folder`.
- Old/legacy entries (absolute path) — returned unchanged for backward compatibility.

## Implementation steps

1. In `_save_export_config`: drop `.resolve()`, build the path as `folder / filename`,
   and store only `filename` in the registry.
2. Add module-level helper `_resolve_config_path`.
3. Update `_load_export_config` and `_delete_export_config` to call `_resolve_config_path`.
4. Add two new tests in `tests/test_export_fovs_mask_customization.py`:
   - `test_save_config_stores_relative_path` — asserts the registry entry is not absolute.
   - `test_load_config_survives_folder_move` — moves the temp dir, re-initialises the
     plugin, and verifies the config can still be loaded.
