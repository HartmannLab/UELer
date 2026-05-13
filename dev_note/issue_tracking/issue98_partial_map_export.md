# Issue #98 — Batch Export Still Sometimes Writes Partial Images

## Problem

Batch export in map mode occasionally produces images with black patches — regions where tiles should appear but are blank. The bug is a follow-up to issue #94, which added a `warnings.warn()` when tile loading failed but still allowed the partially-rendered canvas to be written to disk.

The net result: exports could succeed (from the job runner's perspective) while producing visually incomplete images, with no clear error reported to the user.

---

## Root Cause

**File:** `ueler/viewer/plugin/export_fovs.py` — `_render_map_region_direct()`, post-#94 state:

```python
viewer.load_fov(tile.name, channels)
fov_arrays = viewer.image_cache.get(tile.name)
if not fov_arrays:
    warnings.warn(
        f"Map export: tile '{tile.name}' could not be loaded; it will appear blank in the output.",
        stacklevel=2,
    )
    continue   # ← partial canvas still returned and written to disk
```

The shared `image_cache` can be evicted concurrently by the live viewer UI while export is running (the viewer also loads/evicts FOVs as the user interacts). In some runs, the cache entry is present immediately; in others it has already been evicted by the time `image_cache.get()` is called — producing the "sometimes" pattern.

---

## Fix

**File:** `ueler/viewer/plugin/export_fovs.py`

Two changes to `_render_map_region_direct()`:

1. **Retry once** (50 ms sleep then reload) — handles the common case of a transient eviction.
2. **Raise `RuntimeError`** after the retry if the tile is still missing — prevents a partial canvas from being written to disk.

```python
viewer.load_fov(tile.name, channels)
fov_arrays = viewer.image_cache.get(tile.name)
if not fov_arrays:
    import time
    time.sleep(0.05)
    viewer.load_fov(tile.name, channels)
    fov_arrays = viewer.image_cache.get(tile.name)
if not fov_arrays:
    raise RuntimeError(
        f"Map export: tile '{tile.name}' could not be loaded after retry. "
        "Export aborted to prevent writing a partial image."
    )
```

The `RuntimeError` propagates through `_export_map_roi_worker()` to the job runner (`ueler/export/job.py` line 178), which catches it and records `ExportResult(ok=False, error=str(exc))` — surfacing the message to the user without writing a partial file.

---

## Error Propagation Path

```
_render_map_region_direct()  →  RuntimeError
    ↑ called by
_export_map_roi_worker()      →  propagates (no broad except)
    ↑ called by
job.py ExportJob._run_loop()  →  catches Exception, sets ok=False, error=str(exc)
    ↑ results visible in
BatchExportPlugin UI           →  shows failed items with error message
```

---

## Tests Updated / Added

**File:** `tests/test_export_fovs_batch.py` — in `BatchExportMapROIItemsTests`:

| Test | Change | Assertion |
|------|--------|-----------|
| `test_render_map_region_direct_raises_when_tile_load_fails` | Updated (was `warns_when_tile_load_fails`) | `RuntimeError` raised naming the tile and "partial image" |
| `test_render_map_region_direct_succeeds_on_retry` | New | `load_fov` called twice; export completes when cache is populated on second call |
