# OME-TIFF Loading

> Source: [`dev_note/topic_ome_tiff_loading.md`](https://github.com/HartmannLab/UELer/blob/main/dev_note/topic_ome_tiff_loading.md)

---

## Context

OME-TIFF support adds a parallel data-loading path that operates alongside the existing per-FOV TIFF folder layout.

---

## Key Decisions

- **Auto-detection.** `find_ome_tiff_files()` matches conventional `.ome.tif` / `.ome.tiff` names first, then inspects the remaining `.tif` / `.tiff` files and accepts any whose `tifffile` handle reports `is_ome`. So an OME file with a plain `.tif` name is still found, and no manual flag is required.
- **`OMEFovWrapper`.** Channel access is lazy and downsample-aware; the wrapper presents the same interface as the folder-based loader, so the compositor cannot tell the two apart.
- **Pyramid level selection with a residual.** `_select_level(ds_factor)` returns **both** the coarsest pyramid level that does not overshoot the requested factor **and** the `residual` factor still to apply. The level does the cheap part of the reduction by reading fewer bytes; the residual makes up the difference in memory. Without the residual, a request for factor 6 on a dataset with levels at 1/2/4 would have to either over-fetch or over-decimate.
- **Frame-aware access.** Stacked OME files (multiple Z-planes or time points) carry a `frame_axis`, `frame_count` and `current_frame_index`; slices are cached per frame.
- **Keyframe fallback.** Some files carry OME metadata `tifffile` rejects with an "incompatible keyframe" error. The loader retries with `is_ome=False`, falling back to plain series parsing rather than failing the FOV.

---

## Rendering

OME-TIFF images share the same compositor pipeline as standard TIFFs. Viewport alignment and downsample factor handling have specific fixes:

- Keyframe compatibility — fallbacks for metadata edge cases.
- Lazy max computation — avoids loading full-resolution data to find the channel maximum.
- Memory usage regression addressed for large pyramid levels.

---

## Usage

To open an OME-TIFF dataset, point `base_folder` (or the equivalent runner argument) at the directory containing the `.ome.tiff` file(s). The viewer detects and loads them automatically.

---

## Related Issues

- [#60](https://github.com/HartmannLab/UELer/issues/60)
- [#63](https://github.com/HartmannLab/UELer/issues/63)
