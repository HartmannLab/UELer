# OME-TIFF Loading

> Source: [`dev_note/topic_ome_tiff_loading.md`](https://github.com/HartmannLab/UELer/blob/main/dev_note/topic_ome_tiff_loading.md)

---

## Context

OME-TIFF support adds a parallel data-loading path that operates alongside the existing per-FOV TIFF folder layout.

---

## Key Decisions

- **Auto-detection.** OME-TIFF datasets are detected automatically from `.ome.tif`, `.ome.tiff`, or suffix-less OME TIFF files. No manual flag is required.
- **`OMEFovWrapper`.** Channel access is lazy and downsample-aware; the wrapper presents the same interface as the folder-based loader.
- **Pyramid level selection.** The coarsest pyramid level that meets the requested downsample factor is used to minimize I/O without over-fetching pixels.
- **Frame-aware access.** Stacked OME files (multiple Z-planes or time points) are accessed by frame index; slices are cached per frame.

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
