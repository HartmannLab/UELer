# Issue #110 — Stream-load images from the BioImage Archive (BIA)

## Problem

UELer only read datasets from a local directory. `run_viewer(base_folder, ...)` validated
`base_folder` as an existing local dir and `ImageMaskViewer.__init__` discovered FOVs with
`os.listdir`/`glob`; all reads used `dask_image.imread`/`tifffile` on local paths. Exploring a
public BioImage Archive study therefore required downloading the whole dataset first.

Issue #110 asks to explore a BIA study (an `S-BIAD*` accession) **without** a full download.
The hard part is that BIA studies have **no standard folder layout**.

## Decisions (agreed with the maintainer)

- **Descriptor-first structure handling.** An optional JSON descriptor maps the study file
  tree onto FOVs / channels / masks. Auto-detection is a convenience limited to the two clean
  layouts that mirror UELer's local modes (folder-per-FOV, OME-TIFF-per-FOV).
- **Stream + download/cache fallback.** Byte-range streaming via `fsspec` is only fast for
  pyramidal / cloud-optimised files. Pyramidal OME-TIFFs are streamed; everything else is
  downloaded once into a local cache and read with the existing local loaders.
- **OME-Zarr (NGFF) deferred to a later phase.**
- **Entry point accepts an accession id or a direct base URL.**

## Reference dataset — `S-BIAD2557` (lab's own MIBI study)

Verified structure (folder-per-FOV, matches UELer's local folder mode directly):

```
Files/spatial_murine_iCCAvsHCC/
  image_data/<FOV>/<channel>.tiff                   # per-channel MIBI TIFFs (~8 MB, single-res)
  segmentation/cleaned_mask/<FOV>_cleaned_mask.tiff # <fov>_<name>.tiff — matches load_masks_for_fov
  tumor_border/<FOV>.csv, extracellular_region/, SMA_positive_pixels/   # region CSVs (out of scope)
```

Confirmed facts that shaped the implementation:
- `/info` returns `ftpLink = ftp://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/557/S-BIAD2557`
  (the `pub/databases` path — differs from other studies' `fire` path, so the base **must** be
  resolved from the API, never hardcoded).
- The HTTPS mirror serves an Apache autoindex and honours `Accept-Ranges: bytes`.
- The channel TIFFs are single-resolution → S-BIAD2557 uses the download-and-cache path; files
  are small and fetched per opened channel, so caching is efficient.

## Implementation

New module **`ueler/bia_loader.py`**:
- `_open_remote` (fsspec HTTP handle), `_is_streamable` (pyramid / size heuristic),
  `_download` (atomic download-once cache).
- `BIAStudyIndex` — resolves an accession via the BioStudies `/info` endpoint (or takes a base
  URL) and enumerates files by crawling the Apache autoindex (`_parse_autoindex`), cached.
- `_layout_from_descriptor` / `_auto_detect_layout` → `BIALayout` (`folder` | `ome-tiff`).
- `BIADataSource` — the object handed to the viewer: `list_fovs`, `open_fov`, `load_channel`,
  `prefetch_masks/annotations`, `masks_folder`/`annotations_folder`, `has_masks/has_annotations`.

Reuse of existing loaders (the practicality win):
- Folder channels → `_download` then unchanged `load_one_channel_fov` (squeeze/rechunk/stats)
  pointed at a `<fov>/<channel>.tiff` cache root.
- Masks/annotations → `prefetch_*` downloads matching files into a flat local dir; the viewer
  then runs the unchanged `load_masks_for_fov` / `load_annotations_for_fov`.
- OME FOVs → `OMEFovWrapper`, streamed (new `opener=` parameter feeds a remote handle to
  `tifffile.TiffFile`) when pyramidal, otherwise downloaded and opened locally.

`ueler/data_loader.py`: `OMEFovWrapper.__init__` gains `opener=None` (+ remote-handle close in
`close()`); `extract_ome_channel_names` gains `opener=None`.

`ueler/viewer/main_viewer.py`: `ImageMaskViewer.__init__` gains `data_source=None`. When set:
`_fov_mode = "bia"`; FOV discovery, `load_fov` image population + channel fill, and mask/
annotation prefetch route through the data source. `base_folder` is a local workspace dir so
every existing `.UELer` write (ROIs, checkpoints, widget states, maps, palettes) is unchanged.

`ueler/runner.py`: new `run_viewer_bia(source, *, descriptor=None, local_dir=None, ...)` +
shared `_finalise_viewer` tail; workspace default `~/.ueler/bia/<accession>/` with `cache/`
subdir. Exported from `ueler/__init__.py`. Deps declared in `pyproject.toml`
(`fsspec[http]`, `requests`, `tifffile`).

### Local workspace layout

```
~/.ueler/bia/<accession>/
  .UELer/            # persistent user work (ROIs, checkpoints, widget states, maps, palettes)
  cache/
    channels/<FOV>/<channel>.tiff   # downloaded folder-mode channels
    ome/<FOV>.ome.tiff              # downloaded non-pyramidal OME FOVs
    masks_flat/<FOV>_<name>.tiff    # downloaded masks
```
`cache/` is disposable (re-downloaded on demand); `.UELer/` holds the user's work.

## Tests

`tests/test_issue110_bia_loader.py` (22 tests, network mocked): URL rewrite + autoindex
parsing; accession→HTTPS resolution; descriptor + auto-detect classification (folder / OME);
folder-mode `list_fovs` / `_channels_for` / `load_channel` (reuses `load_one_channel_fov`) /
mask prefetch; OME stream-vs-cache selection; `run_viewer_bia` wiring (workspace created,
`data_source` passed, `_normalise_directory` never called, JSON-file descriptor).

Live smoke test against real `S-BIAD2557`: base URL resolved, **562 FOVs**, 30 channels for
FOV0, masks detected, one channel (8.4 MB) downloaded to cache, mask prefetched — all OK.

Full suite: no new failures/errors vs. baseline (20 failures / 40 errors pre-existing, identical
before and after).

## Known limitations / future work

- Streaming is exercised only by pyramidal OME-TIFF studies; folder-mode + non-pyramidal files
  use the download-and-cache path.
- OME-Zarr (NGFF) not yet supported (phase 2).
- Region CSVs (`tumor_border/` etc. in S-BIAD2557) are not imported.
- Auto-detection finds only the FOV/channel structure; masks generally need a descriptor.
