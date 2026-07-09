"""Stream / cache image loading from the BioImage Archive (BIA).

Issue #110.  Adds an optional *remote* data source so UELer can explore a public
BioImage Archive study (``S-BIAD*``) without downloading the whole dataset up
front.  The design deliberately reuses the existing local loaders in
:mod:`ueler.data_loader`:

* **OME-TIFF studies** (one multi-channel file per FOV) reuse
  :class:`~ueler.data_loader.OMEFovWrapper`.  When the file exposes a usable
  pyramid it is streamed via HTTP byte-range requests (``opener=_open_remote``);
  otherwise the file is downloaded once into the local cache and opened locally.
* **Folder-per-FOV studies** (``<FOV>/<channel>.tiff``, the common MIBI layout,
  e.g. the reference dataset ``S-BIAD2557``) download each requested channel
  file on demand into a cache directory that mirrors ``<FOV>/<channel>.tiff`` and
  then read it with the unchanged :func:`~ueler.data_loader.load_one_channel_fov`.
  Masks are downloaded into a flat cache directory and read with the unchanged
  :func:`~ueler.data_loader.load_masks_for_fov`.

Structure handling is *descriptor-first*: an optional JSON descriptor maps the
study file tree onto FOVs / channels / masks.  When no descriptor is given a
best-effort auto-detection covers the two clean layouts above.

Only ``fsspec[http]``, ``requests``, ``tifffile`` (already project dependencies)
are required — no S3 / OME-Zarr stack.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .data_loader import (
    OMEFovWrapper,
    extract_ome_channel_names,
    load_annotations_for_fov,
    load_masks_for_fov,
    load_one_channel_fov,
)

logger = logging.getLogger(__name__)

BIOSTUDIES_INFO_URL = "https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}/info"
ACCESSION_RE = re.compile(r"^S-[A-Z]+\d+$", re.IGNORECASE)
TIFF_SUFFIXES = (".tiff", ".tif")
OME_SUFFIXES = (".ome.tif", ".ome.tiff")

# Non-pyramidal remote files larger than this are cached rather than streamed,
# since range reads of a single-resolution image give little benefit.
STREAM_SIZE_LIMIT = 256 * 1024 * 1024

# A non-pyramidal OME-TIFF can only be served by downloading the whole file into
# the cache. Refuse to do that above this size (overridable per data source) so a
# giant single-resolution image doesn't trigger a silent multi-GB download.
MAX_OME_CACHE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _ensure_requests():
    try:
        import requests  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "requests is required to resolve BioImage Archive studies for UELer."
        ) from exc
    return requests


def _ensure_fsspec():
    try:
        import fsspec  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "fsspec[http] is required to stream BioImage Archive images for UELer."
        ) from exc
    return fsspec


def _to_https(url: str) -> str:
    """Rewrite an ``ftp://ftp.ebi.ac.uk/...`` link to its HTTPS mirror."""
    if url.startswith("ftp://ftp.ebi.ac.uk/"):
        return "https://ftp.ebi.ac.uk/" + url[len("ftp://ftp.ebi.ac.uk/"):]
    if url.startswith("ftp://"):
        return "https://" + url[len("ftp://"):]
    return url


def _urljoin(base: str, *parts: str) -> str:
    url = base.rstrip("/")
    for part in parts:
        part = str(part).strip("/")
        if part:
            url = f"{url}/{part}"
    return url


def _open_remote(url: str):
    """Return a seekable file-like object backed by HTTP byte-range requests."""
    fsspec = _ensure_fsspec()
    return fsspec.open(url, mode="rb").open()


def _remote_size(url: str) -> Optional[int]:
    requests = _ensure_requests()
    try:
        resp = requests.head(url, allow_redirects=True, timeout=30)
        length = resp.headers.get("Content-Length")
        return int(length) if length is not None else None
    except Exception:
        return None


def _is_streamable(url: str) -> bool:
    """True when byte-range streaming is worthwhile (pyramidal, or small file)."""
    try:
        from .data_loader import _ensure_tifffile

        tifffile = _ensure_tifffile()
        with _open_remote(url) as handle:
            with tifffile.TiffFile(handle) as tif:
                series = tif.series[0]
                levels = list(getattr(series, "levels", ()) or ())
                if len(levels) > 1:
                    return True
    except Exception:
        return False
    size = _remote_size(url)
    return size is not None and size <= STREAM_SIZE_LIMIT


# ---------------------------------------------------------------------------
# Directory enumeration (Apache autoindex crawl)
# ---------------------------------------------------------------------------
_HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)


def _parse_autoindex(html: str) -> List[Tuple[str, bool]]:
    """Parse an Apache autoindex page into ``(name, is_dir)`` entries.

    Sort links (``?C=...``), parent links (absolute ``/...``) and query anchors
    are skipped.  Directories keep their trailing ``/`` in the href.
    """
    entries: List[Tuple[str, bool]] = []
    seen = set()
    for href in _HREF_RE.findall(html):
        if href.startswith("?") or href.startswith("/") or href.startswith(".."):
            continue
        if "://" in href:  # absolute link to another host
            continue
        is_dir = href.endswith("/")
        name = href.rstrip("/")
        if not name or name in seen:
            continue
        seen.add(name)
        entries.append((name, is_dir))
    return entries


class BIAStudyIndex:
    """Resolve a study base URL and lazily enumerate its directory tree."""

    def __init__(self, base_url: str, accession: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.accession = accession
        self._dir_cache: Dict[str, List[Tuple[str, bool]]] = {}

    @classmethod
    def from_source(cls, source: str) -> "BIAStudyIndex":
        """Build an index from an accession id or a direct base URL."""
        source = source.strip()
        if "://" in source:
            return cls(source)
        if ACCESSION_RE.match(source):
            base_url = cls._resolve_accession(source)
            return cls(base_url, accession=source.upper())
        raise ValueError(
            f"Unrecognised BIA source '{source}'. Provide an accession such as "
            "'S-BIAD2557' or a full https base URL."
        )

    @staticmethod
    def _resolve_accession(accession: str) -> str:
        requests = _ensure_requests()
        url = BIOSTUDIES_INFO_URL.format(accession=accession.upper())
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        info = resp.json()
        link = info.get("ftpLink") or info.get("httpLink")
        if not link:
            raise ValueError(f"BioStudies /info for {accession} exposed no ftpLink/httpLink.")
        return _to_https(link)

    def url_for(self, rel_path: str) -> str:
        return _urljoin(self.base_url, rel_path)

    def list_dir(self, rel_path: str = "") -> List[Tuple[str, bool]]:
        """Return ``(name, is_dir)`` children of *rel_path* (cached)."""
        if rel_path in self._dir_cache:
            return self._dir_cache[rel_path]
        requests = _ensure_requests()
        url = self.url_for(rel_path)
        if not url.endswith("/"):
            url += "/"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        entries = _parse_autoindex(resp.text)
        self._dir_cache[rel_path] = entries
        return entries

    def list_subdirs(self, rel_path: str = "") -> List[str]:
        return sorted(name for name, is_dir in self.list_dir(rel_path) if is_dir)

    def list_files(self, rel_path: str = "") -> List[str]:
        return sorted(name for name, is_dir in self.list_dir(rel_path) if not is_dir)


def _has_suffix(name: str, suffixes: Tuple[str, ...]) -> bool:
    lower = name.lower()
    return any(lower.endswith(s) for s in suffixes)


# ---------------------------------------------------------------------------
# Layout classification
# ---------------------------------------------------------------------------
def _normalise_sources(descriptor, list_key, dir_key, glob_key, name_key):
    """Normalise a descriptor's mask/annotation config into a list of sources.

    Each source is ``{"dir": str, "match": str, "name": Optional[str]}``.  Two
    descriptor forms are accepted:

    * A list under ``list_key`` (``masks`` / ``annotations``) of
      ``{"dir", "name"?, "match"?}`` entries — for studies with several mask
      folders and/or ``<fov>.tiff`` naming.
    * The legacy single ``dir_key`` / ``glob_key`` (+ optional ``name_key``).

    When ``name`` is set, the cached file is renamed to ``<fov>_<name>.tiff`` so
    the mask/annotation label becomes ``name``; the default ``match`` is then
    ``{fov}.*``.  When ``name`` is absent the original filename is kept and the
    default ``match`` is ``{fov}_*`` (the label is the ``<fov>_`` suffix, matching
    the local loaders).
    """

    def _default_match(name, per_fov):
        if per_fov:
            return "*.tif*"  # every raster in the per-FOV subdir
        return "{fov}.*" if name else "{fov}_*"

    raw = descriptor.get(list_key)
    sources = []
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            if not isinstance(entry, dict) or not entry.get("dir"):
                continue
            name = entry.get("name")
            name = str(name) if name else None
            per_fov = bool(entry.get("per_fov", False))
            sources.append(
                {
                    "dir": str(entry["dir"]).strip("/"),
                    "match": str(entry.get("match", _default_match(name, per_fov))),
                    "name": name,
                    "per_fov": per_fov,
                }
            )
        return sources

    directory = descriptor.get(dir_key)
    if directory:
        name = descriptor.get(name_key)
        name = str(name) if name else None
        sources.append(
            {
                "dir": str(directory).strip("/"),
                "match": str(descriptor.get(glob_key, _default_match(name, False))),
                "name": name,
                "per_fov": False,
            }
        )
    return sources


class BIALayout:
    """Normalised description of where FOVs / channels / masks live."""

    def __init__(
        self,
        mode: str,
        *,
        base: str = "",
        mask_sources: Optional[List[dict]] = None,
        annotation_sources: Optional[List[dict]] = None,
        fov_dir: str = "",
        fov_glob: str = "*.ome.tiff",
        fov_container: Optional[str] = None,
    ):
        self.mode = mode  # "folder" | "ome-tiff"
        self.base = base.strip("/")
        self.mask_sources = list(mask_sources or [])
        self.annotation_sources = list(annotation_sources or [])
        self.fov_dir = fov_dir.strip("/")
        self.fov_glob = fov_glob
        # None → each FOV is a directory of channel TIFFs; "zip" → each FOV is a
        # <FOV>.zip archive of channel TIFFs (read per-member over HTTP ranges).
        self.fov_container = fov_container


def _layout_from_descriptor(descriptor: Dict[str, object]) -> BIALayout:
    mode = str(descriptor.get("mode", "folder"))
    mask_sources = _normalise_sources(descriptor, "masks", "mask_dir", "mask_glob", "mask_name")
    annotation_sources = _normalise_sources(
        descriptor, "annotations", "annotation_dir", "annotation_glob", "annotation_name"
    )
    if mode == "ome-tiff":
        fov_glob = str(descriptor.get("fov_glob", "*.ome.tiff"))
        fov_dir = os.path.dirname(fov_glob)
        pattern = os.path.basename(fov_glob) or "*.ome.tiff"
        return BIALayout(
            "ome-tiff",
            fov_dir=fov_dir,
            fov_glob=pattern,
            mask_sources=mask_sources,
            annotation_sources=annotation_sources,
        )
    container = descriptor.get("fov_container")
    return BIALayout(
        "folder",
        base=str(descriptor.get("base", "")),
        mask_sources=mask_sources,
        annotation_sources=annotation_sources,
        fov_container=str(container) if container else None,
    )


def _auto_detect_layout(index: BIAStudyIndex, max_depth: int = 4) -> BIALayout:
    """Best-effort detection of a folder-per-FOV or OME-TIFF-per-FOV layout.

    Walks the tree breadth-first from the study root.  Returns the first
    directory whose children are subdirectories that each hold a TIFF (folder
    mode), or a directory containing ``*.ome.tif(f)`` files (OME mode).
    """
    queue: List[Tuple[str, int]] = [("", 0)]
    while queue:
        rel, depth = queue.pop(0)
        try:
            entries = index.list_dir(rel)
        except Exception:
            continue
        files = [n for n, is_dir in entries if not is_dir]
        subdirs = [n for n, is_dir in entries if is_dir]

        if any(_has_suffix(f, OME_SUFFIXES) for f in files):
            return BIALayout("ome-tiff", fov_dir=rel, fov_glob="*.ome.tif*")

        # zip-container FOVs: a directory whose files are predominantly .zip
        zips = [f for f in files if f.lower().endswith(".zip")]
        if zips and len(zips) >= max(1, len(files) // 2):
            return BIALayout("folder", base=rel, fov_container="zip")

        # folder-per-FOV: every sampled subdir holds at least one *plain* TIFF
        # (a lone .ome.tif(f) marks an OME FOV file, handled by the branch above).
        if subdirs and depth < max_depth:
            sampled = subdirs[:8]
            tiff_bearing = 0
            for sub in sampled:
                try:
                    sub_files = index.list_files(_urljoin(rel, sub) if rel else sub)
                except Exception:
                    sub_files = []
                if any(
                    _has_suffix(f, TIFF_SUFFIXES) and not _has_suffix(f, OME_SUFFIXES)
                    for f in sub_files
                ):
                    tiff_bearing += 1
            if tiff_bearing and tiff_bearing >= len(sampled):
                return BIALayout("folder", base=rel)

        if depth < max_depth:
            for sub in subdirs:
                queue.append((_urljoin(rel, sub) if rel else sub, depth + 1))

    raise ValueError(
        "Could not auto-detect the BIA dataset layout. Pass a descriptor "
        "(mode/base/mask_dir/...) describing where FOVs, channels and masks live."
    )


# ---------------------------------------------------------------------------
# Download / cache
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path) -> Path:
    """Download *url* to *dest* once (atomic); reuse an existing complete file."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    requests = _ensure_requests()
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            with requests.get(url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return dest


# ---------------------------------------------------------------------------
# Zip-container FOVs (per-member HTTP byte-range reads)
# ---------------------------------------------------------------------------
def _open_zip(zip_url: str):
    """Open a remote ZIP over an fsspec HTTP handle.

    ``zipfile.ZipFile`` reads the central directory (and later, any single member)
    via byte-range requests, so nothing beyond the requested bytes is downloaded.
    Returns ``(ZipFile, handle)``; close both when done.
    """
    import zipfile

    handle = _open_remote(zip_url)
    try:
        return zipfile.ZipFile(handle), handle
    except Exception:
        try:
            handle.close()
        except Exception:
            pass
        raise


def _zip_members(zip_url: str) -> Dict[str, str]:
    """Map ``channel_stem -> member`` for top-level ``*.tif(f)`` members of a zip."""
    zf, handle = _open_zip(zip_url)
    try:
        mapping: Dict[str, str] = {}
        for member in zf.namelist():
            if member.endswith("/") or "/" in member.strip("/"):
                continue  # only top-level files
            if _has_suffix(member, TIFF_SUFFIXES):
                mapping[os.path.splitext(os.path.basename(member))[0]] = member
        return mapping
    finally:
        zf.close()
        try:
            handle.close()
        except Exception:
            pass


def _extract_member(zip_url: str, member: str, dest: Path) -> Path:
    """Range-read one *member* out of the remote zip and write it to *dest* once."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    zf, handle = _open_zip(zip_url)
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    tmp = Path(tmp_name)
    try:
        with zf.open(member) as src, os.fdopen(fd, "wb") as out:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
    finally:
        zf.close()
        try:
            handle.close()
        except Exception:
            pass
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return dest


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
class BIADataSource:
    """Remote data source handed to :class:`ImageMaskViewer` via ``data_source=``.

    Exposes the small surface the viewer needs in ``_fov_mode == "bia"``:
    ``fov_mode``, ``list_fovs``, ``open_fov``, ``load_channel`` and the mask /
    annotation prefetch helpers (which populate flat local dirs so the existing
    ``load_masks_for_fov`` / ``load_annotations_for_fov`` run unchanged).
    """

    def __init__(
        self,
        source: str,
        *,
        cache_dir: str,
        descriptor: Optional[Dict[str, object]] = None,
        max_download_bytes: Optional[int] = None,
    ):
        self.index = BIAStudyIndex.from_source(source)
        self.cache_dir = Path(cache_dir)
        self.layout = (
            _layout_from_descriptor(descriptor)
            if descriptor
            else _auto_detect_layout(self.index)
        )
        self.fov_mode = self.layout.mode
        # Guard against silently downloading a huge non-pyramidal OME-TIFF via the
        # cache fallback. Pyramidal files stream regardless of size and bypass this.
        self.max_download_bytes = (
            MAX_OME_CACHE_BYTES if max_download_bytes is None else int(max_download_bytes)
        )

        self._channel_root = self.cache_dir / "channels"   # mirrors <fov>/<channel>.tiff
        self._ome_root = self.cache_dir / "ome"
        self._masks_dir = self.cache_dir / "masks_flat"
        self._annotations_dir = self.cache_dir / "annotations_flat"

        self._fov_names: Optional[List[str]] = None
        self._channels_cache: Dict[str, Dict[str, str]] = {}
        self._dir_files: Dict[str, List[str]] = {}  # remote dir -> tiff file names
        self._zip_members_cache: Dict[str, Dict[str, str]] = {}  # fov -> {channel: member}

    @property
    def _is_zip(self) -> bool:
        return self.layout.mode == "folder" and self.layout.fov_container == "zip"

    def _zip_url(self, fov: str) -> str:
        rel = _urljoin(self.layout.base, f"{fov}.zip") if self.layout.base else f"{fov}.zip"
        return self.index.url_for(rel)

    def _zip_channel_map(self, fov: str) -> Dict[str, str]:
        if fov not in self._zip_members_cache:
            self._zip_members_cache[fov] = _zip_members(self._zip_url(fov))
        return self._zip_members_cache[fov]

    # -- discovery ----------------------------------------------------------
    def list_fovs(self) -> List[str]:
        if self._fov_names is not None:
            return self._fov_names
        if self.layout.mode == "ome-tiff":
            files = self.index.list_files(self.layout.fov_dir)
            matches = [f for f in files if fnmatch.fnmatch(f.lower(), self.layout.fov_glob.lower())]
            names = []
            for name in matches:
                lower = name.lower()
                if lower.endswith(".ome.tif"):
                    names.append(name[:-8])
                elif lower.endswith(".ome.tiff"):
                    names.append(name[:-9])
                else:
                    names.append(os.path.splitext(name)[0])
            self._fov_names = sorted(names)
        elif self._is_zip:
            self._fov_names = sorted(
                f[:-4] for f in self.index.list_files(self.layout.base)
                if f.lower().endswith(".zip")
            )
        else:
            self._fov_names = sorted(self.index.list_subdirs(self.layout.base))
        if not self._fov_names:
            raise ValueError("No FOVs found in the BIA study for the resolved layout.")
        return self._fov_names

    def _channels_for(self, fov: str) -> Dict[str, str]:
        if fov in self._channels_cache:
            return self._channels_cache[fov]
        rel = _urljoin(self.layout.base, fov) if self.layout.base else fov
        mapping: Dict[str, str] = {}
        for fname in self.index.list_files(rel):
            if _has_suffix(fname, TIFF_SUFFIXES):
                stem = os.path.splitext(fname)[0]
                mapping[stem] = self.index.url_for(_urljoin(rel, fname))
        self._channels_cache[fov] = mapping
        return mapping

    # -- image loading ------------------------------------------------------
    def open_fov(self, fov: str, ds_factor: int):
        """Return a per-FOV image object.

        OME mode → an :class:`OMEFovWrapper` (streamed if pyramidal, else
        cached).  Folder mode → a ``{channel_name: None}`` dict whose entries the
        viewer fills lazily via :meth:`load_channel`.
        """
        if self.layout.mode == "ome-tiff":
            url = self._ome_url(fov)
            if _is_streamable(url):
                logger.info("[BIA] streaming OME-TIFF FOV %s", fov)
                return OMEFovWrapper(url, ds_factor=ds_factor, opener=_open_remote)
            # Non-pyramidal: the only way to serve it is to cache the whole file.
            size = _remote_size(url)
            if size is not None and size > self.max_download_bytes:
                raise ValueError(
                    f"[BIA] FOV '{fov}' is a {size / 1e9:.1f} GB non-pyramidal OME-TIFF. "
                    "UELer streams pyramidal images only; serving this would download the "
                    "entire file. Use a pyramidal / OME-Zarr copy, download it locally and "
                    "open it with run_viewer(...), or raise the ceiling via "
                    "run_viewer_bia(..., max_download_bytes=<bytes>)."
                )
            if size is None:
                logger.warning(
                    "[BIA] FOV '%s' size is unknown; caching may download a large file.", fov
                )
            logger.info("[BIA] caching OME-TIFF FOV %s (non-pyramidal)", fov)
            local = _download(url, self._ome_root / f"{fov}{self._ome_suffix(url)}")
            return OMEFovWrapper(str(local), ds_factor=ds_factor)
        if self._is_zip:
            return {name: None for name in self._zip_channel_map(fov)}
        return {name: None for name in self._channels_for(fov)}

    def load_channel(self, fov: str, channel: str, channel_max_values, compute_stats: bool = True):
        """Fetch one folder-mode channel (from a plain file or a zip member) and
        read it via the unchanged local loader."""
        if self._is_zip:
            member = self._zip_channel_map(fov).get(channel)
            if member is None:
                logger.warning("[BIA] channel '%s' not found in zip for FOV '%s'", channel, fov)
                return None
            ext = os.path.splitext(member)[1] or ".tiff"
            _extract_member(self._zip_url(fov), member, self._channel_root / fov / f"{channel}{ext}")
        else:
            channels = self._channels_for(fov)
            url = channels.get(channel)
            if url is None:
                logger.warning("[BIA] channel '%s' not found for FOV '%s'", channel, fov)
                return None
            suffix = os.path.splitext(url)[1] or ".tiff"
            _download(url, self._channel_root / fov / f"{channel}{suffix}")
        return load_one_channel_fov(
            fov,
            str(self._channel_root),
            channel_max_values,
            {channel},
            compute_stats=compute_stats,
        )

    def _ome_url(self, fov: str) -> str:
        files = self.index.list_files(self.layout.fov_dir)
        for cand in (f"{fov}.ome.tiff", f"{fov}.ome.tif", f"{fov}.tiff", f"{fov}.tif"):
            for fname in files:
                if fname.lower() == cand.lower():
                    rel = _urljoin(self.layout.fov_dir, fname) if self.layout.fov_dir else fname
                    return self.index.url_for(rel)
        raise ValueError(f"No OME-TIFF file found for FOV '{fov}'.")

    @staticmethod
    def _ome_suffix(url: str) -> str:
        lower = url.lower()
        if lower.endswith(".ome.tiff"):
            return ".ome.tiff"
        if lower.endswith(".ome.tif"):
            return ".ome.tif"
        return os.path.splitext(url)[1] or ".tiff"

    # -- masks / annotations ------------------------------------------------
    @property
    def has_masks(self) -> bool:
        return any(self._source_has_content(s) for s in self.layout.mask_sources)

    @property
    def has_annotations(self) -> bool:
        return any(self._source_has_content(s) for s in self.layout.annotation_sources)

    @property
    def masks_folder(self) -> str:
        return str(self._masks_dir)

    @property
    def annotations_folder(self) -> str:
        return str(self._annotations_dir)

    def _source_files(self, source: dict) -> List[str]:
        """TIFF file names in a flat mask/annotation source directory (cached)."""
        directory = source["dir"]
        if directory not in self._dir_files:
            try:
                self._dir_files[directory] = [
                    f for f in self.index.list_files(directory)
                    if _has_suffix(f, TIFF_SUFFIXES)
                ]
            except Exception:
                self._dir_files[directory] = []
        return self._dir_files[directory]

    def _source_has_content(self, source: dict) -> bool:
        if source.get("per_fov"):
            try:
                return bool(self.index.list_subdirs(source["dir"]))
            except Exception:
                return False
        return bool(self._source_files(source))

    def _source_fov_files(self, source: dict, fov: str) -> Tuple[str, List[str]]:
        """Return ``(rel_dir, tiff_filenames)`` for this source and FOV.

        Flat sources read from ``<dir>/``; per-FOV sources from ``<dir>/<fov>/``.
        """
        if source.get("per_fov"):
            rel_dir = _urljoin(source["dir"], fov)
            try:
                files = [f for f in self.index.list_files(rel_dir) if _has_suffix(f, TIFF_SUFFIXES)]
            except Exception:
                files = []
            return rel_dir, files
        return source["dir"], self._source_files(source)

    def prefetch_masks(self, fov: str) -> None:
        self._prefetch(fov, self.layout.mask_sources, self._masks_dir)

    def prefetch_annotations(self, fov: str) -> None:
        self._prefetch(fov, self.layout.annotation_sources, self._annotations_dir)

    def _prefetch(self, fov: str, sources: List[dict], dest_dir: Path) -> None:
        """Download the FOV's raster files into a flat local dir, renaming to
        ``<fov>_<label>.tiff`` so the local loaders derive a clean mask label.

        * per-FOV source (``<dir>/<fov>/*.tiff``): label = file stem.
        * named flat source: label = the source ``name``.
        * unnamed flat source: original filename (label = its ``<fov>_`` suffix).
        """
        for source in sources:
            pattern = source["match"].format(fov=fov)
            rel_dir, files = self._source_fov_files(source, fov)
            per_fov = bool(source.get("per_fov"))
            for fname in files:
                if not fnmatch.fnmatch(fname, pattern):
                    continue
                ext = os.path.splitext(fname)[1] or ".tiff"
                if per_fov:
                    stem = os.path.splitext(fname)[0]
                    label = f"{source['name']}_{stem}" if source["name"] else stem
                    local_name = f"{fov}_{label}{ext}"
                elif source["name"]:
                    local_name = f"{fov}_{source['name']}{ext}"
                else:
                    local_name = fname
                rel = _urljoin(rel_dir, fname)
                _download(self.index.url_for(rel), Path(dest_dir) / local_name)
