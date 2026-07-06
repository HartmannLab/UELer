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
class BIALayout:
    """Normalised description of where FOVs / channels / masks live."""

    def __init__(
        self,
        mode: str,
        *,
        base: str = "",
        mask_dir: Optional[str] = None,
        mask_glob: str = "{fov}_*",
        annotation_dir: Optional[str] = None,
        annotation_glob: str = "{fov}_*",
        fov_dir: str = "",
        fov_glob: str = "*.ome.tiff",
    ):
        self.mode = mode  # "folder" | "ome-tiff"
        self.base = base.strip("/")
        self.mask_dir = mask_dir.strip("/") if mask_dir else None
        self.mask_glob = mask_glob
        self.annotation_dir = annotation_dir.strip("/") if annotation_dir else None
        self.annotation_glob = annotation_glob
        self.fov_dir = fov_dir.strip("/")
        self.fov_glob = fov_glob


def _layout_from_descriptor(descriptor: Dict[str, object]) -> BIALayout:
    mode = str(descriptor.get("mode", "folder"))
    if mode == "ome-tiff":
        fov_glob = str(descriptor.get("fov_glob", "*.ome.tiff"))
        fov_dir = os.path.dirname(fov_glob)
        pattern = os.path.basename(fov_glob) or "*.ome.tiff"
        return BIALayout(
            "ome-tiff",
            fov_dir=fov_dir,
            fov_glob=pattern,
            mask_dir=descriptor.get("mask_dir"),
            mask_glob=str(descriptor.get("mask_glob", "{fov}_*")),
            annotation_dir=descriptor.get("annotation_dir"),
            annotation_glob=str(descriptor.get("annotation_glob", "{fov}_*")),
        )
    return BIALayout(
        "folder",
        base=str(descriptor.get("base", "")),
        mask_dir=descriptor.get("mask_dir"),
        mask_glob=str(descriptor.get("mask_glob", "{fov}_*")),
        annotation_dir=descriptor.get("annotation_dir"),
        annotation_glob=str(descriptor.get("annotation_glob", "{fov}_*")),
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
    ):
        self.index = BIAStudyIndex.from_source(source)
        self.cache_dir = Path(cache_dir)
        self.layout = (
            _layout_from_descriptor(descriptor)
            if descriptor
            else _auto_detect_layout(self.index)
        )
        self.fov_mode = self.layout.mode

        self._channel_root = self.cache_dir / "channels"   # mirrors <fov>/<channel>.tiff
        self._ome_root = self.cache_dir / "ome"
        self._masks_dir = self.cache_dir / "masks_flat"
        self._annotations_dir = self.cache_dir / "annotations_flat"

        self._fov_names: Optional[List[str]] = None
        self._channels_cache: Dict[str, Dict[str, str]] = {}
        self._mask_index: Optional[List[str]] = None
        self._annotation_index: Optional[List[str]] = None

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
            logger.info("[BIA] caching OME-TIFF FOV %s (non-pyramidal)", fov)
            local = _download(url, self._ome_root / f"{fov}{self._ome_suffix(url)}")
            return OMEFovWrapper(str(local), ds_factor=ds_factor)
        return {name: None for name in self._channels_for(fov)}

    def load_channel(self, fov: str, channel: str, channel_max_values, compute_stats: bool = True):
        """Cache one folder-mode channel file and read it via the local loader."""
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
        return self.layout.mask_dir is not None and bool(self._mask_files())

    @property
    def has_annotations(self) -> bool:
        return self.layout.annotation_dir is not None and bool(self._annotation_files())

    @property
    def masks_folder(self) -> str:
        return str(self._masks_dir)

    @property
    def annotations_folder(self) -> str:
        return str(self._annotations_dir)

    def _mask_files(self) -> List[str]:
        if self._mask_index is None:
            if self.layout.mask_dir is None:
                self._mask_index = []
            else:
                try:
                    self._mask_index = [
                        f for f in self.index.list_files(self.layout.mask_dir)
                        if _has_suffix(f, TIFF_SUFFIXES)
                    ]
                except Exception:
                    self._mask_index = []
        return self._mask_index

    def _annotation_files(self) -> List[str]:
        if self._annotation_index is None:
            if self.layout.annotation_dir is None:
                self._annotation_index = []
            else:
                try:
                    self._annotation_index = [
                        f for f in self.index.list_files(self.layout.annotation_dir)
                        if _has_suffix(f, TIFF_SUFFIXES)
                    ]
                except Exception:
                    self._annotation_index = []
        return self._annotation_index

    def prefetch_masks(self, fov: str) -> None:
        self._prefetch(fov, self.layout.mask_dir, self.layout.mask_glob,
                       self._mask_files(), self._masks_dir)

    def prefetch_annotations(self, fov: str) -> None:
        self._prefetch(fov, self.layout.annotation_dir, self.layout.annotation_glob,
                       self._annotation_files(), self._annotations_dir)

    def _prefetch(self, fov, remote_dir, glob_tmpl, files, dest_dir):
        if remote_dir is None:
            return
        pattern = glob_tmpl.format(fov=fov)
        for fname in files:
            if fnmatch.fnmatch(fname, pattern):
                rel = _urljoin(remote_dir, fname)
                _download(self.index.url_for(rel), Path(dest_dir) / fname)
