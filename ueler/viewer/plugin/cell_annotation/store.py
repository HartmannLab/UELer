"""Storage helpers for the Cell Annotation plugin.

Provides:
- Dataset ID hashing (stable hash of the dataset root path)
- ``.UELer/dataset_<id>/`` path resolution
- Atomic write helpers (temp → fsync → ``os.replace``)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Sub-folder names created under each dataset store.
STORE_SUBDIRS = ("checkpoints", "thumbnails", "selections")


def _dataset_id(dataset_root: str | Path) -> str:
    """Return a short, stable hex identifier for *dataset_root*.

    The ID is the first 16 hex characters of the SHA-256 digest of the
    resolved absolute path, giving a 64-bit collision space that is
    sufficient for local file-system use.
    """
    resolved = str(Path(dataset_root).resolve())
    digest = hashlib.sha256(resolved.encode()).hexdigest()
    return digest[:16]


class DatasetStore:
    """Manages the per-dataset folder tree under ``<root>/.UELer/dataset_<id>/``.

    Parameters
    ----------
    dataset_root:
        Absolute path to the dataset directory (the same ``base_folder``
        that ``ImageMaskViewer`` receives).
    """

    def __init__(self, dataset_root: str | Path) -> None:
        self._root = Path(dataset_root).resolve()
        self._id = _dataset_id(self._root)
        self._store_path = self._root / ".UELer" / f"dataset_{self._id}"

    @property
    def store_path(self) -> Path:
        """Absolute path to the dataset-specific store directory."""
        return self._store_path

    @property
    def dataset_id(self) -> str:
        """Short hex identifier derived from the dataset root path."""
        return self._id

    def ensure_dirs(self) -> None:
        """Create the store directory tree if it does not already exist.

        Creates::

            <root>/.UELer/dataset_<id>/
            <root>/.UELer/dataset_<id>/checkpoints/
            <root>/.UELer/dataset_<id>/thumbnails/
            <root>/.UELer/dataset_<id>/selections/
        """
        self._store_path.mkdir(parents=True, exist_ok=True)
        for sub in STORE_SUBDIRS:
            (self._store_path / sub).mkdir(exist_ok=True)
        logger.debug("DatasetStore ready at %s", self._store_path)

    def subdir(self, name: str) -> Path:
        """Return the path for a named sub-directory within the store."""
        return self._store_path / name


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------

def atomic_replace(src_tmp: str | Path, dst_final: str | Path) -> None:
    """Atomically replace *dst_final* with *src_tmp*.

    The source file is fsync'd and then its parent directory is fsync'd
    (where the OS supports it) before the rename so that the data is
    durable on disk before the name becomes visible to readers.

    Parameters
    ----------
    src_tmp:
        Path to the temporary file that has already been written.
    dst_final:
        Destination path.  Any existing file at this location is replaced
        atomically (on POSIX) or with best-effort on Windows.
    """
    src = Path(src_tmp)
    dst = Path(dst_final)

    # Ensure the destination directory exists.
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Flush and fsync the file data to storage.
    with open(src, "rb") as fh:
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass  # best-effort on platforms that do not support fsync

    # Fsync the parent directory so the directory entry is durable.
    _fsync_dir(src.parent)

    os.replace(src, dst)

    # Fsync the destination directory after the rename.
    _fsync_dir(dst.parent)


def _fsync_dir(directory: Path) -> None:
    """Fsync a directory file descriptor (no-op on Windows)."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # Windows does not allow fsync on directories; silently skip


def atomic_write_json(path: str | Path, obj: object) -> None:
    """Serialise *obj* to JSON and write it atomically to *path*.

    The JSON is first written to a sibling temporary file and then renamed
    over *path* so that readers never observe a partial file.

    Parameters
    ----------
    path:
        Target file path.
    obj:
        Any object that ``json.dumps`` can serialise.
    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file in the same directory to guarantee that
    # ``os.replace`` is an atomic rename (same filesystem).
    fd, tmp_path = tempfile.mkstemp(
        dir=str(dst.parent),
        prefix=f".{dst.name}.tmp",
        suffix=".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        atomic_replace(tmp_path, dst)
    except Exception:
        # Clean up the temp file on failure so no partials are left on disk.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
