"""Storage helpers for the Cell Annotation plugin."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

STORE_SUBDIRS = ("checkpoints", "thumbnails", "selections")


def _dataset_id(dataset_root: str | Path) -> str:
    """Return a short, stable identifier for *dataset_root*."""

    resolved = str(Path(dataset_root).resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


class DatasetStore:
    """Resolve and create the per-dataset Cell Annotation store tree."""

    def __init__(self, dataset_root: str | Path) -> None:
        self._root = Path(dataset_root).resolve()
        self._dataset_id = _dataset_id(self._root)
        self._store_path = self._root / ".UELer" / f"dataset_{self._dataset_id}"

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def store_path(self) -> Path:
        return self._store_path

    def ensure_dirs(self) -> None:
        self._store_path.mkdir(parents=True, exist_ok=True)
        for subdir in STORE_SUBDIRS:
            (self._store_path / subdir).mkdir(exist_ok=True)

    def subdir(self, name: str) -> Path:
        return self._store_path / name


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_replace(src_tmp: str | Path, dst_final: str | Path) -> None:
    """Atomically replace *dst_final* with the already-written temp file."""

    src = Path(src_tmp)
    dst = Path(dst_final)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with open(src, "rb") as handle:
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass

    _fsync_dir(src.parent)
    os.replace(src, dst)
    _fsync_dir(dst.parent)


def atomic_write_json(path: str | Path, obj: object) -> None:
    """Write JSON atomically so readers never observe a partial manifest."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.tmp",
        suffix=".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        atomic_replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
