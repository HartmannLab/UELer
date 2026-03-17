"""Manifest helpers for the Cell Annotation plugin.

The manifest lives at ``<store>/manifest.json`` and acts as the index of
all checkpoints, thumbnails and selections persisted for a dataset.

This module is intentionally a thin stub: the full schema and merge logic
are tracked in sub-issue #18 (AnnData Serializer) and #19 (Manifest &
Thumbnails).  Only the file-I/O primitives are implemented here so that
the rest of the plugin has a stable surface to call.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .store import atomic_write_json

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "manifest.json"


class Manifest:
    """Thin wrapper around ``manifest.json`` for a dataset store.

    Parameters
    ----------
    store_path:
        The ``DatasetStore.store_path`` directory that contains (or will
        contain) the manifest file.
    """

    def __init__(self, store_path: str | Path) -> None:
        self._store_path = Path(store_path)
        self._manifest_path = self._store_path / _MANIFEST_FILENAME
        self._data: Dict[str, Any] = {}

    @property
    def path(self) -> Path:
        """Absolute path to the manifest file."""
        return self._manifest_path

    @property
    def data(self) -> Dict[str, Any]:
        """In-memory representation of the manifest (may be empty)."""
        return self._data

    def load(self) -> Optional[Dict[str, Any]]:
        """Load the manifest from disk.

        Returns the parsed dictionary on success, or ``None`` if the file
        does not exist yet.  Raises ``json.JSONDecodeError`` on corrupt files
        so callers can decide how to recover.
        """
        if not self._manifest_path.exists():
            logger.debug("Manifest not found at %s; starting empty", self._manifest_path)
            return None

        with open(self._manifest_path, "r", encoding="utf-8") as fh:
            self._data = json.load(fh)

        logger.debug("Manifest loaded from %s (%d keys)", self._manifest_path, len(self._data))
        return self._data

    def save_atomic(self) -> None:
        """Persist the current in-memory manifest to disk atomically."""
        atomic_write_json(self._manifest_path, self._data)
        logger.debug("Manifest saved to %s", self._manifest_path)

    def rebuild_from_disk(self) -> Dict[str, Any]:
        """Rebuild the manifest by scanning the store directory.

        This is a stub implementation.  A full scan that walks
        ``checkpoints/``, ``thumbnails/`` and ``selections/`` and
        reconciles their contents with the manifest is tracked in
        sub-issue #19 (Manifest & Thumbnails).

        Returns
        -------
        Dict[str, Any]
            The (currently empty) rebuilt manifest dictionary.
        """
        logger.debug(
            "rebuild_from_disk called for %s (stub – no scan performed)",
            self._store_path,
        )
        self._data = {}
        return self._data
