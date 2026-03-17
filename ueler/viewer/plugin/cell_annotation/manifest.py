"""Manifest helpers for Cell Annotation checkpoint storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .store import atomic_write_json


class Manifest:
    """Thin wrapper around ``manifest.json`` within a dataset store."""

    def __init__(self, store_path: str | Path) -> None:
        self._store_path = Path(store_path)
        self._path = self._store_path / "manifest.json"
        self._data: dict[str, Any] = {}

    @property
    def path(self) -> Path:
        return self._path

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def load(self) -> dict[str, Any] | None:
        if not self._path.exists():
            return None
        with open(self._path, "r", encoding="utf-8") as handle:
            self._data = json.load(handle)
        return self._data

    def save_atomic(self) -> None:
        atomic_write_json(self._path, self._data)

    def rebuild_from_disk(self) -> dict[str, Any]:
        """Stub manifest rebuild used until checkpoint scanning lands.

        TODO: replace this with a directory walk that scans checkpoint, thumbnail,
        and selection artifacts, ignores ``*.partial`` files, and rebuilds the
        persisted DAG metadata in ``manifest.json``.
        """

        self._data = {}
        return self._data
