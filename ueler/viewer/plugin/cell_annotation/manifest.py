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
        """Rebuild ``manifest.json`` from checkpoint-sidecar artifacts on disk."""

        checkpoints_dir = self._store_path / "checkpoints"
        thumbnails = self._artifact_map(self._store_path / "thumbnails")
        selections = self._artifact_map(self._store_path / "selections")
        checkpoints: list[dict[str, Any]] = []

        if checkpoints_dir.exists():
            for metadata_path in sorted(checkpoints_dir.glob("*.json")):
                if self._is_partial(metadata_path):
                    continue
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if not isinstance(payload, dict):
                    continue

                checkpoint = dict(payload)
                checkpoint_id = str(checkpoint.get("id") or metadata_path.stem)
                artifacts = checkpoint.setdefault("artifacts", {})
                if not isinstance(artifacts, dict):
                    artifacts = {}
                    checkpoint["artifacts"] = artifacts
                checkpoint["id"] = checkpoint_id
                artifacts.setdefault("checkpoint", self._relative_path(metadata_path))

                thumbnail_path = thumbnails.get(checkpoint_id)
                if thumbnail_path is not None:
                    artifacts.setdefault("thumbnail", self._relative_path(thumbnail_path))

                selection_path = selections.get(checkpoint_id)
                if selection_path is not None:
                    artifacts.setdefault("selection", self._relative_path(selection_path))

                checkpoints.append(checkpoint)

        self._data = {"checkpoints": checkpoints}
        self.save_atomic()
        return self._data

    def _artifact_map(self, directory: Path) -> dict[str, Path]:
        artifacts: dict[str, Path] = {}
        if not directory.exists():
            return artifacts
        for path in sorted(directory.iterdir()):
            if not path.is_file() or self._is_partial(path):
                continue
            artifacts.setdefault(path.stem, path)
        return artifacts

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self._store_path).as_posix()

    @staticmethod
    def _is_partial(path: Path) -> bool:
        return path.name.endswith(".partial") or ".partial" in path.suffixes
