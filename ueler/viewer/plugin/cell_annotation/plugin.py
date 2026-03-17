"""Cell Annotation plugin scaffolding and lifecycle hooks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ueler.viewer.interfaces import FlowsomParamsProvider, HeatmapStateProvider

from .manifest import Manifest
from .store import DatasetStore


def _flag_enabled() -> bool:
    """Return whether the Cell Annotation feature flag is enabled."""

    return os.environ.get("ENABLE_CELL_ANNOTATION", "").strip().lower() in {"1", "true", "yes"}


class CellAnnotationPlugin:
    """Own the non-UI Cell Annotation store and provider registrations."""

    REGISTRY_KEY = "cell_annotation_plugin"

    def __init__(self, viewer: object) -> None:
        self._viewer = viewer
        self._store: Optional[DatasetStore] = None
        self._manifest: Optional[Manifest] = None
        self._heatmap_provider: Optional[HeatmapStateProvider] = None
        self._flowsom_provider: Optional[FlowsomParamsProvider] = None

    @property
    def store(self) -> Optional[DatasetStore]:
        return self._store

    @property
    def manifest(self) -> Optional[Manifest]:
        return self._manifest

    @property
    def heatmap_provider(self) -> Optional[HeatmapStateProvider]:
        return self._heatmap_provider

    @property
    def flowsom_provider(self) -> Optional[FlowsomParamsProvider]:
        return self._flowsom_provider

    def on_dataset_opened(self, base_folder: str | Path) -> None:
        self._store = DatasetStore(base_folder)
        self._store.ensure_dirs()
        self._manifest = Manifest(self._store.store_path)
        self._manifest.load()

    def on_dataset_closed(self) -> None:
        self._store = None
        self._manifest = None

    def register_heatmap(self, provider: HeatmapStateProvider) -> None:
        self._heatmap_provider = provider

    def register_flowsom(self, provider: FlowsomParamsProvider) -> None:
        self._flowsom_provider = provider
