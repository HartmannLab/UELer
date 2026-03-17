"""Cell Annotation plugin package."""

from __future__ import annotations

from .manifest import Manifest
from .plugin import CellAnnotationPlugin
from .selection_spec import MaterializedSelectionSpec
from .store import DatasetStore, atomic_replace, atomic_write_json

__all__ = [
    "CellAnnotationPlugin",
    "DatasetStore",
    "Manifest",
    "MaterializedSelectionSpec",
    "atomic_replace",
    "atomic_write_json",
]
