"""Cell Annotation plugin package."""

from __future__ import annotations

from .manifest import Manifest
from .plugin import CellAnnotationPlugin
from .serialize import serialize_heatmap_state, validate_artifact, write_h5ad_atomic
from .selection_spec import MaterializedSelectionSpec
from .store import DatasetStore, atomic_replace, atomic_write_json

__all__ = [
    "CellAnnotationPlugin",
    "DatasetStore",
    "Manifest",
    "MaterializedSelectionSpec",
    "serialize_heatmap_state",
    "validate_artifact",
    "write_h5ad_atomic",
    "atomic_replace",
    "atomic_write_json",
]
