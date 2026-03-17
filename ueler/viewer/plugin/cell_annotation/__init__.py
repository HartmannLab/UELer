"""Cell Annotation plugin package.

Provides checkpoint orchestration and `.UELer` persistence for the
Cell Annotation workflow.  Registration is gated by the
``ENABLE_CELL_ANNOTATION`` environment variable.
"""

from __future__ import annotations

from .plugin import CellAnnotationPlugin
from .store import DatasetStore, atomic_replace, atomic_write_json
from .manifest import Manifest

__all__ = [
    "CellAnnotationPlugin",
    "DatasetStore",
    "atomic_replace",
    "atomic_write_json",
    "Manifest",
]
