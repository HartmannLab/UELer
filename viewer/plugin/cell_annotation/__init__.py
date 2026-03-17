"""Compatibility wrapper for the relocated ``cell_annotation`` package."""

from __future__ import annotations

import importlib
import sys

_target = importlib.import_module("ueler.viewer.plugin.cell_annotation")
sys.modules[__name__] = _target

CellAnnotationPlugin = _target.CellAnnotationPlugin
DatasetStore = _target.DatasetStore
Manifest = _target.Manifest
MaterializedSelectionSpec = _target.MaterializedSelectionSpec
atomic_replace = _target.atomic_replace
atomic_write_json = _target.atomic_write_json

__all__ = getattr(
    _target,
    "__all__",
    [
        "CellAnnotationPlugin",
        "DatasetStore",
        "Manifest",
        "MaterializedSelectionSpec",
        "atomic_replace",
        "atomic_write_json",
    ],
)
