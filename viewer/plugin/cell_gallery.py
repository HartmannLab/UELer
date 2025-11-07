"""Compatibility wrapper for the relocated ``cell_gallery`` plugin."""

from __future__ import annotations

import importlib
import sys


_target = importlib.import_module("ueler.viewer.plugin.cell_gallery")
sys.modules[__name__] = _target

CellGalleryDisplay = _target.CellGalleryDisplay
UiComponent = _target.UiComponent
create_gallery = _target.create_gallery
Data = _target.Data

__all__ = getattr(_target, "__all__", None) or [
    "CellGalleryDisplay",
    "UiComponent",
    "create_gallery",
    "Data",
]
