"""Compatibility wrapper for the relocated ``annotation_palette_editor`` module."""

from __future__ import annotations

import importlib
import sys


_target = importlib.import_module("ueler.viewer.annotation_palette_editor")
sys.modules[__name__] = _target

AnnotationPaletteEditor = _target.AnnotationPaletteEditor

__all__ = getattr(_target, "__all__", None) or ["AnnotationPaletteEditor"]
