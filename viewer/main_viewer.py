"""Compatibility wrapper for the relocated ``main_viewer`` module."""

from __future__ import annotations

import importlib
import sys


_target = importlib.import_module("ueler.viewer.main_viewer")
sys.modules[__name__] = _target

ImageMaskViewer = _target.ImageMaskViewer
create_widgets = getattr(_target, "create_widgets", None)
display_ui = getattr(_target, "display_ui", None)

__all__ = getattr(
    _target,
    "__all__",
    ["ImageMaskViewer", "create_widgets", "display_ui"],
)
