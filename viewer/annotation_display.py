"""Compatibility wrapper for the relocated ``annotation_display`` module."""

from __future__ import annotations

import importlib
import sys


_target = importlib.import_module("ueler.viewer.annotation_display")
sys.modules[__name__] = _target

AnnotationDisplay = _target.AnnotationDisplay
UiComponent = _target.UiComponent
Data = _target.Data
linked_controls = _target.linked_controls

__all__ = getattr(
	_target,
	"__all__",
	None,
) or ["AnnotationDisplay", "UiComponent", "Data", "linked_controls"]