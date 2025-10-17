"""Compatibility wrapper for the relocated ``roi_manager`` module."""

from __future__ import annotations

import importlib
import sys


_target = importlib.import_module("ueler.viewer.roi_manager")
sys.modules[__name__] = _target

ROI_COLUMNS = _target.ROI_COLUMNS
ROIManager = _target.ROIManager

__all__ = getattr(_target, "__all__", None) or ["ROI_COLUMNS", "ROIManager"]
