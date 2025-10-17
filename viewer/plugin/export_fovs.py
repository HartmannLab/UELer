"""Compatibility wrapper for the relocated ``export_fovs`` plugin."""

from __future__ import annotations

import importlib
import sys


_target = importlib.import_module("ueler.viewer.plugin.export_fovs")
sys.modules[__name__] = _target

PLACEHOLDER_MESSAGE = _target.PLACEHOLDER_MESSAGE
RunFlowsom = _target.RunFlowsom

__all__ = getattr(_target, "__all__", None) or ["RunFlowsom", "PLACEHOLDER_MESSAGE"]