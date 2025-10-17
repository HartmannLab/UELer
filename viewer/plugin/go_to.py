"""Compatibility wrapper for the relocated ``go_to`` plugin."""

from __future__ import annotations

import importlib
import sys


_target = importlib.import_module("ueler.viewer.plugin.go_to")
sys.modules[__name__] = _target

goTo = _target.goTo
UiComponent = _target.UiComponent

__all__ = getattr(_target, "__all__", None) or ["goTo", "UiComponent"]
