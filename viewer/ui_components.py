"""Compatibility wrapper for the relocated ``ui_components`` module."""

from __future__ import annotations

import importlib
import sys


_target = importlib.import_module("ueler.viewer.ui_components")
sys.modules[__name__] = _target

from ueler.viewer.ui_components import *  # noqa: F401,F403

__all__ = getattr(_target, "__all__", [])
