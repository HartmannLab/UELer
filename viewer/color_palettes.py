"""Compatibility wrapper for the relocated ``color_palettes`` module."""

from __future__ import annotations

import importlib
import sys


_target = importlib.import_module("ueler.viewer.color_palettes")
sys.modules[__name__] = _target

from ueler.viewer.color_palettes import *  # noqa: F401,F403

__all__ = getattr(_target, "__all__", [])
