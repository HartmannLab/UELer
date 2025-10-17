"""Compatibility wrapper for the relocated ``observable`` module."""

from __future__ import annotations

import importlib
import sys


_target = importlib.import_module("ueler.viewer.observable")
sys.modules[__name__] = _target

Observable = _target.Observable

__all__ = getattr(_target, "__all__", None) or ["Observable"]