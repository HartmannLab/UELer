"""Compatibility wrapper for the relocated ``decorators`` module."""

from __future__ import annotations

import importlib
import sys


_target = importlib.import_module("ueler.viewer.decorators")
sys.modules[__name__] = _target

update_status_bar = _target.update_status_bar

__all__ = getattr(_target, "__all__", None) or ["update_status_bar"]