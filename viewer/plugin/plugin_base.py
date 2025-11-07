"""Compatibility wrapper for the relocated ``plugin_base`` module."""

from __future__ import annotations

import importlib
import sys


_target = importlib.import_module("ueler.viewer.plugin.plugin_base")
sys.modules[__name__] = _target

PluginBase = _target.PluginBase

__all__ = getattr(_target, "__all__", None) or ["PluginBase"]