"""Compatibility wrapper for ``ueler.viewer.interfaces``."""

from __future__ import annotations

import importlib
import sys

_target = importlib.import_module("ueler.viewer.interfaces")
sys.modules[__name__] = _target
