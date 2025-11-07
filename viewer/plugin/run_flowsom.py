"""Compatibility wrapper for the relocated ``run_flowsom`` plugin."""

from __future__ import annotations

import importlib
import sys


_target = importlib.import_module("ueler.viewer.plugin.run_flowsom")
sys.modules[__name__] = _target

RunFlowsom = _target.RunFlowsom
UiComponent = _target.UiComponent
Data = _target.Data
linked_controls = _target.linked_controls

__all__ = getattr(
    _target,
    "__all__",
    ["RunFlowsom", "UiComponent", "Data", "linked_controls"],
)
