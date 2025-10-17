from __future__ import annotations

import importlib
import types


_module = importlib.import_module("ueler.viewer.plugin.mask_painter")

__all__ = [
	name
	for name, value in vars(_module).items()
	if not name.startswith("_") and not isinstance(value, types.ModuleType)
]

globals().update({name: getattr(_module, name) for name in __all__})

del importlib
del types
del _module
