from __future__ import annotations

import importlib
import types

try:
	import anywidget  # noqa: F401
except ImportError as exc:  # pragma: no cover - dependency guard
	raise ImportError(
		"UELer requires the 'anywidget' package for jscatter integration. "
		"Install it with either 'pip install anywidget' or, if you are using micromamba, "
		"'micromamba activate <env> && pip install anywidget'. After installing, restart JupyterLab and rerun the viewer."
	) from exc


_module = importlib.import_module("ueler.viewer.plugin.scatter_widget")

__all__ = [
	name
	for name, value in vars(_module).items()
	if not name.startswith("_") and not isinstance(value, types.ModuleType)
]

globals().update({name: getattr(_module, name) for name in __all__})

del importlib
del types
del _module
