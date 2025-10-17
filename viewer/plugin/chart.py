from __future__ import annotations

import importlib
import sys
import types


_widgets = sys.modules.get("ipywidgets")
if _widgets is not None and not hasattr(_widgets, "IntSlider"):
	base_widget = getattr(_widgets, "FloatSlider", getattr(_widgets, "Widget", object))

	class _FallbackIntSlider(base_widget):
		def __init__(self, *args, **kwargs):  # noqa: D401 - compatibility stub
			super().__init__(*args, **kwargs)
			self.min = kwargs.get("min", 0)
			self.max = kwargs.get("max", 10)
			self.step = kwargs.get("step", 1)

	_widgets.IntSlider = _FallbackIntSlider  # type: ignore[attr-defined]


_module = importlib.import_module("ueler.viewer.plugin.chart")

__all__ = [
	name
	for name, value in vars(_module).items()
	if not name.startswith("_") and not isinstance(value, types.ModuleType)
]

globals().update({name: getattr(_module, name) for name in __all__})

del importlib
del sys
del types
del _module
del _widgets
try:
	del _FallbackIntSlider  # type: ignore[name-defined]
	del base_widget  # type: ignore[name-defined]
except NameError:
	pass