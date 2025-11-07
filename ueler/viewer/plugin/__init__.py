"""Viewer plugin subpackage for the packaged UELer namespace."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace


class _IndexStub(list):
	def __init__(self, iterable=()):
		super().__init__(iterable)
		self._positions = {value: pos for pos, value in enumerate(self)}

	def get_loc(self, label):
		if label in self._positions:
			return self._positions[label]
		raise KeyError(label)


class _SeriesLocIndexer:
	def __init__(self, series):
		self._series = series

	def __getitem__(self, key):
		positions, labels = self._series._resolve_index_labels(key)
		values = [self._series._values[pos] for pos in positions]
		if len(values) == 1 and not isinstance(key, (list, tuple, _SeriesStub, _IndexStub, set)):
			return values[0]
		return _SeriesStub(values, index=labels)


class _SeriesILocIndexer:
	def __init__(self, series):
		self._series = series

	def __getitem__(self, key):
		if isinstance(key, slice):
			indices = range(*key.indices(len(self._series)))
		elif isinstance(key, (list, tuple)):
			indices = key
		else:
			indices = [key]
		values = [self._series._values[pos] for pos in indices]
		labels = [self._series.index[pos] for pos in indices]
		if len(values) == 1 and not isinstance(key, (list, tuple)):
			return values[0]
		return _SeriesStub(values, index=labels)


class _SeriesStub:
	def __init__(self, data=(), index=None):
		values = list(data or [])
		self._values = values
		if index is None:
			self.index = _IndexStub(range(len(values)))
		else:
			self.index = _IndexStub(index)
		self._loc = _SeriesLocIndexer(self)
		self._iloc = _SeriesILocIndexer(self)

	def copy(self):
		return _SeriesStub(list(self._values), list(self.index))

	def map(self, mapper):
		if callable(mapper):
			mapped = [mapper(value) for value in self._values]
		else:
			getter = getattr(mapper, "get", None)
			mapped = []
			for value in self._values:
				if getter is not None:
					mapped.append(getter(value))
				else:
					try:
						mapped.append(mapper[value])  # type: ignore[index]
					except Exception:
						mapped.append(None)
		return _SeriesStub(mapped, index=list(self.index))

	def isna(self):
		return _SeriesStub([value is None for value in self._values], index=list(self.index))

	def astype(self, dtype):
		converted = []
		for value in self._values:
			if value is None:
				converted.append(None)
			else:
				try:
					converted.append(dtype(value))
				except Exception:
					converted.append(value)
		return _SeriesStub(converted, index=list(self.index))

	def tolist(self):
		return list(self._values)

	def reindex(self, new_index):
		labels = list(new_index)
		lookup = {label: pos for pos, label in enumerate(self.index)}
		values = []
		for label in labels:
			pos = lookup.get(label)
			values.append(self._values[pos] if pos is not None else None)
		return _SeriesStub(values, index=labels)

	def dropna(self):
		values = [value for value in self._values if value is not None]
		labels = [label for value, label in zip(self._values, self.index) if value is not None]
		return _SeriesStub(values, index=labels)

	def __len__(self):
		return len(self._values)

	def __iter__(self):
		return iter(self._values)

	def __getitem__(self, key):
		if isinstance(key, slice):
			positions = range(*key.indices(len(self)))
			values = [self._values[pos] for pos in positions]
			labels = [self.index[pos] for pos in positions]
			return _SeriesStub(values, index=labels)
		if isinstance(key, (list, tuple, set, _SeriesStub, _IndexStub)):
			labels = list(key)
			return self.loc[labels]
		position = self.index.get_loc(key)
		return self._values[position]

	def __setitem__(self, key, value):
		position = self.index.get_loc(key)
		self._values[position] = value

	def all(self):
		return all(bool(value) for value in self._values)

	@property
	def loc(self):
		return self._loc

	@property
	def iloc(self):
		return self._iloc

	def _resolve_index_labels(self, key):
		if isinstance(key, slice):
			positions = list(range(*key.indices(len(self))))
			labels = [self.index[pos] for pos in positions]
			return positions, labels
		if isinstance(key, _SeriesStub):
			labels = list(key)
		elif isinstance(key, _IndexStub):
			labels = list(key)
		elif isinstance(key, (list, tuple)):
			labels = list(key)
		elif isinstance(key, set):
			labels = list(key)
		else:
			labels = [key]
		positions = []
		for label in labels:
			positions.append(self.index.get_loc(label))
		return positions, labels


def _series_factory(pandas_module):
	candidate = getattr(pandas_module, "Series", None) if pandas_module else None
	if candidate is None:
		return _SeriesStub
	try:
		candidate([0], index=[0])
	except Exception:
		return _SeriesStub
	return candidate


def _ensure_anywidget_stub() -> None:
	if "anywidget" in sys.modules:
		return
	module = types.ModuleType("anywidget")
	module.__bootstrap_stub__ = True  # type: ignore[attr-defined]
	sys.modules["anywidget"] = module


def _ensure_intslider_stub() -> None:
	widgets = sys.modules.get("ipywidgets")
	if widgets is None or hasattr(widgets, "IntSlider"):
		return

	base = getattr(widgets, "FloatSlider", getattr(widgets, "Widget", object))

	class IntSlider(base):  # type: ignore[misc]
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.min = kwargs.get("min", 0)
			self.max = kwargs.get("max", 10)
			self.step = kwargs.get("step", 1)

	widgets.IntSlider = IntSlider  # type: ignore[attr-defined]


def _ensure_widget_exports() -> None:
	widgets = sys.modules.get("ipywidgets")
	if widgets is None:
		widgets = types.ModuleType("ipywidgets")
		sys.modules["ipywidgets"] = widgets

	if not hasattr(widgets, "Layout"):
		class Layout:  # type: ignore[override]
			def __init__(self, **kwargs):
				self.__dict__.update(kwargs)

		widgets.Layout = Layout  # type: ignore[attr-defined]

	base_widget = getattr(widgets, "Widget", object)

	class _WidgetFallback(base_widget):  # type: ignore[type-arg]
		def __init__(self, *args, **kwargs):
			if hasattr(super(), "__init__"):
				try:
					super().__init__(*args, **kwargs)
				except Exception:
					pass
			children = kwargs.get("children", ())
			if children is None and args:
				children = args[0]
			self.children = tuple(children or ())
			self.value = kwargs.get("value")
			self.options = list(kwargs.get("options", ()))
			self.allowed_tags = list(kwargs.get("allowed_tags", ()))
			self.description = kwargs.get("description", "")
			self.tooltip = kwargs.get("tooltip", "")
			self.icon = kwargs.get("icon", "")
			self.button_style = kwargs.get("button_style", "")
			layout = kwargs.get("layout")
			if layout is None:
				layout = getattr(widgets, "Layout", lambda **kw: SimpleNamespace())()
			self.layout = layout

		def observe(self, *_args, **_kwargs):
			return None

		def unobserve(self, *_args, **_kwargs):  # pragma: no cover - rarely used
			return None

		def on_click(self, *_args, **_kwargs):
			return None

		def clear_output(self, *_args, **_kwargs):
			return None

	fallback_names = [
		"Button",
		"Checkbox",
		"Dropdown",
		"FloatSlider",
		"HBox",
		"HTML",
		"RadioButtons",
		"SelectMultiple",
		"Tab",
		"TagsInput",
		"Text",
		"Textarea",
		"ToggleButtons",
		"VBox",
	]

	for name in fallback_names:
		if not hasattr(widgets, name):
			setattr(widgets, name, _WidgetFallback)

	if not hasattr(widgets, "Output"):
		class _OutputFallback(_WidgetFallback):
			def clear_output(self, wait=False):  # pragma: no cover - simple stub
				return None

		widgets.Output = _OutputFallback  # type: ignore[attr-defined]


def _initial_color_series(pd_series, data):
	if hasattr(data, "index"):
		index = data.index
		length = len(index)
	else:
		length = len(data) if hasattr(data, "__len__") else 0
		index = range(length)
	return pd_series([0] * length, index=index)


def _create_jscatter_stub(pd_series):
	module = types.ModuleType("jscatter")

	class _WidgetStub:
		def __init__(self):
			self.layout = SimpleNamespace()
			self.mouse_mode = None
			self._observers = {}
			self.buttons = ()

		def observe(self, callback, names=None):
			names = tuple(names or ())
			self._observers.setdefault(names, []).append(callback)

		def unobserve(self, callback, names=None):  # pragma: no cover - rarely used
			names = tuple(names or ())
			callbacks = self._observers.get(names)
			if callbacks and callback in callbacks:
				callbacks.remove(callback)

	default_color = (0.2, 0.4, 0.8, 0.85)
	_SELECTION_SENTINEL = object()

	class Scatter:
		def __init__(self, *, x, y, data, data_use_index=False):
			self._x = x
			self._y = y
			self._data = data
			self._data_use_index = data_use_index
			self._selection = []
			self._height = 0
			self._size = None
			self._color_state = {"color": default_color}
			self._color_map = {}
			self._color_data = _initial_color_series(pd_series, data)
			self.widget = _WidgetStub()

		def axes(self, *args, **kwargs):  # pragma: no cover - minimal stub
			return None

		def height(self, value=None):
			if value is not None:
				self._height = value
			return self._height

		def size(self, default=None, **kwargs):
			self._size = default
			return None

		def color(self, **kwargs):
			if not kwargs:
				return self._color_state
			if "by" in kwargs:
				by_values = kwargs.get("by") or []
				mapping = kwargs.get("map") or []
				index = getattr(self._data, "index", range(len(by_values)))
				self._color_data = pd_series(by_values, index=index)
				self._color_map = {idx: mapping[idx] for idx in range(len(mapping))}
				self._color_state = {
					"by": by_values,
					"map": mapping,
					"norm": kwargs.get("norm"),
				}
			else:
				self._color_state = {
					"color": kwargs.get("default", default_color),
					"selected": kwargs.get("selected"),
				}
			return self._color_state

		def tooltip(self, *args, **kwargs):  # pragma: no cover - minimal stub
			return None

		def show(self, buttons=None):
			self.widget.buttons = tuple(buttons or ())
			return self.widget

		def selection(self, values=_SELECTION_SENTINEL):
			if values is _SELECTION_SENTINEL:
				return list(self._selection)
			if values is None:
				self._selection = []
			else:
				self._selection = list(values)
			return list(self._selection)

	module.Scatter = Scatter  # type: ignore[attr-defined]
	module.compose = lambda *entries, **__kwargs: entries  # type: ignore[attr-defined]
	return module


def _ensure_jscatter_stub() -> None:
	if "jscatter" in sys.modules:
		return
	try:  # pragma: no cover - prefer real dependency when available
		import jscatter  # type: ignore  # noqa: F401
	except Exception:
		pandas_module = sys.modules.get("pandas")
		pd_series = _series_factory(pandas_module)
		sys.modules["jscatter"] = _create_jscatter_stub(pd_series)
	else:  # pragma: no cover - real dependency present
		sys.modules.setdefault("jscatter", jscatter)  # type: ignore[name-defined]


_ensure_intslider_stub()
_ensure_jscatter_stub()
_ensure_widget_exports()
_ensure_anywidget_stub()

__all__: list[str] = []

del sys
del types
del _ensure_intslider_stub
del _ensure_jscatter_stub
del _ensure_anywidget_stub
del _ensure_widget_exports
del _series_factory
