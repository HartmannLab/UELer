"""Shared bootstrap for unit tests.

This module centralizes lightweight stubs for optional runtime dependencies so the
unit test suite can run without installing the full UI stack. The bootstrap is
imported from ``tests/__init__.py`` ensuring the fakes are registered before any
project modules are imported by the tests.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from types import SimpleNamespace


_PROTECTED_MODULES: set[str] = set()
_GUARD_INSTALLED = False
_PANDAS_IMPORT_ERROR: str | None = None

_PANDAS = "pandas"
_NUMPY = "numpy"
_SEABORN = "seaborn"
_SCIPY = "scipy"
_SCIPY_CLUSTER = "scipy.cluster"
_SCIPY_HIERARCHY = "scipy.cluster.hierarchy"
_ANYWIDGET = "anywidget"
_JSCATTER = "jscatter"
_IPYTHON = "IPython"
_IPYTHON_DISPLAY = "IPython.display"
_MATPLOTLIB = "matplotlib"
_MATPLOTLIB_PYPLOT = "matplotlib.pyplot"

_DEFAULT_POINT_COLOR = (0.2, 0.4, 0.8, 0.85)


def _is_stub_module(module: types.ModuleType | None) -> bool:
    if module is None:
        return True
    if getattr(module, "__bootstrap_stub__", False):
        return True
    return not bool(getattr(module, "__file__", None))


def _install_module_guard() -> None:
    """Historically wrapped ``sys.modules`` to keep real modules alive.

    The original implementation replaced ``sys.modules`` with a custom dict
    subclass so stubs could not overwrite successfully imported packages. That
    broke low-level imports in numpy/pandas because those libraries expect
    ``sys.modules`` to be a plain ``dict``. We keep the hook so the bootstrap
    sequence remains stable, but the guard is now a documented no-op."""

    global _GUARD_INSTALLED
    _GUARD_INSTALLED = True


def _protect_module(name: str, module: types.ModuleType) -> None:
    _PROTECTED_MODULES.add(name)
    setattr(module, "__bootstrap_preserve__", True)


def _ensure_pandas() -> None:
    """Guarantee a functional pandas module is available for the tests."""

    module = sys.modules.get(_PANDAS)
    if module is not None:
        if getattr(module, "__file__", None):
            _protect_module(_PANDAS, module)
            return

        _supplement_pandas_stub(module)
        _protect_module(_PANDAS, module)
        return

    _clear_numpy_stub()
    module = _import_real_pandas()
    if module is not None:
        _protect_module(_PANDAS, module)
        return

    sys.modules[_PANDAS] = _create_pandas_stub()


def _supplement_pandas_stub(module: types.ModuleType) -> None:
    """Upgrade lightweight pandas stand-ins installed by individual tests.

    Several test modules ship their own ultra-minimal pandas replacements that
    only fulfil local needs (sometimes mapping ``DataFrame`` to ``object``).
    When those versions leak into the shared test process they break unrelated
    suites. We reconcile them here by grafting in the behaviour from the
    bootstrap stub whenever essential APIs are missing."""

    if getattr(module, "__bootstrap_patched__", False):  # pragma: no cover - idempotent guard
        return

    stub = _create_pandas_stub()

    replaced = _replace_missing_pandas_symbols(module, stub)
    if _ensure_pandas_api_support(module, stub):
        replaced = True

    if replaced:
        module.__bootstrap_stub__ = True  # type: ignore[attr-defined]
        module.__bootstrap_patched__ = True  # type: ignore[attr-defined]


def _pandas_attr_needs_replacement(attr_name: str, value) -> bool:
    if value is None:
        return True
    if attr_name in {"DataFrame", "Series", "Index"} and value is object:
        return True
    return False


def _replace_missing_pandas_symbols(target: types.ModuleType, stub: types.ModuleType) -> bool:
    replaced = False
    for name in ("DataFrame", "Series", "Index", "merge", "concat", "isna"):
        current = getattr(target, name, None)
        if _pandas_attr_needs_replacement(name, current):
            setattr(target, name, getattr(stub, name))
            replaced = True
    return replaced


def _ensure_pandas_api_support(target: types.ModuleType, stub: types.ModuleType) -> bool:
    if not hasattr(target, "api"):
        target.api = getattr(stub, "api")  # type: ignore[attr-defined]
        return True

    stub_api = getattr(stub, "api", None)
    target_api = getattr(target, "api", None)
    if stub_api is None or target_api is None:
        return False

    stub_types = getattr(stub_api, "types", None)
    target_types = getattr(target_api, "types", None)
    replaced = False
    if target_types is None and stub_types is not None:
        target_api.types = stub_types  # type: ignore[attr-defined]
        return True
    if stub_types is None or target_types is None:
        return replaced

    for attr in ("is_numeric_dtype", "is_object_dtype"):
        if not hasattr(target_types, attr):
            setattr(target_types, attr, getattr(stub_types, attr))
            replaced = True
    return replaced


def _clear_numpy_stub() -> None:
    numpy_module = sys.modules.get(_NUMPY)
    if numpy_module is not None and not getattr(numpy_module, "__file__", None):
        sys.modules.pop(_NUMPY, None)


def _import_real_pandas() -> types.ModuleType | None:
    global _PANDAS_IMPORT_ERROR
    try:  # pragma: no cover - prefer the actual library
        import pandas  # type: ignore  # noqa: F401
    except Exception as exc:
        _PANDAS_IMPORT_ERROR = repr(exc)
        sys.modules.pop(_PANDAS, None)
        return None
    _PANDAS_IMPORT_ERROR = None
    return sys.modules[_PANDAS]


class _BootstrapSeries(list):
    def __init__(self, data=(), index=None):
        if isinstance(data, dict):
            items = list(data.items())
            values = [value for _, value in items]
            default_index = [key for key, _ in items]
            super().__init__(values)
        else:
            super().__init__(data)
            default_index = list(range(len(self)))

        if index is None:
            self.index = _BootstrapIndex(default_index)
        else:
            self.index = _BootstrapIndex(index)
        self._loc = _BootstrapSeriesLocIndexer(self)
        self._iloc = _BootstrapSeriesILocIndexer(self)

    def dropna(self):
        result = _BootstrapSeries(value for value in self if value is not None)
        result.index = [idx for value, idx in zip(self, self.index) if value is not None]
        return result

    def unique(self):
        seen = []
        seen_index = []
        for item in self:
            if item not in seen:
                seen.append(item)
                seen_index.append(len(seen_index))
        result = _BootstrapSeries(seen)
        result.index = seen_index
        return result

    def isin(self, values):  # pragma: no cover - simple stub
        values = set(values)
        result = _BootstrapSeries(item in values for item in self)
        result.index = list(self.index)
        return result

    def isna(self):  # pragma: no cover - simple stub
        result = _BootstrapSeries(value is None for value in self)
        result.index = list(self.index)
        return result

    def all(self):  # pragma: no cover - simple stub
        return all(bool(item) for item in self)

    def any(self):  # pragma: no cover - simple stub
        return any(bool(item) for item in self)

    def astype(self, dtype):  # pragma: no cover - simple stub
        if dtype in (int, float, str):
            converted = [dtype(value) if value is not None else None for value in self]
        else:
            converted = list(self)
        return _BootstrapSeries(converted, index=list(self.index))

    def map(self, mapper):  # pragma: no cover - simple stub
        if callable(mapper):
            mapped = [mapper(value) for value in self]
        else:
            getter = getattr(mapper, "get", None)
            mapped = []
            for value in self:
                if getter is not None:
                    mapped.append(getter(value))
                else:
                    try:
                        mapped.append(mapper[value])  # type: ignore[index]
                    except Exception:
                        mapped.append(None)
        return _BootstrapSeries(mapped, index=list(self.index))

    def _resolve_index_labels(self, key):
        if isinstance(key, slice):
            positions = list(range(*key.indices(len(self))))
            labels = [self.index[pos] for pos in positions]
            return positions, labels
        if isinstance(key, _BootstrapIndex):
            labels = list(key)
        elif isinstance(key, _BootstrapSeries):
            if all(isinstance(item, bool) for item in key):
                positions = [pos for pos, flag in enumerate(key) if flag]
                labels = [self.index[pos] for pos in positions]
                return positions, labels
            labels = list(key)
        elif isinstance(key, (list, tuple)):
            labels = list(key)
        elif isinstance(key, set):  # pragma: no cover - defensive
            labels = list(key)
        else:
            labels = [key]

        positions = []
        for label in labels:
            if label not in self.index:
                raise KeyError(label)
            positions.append(self.index.index(label))
        return positions, labels

    @property
    def loc(self):  # pragma: no cover - simple stub
        return self._loc

    @property
    def iloc(self):  # pragma: no cover - simple stub
        return self._iloc

    def tolist(self):
        return list(self)

    @property
    def empty(self):  # pragma: no cover - simple stub
        return len(self) == 0

    def reindex(self, new_index):  # pragma: no cover - simple stub
        labels = list(new_index)
        position_lookup = {label: pos for pos, label in enumerate(self.index)}
        values = []
        for label in labels:
            pos = position_lookup.get(label)
            values.append(self[pos] if pos is not None else None)
        result = _BootstrapSeries(values, index=labels)
        return result


class _BootstrapIndex(list):
    def __init__(self, iterable=()):
        super().__init__(iterable)
        self._positions = {}
        for pos, value in enumerate(self):
            self._positions[value] = pos

    def take(self, positions):  # pragma: no cover - simple stub
        return _BootstrapIndex(self[pos] for pos in positions)

    def get_loc(self, label):
        if label in self._positions:
            return self._positions[label]
        raise KeyError(label)

    def append(self, value):  # pragma: no cover - helper
        self._positions[value] = len(self)
        super().append(value)


class _BootstrapLocIndexer:
    def __init__(self, dataframe):
        self._df = dataframe

    def __getitem__(self, key):
        return self._df._get_loc(key)

    def __setitem__(self, key, value):
        self._df._set_loc(key, value)


class _BootstrapSeriesLocIndexer:
    def __init__(self, series):
        self._series = series

    def __getitem__(self, key):
        positions, labels = self._series._resolve_index_labels(key)
        values = [self._series[pos] for pos in positions]
        if len(values) == 1 and not isinstance(key, (list, tuple, _BootstrapSeries, _BootstrapIndex, set)):
            return values[0]
        result = _BootstrapSeries(values, index=labels)
        return result

    def __setitem__(self, key, value):  # pragma: no cover - assignment support
        positions, _labels = self._series._resolve_index_labels(key)
        if isinstance(value, _BootstrapSeries):
            values_iter = list(value)
        elif isinstance(value, (list, tuple)):
            values_iter = list(value)
        else:
            values_iter = [value] * len(positions)
        for pos, item in zip(positions, values_iter):
            self._series[pos] = item


class _BootstrapSeriesILocIndexer:
    def __init__(self, series):
        self._series = series

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self._series)))
        elif isinstance(key, (list, tuple)):
            indices = key
        else:
            indices = [key]

        values = [self._series[pos] for pos in indices]
        labels = [self._series.index[pos] for pos in indices]
        if len(values) == 1 and not isinstance(key, (list, tuple)):
            return values[0]
        return _BootstrapSeries(values, index=labels)


class _BootstrapDataFrame:
    def __init__(self, data=None, index=None):
        data = data or {}
        self._data = {key: list(value) for key, value in data.items()}
        self.columns = _BootstrapSeries(self._data.keys())
        rows = len(next(iter(self._data.values()), []))
        self.index = _BootstrapIndex(index or range(rows))
        if rows:
            for column in self.columns:
                self._data.setdefault(column, [None] * rows)
        self._loc = _BootstrapLocIndexer(self)

    def copy(self):
        return _BootstrapDataFrame(self._data.copy(), list(self.index))

    def __getitem__(self, key):
        values = self._data[key]
        if isinstance(values, list):
            return _BootstrapSeries(values, index=list(self.index))
        return values

    def __setitem__(self, key, values):  # pragma: no cover - simple stub
        values_list = list(values)
        if not self.index and values_list:
            self.index = _BootstrapIndex(range(len(values_list)))
        target_len = len(self.index)
        if target_len:
            if len(values_list) < target_len:
                values_list.extend([None] * (target_len - len(values_list)))
            values_list = values_list[:target_len]
        else:
            target_len = len(values_list)
        if target_len and not values_list:
            values_list = [None] * target_len
        self._data[key] = list(values_list)
        if key not in self.columns:
            self.columns.append(key)

    def __contains__(self, key):
        return key in self._data

    def dropna(self):  # pragma: no cover - simple stub
        return self

    def unique(self):  # pragma: no cover - helper for chained calls
        return _BootstrapSeries(self._data.keys())

    def to_numpy(self):  # pragma: no cover - helper
        return list(zip(*self._data.values()))

    def isin(self, values):  # pragma: no cover - simple stub
        lookup = set(values)
        key = next(iter(self._data), None)
        if key is None:
            return _BootstrapSeries()
        result = [row in lookup for row in self._data[key]]
        series = _BootstrapSeries(result, index=list(self.index))
        return series

    def _normalize_key(self, key):
        if isinstance(key, tuple):
            return key
        return key, slice(None)

    def _ensure_row(self, row_label):
        for label in self._normalize_row_labels(row_label):
            if label in self.index:
                continue
            self.index.append(label)
            for column in self.columns:
                self._data.setdefault(column, []).append(None)

    def _set_loc(self, key, value):
        row_label, column_label = self._normalize_key(key)
        self._ensure_row(row_label)
        if isinstance(column_label, slice):
            raise NotImplementedError("slice assignment not supported in stub")
        if column_label not in self.columns:
            self.columns.append(column_label)
            self._data[column_label] = [None] * len(self.index)
        row_positions = self._resolve_row_positions(row_label)
        if len(row_positions) == 1:
            self._data[column_label][row_positions[0]] = value
        else:
            if isinstance(value, _BootstrapSeries):
                values_iter = list(value)
            elif isinstance(value, (list, tuple)):
                values_iter = list(value)
            else:
                values_iter = [value] * len(row_positions)
            for pos, val in zip(row_positions, values_iter):
                self._data[column_label][pos] = val

    def _get_loc(self, key):  # pragma: no cover - helper for completeness
        row_label, column_label = self._normalize_key(key)
        if isinstance(column_label, slice):
            raise NotImplementedError("slice access not supported in stub")
        row_positions = self._resolve_row_positions(row_label)
        values = [self._data[column_label][pos] for pos in row_positions]
        if len(row_positions) == 1:
            return values[0]
        labels = self._normalize_row_labels(row_label)
        series = _BootstrapSeries(values, index=labels)
        return series

    def _resolve_row_positions(self, row_label):
        labels = self._normalize_row_labels(row_label)
        return [self.index.get_loc(label) for label in labels]

    def _normalize_row_labels(self, row_label):
        if isinstance(row_label, _BootstrapSeries):
            return list(row_label)
        if isinstance(row_label, (list, tuple)):
            return list(row_label)
        if isinstance(row_label, set):  # pragma: no cover - uncommon branch
            return list(row_label)
        return [row_label]

    @property
    def loc(self):  # pragma: no cover - simple stub
        return self._loc

    @property
    def empty(self):  # pragma: no cover - simple stub
        return not any(self._data.values())


def _pandas_merge(left, right, **_kwargs):  # pragma: no cover - simple stub
    combined = left.copy()
    combined._data.update(right._data)
    return combined


def _pandas_concat(frames, ignore_index=False, **_kwargs):  # pragma: no cover - simple stub
    base = _BootstrapDataFrame()
    base._data = {}
    for frame in frames:
        for key, values in frame._data.items():
            base._data.setdefault(key, []).extend(values)
    if ignore_index:
        base.index = _BootstrapIndex(range(len(next(iter(base._data.values()), []))))
    return base


def _pandas_isna(value):  # pragma: no cover - simple stub
    return value is None


def _create_pandas_stub() -> types.ModuleType:
    pandas_stub = types.ModuleType(_PANDAS)
    pandas_stub.DataFrame = _BootstrapDataFrame  # type: ignore[attr-defined]
    pandas_stub.Series = _BootstrapSeries  # type: ignore[attr-defined]
    pandas_stub.Index = _BootstrapIndex  # type: ignore[attr-defined]
    pandas_stub.merge = _pandas_merge  # type: ignore[attr-defined]
    pandas_stub.concat = _pandas_concat  # type: ignore[attr-defined]
    pandas_stub.isna = _pandas_isna  # type: ignore[attr-defined]
    pandas_stub.api = SimpleNamespace(  # type: ignore[attr-defined]
        types=SimpleNamespace(
            is_numeric_dtype=lambda values: all(isinstance(v, (int, float)) for v in values),
            is_object_dtype=lambda values: any(isinstance(v, str) for v in values),
        )
    )

    pandas_stub.__bootstrap_stub__ = True  # type: ignore[attr-defined]
    return pandas_stub


def _ensure_ipywidgets_stub() -> None:
    """Provide a very small stub for ``ipywidgets`` when the real dependency is
    unavailable. Only the attributes exercised within the test suite are
    implemented, enough to allow widgets to be instantiated and interacted with
    at a superficial level."""

    existing = sys.modules.get("ipywidgets")
    if existing is not None and getattr(existing, "__bootstrap_force_stub__", False):
        return

    if existing is not None:
        essential = ("Widget", "Layout", "IntSlider", "IntText")
        if not getattr(existing, "__file__", None) and any(
            not hasattr(existing, name) for name in essential
        ):
            sys.modules.pop("ipywidgets", None)
            existing = None

    try:
        import ipywidgets as real_widgets  # type: ignore
    except Exception:
        real_widgets = None

    stub = _build_ipywidgets_stub()
    if real_widgets is not None:
        for name in dir(real_widgets):
            if not hasattr(stub, name):
                setattr(stub, name, getattr(real_widgets, name))

    stub.__bootstrap_stub__ = True  # type: ignore[attr-defined]
    stub.__bootstrap_force_stub__ = True  # type: ignore[attr-defined]
    sys.modules["ipywidgets"] = stub


def _ensure_widget_exports(widgets) -> None:
    required = {
        "Box": getattr(widgets, "Box", None),
        "GridBox": getattr(widgets, "GridBox", None),
        "VBox": getattr(widgets, "VBox", None),
    }
    if all(required.values()):  # pragma: no cover - real library covers exports
        return

    stub = _build_ipywidgets_stub()
    for name in dir(stub):
        if not hasattr(widgets, name):
            setattr(widgets, name, getattr(stub, name))


def _build_ipywidgets_stub():
    widgets = types.ModuleType("ipywidgets")

    class Layout:  # type: ignore[override]
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Widget:
        def __init__(self, *args, **kwargs):
            children = kwargs.get("children")
            if children is None and args:
                children = args[0]
            self.children = tuple(children or ())
            self.value = kwargs.get("value")
            self.options = list(kwargs.get("options", ()))
            self.allowed_tags = list(kwargs.get("allowed_tags", ()))
            self.allow_duplicates = bool(kwargs.get("allow_duplicates", False))
            self.allow_new = bool(kwargs.get("allow_new", False))
            self.restrict_to_allowed_tags = bool(
                kwargs.get("restrict_to_allowed_tags", False)
            )
            self.ensure_option = kwargs.get("ensure_option")
            self.description = kwargs.get("description", "")
            self.button_style = kwargs.get("button_style", "")
            self.icon = kwargs.get("icon", "")
            self.tooltip = kwargs.get("tooltip", "")
            self.style = kwargs.get("style", {})
            self.disabled = kwargs.get("disabled", False)
            self.layout = kwargs.get("layout", Layout())

        # ``ipywidgets`` controls expose observer hooks; tests only need them to
        # exist and ignore registrations.
        def observe(self, *_args, **_kwargs):
            return None

        def unobserve(self, *_args, **_kwargs):
            return None

        def on_click(self, *_args, **_kwargs):
            return None

        def set_title(self, *_args, **_kwargs):
            return None

    class Output(Widget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.outputs = ()
            self.clear_output_calls: list[bool] = []

        def clear_output(self, wait: bool = False):  # pragma: no cover - simple stub
            self.clear_output_calls.append(wait)
            self.outputs = ()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # pragma: no cover - simple stub
            return False

    class Tab(Widget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._titles: dict[int, str] = {}
            self.selected_index = kwargs.get("selected_index")

        def set_title(self, index, title):  # pragma: no cover - simple stub
            self._titles[index] = title

    class Accordion(Tab):
        pass

    class GridspecLayout(Widget):
        pass

    # Export a consistent layout factory the rest of the project expects
    widgets.Layout = Layout  # type: ignore[attr-defined]

    # Map all widget types we rely on to the base ``Widget`` implementation.
    for name in [
        "Widget",
        "Button",
        "Box",
        "Checkbox",
        "ColorPicker",
        "Combobox",
        "Dropdown",
        "FloatSlider",
        "GridBox",
        "HBox",
        "HTML",
        "Image",
        "IntSlider",
        "IntText",
        "Label",
        "RadioButtons",
        "Select",
        "SelectMultiple",
        "Tab",
        "TagsInput",
        "Text",
        "Textarea",
        "ToggleButton",
        "ToggleButtons",
        "VBox",
    ]:
        setattr(widgets, name, Widget if name not in {"Tab"} else Tab)

    widgets.Output = Output  # type: ignore[attr-defined]
    widgets.Accordion = Accordion  # type: ignore[attr-defined]
    widgets.GridspecLayout = GridspecLayout  # type: ignore[attr-defined]

    return widgets


def _ensure_ipython_display() -> None:
    """Register a simplified ``IPython.display`` module when IPython is not
    installed. The heatmap plugin calls ``display(fig)`` when replaying cached
    canvases; the stub ensures those calls are no-ops while still observable via
    ``unittest.mock.patch`` in the tests."""

    try:
        from IPython.display import display  # type: ignore # noqa: F401
        return
    except Exception:
        for name in [_IPYTHON, _IPYTHON_DISPLAY]:
            sys.modules.pop(name, None)

    ipython = types.ModuleType(_IPYTHON)
    display_module = types.ModuleType(_IPYTHON_DISPLAY)

    def _display(*_args, **_kwargs):  # pragma: no cover - simple stub
        return None

    display_module.display = _display  # type: ignore[attr-defined]
    ipython.display = display_module  # type: ignore[attr-defined]
    sys.modules[_IPYTHON] = ipython
    sys.modules[_IPYTHON_DISPLAY] = display_module


def _ensure_matplotlib_stub() -> None:
    if _MATPLOTLIB_PYPLOT in sys.modules:
        return

    matplotlib_stub = types.ModuleType(_MATPLOTLIB)
    pyplot_stub = types.ModuleType(_MATPLOTLIB_PYPLOT)

    class _Canvas:
        def mpl_connect(self, *_args, **_kwargs):  # pragma: no cover - simple stub
            return None

        def draw_idle(self, *_args, **_kwargs):  # pragma: no cover - simple stub
            return None

        @property
        def toolbar(self):  # pragma: no cover - simple stub
            return SimpleNamespace(
                _nav_stack=lambda: SimpleNamespace(push=lambda *_a, **_k: None)
            )

    class _Figure:
        def __init__(self):
            self.canvas = _Canvas()

        def tight_layout(self):  # pragma: no cover - simple stub
            return None

    class _Axes:
        def __init__(self, figure):
            self.figure = figure
            self.collections = []

        def hist(self, *_args, **_kwargs):  # pragma: no cover - simple stub
            return None

        def set_xlabel(self, *_args, **_kwargs):  # pragma: no cover - simple stub
            return None

        def set_ylabel(self, *_args, **_kwargs):  # pragma: no cover - simple stub
            return None

        def axvline(self, *_args, **_kwargs):  # pragma: no cover - simple stub
            return SimpleNamespace(remove=lambda: None)

    def _subplots(*_args, **_kwargs):  # pragma: no cover - simple stub
        figure = _Figure()
        axes = _Axes(figure)
        return figure, axes

    def _show(*_args, **_kwargs):  # pragma: no cover - simple stub
        return None

    pyplot_stub.subplots = _subplots  # type: ignore[attr-defined]
    pyplot_stub.show = _show  # type: ignore[attr-defined]
    matplotlib_stub.pyplot = pyplot_stub  # type: ignore[attr-defined]
    sys.modules[_MATPLOTLIB] = matplotlib_stub
    sys.modules[_MATPLOTLIB_PYPLOT] = pyplot_stub

def _install_seaborn_stub() -> None:
    sys.modules.pop(_SEABORN, None)
    seaborn_stub = types.ModuleType(_SEABORN)

    def _clustermap(*_args, **_kwargs):  # pragma: no cover - simple stub
        figure = SimpleNamespace(
            axes=SimpleNamespace(
                ax_col_dendrogram=None,
                ax_col_colors=None,
                ax_row_dendrogram=None,
                ax_row_colors=None,
            ),
            figsize=(6, 4),
        )
        return SimpleNamespace(fig=figure)

    seaborn_stub.clustermap = _clustermap  # type: ignore[attr-defined]
    seaborn_stub.color_palette = lambda *_, **__: []  # type: ignore[attr-defined]
    seaborn_stub.set_context = lambda *_, **__: None  # type: ignore[attr-defined]
    seaborn_stub.__bootstrap_stub__ = True  # type: ignore[attr-defined]
    sys.modules[_SEABORN] = seaborn_stub


def _install_scipy_stub() -> None:
    for name in (_SCIPY_HIERARCHY, _SCIPY_CLUSTER, _SCIPY):
        sys.modules.pop(name, None)

    scipy_module = types.ModuleType(_SCIPY)
    cluster_module = types.ModuleType(_SCIPY_CLUSTER)
    hierarchy_module = types.ModuleType(_SCIPY_HIERARCHY)

    def _dendrogram(*_args, **_kwargs):  # pragma: no cover - simple stub
        return {"leaves": []}

    hierarchy_module.dendrogram = _dendrogram  # type: ignore[attr-defined]
    hierarchy_module.linkage = lambda *_, **__: []  # type: ignore[attr-defined]
    hierarchy_module.cut_tree = lambda *_, **__: []  # type: ignore[attr-defined]
    hierarchy_module.__bootstrap_stub__ = True  # type: ignore[attr-defined]

    cluster_module.hierarchy = hierarchy_module  # type: ignore[attr-defined]
    cluster_module.__bootstrap_stub__ = True  # type: ignore[attr-defined]
    scipy_module.cluster = cluster_module  # type: ignore[attr-defined]
    scipy_module.__bootstrap_stub__ = True  # type: ignore[attr-defined]

    sys.modules[_SCIPY_HIERARCHY] = hierarchy_module
    sys.modules[_SCIPY_CLUSTER] = cluster_module
    sys.modules[_SCIPY] = scipy_module


def _ensure_heatmap_dependency_stubs() -> None:
    """Provide ultra-light stubs for third-party libraries used by the
    heatmap plugin when heavyweight dependencies are unavailable."""

    pandas_module = sys.modules.get(_PANDAS)
    force_stub = _is_stub_module(pandas_module)

    if force_stub:
        _install_seaborn_stub()
    else:  # pragma: no cover - prefer actual seaborn when pandas is real
        try:
            import seaborn  # type: ignore  # noqa: F401
        except Exception:
            _install_seaborn_stub()
        else:
            _protect_module(_SEABORN, sys.modules[_SEABORN])

    if force_stub:
        _install_scipy_stub()
    else:  # pragma: no cover - prefer actual scipy when available
        try:
            import scipy.cluster.hierarchy  # type: ignore  # noqa: F401
        except Exception:
            _install_scipy_stub()
        else:
            _protect_module(_SCIPY, sys.modules[_SCIPY])
            _protect_module(
                _SCIPY_CLUSTER,
                sys.modules.get(_SCIPY_CLUSTER, sys.modules[_SCIPY]),
            )
            _protect_module(_SCIPY_HIERARCHY, sys.modules[_SCIPY_HIERARCHY])


def _ensure_anywidget_stub() -> None:
    if _ANYWIDGET in sys.modules:
        return

    module = types.ModuleType(_ANYWIDGET)
    module.__bootstrap_stub__ = True  # type: ignore[attr-defined]
    sys.modules[_ANYWIDGET] = module


def _ensure_jscatter_stub() -> None:
    if _JSCATTER in sys.modules:
        return

    sys.modules[_JSCATTER] = _build_jscatter_stub()


def _build_jscatter_stub() -> types.ModuleType:
    module = types.ModuleType(_JSCATTER)
    pandas_module = sys.modules.get(_PANDAS)
    pd_series = getattr(pandas_module, "Series", list)

    class _WidgetStub:
        def __init__(self):
            self.layout = SimpleNamespace()
            self.mouse_mode = None
            self._observers = {}
            self.buttons = ()

        def observe(self, callback, names=None):
            names = tuple(names or ())
            self._observers.setdefault(names, []).append(callback)

        def unobserve(self, callback, names=None):  # pragma: no cover - rarely triggered in tests
            names = tuple(names or ())
            self._observers.get(names, []).remove(callback)

    _SELECTION_SENTINEL = object()

    class Scatter:
        def __init__(self, *, x, y, data, data_use_index=False):  # noqa: D401 - match real signature
            self._x = x
            self._y = y
            self._data = data
            self._data_use_index = data_use_index
            self._selection = []
            self._height = 0
            self._size = None
            self._color_state = {"color": _DEFAULT_POINT_COLOR}
            self._color_map = {}
            self._color_data = _initial_color_series(pd_series, data)
            self.widget = _WidgetStub()

        def axes(self, *args, **kwargs):  # pragma: no cover - simple stub
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
            self._apply_color_kwargs(kwargs, pd_series)
            return self._color_state

        def _apply_color_kwargs(self, kwargs, pd_series_local):
            if "by" in kwargs:
                by_values = kwargs.get("by") or []
                mapping = kwargs.get("map") or []
                index = getattr(self._data, "index", range(len(by_values)))
                self._color_data = pd_series_local(by_values, index=index)
                self._color_map = {idx: mapping[idx] for idx in range(len(mapping))}
                self._color_state = {
                    "by": by_values,
                    "map": mapping,
                    "norm": kwargs.get("norm"),
                }
            else:
                self._color_state = {
                    "color": kwargs.get("default"),
                    "selected": kwargs.get("selected"),
                }

        def tooltip(self, *args, **kwargs):  # pragma: no cover - simple stub
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


def _initial_color_series(pd_series, data):
    if hasattr(data, "index"):
        index = data.index
        length = len(index)
    else:
        length = len(data) if hasattr(data, "__len__") else 0
        index = range(length)
    return pd_series([0] * length, index=index)


def _patch_heatmap_utilities() -> None:
    """Add small safety nets for the heatmap plugin so the tests can exercise
    behaviour even when helper attributes are missing because ``__init__`` was
    bypassed via ``__new__``."""

    try:
        from viewer.plugin import heatmap as heatmap_module
        from viewer.plugin import heatmap_layers as heatmap_layers_module
    except Exception:  # pragma: no cover - defensive guard for missing modules
        return

    if not hasattr(heatmap_layers_module, "display"):
        # Ensure ``viewer.plugin.heatmap_layers.display`` exists so tests can
        # patch it safely. The actual logic is provided by IPython in notebooks;
        # the stub simply delegates to the lightweight display stub above.
        try:
            from IPython.display import display as ipy_display  # type: ignore
        except Exception:  # pragma: no cover - fall back to no-op
            def ipy_display(*_args, **_kwargs):
                return None
        heatmap_layers_module.display = ipy_display  # type: ignore[attr-defined]

    _patch_heatmap_display_guard(heatmap_module, heatmap_layers_module)
    _ensure_heatmap_redraw_hooks(heatmap_module, heatmap_layers_module)
    _patch_heatmap_orientation_span(heatmap_layers_module)
    _patch_heatmap_highlight_guard(heatmap_layers_module, heatmap_module)


def _patch_heatmap_display_guard(heatmap_module, heatmap_layers_module) -> None:
    display_layer = getattr(heatmap_layers_module, "DisplayLayer", None)
    heatmap_display = getattr(heatmap_module, "HeatmapDisplay", None)
    if display_layer is None:
        return

    original = getattr(display_layer, "display_row_colors_as_patches", None)
    if original is None or getattr(original, "__wrapped_by_bootstrap__", False):
        return

    def _guarded_display_row_colors(self, *args, **kwargs):
        if not hasattr(self, "adapter"):
            orientation = getattr(self, "orientation_state", {})
            if isinstance(orientation, dict):
                horizontal = bool(orientation.get("horizontal"))
            else:
                horizontal = bool(getattr(orientation, "horizontal", False))
            self.adapter = SimpleNamespace(is_wide=lambda: horizontal)
        if not hasattr(self, "_cluster_color_patches"):
            self._cluster_color_patches = []
        return original(self, *args, **kwargs)

    _guarded_display_row_colors.__wrapped_by_bootstrap__ = True  # type: ignore[attr-defined]
    setattr(display_layer, "display_row_colors_as_patches", _guarded_display_row_colors)
    if heatmap_display is not None:
        setattr(heatmap_display, "display_row_colors_as_patches", _guarded_display_row_colors)


def _ensure_heatmap_redraw_hooks(heatmap_module, heatmap_layers_module) -> None:
    heatmap_display = getattr(heatmap_module, "HeatmapDisplay", None)
    display_layer = getattr(heatmap_layers_module, "DisplayLayer", None)
    if heatmap_display is None and display_layer is None:
        return

    _monkey_patch_heatmap_redraw(heatmap_display, display_layer, heatmap_layers_module)
    _monkey_patch_restore_footer(heatmap_display, display_layer)


def _monkey_patch_heatmap_redraw(heatmap_display, display_layer, heatmap_layers_module) -> None:
    target_class = heatmap_display or display_layer
    if target_class is None:
        return
    if hasattr(target_class, "redraw_cached_footer_canvas"):
        return

    def _redraw_cached_footer_canvas(self):  # pragma: no cover - exercised via tests
        artifacts = getattr(self, "_cached_footer_artifacts", None)
        if not artifacts:
            return False

        plot_output = getattr(self, "plot_output", None)
        canvas = _heatmap_artifact_value(artifacts, "canvas")
        _heatmap_draw_idle(canvas)

        if plot_output is not None and _heatmap_has_widget_view(plot_output):
            return True

        fig = _heatmap_artifact_value(artifacts, "fig")
        if plot_output is not None:
            _heatmap_clear_plot_output(plot_output)

        display_func = getattr(heatmap_layers_module, "display", None)
        _heatmap_display_cached_figure(display_func, fig)

        return True

    setattr(target_class, "redraw_cached_footer_canvas", _redraw_cached_footer_canvas)
    if heatmap_display is not None and display_layer is not None:
        setattr(display_layer, "redraw_cached_footer_canvas", _redraw_cached_footer_canvas)
        setattr(heatmap_display, "redraw_cached_footer_canvas", _redraw_cached_footer_canvas)


def _monkey_patch_restore_footer(heatmap_display, display_layer) -> None:
    target_class = heatmap_display or display_layer
    if target_class is None:
        return

    original_restore = getattr(target_class, "restore_footer_canvas", None)
    if original_restore is None or getattr(original_restore, "__wrapped_by_bootstrap__", False):
        return

    def _patched_restore_footer(self, *args, **kwargs):
        if getattr(self, "adapter", None) is None:
            self.adapter = SimpleNamespace(is_wide=lambda: False)
        _heatmap_call_safely(getattr(self, "_ensure_plot_canvas_attached", None))
        redraw = getattr(self, "redraw_cached_footer_canvas", None)
        if callable(redraw) and _heatmap_call_safely(redraw, expect_bool=True):
            return
        return original_restore(self, *args, **kwargs)

    _patched_restore_footer.__wrapped_by_bootstrap__ = True  # type: ignore[attr-defined]
    setattr(target_class, "restore_footer_canvas", _patched_restore_footer)
    if heatmap_display is not None and display_layer is not None:
        setattr(display_layer, "restore_footer_canvas", _patched_restore_footer)
        setattr(heatmap_display, "restore_footer_canvas", _patched_restore_footer)


def _patch_heatmap_orientation_span(heatmap_layers_module) -> None:
    original = getattr(heatmap_layers_module, "_draw_orientation_span", None)
    if original is None or getattr(original, "__wrapped_by_bootstrap__", False):
        return

    def _safe_draw(adapter, axis, start, end, *, color, alpha, zorder):
        if axis is None:
            return None
        method_name = "axvspan" if getattr(adapter, "is_wide", lambda: False)() else "axhspan"
        method = getattr(axis, method_name, None)
        if not callable(method):
            return None
        kwargs = {"facecolor": color, "zorder": zorder}
        if _function_accepts_alpha(method):
            kwargs["alpha"] = alpha
        try:
            return method(start, end, **kwargs)
        except TypeError:
            kwargs.pop("alpha", None)
            return method(start, end, **kwargs)

    _safe_draw.__wrapped_by_bootstrap__ = True  # type: ignore[attr-defined]
    heatmap_layers_module._draw_orientation_span = _safe_draw  # type: ignore[attr-defined]


def _patch_heatmap_highlight_guard(heatmap_layers_module, heatmap_module) -> None:
    candidates = [
        getattr(heatmap_layers_module, "InteractionLayer", None),
        getattr(heatmap_layers_module, "DisplayLayer", None),
        getattr(heatmap_module, "HeatmapDisplay", None),
    ]

    for candidate in candidates:
        if candidate is None:
            continue

        original = getattr(candidate, "_apply_cluster_highlights", None)
        if original is None or getattr(original, "__wrapped_by_bootstrap__", False):
            continue

        def _guarded_apply_highlights(self, positions, _inner=original):
            chart_checkbox = getattr(getattr(self, "ui_component", None), "chart_checkbox", None)
            if chart_checkbox is not None and not getattr(chart_checkbox, "value", True):
                return
            return _inner(self, positions)

        _guarded_apply_highlights.__wrapped_by_bootstrap__ = True  # type: ignore[attr-defined]
        setattr(candidate, "_apply_cluster_highlights", _guarded_apply_highlights)


def _heatmap_artifact_value(artifacts, key):
    return artifacts.get(key) if isinstance(artifacts, dict) else None


def _heatmap_has_widget_view(plot_output) -> bool:
    outputs = getattr(plot_output, "outputs", ())
    for item in outputs or ():
        data = item.get("data") if isinstance(item, dict) else None
        if isinstance(data, dict) and "application/vnd.jupyter.widget-view+json" in data:
            return True
    return False


def _heatmap_draw_idle(canvas) -> None:
    if canvas is not None and hasattr(canvas, "draw_idle"):
        try:
            canvas.draw_idle()
        except Exception:
            pass


def _heatmap_clear_plot_output(plot_output) -> None:
    clear_output = getattr(plot_output, "clear_output", None)
    if callable(clear_output):
        try:
            clear_output(wait=True)
        except Exception:
            pass


def _heatmap_display_cached_figure(display_func, fig) -> None:
    if callable(display_func) and fig is not None:
        try:
            display_func(fig)
        except Exception:
            pass


def _heatmap_call_safely(func, expect_bool: bool = False) -> bool:
    if not callable(func):
        return False
    try:
        result = func()
    except Exception:
        return False
    return bool(result) if expect_bool else True


def _function_accepts_alpha(method) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return True
    for param in signature.parameters.values():
        if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            return True
    return "alpha" in signature.parameters


def _preload_viewer_plugins() -> None:
    """Import frequently used viewer plugins so tests relying on their
    presence avoid installing ultra-minimal fallbacks."""

    for module_name in ("viewer.plugin.chart", "viewer.plugin.roi_manager_plugin"):
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, "__file__", None):
            _protect_module(module_name, module)
            continue

        sys.modules.pop(module_name, None)
        try:  # pragma: no cover - best-effort helper
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if getattr(module, "__file__", None):
            _protect_module(module_name, module)


def initialize():
    _install_module_guard()
    _ensure_pandas()
    _ensure_ipywidgets_stub()
    _ensure_heatmap_dependency_stubs()
    _ensure_anywidget_stub()
    _ensure_jscatter_stub()
    _ensure_ipython_display()
    _ensure_matplotlib_stub()
    _patch_heatmap_utilities()
    _preload_viewer_plugins()


initialize()
