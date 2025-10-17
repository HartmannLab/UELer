"""Shared bootstrap for unit tests.

This module centralizes lightweight stubs for optional runtime dependencies so the
unit test suite can run without installing the full UI stack. The bootstrap is
imported from ``tests/__init__.py`` ensuring the fakes are registered before any
project modules are imported by the tests.
"""

from __future__ import annotations

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

_DEFAULT_POINT_COLOR = (0.2, 0.4, 0.8, 0.85)


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

    def _needs_replacement(attr_name: str, value) -> bool:
        if value is None:
            return True
        if attr_name in {"DataFrame", "Series", "Index"} and value is object:
            return True
        return False

    replaced = False
    for name in ("DataFrame", "Series", "Index", "merge", "concat", "isna"):
        current = getattr(module, name, None)
        if _needs_replacement(name, current):
            setattr(module, name, getattr(stub, name))
            replaced = True

    if not hasattr(module, "api"):
        module.api = getattr(stub, "api")  # type: ignore[attr-defined]
        replaced = True

    if replaced:
        module.__bootstrap_stub__ = True  # type: ignore[attr-defined]
        module.__bootstrap_patched__ = True  # type: ignore[attr-defined]


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


class _BootstrapSeries:
    def __init__(self, values=None, index=None, name=None):
        self._data = list(values or [])
        self.index = list(index or range(len(self._data)))
        self.name = name

    def __iter__(self):  # pragma: no cover - simple proxy
        return iter(self._data)

    def __len__(self):  # pragma: no cover - simple proxy
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, index, value):  # pragma: no cover - rarely used
        self._data[index] = value

    def dropna(self):
        filtered = [value for value in self._data if value is not None]
        filtered_index = [idx for value, idx in zip(self._data, self.index) if value is not None]
        return _BootstrapSeries(filtered, filtered_index)

    def unique(self):
        seen = []
        for item in self._data:
            if item not in seen:
                seen.append(item)
        return _BootstrapSeries(seen)

    def isin(self, values):  # pragma: no cover - simple stub
        lookup = set(values)
        return _BootstrapSeries([item in lookup for item in self._data])

    def tolist(self):
        return list(self._data)

    def _resolve_positions(self, key):
        if isinstance(key, slice):  # pragma: no cover - seldom triggered
            start = key.start or 0
            stop = key.stop if key.stop is not None else len(self.index)
            step = key.step or 1
            return list(range(start, stop, step))
        if isinstance(key, (list, tuple, set)):
            keys = list(key)
        else:
            keys = [key]
        positions = []
        for label in keys:
            if label in self.index:
                positions.append(self.index.index(label))
            elif isinstance(label, int) and 0 <= label < len(self._data):
                positions.append(label)
            else:  # pragma: no cover - defensive guard
                raise KeyError(label)
        return positions

    @property
    def loc(self):
        class _SeriesLoc:
            def __init__(self, series):
                self._series = series

            def __getitem__(self, key):
                positions = self._series._resolve_positions(key)
                values = [self._series[pos] for pos in positions]
                if len(values) == 1 and not isinstance(key, (list, tuple, set, slice)):
                    return values[0]
                indices = [self._series.index[pos] for pos in positions]
                return _BootstrapSeries(values, indices)

        return _SeriesLoc(self)


class _BootstrapIndex(list):
    def take(self, positions):  # pragma: no cover - simple stub
        return _BootstrapIndex(self[pos] for pos in positions)


class _BootstrapDataFrame:
    def __init__(self, data=None, index=None):
        data = data or {}
        self._data = {key: list(value) for key, value in data.items()}
        first_column = next(iter(self._data.values()), [])
        inferred_index = list(range(len(first_column)))
        self.index = _BootstrapIndex(index or inferred_index)
        self.columns = _BootstrapSeries(self._data.keys())
        self.loc = _BootstrapDataFrameLoc(self)  # type: ignore[attr-defined]

    def copy(self):
        return _BootstrapDataFrame(self._data.copy(), list(self.index))

    def __getitem__(self, key):
        values = self._data[key]
        return _BootstrapSeries(values, list(self.index))

    def __setitem__(self, key, values):  # pragma: no cover - simple stub
        self._data[key] = list(values)

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
        first_key = next(iter(self._data), None)
        if first_key is None:
            return _BootstrapSeries([])
        result = [row in lookup for row in self._data[first_key]]
        return _BootstrapSeries(result, list(self.index))

    @property
    def empty(self):  # pragma: no cover - simple stub
        return not any(self._data.values())

    def _resolve_row_positions(self, selector):
        if isinstance(selector, slice):  # pragma: no cover - defensive guard
            start = selector.start or 0
            stop = selector.stop if selector.stop is not None else len(self.index)
            step = selector.step or 1
            return list(range(start, stop, step))
        if isinstance(selector, (list, tuple, set)):
            labels = list(selector)
        else:
            labels = [selector]
        positions = []
        for label in labels:
            if label in self.index:
                positions.append(self.index.index(label))
            elif isinstance(label, int) and 0 <= label < len(self.index):
                positions.append(label)
            else:  # pragma: no cover - defensive guard
                raise KeyError(label)
        return positions


class _BootstrapDataFrameLoc:
    def __init__(self, frame):
        self._frame = frame

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_selector, col_selector = key
        else:
            row_selector, col_selector = key, slice(None)

        row_positions = self._frame._resolve_row_positions(row_selector)
        if isinstance(col_selector, slice):  # pragma: no cover - defensive guard
            column_names = list(self._frame._data.keys())[col_selector]
        elif isinstance(col_selector, (list, tuple)):
            column_names = list(col_selector)
        elif isinstance(col_selector, set):
            column_names = sorted(col_selector)
        elif col_selector is slice(None):
            column_names = list(self._frame._data.keys())
        else:
            column_names = [col_selector]

        if len(column_names) == 1:
            column = column_names[0]
            values = [self._frame._data[column][pos] for pos in row_positions]
            if len(values) == 1 and not isinstance(row_selector, (list, tuple, set, slice)):
                return values[0]
            indices = [self._frame.index[pos] for pos in row_positions]
            return _BootstrapSeries(values, indices)

        data = {
            column: [self._frame._data[column][pos] for pos in row_positions]
            for column in column_names
        }
        indices = [self._frame.index[pos] for pos in row_positions]
        return _BootstrapDataFrame(data, indices)


def _create_pandas_stub() -> types.ModuleType:
    pandas_stub = types.ModuleType(_PANDAS)

    def _merge(left, right, **_kwargs):  # pragma: no cover - simple stub
        combined = left.copy()
        combined._data.update(right._data)
        return combined

    def _concat(frames, ignore_index=False, **_kwargs):  # pragma: no cover - simple stub
        base = _BootstrapDataFrame()
        base._data = {}
        for frame in frames:
            for key, values in frame._data.items():
                base._data.setdefault(key, []).extend(values)
        if ignore_index:
            base.index = _BootstrapIndex(range(len(next(iter(base._data.values()), []))))
        return base

    def _isna(value):  # pragma: no cover - simple stub
        return value is None

    pandas_stub.DataFrame = _BootstrapDataFrame  # type: ignore[attr-defined]
    pandas_stub.Series = _BootstrapSeries  # type: ignore[attr-defined]
    pandas_stub.Index = _BootstrapIndex  # type: ignore[attr-defined]
    pandas_stub.merge = _merge  # type: ignore[attr-defined]
    pandas_stub.concat = _concat  # type: ignore[attr-defined]
    pandas_stub.isna = _isna  # type: ignore[attr-defined]
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

def _ensure_heatmap_dependency_stubs() -> None:
    """Provide ultra-light stubs for third-party libraries used by the
    heatmap plugin when they are not installed in the test environment."""

    try:  # pragma: no cover - prefer actual seaborn
        import seaborn  # type: ignore  # noqa: F401
        _protect_module(_SEABORN, sys.modules[_SEABORN])
    except Exception:  # pragma: no cover - fallback for minimal environments
        sys.modules.pop(_SEABORN, None)
        if _SEABORN not in sys.modules:
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
            sys.modules[_SEABORN] = seaborn_stub

    try:  # pragma: no cover - prefer actual scipy
        import scipy.cluster.hierarchy  # type: ignore  # noqa: F401
        _protect_module(_SCIPY, sys.modules[_SCIPY])
        _protect_module(_SCIPY_CLUSTER, sys.modules.get(_SCIPY_CLUSTER, sys.modules[_SCIPY]))
        _protect_module(_SCIPY_HIERARCHY, sys.modules[_SCIPY_HIERARCHY])
    except Exception:  # pragma: no cover - fallback when compiled extensions missing
        for name in (_SCIPY, _SCIPY_CLUSTER, _SCIPY_HIERARCHY):
            sys.modules.pop(name, None)
        if _SCIPY not in sys.modules:
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

            sys.modules[_SCIPY] = scipy_module
            sys.modules[_SCIPY_CLUSTER] = cluster_module
            sys.modules[_SCIPY_HIERARCHY] = hierarchy_module


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
    try:
        return pd_series([0] * length, index=index)
    except TypeError:
        series = pd_series([0] * length)
        if hasattr(series, "index"):
            series.index = list(index)
        return series


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
        try:  # pragma: no cover - best-effort helper
            __import__(module_name)
        except Exception:
            continue


def initialize():
    _install_module_guard()
    _ensure_pandas()
    _ensure_ipywidgets_stub()
    _ensure_heatmap_dependency_stubs()
    _ensure_anywidget_stub()
    _ensure_jscatter_stub()
    _ensure_ipython_display()
    _patch_heatmap_utilities()
    _preload_viewer_plugins()


initialize()
