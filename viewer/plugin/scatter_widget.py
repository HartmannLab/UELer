from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from ipywidgets import Layout, Widget

try:
    import anywidget  # noqa: F401
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "UELer requires the 'anywidget' package for jscatter integration. "
        "Install it with either 'pip install anywidget' or, if you are using micromamba, "
        "'micromamba activate <env> && pip install anywidget'. After installing, restart JupyterLab and rerun the viewer."
    ) from exc

from jscatter import Scatter

SelectionListener = Callable[[Set[Union[int, str]], str], None]
HoverListener = Callable[[Optional[Union[int, str]]], None]

_DEFAULT_TOOLBAR = ["pan_zoom", "lasso", "reset", "download"]
_DEFAULT_SELECTED_COLOR = (1, 0, 0, 1)
_DEFAULT_POINT_COLOR = (0.2, 0.4, 0.8, 0.85)


def _normalize_indices(values: Iterable[Union[int, str, np.integer]]) -> Set[Union[int, str]]:
    """Convert incoming selection ids to hashable scalars."""
    normalized: Set[Union[int, str]] = set()
    for value in values:
        if isinstance(value, np.integer):
            normalized.add(int(value))
            continue
        if isinstance(value, (int, str)):
            normalized.add(value)
            continue
        try:
            normalized.add(int(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive path
            normalized.add(value)  # fallback to original representation
    return normalized


@dataclass
class ScatterViewState:
    """Encapsulate the state needed to render and interact with a scatter plot."""

    identifier: str
    title: str
    scatter: Scatter
    height: int
    toolbar: Tuple[str, ...]


class ScatterPlotWidget:
    """Wrapper around :class:`jscatter.Scatter` with convenient callbacks."""

    def __init__(
        self,
        identifier: str,
        data: pd.DataFrame,
        x: str,
        y: str,
        *,
        color: Optional[str] = None,
        point_size: float = 10.0,
        title: Optional[str] = None,
        tooltip_fields: Optional[Sequence[str]] = None,
        height: int = 320,
        toolbar: Optional[Sequence[str]] = None,
    ) -> None:
        if x not in data.columns or y not in data.columns:
            raise KeyError(f"Required columns '{x}' or '{y}' not found in scatter DataFrame")

        self.identifier = identifier
        self._data = data
        self._x = x
        self._y = y
        self._color = color
        self._height = int(height)
        self._toolbar = tuple(toolbar or _DEFAULT_TOOLBAR)
        self._selection_callbacks: List[SelectionListener] = []
        self._hover_callbacks: List[HoverListener] = []
        self._suspend_selection = False

        tooltip_candidates: List[str] = [x, y]
        if tooltip_fields:
            tooltip_candidates.extend(list(tooltip_fields))
        if color and color not in tooltip_candidates:
            tooltip_candidates.append(color)
        tooltip_properties = list(dict.fromkeys(filter(self._data.__contains__, tooltip_candidates)))

        self._scatter = Scatter(
            x=x,
            y=y,
            data=self._data,
            data_use_index=True,
        )
        self._scatter.axes(axes=True, grid=True, labels=[x, y])
        self._scatter.height(self._height)
        self._scatter.size(default=point_size)
        if color and color in self._data.columns:
            self._scatter.color(by=color, selected=_DEFAULT_SELECTED_COLOR)
        else:
            self._scatter.color(default=_DEFAULT_POINT_COLOR, selected=_DEFAULT_SELECTED_COLOR)
        self._scatter.tooltip(True, properties=tooltip_properties)
        self._jwidget = self._scatter.widget
        self._jwidget.layout = Layout(width="100%", height=f"{self._height}px")
        self._jwidget.mouse_mode = "panZoom"

        self._selection_handler = self._create_selection_handler()
        self._hover_handler = self._create_hover_handler()
        self._jwidget.observe(self._selection_handler, names="selection")
        self._jwidget.observe(self._hover_handler, names="hovering")

        self.state = ScatterViewState(
            identifier=identifier,
            title=title or f"{y} vs {x}",
            scatter=self._scatter,
            height=self._height,
            toolbar=self._toolbar,
        )

    # ------------------------------------------------------------------
    # Event wiring helpers
    # ------------------------------------------------------------------
    def _create_selection_handler(self):
        def _handler(change):
            if self._suspend_selection:
                return
            raw = self._scatter.selection()
            if raw is None:
                normalized: Set[Union[int, str]] = set()
            elif isinstance(raw, pd.Index):
                normalized = _normalize_indices(raw.tolist())
            else:
                normalized = _normalize_indices(raw)
            self._emit_selection(normalized, origin="widget")

        return _handler

    def _create_hover_handler(self):
        def _handler(change):
            new_value = change.get("new")
            if new_value is None or new_value == -1:
                hovered: Optional[Union[int, str]] = None
            else:
                try:
                    position = int(new_value)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    hovered = None
                else:
                    if 0 <= position < len(self._data.index):
                        hovered = self._data.index[position]
                    else:  # pragma: no cover - defensive guard
                        hovered = None
            for callback in self._hover_callbacks:
                callback(hovered)

        return _handler

    def _emit_selection(self, indices: Set[Union[int, str]], origin: str) -> None:
        for callback in self._selection_callbacks:
            callback(indices, origin)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def scatter(self) -> Scatter:
        return self._scatter

    def widget(self) -> Widget:
        """Return a widget view equipped with the configured toolbar."""
        return self._scatter.show(buttons=list(self.state.toolbar))

    def compose_entry(self) -> Tuple[Scatter, str]:
        """Provide a tuple that can be fed into :func:`jscatter.compose`."""
        return (self._scatter, self.state.title)

    def add_selection_listener(self, callback: SelectionListener) -> None:
        self._selection_callbacks.append(callback)

    def add_hover_listener(self, callback: HoverListener) -> None:
        self._hover_callbacks.append(callback)

    def apply_selection(
        self,
        indices: Iterable[Union[int, str]],
        *,
        announce: bool = True,
    ) -> Set[Union[int, str]]:
        valid = [idx for idx in indices if idx in self._data.index]
        normalized = _normalize_indices(valid)
        self._suspend_selection = True
        try:
            if valid:
                self._scatter.selection(valid)
            else:
                self._scatter.selection(None)
        finally:
            self._suspend_selection = False
        if announce:
            self._emit_selection(normalized, origin="external")
        return normalized

    def clear_selection(self, announce: bool = True) -> None:
        self.apply_selection([], announce=announce)

    def set_point_size(self, size: float) -> None:
        self._scatter.size(default=size)

    def set_mouse_mode(self, mode: str) -> None:
        self._jwidget.mouse_mode = mode

    def set_categorical_colors(
        self,
        categories: pd.Series,
        color_map: Mapping[Union[int, str], Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]],
        *,
        default_color: Union[str, Tuple[float, float, float], Tuple[float, float, float, float], None] = None,
    ) -> None:
        if categories is None:
            return
        if not isinstance(categories, pd.Series):
            categories = pd.Series(categories)
        if categories.empty:
            return
        aligned = categories.reindex(self._data.index)
        if aligned.isna().all():
            return
        current_colors = self._scatter.color()
        if default_color is None:
            default_color = current_colors.get("color", _DEFAULT_POINT_COLOR)
        sentinel = "__missing__"
        normalized_map = {}
        for key, value in color_map.items():
            if isinstance(value, np.ndarray):
                normalized_map[key] = tuple(float(part) for part in value.tolist())
            elif isinstance(value, (list, tuple)):
                normalized_map[key] = tuple(float(part) for part in value)
            else:
                normalized_map[key] = value
        while sentinel in normalized_map:
            sentinel += "_"
        if isinstance(default_color, np.ndarray):
            default_color = tuple(float(part) for part in default_color.tolist())
        elif isinstance(default_color, (list, tuple)):
            default_color = tuple(float(part) for part in default_color)
        normalized_map[sentinel] = default_color
        working = aligned.astype(object)
        working.loc[working.isna()] = sentinel
        categories_order = list(normalized_map.keys())
        colors = list(normalized_map.values())
        encoding = {category: idx for idx, category in enumerate(categories_order)}
        encoded_values = working.map(encoding)
        if encoded_values.isna().any():
            return
        self._scatter.color(by=encoded_values.tolist(), map=colors, norm=(0, len(colors) - 1))

    def dispose(self) -> None:
        self._jwidget.unobserve(self._selection_handler, names="selection")
        self._jwidget.unobserve(self._hover_handler, names="hovering")
        self._selection_callbacks.clear()
        self._hover_callbacks.clear()

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def columns(self) -> Tuple[str, str]:
        return self._x, self._y

    @property
    def color_column(self) -> Optional[str]:
        return self._color
