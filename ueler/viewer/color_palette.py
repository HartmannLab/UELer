"""Small color palette widget for selecting colors.

This provides a compact, dependency-light ipywidgets widget that displays a
grid of color swatches and exposes a simple selection API. It is intentionally
minimal so it can be used in plugins like the ROI manager or annotation
editor.
"""
from typing import Callable, Iterable, List, Optional

from ipywidgets import Box, Button, GridBox, Layout


# Default palette requested by the user
DEFAULT_COLORS = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]


class ColorPalette(Box):
    """A simple color-palette selector.

    Parameters
    - colors: iterable of color strings (hex or css color)
    - multi: allow multiple selection when True
    - columns: number of columns in the swatch grid
    - swatch_size: CSS size for each swatch (e.g. '28px')
    """

    # Internally-held selection value. Exposed via the `value` property.
    _value = None

    def __init__(self,
                 colors: Optional[Iterable[str]] = None,
                 multi: bool = False,
                 columns: int = 8,
                 swatch_size: str = "28px",
                 **kwargs):
        super().__init__(**kwargs)
        self.multi = bool(multi)
        self.columns = max(1, int(columns))
        self.swatch_size = swatch_size
        self.colors: List[str] = list(colors) if colors is not None else list(DEFAULT_COLORS)
        self._selected = [] if self.multi else None
        self._callbacks: List[Callable] = []

        # Build swatches
        self._buttons = []
        self._grid = GridBox(layout=Layout(grid_template_columns=f"repeat({self.columns}, {self.swatch_size})", grid_gap='6px'))
        self._rebuild_grid()

        # Use a single child Box so the widget can be placed anywhere
        self.children = (self._grid,)

    def _make_button(self, color: str) -> Button:
        btn = Button(layout=Layout(width=self.swatch_size, height=self.swatch_size, padding='0px', margin='0px'), tooltip=color)
        # Use style.button_color where available; fallback to setting style via layout
        try:
            btn.style.button_color = color
        except Exception:
            # Some widget themes may not support .style.button_color; ignore
            pass

        # store raw color for click handler
        btn._color = color
        btn.on_click(lambda _, c=color: self._on_swatch_click(c))
        self._apply_button_border(btn, selected=False)
        return btn

    def _apply_button_border(self, btn: Button, selected: bool):
        # Bold border when selected, thin otherwise
        if selected:
            btn.layout.border = '3px solid #444444'
        else:
            btn.layout.border = '1px solid rgba(0,0,0,0.15)'

    def _rebuild_grid(self):
        self._buttons = [self._make_button(c) for c in self.colors]
        self._grid.children = tuple(self._buttons)

    def _on_swatch_click(self, color: str):
        old = self.value
        if self.multi:
            if color in (self._selected or []):
                self._selected.remove(color)
            else:
                self._selected.append(color)
            new = list(self._selected)
            self._value = new
        else:
            if self.value == color:
                self._value = None
            else:
                self._value = color
        self._refresh_button_states()
        for cb in self._callbacks:
            try:
                cb({'old': old, 'new': self.value})
            except Exception:
                # best-effort callbacks
                pass

    def _refresh_button_states(self):
        if self.multi:
            selected = set(self._selected or [])
            for btn in self._buttons:
                self._apply_button_border(btn, btn._color in selected)
        else:
            for btn in self._buttons:
                self._apply_button_border(btn, btn._color == self.value)

    # Public API
    def select(self, color: Optional[str]):
        """Programmatically select a color (or None to clear)."""
        if color is None:
            old = self.value
            if self.multi:
                self._selected = []
                self._value = []
            else:
                self._value = None
            self._refresh_button_states()
            for cb in self._callbacks:
                cb({'old': old, 'new': self.value})
            return
        if color not in self.colors:
            raise ValueError(f"color {color!r} is not in the palette")
        # simulate click behavior
        if self.multi:
            if color in (self._selected or []):
                self._selected.remove(color)
            else:
                self._selected.append(color)
            self._value = list(self._selected)
        else:
            self._value = color
        self._refresh_button_states()

    def on_change(self, callback: Callable[[dict], None]):
        """Register a callback called with a dict {'old':..., 'new':...} on change."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def set_colors(self, colors: Iterable[str]):
        """Replace the palette colors."""
        self.colors = list(colors)
        if self.multi:
            self._selected = []
            self._value = []
        else:
            self._value = None
        self._rebuild_grid()

    def add_color(self, color: str):
        if color in self.colors:
            return
        self.colors.append(color)
        self._rebuild_grid()

    def remove_color(self, color: str):
        if color not in self.colors:
            return
        self.colors.remove(color)
        if self.multi and color in (self._selected or []):
            self._selected.remove(color)
        elif not self.multi and self.value == color:
            self._value = None
        self._rebuild_grid()

    @property
    def value(self):
        """Currently selected color (str) or list[str] for multi-select, or None."""
        return self._value

    @value.setter
    def value(self, new):
        old = getattr(self, '_value', None)
        self._value = new
        # notify callbacks
        if hasattr(self, '_callbacks'):
            for cb in self._callbacks:
                try:
                    cb({'old': old, 'new': new})
                except Exception:
                    pass
