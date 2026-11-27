"""How a cell selection looks, and the one routine that draws it into an image.

A selected cell is drawn in the selection colour with an outline thicker than a
mask outline and a light tint of the same colour inside, so it stays
recognisable even when the masks are drawn in the selection colour too. The
image (single FOV, map mode, channel grid) draws through
:func:`paint_selection`; the scatter, histogram and Gallery read the colour from
the same :class:`SelectionStyle`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from matplotlib.colors import to_hex, to_rgb

from ueler.constants import (
    PLOT_BASE_COLOR,
    SELECTION_COLOR,
    SELECTION_FILL_OPACITY,
    SELECTION_OUTLINE_WIDTH,
)
from ueler.rendering.engine import scale_outline_thickness, thicken_outline
from skimage.segmentation import find_boundaries


def _hex(value, fallback: str) -> str:
    try:
        return to_hex(to_rgb(value))
    except (TypeError, ValueError):
        return fallback


@dataclass(frozen=True)
class SelectionStyle:
    """The selection colour, outline width (px at full resolution) and fill opacity."""

    color: str = SELECTION_COLOR
    outline_width: int = SELECTION_OUTLINE_WIDTH
    fill_opacity: float = SELECTION_FILL_OPACITY
    base_color: str = PLOT_BASE_COLOR

    @classmethod
    def build(cls, color=None, outline_width=None, fill_opacity=None, base_color=None) -> "SelectionStyle":
        """A style from loose widget values; anything unusable falls back to the default."""
        try:
            width = max(1, int(outline_width))
        except (TypeError, ValueError):
            width = SELECTION_OUTLINE_WIDTH
        try:
            opacity = min(1.0, max(0.0, float(fill_opacity)))
        except (TypeError, ValueError):
            opacity = SELECTION_FILL_OPACITY
        return cls(
            color=_hex(color, SELECTION_COLOR),
            outline_width=width,
            fill_opacity=opacity,
            base_color=_hex(base_color, PLOT_BASE_COLOR),
        )

    @property
    def rgb(self) -> Tuple[float, float, float]:
        return tuple(float(c) for c in to_rgb(self.color))  # type: ignore[return-value]

    @property
    def base_rgb(self) -> Tuple[float, float, float]:
        return tuple(float(c) for c in to_rgb(self.base_color))  # type: ignore[return-value]


def selection_edges(labels: np.ndarray, outline_width: int, downsample_factor: int = 1) -> np.ndarray:
    """The outline pixels of the non-zero cells in *labels*, thickened to the style's width.

    The width is given at full resolution and scaled with the downsample factor,
    like mask outlines. *labels* is usually a crop of the view, so a cell that
    fills it has no edge in it; the fill tint then still marks it.
    """
    edges = find_boundaries(labels, mode="inner") & (labels != 0)
    width = scale_outline_thickness(outline_width, downsample_factor)
    if width > 1:
        edges = thicken_outline(edges, width - 1)
    return edges


def paint_selection(
    image: np.ndarray,
    labels: Optional[np.ndarray],
    style: SelectionStyle,
    *,
    downsample_factor: int = 1,
    offset: Tuple[int, int] = (0, 0),
) -> bool:
    """Draw the cells of *labels* into *image* in place: a tint inside, the outline on top.

    *labels* holds cell ids (non-zero = selected). Its pixel ``(r, c)`` lands on
    ``image[r + offset[0], c + offset[1]]``; whatever falls outside *image* is
    clipped. Integer images are taken as 0–255, float images as 0–1, and an
    alpha channel is left alone. Returns whether anything was drawn.
    """
    if labels is None:
        return False
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.size == 0 or not np.any(labels):
        return False
    if image.ndim != 3 or image.shape[2] < 3:
        return False

    r0, c0 = int(offset[0]), int(offset[1])
    height, width = image.shape[:2]
    rows = slice(max(0, r0), min(height, r0 + labels.shape[0]))
    cols = slice(max(0, c0), min(width, c0 + labels.shape[1]))
    if rows.start >= rows.stop or cols.start >= cols.stop:
        return False

    edges_full = selection_edges(labels, style.outline_width, downsample_factor)
    sub = (slice(rows.start - r0, rows.stop - r0), slice(cols.start - c0, cols.stop - c0))
    edges = edges_full[sub]
    fill = (labels[sub] != 0) & ~edges
    if not (np.any(edges) or np.any(fill)):
        return False

    view = image[rows, cols, :3]
    scale = 255.0 if np.issubdtype(image.dtype, np.integer) else 1.0
    colour = np.asarray(style.rgb, dtype=np.float64) * scale
    opacity = style.fill_opacity
    if opacity > 0 and np.any(fill):
        blended = view[fill].astype(np.float64) * (1.0 - opacity) + colour * opacity
        view[fill] = blended.astype(image.dtype) if scale == 255.0 else blended
    view[edges] = colour.astype(image.dtype) if scale == 255.0 else colour
    return True


def style_of(viewer) -> SelectionStyle:
    """The viewer's selection style, or the default one for a viewer without it (a test stub)."""
    style = getattr(viewer, "selection_style", None)
    return style if isinstance(style, SelectionStyle) else SelectionStyle()
