"""Shared layout helpers for ipywidgets containers.

These helpers return new ``Layout`` instances tuned for common UELer UI
patterns, keeping child widths within their parent boxes to avoid the
shallow horizontal scrollbars reported in issue #39.
"""

from __future__ import annotations

from ipywidgets import Layout


def column_block_layout(**overrides) -> Layout:
    """Layout that fills the parent width without exceeding its bounds."""
    base = {
        "width": "98%",
        "box_sizing": "border-box",
    }
    base.update(overrides)
    return Layout(**base)


def flex_fill_layout(**overrides) -> Layout:
    """Layout that flexes within rows while leaving room for siblings."""
    base = {
        "flex": "1 1 0%",
        "min_width": "0",
        "width": "auto",
    }
    base.update(overrides)
    return Layout(**base)


# Backwards-compatible helpers expected by the legacy UI module
def constrained_column(*, max_width: str = "100%", gap: str | None = None, **overrides) -> Layout:
    """Return a column layout constrained to the given max width.

    Accepts optional `gap` and any additional Layout overrides.
    """
    props = {
        "width": "100%",
        "max_width": max_width,
        "box_sizing": "border-box",
    }
    if gap is not None:
        props["gap"] = gap
    props.update(overrides)
    return Layout(**props)


def scroll_column(*, gap: str | None = None, padding: str | None = None, max_width: str | None = None, **overrides) -> Layout:
    """Return a vertically scrollable column layout."""
    props = {
        "width": "100%",
        "overflow_y": "auto",
        "box_sizing": "border-box",
    }
    if gap is not None:
        props["gap"] = gap
    if padding is not None:
        props["padding"] = padding
    if max_width is not None:
        props["max_width"] = max_width
    props.update(overrides)
    return Layout(**props)


def fixed_panel(*, max_width: str = "360px", gap: str | None = None, **overrides) -> Layout:
    """Return a fixed-width panel layout for sidebars/control columns."""
    props = {
        "width": max_width,
        "flex": "0 0 auto",
        "box_sizing": "border-box",
    }
    if gap is not None:
        props["gap"] = gap
    props.update(overrides)
    return Layout(**props)


def flex_row(*, gap: str | None = None, align: str | None = None, **overrides) -> Layout:
    """Return a row layout using flexbox with optional gap and alignment."""
    props = {
        "display": "flex",
        "width": "100%",
        "box_sizing": "border-box",
    }
    if gap is not None:
        props["gap"] = gap
    if align is not None:
        # map 'align' to align_items
        props["align_items"] = align
    props.update(overrides)
    return Layout(**props)


def flex_item(*, flex: str = "1 1 0%", min_width: str = "0", **overrides) -> Layout:
    """Return a layout suitable for a flex child that can grow/shrink."""
    props = {"flex": flex, "min_width": min_width}
    props.update(overrides)
    return Layout(**props)
