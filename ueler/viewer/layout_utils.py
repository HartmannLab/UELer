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
