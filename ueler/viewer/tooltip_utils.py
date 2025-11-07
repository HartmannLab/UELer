"""Helpers for formatting tooltip values in the viewer."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

__all__ = ["format_tooltip_value"]


def format_tooltip_value(value: Any) -> str:
    """Format marker values for tooltips with a scientific-notation fallback.

    Tiny non-zero values that would round to 0.00 under fixed-point formatting are
    displayed in scientific notation instead so near-zero intensities remain
    visible. Non-numeric values are coerced to ``str`` without additional
    formatting.
    """

    if isinstance(value, Real):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value)

        if math.isnan(numeric_value):
            return "nan"
        if math.isinf(numeric_value):
            return "inf" if numeric_value > 0 else "-inf"

        formatted = f"{numeric_value:.2f}"
        if formatted in ("0.00", "-0.00") and not math.isclose(numeric_value, 0.0):
            return f"{numeric_value:.2e}"
        return formatted

    return str(value)
