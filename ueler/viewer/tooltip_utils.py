"""Helpers for formatting tooltip values in the viewer."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Optional

import pandas as pd

__all__ = ["format_tooltip_value", "resolve_cell_record"]


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


def resolve_cell_record(
    cell_table: Any,
    *,
    fov_value: Any,
    mask_name: Optional[str],
    mask_id: Any,
    fov_key: str,
    label_key: str,
    mask_key: Optional[str] = None,
) -> Optional[pd.Series]:
    """Return the first matching cell-table row for a hover tooltip.

    Parameters
    ----------
    cell_table:
        Source table storing per-cell data. Expected to behave like a pandas
        :class:`~pandas.DataFrame`.
    fov_value:
        The current field of view identifier.
    mask_name:
        The logical mask name for the label currently hovered (``None`` when
        unknown).
    mask_id:
        The integer label extracted from the mask raster.
    fov_key, label_key:
        Column names to use when filtering the table for the active FOV and
        cell label.
    mask_key:
        Optional column name that distinguishes multiple mask sources.

    Returns
    -------
    pandas.Series or ``None``
        The first matching row when present; ``None`` if the table is missing
        required columns or no row satisfies the filter.
    """

    if cell_table is None:
        return None

    if not hasattr(cell_table, "loc"):
        return None

    df = cell_table

    try:
        candidate = df
        if fov_key not in candidate.columns or label_key not in candidate.columns:
            return None
    except AttributeError:
        return None

    mask_filter = candidate[label_key] == mask_id

    if mask_key and mask_key in candidate.columns and mask_name is not None:
        mask_filter = mask_filter & (candidate[mask_key] == mask_name)

    if fov_value is not None and fov_key in candidate.columns:
        mask_filter = mask_filter & (candidate[fov_key] == fov_value)

    matches = candidate.loc[mask_filter]
    if matches.empty:
        return None

    return matches.iloc[0]
