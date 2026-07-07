"""Shared helpers for the Scatter-plot and Histogram plugins (issue #112).

This module is intentionally prefixed with ``_`` so the plugin auto-loader in
``main_viewer.dynamically_load_plugins`` (which skips files starting with ``_``)
does not treat it as a plugin.  It holds the data-preparation, subset-control,
link-checkbox, and viewer-highlight logic that was previously private to the
combined ``ChartDisplay`` and is now reused by both plugins so their behaviour
stays consistent.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence, Set, Union

import numpy as np
import pandas as pd

import ipywidgets as _ipywidgets

Checkbox = getattr(_ipywidgets, "Checkbox")
Dropdown = getattr(_ipywidgets, "Dropdown")
Layout = getattr(_ipywidgets, "Layout")
SelectMultiple = getattr(_ipywidgets, "SelectMultiple")

_logger = logging.getLogger(__name__)


def prepare_dataframe(
    viewer,
    *,
    subset_on,
    subset_values: Sequence,
    impose_fov: bool,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return the cell table filtered by the subset/current-FOV controls.

    Mirrors the previous ``ChartDisplay._prepare_dataframe`` so both the
    scatter and histogram plugins filter identically.
    """
    cell_table = viewer.cell_table.copy()
    subset_values = list(subset_values) if subset_values else []
    if subset_on and subset_values:
        if subset_on not in cell_table.columns:
            raise KeyError(f"Subset column '{subset_on}' not found in cell table.")
        cell_table = cell_table[cell_table[subset_on].isin(subset_values)]
    if impose_fov:
        current_fov = viewer.ui_component.image_selector.value
        cell_table = cell_table[cell_table[viewer.fov_key] == current_fov]
    columns = [col for col in columns if col in cell_table.columns]
    if columns:
        cell_table = cell_table.dropna(subset=columns)
    return cell_table


def build_subset_controls(viewer):
    """Create the ``(subset_on_dropdown, subset_selector, impose_fov_checkbox)`` widgets."""
    subset_columns = [
        col
        for col in viewer.cell_table.columns
        if pd.api.types.is_numeric_dtype(viewer.cell_table[col])
        or pd.api.types.is_object_dtype(viewer.cell_table[col])
    ]
    subset_on_dropdown = Dropdown(
        options=subset_columns,
        description="Subset on:",
        style={"description_width": "auto"},
        layout=Layout(width="100%"),
    )
    subset_selector = SelectMultiple(
        options=[],
        description="Subset:",
        style={"description_width": "auto"},
        layout=Layout(width="100%"),
    )
    impose_fov_checkbox = Checkbox(
        value=False,
        description="Current FOV",
        style={"description_width": "auto"},
        layout=Layout(width="140px"),
    )
    return subset_on_dropdown, subset_selector, impose_fov_checkbox


def build_link_checkboxes():
    """Create the ``(mv_linked_checkbox, cell_gallery_linked_checkbox)`` widgets."""
    mv_linked_checkbox = Checkbox(
        value=False,
        description="Main viewer",
        style={"description_width": "auto"},
    )
    cell_gallery_linked_checkbox = Checkbox(
        value=False,
        description="Cell gallery",
        style={"description_width": "auto"},
    )
    return mv_linked_checkbox, cell_gallery_linked_checkbox


def subset_options_for(viewer, selected_column) -> list:
    """Return the sorted unique values of ``selected_column`` (for the subset selector)."""
    if not selected_column or selected_column not in viewer.cell_table.columns:
        return []
    unique_values = viewer.cell_table[selected_column].dropna().unique().tolist()
    return sorted(unique_values)


def sync_mask_highlights_from_selection(
    viewer, indices: Set[Union[int, str]]
) -> None:
    """Translate a set of cell-table row indices into mask highlights in the viewer.

    Works in both single-FOV and map mode.  Extracted verbatim from the previous
    ``ChartDisplay._sync_mask_highlights_from_selection``.
    """
    try:
        image_display = getattr(viewer, "image_display", None)
        if image_display is None:
            return
        mask_key = getattr(viewer, "mask_key", None)
        if not mask_key:
            return

        if not indices:
            image_display.set_mask_ids(mask_name=mask_key, mask_ids=[])
            return

        cell_table = viewer.cell_table
        fov_col = viewer.fov_key
        lbl_col = viewer.label_key

        valid_indices = [idx for idx in indices if idx in cell_table.index]
        if not valid_indices:
            return

        active_fov = viewer.get_active_fov()
        if active_fov:
            # Single-FOV mode: highlight only cells in the active FOV.
            rows = cell_table.loc[valid_indices, [fov_col, lbl_col]]
            mask_ids = (
                rows.loc[rows[fov_col] == active_fov, lbl_col].astype(int).tolist()
            )
            image_display.set_mask_ids(mask_name=mask_key, mask_ids=mask_ids)
        else:
            # Map mode: pass explicit (fov, mask_id) pairs so each selection is
            # correctly routed to its tile viewport.
            rows = cell_table.loc[valid_indices, [fov_col, lbl_col]]
            fov_mask_pairs = list(
                zip(rows[fov_col].astype(str), rows[lbl_col].astype(int))
            )
            image_display.set_mask_ids(
                mask_name=mask_key, mask_ids=[], fov_mask_pairs=fov_mask_pairs
            )
    except Exception:
        if getattr(viewer, "_debug", False):
            import traceback

            traceback.print_exc()


def normalize_indices(indices: Iterable[Union[int, str]]) -> Set[Union[int, str]]:
    """Coerce numpy integers to plain ints while leaving other ids untouched."""
    return {int(idx) if isinstance(idx, np.integer) else idx for idx in indices}


__all__ = [
    "prepare_dataframe",
    "build_subset_controls",
    "build_link_checkboxes",
    "subset_options_for",
    "sync_mask_highlights_from_selection",
    "normalize_indices",
]
