"""Annotation display helpers exposed via the packaged viewer namespace."""
from __future__ import annotations

from collections import OrderedDict

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.backend_bases import MouseButton
from matplotlib.text import Annotation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage

try:  # pragma: no cover - optional dependency during fast tests
    import pandas as pd  # type: ignore[import-error]  # noqa: F401
except Exception:  # pragma: no cover - optional dependency missing in fast tests
    pd = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency during fast tests
    import seaborn as sns  # type: ignore[import-error]  # noqa: F401
except Exception:  # pragma: no cover - optional dependency missing in fast tests
    sns = None  # type: ignore[assignment]


def _require_widgets():
    try:
        import ipywidgets as widgets  # type: ignore
    except Exception as exc:  # pragma: no cover - widget dependency missing
        raise ImportError("ipywidgets is required for AnnotationDisplay") from exc
    return widgets

from ueler.image_utils import color_one_image, estimate_color_range, process_single_crop
from ueler.viewer.observable import Observable

__all__ = ["AnnotationDisplay", "UiComponent", "Data", "linked_controls"]


class AnnotationDisplay:
    def __init__(self, main_viewer, width, height):
        widgets = _require_widgets()
        self._checkbox_cls = getattr(widgets, "Checkbox")
        self._layout_cls = getattr(widgets, "Layout")
        self._vbox_cls = getattr(widgets, "VBox")

        setattr(self, "SidePlots_id", "annotation_display_output")
        self.displayed_name = "Cell tooltip label"
        self.main_viewer = main_viewer
        self.width = width
        self.height = height

        self.ui_component = UiComponent()
        self.data = Data()

        # Get columns that are of integer or string types
        if self.main_viewer.cell_table is not None:
            cell_table = self.main_viewer.cell_table
            label_columns = cell_table.select_dtypes(include=["int", "int64", "object"]).columns.tolist()

            # Create checkboxes for each label column
            self.ui_component.label_checkboxes = []
            for col in label_columns:
                checkbox = self._checkbox_cls(value=False, description=col, disabled=False)
                self.ui_component.label_checkboxes.append(checkbox)
                checkbox.observe(self.on_label_checkbox_change, names="value")
        else:
            self.ui_component.label_checkboxes = []

        self.initiate_ui()

    def initiate_ui(self):
        # Create a VBox for label checkboxes
        label_selection_box = self._vbox_cls(
            self.ui_component.label_checkboxes, layout=self._layout_cls(flex_flow="column wrap")
        )

        # Update the UI to include the label selection box
        self.ui = self._vbox_cls(
            [label_selection_box], layout=self._layout_cls(max_height=f"{self.height}in")
        )

    def on_label_checkbox_change(self, change):
        # Update the list of selected labels
        selected_labels = [cb.description for cb in self.ui_component.label_checkboxes if cb.value]
        self.main_viewer.selected_tooltip_labels = selected_labels


class UiComponent:
    """Simple namespace for widgets attached dynamically during runtime."""

    def __init__(self):
        # Attributes are set after instantiation to mirror the legacy behaviour.
        pass


class Data:
    """Container for data attributes filled in by the main viewer."""

    def __init__(self):
        # Legacy module expects to assign fields dynamically post-construction.
        pass


class linked_controls:
    """Backwards-compatible placeholder for linked control state."""

    def __init__(self):
        # Kept for compatibility with callers that mutate attributes directly.
        pass
