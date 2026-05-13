"""Plugin for writing string values into cell table columns."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ipywidgets import (
    Button,
    Combobox,
    HTML,
    HBox,
    Layout,
    Text,
    VBox,
)

from .plugin_base import PluginBase
from ..layout_utils import column_block_layout, content_widget_layout


class CellTableEditorPlugin(PluginBase):
    """Assign a string value to a cell table column for all selected cells."""

    def __init__(self, main_viewer, width: int = 6, height: int = 3) -> None:
        super().__init__(main_viewer, width, height)
        self.displayed_name = "Cell Table Editor"
        self.SidePlots_id = "cell_table_editor_output"
        self.main_viewer = main_viewer
        self.ui_component = SimpleNamespace()

        self._build_widgets()
        self._build_layout()

        self.setup_widget_observers()
        self.initialized = True

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        style_auto = {"description_width": "auto"}
        full_width = Layout(width="calc(100% - 5px)", max_width="calc(100% - 5px)")

        self.ui_component.column_combo = Combobox(
            options=self._get_column_options(),
            value="",
            placeholder="Column name (new or existing)",
            description="Column:",
            ensure_option=False,
            layout=full_width,
            style=style_auto,
        )

        self.ui_component.value_input = Text(
            value="",
            placeholder="String value to assign",
            description="Value:",
            layout=full_width,
            style=style_auto,
        )

        self.ui_component.apply_btn = Button(
            description="Apply to selected cells",
            button_style="primary",
            disabled=True,
            tooltip="Write the value to the specified column for all selected cells",
            icon="pencil",
            layout=Layout(width="auto"),
        )
        self.ui_component.apply_btn.on_click(self._on_apply_clicked)

        self.ui_component.status_label = HTML(value="<i>No cells selected.</i>")

    def _build_layout(self) -> None:
        self.ui = VBox(
            [
                self.ui_component.column_combo,
                self.ui_component.value_input,
                self.ui_component.apply_btn,
                self.ui_component.status_label,
            ],
            layout=Layout(width="100%", gap="6px", padding="4px"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_column_options(self) -> list[str]:
        ct = getattr(self.main_viewer, "cell_table", None)
        if ct is None:
            return []
        system_keys = {
            getattr(self.main_viewer, "fov_key", "fov"),
            getattr(self.main_viewer, "x_key", "X"),
            getattr(self.main_viewer, "y_key", "Y"),
            getattr(self.main_viewer, "label_key", "label"),
            getattr(self.main_viewer, "mask_key", "mask"),
        }
        return [c for c in ct.columns if c not in system_keys]

    def _get_selected_masks(self):
        image_display = getattr(self.main_viewer, "image_display", None)
        if image_display is None:
            return set()
        return getattr(image_display, "selected_masks_label", set())

    def _refresh_apply_btn(self) -> None:
        has_selection = bool(self._get_selected_masks())
        self.ui_component.apply_btn.disabled = not has_selection
        if not has_selection:
            self.ui_component.status_label.value = "<i>No cells selected.</i>"

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_apply_clicked(self, _button) -> None:
        selections = self._get_selected_masks()
        if not selections:
            self.ui_component.status_label.value = (
                "<span style='color:#c62828'>No cells selected.</span>"
            )
            return

        ct = getattr(self.main_viewer, "cell_table", None)
        if ct is None:
            self.ui_component.status_label.value = (
                "<span style='color:#c62828'>No cell table loaded.</span>"
            )
            return

        column_name = self.ui_component.column_combo.value.strip()
        if not column_name:
            self.ui_component.status_label.value = (
                "<span style='color:#c62828'>Please enter a column name.</span>"
            )
            return

        value = self.ui_component.value_input.value

        fov_key = getattr(self.main_viewer, "fov_key", "fov")
        label_key = getattr(self.main_viewer, "label_key", "label")

        # Initialize new column with empty string
        if column_name not in ct.columns:
            ct[column_name] = ""

        updated = 0
        for sel in selections:
            fov = getattr(sel, "fov", None)
            mask_id = getattr(sel, "mask_id", None)
            if fov is None or mask_id is None:
                continue
            try:
                # Cast mask_id to match the column dtype to avoid type mismatches
                col_dtype = ct[label_key].dtype
                try:
                    typed_mask_id = col_dtype.type(mask_id)
                except Exception:
                    typed_mask_id = mask_id

                mask = (ct[fov_key] == fov) & (ct[label_key] == typed_mask_id)
                n = mask.sum()
                if n > 0:
                    ct.loc[mask, column_name] = value
                    updated += int(n)
            except Exception:
                continue

        if updated > 0:
            self.main_viewer.inform_plugins("on_cell_table_change")
            # Refresh column options to include the (possibly new) column
            self.ui_component.column_combo.options = self._get_column_options()
            self.ui_component.status_label.value = (
                f"<span style='color:#2e7d32'>Updated {updated} row(s) in "
                f"<b>{column_name}</b>.</span>"
            )
        else:
            self.ui_component.status_label.value = (
                "<span style='color:#f9a825'>No matching rows found in the cell table.</span>"
            )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_cell_table_change(self) -> None:
        self.ui_component.column_combo.options = self._get_column_options()

    def on_fov_change(self) -> None:
        self._refresh_apply_btn()

    def on_map_mode_activate(self) -> None:
        self._refresh_apply_btn()

    def on_map_mode_deactivate(self) -> None:
        self._refresh_apply_btn()

    def on_mv_update_display(self) -> None:
        self._refresh_apply_btn()

    def on_selection_change(self) -> None:
        self._refresh_apply_btn()
