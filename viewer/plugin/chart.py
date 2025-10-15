from __future__ import annotations

import itertools
from collections import OrderedDict
from typing import Iterable, Mapping, Optional, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import (
    Button,
    Checkbox,
    Dropdown,
    FloatSlider,
    HBox,
    HTML,
    IntSlider,
    Layout,
    Output,
    SelectMultiple,
    Tab,
    ToggleButtons,
    VBox,
)
from jscatter import compose

from viewer.decorators import update_status_bar
from viewer.observable import Observable
from viewer.plugin.plugin_base import PluginBase
from viewer.plugin.scatter_widget import ScatterPlotWidget


_SELECTION_NOTICE = (
    "<i>No scatter plots generated yet. Choose axes, then click <b>Plot</b>.</i>"
)


class ChartDisplay(PluginBase):
    def __init__(self, main_viewer, width: float, height: float):
        super().__init__(main_viewer, width, height)
        self.SidePlots_id = "chart_output"
        self.displayed_name = "Chart"
        self.main_viewer = main_viewer
        self.width = width
        self.height = height

        self.point_size = 10.0
        self.histogram_line = None
        self.cutoff: Optional[float] = None

        self.selected_indices: Observable = Observable(set())

        self._scatter_views: "OrderedDict[str, ScatterPlotWidget]" = OrderedDict()
        self._id_counter = itertools.count(1)

        self.ui_component = UiComponent(self.main_viewer)
        self._hist_output = Output(layout=Layout(width="100%"))
        self._plot_placeholder = HTML(
            value=_SELECTION_NOTICE,
            layout=Layout(width="100%"),
        )
        self._plot_host = VBox(
            [self._plot_placeholder], layout=Layout(width="100%", gap="8px")
        )

        self._wide_notice = HTML(
            value=(
                "<b>Multiple scatter plots are active.</b> Controls and plots appear in the footer."
            ),
            layout=Layout(width="100%", padding="8px"),
        )
        self._section_location = "vertical"

        self._wire_events()
        self._build_layout()
        self.setup_widget_observers()

    # ------------------------------------------------------------------
    # UI wiring
    # ------------------------------------------------------------------
    def _wire_events(self) -> None:
        self.ui_component.plot_button.on_click(self.plot_chart)
        self.ui_component.trace_button.on_click(self.trace_cells)
        self.ui_component.point_size_slider.observe(
            self._on_point_size_change, names="value"
        )
        self.ui_component.subset_on_dropdown.observe(
            self.on_subset_on_dropdown_change, names="value"
        )
        self.ui_component.remove_scatter_button.on_click(
            self._remove_selected_scatter
        )
        self.ui_component.clear_scatter_button.on_click(
            self._clear_all_scatter_views
        )
        self.ui_component.clear_selection_button.on_click(
            lambda _btn: self._apply_external_selection(set())
        )
        self.ui_component.scatter_set_selector.observe(
            self._on_scatter_selector_change, names="value"
        )

    def _build_layout(self) -> None:
        histogram_controls = HBox(
            [self.ui_component.bin_slider, self.ui_component.above_below_buttons],
            layout=Layout(gap="12px"),
        )

        scatter_controls = VBox(
            [
                self.ui_component.point_size_slider,
                HBox(
                    [
                        self.ui_component.scatter_set_selector,
                        self.ui_component.remove_scatter_button,
                    ],
                    layout=Layout(gap="8px", align_items="center"),
                ),
                HBox(
                    [
                        self.ui_component.clear_scatter_button,
                        self.ui_component.clear_selection_button,
                    ],
                    layout=Layout(gap="8px"),
                ),
            ],
            layout=Layout(width="100%", gap="8px"),
        )

        subset_controls = VBox(
            [
                self.ui_component.subset_on_dropdown,
                self.ui_component.subset_selector,
                self.ui_component.impose_fov_checkbox,
            ],
            layout=Layout(width="100%", gap="8px"),
        )

        trace_controls = VBox(
            [self.ui_component.trace_button], layout=Layout(width="100%", gap="8px")
        )

        link_controls = VBox(
            [
                self.ui_component.mv_linked_checkbox,
                self.ui_component.cell_gallery_linked_checkbox,
            ],
            layout=Layout(width="100%", gap="8px"),
        )

        self._plot_tabs = Tab(
            children=[
                histogram_controls,
                scatter_controls,
                subset_controls,
                trace_controls,
                link_controls,
            ]
        )
        self._plot_tabs.set_title(0, "Histogram")
        self._plot_tabs.set_title(1, "Scatter plot")
        self._plot_tabs.set_title(2, "Subset")
        self._plot_tabs.set_title(3, "Trace")
        self._plot_tabs.set_title(4, "Linked plugins")

        chart_widgets = VBox(
            [
                HBox(
                    [
                        self.ui_component.x_axis_selector,
                        self.ui_component.y_axis_selector,
                        self.ui_component.color_selector,
                    ],
                    layout=Layout(gap="8px", align_items="center"),
                ),
                self.ui_component.plot_button,
                self._plot_tabs,
            ],
            layout=Layout(width="100%", gap="10px"),
        )

        self.controls_section = VBox(
            [chart_widgets],
            layout=Layout(width="100%", gap="12px"),
        )
        self.plot_section = VBox(
            [self._plot_host],
            layout=Layout(width="100%", flex="1 1 auto"),
        )

        self.ui = VBox(
            [self.controls_section, self.plot_section],
            layout=Layout(width="100%", max_height="600px", gap="12px"),
        )

    # ------------------------------------------------------------------
    # Plotting actions
    # ------------------------------------------------------------------
    @update_status_bar
    def plot_chart(self, _button):
        x_col = self.ui_component.x_axis_selector.value
        y_col = self.ui_component.y_axis_selector.value
        c_col = self.ui_component.color_selector.value

        if x_col == "None" and y_col == "None":
            print("Please select columns for at least the x axis.")
            return

        if y_col == "None":
            self._render_histogram(x_col)
            return

        required_columns = [
            col for col in [x_col, y_col, c_col] if col and col != "None"
        ]
        data = self._prepare_dataframe(required_columns)
        if data.empty:
            self._plot_host.children = [HTML("<i>No rows match the current filters.</i>")]
            return

        scatter_id = f"scatter-{next(self._id_counter)}"
        scatter = ScatterPlotWidget(
            identifier=scatter_id,
            data=data,
            x=x_col,
            y=y_col,
            color=c_col if c_col != "None" else None,
            point_size=self.point_size,
            title=self._scatter_title(x_col, y_col, c_col),
            tooltip_fields=self._tooltip_fields(x_col, y_col, c_col),
            height=320,
        )
        scatter.add_selection_listener(self._on_scatter_selection)
        scatter.add_hover_listener(self._on_scatter_hover)
        self._scatter_views[scatter_id] = scatter
        self._update_scatter_controls(selected_id=scatter_id)
        self._render_scatter_area()
        self._sync_panel_location()

    def _render_histogram(self, x_col: str) -> None:
        data = self._prepare_dataframe([x_col])
        if data.empty:
            self._plot_host.children = [HTML("<i>No rows available for histogram.</i>")]
            return

        self._hist_output.clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(self.width * 0.9, self.height))
        with self._hist_output:
            ax.hist(
                data[x_col],
                bins=self.ui_component.bin_slider.value,
                color="tab:blue",
                alpha=0.75,
            )
            ax.set_xlabel(x_col)
            ax.set_ylabel("Cell count")
            self.histogram_line = None

            def onclick(event):
                if event.inaxes != ax:
                    return
                self.cutoff = event.xdata
                if self.histogram_line is not None:
                    try:
                        self.histogram_line.remove()
                    except ValueError:
                        pass
                self.histogram_line = ax.axvline(
                    self.cutoff, color="red", linestyle="--"
                )
                fig.canvas.draw_idle()
                print(f"Cutoff set at: {self.cutoff:.3f}")
                self.highlight_cells()

            fig.canvas.mpl_connect("button_press_event", onclick)
            fig.tight_layout()
            plt.show(fig)

        self._plot_host.children = [self._hist_output]

    # ------------------------------------------------------------------
    # Selection + linking helpers
    # ------------------------------------------------------------------
    def _on_scatter_selection(
        self, indices: Set[Union[int, str]], origin: str
    ) -> None:
        normalized = {
            int(idx) if isinstance(idx, np.integer) else idx for idx in indices
        }
        self.selected_indices.value = normalized
        if self.ui_component.mv_linked_checkbox.value and len(normalized) == 1:
            self._focus_main_viewer(next(iter(normalized)))

    def _on_scatter_hover(self, index: Optional[Union[int, str]]) -> None:
        # Reserved for future hover-linked integrations.
        return None

    def _apply_external_selection(
        self, indices: Iterable[Union[int, str]]
    ) -> None:
        normalized = {
            int(idx) if isinstance(idx, np.integer) else idx for idx in indices
        }
        for scatter in self._scatter_views.values():
            scatter.apply_selection(normalized, announce=False)
        self.selected_indices.value = normalized

    def apply_color_mapping(
        self,
        categories: pd.Series,
        color_map: Mapping[Union[int, str], Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]],
        *,
        default_color: Union[str, Tuple[float, float, float], Tuple[float, float, float, float]] = "grey",
    ) -> None:
        if categories is None or not self._scatter_views:
            return
        if not isinstance(categories, pd.Series):
            categories = pd.Series(categories)
        for scatter in self._scatter_views.values():
            scatter.set_categorical_colors(categories, color_map, default_color=default_color)

    def _focus_main_viewer(self, index: Union[int, str]) -> None:
        if index not in self.main_viewer.cell_table.index:
            return
        row = self.main_viewer.cell_table.loc[index]
        fov = row[self.main_viewer.fov_key]
        x = row[self.main_viewer.x_key]
        y = row[self.main_viewer.y_key]
        self.main_viewer.ui_component.image_selector.value = fov
        ax = getattr(self.main_viewer.image_display, "ax", None)
        if ax is None:
            return
        nav_stack = ax.figure.canvas.toolbar._nav_stack  # type: ignore[attr-defined]
        current_view = nav_stack()
        ax.set_xlim(x - 100, x + 100)
        ax.set_ylim(y - 100, y + 100)
        nav_stack.push(current_view)
        ax.figure.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Trace + highlight
    # ------------------------------------------------------------------
    def trace_cells(self, _button) -> None:
        if not self.main_viewer.image_display.selected_masks_label:
            print("No cells selected.")
            return
        x_col = self.ui_component.x_axis_selector.value
        y_col = self.ui_component.y_axis_selector.value
        if x_col == "None" or y_col == "None":
            print("Please select both x and y axes to trace cells.")
            return
        current_fov = self.main_viewer.ui_component.image_selector.value
        mask_ids = [
            mask_id
            for _mask_name, mask_id in self.main_viewer.image_display.selected_masks_label
        ]
        if not mask_ids:
            print("No cells selected for tracing.")
            return
        cell_table = self.main_viewer.cell_table
        in_fov = cell_table[self.main_viewer.fov_key] == current_fov
        traced = cell_table.loc[
            in_fov & cell_table[self.main_viewer.label_key].isin(mask_ids)
        ]
        if traced.empty:
            print("No matching cells found in the current FOV.")
            return
        self._apply_external_selection(traced.index)

    def highlight_cells(self) -> None:
        x_col = self.ui_component.x_axis_selector.value
        if x_col == "None" or self.cutoff is None:
            print("X-axis not selected or cutoff unset.")
            return
        cell_table = self.main_viewer.cell_table
        select_above = self.ui_component.above_below_buttons.value == "above"
        comparator = np.greater if select_above else np.less
        within_fov = (
            cell_table[self.main_viewer.fov_key]
            == self.main_viewer.ui_component.image_selector.value
        )
        matches = comparator(cell_table[x_col], self.cutoff)
        mask_ids = cell_table.loc[
            within_fov & matches, self.main_viewer.label_key
        ].tolist()
        self.main_viewer.image_display.set_mask_ids(
            mask_name=self.main_viewer.mask_key,
            mask_ids=mask_ids,
        )

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _prepare_dataframe(self, columns: Sequence[str]) -> pd.DataFrame:
        cell_table = self.main_viewer.cell_table.copy()
        subset_on = self.ui_component.subset_on_dropdown.value
        subset_values = list(self.ui_component.subset_selector.value)
        if subset_on and subset_values:
            if subset_on not in cell_table.columns:
                raise KeyError(
                    f"Subset column '{subset_on}' not found in cell table."
                )
            cell_table = cell_table[cell_table[subset_on].isin(subset_values)]
        if self.ui_component.impose_fov_checkbox.value:
            current_fov = self.main_viewer.ui_component.image_selector.value
            cell_table = cell_table[
                cell_table[self.main_viewer.fov_key] == current_fov
            ]
        columns = [col for col in columns if col in cell_table.columns]
        if columns:
            cell_table = cell_table.dropna(subset=columns)
        return cell_table

    def _tooltip_fields(self, x: str, y: str, color: Optional[str]) -> Sequence[str]:
        fields = [x, y, self.main_viewer.fov_key, self.main_viewer.label_key]
        if color and color != "None":
            fields.append(color)
        # Preserve ordering but remove duplicates
        return list(dict.fromkeys(fields))

    # ------------------------------------------------------------------
    # Scatter management helpers
    # ------------------------------------------------------------------
    def _scatter_title(self, x: str, y: str, color: Optional[str]) -> str:
        base = f"{y} vs {x}"
        if color and color != "None":
            return f"{base} • color: {color}"
        return base

    def _render_scatter_area(self) -> None:
        if not self._scatter_views:
            self._plot_host.children = [self._plot_placeholder]
            return
        if len(self._scatter_views) == 1:
            view = next(iter(self._scatter_views.values()))
            self._plot_host.children = [view.widget()]
            return
        entries = [view.compose_entry() for view in self._scatter_views.values()]
        cols = min(2, len(entries))
        grid = compose(
            entries,
            cols=cols,
            sync_selection=True,
            sync_hover=True,
            row_height=320,
        )
        self._plot_host.children = [grid]

    def _update_scatter_controls(self, selected_id: Optional[str] = None) -> None:
        options = [
            (view.state.title, view.identifier)
            for view in self._scatter_views.values()
        ]
        self.ui_component.scatter_set_selector.options = options or []
        if selected_id:
            self.ui_component.scatter_set_selector.value = selected_id
        elif options:
            current_value = self.ui_component.scatter_set_selector.value
            if current_value not in dict(options):
                self.ui_component.scatter_set_selector.value = options[-1][1]
        else:
            self.ui_component.scatter_set_selector.value = None
        has_plots = bool(options)
        self.ui_component.remove_scatter_button.disabled = (
            not has_plots or not self.ui_component.scatter_set_selector.value
        )
        self.ui_component.clear_scatter_button.disabled = not has_plots
        self.ui_component.clear_selection_button.disabled = not has_plots

    def _remove_selected_scatter(self, _button) -> None:
        selected_id = self.ui_component.scatter_set_selector.value
        if not selected_id or selected_id not in self._scatter_views:
            return
        scatter = self._scatter_views.pop(selected_id)
        scatter.dispose()
        self._update_scatter_controls()
        self._render_scatter_area()
        self._sync_panel_location()

    def _clear_all_scatter_views(self, _button) -> None:
        for scatter in self._scatter_views.values():
            scatter.dispose()
        self._scatter_views.clear()
        self._update_scatter_controls()
        self._render_scatter_area()
        self._sync_panel_location()
        self.selected_indices.value = set()

    def _on_scatter_selector_change(self, change) -> None:
        value = change.get("new")
        self.ui_component.remove_scatter_button.disabled = value is None

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _has_multiple_scatter(self) -> bool:
        return len(self._scatter_views) > 1

    def _place_sections_vertical(self) -> None:
        if self._section_location == "vertical":
            return
        self.ui.children = [self.controls_section, self.plot_section]
        self._section_location = "vertical"

    def _place_sections_horizontal(self) -> None:
        if self._section_location == "horizontal":
            return
        self.ui.children = [self._wide_notice]
        self._section_location = "horizontal"

    def _sync_panel_location(self) -> bool:
        previous_location = self._section_location
        if self._has_multiple_scatter():
            self._place_sections_horizontal()
        else:
            self._place_sections_vertical()
        layout_changed = previous_location != self._section_location
        if (
            layout_changed
            and hasattr(self.main_viewer, "refresh_bottom_panel")
        ):
            self.main_viewer.refresh_bottom_panel()
        return layout_changed

    def wide_panel_layout(self):
        if self._has_multiple_scatter():
            self._place_sections_horizontal()
            return {
                "title": self.displayed_name,
                "control": self.controls_section,
                "content": self.plot_section,
            }
        self._place_sections_vertical()
        return None

    def after_all_plugins_loaded(self):
        super().after_all_plugins_loaded()
        layout_changed = self._sync_panel_location()
        if (
            not layout_changed
            and hasattr(self.main_viewer, "refresh_bottom_panel")
        ):
            self.main_viewer.refresh_bottom_panel()

    # ------------------------------------------------------------------
    # Widget linkage
    # ------------------------------------------------------------------
    def on_subset_on_dropdown_change(self, change):
        selected_column = change.get("new")
        if (
            not selected_column
            or selected_column not in self.main_viewer.cell_table.columns
        ):
            self.ui_component.subset_selector.options = []
            return
        unique_values = (
            self.main_viewer.cell_table[selected_column].dropna().unique().tolist()
        )
        self.ui_component.subset_selector.options = sorted(unique_values)

    def _on_point_size_change(self, change) -> None:
        if change.get("name") != "value":
            return
        new_size = change.get("new")
        if new_size is None:
            return
        self.point_size = float(new_size)
        for scatter in self._scatter_views.values():
            scatter.set_point_size(self.point_size)

    def setup_observe(self):
        def forward_to_cell_gallery(indices):
            if self.ui_component.cell_gallery_linked_checkbox.value:
                self.main_viewer.SidePlots.cell_gallery_output.set_selected_cells(
                    indices
                )

        self.selected_indices.add_observer(forward_to_cell_gallery)

    def color_points(self, selected_indices, selected_colors=None):
        self._apply_external_selection(selected_indices)


class UiComponent:
    def __init__(self, viewer):
        widget_style = {"description_width": "auto"}
        dropdown_options = ["None"] + viewer.cell_table.columns.tolist()

        self.x_axis_selector = Dropdown(
            options=dropdown_options,
            value="None",
            description="X:",
            style=widget_style,
            layout=Layout(width="150px"),
        )
        self.y_axis_selector = Dropdown(
            options=dropdown_options,
            value="None",
            description="Y:",
            style=widget_style,
            layout=Layout(width="150px"),
        )
        self.color_selector = Dropdown(
            options=dropdown_options,
            value="None",
            description="Color:",
            style=widget_style,
            layout=Layout(width="150px"),
        )
        self.impose_fov_checkbox = Checkbox(
            value=False,
            description="Current FOV",
            style=widget_style,
            layout=Layout(width="140px"),
        )
        self.plot_button = Button(
            description="Plot",
            button_style="",
            tooltip="Plot the selected columns",
            icon="line-chart",
            layout=Layout(width="120px"),
        )

        self.bin_slider = IntSlider(
            value=50,
            min=10,
            max=200,
            step=1,
            description="Bins:",
            continuous_update=False,
            style={'description_width': 'auto'},
            layout=Layout(width="250px"),
        )
        self.above_below_buttons = ToggleButtons(
            options=["below", "above"],
            description="Highlight:",
            style={'description_width': 'auto'},
            layout=Layout(width="250px"),
        )
        self.point_size_slider = FloatSlider(
            value=10.0,
            min=0.1,
            max=40.0,
            step=0.1,
            description="Point Size:",
            continuous_update=False,
            style={'description_width': 'auto'},
            layout=Layout(width="250px"),
        )

        subset_columns = [
            col
            for col in viewer.cell_table.columns
            if pd.api.types.is_numeric_dtype(viewer.cell_table[col])
            or pd.api.types.is_object_dtype(viewer.cell_table[col])
        ]
        self.subset_on_dropdown = Dropdown(
            options=subset_columns,
            description="Subset on:",
            style={'description_width': 'auto'},
            layout=Layout(width="100%"),
        )
        self.subset_selector = SelectMultiple(
            options=[],
            description="Subset:",
            style={'description_width': 'auto'},
            layout=Layout(width="100%"),
        )
        self.trace_button = Button(
            description="Trace",
            button_style="",
            tooltip="Select the traced cells in every scatter plot",
            icon="search",
            layout=Layout(width="120px"),
        )
        self.mv_linked_checkbox = Checkbox(
            value=False,
            description="Main viewer",
            style=widget_style,
        )
        self.cell_gallery_linked_checkbox = Checkbox(
            value=False,
            description="Cell gallery",
            style=widget_style,
        )
        self.scatter_set_selector = Dropdown(
            options=[],
            description="Plots:",
            style={'description_width': 'auto'},
            layout=Layout(width="250px"),
        )
        self.remove_scatter_button = Button(
            description="Remove",
            icon="trash",
            button_style="warning",
            layout=Layout(width="110px"),
            disabled=True,
        )
        self.clear_scatter_button = Button(
            description="Clear all",
            icon="times",
            button_style="danger",
            layout=Layout(width="110px"),
            disabled=True,
        )
        self.clear_selection_button = Button(
            description="Clear selection",
            icon="eraser",
            layout=Layout(width="140px"),
            disabled=True,
        )