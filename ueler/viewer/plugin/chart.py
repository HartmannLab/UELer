from __future__ import annotations

import itertools
import logging
import os
from collections import OrderedDict

_logger = logging.getLogger(__name__)
from typing import Iterable, Mapping, Optional, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ipywidgets as _ipywidgets

Box = getattr(_ipywidgets, "Box")
Button = getattr(_ipywidgets, "Button")
Checkbox = getattr(_ipywidgets, "Checkbox")
Dropdown = getattr(_ipywidgets, "Dropdown")
FloatSlider = getattr(_ipywidgets, "FloatSlider", getattr(_ipywidgets, "Widget"))
HBox = getattr(_ipywidgets, "HBox")
HTML = getattr(_ipywidgets, "HTML")
Layout = getattr(_ipywidgets, "Layout")
Output = getattr(_ipywidgets, "Output")
SelectMultiple = getattr(_ipywidgets, "SelectMultiple")
Tab = getattr(_ipywidgets, "Tab")
VBox = getattr(_ipywidgets, "VBox")

from jscatter import compose

from ueler.viewer.decorators import update_status_bar
from ueler.viewer.observable import Observable

from . import _chart_common
from .plugin_base import PluginBase
from .scatter_widget import ScatterPlotWidget


_SELECTION_NOTICE = (
    "<i>No scatter plots generated yet. Choose axes, then click <b>Plot</b>.</i>"
)


class ChartDisplay(PluginBase):
    def __init__(self, main_viewer, width: float, height: float):
        super().__init__(main_viewer, width, height)
        self.SidePlots_id = "chart_output"
        self.displayed_name = "Scatter plot"
        self.main_viewer = main_viewer
        self.width = width
        self.height = height

        self.point_size = 10.0

        self.selected_indices: Observable = Observable(set())

        self._scatter_views: "OrderedDict[str, ScatterPlotWidget]" = OrderedDict()
        # (x, y) channel pair per scatter id — used to lay out the triangular
        # matrix (ScatterViewState does not expose x/y). (#113)
        self._scatter_pairs: "OrderedDict[str, Tuple[str, str]]" = OrderedDict()
        # Channels of the most recent "Plot all pairs" — anchors the triangular
        # matrix so later single-pair plots append below it rather than resetting.
        self._multipair_channels_last: list = []
        self._id_counter = itertools.count(1)
        self._observers_registered = False

        self.ui_component = UiComponent(self.main_viewer)
        self._plot_placeholder = HTML(
            value=_SELECTION_NOTICE,
            layout=Layout(width="100%"),
        )
        self._plot_host = VBox(
            children=[self._plot_placeholder], layout=Layout(width="100%", gap="8px")
        )

        self._wide_notice = HTML(
            value=(
                "<b>Multiple scatter plots are active.</b> Controls and plots appear in the footer."
            ),
            layout=Layout(width="100%", padding="8px"),
        )
        self._section_location = "vertical"

        self.single_point_click_state = 0
        backend_env = os.environ.get("UELER_SCATTER_BACKEND")
        default_backend = "static" if os.environ.get("VSCODE_PID") else "widget"
        backend = (backend_env or default_backend).lower()
        if backend not in {"widget", "static"}:
            backend = default_backend
        self._scatter_backend = backend
        self._scatter_fallback_notice = HTML(
            value=(
                "<b>Scatter fallback active.</b> Interactive scatter widgets are disabled in this "
                "environment; showing a static Matplotlib plot instead. Set "
                "<code>UELER_SCATTER_BACKEND=widget</code> (after enabling widget support) to "
                "force the interactive scatter backend."
            ),
            layout=Layout(width="100%"),
        )

        self._wire_events()
        self._build_layout()
        self.setup_widget_observers()

    # ------------------------------------------------------------------
    # UI wiring
    # ------------------------------------------------------------------
    def _wire_events(self) -> None:
        self.ui_component.plot_button.on_click(self.plot_chart)
        self.ui_component.plot_pairs_button.on_click(self.plot_all_pairs)
        self.ui_component.channel_selector_bundle.load_button.on_click(
            lambda _btn: _chart_common.apply_marker_set_to_selector(
                self.ui_component.channel_selector_bundle, self.main_viewer
            )
        )
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
        # Multi-pair selector is now the always-visible picker on top (#113).
        multipair_controls = VBox(
            children=[
                self.ui_component.channel_selector_bundle.box,
                self.ui_component.plot_pairs_button,
            ],
            layout=Layout(width="100%", gap="8px"),
        )

        # Single-pair X/Y/Color selector moved into a tab (#113).
        singlepair_controls = VBox(
            children=[
                HBox(
                    children=[
                        self.ui_component.x_axis_selector,
                        self.ui_component.y_axis_selector,
                        self.ui_component.color_selector,
                    ],
                    layout=Layout(gap="8px", align_items="center"),
                ),
                self.ui_component.plot_button,
            ],
            layout=Layout(width="100%", gap="8px"),
        )

        scatter_controls = VBox(
            children=[
                self.ui_component.point_size_slider,
                HBox(
                    children=[
                        self.ui_component.scatter_set_selector,
                        self.ui_component.remove_scatter_button,
                    ],
                    layout=Layout(gap="8px", align_items="center"),
                ),
                HBox(
                    children=[
                        self.ui_component.clear_scatter_button,
                        self.ui_component.clear_selection_button,
                    ],
                    layout=Layout(gap="8px"),
                ),
            ],
            layout=Layout(width="100%", gap="8px"),
        )

        subset_controls = VBox(
            children=[
                self.ui_component.subset_on_dropdown,
                self.ui_component.subset_selector,
                self.ui_component.impose_fov_checkbox,
            ],
            layout=Layout(width="100%", gap="8px"),
        )

        trace_controls = VBox(
            children=[self.ui_component.trace_button],
            layout=Layout(width="100%", gap="8px"),
        )

        link_controls = VBox(
            children=[
                self.ui_component.mv_linked_checkbox,
                self.ui_component.cell_gallery_linked_checkbox,
            ],
            layout=Layout(width="100%", gap="8px"),
        )

        self._plot_tabs = Tab(
            children=[
                scatter_controls,
                singlepair_controls,
                subset_controls,
                trace_controls,
                link_controls,
            ]
        )
        self._plot_tabs.set_title(0, "Scatter plot")
        self._plot_tabs.set_title(1, "Single-pair")
        self._plot_tabs.set_title(2, "Subset")
        self._plot_tabs.set_title(3, "Trace")
        self._plot_tabs.set_title(4, "Linked plugins")

        chart_widgets = VBox(
            children=[
                multipair_controls,
                self._plot_tabs,
            ],
            layout=Layout(width="100%", gap="10px"),
        )

        self.controls_section = VBox(
            children=[chart_widgets],
            layout=Layout(width="100%", gap="12px"),
        )
        self.plot_section = VBox(
            children=[self._plot_host],
            layout=Layout(width="100%", flex="1 1 auto"),
        )

        self.ui = VBox(
            children=[self.controls_section, self.plot_section],
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

        if x_col == "None" or y_col == "None":
            _logger.warning("Please select columns for both the x and y axes.")
            return

        required_columns = [
            col for col in [x_col, y_col, c_col] if col and col != "None"
        ]
        data = self._prepare_dataframe(required_columns)
        if data.empty:
            self._plot_host.children = [HTML("<i>No rows match the current filters.</i>")]
            return

        if self._scatter_backend == "static":
            self._render_scatter_matplotlib(data, x_col, y_col, c_col)
            return

        self._add_scatter_view(data, x_col, y_col, c_col)
        self._render_scatter_area()
        self._sync_panel_location()

    @update_status_bar
    def plot_all_pairs(self, _button):
        """Generate a scatter plot for every pairwise combination of the selected channels."""
        channels = [
            col
            for col in self.ui_component.multipair_channels.value
            if col and col != "None"
        ]
        if len(channels) < 2:
            _logger.warning("Select at least two channels to plot all pairs.")
            return
        # Anchor the triangular matrix on this channel set (#113).
        self._multipair_channels_last = list(dict.fromkeys(channels))
        channels = self._multipair_channels_last
        c_col = self.ui_component.color_selector.value

        required_columns = list(dict.fromkeys(channels + ([c_col] if c_col != "None" else [])))
        data = self._prepare_dataframe(required_columns)
        if data.empty:
            self._plot_host.children = [HTML("<i>No rows match the current filters.</i>")]
            return

        if self._scatter_backend == "static":
            # The static fallback renders a single Matplotlib axes; plot the
            # first pair so the environment still gets a usable figure.
            x_col, y_col = channels[0], channels[1]
            self._render_scatter_matplotlib(data, x_col, y_col, c_col)
            return

        last_id = None
        for x_col, y_col in itertools.combinations(channels, 2):
            last_id = self._add_scatter_view(data, x_col, y_col, c_col)
        if last_id is not None:
            self._update_scatter_controls(selected_id=last_id)
        self._render_scatter_area()
        self._sync_panel_location()

    def _add_scatter_view(
        self, data: pd.DataFrame, x_col: str, y_col: str, c_col: str
    ) -> str:
        """Create a scatter widget for ``x_col`` vs ``y_col`` and register it."""
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
        self._scatter_pairs[scatter_id] = (x_col, y_col)
        self._update_scatter_controls(selected_id=scatter_id)
        return scatter_id

    # ------------------------------------------------------------------
    # Selection + linking helpers
    # ------------------------------------------------------------------
    def _on_scatter_selection(
        self, indices: Set[Union[int, str]], origin: str
    ) -> None:
        self._commit_scatter_selection(
            indices,
            focus_single=(origin == "widget"),
        )

    def _on_scatter_hover(self, index: Optional[Union[int, str]]) -> None:
        # Reserved for future hover-linked integrations.
        return None

    def _apply_external_selection(
        self, indices: Iterable[Union[int, str]]
    ) -> None:
        self._commit_scatter_selection(indices)

    def _commit_scatter_selection(
        self,
        indices: Iterable[Union[int, str]],
        *,
        focus_single: bool = False,
    ) -> Set[Union[int, str]]:
        normalized = {
            int(idx) if isinstance(idx, np.integer) else idx for idx in indices
        }
        self._update_single_point_state(normalized)
        for scatter in self._scatter_views.values():
            scatter.apply_selection(normalized, announce=False)
        self.selected_indices.value = normalized
        if self.ui_component.mv_linked_checkbox.value:
            if focus_single and len(normalized) == 1:
                self._focus_main_viewer(next(iter(normalized)))
            self._sync_mask_highlights_from_selection(normalized)
        return normalized

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
        self.main_viewer.focus_on_cell(fov, x, y, radius=100.0)

    def _sync_mask_highlights_from_selection(
        self, indices: Set[Union[int, str]]
    ) -> None:
        """Translate scatter selection row indices to mask highlights in the viewer.

        Called whenever the scatter selection changes and the main-viewer
        link is active.  Works in both single-FOV and map mode.
        """
        _chart_common.sync_mask_highlights_from_selection(self.main_viewer, indices)

    # ------------------------------------------------------------------
    # Trace + highlight
    # ------------------------------------------------------------------
    def trace_cells(self, _button) -> None:
        selections = list(self.main_viewer.image_display.selected_masks_label)
        if not selections:
            _logger.warning("No cells selected.")
            return
        x_col = self.ui_component.x_axis_selector.value
        y_col = self.ui_component.y_axis_selector.value
        if x_col == "None" or y_col == "None":
            _logger.warning("Please select both x and y axes to trace cells.")
            return
        current_fov = self.main_viewer.ui_component.image_selector.value
        mask_ids = [
            selection.mask_id
            for selection in selections
            if getattr(selection, "fov", current_fov) == current_fov
        ]
        if not mask_ids:
            _logger.warning("No cells selected for tracing.")
            return
        cell_table = self.main_viewer.cell_table
        in_fov = cell_table[self.main_viewer.fov_key] == current_fov
        traced = cell_table.loc[
            in_fov & cell_table[self.main_viewer.label_key].isin(mask_ids)
        ]
        if traced.empty:
            _logger.warning("No matching cells found in the current FOV.")
            return
        self._apply_external_selection(traced.index)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _prepare_dataframe(self, columns: Sequence[str]) -> pd.DataFrame:
        return _chart_common.prepare_dataframe(
            self.main_viewer,
            subset_on=self.ui_component.subset_on_dropdown.value,
            subset_values=self.ui_component.subset_selector.value,
            impose_fov=self.ui_component.impose_fov_checkbox.value,
            columns=columns,
        )

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
        grid = self._triangular_grid()
        if grid is not None:
            self._plot_host.children = [grid]
            return
        # Fallback: no active pairwise matrix (purely single-pair or manually
        # added/removed views) — keep the previous 2-column compose grid.
        entries = [view.compose_entry() for view in self._scatter_views.values()]
        cols = min(2, len(entries))
        grid = compose(
            entries,
            cols=cols,
            sync_selection=False,
            sync_hover=True,
            row_height=320,
        )
        self._plot_host.children = [grid]

    def _triangular_grid(self):
        """Lay the pairwise scatters out as an upper-triangular matrix (#113).

        Built as a ``VBox`` of ``HBox`` rows (plain flexbox) rather than a
        CSS-grid ``GridBox``: every row has exactly ``N-1`` equal-flex cells so
        the columns line up, blank cells fill the lower triangle, and each cell
        lets its scatter self-size (no fixed height, so axes are never clipped).

        Returns the ``VBox`` when a full pairwise matrix for
        ``self._multipair_channels_last`` is present — with any extra
        (single-pair) views appended as new rows below it — or ``None`` when
        there is no active matrix to anchor the layout (callers then fall back
        to ``compose``).
        """
        channels = self._multipair_channels_last
        if len(channels) < 2:
            return None
        matrix_pairs = list(itertools.combinations(channels, 2))
        present = set(self._scatter_pairs.values())
        # The matrix is only "active" while all of its pairs are still shown.
        if not all(pair in present for pair in matrix_pairs):
            return None

        n = len(channels)
        cols = n - 1
        matrix_pair_set = set(matrix_pairs)
        view_by_pair = {
            self._scatter_pairs[sid]: view
            for sid, view in self._scatter_views.items()
        }

        rows = []
        # Matrix rows: row i has i leading blanks, then plots for (i, j), j>i.
        for i in range(n - 1):
            cells = [self._blank_cell() for _ in range(i)]
            for j in range(i + 1, n):
                cells.append(self._plot_cell(view_by_pair[(channels[i], channels[j])]))
            rows.append(self._grid_row(cells))

        # Extra views (e.g. single-pair plots added after the matrix): append as
        # new full rows below, flowing left→right, padded with blanks to N-1.
        extras = [
            view
            for sid, view in self._scatter_views.items()
            if self._scatter_pairs.get(sid) not in matrix_pair_set
        ]
        for start in range(0, len(extras), cols):
            chunk = extras[start:start + cols]
            cells = [self._plot_cell(view) for view in chunk]
            cells.extend(self._blank_cell() for _ in range(cols - len(chunk)))
            rows.append(self._grid_row(cells))

        return VBox(children=rows, layout=Layout(width="100%", gap="8px"))

    @staticmethod
    def _plot_cell(view) -> "Box":
        return Box(
            children=[view.widget()],
            layout=Layout(flex="1 1 0%", min_width="0"),
        )

    @staticmethod
    def _blank_cell() -> "Box":
        return Box(children=[], layout=Layout(flex="1 1 0%", min_width="0"))

    @staticmethod
    def _grid_row(cells) -> "HBox":
        return HBox(
            children=cells,
            layout=Layout(width="100%", gap="4px", align_items="stretch"),
        )

    def _render_scatter_matplotlib(
        self, data: pd.DataFrame, x_col: str, y_col: str, c_col: Optional[str]
    ) -> None:
        scatter_kwargs = {}
        color_values = None
        if c_col and c_col != "None" and c_col in data.columns:
            color_values = data[c_col]
            scatter_kwargs["c"] = color_values
        out = Output(layout=Layout(width="100%"))
        with out:
            fig, ax = plt.subplots(figsize=(self.width * 0.9, self.height))
            ax.scatter(data[x_col], data[y_col], s=self.point_size, **scatter_kwargs)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            if color_values is not None:
                try:
                    fig.colorbar(ax.collections[0], ax=ax)
                except Exception:
                    pass
            fig.tight_layout()
            plt.show(fig)
        self._scatter_views.clear()
        self._update_scatter_controls()
        if self._scatter_backend == "static":
            self._plot_host.children = [self._scatter_fallback_notice, out]
        else:
            self._plot_host.children = [out]

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
        self._scatter_pairs.pop(selected_id, None)
        scatter.dispose()
        self._update_scatter_controls()
        self._render_scatter_area()
        self._sync_panel_location()

    def _clear_all_scatter_views(self, _button) -> None:
        for scatter in self._scatter_views.values():
            scatter.dispose()
        self._scatter_views.clear()
        self._scatter_pairs.clear()
        self._multipair_channels_last = []
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
        _logger.debug("[chart] sync panel location: %s", self._section_location)
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
            _logger.debug("[chart] refreshing bottom panel due to layout change")
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

    def on_marker_sets_changed(self):
        """Keep the marker-set dropdown in sync with the left panel (#113)."""
        _chart_common.refresh_marker_set_options(
            self.ui_component.channel_selector_bundle, self.main_viewer
        )

    def after_all_plugins_loaded(self):
        super().after_all_plugins_loaded()
        # Marker sets are restored from widget_states.json after plugin __init__.
        self.on_marker_sets_changed()
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
        self.ui_component.subset_selector.options = _chart_common.subset_options_for(
            self.main_viewer, selected_column
        )

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
        if self._observers_registered:
            return

        def forward_to_cell_gallery(indices):
            if self.ui_component.cell_gallery_linked_checkbox.value:
                if self.single_point_click_state == 1:
                    return
                self.main_viewer.SidePlots.cell_gallery_output.set_selected_cells(
                    indices
                )

        self.selected_indices.add_observer(forward_to_cell_gallery)
        self._observers_registered = True

    def color_points(self, selected_indices, selected_colors=None):
        _ = selected_colors  # legacy parameter retained for compatibility
        self._apply_external_selection(selected_indices)

    def _update_single_point_state(self, normalized: Set[Union[int, str]]) -> None:
        self.single_point_click_state = 1 if len(normalized) == 1 else 0


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
        self.plot_button = Button(
            description="Plot",
            button_style="",
            tooltip="Plot the selected columns",
            icon="line-chart",
            layout=Layout(width="120px"),
        )

        # Multi-pair scatter: pick several channels, plot every pairwise
        # combination. Uses the left-panel-consistent channel picker (with
        # marker-set loading) shared with the histogram plugin (#113).
        # ``multipair_channels`` stays as an alias to the bundle's TagsInput so
        # ``plot_all_pairs`` (which reads ``multipair_channels.value``) is unchanged.
        self.channel_selector_bundle = _chart_common.build_channel_selector(viewer)
        self.multipair_channels = self.channel_selector_bundle.tags
        self.plot_pairs_button = Button(
            description="Plot all pairs",
            button_style="",
            tooltip="Plot a scatter for every pairwise combination of the selected channels",
            icon="th",
            layout=Layout(width="150px"),
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

        (
            self.subset_on_dropdown,
            self.subset_selector,
            self.impose_fov_checkbox,
        ) = _chart_common.build_subset_controls(viewer)
        self.trace_button = Button(
            description="Trace",
            button_style="",
            tooltip="Select the traced cells in every scatter plot",
            icon="search",
            layout=Layout(width="120px"),
        )
        (
            self.mv_linked_checkbox,
            self.cell_gallery_linked_checkbox,
        ) = _chart_common.build_link_checkboxes()
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
