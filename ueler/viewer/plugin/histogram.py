"""Standalone Histogram plugin, split out of the combined Chart plugin (issue #112).

Previously histograms and scatter plots shared one render host inside
``ChartDisplay`` so only one could be visible at a time.  This plugin owns its
own render area, so a histogram and a scatter plot can now be open together.

Beyond feature-parity with the old histogram (single-channel cutoff + above/below
highlighting driven into the viewer), it supports the issue's "further
considerations":

* **Multiple histograms** — pick several channels and see them all at once.
* **Linked brushing** — drag a range on one histogram and (a) that cell
  selection is reflected in the viewer / cell gallery and (b) the selected
  subset's distribution is overlaid on *every* histogram, so you can see how a
  selection on one channel distributes across the others.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Set, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from IPython.display import display

import ipywidgets as _ipywidgets

Button = getattr(_ipywidgets, "Button")
Dropdown = getattr(_ipywidgets, "Dropdown")
HBox = getattr(_ipywidgets, "HBox")
HTML = getattr(_ipywidgets, "HTML")
IntSlider = getattr(_ipywidgets, "IntSlider", None)
Layout = getattr(_ipywidgets, "Layout")
Output = getattr(_ipywidgets, "Output")
SelectMultiple = getattr(_ipywidgets, "SelectMultiple")
Tab = getattr(_ipywidgets, "Tab")
ToggleButtons = getattr(_ipywidgets, "ToggleButtons")
VBox = getattr(_ipywidgets, "VBox")

if IntSlider is None:  # pragma: no cover - fallback for stub environments
    _base_slider = getattr(_ipywidgets, "Widget")

    class IntSlider(_base_slider):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.min = kwargs.get("min", 0)
            self.max = kwargs.get("max", 10)
            self.step = kwargs.get("step", 1)

    _ipywidgets.IntSlider = IntSlider  # type: ignore[attr-defined]

from ueler.viewer.decorators import update_status_bar
from ueler.viewer.observable import Observable

from . import _chart_common
from .plugin_base import PluginBase

_logger = logging.getLogger(__name__)

_SELECTION_NOTICE = (
    "<i>No histograms yet. Choose one or more channels, then click <b>Plot</b>.</i>"
)

_BASE_COLOR = "tab:blue"
_OVERLAY_COLOR = "tab:orange"


class HistogramDisplay(PluginBase):
    def __init__(self, main_viewer, width: float, height: float):
        super().__init__(main_viewer, width, height)
        self.SidePlots_id = "histogram_output"
        self.displayed_name = "Histogram"
        self.main_viewer = main_viewer
        self.width = width
        self.height = height

        # Cutoff-mode state (feature parity with the old histogram).
        self.cutoff: Optional[float] = None
        self._active_histogram_column: Optional[str] = None

        # Brush-mode state (linked selection across histograms).
        self._brush_selection: Optional[tuple] = None  # (channel, lo, hi)
        self.selected_indices: Observable = Observable(set())
        self.single_point_click_state = 0

        self._channels: list = []
        self._plot_data = None
        self._span_selectors: list = []  # keep SpanSelector refs alive
        self._observers_registered = False

        self.ui_component = UiComponent(self.main_viewer)
        self._plot_placeholder = HTML(value=_SELECTION_NOTICE, layout=Layout(width="100%"))
        self._plot_host = VBox(
            children=[self._plot_placeholder], layout=Layout(width="100%", gap="8px")
        )

        self._wire_events()
        self._build_layout()
        self.setup_widget_observers()

    # ------------------------------------------------------------------
    # UI wiring
    # ------------------------------------------------------------------
    def _wire_events(self) -> None:
        self.ui_component.plot_button.on_click(self.plot_histograms)
        self.ui_component.bin_slider.observe(self._on_bin_slider_change, names="value")
        self.ui_component.above_below_buttons.observe(
            self._on_above_below_change, names="value"
        )
        self.ui_component.interaction_mode.observe(
            self._on_interaction_mode_change, names="value"
        )
        self.ui_component.subset_on_dropdown.observe(
            self.on_subset_on_dropdown_change, names="value"
        )
        self.ui_component.clear_selection_button.on_click(
            lambda _btn: self.clear_selection()
        )

    def _build_layout(self) -> None:
        plot_controls = VBox(
            children=[
                self.ui_component.channel_selector,
                HBox(
                    children=[
                        self.ui_component.bin_slider,
                        self.ui_component.interaction_mode,
                    ],
                    layout=Layout(gap="12px", align_items="center"),
                ),
                HBox(
                    children=[
                        self.ui_component.above_below_buttons,
                        self.ui_component.clear_selection_button,
                    ],
                    layout=Layout(gap="12px", align_items="center"),
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

        link_controls = VBox(
            children=[
                self.ui_component.mv_linked_checkbox,
                self.ui_component.cell_gallery_linked_checkbox,
            ],
            layout=Layout(width="100%", gap="8px"),
        )

        self._plot_tabs = Tab(children=[plot_controls, subset_controls, link_controls])
        self._plot_tabs.set_title(0, "Histogram")
        self._plot_tabs.set_title(1, "Subset")
        self._plot_tabs.set_title(2, "Linked plugins")

        controls = VBox(
            children=[self.ui_component.plot_button, self._plot_tabs],
            layout=Layout(width="100%", gap="10px"),
        )
        self.controls_section = VBox(children=[controls], layout=Layout(width="100%", gap="12px"))
        self.plot_section = VBox(
            children=[self._plot_host], layout=Layout(width="100%", flex="1 1 auto")
        )
        self.ui = VBox(
            children=[self.controls_section, self.plot_section],
            layout=Layout(width="100%", max_height="700px", gap="12px"),
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    @update_status_bar
    def plot_histograms(self, _button):
        channels = [
            col for col in self.ui_component.channel_selector.value if col and col != "None"
        ]
        if not channels:
            _logger.warning("Select at least one channel to plot a histogram.")
            return
        data = self._prepare_dataframe(channels)
        if data.empty:
            self._plot_host.children = [HTML("<i>No rows available for histogram.</i>")]
            return
        self._channels = channels
        self._plot_data = data
        # A fresh plot invalidates any previous cutoff/brush that referenced
        # columns which may no longer be shown.
        if self._active_histogram_column not in channels:
            self._active_histogram_column = None
            self.cutoff = None
        if self._brush_selection is not None and self._brush_selection[0] not in channels:
            self._brush_selection = None
        self._render()

    def _render(self) -> None:
        """Rebuild the figure (one subplot per channel) and swap it into the host."""
        data = self._plot_data
        channels = self._channels
        if data is None or not channels:
            self._plot_host.children = [self._plot_placeholder]
            return

        selected_idx = self.selected_indices.value or set()
        bins = self.ui_component.bin_slider.value
        brush_mode = self.ui_component.interaction_mode.value == "Brush"

        out = Output(layout=Layout(width="100%"))
        self._span_selectors = []
        n = len(channels)
        with out:
            fig, axes = plt.subplots(
                n, 1, figsize=(self.width * 0.9, self.height * n), squeeze=False
            )
            axes = [row[0] for row in axes]
            for ax, channel in zip(axes, channels):
                # Share one set of bin edges (computed from the full data range)
                # between the base and overlay histograms so the selected-subset
                # overlay sits on the same grid and is directly comparable (#112
                # reply). Passing the bin *count* to each hist() would let
                # Matplotlib recompute edges over each call's own data range,
                # squeezing a narrow subset into its own bins.
                edges = self._histogram_bin_edges(channel, bins)
                ax.hist(data[channel], bins=edges, color=_BASE_COLOR, alpha=0.6, label="All")
                # Overlay the currently-selected subset's distribution.
                if selected_idx:
                    valid = [i for i in selected_idx if i in data.index]
                    if valid:
                        ax.hist(
                            data.loc[valid, channel],
                            bins=edges,
                            color=_OVERLAY_COLOR,
                            alpha=0.7,
                            label="Selected",
                        )
                        ax.legend(loc="upper right", fontsize="small")
                ax.set_xlabel(channel)
                ax.set_ylabel("Cell count")
                if (
                    self._active_histogram_column == channel
                    and self.cutoff is not None
                ):
                    ax.axvline(self.cutoff, color="red", linestyle="--")
                self._attach_axis_interactions(fig, ax, channel, brush_mode)
            fig.tight_layout()
            # Emit the interactive canvas exactly once (mirrors the heatmap fix
            # from issue #108); works across the widget and headless backends.
            display(fig.canvas)

        self._plot_host.children = [out]

    def _attach_axis_interactions(self, fig, ax, channel, brush_mode) -> None:
        """Wire cutoff clicks and (optionally) range brushing to a single subplot."""

        def onclick(event):
            if event.inaxes != ax or brush_mode:
                return
            if event.xdata is None:
                return
            self.cutoff = float(event.xdata)
            self._active_histogram_column = channel
            _logger.info("Cutoff set at %.3f on channel %s", self.cutoff, channel)
            self.highlight_cells(push_to_gallery=True)
            self._render()

        fig.canvas.mpl_connect("button_press_event", onclick)

        if brush_mode:
            try:
                from matplotlib.widgets import SpanSelector

                selector = SpanSelector(
                    ax,
                    lambda lo, hi, _ch=channel: self._on_brush(_ch, lo, hi),
                    "horizontal",
                    useblit=False,
                    props=dict(alpha=0.2, facecolor=_OVERLAY_COLOR),
                )
                self._span_selectors.append(selector)
            except Exception:  # pragma: no cover - defensive (headless / old mpl)
                _logger.debug("SpanSelector unavailable; brush mode disabled", exc_info=True)

    def _histogram_bin_edges(self, channel: str, bins: int):
        """Bin edges computed over the *full* plotted data for ``channel``.

        Shared by the base and subset-overlay histograms so both sit on the same
        grid; independent of the current selection (#112 reply). ``_plot_data``
        is already NaN-dropped on the plotted channels by ``_prepare_dataframe``.
        """
        return np.histogram_bin_edges(self._plot_data[channel], bins=bins)

    # ------------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------------
    def _cells_in_range(self, channel: str, lo: float, hi: float) -> Set[Union[int, str]]:
        """Row indices of the (filtered) data whose ``channel`` value is within [lo, hi]."""
        data = self._plot_data
        if data is None or channel not in data.columns:
            return set()
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        mask = data[channel].between(lo, hi)
        return set(data.index[mask])

    def _on_brush(self, channel: str, lo: float, hi: float) -> None:
        if lo == hi:
            return
        self._brush_selection = (channel, lo, hi)
        indices = _chart_common.normalize_indices(self._cells_in_range(channel, lo, hi))
        self._update_single_point_state(indices)
        # Publish the selection (drives the cell-gallery observer when linked).
        self.selected_indices.value = indices
        if self.ui_component.mv_linked_checkbox.value:
            _chart_common.sync_mask_highlights_from_selection(self.main_viewer, indices)
        # Redraw so every histogram shows the selected subset overlaid.
        self._render()

    def highlight_cells(self, *, push_to_gallery: bool = False) -> None:
        """Cutoff-based highlight: select cells above/below the active-channel cutoff."""
        channel = self._active_histogram_column
        if channel is None or self.cutoff is None:
            _logger.warning("No active channel or cutoff set.")
            return
        cell_table = self.main_viewer.cell_table
        if channel not in cell_table.columns:
            return
        select_above = self.ui_component.above_below_buttons.value == "above"
        comparator = np.greater if select_above else np.less
        matches = comparator(cell_table[channel], self.cutoff)
        active_fov = self.main_viewer.get_active_fov()
        if active_fov:
            within_fov = cell_table[self.main_viewer.fov_key] == active_fov
            mask_ids = cell_table.loc[
                within_fov & matches, self.main_viewer.label_key
            ].tolist()
            self.main_viewer.image_display.set_mask_ids(
                mask_name=self.main_viewer.mask_key, mask_ids=mask_ids
            )
        else:
            fov_col = self.main_viewer.fov_key
            lbl_col = self.main_viewer.label_key
            matched_rows = cell_table.loc[matches, [fov_col, lbl_col]]
            fov_mask_pairs = list(
                zip(matched_rows[fov_col].astype(str), matched_rows[lbl_col].astype(int))
            )
            self.main_viewer.image_display.set_mask_ids(
                mask_name=self.main_viewer.mask_key, mask_ids=[], fov_mask_pairs=fov_mask_pairs
            )
        if push_to_gallery:
            self.selected_indices.value = set(cell_table.loc[matches].index)

    def clear_selection(self) -> None:
        self._brush_selection = None
        self.selected_indices.value = set()
        if self.ui_component.mv_linked_checkbox.value:
            _chart_common.sync_mask_highlights_from_selection(self.main_viewer, set())
        self._render()

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def _prepare_dataframe(self, columns: Sequence[str]):
        return _chart_common.prepare_dataframe(
            self.main_viewer,
            subset_on=self.ui_component.subset_on_dropdown.value,
            subset_values=self.ui_component.subset_selector.value,
            impose_fov=self.ui_component.impose_fov_checkbox.value,
            columns=columns,
        )

    def _update_single_point_state(self, normalized: Set[Union[int, str]]) -> None:
        self.single_point_click_state = 1 if len(normalized) == 1 else 0

    # ------------------------------------------------------------------
    # Widget callbacks / observers
    # ------------------------------------------------------------------
    def on_subset_on_dropdown_change(self, change):
        selected_column = change.get("new")
        self.ui_component.subset_selector.options = _chart_common.subset_options_for(
            self.main_viewer, selected_column
        )

    def _on_bin_slider_change(self, change) -> None:
        if change.get("name") != "value":
            return
        if self._plot_data is not None:
            self._render()

    def _on_above_below_change(self, _change) -> None:
        if self._active_histogram_column is not None and self.cutoff is not None:
            self.highlight_cells(push_to_gallery=True)

    def _on_interaction_mode_change(self, _change) -> None:
        if self._plot_data is not None:
            self._render()

    def setup_observe(self):
        if self._observers_registered:
            return

        def forward_to_cell_gallery(indices):
            if self.ui_component.cell_gallery_linked_checkbox.value:
                if self.single_point_click_state == 1:
                    return
                self.main_viewer.SidePlots.cell_gallery_output.set_selected_cells(indices)

        self.selected_indices.add_observer(forward_to_cell_gallery)
        self._observers_registered = True


class UiComponent:
    def __init__(self, viewer):
        numeric_columns = [
            col
            for col in viewer.cell_table.columns
            if pd.api.types.is_numeric_dtype(viewer.cell_table[col])
        ]
        self.channel_selector = SelectMultiple(
            options=numeric_columns,
            description="Channels:",
            style={"description_width": "auto"},
            layout=Layout(width="100%", height="140px"),
        )
        self.plot_button = Button(
            description="Plot",
            button_style="",
            tooltip="Plot a histogram for each selected channel",
            icon="bar-chart",
            layout=Layout(width="120px"),
        )
        self.bin_slider = IntSlider(
            value=50,
            min=10,
            max=200,
            step=1,
            description="Bins:",
            continuous_update=True,
            style={"description_width": "auto"},
            layout=Layout(width="250px"),
        )
        self.interaction_mode = ToggleButtons(
            options=["Cutoff", "Brush"],
            value="Cutoff",
            description="Interaction:",
            tooltips=[
                "Click a histogram to set an above/below cutoff",
                "Drag a range to select cells and overlay the selection on every histogram",
            ],
            style={"description_width": "auto"},
        )
        self.above_below_buttons = ToggleButtons(
            options=["below", "above"],
            description="Highlight:",
            style={"description_width": "auto"},
            layout=Layout(width="250px"),
        )
        self.clear_selection_button = Button(
            description="Clear selection",
            icon="eraser",
            layout=Layout(width="150px"),
        )
        (
            self.subset_on_dropdown,
            self.subset_selector,
            self.impose_fov_checkbox,
        ) = _chart_common.build_subset_controls(viewer)
        (
            self.mv_linked_checkbox,
            self.cell_gallery_linked_checkbox,
        ) = _chart_common.build_link_checkboxes()
