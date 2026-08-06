"""Standalone Histogram plugin, split out of the combined Chart plugin (issue #112).

Previously histograms and scatter plots shared one render host inside
``ChartDisplay`` so only one could be visible at a time.  This plugin owns its
own render area, so a histogram and a scatter plot can now be open together.

Rendering uses **Bokeh** (via ``jupyter_bokeh``'s ``BokehModel``), replacing the
original matplotlib/``ipympl`` path — the fragility issue #107 moved away from.
Bokeh gives native, kernel-backed interactivity:

* **Multiple histograms** — pick several channels and see them all at once.
* **Linked brushing** — drag a range on one histogram (a ``BoxSelectTool``) and
  (a) that cell selection is reflected in the viewer / cell gallery and (b) the
  selected subset's distribution is overlaid on *every* histogram, so you can
  see how a selection on one channel distributes across the others.
* **Cutoff mode** — tap a histogram to set an above/below threshold that
  highlights cells in the viewer (feature parity with the old histogram).
* **Multi-channel gating (#127)** — each histogram keeps its *own* gate term (a
  brushed range or an above/below cutoff) in ``_gates``, and the published
  selection is the **intersection** of every active term.  Acting on one channel
  replaces only that channel's term and leaves the other histograms' markers,
  axes and zoom untouched — nothing is replotted on selection.  Double-tap a
  histogram to drop just that channel's term.  Switching between Cutoff and
  Brush does not replot either: both gestures are wired on every figure and the
  toggle only re-points ``toolbar.active_drag`` (#127 reply).

Python owns all binning (so the logic stays unit-testable); Bokeh only draws the
bars + brush and routes its events back to Python callbacks in the kernel.  When
the Bokeh stack is unavailable (headless / CI), the plugin degrades to a notice
and its selection logic remains callable.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd

import ipywidgets as _ipywidgets

Button = getattr(_ipywidgets, "Button")
Dropdown = getattr(_ipywidgets, "Dropdown")
HBox = getattr(_ipywidgets, "HBox")
HTML = getattr(_ipywidgets, "HTML")
IntSlider = getattr(_ipywidgets, "IntSlider", None)
Layout = getattr(_ipywidgets, "Layout")
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

# Bokeh + jupyter_bokeh are optional at import time so the plugin still imports
# headlessly (unit tests / CI). ``bokeh`` alone is enough to *build* a layout
# (used by tests); ``jupyter_bokeh`` is additionally needed to render it as an
# interactive ipywidget with kernel-side event callbacks.
try:  # pragma: no cover - exercised via the real notebook stack
    from bokeh.plotting import figure as _bk_figure
    from bokeh.models import (
        BoxAnnotation,
        BoxSelectTool,
        ColumnDataSource,
        PanTool,
        Span,
    )
    from bokeh.events import DoubleTap, SelectionGeometry, Tap
    from bokeh.layouts import column as _bk_column

    _BOKEH_OK = True
except Exception:  # pragma: no cover - bokeh missing
    _BOKEH_OK = False

try:  # pragma: no cover - exercised via the real notebook stack
    from jupyter_bokeh import BokehModel

    _JBOKEH_OK = True
except Exception:  # pragma: no cover - jupyter_bokeh missing
    _JBOKEH_OK = False

from ueler.viewer.decorators import update_status_bar
from ueler.viewer.observable import Observable

from . import _chart_common
from .plugin_base import PluginBase

_logger = logging.getLogger(__name__)

# BokehJS must be present in the notebook frontend for a `BokehModel` to render.
# JupyterLab's jupyter_bokeh extension loads it automatically, but VSCode's
# notebook frontend does not — there the widget stays blank until something calls
# `output_notebook()`. We load it once, up front, so the histogram renders without
# the user having to run a priming cell (#112 reply). Guarded so it is a no-op
# outside an interactive kernel (unit tests) and loads at most once per session.
_bokehjs_loaded = False


def _ensure_bokehjs() -> None:
    global _bokehjs_loaded
    if _bokehjs_loaded or not (_BOKEH_OK and _JBOKEH_OK):
        return
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return  # not in an interactive kernel (e.g. unit tests / headless)
        from bokeh.io import output_notebook

        output_notebook(hide_banner=True)
        _bokehjs_loaded = True
    except Exception:  # pragma: no cover - defensive; never block plugin load
        _logger.debug("Could not preload BokehJS via output_notebook()", exc_info=True)


_SELECTION_NOTICE = (
    "<i>No histograms yet. Choose one or more channels, then click <b>Plot</b>.</i>"
)
_BOKEH_MISSING_NOTICE = (
    "<b>Interactive histograms require Bokeh.</b> Install <code>bokeh</code> and "
    "<code>jupyter_bokeh</code> (both are UELer dependencies) and restart the kernel."
)

_BASE_COLOR = "#1f77b4"        # matplotlib tab:blue
_OVERLAY_COLOR = "#ff7f0e"     # matplotlib tab:orange
_BAND_COLOR = "#2ca02c"        # matplotlib tab:green — persistent gate band
_FIGURE_HEIGHT = 220
_ROW_OVERHEAD = 40             # approx. per-figure title + axis label DOM height
_MAX_PLOT_HEIGHT = 560         # scroll the stack once it exceeds this many px


# Gate-term kinds (issue #127). A term is ``(kind, a, b)``:
#   ("range",  lo,        hi)     — brushed [lo, hi] on that channel
#   ("cutoff", direction, value)  — "above"/"below" *value* on that channel
_RANGE = "range"
_CUTOFF = "cutoff"


def bin_counts(values, edges) -> np.ndarray:
    """Histogram counts of ``values`` over the explicit bin ``edges``.

    Pure helper (no Bokeh) so binning stays unit-testable. Empty input yields an
    all-zero vector of length ``len(edges) - 1``.
    """
    arr = np.asarray(list(values), dtype=float)
    counts, _ = np.histogram(arr, bins=edges)
    return counts


class HistogramDisplay(PluginBase):
    def __init__(self, main_viewer, width: float, height: float):
        super().__init__(main_viewer, width, height)
        self.SidePlots_id = "histogram_output"
        self.displayed_name = "Histogram"
        self.main_viewer = main_viewer
        self.width = width
        self.height = height

        # Cutoff-mode state. ``cutoff``/``_active_histogram_column`` track the
        # *last* cutoff the user set; the authoritative per-channel state lives in
        # ``_gates``. Kept because the viewer re-triggers ``highlight_cells()`` on
        # a FOV change and callers/tests set them directly.
        self.cutoff: Optional[float] = None
        self._active_histogram_column: Optional[str] = None

        # Per-channel gate terms, ANDed into one selection (#127). Replaces the
        # old single ``_brush_selection`` tuple, which could only gate on the
        # last channel touched.
        self._gates: dict = {}  # channel -> (kind, a, b)
        self.selected_indices: Observable = Observable(set())
        self.single_point_click_state = 0

        self._channels: list = []
        self._plot_data = None
        # Bokeh render state: per-channel selected-overlay sources (each also
        # carrying that channel's persistent range band) + cutoff spans.
        self._sources: dict = {}
        self._spans: dict = {}
        # Live figures + their box-select tools, kept so switching the
        # interaction mode can flip the drag gesture in place instead of
        # rebuilding the stack (#127 reply).
        self._figures: dict = {}
        self._box_tools: dict = {}
        self._bokeh_model = None
        self._observers_registered = False

        self.ui_component = UiComponent(self.main_viewer)
        self._plot_placeholder = HTML(value=_SELECTION_NOTICE, layout=Layout(width="100%"))
        self._plot_host = VBox(
            children=[self._plot_placeholder], layout=Layout(width="100%", gap="8px")
        )

        self._wire_events()
        self._build_layout()
        self.setup_widget_observers()
        # Load BokehJS while the viewer cell is executing (a reliable display
        # context), so the histogram renders on first Plot even in VSCode.
        _ensure_bokehjs()

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
        # Toggling the link must act on the highlight immediately (#129), not
        # only on the next selection.
        self.ui_component.mv_linked_checkbox.observe(
            self._on_mv_link_change, names="value"
        )
        self.ui_component.subset_on_dropdown.observe(
            self.on_subset_on_dropdown_change, names="value"
        )
        self.ui_component.clear_selection_button.on_click(
            lambda _btn: self.clear_selection()
        )
        self.ui_component.channel_selector_bundle.load_button.on_click(
            lambda _btn: _chart_common.apply_marker_set_to_selector(
                self.ui_component.channel_selector_bundle, self.main_viewer
            )
        )

    def on_marker_sets_changed(self):
        """Keep the marker-set dropdown in sync with the left panel (#113)."""
        _chart_common.refresh_marker_set_options(
            self.ui_component.channel_selector_bundle, self.main_viewer
        )

    def after_all_plugins_loaded(self):
        super().after_all_plugins_loaded()
        # Marker sets are restored from widget_states.json after plugin __init__,
        # so populate the dropdown once everything is loaded.
        self.on_marker_sets_changed()

    def _build_layout(self) -> None:
        plot_controls = VBox(
            children=[
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
                self.ui_component.gate_summary,
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
            children=[
                self.ui_component.channel_selector_bundle.box,
                self.ui_component.plot_button,
                self._plot_tabs,
            ],
            layout=Layout(width="100%", gap="10px"),
        )
        self.controls_section = VBox(children=[controls], layout=Layout(width="100%", gap="12px"))
        # Bound the plot area and scroll it, so stacking several histograms shows
        # an internal scrollbar instead of overflowing the plugin (#112 reply 2).
        # Controls stay fixed above the scroll region.
        # The scroll region is applied to the BokehModel widget in _render (not
        # here): Bokeh manages the column's height on its own DOM node, so a
        # max-height on this outer VBox never sees the overflow. See _render /
        # _scroll_height (#112 reply 2).
        self.plot_section = VBox(
            children=[self._plot_host], layout=Layout(width="100%")
        )
        self.ui = VBox(
            children=[self.controls_section, self.plot_section],
            layout=Layout(width="100%", gap="12px"),
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
        # A fresh plot invalidates any gate term on a column that is no longer
        # shown (#127) — the remaining terms keep gating.
        if self._active_histogram_column not in channels:
            self._active_histogram_column = None
            self.cutoff = None
        dropped = [ch for ch in self._gates if ch not in channels]
        for channel in dropped:
            del self._gates[channel]
        self._render()
        if dropped:
            # The gate changed, so the published selection must follow.
            self._apply_gate(publish=True, highlight=self.ui_component.mv_linked_checkbox.value)

    def _render(self) -> None:
        """Rebuild the Bokeh layout (one figure per channel) and host it."""
        data = self._plot_data
        channels = self._channels
        # Drop references to the outgoing figures up front; _build_figures
        # repopulates them, and a path that bails out must not leave a mode
        # switch pointing at detached figures.
        self._figures = {}
        self._box_tools = {}
        if data is None or not channels:
            self._plot_host.children = [self._plot_placeholder]
            return
        if not (_BOKEH_OK and _JBOKEH_OK):
            self._plot_host.children = [HTML(_BOKEH_MISSING_NOTICE)]
            return
        _ensure_bokehjs()  # backstop (idempotent) in case init couldn't load it

        layout, self._sources, self._spans = self._build_figures()
        self._bokeh_model = BokehModel(layout)
        # Scroll the stack once it gets tall. The scroll must live on the
        # BokehModel widget itself — Bokeh sizes the column on its own DOM node,
        # so a max-height on the parent VBox never triggers overflow (#112 reply 2).
        scroll_h = self._scroll_height()
        if scroll_h is not None:
            self._bokeh_model.layout.height = scroll_h
            self._bokeh_model.layout.overflow = "hidden auto"
        self._plot_host.children = [self._bokeh_model]
        # Reflect any existing selection / gate terms on the freshly built figures.
        self._refresh_overlays()
        self._refresh_gate_markers()

    def _scroll_height(self):
        """Return a fixed pixel height ('<N>px') once the stack exceeds the cap,
        else ``None`` (few histograms render at natural height, no scrollbar)."""
        total = len(self._channels) * (_FIGURE_HEIGHT + _ROW_OVERHEAD)
        return f"{_MAX_PLOT_HEIGHT}px" if total > _MAX_PLOT_HEIGHT else None

    def _build_figures(self):
        """Build per-channel Bokeh figures. Returns ``(layout, sources, spans)``.

        Uses only ``bokeh`` (not ``jupyter_bokeh``) so it is unit-testable.
        Python computes the bins; each figure draws a ``quad`` for the full
        counts and a second ``quad`` (fed by ``sources[channel]``) for the
        selected-subset overlay, both on the **same** edges.

        Each figure also gets a ``BoxAnnotation`` (in ``sources[channel]["band"]``)
        that draws this channel's brushed range, and a ``Span`` for its cutoff.
        Those are *ours*, not Bokeh's transient box-select overlay, so a gate term
        stays visible after the user acts on a different histogram (#127).

        Both gestures (box-select and tap) are wired on **every** figure whatever
        the current mode, so a mode switch only has to re-point the active drag
        tool — see ``_apply_interaction_mode`` (#127 reply).
        """
        bins = self.ui_component.bin_slider.value
        data = self._plot_data

        figures = []
        sources: dict = {}
        spans: dict = {}
        self._figures = {}
        self._box_tools = {}
        for channel in self._channels:
            edges = self._histogram_bin_edges(channel, bins)
            left = edges[:-1].tolist()
            right = edges[1:].tolist()
            full = bin_counts(data[channel], edges).tolist()
            full_src = ColumnDataSource(dict(left=left, right=right, top=full))
            sel_src = ColumnDataSource(
                dict(left=left, right=right, top=[0] * len(full))
            )

            p = _bk_figure(
                height=_FIGURE_HEIGHT,
                sizing_mode="stretch_width",
                tools="pan,wheel_zoom,reset",
                title=channel,
            )
            full_r = p.quad(
                left="left", right="right", bottom=0, top="top",
                source=full_src, fill_color=_BASE_COLOR, line_color="white",
                fill_alpha=0.6, legend_label="All",
            )
            sel_r = p.quad(
                left="left", right="right", bottom=0, top="top",
                source=sel_src, fill_color=_OVERLAY_COLOR, line_color="white",
                fill_alpha=0.75, legend_label="Selected",
            )
            # Pin the selection glyphs to the base glyph so a box-select gesture
            # never mutes the bars: what the user reads is our own band + overlay,
            # which survives acting on another histogram (#127).
            for renderer in (full_r, sel_r):
                renderer.selection_glyph = renderer.glyph
                renderer.nonselection_glyph = renderer.glyph

            p.xaxis.axis_label = channel
            p.yaxis.axis_label = "Cell count"
            p.legend.click_policy = "hide"

            span = Span(
                location=0, dimension="height",
                line_color="red", line_dash="dashed", line_width=2, visible=False,
            )
            p.add_layout(span)

            band = BoxAnnotation(
                left=0, right=0, fill_color=_BAND_COLOR, fill_alpha=0.12,
                line_color=_BAND_COLOR, line_alpha=0.6, line_width=1, visible=False,
            )
            p.add_layout(band)

            # Double-tap clears just this channel's gate term (#127).
            p.on_event(DoubleTap, self._make_clear_gate_handler(channel))

            # The box-select tool exists on every figure; whether it *owns* the
            # click-drag gesture is decided by _apply_interaction_mode below.
            # Adding the tool is not enough — it has to be the active drag,
            # otherwise the default (pan) still handles click-drag and no range
            # can be brushed (#112 reply).
            box = BoxSelectTool(dimensions="width")
            p.add_tools(box)
            p.on_event(SelectionGeometry, self._make_range_handler(channel))
            p.on_event(Tap, self._make_tap_handler(channel))

            figures.append(p)
            sources[channel] = {"selected": sel_src, "edges": edges, "band": band}
            spans[channel] = span
            self._figures[channel] = p
            self._box_tools[channel] = box

        self._apply_interaction_mode()
        layout = _bk_column(*figures, sizing_mode="stretch_width")
        return layout, sources, spans

    def _brush_mode(self) -> bool:
        return self.ui_component.interaction_mode.value == "Brush"

    @staticmethod
    def _pan_tool(p):
        """The figure's PanTool, so cutoff mode can hand the drag back to it."""
        for tool in p.tools:
            if isinstance(tool, PanTool):
                return tool
        return "auto"

    def _apply_interaction_mode(self) -> None:
        """Point each live figure's drag gesture at the current mode (#127 reply).

        Brush mode makes the ``BoxSelectTool`` the active drag; cutoff mode hands
        the drag back to pan so a click can register as a tap. Both are property
        writes on figures that already exist, so switching modes keeps the bars,
        the gate markers, the overlay and — the point of the fix — the user's
        zoom/pan. Rebuilding the figures (the old behaviour) threw all of that away.
        """
        brush_mode = self._brush_mode()
        for channel, p in self._figures.items():
            box = self._box_tools.get(channel)
            if box is None:
                continue
            p.toolbar.active_drag = box if brush_mode else self._pan_tool(p)

    def _make_range_handler(self, channel: str):
        """Bokeh ``SelectionGeometry`` → ``handle_range`` (brush mode)."""

        def _handler(event):
            # SelectionGeometry fires during the drag too; only act on the final
            # (mouse-up) event so we compute the selection once per gesture.
            if not getattr(event, "final", True):
                return
            geom = getattr(event, "geometry", None) or {}
            x0 = geom.get("x0")
            x1 = geom.get("x1")
            if x0 is None or x1 is None:
                return
            self.handle_range(channel, float(x0), float(x1))

        return _handler

    def _make_tap_handler(self, channel: str):
        """Bokeh ``Tap`` → set the cutoff for ``channel`` (cutoff mode only).

        Registered on every figure, so it has to ignore taps raised while the
        user is brushing: a bare click is ambiguous, and Bokeh fires ``Tap`` for
        one whatever the active tool is. A box-select gesture is *not* ambiguous,
        so ``_make_range_handler`` accepts it in either mode.
        """

        def _handler(event):
            if self._brush_mode():
                return
            x = getattr(event, "x", None)
            if x is None:
                return
            self.cutoff = float(x)
            self._active_histogram_column = channel
            _logger.info("Cutoff set at %.3f on channel %s", self.cutoff, channel)
            # highlight_cells() records the cutoff as this channel's gate term and
            # refreshes the overlay + markers itself.
            self.highlight_cells(push_to_gallery=True)

        return _handler

    def _make_clear_gate_handler(self, channel: str):
        """Bokeh ``DoubleTap`` → drop *channel*'s gate term, keeping the others (#127)."""

        def _handler(_event):
            self.clear_gate(channel)

        return _handler

    def _refresh_overlays(self, indices: Optional[Set[Union[int, str]]] = None) -> None:
        """Recompute the selected-subset bar counts for every built figure.

        Touches only the ``selected`` ColumnDataSource of each figure — never the
        figures, axes or bin edges — so refreshing the overlay cannot disturb a
        zoom/pan or another channel's gate marker (#127).
        """
        if not self._sources or self._plot_data is None:
            return
        selected = (self.selected_indices.value if indices is None else indices) or set()
        valid = [i for i in selected if i in self._plot_data.index]
        for channel, info in self._sources.items():
            edges = info["edges"]
            if valid:
                counts = bin_counts(self._plot_data.loc[valid, channel], edges)
            else:
                counts = np.zeros(len(edges) - 1, dtype=int)
            info["selected"].data = dict(
                left=edges[:-1].tolist(),
                right=edges[1:].tolist(),
                top=counts.tolist(),
            )

    def _refresh_gate_markers(self) -> None:
        """Draw every gated channel's own marker — range band or cutoff line.

        Each histogram shows *its* term (#127), not just the last one touched, so a
        gate can be built up channel by channel. Channels without a term show
        nothing. Only annotation properties are written; no figure is rebuilt.
        """
        for channel, span in self._spans.items():
            term = self._gates.get(channel)
            if term is not None and term[0] == _CUTOFF:
                span.location = term[2]
                span.visible = True
            else:
                span.visible = False

        for channel, info in self._sources.items():
            band = info.get("band")
            if band is None:
                continue
            term = self._gates.get(channel)
            if term is not None and term[0] == _RANGE:
                band.left, band.right = term[1], term[2]
                band.visible = True
            else:
                band.visible = False

        self._refresh_gate_summary()

    # Legacy name kept for callers/tests that predate per-channel gating (#127).
    def _refresh_cutoff_spans(self) -> None:
        self._refresh_gate_markers()

    def _refresh_gate_summary(self) -> None:
        """Restate the active gate in words, so it is readable at a glance."""
        label = getattr(self.ui_component, "gate_summary", None)
        if label is None:
            return
        label.value = f"<i>{self.gate_description()}</i>"

    def gate_description(self) -> str:
        """Human-readable rendering of the current gate (``AND`` over all terms)."""
        if not self._gates:
            return "No gate — brush or tap a histogram to start one."
        parts = []
        for channel, (kind, a, b) in self._gates.items():
            if kind == _RANGE:
                lo, hi = (a, b) if a <= b else (b, a)
                parts.append(f"{channel} ∈ [{lo:.3g}, {hi:.3g}]")
            else:
                parts.append(f"{channel} {'>' if a == 'above' else '<'} {b:.3g}")
        return "Gate: " + " AND ".join(parts)

    def _histogram_bin_edges(self, channel: str, bins: int):
        """Bin edges computed over the *full* plotted data for ``channel``.

        Shared by the base and subset-overlay bars so both sit on the same grid;
        independent of the current selection (#112 reply). ``_plot_data`` is
        already NaN-dropped on the plotted channels by ``_prepare_dataframe``.
        """
        return np.histogram_bin_edges(self._plot_data[channel], bins=bins)

    # ------------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------------
    def _gate_frame(self):
        """The frame the gate terms are evaluated on.

        The plotted (subset-filtered, NaN-dropped) ``_plot_data`` when a plot
        exists, so a gate always means the same thing whichever mode produced its
        terms; the full cell table otherwise, which is what a cutoff set before any
        plot (or re-applied on a FOV change) has to fall back to.
        """
        if self._plot_data is not None:
            return self._plot_data
        return getattr(self.main_viewer, "cell_table", None)

    @staticmethod
    def _term_mask(frame, channel: str, term: tuple):
        """Boolean mask of the rows of ``frame`` satisfying one gate ``term``."""
        kind, a, b = term
        if kind == _RANGE:
            lo, hi = (a, b) if a <= b else (b, a)
            return frame[channel].between(lo, hi)
        comparator = np.greater if a == "above" else np.less
        return pd.Series(comparator(frame[channel], b), index=frame.index)

    def _cells_in_range(self, channel: str, lo: float, hi: float) -> Set[Union[int, str]]:
        """Row indices of the (filtered) data whose ``channel`` value is within [lo, hi]."""
        data = self._plot_data
        if data is None or channel not in data.columns:
            return set()
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        mask = data[channel].between(lo, hi)
        return set(data.index[mask])

    def gated_indices(self) -> Set[Union[int, str]]:
        """Row indices satisfying **every** active gate term (#127).

        The intersection is the whole point: a range on CD4 plus a cutoff on CD8
        selects the cells that pass both. With a single term this is exactly the
        old single-channel behaviour. Terms naming a column absent from the frame
        are skipped rather than emptying the gate.
        """
        frame = self._gate_frame()
        if frame is None or not self._gates:
            return set()
        mask = None
        for channel, term in self._gates.items():
            if channel not in frame.columns:
                continue
            channel_mask = self._term_mask(frame, channel, term)
            mask = channel_mask if mask is None else (mask & channel_mask)
        if mask is None:
            return set()
        return set(frame.index[mask])

    def _apply_gate(self, *, publish: bool = True, highlight: bool = False) -> None:
        """Recompute the gated selection and reflect it everywhere.

        Deliberately narrow: it publishes ``selected_indices``, optionally pushes
        mask highlights, and rewrites only the overlay sources + gate annotations.
        It never calls ``_render``/``plot_histograms``, so a selection cannot
        replot the stack or lose a zoom/pan (#127; cf. #109, #119).
        """
        indices = _chart_common.normalize_indices(self.gated_indices())
        if publish:
            self._update_single_point_state(indices)
            self.selected_indices.value = indices
        if highlight:
            _chart_common.sync_mask_highlights_from_selection(self.main_viewer, indices)
        self._refresh_overlays(indices)
        self._refresh_gate_markers()

    def set_gate(self, channel: str, term: tuple, *, publish: bool = True) -> None:
        """Set (or replace) ``channel``'s gate term, leaving other channels' alone."""
        self._gates[channel] = term
        self._apply_gate(
            publish=publish, highlight=self.ui_component.mv_linked_checkbox.value
        )

    def clear_gate(self, channel: str) -> None:
        """Drop just ``channel``'s gate term; the remaining terms keep gating (#127)."""
        if self._gates.pop(channel, None) is None:
            return
        if self._active_histogram_column == channel:
            self._active_histogram_column = None
            self.cutoff = None
        _logger.info("Cleared the gate term on channel %s", channel)
        self._apply_gate(
            publish=True, highlight=self.ui_component.mv_linked_checkbox.value
        )

    def handle_range(self, channel: str, lo: float, hi: float) -> None:
        """Record a brushed [lo, hi] on ``channel`` as that channel's gate term.

        Pure of Bokeh (event handlers delegate here), so it is unit-testable with
        plain floats. The published selection is the intersection over all gated
        channels (#127); a brush supersedes any cutoff previously set on the same
        channel. Other histograms keep their own markers and zoom.
        """
        if lo == hi:
            return
        if self._active_histogram_column == channel:
            # This channel is now range-gated, so its cutoff no longer applies.
            self._active_histogram_column = None
            self.cutoff = None
        self.set_gate(channel, (_RANGE, float(lo), float(hi)))

    # Backwards-compatible alias (kept for callers/tests using the old name).
    def _on_brush(self, channel: str, lo: float, hi: float) -> None:
        self.handle_range(channel, lo, hi)

    def highlight_cells(self, *, push_to_gallery: bool = False) -> None:
        """Apply the gate, recording the pending cutoff as its channel's term.

        The cutoff now participates in the same intersection as brushed ranges
        (#127): tapping channel A while channel B is gated selects the cells that
        pass both. ``cutoff``/``_active_histogram_column`` carry the cutoff the user
        just set (or one a caller assigned directly, e.g. the viewer re-applying a
        highlight after a FOV change); it is folded into ``_gates`` here, with the
        above/below direction captured now so each channel keeps its own.

        Mask highlights are pushed on exactly the same condition as a brush — the
        "Main viewer" link (#129). This used to highlight unconditionally, which
        made a cutoff tap (and the FOV-change re-apply that goes through here)
        outline cells in the viewer even with the link switched off.
        """
        channel = self._active_histogram_column
        if channel is not None and self.cutoff is not None:
            direction = self.ui_component.above_below_buttons.value or "below"
            self._gates[channel] = (_CUTOFF, direction, float(self.cutoff))
        if not self._gates:
            _logger.warning("No active channel or cutoff set.")
            return
        frame = self._gate_frame()
        if frame is None or not any(ch in frame.columns for ch in self._gates):
            return
        self._apply_gate(
            publish=push_to_gallery,
            highlight=self.ui_component.mv_linked_checkbox.value,
        )

    def clear_selection(self) -> None:
        """Drop every gate term (both kinds) and the selection it produced."""
        self._gates = {}
        self.cutoff = None
        self._active_histogram_column = None
        self.selected_indices.value = set()
        if self.ui_component.mv_linked_checkbox.value:
            _chart_common.sync_mask_highlights_from_selection(self.main_viewer, set())
        self._refresh_overlays()
        self._refresh_gate_markers()

    def show_external_selection(self, row_indices: Iterable[Union[int, str]]) -> None:
        """Overlay an externally-supplied selection as the "Selected" distribution.

        Entry point for *other* plugins (e.g. the heatmap "Histogram" link) to
        push a set of cell-table row indices into this plugin. The selection is
        published on ``selected_indices`` (so cell-gallery forwarding still works
        when linked) and drawn on every plotted histogram via the same overlay
        machinery that brush selections use. Indices that fall outside the
        currently plotted data are ignored by ``_refresh_overlays``, so the
        overlay reflects whatever channels/subset the histogram is showing.

        An external selection replaces the local gate rather than intersecting with
        it (#127) — the incoming indices come from another plugin's own criteria, so
        leaving stale gate terms drawn would misrepresent what is selected.
        """
        # A programmatic push is never a single-point viewer focus.
        self.single_point_click_state = 0
        self._gates = {}
        self.cutoff = None
        self._active_histogram_column = None
        self.selected_indices.value = _chart_common.normalize_indices(row_indices)
        if self.ui_component.mv_linked_checkbox.value:
            _chart_common.sync_mask_highlights_from_selection(
                self.main_viewer, self.selected_indices.value
            )
        self._refresh_overlays()
        self._refresh_gate_markers()

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
        """Flip the direction of the *last* cutoff; other channels keep theirs (#127)."""
        channel = self._active_histogram_column
        if channel is None or self.cutoff is None:
            return
        self.highlight_cells(push_to_gallery=True)

    def _on_interaction_mode_change(self, _change) -> None:
        """Switch the drag gesture in place; never replot (#127 reply).

        This used to call ``_render()``, which rebuilt the whole Bokeh layout —
        the gate survived (it lives in ``_gates``) but the plot visibly reset,
        losing zoom/pan, exactly the kind of refresh #127 set out to remove.
        """
        self._apply_interaction_mode()

    def _on_mv_link_change(self, _change) -> None:
        self.sync_main_viewer_link()

    def sync_main_viewer_link(self) -> None:
        """Push or withdraw this plugin's mask highlights for the current link state (#129).

        Unchecking "Main viewer" has to *take the highlight away*, not merely stop
        updating it: outlines drawn while linked would otherwise stay on the canvas
        and read as a live link. Clearing goes through
        ``sync_mask_highlights_from_selection(..., set())`` so the
        ``linked_selection_indices`` record (#119) is dropped too and a later FOV
        switch cannot resurrect the highlight. Re-checking re-projects the current
        selection onto the active FOV.

        A no-op when this plugin has published no selection, so toggling an idle
        histogram cannot wipe a highlight the scatter or heatmap plugin owns. The
        checkbox is read directly rather than from the ``change`` payload, which
        keeps the handler independent of the observer's change object.
        """
        indices = self.selected_indices.value or set()
        if not indices:
            return
        linked = bool(self.ui_component.mv_linked_checkbox.value)
        _chart_common.sync_mask_highlights_from_selection(
            self.main_viewer, indices if linked else set()
        )

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
        # Left-panel-consistent channel picker with marker-set loading (#113).
        # ``channel_selector`` stays as an alias to the bundle's TagsInput so
        # ``plot_histograms`` (which reads ``channel_selector.value``) is unchanged.
        self.channel_selector_bundle = _chart_common.build_channel_selector(viewer)
        self.channel_selector = self.channel_selector_bundle.tags
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
            continuous_update=False,
            style={"description_width": "auto"},
            layout=Layout(width="250px"),
        )
        self.interaction_mode = ToggleButtons(
            options=["Cutoff", "Brush"],
            value="Cutoff",
            description="Interaction:",
            tooltips=[
                "Click a histogram to set an above/below cutoff on that channel",
                "Drag a range to gate on that channel",
            ],
            style={"description_width": "auto"},
        )
        # Gating is only useful if the user can see which terms are active (#127).
        self.gate_summary = HTML(
            value="<i>No gate — brush or tap a histogram to start one.</i>",
            layout=Layout(width="100%"),
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
            tooltip=(
                "Clear every channel's gate term "
                "(double-click a histogram to clear just that channel)"
            ),
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
