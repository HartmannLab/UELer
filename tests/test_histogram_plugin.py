"""Tests for the standalone Histogram plugin (issue #112).

Covers feature-parity with the old histogram (cutoff → above/below highlight →
cell-gallery forwarding) plus the new linked-brushing behaviour (a range brush
selects cells and feeds ``selected_indices``, which drives the cross-histogram
overlay and the cell gallery).
"""
from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

# ---------------------------------------------------------------------------
# Lightweight stubs for optional UI / scientific dependencies
# ---------------------------------------------------------------------------
try:
    import ipywidgets as _ipywidgets  # type: ignore
except ImportError:  # pragma: no cover - stub fallback
    _ipywidgets = types.ModuleType("ipywidgets")
    sys.modules["ipywidgets"] = _ipywidgets


class _Layout:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Widget:
    def __init__(self, *args, **kwargs):
        children = kwargs.get("children")
        if children is None and args:
            children = args[0]
        self.children = tuple(children or ())
        self.value = kwargs.get("value")
        self.options = kwargs.get("options", [])
        self.layout = kwargs.get("layout", _Layout())
        self.description = kwargs.get("description", "")
        self._observers: list = []

    def observe(self, callback, names=None):
        self._observers.append((callback, names))

    def _trigger(self, new_value):
        self.value = new_value
        for cb, _ in self._observers:
            cb(SimpleNamespace(new=new_value))

    def on_click(self, *_, **__):
        return None

    def clear_output(self, *_, **__):
        return None

    def set_title(self, *_, **__):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


for _wname in [
    "Button", "Checkbox", "Dropdown", "FloatSlider", "HBox", "HTML",
    "IntSlider", "IntText", "Layout", "Output", "SelectMultiple", "Tab",
    "TagsInput", "Text", "ToggleButtons", "VBox", "Widget",
]:
    if not hasattr(_ipywidgets, _wname):
        setattr(_ipywidgets, _wname, _Widget)

if not hasattr(_ipywidgets, "Layout"):
    _ipywidgets.Layout = _Layout  # type: ignore[attr-defined]

for _mod in ["seaborn_image", "tifffile", "cv2", "dask"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

import numpy as np
import pandas as pd

from ueler.viewer.plugin.histogram import HistogramDisplay


def _bokeh_available() -> bool:
    """True when Bokeh is importable (enough to build a figure/layout)."""
    from ueler.viewer.plugin import histogram as _h

    return bool(_h._BOKEH_OK)


def _bokeh_stack_available() -> bool:
    """True when both bokeh and jupyter_bokeh are importable (full interactive render)."""
    from ueler.viewer.plugin import histogram as _h

    return bool(_h._BOKEH_OK and _h._JBOKEH_OK)


# ---------------------------------------------------------------------------
# Minimal viewer stub (mirrors test_chart_cell_gallery_link)
# ---------------------------------------------------------------------------

class _FakeImageDisplay:
    def __init__(self):
        self.last_mask_ids: list = []
        self.last_fov_mask_pairs = None

    def set_mask_ids(self, *, mask_name, mask_ids, fov_mask_pairs=None):
        if fov_mask_pairs is not None:
            self.last_fov_mask_pairs = list(fov_mask_pairs)
            self.last_mask_ids = []
        else:
            self.last_mask_ids = list(mask_ids)
            self.last_fov_mask_pairs = None


class _FakeCellGallery:
    def __init__(self):
        self.received: object = None

    def set_selected_cells(self, indices):
        self.received = indices


def _make_viewer(cell_table: "pd.DataFrame") -> SimpleNamespace:
    gallery = _FakeCellGallery()
    image_display = _FakeImageDisplay()
    ui_component = SimpleNamespace(image_selector=SimpleNamespace(value="fov1"))
    side_plots = SimpleNamespace(cell_gallery_output=gallery)
    return SimpleNamespace(
        cell_table=cell_table,
        fov_key="fov",
        label_key="label",
        mask_key="cells",
        ui_component=ui_component,
        image_display=image_display,
        SidePlots=side_plots,
        get_active_fov=lambda: ui_component.image_selector.value,
    )


def _make_histogram(viewer: SimpleNamespace, *, patch_render=True) -> HistogramDisplay:
    hist = HistogramDisplay(viewer, width=4, height=3)
    hist.setup_observe()
    if patch_render:
        # Decouple selection logic from real Matplotlib figure building.
        hist._render_calls = 0

        def _fake_render():
            hist._render_calls += 1

        hist._render = _fake_render
    return hist


def _two_fov_table() -> "pd.DataFrame":
    return pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov1", "fov2", "fov2"],
            "label": [1, 2, 3, 4, 5],
            "intensity": [1.0, 5.0, 9.0, 3.0, 7.0],
            "area": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


# ---------------------------------------------------------------------------
# Cutoff-mode parity tests
# ---------------------------------------------------------------------------

class TestHistogramCutoffGalleryLink(unittest.TestCase):
    """highlight_cells() → selected_indices → cell gallery forwarding (parity)."""

    def setUp(self):
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer)
        self.gallery: _FakeCellGallery = self.viewer.SidePlots.cell_gallery_output
        self.hist._active_histogram_column = "intensity"
        self.hist.ui_component.above_below_buttons.value = "above"
        self.hist.ui_component.cell_gallery_linked_checkbox.value = True
        self.hist.cutoff = 4.0

    def test_gallery_receives_all_fov_indices_above_cutoff(self):
        self.hist.highlight_cells(push_to_gallery=True)
        self.assertIsNotNone(self.gallery.received)
        table = self.viewer.cell_table
        expected = set(table.loc[table["intensity"] > 4.0].index)
        self.assertEqual(set(self.gallery.received), expected)

    def test_gallery_not_updated_when_checkbox_off(self):
        self.hist.ui_component.cell_gallery_linked_checkbox.value = False
        self.hist.highlight_cells(push_to_gallery=True)
        self.assertIsNone(self.gallery.received)

    def test_gallery_receives_indices_below_cutoff(self):
        self.hist.ui_component.above_below_buttons.value = "below"
        self.hist.highlight_cells(push_to_gallery=True)
        table = self.viewer.cell_table
        expected = set(table.loc[table["intensity"] < 4.0].index)
        self.assertEqual(set(self.gallery.received), expected)

    def test_image_display_limited_to_current_fov(self):
        self.hist.highlight_cells(push_to_gallery=True)
        img = self.viewer.image_display
        self.assertNotIn(1, img.last_mask_ids)
        self.assertIn(2, img.last_mask_ids)
        self.assertIn(3, img.last_mask_ids)
        for label in (4, 5):
            self.assertNotIn(label, img.last_mask_ids)

    def test_map_mode_uses_fov_mask_pairs(self):
        self.viewer.get_active_fov = lambda: None
        self.hist.highlight_cells(push_to_gallery=True)
        pairs = self.viewer.image_display.last_fov_mask_pairs
        self.assertIsNotNone(pairs)
        # intensity > 4.0 → labels 2,3 (fov1) and 5 (fov2)
        self.assertIn(("fov1", 2), pairs)
        self.assertIn(("fov2", 5), pairs)

    def test_no_crash_when_cutoff_is_none(self):
        self.hist.cutoff = None
        self.hist.highlight_cells(push_to_gallery=True)
        self.assertIsNone(self.gallery.received)

    def test_no_crash_when_no_active_channel(self):
        self.hist._active_histogram_column = None
        self.hist.highlight_cells(push_to_gallery=True)
        self.assertIsNone(self.gallery.received)

    def test_auto_rerender_does_not_update_gallery(self):
        self.gallery.received = {99}
        self.hist.highlight_cells(push_to_gallery=False)
        self.assertEqual(self.gallery.received, {99})


# ---------------------------------------------------------------------------
# Brush-mode / linked-selection tests
# ---------------------------------------------------------------------------

class TestHistogramBrushLinking(unittest.TestCase):
    def setUp(self):
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer)
        self.gallery: _FakeCellGallery = self.viewer.SidePlots.cell_gallery_output
        self.hist._plot_data = self.viewer.cell_table.copy()
        self.hist._channels = ["intensity", "area"]

    def test_cells_in_range_returns_indices_within_bounds(self):
        idx = self.hist._cells_in_range("intensity", 4.0, 8.0)
        table = self.viewer.cell_table
        expected = set(table.loc[table["intensity"].between(4.0, 8.0)].index)
        self.assertEqual(idx, expected)

    def test_cells_in_range_handles_reversed_bounds(self):
        forward = self.hist._cells_in_range("intensity", 4.0, 8.0)
        reversed_ = self.hist._cells_in_range("intensity", 8.0, 4.0)
        self.assertEqual(forward, reversed_)

    def test_brush_publishes_selected_indices(self):
        self.hist._on_brush("intensity", 4.0, 8.0)
        table = self.viewer.cell_table
        expected = set(table.loc[table["intensity"].between(4.0, 8.0)].index)
        self.assertEqual(set(self.hist.selected_indices.value), expected)

    def test_brush_forwards_to_gallery_when_linked(self):
        self.hist.ui_component.cell_gallery_linked_checkbox.value = True
        # 3 cells in range so the single-point guard does not suppress forwarding.
        self.hist._on_brush("area", 15.0, 55.0)
        self.assertIsNotNone(self.gallery.received)
        self.assertEqual(set(self.gallery.received), {1, 2, 3, 4})

    def test_brush_highlights_viewer_when_mv_linked(self):
        self.hist.ui_component.mv_linked_checkbox.value = True
        self.hist._on_brush("intensity", 4.0, 10.0)
        # fov1 labels with intensity in [4,10]: labels 2 (5.0), 3 (9.0)
        self.assertIn(2, self.viewer.image_display.last_mask_ids)
        self.assertIn(3, self.viewer.image_display.last_mask_ids)

    def test_handle_range_is_the_brush_alias(self):
        """`_on_brush` delegates to the public `handle_range` (same effect)."""
        self.hist._on_brush("intensity", 4.0, 8.0)
        via_alias = set(self.hist.selected_indices.value)
        self.hist.clear_selection()
        self.hist.handle_range("intensity", 4.0, 8.0)
        self.assertEqual(set(self.hist.selected_indices.value), via_alias)

    def test_clear_selection_empties_indices(self):
        self.hist._on_brush("intensity", 4.0, 8.0)
        self.assertTrue(self.hist.selected_indices.value)
        self.hist.clear_selection()
        self.assertEqual(self.hist.selected_indices.value, set())

    def test_bin_counts_matches_numpy_histogram(self):
        from ueler.viewer.plugin.histogram import bin_counts

        edges = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        counts = bin_counts(self.viewer.cell_table["intensity"], edges)
        expected, _ = np.histogram(self.viewer.cell_table["intensity"], bins=edges)
        self.assertTrue(np.array_equal(counts, expected))

    @unittest.skipUnless(_bokeh_available(), "bokeh not available")
    def test_overlay_source_counts_match_selection(self):
        """After a brush, each figure's 'selected' source counts the selected cells."""
        layout, sources, spans = self.hist._build_figures()
        self.hist._sources, self.hist._spans = sources, spans
        self.hist.handle_range("intensity", 4.0, 8.0)
        # intensity in [4,8] → rows with 5.0 and 7.0 → 2 cells.
        total = sum(sources["intensity"]["selected"].data["top"])
        self.assertEqual(total, 2)

    def test_bin_edges_span_full_data_range(self):
        """Edges cover the full column range and have bins+1 entries (#112 reply)."""
        edges = self.hist._histogram_bin_edges("intensity", 20)
        col = self.viewer.cell_table["intensity"]
        self.assertEqual(len(edges), 21)
        self.assertLessEqual(edges[0], col.min())
        self.assertGreaterEqual(edges[-1], col.max())

    def test_bin_edges_independent_of_selection(self):
        """The overlay must reuse the full-data edges, not the subset's own range.

        Regression for the #112 reply: a narrow subset selection must not change
        the bin grid, otherwise the overlay is squeezed into its own bins and is
        not comparable to the full histogram.
        """
        before = self.hist._histogram_bin_edges("intensity", 15)
        # Select a narrow subset, then recompute — edges must be unchanged.
        self.hist._on_brush("intensity", 4.5, 5.5)
        after = self.hist._histogram_bin_edges("intensity", 15)
        self.assertTrue(np.array_equal(before, after))


# ---------------------------------------------------------------------------
# Multi-channel plot state
# ---------------------------------------------------------------------------

class TestHistogramMultiChannel(unittest.TestCase):
    def setUp(self):
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer)

    def test_plot_histograms_records_selected_channels(self):
        self.hist.ui_component.channel_selector.value = ("intensity", "area")
        self.hist.plot_histograms(None)
        self.assertEqual(self.hist._channels, ["intensity", "area"])
        self.assertIsNotNone(self.hist._plot_data)
        self.assertGreater(self.hist._render_calls, 0)

    def test_plot_histograms_no_channels_is_noop(self):
        self.hist.ui_component.channel_selector.value = ()
        self.hist.plot_histograms(None)
        self.assertEqual(self.hist._channels, [])

    def test_ensure_bokehjs_is_noop_outside_a_kernel(self):
        """Preloading BokehJS must not raise or mark loaded when there's no IPython kernel."""
        from ueler.viewer.plugin import histogram as h

        h._bokehjs_loaded = False
        h._ensure_bokehjs()  # unit tests have no interactive kernel → no-op
        self.assertFalse(h._bokehjs_loaded)

    def test_scroll_height_kicks_in_only_when_stack_is_tall(self):
        """`_scroll_height` returns a fixed px height once the stack exceeds the cap.

        The scroll is applied to the BokehModel in `_render`; ipywidgets 8 removed
        the per-axis overflow traits, and a `max-height` on the parent VBox does not
        clip the Bokeh column, so the height must live on the model itself (#112 reply 2).
        """
        from ueler.viewer.plugin.histogram import (
            _FIGURE_HEIGHT, _MAX_PLOT_HEIGHT, _ROW_OVERHEAD,
        )

        per = _FIGURE_HEIGHT + _ROW_OVERHEAD
        few = max(1, _MAX_PLOT_HEIGHT // per)          # fits within the cap
        many = (_MAX_PLOT_HEIGHT // per) + 2           # exceeds the cap

        self.hist._channels = ["c%d" % i for i in range(few)]
        self.assertIsNone(self.hist._scroll_height())

        self.hist._channels = ["c%d" % i for i in range(many)]
        self.assertEqual(self.hist._scroll_height(), f"{_MAX_PLOT_HEIGHT}px")


class TestHistogramChannelSelector(unittest.TestCase):
    """Left-panel-consistent channel selector + marker-set loading (#113)."""

    def setUp(self):
        self.viewer = _make_viewer(_two_fov_table())
        # A left-panel channel selector we can assert is NOT mutated by loading.
        self.viewer.ui_component.channel_selector = SimpleNamespace(value=("untouched",))
        self.viewer.marker_sets = {}
        self.hist = _make_histogram(self.viewer)

    def test_channel_selector_is_shared_bundle(self):
        bundle = self.hist.ui_component.channel_selector_bundle
        self.assertIs(self.hist.ui_component.channel_selector, bundle.tags)

    def test_load_marker_set_populates_channels_locally(self):
        self.viewer.marker_sets = {
            "T cells": {"selected_channels": ["intensity", "area"]}
        }
        self.hist.on_marker_sets_changed()
        bundle = self.hist.ui_component.channel_selector_bundle
        bundle.marker_set_dropdown.value = "T cells"
        from ueler.viewer.plugin import _chart_common

        _chart_common.apply_marker_set_to_selector(bundle, self.viewer)
        self.assertEqual(list(bundle.tags.value), ["intensity", "area"])
        # Loading a set into the plugin must not disturb the left-panel selector.
        self.assertEqual(self.viewer.ui_component.channel_selector.value, ("untouched",))

    def test_load_marker_set_filters_unknown_channels(self):
        self.viewer.marker_sets = {
            "mixed": {"selected_channels": ["intensity", "does_not_exist", "fov"]}
        }
        self.hist.on_marker_sets_changed()
        bundle = self.hist.ui_component.channel_selector_bundle
        bundle.marker_set_dropdown.value = "mixed"
        from ueler.viewer.plugin import _chart_common

        _chart_common.apply_marker_set_to_selector(bundle, self.viewer)
        # Only numeric cell-table columns survive; "fov" (object) and the absent
        # channel are filtered out.
        self.assertEqual(list(bundle.tags.value), ["intensity"])


class TestHistogramBokehLayout(unittest.TestCase):
    """Build the Bokeh layout (bokeh only; no jupyter_bokeh needed)."""

    def setUp(self):
        if not _bokeh_available():
            self.skipTest("bokeh not available in this environment")
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer)
        self.hist._plot_data = self.viewer.cell_table.copy()

    def test_build_figures_one_per_channel_with_shared_edges(self):
        self.hist._channels = ["intensity", "area"]
        layout, sources, spans = self.hist._build_figures()
        # A figure (and sources/spans) per channel.
        self.assertEqual(set(sources), {"intensity", "area"})
        self.assertEqual(set(spans), {"intensity", "area"})
        # The selected overlay shares the same bin edges as the full histogram.
        edges = sources["intensity"]["edges"]
        self.assertEqual(
            sources["intensity"]["selected"].data["left"], edges[:-1].tolist()
        )

    def test_cutoff_span_shows_only_on_active_channel(self):
        self.hist._channels = ["intensity", "area"]
        _layout, sources, spans = self.hist._build_figures()
        self.hist._sources, self.hist._spans = sources, spans
        self.hist._active_histogram_column = "intensity"
        self.hist.cutoff = 5.0
        self.hist._refresh_cutoff_spans()
        self.assertTrue(spans["intensity"].visible)
        self.assertEqual(spans["intensity"].location, 5.0)
        self.assertFalse(spans["area"].visible)

    def test_brush_mode_activates_box_select_drag(self):
        """Brush mode must set the BoxSelectTool as the active drag gesture (#112 reply).

        Without this, click-drag falls back to pan and no range can be brushed.
        """
        from bokeh.models import BoxSelectTool

        self.hist.ui_component.interaction_mode.value = "Brush"
        self.hist._channels = ["intensity", "area"]
        layout, _sources, _spans = self.hist._build_figures()
        for fig in layout.children:
            self.assertIsInstance(fig.toolbar.active_drag, BoxSelectTool)

    def test_cutoff_mode_does_not_activate_box_select(self):
        """Cutoff mode leaves drag as pan/auto so tapping to set a cutoff still works."""
        from bokeh.models import BoxSelectTool

        self.hist.ui_component.interaction_mode.value = "Cutoff"
        self.hist._channels = ["intensity"]
        layout, _sources, _spans = self.hist._build_figures()
        for fig in layout.children:
            self.assertNotIsInstance(fig.toolbar.active_drag, BoxSelectTool)


class TestHistogramRendering(unittest.TestCase):
    """Exercise the full interactive render path (skipped without the Bokeh stack)."""

    def setUp(self):
        if not _bokeh_stack_available():
            self.skipTest("bokeh + jupyter_bokeh not available in this environment")
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer, patch_render=False)

    def test_render_hosts_a_single_bokeh_model(self):
        self.hist.ui_component.channel_selector.value = ("intensity", "area")
        self.hist.plot_histograms(None)
        # A single BokehModel widget hosting the multi-figure column is swapped in.
        self.assertEqual(len(self.hist._plot_host.children), 1)
        self.assertIs(self.hist._plot_host.children[0], self.hist._bokeh_model)

    def test_render_with_narrow_subset_overlay_does_not_crash(self):
        """Rendering the shared-edge overlay for a narrow subset builds cleanly (#112)."""
        self.hist.ui_component.channel_selector.value = ("intensity",)
        self.hist.plot_histograms(None)
        self.hist.selected_indices.value = {1}  # intensity 5.0 (range is 1..9)
        self.hist._render()
        self.assertEqual(len(self.hist._plot_host.children), 1)

    def test_tall_stack_applies_scroll_to_the_model(self):
        """A tall histogram stack sets a fixed height + overflow on the BokehModel (#112 reply 2)."""
        from ueler.viewer.plugin.histogram import _MAX_PLOT_HEIGHT

        # Enough numeric channels to exceed the scroll cap.
        table = pd.DataFrame(
            {
                "fov": ["f1", "f1", "f1"],
                "label": [1, 2, 3],
                **{f"m{i}": [1.0, 2.0, 3.0] for i in range(5)},
            }
        )
        viewer = _make_viewer(table)
        hist = _make_histogram(viewer, patch_render=False)
        hist.ui_component.channel_selector.value = tuple(f"m{i}" for i in range(5))
        hist.plot_histograms(None)
        self.assertEqual(hist._bokeh_model.layout.height, f"{_MAX_PLOT_HEIGHT}px")
        self.assertIn("auto", hist._bokeh_model.layout.overflow)

    def test_short_stack_leaves_model_unconstrained(self):
        """A single histogram renders at natural height (no scroll height on the model)."""
        self.hist.ui_component.channel_selector.value = ("intensity",)
        self.hist.plot_histograms(None)
        self.assertIsNone(self.hist._bokeh_model.layout.height)


if __name__ == "__main__":
    unittest.main()
