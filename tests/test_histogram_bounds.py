"""Tests for user-adjustable histogram bounds (issue #138).

Each histogram carries its own binning window: without one it bins over the
channel's full data extent (the original behaviour), with one it puts every bin
inside the chosen range and re-bins in place — without rebuilding the figure, and
without disturbing the other channels, the gate terms or the selection overlay.
"""
from __future__ import annotations

import unittest

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

import numpy as np
import pandas as pd

from tests.test_histogram_plugin import (
    _bokeh_available,
    _figures_of,
    _make_histogram,
    _make_viewer,
    _two_fov_table,
)

from ueler.viewer.plugin import histogram as _h


def _long_tail_table() -> "pd.DataFrame":
    """A channel with most of its mass near zero and one far outlier.

    This is the shape the issue is about: binning over ``[0, 1000]`` puts every
    cell but one in the first bin, so the structure near zero is unreadable.
    """
    return pd.DataFrame(
        {
            "fov": ["fov1"] * 6,
            "label": [1, 2, 3, 4, 5, 6],
            "intensity": [0.0, 1.0, 2.0, 3.0, 4.0, 1000.0],
            "area": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


class TestBinRange(unittest.TestCase):
    """``_bin_range`` / ``_channel_extent``: the window the bins are laid over."""

    def setUp(self):
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer)
        self.hist._plot_data = self.viewer.cell_table.copy()

    def test_no_bound_means_full_extent(self):
        """An unbounded channel bins exactly as it did before #138."""
        self.assertIsNone(self.hist._bin_range("intensity"))
        edges = self.hist._histogram_bin_edges("intensity", 10)
        expected = np.histogram_bin_edges(self.hist._plot_data["intensity"], bins=10)
        np.testing.assert_allclose(edges, expected)

    def test_bound_confines_every_bin_to_the_window(self):
        self.hist._bounds["intensity"] = (2.0, 6.0)
        edges = self.hist._histogram_bin_edges("intensity", 8)
        self.assertAlmostEqual(float(edges[0]), 2.0)
        self.assertAlmostEqual(float(edges[-1]), 6.0)
        self.assertEqual(len(edges), 9)

    def test_values_outside_the_window_are_left_out_of_the_counts(self):
        """Not clipped into the end bins, which would invent a spike (#138)."""
        self.hist._bounds["intensity"] = (2.0, 6.0)
        edges = self.hist._histogram_bin_edges("intensity", 4)
        counts = _h.bin_counts(self.hist._plot_data["intensity"], edges)
        # intensities are 1, 5, 9, 3, 7 — only 3 and 5 are inside [2, 6].
        self.assertEqual(int(counts.sum()), 2)

    def test_bound_is_clamped_to_the_current_data_extent(self):
        """A bound set before a subset change cannot bin over an empty window."""
        self.hist._bounds["intensity"] = (-100.0, 100.0)
        self.assertEqual(self.hist._bin_range("intensity"), (1.0, 9.0))

    def test_inverted_bound_is_read_in_order(self):
        self.hist._bounds["intensity"] = (7.0, 3.0)
        self.assertEqual(self.hist._bin_range("intensity"), (3.0, 7.0))

    def test_degenerate_channel_has_no_extent(self):
        """A constant or empty column gives no window with width."""
        self.hist._plot_data = pd.DataFrame({"flat": [2.0, 2.0, 2.0]})
        self.assertIsNone(self.hist._channel_extent("flat"))
        self.hist._bounds["flat"] = (0.0, 1.0)
        self.assertIsNone(self.hist._bin_range("flat"))

    def test_unknown_channel_is_not_an_error(self):
        self.assertIsNone(self.hist._channel_extent("nope"))
        self.assertIsNone(self.hist._bin_range("nope"))


class TestSetAndClearBounds(unittest.TestCase):
    """``set_channel_bounds`` / ``clear_channel_bounds`` re-bin in place."""

    def setUp(self):
        if not _bokeh_available():
            self.skipTest("bokeh not available in this environment")
        self.viewer = _make_viewer(_long_tail_table())
        self.hist = _make_histogram(self.viewer, patch_render=False)
        self.hist._plot_data = self.viewer.cell_table.copy()
        self.hist._channels = ["intensity", "area"]
        _layout, sources, spans = self.hist._build_figures()
        self.hist._sources, self.hist._spans = sources, spans

    def test_setting_bounds_redraws_the_base_bars_over_the_window(self):
        self.hist.set_channel_bounds("intensity", 0.0, 4.0)
        info = self.hist._sources["intensity"]
        self.assertAlmostEqual(float(info["edges"][0]), 0.0)
        self.assertAlmostEqual(float(info["edges"][-1]), 4.0)
        self.assertAlmostEqual(info["full_src"].data["left"][0], 0.0)
        self.assertAlmostEqual(info["full_src"].data["right"][-1], 4.0)
        # The outlier at 1000 is outside the window, so 5 of the 6 cells are drawn.
        self.assertEqual(int(np.asarray(info["full_src"].data["top"]).sum()), 5)

    def test_setting_bounds_moves_that_figure_x_range(self):
        self.hist.set_channel_bounds("intensity", 1.0, 3.0)
        x_range = self.hist._figures["intensity"].x_range
        self.assertAlmostEqual(float(x_range.start), 1.0)
        self.assertAlmostEqual(float(x_range.end), 3.0)

    def test_bounds_are_per_channel(self):
        """Bounding one channel leaves the others binned over their own extent."""
        before = self.hist._sources["area"]["edges"].copy()
        self.hist.set_channel_bounds("intensity", 0.0, 4.0)
        np.testing.assert_allclose(self.hist._sources["area"]["edges"], before)
        self.assertNotIn("area", self.hist._bounds)

    def test_setting_bounds_never_replots(self):
        """The slider that triggered the change must survive it (#127 rule, #138)."""
        figure_before = self.hist._figures["intensity"]
        source_before = self.hist._sources["intensity"]["full_src"]
        calls = []
        self.hist._render = lambda: calls.append(1)
        self.hist.set_channel_bounds("intensity", 0.0, 4.0)
        self.assertEqual(calls, [])
        self.assertIs(self.hist._figures["intensity"], figure_before)
        self.assertIs(self.hist._sources["intensity"]["full_src"], source_before)

    def test_zero_width_bounds_are_ignored(self):
        self.hist.set_channel_bounds("intensity", 3.0, 3.0)
        self.assertNotIn("intensity", self.hist._bounds)

    def test_inverted_bounds_are_stored_in_order(self):
        self.hist.set_channel_bounds("intensity", 4.0, 1.0)
        self.assertEqual(self.hist._bounds["intensity"], (1.0, 4.0))

    def test_clearing_restores_the_full_extent(self):
        self.hist.set_channel_bounds("intensity", 0.0, 4.0)
        self.hist.clear_channel_bounds("intensity")
        self.assertNotIn("intensity", self.hist._bounds)
        info = self.hist._sources["intensity"]
        self.assertAlmostEqual(float(info["edges"][-1]), 1000.0)
        self.assertEqual(int(np.asarray(info["full_src"].data["top"]).sum()), 6)

    def test_clearing_puts_the_slider_handles_back(self):
        slider = self.hist._bound_sliders["intensity"]
        self.hist.set_channel_bounds("intensity", 0.0, 4.0)
        slider.value = (0.0, 4.0)
        self.hist.clear_channel_bounds("intensity")
        self.assertEqual(tuple(slider.value), (0.0, 1000.0))

    def test_clearing_an_unbounded_channel_is_a_no_op(self):
        calls = []
        self.hist._rebin_channel = lambda channel: calls.append(channel)
        self.hist.clear_channel_bounds("intensity")
        self.assertEqual(calls, [])


class TestBoundsAndSelection(unittest.TestCase):
    """Bounds are display state: the gate and the overlay follow, but do not change."""

    def setUp(self):
        if not _bokeh_available():
            self.skipTest("bokeh not available in this environment")
        self.viewer = _make_viewer(_long_tail_table())
        self.hist = _make_histogram(self.viewer, patch_render=False)
        self.hist._plot_data = self.viewer.cell_table.copy()
        self.hist._channels = ["intensity"]
        _layout, sources, spans = self.hist._build_figures()
        self.hist._sources, self.hist._spans = sources, spans

    def test_selection_overlay_is_rebinned_onto_the_new_grid(self):
        self.hist.selected_indices.value = {0, 1, 2}
        self.hist._refresh_overlays()
        self.hist.set_channel_bounds("intensity", 0.0, 4.0)
        selected = self.hist._sources["intensity"]["selected"].data
        self.assertEqual(selected["left"], self.hist._sources["intensity"]["edges"][:-1].tolist())
        self.assertEqual(int(np.asarray(selected["top"]).sum()), 3)

    def test_bounds_do_not_narrow_the_gate(self):
        """A gate keeps selecting the cells it names, in or out of the window (#138)."""
        self.hist.handle_range("intensity", 0.0, 1000.0)
        before = set(self.hist.selected_indices.value)
        self.hist.set_channel_bounds("intensity", 0.0, 4.0)
        self.hist._apply_gate(publish=True, highlight=False)
        self.assertEqual(set(self.hist.selected_indices.value), before)
        self.assertEqual(len(before), 6)

    def test_gate_band_survives_a_bounds_change(self):
        self.hist.handle_range("intensity", 1.0, 3.0)
        self.hist.set_channel_bounds("intensity", 0.0, 4.0)
        band = self.hist._sources["intensity"]["band"]
        self.assertTrue(band.visible)
        self.assertEqual((band.left, band.right), (1.0, 3.0))


class TestBoundsLifecycle(unittest.TestCase):
    """Bounds survive a rebuild and are dropped with the channel they belong to."""

    def setUp(self):
        if not _bokeh_available():
            self.skipTest("bokeh not available in this environment")
        self.viewer = _make_viewer(_long_tail_table())
        self.hist = _make_histogram(self.viewer, patch_render=False)

    def test_bounds_survive_a_bin_count_change(self):
        self.hist._plot_data = self.viewer.cell_table.copy()
        self.hist._channels = ["intensity"]
        self.hist._bounds["intensity"] = (0.0, 4.0)
        _layout, sources, _spans = self.hist._build_figures()
        self.assertAlmostEqual(float(sources["intensity"]["edges"][-1]), 4.0)

    def test_slider_handles_reflect_an_existing_bound(self):
        self.hist._plot_data = self.viewer.cell_table.copy()
        self.hist._channels = ["intensity"]
        self.hist._bounds["intensity"] = (0.0, 4.0)
        self.hist._build_figures()
        slider = self.hist._bound_sliders["intensity"]
        self.assertEqual(tuple(slider.value), (0.0, 4.0))
        self.assertAlmostEqual(float(slider.start), 0.0)
        self.assertAlmostEqual(float(slider.end), 1000.0)

    def test_plotting_drops_bounds_of_channels_no_longer_shown(self):
        self.hist._render = lambda: None
        self.hist.ui_component.channel_selector.value = ["intensity"]
        self.hist._bounds = {"intensity": (0.0, 4.0), "area": (10.0, 20.0)}
        self.hist.plot_histograms(None)
        self.assertEqual(set(self.hist._bounds), {"intensity"})

    def test_a_constant_channel_still_gets_a_row(self):
        """A disabled slider, so the stack does not change height with the data."""
        self.hist._plot_data = pd.DataFrame({"flat": [2.0, 2.0, 2.0]})
        self.hist._channels = ["flat"]
        layout, _sources, _spans = self.hist._build_figures()
        self.assertEqual(len(_figures_of(layout)), 1)
        bounds_row = layout.children[0].children[1]
        self.assertTrue(bounds_row.children[0].disabled)
        self.assertNotIn("flat", self.hist._bound_sliders)


class TestBoundsWidgets(unittest.TestCase):
    """The slider row that sits under each figure."""

    def setUp(self):
        if not _bokeh_available():
            self.skipTest("bokeh not available in this environment")
        self.viewer = _make_viewer(_long_tail_table())
        self.hist = _make_histogram(self.viewer, patch_render=False)
        self.hist._plot_data = self.viewer.cell_table.copy()
        self.hist._channels = ["intensity", "area"]
        self.layout, sources, spans = self.hist._build_figures()
        self.hist._sources, self.hist._spans = sources, spans

    def test_every_channel_gets_a_slider_below_its_figure(self):
        from bokeh.models import RangeSlider

        self.assertEqual(set(self.hist._bound_sliders), {"intensity", "area"})
        for child in self.layout.children:
            figure, bounds_row = child.children
            self.assertIsInstance(bounds_row.children[0], RangeSlider)
            self.assertIn(figure.title.text, {"intensity", "area"})

    def test_slider_spans_the_channel_extent(self):
        slider = self.hist._bound_sliders["area"]
        self.assertAlmostEqual(float(slider.start), 10.0)
        self.assertAlmostEqual(float(slider.end), 60.0)
        self.assertEqual(tuple(slider.value), (10.0, 60.0))

    def test_slider_value_is_wired_to_the_rebin(self):
        """Bound to ``value_throttled``, so dragging does not rebin per pixel."""
        handler = self.hist._make_bounds_handler("intensity")
        handler("value_throttled", (0.0, 1000.0), (0.0, 4.0))
        self.assertEqual(self.hist._bounds["intensity"], (0.0, 4.0))
        self.assertIn("value_throttled", self.hist._bound_sliders["intensity"]._callbacks)

    def test_x_range_follows_the_handles_client_side(self):
        """A ``js_link`` per handle, so the window moves without a kernel round-trip."""
        links = self.hist._bound_sliders["intensity"].js_property_callbacks.get("change:value", [])
        self.assertEqual(len(links), 2)

    def test_full_range_button_clears_the_bound(self):
        from bokeh.models import Button

        bounds_row = self.layout.children[0].children[1]
        self.assertIsInstance(bounds_row.children[1], Button)
        self.hist.set_channel_bounds("intensity", 0.0, 4.0)
        self.hist._make_clear_bounds_handler("intensity")()
        self.assertNotIn("intensity", self.hist._bounds)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
