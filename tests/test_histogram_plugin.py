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


def _real_matplotlib() -> bool:
    """True when the *real* matplotlib.pyplot is importable (not a bootstrap stub)."""
    mod = sys.modules.get("matplotlib.pyplot")
    return bool(mod is not None and getattr(mod, "__file__", None))


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

    def test_brush_triggers_rerender_for_overlay(self):
        before = self.hist._render_calls
        self.hist._on_brush("intensity", 4.0, 8.0)
        self.assertGreater(self.hist._render_calls, before)

    def test_clear_selection_empties_indices(self):
        self.hist._on_brush("intensity", 4.0, 8.0)
        self.assertTrue(self.hist.selected_indices.value)
        self.hist.clear_selection()
        self.assertEqual(self.hist.selected_indices.value, set())

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


class TestHistogramRendering(unittest.TestCase):
    """Exercise the real Matplotlib render path (skipped without real matplotlib)."""

    def setUp(self):
        if not _real_matplotlib():
            self.skipTest("real matplotlib not available in this environment")
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer, patch_render=False)

    def test_render_builds_output_for_multiple_channels(self):
        self.hist.ui_component.channel_selector.value = ("intensity", "area")
        self.hist.plot_histograms(None)
        # A single Output hosting the (multi-subplot) figure is swapped in.
        self.assertEqual(len(self.hist._plot_host.children), 1)

    def test_render_with_narrow_subset_overlay_does_not_crash(self):
        """Rendering the shared-edge overlay for a narrow subset builds cleanly (#112)."""
        self.hist.ui_component.channel_selector.value = ("intensity",)
        self.hist.plot_histograms(None)
        # Narrow selection well inside the full range; overlay uses full-data edges.
        self.hist.selected_indices.value = {1}  # intensity 5.0 (range is 1..9)
        self.hist._render()
        self.assertEqual(len(self.hist._plot_host.children), 1)


if __name__ == "__main__":
    unittest.main()
