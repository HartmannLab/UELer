"""Tests for the Heatmap plugin's permanent wide-footer allocation (#121 reply).

Following the reply to issue #121, the "Heatmap" plugin is permanently allocated to
the wide-footer panel and always renders in the wide (horizontal) orientation. The
old footer/side + orientation toggle (the "Horizontal layout" checkbox and the
``_sync_panel_location`` / ``_place_sections_*`` machinery) has been removed. This
mirrors the Scatter plot and Chart (heatmap) plugins:

- ``footer_only`` is ``True`` (skipped in the side accordion).
- ``adapter.is_wide()`` is always ``True``.
- ``wide_panel_layout()`` always returns the footer dict (never ``None``).
- the ``horizontal_layout_checkbox`` no longer exists on the UiComponent.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import ueler.viewer.plugin.heatmap_layers as heatmap_layers
from ueler.viewer.plugin.heatmap import HeatmapDisplay, UiComponent


def _cell_table() -> "pd.DataFrame":
    return pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov2"],
            "label": [1, 2, 3],
            "cluster": ["A", "B", "A"],
            "intensity": [1.0, 5.0, 9.0],
            "area": [10.0, 20.0, 30.0],
        }
    )


def _make_viewer() -> "SimpleNamespace":
    return SimpleNamespace(
        cell_table=_cell_table(),
        marker_sets={},
        ui_component=SimpleNamespace(channel_selector=SimpleNamespace(value=())),
    )


def _make_heatmap() -> "HeatmapDisplay":
    return HeatmapDisplay(_make_viewer(), width=400, height=400)


class TestHeatmapUiComponentNoToggle(unittest.TestCase):
    def test_ui_component_has_no_horizontal_layout_checkbox(self):
        parent = MagicMock()
        parent.main_viewer = _make_viewer()
        ui = UiComponent(parent)
        self.assertFalse(hasattr(ui, "horizontal_layout_checkbox"))


class TestHeatmapPermanentWideFooter(unittest.TestCase):
    def setUp(self):
        self.heatmap = _make_heatmap()

    def test_heatmap_is_footer_only(self):
        self.assertTrue(self.heatmap.footer_only)

    def test_heatmap_adapter_is_always_wide(self):
        self.assertTrue(self.heatmap.adapter.is_wide())

    def test_wide_panel_layout_always_returns_footer_dict(self):
        layout = self.heatmap.wide_panel_layout()
        self.assertIsNotNone(layout)
        self.assertEqual(layout["title"], "Heatmap")
        self.assertIs(layout["control"], self.heatmap.controls_section)
        self.assertIs(layout["content"], self.heatmap.plot_section)

    def test_wide_panel_layout_is_footer_even_if_adapter_flipped(self):
        # Defensive: the plugin never leaves wide mode, but wide_panel_layout must
        # not gate on orientation — it always returns the footer dict.
        self.heatmap.adapter.mode = "vertical"
        layout = self.heatmap.wide_panel_layout()
        self.assertIsNotNone(layout)
        self.assertIs(layout["content"], self.heatmap.plot_section)


class TestHeatmapCanvasFillsFooter(unittest.TestCase):
    """#121 reply 2: a fresh heatmap fills the footer width, while a remembered
    (user-resized) size stays fixed.

    ``_refresh_plot`` sets the ipympl canvas ``layout.width`` — ``'100%'`` on a fresh
    render so ipympl fits the figure to the full footer width, and ``'auto'`` when the
    #109 resize-remember path restores a user-set figure size.
    """

    def setUp(self):
        self._orig_display = heatmap_layers.display
        heatmap_layers.display = lambda *_a, **_k: None
        self._was_interactive = plt.isinteractive()

    def tearDown(self):
        heatmap_layers.display = self._orig_display
        if self._was_interactive:
            plt.ion()
        else:
            plt.ioff()

    def _make_heatmap(self):
        from ipywidgets import Output, VBox

        heatmap = HeatmapDisplay.__new__(HeatmapDisplay)
        heatmap.adapter = SimpleNamespace(is_wide=lambda: True)
        heatmap._restoring_plot_section = False
        heatmap.plot_output = Output()
        heatmap.plot_section = VBox([heatmap.plot_output])

        self.canvas = SimpleNamespace(
            layout=SimpleNamespace(width=None),
            draw=lambda: None,
            new_timer=lambda interval=0: None,
        )

        def _fake_generate(figsize_override=None):
            heatmap.data = SimpleNamespace(
                g=SimpleNamespace(fig=SimpleNamespace(canvas=self.canvas))
            )

        heatmap.generate_heatmap = _fake_generate
        return heatmap

    def test_fresh_render_fills_footer_width(self):
        heatmap = self._make_heatmap()
        heatmap._refresh_plot()
        self.assertEqual(self.canvas.layout.width, "100%")

    def test_restored_size_keeps_canvas_width_auto(self):
        heatmap = self._make_heatmap()
        heatmap._refresh_plot(restore_size=(10.0, 3.0))
        self.assertEqual(self.canvas.layout.width, "auto")

    def test_apply_canvas_width_is_defensive_without_layout(self):
        heatmap = self._make_heatmap()
        # A stub canvas with no ``layout`` must not raise.
        heatmap._apply_canvas_width(SimpleNamespace(), None)
        heatmap._apply_canvas_width(None, None)


if __name__ == "__main__":
    unittest.main()
