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

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
