"""Chart (heatmap) plugin is permanently allocated to the wide-footer panel (#121)."""

import unittest
from types import SimpleNamespace

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

import pandas as pd

from ueler.viewer.plugin.chart_heatmap import ChartDisplay


def _make_viewer() -> SimpleNamespace:
    return SimpleNamespace(
        _debug=False,
        base_folder="/tmp",
        cell_table=pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
        fov_key="fov",
        label_key="label",
        SidePlots=SimpleNamespace(),
        BottomPlots=SimpleNamespace(),
    )


class ChartHeatmapFooterTests(unittest.TestCase):
    def setUp(self):
        self.chart = ChartDisplay(_make_viewer(), width=6, height=4)

    def test_chart_heatmap_is_footer_only(self):
        self.assertTrue(self.chart.footer_only)

    def test_wide_panel_layout_always_exposes_controls_and_plots(self):
        def _assert_footer_layout():
            layout = self.chart.wide_panel_layout()
            self.assertIsNotNone(layout)
            self.assertEqual(layout["title"], self.chart.displayed_name)
            self.assertIs(layout["control"], self.chart.controls_section)
            self.assertIs(layout["content"], self.chart.plot_section)

        # Footer regardless of how many scatters are active (#121).
        _assert_footer_layout()
        self.chart._scatter_views["s1"] = SimpleNamespace()
        _assert_footer_layout()
        self.chart._scatter_views["s2"] = SimpleNamespace()
        _assert_footer_layout()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
