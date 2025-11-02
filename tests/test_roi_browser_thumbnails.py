import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.test_roi_manager_tags import make_plugin


class _FakeFrame:
    def __init__(self, records):
        self._records = list(records)

    @property
    def empty(self):
        return not self._records

    def to_dict(self, orient):
        if orient != "records":
            raise ValueError("Unsupported orientation")
        return list(self._records)


class TestROIManagerBrowserThumbnails(unittest.TestCase):
    def test_downsample_uses_calculator_directly(self):
        plugin = make_plugin()
        record = {"width": 2048, "height": 1024, "fov": "FOV1"}

        with patch(
            "ueler.viewer.plugin.roi_manager_plugin.calculate_downsample_factor",
            return_value=8,
        ) as calc:
            factor = plugin._determine_browser_downsample(record)

        self.assertEqual(factor, 8)
        calc.assert_called_once_with(2048.0, 1024.0, ignore_zoom=False)

    def test_downsample_falls_back_to_bounds(self):
        plugin = make_plugin()
        record = {
            "width": None,
            "height": 0,
            "x_min": 0,
            "x_max": 4096,
            "y_min": 0,
            "y_max": 2048,
            "fov": "FOV1",
        }

        with patch(
            "ueler.viewer.plugin.roi_manager_plugin.calculate_downsample_factor",
            return_value=4,
        ) as calc:
            factor = plugin._determine_browser_downsample(record)

        self.assertEqual(factor, 4)
        calc.assert_called_once_with(4096.0, 2048.0, ignore_zoom=False)

    def test_refresh_gallery_applies_enforced_downsample_and_layout(self):
        plugin = make_plugin()
        plugin.BROWSER_COLUMNS = 4

        records = [
            {
                "roi_id": f"roi-{idx}",
                "fov": "FOV1",
                "width": 1200,
                "height": 800,
                "x_min": 0,
                "x_max": 1200,
                "y_min": 0,
                "y_max": 800,
            }
            for idx in range(4)
        ]

        plugin._filtered_browser_dataframe = lambda: _FakeFrame(records)
        plugin._build_marker_profile = lambda *_: SimpleNamespace(selected_channels=("ch1",), channel_settings={})

        downsample_calls = []

        def _render_stub(_record, _profile, downsample):
            downsample_calls.append(downsample)
            return np.ones((2, 2, 3), dtype=np.float32)
        plugin._render_roi_tile = _render_stub

        import matplotlib.pyplot as plt

        class _Canvas:
            def __init__(self):
                self.toolbar_visible = False
                self.header_visible = False
                self.footer_visible = False

            def mpl_connect(self, *_):
                return 1

        class _Figure:
            def __init__(self):
                self.canvas = _Canvas()

            def subplots_adjust(self, **_):
                return None

        class _Axis:
            def axis(self, *_):
                return None

            def text(self, *_args, **_kwargs):
                return None

            def imshow(self, *_args, **_kwargs):
                return None

            def remove(self):
                return None

        def _subplots(rows, cols, **_):
            fig = _Figure()
            axes = [[_Axis() for _ in range(cols)] for _ in range(rows)]
            return fig, axes if rows > 1 else axes[0]

        with patch(
            "matplotlib.pyplot.subplots",
            side_effect=_subplots,
        ), patch(
            "matplotlib.pyplot.show",
            lambda *_: None,
        ), patch(
            "ueler.viewer.plugin.roi_manager_plugin.calculate_downsample_factor",
            return_value=2,
        ):
            plugin._refresh_browser_gallery()

        self.assertEqual(downsample_calls, [2, 2, 2, 2])

        layout = plugin.ui_component.browser_output.layout
        self.assertEqual(layout.width, "960px")
        self.assertEqual(layout.min_width, "960px")
        self.assertEqual(layout.max_width, "960px")
        self.assertEqual(layout.align_self, "flex-start")
        self.assertEqual(layout.overflow_x, "auto")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
