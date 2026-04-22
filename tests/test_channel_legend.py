import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from ueler.viewer.image_display import ImageDisplay
from ueler.viewer.main_viewer import (
    ImageMaskViewer,
    _dedupe_channel_sequence,
    _channel_color_dropdown_layout,
    _channel_header_row_layout,
    _channel_slider_layout,
    _channel_visibility_checkbox_layout,
)


class ChannelLegendTests(unittest.TestCase):
    def test_build_channel_legend_html_escapes_names(self) -> None:
        html = ImageMaskViewer._build_channel_legend_html([
            ("<A>", (1.0, 0.0, 0.0)),
        ])
        self.assertIn("&lt;A&gt;", html)
        self.assertIn("rgb(255, 0, 0)", html)

    def test_image_display_channel_legend_updates(self) -> None:
        display = ImageDisplay(10, 10)
        entries = [("A", (1.0, 0.0, 0.0))]
        display.update_channel_legend(entries, enabled=True)
        self.assertIsNotNone(display.channel_legend_box)

        display.update_channel_legend(entries, enabled=False)
        self.assertIsNone(display.channel_legend_box)

    def test_build_channel_legend_html_wraps_long_names(self) -> None:
        html = ImageMaskViewer._build_channel_legend_html([
            ("VeryLongChannelNameWithoutSpacesForOverflowChecks", (0.0, 1.0, 0.0)),
        ])
        self.assertIn("overflow-wrap: anywhere", html)
        self.assertIn("word-break: break-word", html)
        self.assertIn("max-width: 100%", html)
        self.assertIn("overflow: hidden", html)

    def test_channel_control_layout_helpers_are_compact(self) -> None:
        dropdown_layout = _channel_color_dropdown_layout(["Red", "DarkTurquoise"])
        self.assertEqual(getattr(dropdown_layout, "min_width", None), "68px")
        self.assertEqual(getattr(dropdown_layout, "flex", None), "0 0 auto")
        self.assertEqual(getattr(dropdown_layout, "margin", None), "0 0 0 -5px")
        self.assertTrue(str(getattr(dropdown_layout, "width", "")).endswith("ch"))
        self.assertTrue(str(getattr(dropdown_layout, "max_width", "")).endswith("ch"))

        checkbox_layout = _channel_visibility_checkbox_layout()
        self.assertEqual(getattr(checkbox_layout, "width", None), "20px")
        self.assertEqual(getattr(checkbox_layout, "max_width", None), "20px")
        self.assertEqual(getattr(checkbox_layout, "margin", None), "0")

        header_layout = _channel_header_row_layout()
        self.assertEqual(getattr(header_layout, "gap", None), "4px")
        self.assertEqual(getattr(header_layout, "overflow", None), "hidden")
        self.assertEqual(getattr(header_layout, "max_width", None), "calc(100% - 5px)")

        slider_layout = _channel_slider_layout()
        self.assertEqual(getattr(slider_layout, "width", None), "calc(100% - 5px)")
        self.assertEqual(getattr(slider_layout, "max_width", None), "calc(100% - 5px)")
        self.assertEqual(getattr(slider_layout, "flex", None), "0 0 auto")
        self.assertEqual(getattr(slider_layout, "box_sizing", None), "border-box")

    def test_dedupe_channel_sequence_preserves_order(self) -> None:
        deduped = _dedupe_channel_sequence(("CD45", "CD3", "CD45", "CD8", "CD3"))
        self.assertEqual(deduped, ("CD45", "CD3", "CD8"))

    def test_dedupe_channel_sequence_coerces_to_strings(self) -> None:
        deduped = _dedupe_channel_sequence((1, "1", 2, "2", 1))
        self.assertEqual(deduped, ("1", "2"))

    def test_apply_marker_set_dedupes_and_skips_manual_refresh_when_selector_changes(self) -> None:
        viewer = ImageMaskViewer.__new__(ImageMaskViewer)
        viewer.marker_sets = {
            "set1": {
                "selected_channels": ["CD45", "CD3", "CD45"],
                "channel_settings": {},
            }
        }
        viewer.ui_component = SimpleNamespace(
            channel_selector=SimpleNamespace(value=("CD20",)),
            color_controls={},
            contrast_min_controls={},
            contrast_max_controls={},
        )
        viewer._merge_channel_max = MagicMock()
        viewer._sync_channel_controls = MagicMock()
        viewer.update_controls = MagicMock()
        viewer.update_display = MagicMock()
        viewer.current_downsample_factor = 1

        ok = viewer._apply_marker_set("set1", silent=True)

        self.assertTrue(ok)
        self.assertEqual(viewer.ui_component.channel_selector.value, ("CD45", "CD3"))
        viewer.update_controls.assert_not_called()

    def test_apply_marker_set_refreshes_once_when_selector_unchanged(self) -> None:
        viewer = ImageMaskViewer.__new__(ImageMaskViewer)
        viewer.marker_sets = {
            "set1": {
                "selected_channels": ["CD45", "CD3", "CD45"],
                "channel_settings": {},
            }
        }
        viewer.ui_component = SimpleNamespace(
            channel_selector=SimpleNamespace(value=("CD45", "CD3")),
            color_controls={},
            contrast_min_controls={},
            contrast_max_controls={},
        )
        viewer._merge_channel_max = MagicMock()
        viewer._sync_channel_controls = MagicMock()
        viewer.update_controls = MagicMock()
        viewer.update_display = MagicMock()
        viewer.current_downsample_factor = 1

        ok = viewer._apply_marker_set("set1", silent=True)

        self.assertTrue(ok)
        viewer.update_controls.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
