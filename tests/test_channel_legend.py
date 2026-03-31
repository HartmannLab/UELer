import unittest

from ueler.viewer.image_display import ImageDisplay
from ueler.viewer.main_viewer import ImageMaskViewer


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


if __name__ == "__main__":
    unittest.main()
