"""Unit tests for the shared tile-grid gallery widget (issue #107)."""

import base64
import unittest

import numpy as np

from ueler.viewer.plugin.tile_gallery_widget import (
    TileGalleryWidget,
    array_to_data_uri,
    parse_clicked_id,
    text_placeholder_uri,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _decode(uri: str) -> bytes:
    assert uri.startswith("data:image/png;base64,"), uri[:32]
    return base64.b64decode(uri.split(",", 1)[1])


class ArrayToDataUriTestCase(unittest.TestCase):
    def test_float_rgb_produces_valid_png(self):
        arr = np.clip(np.random.default_rng(0).random((8, 8, 3)), 0.0, 1.0)
        raw = _decode(array_to_data_uri(arr))
        self.assertEqual(raw[:8], _PNG_MAGIC)

    def test_uint8_input_produces_valid_png(self):
        arr = (np.random.default_rng(1).random((6, 10, 3)) * 255).astype(np.uint8)
        raw = _decode(array_to_data_uri(arr))
        self.assertEqual(raw[:8], _PNG_MAGIC)

    def test_grayscale_input_is_expanded_to_rgb(self):
        arr = np.linspace(0.0, 1.0, 16).reshape(4, 4)
        raw = _decode(array_to_data_uri(arr))
        self.assertEqual(raw[:8], _PNG_MAGIC)

    def test_empty_array_returns_empty_string(self):
        self.assertEqual(array_to_data_uri(np.zeros((0, 0, 3))), "")


class TextPlaceholderTestCase(unittest.TestCase):
    def test_placeholder_is_valid_png_data_uri(self):
        raw = _decode(text_placeholder_uri("No channels"))
        self.assertEqual(raw[:8], _PNG_MAGIC)


class ParseClickedIdTestCase(unittest.TestCase):
    def test_strips_nonce_suffix(self):
        self.assertEqual(parse_clicked_id("42|7"), "42")

    def test_preserves_ids_containing_pipe(self):
        # rsplit keeps a pipe that is part of the id itself.
        self.assertEqual(parse_clicked_id("roi|abc|3"), "roi|abc")

    def test_empty_payload_returns_empty(self):
        self.assertEqual(parse_clicked_id(""), "")
        self.assertEqual(parse_clicked_id(None), "")


class TileGalleryWidgetTestCase(unittest.TestCase):
    def test_trait_round_trip_and_click_observer(self):
        widget = TileGalleryWidget(columns=3)
        self.assertEqual(int(widget.columns), 3)

        fired = []
        widget.observe(lambda change: fired.append(change["new"]), names="clicked")

        widget.tiles = [{"id": "a", "src": array_to_data_uri(np.zeros((2, 2, 3))), "label": "A"}]
        self.assertEqual(len(widget.tiles), 1)

        widget.clicked = "a|0"
        # Re-clicking the same tile must still fire because of the changing nonce.
        widget.clicked = "a|1"
        self.assertEqual(fired, ["a|0", "a|1"])
        self.assertEqual(parse_clicked_id(fired[-1]), "a")


if __name__ == "__main__":
    unittest.main()
