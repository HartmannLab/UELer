"""Lossy frames for full-canvas updates, PNG for everything else."""

import types
import unittest

import numpy as np

from ueler.viewer.frame_encoding import (
    LOSSY_MIN_CHANGED_FRACTION,
    available_frame_formats,
    install_frame_encoder,
)
from ueler.viewer.main_viewer import ImageMaskViewer


class _Canvas:
    """Just enough of matplotlib's WebAgg canvas for ``get_diff_image``."""

    def __init__(self, pixels, last=None):
        self.pixels = pixels
        self._png_is_old = True
        self._force_full = False
        self._last_buff = last if last is not None else pixels.view(np.uint32).squeeze(2).copy()
        self.modes = []
        self.png_calls = 0

    def get_renderer(self):
        return types.SimpleNamespace(buffer_rgba=lambda: self.pixels)

    def set_image_mode(self, mode):
        self.modes.append(mode)

    def get_diff_image(self):  # stands in for matplotlib's PNG path
        self.png_calls += 1
        self._png_is_old = False
        return b"\x89PNG"


def _pixels(seed=0, size=64):
    rng = np.random.default_rng(seed)
    rgba = rng.integers(0, 255, size=(size, size, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    return rgba


def _canvas(fmt, *, changed_fraction=1.0, preview=False):
    old = _pixels(1)
    new = old.copy()
    rows = int(round(new.shape[0] * changed_fraction))
    new[:rows] = _pixels(2)[:rows]
    canvas = _Canvas(new, last=old.view(np.uint32).squeeze(2).copy())
    install_frame_encoder(canvas, lambda: fmt, lambda: preview)
    return canvas


class FrameEncoderTests(unittest.TestCase):
    def test_png_setting_keeps_the_original_path(self):
        canvas = _canvas("png")
        self.assertEqual(canvas.get_diff_image(), b"\x89PNG")
        self.assertEqual(canvas.png_calls, 1)

    def test_jpeg_full_update_is_sent_as_a_full_jpeg_frame(self):
        canvas = _canvas("jpeg")
        data = canvas.get_diff_image()
        self.assertEqual(data[:2], b"\xff\xd8")
        self.assertEqual(canvas.modes, ["full"])
        self.assertEqual(canvas.png_calls, 0)
        self.assertFalse(canvas._png_is_old)
        np.testing.assert_array_equal(canvas._last_buff, canvas.pixels.view(np.uint32).squeeze(2))

    def test_webp_when_available(self):
        if "webp" not in available_frame_formats():
            self.skipTest("Pillow built without WebP")
        data = _canvas("webp").get_diff_image()
        self.assertEqual(data[:4], b"RIFF")

    def test_small_update_stays_a_png_diff(self):
        canvas = _canvas("jpeg", changed_fraction=LOSSY_MIN_CHANGED_FRACTION / 4)
        self.assertEqual(canvas.get_diff_image(), b"\x89PNG")

    def test_forced_full_frame_goes_lossy_even_without_changes(self):
        canvas = _canvas("jpeg", changed_fraction=0.0)
        canvas._force_full = True
        self.assertEqual(canvas.get_diff_image()[:2], b"\xff\xd8")
        self.assertFalse(canvas._force_full)

    def test_drag_preview_frames_stay_png(self):
        canvas = _canvas("jpeg", preview=True)
        self.assertEqual(canvas.get_diff_image(), b"\x89PNG")

    def test_nothing_new_to_send(self):
        canvas = _canvas("jpeg")
        canvas._png_is_old = False
        canvas.get_diff_image()
        self.assertEqual(canvas.png_calls, 1)  # the original's own no-op


class SettingTests(unittest.TestCase):
    def test_png_is_always_available(self):
        self.assertIn("png", available_frame_formats())

    def test_unknown_value_falls_back_to_png(self):
        viewer = types.SimpleNamespace(frame_format="jpeg")
        ImageMaskViewer.on_frame_format_change(viewer, {"new": "gif"})
        self.assertEqual(viewer.frame_format, "png")
        ImageMaskViewer.on_frame_format_change(viewer, {"new": "jpeg"})
        self.assertEqual(viewer.frame_format, "jpeg")


if __name__ == "__main__":
    unittest.main()
