"""Screen-aware downsample factor and the float32 composite."""

import unittest

import numpy as np

from ueler.image_utils import calculate_downsample_factor, screen_downsample_factor
from ueler.rendering.engine import ChannelRenderSettings, _composite_channels


class ScreenDownsampleFactorTests(unittest.TestCase):
    def test_never_blurrier_than_the_screen(self):
        # 2048 image px on 600 device px: factor 2 renders 1024 >= 600; 4 would give 512 < 600.
        self.assertEqual(screen_downsample_factor(2048, 2048, 600, 600), 2)
        # Retina: the same canvas is 1200 device px, so full resolution is needed.
        self.assertEqual(screen_downsample_factor(2048, 2048, 1200, 1200), 1)

    def test_never_more_than_the_budget(self):
        # A screen larger than the image budget keeps the 2048 cap's factor.
        self.assertEqual(screen_downsample_factor(8192, 8192, 4000, 4000), calculate_downsample_factor(8192, 8192))

    def test_both_axes_must_be_covered(self):
        # 2048 wide but only 700 tall in a 600 × 600 view: 700/2 < 600, so factor 1.
        self.assertEqual(screen_downsample_factor(2048, 700, 600, 600), 1)

    def test_downsample_off_is_native(self):
        self.assertEqual(screen_downsample_factor(8192, 8192, 600, 600, ignore_zoom=True), 1)

    def test_unknown_or_tiny_screen_falls_back_to_the_budget(self):
        self.assertEqual(screen_downsample_factor(4096, 4096, None, None), 2)
        self.assertEqual(screen_downsample_factor(4096, 4096, 1, 1), 2)

    def test_zoomed_in_is_native(self):
        self.assertEqual(screen_downsample_factor(300, 300, 600, 600), 1)


def _reference_composite(arrays, settings, shape):
    """The float64 formulation the engine used before fix 2."""
    composite = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    for name, array in arrays.items():
        st = settings[name]
        scale = max(st.contrast_max - st.contrast_min, np.finfo(np.float32).eps)
        normalised = np.clip((array - st.contrast_min) / scale, 0.0, 1.0)
        composite += normalised[..., np.newaxis] * np.asarray(st.color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(composite, 0.0, 1.0)


class Float32CompositeTests(unittest.TestCase):
    def test_matches_the_float64_formulation(self):
        rng = np.random.default_rng(0)
        shape = (64, 48)
        arrays = {
            "a": rng.integers(0, 4000, shape, dtype=np.uint16),
            "b": rng.integers(0, 4000, shape, dtype=np.uint16),
            "c": rng.random(shape) * 50.0,
        }
        settings = {
            "a": ChannelRenderSettings(color=(1.0, 0.0, 0.0), contrast_min=10.0, contrast_max=900.0),
            "b": ChannelRenderSettings(color=(0.5, 1.0, 0.25), contrast_min=0.0, contrast_max=3000.0),
            "c": ChannelRenderSettings(color=(0.0, 0.0, 1.0), contrast_min=5.0, contrast_max=5.0),
        }
        region = (0, shape[1], 0, shape[0])
        got = _composite_channels(arrays, list(arrays), settings, region, 1, shape)
        expected = _reference_composite(arrays, settings, shape)
        self.assertEqual(got.dtype, np.float32)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_does_not_modify_the_source_array(self):
        array = np.full((4, 4), 100, dtype=np.uint16)
        settings = {"a": ChannelRenderSettings(color=(1.0, 1.0, 1.0), contrast_min=0.0, contrast_max=200.0)}
        _composite_channels({"a": array}, ["a"], settings, (0, 4, 0, 4), 1, (4, 4))
        self.assertTrue((array == 100).all())

    def test_shape_mismatch_is_cropped(self):
        array = np.ones((5, 5), dtype=np.uint16)
        settings = {"a": ChannelRenderSettings(color=(0.0, 1.0, 0.0), contrast_min=0.0, contrast_max=1.0)}
        got = _composite_channels({"a": array}, ["a"], settings, (0, 5, 0, 5), 1, (4, 6))
        self.assertEqual(got.shape, (4, 6, 3))
        self.assertTrue((got[:4, :5, 1] == 1.0).all())
        self.assertTrue((got[:, 5, 1] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
