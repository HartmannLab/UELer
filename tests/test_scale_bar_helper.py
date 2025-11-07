import unittest

from ueler.viewer.scale_bar import compute_scale_bar_spec, effective_pixel_size_nm


class ScaleBarHelperTests(unittest.TestCase):
    def test_compute_scale_bar_spec_rounds_to_engineering_lengths(self) -> None:
        spec = compute_scale_bar_spec(image_width_px=1024, pixel_size_nm=390.0, max_fraction=0.1)
        self.assertAlmostEqual(spec.physical_length_um, 20.0)
        self.assertGreater(spec.pixel_length, 0)
        self.assertLessEqual(spec.pixel_length, 102.4)
        self.assertTrue(spec.label.endswith("µm"))

    def test_compute_scale_bar_spec_handles_millimetres(self) -> None:
        spec = compute_scale_bar_spec(image_width_px=20000, pixel_size_nm=1000.0, max_fraction=0.1)
        self.assertIn("mm", spec.label)
        self.assertGreaterEqual(spec.physical_length_um, 1000.0)

    def test_effective_pixel_size_nm_scales_with_downsample(self) -> None:
        self.assertEqual(effective_pixel_size_nm(400.0, 4), 1600.0)

    def test_compute_scale_bar_spec_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            compute_scale_bar_spec(image_width_px=0, pixel_size_nm=100.0)
        with self.assertRaises(ValueError):
            compute_scale_bar_spec(image_width_px=100, pixel_size_nm=0)
        with self.assertRaises(ValueError):
            compute_scale_bar_spec(image_width_px=100, pixel_size_nm=100.0, max_fraction=0)

    def test_compute_scale_bar_spec_scales_when_pixel_size_expands(self) -> None:
        base = compute_scale_bar_spec(image_width_px=256, pixel_size_nm=500.0, max_fraction=0.1)
        doubled = compute_scale_bar_spec(image_width_px=256, pixel_size_nm=1000.0, max_fraction=0.1)
        # When pixel size doubles, the physical length should increase proportionally
        # while pixel length stays similar (both select from same rounding sequence)
        self.assertAlmostEqual(doubled.physical_length_um, base.physical_length_um * 2.0)


if __name__ == "__main__":
    unittest.main()
