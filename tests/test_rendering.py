import unittest

import numpy as np

from ueler.viewer.rendering import (
    AnnotationRenderSettings,
    ChannelRenderSettings,
    MaskRenderSettings,
    render_crop_to_array,
    render_fov_to_array,
    render_roi_to_array,
)


class RenderingHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.channels = {
            "A": np.ones((4, 4), dtype=np.float32),
            "B": np.full((4, 4), 0.5, dtype=np.float32),
        }
        self.settings = {
            "A": ChannelRenderSettings(color=(1.0, 0.0, 0.0), contrast_min=0.0, contrast_max=1.0),
            "B": ChannelRenderSettings(color=(0.0, 1.0, 0.0), contrast_min=0.0, contrast_max=1.0),
        }

    def test_render_fov_to_array_combines_channels(self) -> None:
        result = render_fov_to_array(
            "FOV",
            self.channels,
            ("A", "B"),
            self.settings,
            downsample_factor=1,
        )
        self.assertEqual(result.shape, (4, 4, 3))
        np.testing.assert_allclose(result[0, 0], [1.0, 0.5, 0.0], atol=1e-6)

    def test_render_fov_to_array_applies_annotation(self) -> None:
        annotation = np.zeros((4, 4), dtype=np.int32)
        annotation[0, 0] = 1
        colormap = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        annotation_settings = AnnotationRenderSettings(
            array=annotation,
            colormap=colormap,
            alpha=0.5,
            mode="combined",
        )
        result = render_fov_to_array(
            "FOV",
            self.channels,
            ("A", "B"),
            self.settings,
            downsample_factor=1,
            annotation=annotation_settings,
        )
        np.testing.assert_allclose(result[0, 0], [0.5, 0.25, 0.5], atol=1e-6)

    def test_render_fov_to_array_with_masks(self) -> None:
        mask_array = np.zeros((4, 4), dtype=bool)
        mask_array[1, 1] = True
        mask = MaskRenderSettings(array=mask_array, color=(0.0, 0.0, 1.0))
        result = render_fov_to_array(
            "FOV",
            self.channels,
            ("A", "B"),
            self.settings,
            downsample_factor=1,
            masks=[mask],
        )
        np.testing.assert_allclose(result[1, 1], [0.0, 0.0, 1.0], atol=1e-6)

    def test_render_crop_to_array_uses_requested_region(self) -> None:
        crop = render_crop_to_array(
            "FOV",
            self.channels,
            ("A", "B"),
            self.settings,
            center_xy=(1.5, 1.5),
            size_px=2,
            downsample_factor=1,
        )
        full = render_fov_to_array(
            "FOV",
            self.channels,
            ("A", "B"),
            self.settings,
            downsample_factor=1,
            region_xy=(1, 3, 1, 3),
            region_ds=(1, 3, 1, 3),
        )
        self.assertEqual(crop.shape, (2, 2, 3))
        np.testing.assert_allclose(crop, full, atol=1e-6)

    def test_render_roi_to_array_requires_valid_definition(self) -> None:
        with self.assertRaises(ValueError):
            render_roi_to_array(
                "FOV",
                self.channels,
                ("A", "B"),
                self.settings,
                roi_definition={"x": 0.0, "y": 0.0},
                downsample_factor=1,
            )


if __name__ == "__main__":
    unittest.main()
