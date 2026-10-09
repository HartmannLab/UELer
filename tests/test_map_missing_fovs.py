"""Map descriptors that list FOVs absent from the base folder (map-mode crash fix)."""

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

import tests.test_map_mode_activation  # noqa: F401  - installs the viewer import stubs
from ueler.viewer.main_viewer import ImageMaskViewer
from ueler.viewer.map_descriptor_loader import MapFOVSpec, SlideDescriptor


def _spec(name: str, slide_id: str = "slide") -> MapFOVSpec:
    return MapFOVSpec(
        name=name,
        slide_id=slide_id,
        center_um=(0.0, 0.0),
        frame_size_px=(10, 10),
        fov_size_um=10.0,
        metadata={},
    )


def _descriptor(slide_id: str, *names: str) -> SlideDescriptor:
    return SlideDescriptor(
        slide_id=slide_id,
        source_path=Path("map.json"),
        export_datetime=None,
        fovs=tuple(_spec(name, slide_id) for name in names),
    )


class MapMissingFovTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.viewer = ImageMaskViewer.__new__(ImageMaskViewer)
        self.viewer._map_mode_messages = []
        self.viewer._fov_mode = "folder"
        self.viewer.base_folder = self._tmp.name
        self.viewer.image_cache = OrderedDict()
        self.viewer.frame_index_by_fov = {}
        self.viewer.current_frame_index = 0
        self.viewer.available_fovs = ["FOV_A", "FOV_B"]

    def tearDown(self):
        self._tmp.cleanup()

    def test_unavailable_fovs_are_dropped_with_one_warning(self):
        slides = {"s1": _descriptor("s1", "FOV_A", "FOV_X", "FOV_B", "FOV_Y")}

        kept = self.viewer._drop_unavailable_map_fovs(slides)

        self.assertEqual([spec.name for spec in kept["s1"].fovs], ["FOV_A", "FOV_B"])
        self.assertEqual(len(self.viewer._map_mode_messages), 1)
        message = self.viewer._map_mode_messages[0]
        self.assertIn("2 of 4 FOVs in map 's1'", message)
        self.assertIn("FOV_X", message)

    def test_map_without_any_available_fov_is_dropped(self):
        slides = {
            "s1": _descriptor("s1", "FOV_A"),
            "s2": _descriptor("s2", "FOV_X", "FOV_Y"),
        }

        kept = self.viewer._drop_unavailable_map_fovs(slides)

        self.assertEqual(list(kept), ["s1"])
        self.assertIs(kept["s1"], slides["s1"])
        self.assertTrue(any("'s2' has no FOVs" in msg for msg in self.viewer._map_mode_messages))

    def test_load_fov_raises_for_missing_folder_without_caching_none(self):
        with self.assertRaises(FileNotFoundError):
            self.viewer.load_fov("FOV_MISSING", ("CD3",))

        self.assertNotIn("FOV_MISSING", self.viewer.image_cache)

    def test_render_fov_region_returns_blank_tile_for_missing_fov(self):
        self.viewer.is_no_image_mode_enabled = lambda: False

        image = self.viewer._render_fov_region(
            "FOV_MISSING", ("CD3",), 1, (0, 10, 0, 8), (0, 5, 0, 4)
        )

        self.assertEqual(image.shape, (4, 5, 3))
        self.assertFalse(np.any(image))
        self.assertTrue(any("FOV_MISSING" in msg for msg in self.viewer._map_mode_messages))


if __name__ == "__main__":
    unittest.main()
