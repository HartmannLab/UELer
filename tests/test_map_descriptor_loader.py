from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ueler.viewer.map_descriptor_loader import MapDescriptorLoader


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "map_descriptors"


def _copy_fixture(name: str, destination: Path) -> Path:
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIXTURE_DIR / name, target)
    return target


class MapDescriptorLoaderTests(unittest.TestCase):
    def setUp(self) -> None:  # pragma: no cover - setup boilerplate
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:  # pragma: no cover - cleanup boilerplate
        self._tmp.cleanup()

    def test_loader_parses_valid_descriptor(self) -> None:
        _copy_fixture("valid_slide.json", self.tmp_path)
        loader = MapDescriptorLoader()

        result = loader.load_from_directory(self.tmp_path)

        self.assertFalse(result.errors)
        self.assertFalse(result.warnings)
        self.assertIn("402", result.slides)

        slide = result.slides["402"]
        self.assertEqual(slide.export_datetime, "2024-11-04T12:45:08.914Z")
        self.assertEqual([fov.name for fov in slide.fovs], ["FOV_A", "FOV_B"])
        self.assertEqual(slide.fovs[0].center_um, (13880.0, 47536.0))
        self.assertEqual(slide.fovs[0].frame_size_px, (1024, 1024))

    def test_loader_reports_mixed_unit_error(self) -> None:
        _copy_fixture("mixed_units.json", self.tmp_path)
        loader = MapDescriptorLoader()

        result = loader.load_from_directory(self.tmp_path)

        self.assertFalse(result.slides)
        self.assertTrue(result.errors)
        self.assertIn("mixed coordinate units", result.errors[0])

    def test_loader_reports_malformed_descriptor(self) -> None:
        _copy_fixture("malformed.json", self.tmp_path)
        loader = MapDescriptorLoader()

        result = loader.load_from_directory(self.tmp_path)

        self.assertFalse(result.slides)
        self.assertTrue(result.errors)
        self.assertIn("must define 'fovs' as a list", result.errors[0])


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    unittest.main()