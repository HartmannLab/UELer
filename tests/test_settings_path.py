"""Tests for the configurable ``.UELer`` settings folder (issue #137)."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ueler.viewer.checkpoint_store import CheckpointStore
from ueler.viewer.main_viewer import ImageMaskViewer
from ueler.viewer.roi_manager import ROIManager
from ueler.viewer.settings_paths import (
    resolve_settings_folder,
    resolve_settings_root,
    viewer_settings_folder,
)


class ResolveSettingsPathsTests(unittest.TestCase):
    def test_default_settings_folder_is_inside_base_folder(self):
        self.assertEqual(
            resolve_settings_folder("/data/experiment"),
            Path("/data/experiment/.UELer"),
        )

    def test_settings_path_redirects_under_base_folder_name(self):
        self.assertEqual(
            resolve_settings_folder("/data/experiment", "/custom/storage"),
            Path("/custom/storage/experiment/.UELer"),
        )

    def test_settings_root_omits_the_UELer_suffix(self):
        self.assertEqual(
            resolve_settings_root("/data/experiment", "/custom/storage"),
            Path("/custom/storage/experiment"),
        )

    def test_viewer_settings_folder_prefers_the_viewer_attribute(self):
        viewer = MagicMock()
        viewer.settings_folder = Path("/custom/storage/experiment/.UELer")
        viewer.base_folder = "/data/experiment"
        self.assertEqual(viewer_settings_folder(viewer), Path("/custom/storage/experiment/.UELer"))

    def test_viewer_settings_folder_falls_back_to_base_folder(self):
        """Lightweight test doubles (e.g. _ViewerStub) only set base_folder."""

        stub = MagicMock(spec=["base_folder"])
        stub.base_folder = "/data/experiment"
        self.assertEqual(viewer_settings_folder(stub), Path("/data/experiment/.UELer"))

    def test_viewer_settings_folder_none_without_base_folder(self):
        stub = MagicMock(spec=[])
        self.assertIsNone(viewer_settings_folder(stub))


class ROIManagerSettingsDirTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_storage_dir_is_unchanged(self):
        manager = ROIManager(self.base)
        self.assertEqual(manager.storage_dir, os.path.join(self.base, ".UELer"))

    def test_settings_dir_override_relocates_storage(self):
        override = os.path.join(self.base, "elsewhere", ".UELer")
        manager = ROIManager(self.base, settings_dir=override)
        self.assertEqual(manager.storage_dir, override)
        self.assertTrue(os.path.isdir(override))
        self.assertFalse(os.path.exists(os.path.join(self.base, ".UELer")))


class CheckpointStoreStorageRootTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dataset_root = os.path.join(self._tmp.name, "dataset")
        os.makedirs(self.dataset_root, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_default_storage_lives_under_dataset_root(self):
        store = CheckpointStore(self.dataset_root)
        self.assertTrue(str(store._dataset_dir).startswith(str(Path(self.dataset_root).resolve())))

    def test_storage_root_relocates_files_but_keeps_dataset_identity(self):
        storage_root = os.path.join(self._tmp.name, "elsewhere")
        default_store = CheckpointStore(self.dataset_root)
        redirected_store = CheckpointStore(self.dataset_root, storage_root=storage_root)

        self.assertTrue(str(redirected_store._dataset_dir).startswith(str(Path(storage_root).resolve())))
        # Same dataset_root must hash to the same dataset_id regardless of storage_root.
        self.assertEqual(default_store._dataset_id, redirected_store._dataset_id)


class ImageMaskViewerSettingsPathTests(unittest.TestCase):
    """Constructs a real ImageMaskViewer, heavily mocked, per tests/test_fov_detection_fix.py."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_settings_path_base_")
        with open(os.path.join(self.test_dir, "test_image_1.ome.tif"), "w") as f:
            f.write("dummy content")
        with open(os.path.join(self.test_dir, "test_image_2.ome.tif"), "w") as f:
            f.write("dummy content")
        self.settings_root_dir = tempfile.mkdtemp(prefix="test_settings_path_custom_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        shutil.rmtree(self.settings_root_dir, ignore_errors=True)

    def _build_viewer(self, **kwargs):
        with patch("ueler.viewer.main_viewer.ImageMaskViewer._initialize_map_descriptors"), \
             patch("ueler.viewer.main_viewer.ImageMaskViewer.load_status_images"), \
             patch("ueler.viewer.main_viewer.create_widgets") as mock_create_widgets, \
             patch("ueler.viewer.main_viewer.ImageDisplay"), \
             patch("ueler.viewer.main_viewer.plt.show"), \
             patch("ueler.viewer.main_viewer.ImageMaskViewer.update_controls"), \
             patch("ueler.viewer.main_viewer.ImageMaskViewer.on_image_change"), \
             patch("ueler.viewer.main_viewer.ImageMaskViewer.update_display"), \
             patch("ueler.viewer.main_viewer.ImageMaskViewer.load_widget_states"), \
             patch("ueler.viewer.main_viewer.ImageMaskViewer.update_marker_set_dropdown"):

            def side_effect_create_widgets(viewer):
                viewer.ui_component = MagicMock()
                viewer.ui_component.pixel_size_inttext = MagicMock()
                viewer.ui_component.pixel_size_inttext.value = 390
                viewer.ui_component.mask_outline_thickness_slider = MagicMock()
                viewer.ui_component.annotation_editor_host = MagicMock()

            mock_create_widgets.side_effect = side_effect_create_widgets

            def mock_load_fov(self, fov_name, requested_channels=None):
                mock_img = MagicMock()
                mock_img.shape = (100, 100)
                self.image_cache[fov_name] = {"DAPI": mock_img}

            with patch("ueler.viewer.main_viewer.ImageMaskViewer.load_fov", side_effect=mock_load_fov, autospec=True):
                return ImageMaskViewer(self.test_dir, **kwargs)

    def test_default_settings_folder_lives_inside_base_folder(self):
        viewer = self._build_viewer()
        expected = Path(self.test_dir) / ".UELer"
        self.assertEqual(viewer.settings_folder, expected)
        self.assertEqual(Path(viewer.roi_manager.storage_dir), expected)

    def test_settings_path_redirects_settings_folder_and_roi_manager(self):
        viewer = self._build_viewer(settings_path=self.settings_root_dir)
        expected = Path(self.settings_root_dir) / Path(self.test_dir).name / ".UELer"

        self.assertEqual(viewer.settings_folder, expected)
        self.assertEqual(Path(viewer.roi_manager.storage_dir), expected)
        self.assertTrue(expected.is_dir())
        self.assertFalse((Path(self.test_dir) / ".UELer").exists())


if __name__ == "__main__":
    unittest.main()
