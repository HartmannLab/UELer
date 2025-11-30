import unittest
import os
import shutil
import sys
from unittest.mock import MagicMock, patch

# Mock dependencies that might be missing or broken in the test env
sys.modules["seaborn_image"] = MagicMock()
sys.modules["skimage"] = MagicMock()
sys.modules["skimage.filters"] = MagicMock()
sys.modules["skimage.measure"] = MagicMock()
sys.modules["skimage.io"] = MagicMock()
sys.modules["skimage.segmentation"] = MagicMock()

from ueler.viewer.main_viewer import ImageMaskViewer

class TestFOVDetection(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_fov_detection_tmp"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_ignore_hidden_folders_and_detect_ome(self):
        # Create .UELer folder
        os.makedirs(os.path.join(self.test_dir, ".UELer"), exist_ok=True)
        
        # Create dummy OME-TIFF files (need 2 because viewer loads index 1)
        with open(os.path.join(self.test_dir, "test_image_1.ome.tif"), "w") as f:
            f.write("dummy content")
        with open(os.path.join(self.test_dir, "test_image_2.ome.tif"), "w") as f:
            f.write("dummy content")

        # Mock methods that require actual data loading or UI
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

            # Mock image cache to avoid KeyError when accessing loaded FOV
            def mock_load_fov(self, fov_name, requested_channels=None):
                # Populate cache with dummy data
                mock_img = MagicMock()
                mock_img.shape = (100, 100)
                self.image_cache[fov_name] = {"DAPI": mock_img}
            
            with patch("ueler.viewer.main_viewer.ImageMaskViewer.load_fov", side_effect=mock_load_fov, autospec=True):
                viewer = ImageMaskViewer(self.test_dir)
                
                self.assertEqual(viewer._fov_mode, "ome-tiff")
                self.assertEqual(viewer.available_fovs, ["test_image_1", "test_image_2"])

if __name__ == "__main__":
    unittest.main()
