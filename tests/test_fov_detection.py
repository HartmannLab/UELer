# test_fov_detection.py

import os
import tempfile
import unittest
import glob


class MockImageMaskViewer:
    """Mock version of ImageMaskViewer for testing FOV detection logic."""
    
    def __init__(self, base_folder):
        self.base_folder = base_folder
    
    def _has_tiff_files(self, fov_name):
        """Check if a directory contains .tif or .tiff files, including in 'rescaled' subdirectory."""
        fov_path = os.path.join(self.base_folder, fov_name)
        
        # Check for 'rescaled' subfolder first
        rescaled_path = os.path.join(fov_path, 'rescaled')
        if os.path.isdir(rescaled_path):
            channel_folder = rescaled_path
        else:
            channel_folder = fov_path
        
        # Get list of TIFF files in the channel folder
        tiff_files = glob.glob(os.path.join(channel_folder, '*.tiff'))
        tiff_files += glob.glob(os.path.join(channel_folder, '*.tif'))
        
        return bool(tiff_files)


class TestFOVDetection(unittest.TestCase):
    """Test FOV detection logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_has_tiff_files_with_tiff_files(self):
        """Test _has_tiff_files returns True when directory contains .tiff files."""
        viewer = MockImageMaskViewer(self.temp_dir)

        # Create a subdirectory with .tiff files
        fov_dir = os.path.join(self.temp_dir, "FOV1")
        os.makedirs(fov_dir)
        
        # Create a dummy .tiff file
        with open(os.path.join(fov_dir, "channel1.tiff"), "w") as f:
            f.write("dummy")

        self.assertTrue(viewer._has_tiff_files("FOV1"))

    def test_has_tiff_files_with_tif_files(self):
        """Test _has_tiff_files returns True when directory contains .tif files."""
        viewer = MockImageMaskViewer(self.temp_dir)

        fov_dir = os.path.join(self.temp_dir, "FOV2")
        os.makedirs(fov_dir)
        
        with open(os.path.join(fov_dir, "channel1.tif"), "w") as f:
            f.write("dummy")

        self.assertTrue(viewer._has_tiff_files("FOV2"))

    def test_has_tiff_files_with_rescaled_subdirectory(self):
        """Test _has_tiff_files checks rescaled subdirectory for TIFF files."""
        viewer = MockImageMaskViewer(self.temp_dir)

        fov_dir = os.path.join(self.temp_dir, "FOV3")
        rescaled_dir = os.path.join(fov_dir, "rescaled")
        os.makedirs(rescaled_dir)
        
        # Put TIFF files in rescaled subdirectory
        with open(os.path.join(rescaled_dir, "channel1.tiff"), "w") as f:
            f.write("dummy")

        self.assertTrue(viewer._has_tiff_files("FOV3"))

    def test_has_tiff_files_without_tiff_files(self):
        """Test _has_tiff_files returns False when directory has no TIFF files."""
        viewer = MockImageMaskViewer(self.temp_dir)

        fov_dir = os.path.join(self.temp_dir, "FOV4")
        os.makedirs(fov_dir)
        
        # Create non-TIFF files
        with open(os.path.join(fov_dir, "data.txt"), "w") as f:
            f.write("dummy")

        self.assertFalse(viewer._has_tiff_files("FOV4"))

    def test_has_tiff_files_empty_directory(self):
        """Test _has_tiff_files returns False for empty directory."""
        viewer = MockImageMaskViewer(self.temp_dir)

        fov_dir = os.path.join(self.temp_dir, "FOV5")
        os.makedirs(fov_dir)

        self.assertFalse(viewer._has_tiff_files("FOV5"))

    def test_has_tiff_files_nonexistent_directory(self):
        """Test _has_tiff_files returns False for nonexistent directory."""
        viewer = MockImageMaskViewer(self.temp_dir)

        self.assertFalse(viewer._has_tiff_files("NonExistentFOV"))

    def test_available_fovs_filters_directories_with_tiff_files(self):
        """Test that available_fovs logic only includes directories with TIFF files."""
        # Create test directories
        fov1_dir = os.path.join(self.temp_dir, "FOV1")
        fov2_dir = os.path.join(self.temp_dir, "FOV2") 
        empty_dir = os.path.join(self.temp_dir, "EmptyDir")
        dot_dir = os.path.join(self.temp_dir, ".ueler")
        
        os.makedirs(fov1_dir)
        os.makedirs(fov2_dir)
        os.makedirs(empty_dir)
        os.makedirs(dot_dir)
        
        # Add TIFF files to FOV1 and FOV2
        with open(os.path.join(fov1_dir, "channel1.tiff"), "w") as f:
            f.write("dummy")
        with open(os.path.join(fov2_dir, "channel1.tif"), "w") as f:
            f.write("dummy")
        
        viewer = MockImageMaskViewer(self.temp_dir)
        
        # Simulate the list comprehension logic from main_viewer.py
        available_fovs = [fov for fov in os.listdir(self.temp_dir)
                        if os.path.isdir(os.path.join(self.temp_dir, fov)) and viewer._has_tiff_files(fov)]
        
        # Should only include FOV1 and FOV2
        self.assertIn("FOV1", available_fovs)
        self.assertIn("FOV2", available_fovs)
        self.assertNotIn("EmptyDir", available_fovs)
        self.assertNotIn(".ueler", available_fovs)


if __name__ == '__main__':
    unittest.main()