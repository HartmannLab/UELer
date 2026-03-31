
import unittest
import numpy as np
from unittest.mock import MagicMock
from ueler.rendering.engine import render_fov_to_array, ChannelRenderSettings

class MockOMEFovWrapper:
    def __init__(self, shape, ds_factor=1):
        self._shape = shape # (H, W)
        self.ds_factor = ds_factor
        self.is_ome_tiff = True
        self.channels = {"DAPI": np.zeros(shape, dtype=np.uint8)} # Placeholder

    @property
    def shape(self):
        return self._shape

    def get(self, key):
        if key in self.channels:
            return self[key]
        return None

    def __contains__(self, key):
        return key in self.channels

    def __getitem__(self, key):
        if key not in self.channels:
             raise KeyError(key)
        # Simulate returning downsampled array
        full_h, full_w = self._shape
        ds_h = max(1, full_h // self.ds_factor)
        ds_w = max(1, full_w // self.ds_factor)
        # Return a mock array with correct shape
        return np.zeros((ds_h, ds_w), dtype=np.float32)

    def set_downsample_factor(self, factor):
        self.ds_factor = factor

class TestOMERenderingFix(unittest.TestCase):
    def test_render_fov_to_array_ome_tiff(self):
        # Setup
        full_shape = (1000, 1000)
        ds_factor = 10
        wrapper = MockOMEFovWrapper(full_shape, ds_factor=ds_factor)
        
        channel_settings = {
            "DAPI": ChannelRenderSettings(color=(1.0, 0.0, 0.0), contrast_min=0, contrast_max=1)
        }
        
        # Test rendering
        # This should NOT raise ValueError
        # And should return array of size roughly 100x100
        
        result = render_fov_to_array(
            fov_name="test_fov",
            channel_arrays=wrapper,
            selected_channels=["DAPI"],
            channel_settings=channel_settings,
            downsample_factor=ds_factor
        )
        
        self.assertEqual(result.shape, (100, 100, 3))
        
    def test_render_fov_to_array_ome_tiff_zoom_out(self):
        # Simulate the crash scenario: zooming out
        # ds_factor increases
        full_shape = (349 * 8, 349 * 8) # Based on traceback numbers roughly
        # Traceback: shapes (349,349,3) (175,175,3)
        # If ds=8, shape=349. If ds=16, shape=175.
        
        ds_factor = 16
        wrapper = MockOMEFovWrapper(full_shape, ds_factor=ds_factor)
        
        channel_settings = {
            "DAPI": ChannelRenderSettings(color=(1.0, 0.0, 0.0), contrast_min=0, contrast_max=1)
        }
        
        # If the fix is working, this should pass.
        # If not, it might fail with shape mismatch if logic is wrong.
        
        result = render_fov_to_array(
            fov_name="test_fov",
            channel_arrays=wrapper,
            selected_channels=["DAPI"],
            channel_settings=channel_settings,
            downsample_factor=ds_factor
        )
        
        # Expected shape: ceil(2792 / 16) = 175
        expected_dim = int(np.ceil(full_shape[0] / ds_factor))
        self.assertEqual(result.shape, (expected_dim, expected_dim, 3))

if __name__ == '__main__':
    unittest.main()
