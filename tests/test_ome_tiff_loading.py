import unittest
import os
from unittest.mock import MagicMock, patch
import numpy as np
from ueler.data_loader import OMEFovWrapper, extract_ome_channel_names

class TestOMETiffLoading(unittest.TestCase):
    def test_extract_ome_channel_names(self):
        # Mock tifffile
        with patch("ueler.data_loader._ensure_tifffile") as mock_ensure:
            mock_tifffile = MagicMock()
            mock_ensure.return_value = mock_tifffile
            
            mock_tif = MagicMock()
            mock_tifffile.TiffFile.return_value.__enter__.return_value = mock_tif
            
            # Mock OME metadata
            xml_str = """
            <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
                <Image ID="Image:0">
                    <Pixels>
                        <Channel ID="Channel:0:0" Name="DAPI" />
                        <Channel ID="Channel:0:1" Name="CD4" />
                        <Channel ID="Channel:0:2" Name="CD8" />
                    </Pixels>
                </Image>
            </OME>
            """
            mock_tif.ome_metadata = xml_str
            
            names = extract_ome_channel_names("dummy.ome.tif")
            self.assertEqual(names, ["DAPI", "CD4", "CD8"])

    def test_ome_fov_wrapper(self):
        with patch("ueler.data_loader._ensure_tifffile") as mock_ensure_tiff, \
             patch("ueler.data_loader._ensure_imread") as mock_ensure_imread:
            
            # Mock tifffile for metadata
            mock_tifffile = MagicMock()
            mock_ensure_tiff.return_value = mock_tifffile
            mock_tif = MagicMock()
            mock_tifffile.TiffFile.return_value.__enter__.return_value = mock_tif
            
            xml_str = """
            <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
                <Image ID="Image:0">
                    <Pixels>
                        <Channel ID="Channel:0:0" Name="DAPI" />
                        <Channel ID="Channel:0:1" Name="CD4" />
                    </Pixels>
                </Image>
            </OME>
            """
            mock_tif.ome_metadata = xml_str
            
            # Mock dask_image.imread
            mock_imread = MagicMock()
            mock_ensure_imread.return_value = mock_imread
            
            # Mock dask array
            # Shape (C, Y, X) = (2, 100, 100)
            mock_arr = MagicMock()
            mock_arr.shape = (2, 100, 100)
            mock_arr.ndim = 3
            mock_arr.chunks = ((1, 1), (100,), (100,))
            mock_arr.rechunk.return_value = mock_arr
            
            # Mock slicing
            def getitem(key):
                # key is tuple of slices
                # We expect (idx)
                return "sliced_array"
            
            mock_arr.__getitem__.side_effect = getitem
            
            mock_imread.return_value = mock_arr
            
            wrapper = OMEFovWrapper("dummy.ome.tif", ds_factor=2)
            
            # Verify rechunk called
            mock_arr.rechunk.assert_called_with({-1: 1024, -2: 1024})
            
            self.assertEqual(wrapper.get_channel_names(), ["DAPI", "CD4"])
            self.assertIn("DAPI", wrapper)
            self.assertIn("CD4", wrapper)
            
            # Test __getitem__
            # wrapper["DAPI"] should call mock_arr[0] (no downsampling)
            res = wrapper["DAPI"]
            self.assertEqual(res, "sliced_array")
            
            # Verify call args
            # mock_arr.__getitem__ called with (0)
            args = mock_arr.__getitem__.call_args[0][0]
            self.assertEqual(args, 0)

if __name__ == "__main__":
    unittest.main()
