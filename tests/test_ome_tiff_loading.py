import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np

from ueler.data_loader import OMEFovWrapper, extract_ome_channel_names


class _FakeReduction:
    def __init__(self, value):
        self.value = value

    def compute(self):  # pragma: no cover - trivial helper
        return self.value


class FakeZarrArray:
    def __init__(self, data):
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.ndim = self._data.ndim
        self.dtype = self._data.dtype

    def __getitem__(self, key):
        return FakeZarrArray(self._data[key])

    def persist(self):  # pragma: no cover - compatibility shim
        return self

    def compute(self):
        return self._data

    def max(self):
        return _FakeReduction(self._data.max())

    def __array__(self):  # pragma: no cover - fallback for numpy conversions
        return self._data


class FakeLevel:
    def __init__(self, axes, shape):
        self.axes = axes
        self.shape = shape


class FakeSeries:
    def __init__(self, arrays, axes="CYX"):
        self.axes = axes
        self.shape = arrays[0].shape
        self._arrays = arrays
        self.levels = tuple(FakeLevel(axes, array.shape) for array in arrays)

    def aszarr(self, level=None):
        idx = 0 if level is None else level
        return FakeZarrArray(self._arrays[idx])


class DummyTifHandle:
    def __init__(self, series):
        self.series = [series]
        self.closed = False

    def close(self):  # pragma: no cover - simple flag setter
        self.closed = True


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

    def _pyramid_arrays(self):
        base = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8)
        level_2 = base[:, ::2, ::2]
        level_4 = base[:, ::4, ::4]
        return [base, level_2, level_4]

    def _build_specs(self, arrays):
        specs = []
        base_height = arrays[0].shape[-2]
        for idx, array in enumerate(arrays):
            height = array.shape[-2]
            scale = max(1, base_height // height)
            specs.append(
                {
                    "level_index": idx,
                    "axes": "CYX",
                    "shape": array.shape,
                    "scale": scale,
                    "array": None,
                }
            )
        return specs

    @contextmanager
    def _patched_wrapper(self, ds_factor):
        arrays = self._pyramid_arrays()
        series = FakeSeries(arrays)
        fake_tif_handle = DummyTifHandle(series)

        tifffile_module = MagicMock()
        tifffile_module.TiffFile.return_value = fake_tif_handle

        fake_da_module = MagicMock()
        fake_da_module.from_zarr.side_effect = lambda store: store

        with patch("ueler.data_loader._ensure_tifffile", return_value=tifffile_module), \
             patch("ueler.data_loader._ensure_dask", return_value=(MagicMock(), fake_da_module)), \
             patch("ueler.data_loader.extract_ome_channel_names", return_value=["DAPI", "CD4"]), \
             patch.object(OMEFovWrapper, "_init_levels", return_value=None):
            wrapper = OMEFovWrapper("dummy.ome.tif", ds_factor=ds_factor)
            wrapper._series = series
            wrapper._level_specs = self._build_specs(arrays)
            wrapper._level_count = len(wrapper._level_specs)
            wrapper.channel_names = ["DAPI", "CD4"]
            wrapper._name_to_index = {name: idx for idx, name in enumerate(wrapper.channel_names)}
            wrapper._channel_cache.clear()
            yield wrapper, fake_tif_handle

    def test_ome_wrapper_uses_pyramid_level(self):
        with self._patched_wrapper(ds_factor=4) as (wrapper, _):
            tile = wrapper["DAPI"].compute()
            self.assertEqual(tile.shape, (2, 2))

    def test_residual_stride_applies_to_selected_level(self):
        with self._patched_wrapper(ds_factor=6) as (wrapper, _):
            tile = wrapper["CD4"].compute()
            self.assertEqual(tile.shape, (1, 1))

    def test_set_downsample_factor_clears_cache(self):
        with self._patched_wrapper(ds_factor=2) as (wrapper, _):
            coarse = wrapper["DAPI"].compute()
            self.assertEqual(coarse.shape, (4, 4))
            wrapper.set_downsample_factor(1)
            fine = wrapper["DAPI"].compute()
            self.assertEqual(fine.shape, (8, 8))

    def test_wrapper_close_closes_underlying_file(self):
        with self._patched_wrapper(ds_factor=2) as (wrapper, fake_tif):
            wrapper.close()
            self.assertTrue(fake_tif.closed)

if __name__ == "__main__":
    unittest.main()
