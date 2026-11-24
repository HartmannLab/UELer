"""Tests for the decoded-channel cache on the render path."""

import unittest

import numpy as np

from ueler.rendering.channel_cache import ChannelLevelCache
from ueler.rendering.engine import ChannelRenderSettings, render_fov_to_array
from ueler.viewer.main_viewer import _cached_channel_reader


class _LazyArray:
    """Minimal stand-in for a Dask array that counts how often it is decoded."""

    def __init__(self, data, counter=None):
        self._data = data
        self.counter = counter if counter is not None else {"compute": 0}

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim

    def __getitem__(self, item):
        return _LazyArray(self._data[item], self.counter)

    def compute(self):
        self.counter["compute"] += 1
        return self._data.copy()


def _image(height=64, width=48, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4000, size=(height, width), dtype=np.uint16)


class FullResolutionPathTests(unittest.TestCase):
    def test_regions_match_lazy_slicing_at_every_factor(self):
        data = _image()
        source = _LazyArray(data)
        cache = ChannelLevelCache()
        for factor in (1, 2, 3, 4, 8):
            for region in ((0, 48, 0, 64), (8, 40, 16, 64), (5, 33, 7, 50)):
                xmin, xmax, ymin, ymax = region
                got = cache.read_region("fov", "CD3", source, region, factor)
                expected = data[ymin:ymax:factor, xmin:xmax:factor]
                np.testing.assert_array_equal(got, expected)

    def test_source_is_decoded_once_across_renders(self):
        source = _LazyArray(_image())
        cache = ChannelLevelCache()
        for factor, region in ((1, (0, 48, 0, 64)), (4, (0, 48, 0, 64)), (2, (8, 24, 8, 40))):
            cache.read_region("fov", "CD3", source, region, factor)
        self.assertEqual(source.counter["compute"], 1)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 2)
        # Full resolution, plus the small level copy kept for the factor-4 read.
        self.assertEqual(cache.nbytes, 64 * 48 * 2 + 16 * 12 * 2)

    def test_in_memory_source_is_not_cached(self):
        cache = ChannelLevelCache()
        self.assertIsNone(cache.read_region("fov", "CD3", _image(), (0, 48, 0, 64), 1))
        self.assertEqual(len(cache), 0)

    def test_non_2d_source_is_not_cached(self):
        cache = ChannelLevelCache()
        source = _LazyArray(np.zeros((2, 8, 8), dtype=np.uint16))
        self.assertIsNone(cache.read_region("fov", "CD3", source, (0, 8, 0, 8), 1))

    def test_replaced_source_is_decoded_again(self):
        cache = ChannelLevelCache()
        old = _LazyArray(_image(seed=1))
        new_data = _image(seed=2)
        new = _LazyArray(new_data)
        cache.read_region("fov", "CD3", old, (0, 48, 0, 64), 1)
        got = cache.read_region("fov", "CD3", new, (0, 48, 0, 64), 1)
        np.testing.assert_array_equal(got, new_data)
        self.assertEqual(new.counter["compute"], 1)
        self.assertEqual(cache.nbytes, new_data.nbytes)


class DownsampledLevelPathTests(unittest.TestCase):
    def _cache_too_small_for_full_resolution(self, data):
        # Per-entry cap is max_bytes / 8: below the full raster, above a factor-4 level.
        return ChannelLevelCache(max_bytes=data.nbytes * 8 // 4)

    def test_aligned_region_is_served_from_the_level(self):
        data = _image(64, 64)
        source = _LazyArray(data)
        cache = self._cache_too_small_for_full_resolution(data)
        for region in ((0, 64, 0, 64), (8, 40, 16, 64), (4, 62, 0, 61)):
            xmin, xmax, ymin, ymax = region
            got = cache.read_region("fov", "CD3", source, region, 4)
            np.testing.assert_array_equal(got, data[ymin:ymax:4, xmin:xmax:4])
        self.assertEqual(source.counter["compute"], 1)
        self.assertEqual(cache.nbytes, 16 * 16 * 2)

    def test_misaligned_region_and_factor_one_fall_back(self):
        data = _image(64, 64)
        source = _LazyArray(data)
        cache = self._cache_too_small_for_full_resolution(data)
        self.assertIsNone(cache.read_region("fov", "CD3", source, (3, 64, 0, 64), 4))
        self.assertIsNone(cache.read_region("fov", "CD3", source, (0, 64, 0, 64), 1))

    def test_level_over_the_entry_cap_falls_back(self):
        data = _image(64, 64)
        cache = ChannelLevelCache(max_bytes=64)
        self.assertIsNone(cache.read_region("fov", "CD3", _LazyArray(data), (0, 64, 0, 64), 2))
        self.assertEqual(len(cache), 0)


class BudgetTests(unittest.TestCase):
    def test_least_recently_used_channel_is_evicted_first(self):
        data = _image(32, 32)  # 2 KiB per channel
        cache = ChannelLevelCache(max_bytes=data.nbytes * 2, max_entry_fraction=1.0)
        a, b, c = (_LazyArray(data) for _ in range(3))
        cache.read_region("fov", "a", a, (0, 32, 0, 32), 1)
        cache.read_region("fov", "b", b, (0, 32, 0, 32), 1)
        cache.read_region("fov", "a", a, (0, 32, 0, 32), 1)  # a is now the most recent
        cache.read_region("fov", "c", c, (0, 32, 0, 32), 1)
        self.assertLessEqual(cache.nbytes, cache.max_bytes)
        cache.read_region("fov", "a", a, (0, 32, 0, 32), 1)
        self.assertEqual(a.counter["compute"], 1)
        cache.read_region("fov", "b", b, (0, 32, 0, 32), 1)
        self.assertEqual(b.counter["compute"], 2)

    def test_drop_fov_and_disable(self):
        data = _image(16, 16)
        cache = ChannelLevelCache()
        cache.read_region("fov1", "a", _LazyArray(data), (0, 16, 0, 16), 1)
        cache.read_region("fov2", "a", _LazyArray(data), (0, 16, 0, 16), 1)
        cache.drop_fov("fov1")
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, data.nbytes)
        cache.set_max_bytes(0)
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.read_region("fov2", "a", _LazyArray(data), (0, 16, 0, 16), 1))


class LevelCopyTests(unittest.TestCase):
    """Coarse reads keep a small level copy that outlives full resolution."""

    def test_overview_is_served_from_the_level_copy_after_full_resolution_is_evicted(self):
        data = _image(64, 64)
        source = _LazyArray(data)
        cache = ChannelLevelCache(max_entry_fraction=1.0)
        cache.read_region("tile", "CD3", source, (0, 64, 0, 64), 8)
        cache._remove(("tile", "CD3", 1))  # full resolution pushed out by other tiles
        got = cache.read_region("tile", "CD3", source, (0, 64, 0, 64), 8)
        np.testing.assert_array_equal(got, data[::8, ::8])
        self.assertEqual(source.counter["compute"], 1)

    def test_no_copy_for_fine_factors_or_misaligned_regions(self):
        cache = ChannelLevelCache(max_entry_fraction=1.0)
        source = _LazyArray(_image(64, 64))
        cache.read_region("tile", "CD3", source, (0, 64, 0, 64), 2)
        cache.read_region("tile", "CD3", source, (3, 64, 0, 64), 8)
        self.assertEqual(len(cache._levels), 0)

    def test_level_copies_stay_within_their_share_and_the_total_budget(self):
        data = _image(64, 64)  # 8 KiB full, 128 B level at factor 8
        cache = ChannelLevelCache(max_bytes=4 * data.nbytes, max_entry_fraction=1.0, level_fraction=1 / 64)
        for i in range(8):
            cache.read_region(f"tile{i}", "CD3", _LazyArray(data), (0, 64, 0, 64), 8)
        self.assertLessEqual(cache._level_nbytes, cache.max_level_bytes)
        self.assertLessEqual(cache.nbytes, cache.max_bytes)

    def test_drop_fov_and_replaced_source_drop_the_copy(self):
        cache = ChannelLevelCache(max_entry_fraction=1.0)
        old = _LazyArray(_image(64, 64, seed=1))
        cache.read_region("tile", "CD3", old, (0, 64, 0, 64), 8)
        cache._remove(("tile", "CD3", 1))
        new_data = _image(64, 64, seed=2)
        got = cache.read_region("tile", "CD3", _LazyArray(new_data), (0, 64, 0, 64), 8)
        np.testing.assert_array_equal(got, new_data[::8, ::8])
        cache.drop_fov("tile")
        self.assertEqual((len(cache), cache.nbytes), (0, 0))


class EngineIntegrationTests(unittest.TestCase):
    def test_render_is_identical_with_and_without_the_cache(self):
        channels = {"CD3": _image(seed=3), "CD8": _image(seed=4)}
        settings = {
            "CD3": ChannelRenderSettings(color=(1.0, 0.0, 0.0), contrast_min=100.0, contrast_max=3000.0),
            "CD8": ChannelRenderSettings(color=(0.0, 0.5, 1.0), contrast_min=0.0, contrast_max=2500.0),
        }
        cache = ChannelLevelCache()
        for factor, region in ((1, (0, 48, 0, 64)), (2, (8, 40, 16, 64)), (4, (0, 48, 0, 64))):
            kwargs = dict(downsample_factor=factor, region_xy=region)
            plain = render_fov_to_array(
                "fov", {k: _LazyArray(v) for k, v in channels.items()}, ("CD3", "CD8"), settings, **kwargs
            )
            lazy = {k: _LazyArray(v) for k, v in channels.items()}
            cached = render_fov_to_array(
                "fov", lazy, ("CD3", "CD8"), settings,
                channel_reader=_cached_channel_reader(cache, "fov"), **kwargs,
            )
            np.testing.assert_array_equal(cached, plain)

    def test_no_cache_means_no_reader(self):
        self.assertIsNone(_cached_channel_reader(None, "fov"))


if __name__ == "__main__":
    unittest.main()
