"""Map activation reads each tile's channels once and renders once."""

import types
import unittest
from unittest.mock import patch

import dask
import dask.array as da
import numpy as np

import tests.bootstrap  # noqa: F401

from tests.test_map_mode_activation import MapModeActivationTests, _StubLayer
from ueler.data_loader import _build_channel_stat_tasks, _channel_stat_values
from ueler.rendering.channel_cache import ChannelLevelCache
from ueler.viewer.main_viewer import ImageMaskViewer


class ChannelStatValuesTests(unittest.TestCase):
    def _lazy_values(self, data):
        pct, mx = _build_channel_stat_tasks(da.from_array(data, chunks=data.shape))
        return tuple(float(v) for v in dask.compute(pct, mx))

    def test_matches_the_lazy_tasks_for_integers(self):
        data = np.random.default_rng(0).integers(0, 4000, size=(256, 256), dtype=np.uint16)
        self.assertEqual(_channel_stat_values(data), self._lazy_values(data))

    def test_matches_the_lazy_tasks_for_floats_with_nan(self):
        data = np.random.default_rng(1).random((128, 128)).astype(np.float32)
        data[::7, ::5] = np.nan
        np.testing.assert_allclose(_channel_stat_values(data), self._lazy_values(data), rtol=1e-6)


class _CountingLazy:
    """A lazy channel that fails the test if its file is read for the stats."""

    def __init__(self, data):
        self._data = data
        self.shape, self.dtype, self.ndim = data.shape, data.dtype, data.ndim
        self.computes = 0

    def compute(self):
        self.computes += 1
        return self._data.copy()


class VisitedTileStatsTests(unittest.TestCase):
    def test_stats_use_the_raster_the_render_decoded(self):
        data = np.random.default_rng(2).integers(0, 4000, size=(64, 64), dtype=np.uint16)
        source = _CountingLazy(data)
        cache = ChannelLevelCache(max_entry_fraction=1.0)
        cache.read_region("tile", "CD3", source, (0, 64, 0, 64), 4)
        synced = []
        viewer = types.SimpleNamespace(
            image_cache={"tile": {"CD3": source}},
            channel_level_cache=cache,
            _stats_computed_fovs=set(),
            channel_max_values={},
            _sync_channel_controls=synced.append,
        )
        with patch("ueler.viewer.main_viewer._build_channel_stat_tasks", side_effect=AssertionError("re-read")):
            ImageMaskViewer._update_stats_for_visited_tile(viewer, "tile", ("CD3",))
        self.assertEqual(source.computes, 1)  # the render's decode only
        self.assertIn("tile\x00CD3", viewer._stats_computed_fovs)
        self.assertIn("CD3", viewer.channel_max_values)
        self.assertEqual(synced, ["CD3"])


class ActivationViewTests(MapModeActivationTests):
    """Reuses the map activation fixture; only the tests below run here."""

    def _activate(self):
        layer = _StubLayer(0.5, (0.0, 1_024.0, 0.0, 2_048.0))
        self.viewer._map_descriptors = {"slide-1": object()}
        with patch.object(ImageMaskViewer, "_get_map_layer", lambda self, map_id: layer), patch(
            "ueler.viewer.main_viewer.adopt_current_view"
        ) as adopt:
            self.viewer._activate_map_mode("slide-1")
        return adopt

    def test_on_screen_activation_renders_once_at_the_screen_factor(self):
        self.viewer.initialized = True
        adopt = self._activate()
        adopt.assert_called_once_with(self.viewer.image_display)
        self.assertFalse(self.viewer.image_display._map_needs_initial_render)

    def test_activation_before_display_keeps_the_first_draw_render(self):
        self.viewer.initialized = True
        self.viewer._widget_displayed = False
        adopt = self._activate()
        adopt.assert_not_called()
        self.assertTrue(self.viewer.image_display._map_needs_initial_render)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for case in (ChannelStatValuesTests, VisitedTileStatsTests):
        suite.addTests(loader.loadTestsFromTestCase(case))
    for name in ("test_on_screen_activation_renders_once_at_the_screen_factor", "test_activation_before_display_keeps_the_first_draw_render"):
        suite.addTest(ActivationViewTests(name))
    return suite


if __name__ == "__main__":
    unittest.main()
