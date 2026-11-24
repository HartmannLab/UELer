"""Rendered map tiles are evicted by size, so the overview survives zooming in."""

import unittest
from collections import OrderedDict

import numpy as np

from ueler.viewer.virtual_map_layer import VirtualMapLayer


def _layer(cache, *, capacity=2, max_bytes=None):
    layer = object.__new__(VirtualMapLayer)
    layer._cache = cache
    layer._cache_capacity = capacity
    layer._cache_max_bytes = max_bytes
    return layer


def _tile(px):
    return np.zeros((px, px, 3), dtype=np.float32)


class TileCacheBudgetTests(unittest.TestCase):
    def test_count_limit_without_a_byte_budget(self):
        cache = OrderedDict()
        layer = _layer(cache, capacity=2)
        for i in range(3):
            layer._cache_store(("t", i), _tile(4))
        self.assertEqual(list(cache), [("t", 1), ("t", 2)])

    def test_byte_budget_keeps_many_small_overview_tiles(self):
        cache = OrderedDict()
        overview = _tile(16)  # 3 KiB
        layer = _layer(cache, capacity=2, max_bytes=64 * overview.nbytes)  # a count limit of 2 would keep 2 tiles
        for i in range(25):
            layer._cache_store(("overview", i), overview.copy())
        for i in range(4):
            layer._cache_store(("zoomed", i), _tile(32))  # 4x the bytes each
        self.assertTrue(all(("overview", i) in cache for i in range(25)))

    def test_byte_budget_evicts_oldest_first(self):
        cache = OrderedDict()
        layer = _layer(cache, max_bytes=2 * _tile(8).nbytes)
        for i in range(3):
            layer._cache_store(("t", i), _tile(8))
        self.assertEqual(list(cache), [("t", 1), ("t", 2)])


if __name__ == "__main__":
    unittest.main()
