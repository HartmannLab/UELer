"""Unit tests for the scatter-plot axis domain padding (#118).

``_padded_domain`` frames each jscatter axis with a small margin so that data
points at the extremes are not clipped at the canvas edge.
"""

import unittest

import numpy as np

from ueler.viewer.plugin.scatter_widget import _padded_domain


class PaddedDomainTests(unittest.TestCase):
    def test_normal_range_pads_five_percent_each_side(self):
        lo, hi = _padded_domain([0.0, 10.0], fraction=0.05)
        self.assertAlmostEqual(lo, -0.5)
        self.assertAlmostEqual(hi, 10.5)

    def test_padding_keeps_all_points_inside(self):
        values = [1.0, 3.5, 7.0, 9.0]
        lo, hi = _padded_domain(values)
        self.assertLess(lo, min(values))
        self.assertGreater(hi, max(values))

    def test_all_equal_column_gets_symmetric_nonzero_pad(self):
        lo, hi = _padded_domain([5.0, 5.0, 5.0], fraction=0.05)
        self.assertLess(lo, 5.0)
        self.assertGreater(hi, 5.0)
        # Symmetric around the constant value.
        self.assertAlmostEqual(5.0 - lo, hi - 5.0)

    def test_all_zero_column_falls_back_to_unit_pad(self):
        lo, hi = _padded_domain([0.0, 0.0])
        self.assertEqual((lo, hi), (-1.0, 1.0))

    def test_empty_input_returns_unit_domain(self):
        self.assertEqual(_padded_domain([]), (-1.0, 1.0))

    def test_all_non_finite_returns_unit_domain(self):
        self.assertEqual(_padded_domain([np.nan, np.inf, -np.inf]), (-1.0, 1.0))

    def test_non_finite_values_are_ignored(self):
        lo, hi = _padded_domain([0.0, np.nan, 10.0, np.inf], fraction=0.05)
        self.assertAlmostEqual(lo, -0.5)
        self.assertAlmostEqual(hi, 10.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
