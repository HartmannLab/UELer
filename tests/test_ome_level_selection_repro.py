
import unittest
from unittest.mock import MagicMock
import math

# Mocking the OMEFovWrapper class structure for testing _select_level logic
class MockOMEFovWrapper:
    def __init__(self, level_scales):
        self._level_specs = [{"scale": s, "level_index": i} for i, s in enumerate(level_scales)]

    def _select_level(self, ds_factor: int):
        ds = max(1, int(ds_factor) or 1)
        best = self._level_specs[0]
        for level in self._level_specs:
            if level["scale"] <= ds and level["scale"] >= best["scale"]:
                best = level
        residual = max(1, int(math.ceil(ds / best["scale"])))
        return best, residual

class TestOMELevelSelection(unittest.TestCase):
    def test_select_level_exact_match(self):
        # Levels: 1, 2, 4, 8, 16
        wrapper = MockOMEFovWrapper([1, 2, 4, 8, 16])
        
        # Request ds=6
        # Current logic:
        # <= 6: 1, 2, 4. Max is 4.
        # Best = 4.
        # Residual = ceil(6/4) = 2.
        # Effective = 4 * 2 = 8.
        # We want 6. We get 8. (Over-downsampled -> smaller image)
        
        best, residual = wrapper._select_level(6)
        effective = best["scale"] * residual
        
        print(f"Request: 6. Got Level Scale: {best['scale']}, Residual: {residual}, Effective: {effective}")
        
        # This asserts the CURRENT BROKEN behavior
        self.assertEqual(effective, 8) 
        self.assertNotEqual(effective, 6)

if __name__ == '__main__':
    unittest.main()
