import unittest
from unittest.mock import MagicMock
import math
from ueler.data_loader import OMEFovWrapper

# Mocking the OMEFovWrapper class structure for testing _select_level logic
class MockOMEFovWrapper(OMEFovWrapper):
    def __init__(self, level_scales, base_shape=(1000, 1000)):
        self._closed = False
        self._channel_cache = {}
        self._level_specs = []
        base_h, base_w = base_shape
        for i, s in enumerate(level_scales):
            # Calculate shape based on scale (perfect scaling)
            h = math.ceil(base_h / s)
            w = math.ceil(base_w / s)
            self._level_specs.append({
                "scale": s, 
                "level_index": i,
                "shape": (h, w),
                "axes": "YX"
            })
            
    def set_imperfect_level(self, index, shape):
        self._level_specs[index]["shape"] = shape

    # We inherit _select_level from OMEFovWrapper (which we imported)
    # But we need to make sure _axis_size is available if it's used
    # OMEFovWrapper has _axis_size as static method, so it should be fine.

class TestOMELevelSelectionFix(unittest.TestCase):
    def test_select_level_exact_match(self):
        # Levels: 1, 2, 4, 8, 16
        wrapper = MockOMEFovWrapper([1, 2, 4, 8, 16])
        
        # Request ds=6
        # Exact divisors of 6: 1, 2.
        # Max scale is 2. Best = Level 2.
        # Residual = 6 // 2 = 3.
        # Effective = 2 * 3 = 6.
        
        best, residual = wrapper._select_level(6)
        effective = best["scale"] * residual
        
        print(f"Request: 6. Got Level Scale: {best['scale']}, Residual: {residual}, Effective: {effective}")
        
        self.assertEqual(effective, 6)
        self.assertEqual(best["scale"], 2)
        self.assertEqual(residual, 3)

    def test_select_level_fallback(self):
        # Levels: 2, 4, 8 (Missing 1, base shape 1000x1000)
        # Note: OMEFovWrapper usually has scale 1 level, but let's test robustness
        wrapper = MockOMEFovWrapper([2, 4, 8])
        
        # Request ds=3
        # Exact divisors of 3: None.
        # Fallback: <= 3: 2. Best = 2.
        # Residual = ceil(3/2) = 2.
        # Effective = 2 * 2 = 4.
        
        best, residual = wrapper._select_level(3)
        effective = best["scale"] * residual
        
        print(f"Request: 3. Got Level Scale: {best['scale']}, Residual: {residual}, Effective: {effective}")
        
        self.assertEqual(effective, 4)
        self.assertEqual(best["scale"], 2)

    def test_imperfect_pyramid_coverage(self):
        # Base: 1000x1000.
        # Level 1 (Scale 2): 490x490 (instead of 500x500).
        wrapper = MockOMEFovWrapper([1, 2])
        wrapper.set_imperfect_level(1, (490, 490))
        
        # Request ds=6.
        # Expected size: ceil(1000/6) = 167.
        
        # If we picked Level 1 (Scale 2):
        # Residual = 6 // 2 = 3.
        # Actual size: ceil(490/3) = 164.
        # 164 < 167 -> Fail.
        
        # Should fallback to Level 0 (Scale 1).
        # Residual = 6 // 1 = 6.
        # Actual size: ceil(1000/6) = 167.
        # 167 >= 167 -> Pass.
        
        best, residual = wrapper._select_level(6)
        
        print(f"Imperfect Pyramid Request: 6. Got Level Scale: {best['scale']}, Residual: {residual}")
        
        self.assertEqual(best["scale"], 1)
        self.assertEqual(residual, 6)

if __name__ == '__main__':
    unittest.main()
