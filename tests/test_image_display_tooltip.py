import unittest

from ueler.viewer.tooltip_utils import format_tooltip_value


class TooltipFormattingTests(unittest.TestCase):
    def test_regular_value_uses_fixed_point(self) -> None:
        self.assertEqual(format_tooltip_value(12.345), "12.35")

    def test_small_value_uses_scientific_notation(self) -> None:
        self.assertEqual(format_tooltip_value(1.12e-6), "1.12e-06")

    def test_zero_value_stays_zero(self) -> None:
        self.assertEqual(format_tooltip_value(0.0), "0.00")

    def test_negative_small_value_preserves_sign(self) -> None:
        self.assertEqual(format_tooltip_value(-3.0e-5), "-3.00e-05")


if __name__ == "__main__":
    unittest.main()
