import unittest


class TestColorPalette(unittest.TestCase):
    def test_default_palette_and_selection(self):
        from ueler.viewer.color_palette import ColorPalette, DEFAULT_COLORS

        p = ColorPalette()
        # default colors preserved
        self.assertEqual(p.colors, DEFAULT_COLORS)

        # programmatic selection (single-select)
        p.select(DEFAULT_COLORS[2])
        self.assertEqual(p.value, DEFAULT_COLORS[2])

        # clear selection
        p.select(None)
        self.assertIsNone(p.value)

    def test_multi_select(self):
        from ueler.viewer.color_palette import ColorPalette

        p = ColorPalette(multi=True)
        p.select(p.colors[0])
        p.select(p.colors[1])
        # order may vary, but the two entries must be present
        self.assertCountEqual(p.value, [p.colors[0], p.colors[1]])


if __name__ == '__main__':
    unittest.main()
