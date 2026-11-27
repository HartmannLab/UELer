"""One selection colour and style, drawn the same way everywhere.

A selected cell is drawn in the selection colour with an outline thicker than a
mask outline and a light tint inside. The style is set under Advanced Settings →
Selection and colours; a change redraws the image and recolours the scatter,
histogram and the Gallery's outline (unless the Gallery's was changed by hand).
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

import numpy as np

from ueler.constants import PLOT_BASE_COLOR, SELECTION_COLOR
from ueler.viewer.image_display import ImageDisplay, MaskSelection
from ueler.viewer.main_viewer import ImageMaskViewer
from ueler.viewer.plugin.histogram import HistogramDisplay
from ueler.viewer.plugin.plugin_base import PluginBase
from ueler.viewer.plugin.scatter_widget import ScatterPlotWidget
from ueler.viewer.selection_style import SelectionStyle, paint_selection, selection_edges, style_of

AMBER = np.array([1.0, 176 / 255, 0.0])


def _square(size=12, lo=3, hi=9, cell_id=4):
    labels = np.zeros((size, size), dtype=np.int32)
    labels[lo:hi, lo:hi] = cell_id
    return labels


class SelectionStyleTests(unittest.TestCase):
    def test_defaults(self):
        style = SelectionStyle()
        self.assertEqual(style.color, SELECTION_COLOR)
        self.assertEqual(style.color, "#ffb000")
        self.assertEqual(style.outline_width, 2)
        self.assertAlmostEqual(style.fill_opacity, 0.25)
        self.assertEqual(style.base_color, PLOT_BASE_COLOR)
        np.testing.assert_allclose(style.rgb, AMBER)

    def test_build_takes_loose_widget_values_and_falls_back(self):
        style = SelectionStyle.build("#FF00FF", "3", 1.7, "red")
        self.assertEqual((style.color, style.outline_width, style.fill_opacity, style.base_color), ("#ff00ff", 3, 1.0, "#ff0000"))
        style = SelectionStyle.build("not a colour", None, "x", None)
        self.assertEqual(style, SelectionStyle())
        self.assertEqual(SelectionStyle.build(outline_width=0).outline_width, 1)
        self.assertEqual(SelectionStyle.build(fill_opacity=-1).fill_opacity, 0.0)

    def test_style_of_a_viewer_without_one_is_the_default(self):
        self.assertEqual(style_of(SimpleNamespace()), SelectionStyle())
        self.assertEqual(style_of(mock.MagicMock()), SelectionStyle())
        custom = SelectionStyle(color="#00ff00")
        self.assertIs(style_of(SimpleNamespace(selection_style=custom)), custom)


class PaintSelectionTests(unittest.TestCase):
    def test_outline_is_thicker_than_a_mask_outline_and_inside_is_tinted(self):
        image = np.zeros((12, 12, 3), dtype=np.float32)
        self.assertTrue(paint_selection(image, _square(), SelectionStyle()))
        # 2 px: the cell's own border ring and the ring inside it are amber ...
        np.testing.assert_allclose(image[3, 5], AMBER, atol=1e-6)
        np.testing.assert_allclose(image[4, 5], AMBER, atol=1e-6)
        # ... the centre is a 25 % tint over black ...
        np.testing.assert_allclose(image[6, 6], AMBER * 0.25, atol=1e-6)
        # ... and the image far from the cell is untouched.
        np.testing.assert_allclose(image[0, 11], 0.0)

    def test_fill_opacity_zero_draws_the_outline_only(self):
        image = np.full((12, 12, 3), 0.5, dtype=np.float32)
        paint_selection(image, _square(), SelectionStyle(fill_opacity=0.0))
        np.testing.assert_allclose(image[6, 6], 0.5)
        np.testing.assert_allclose(image[3, 5], AMBER, atol=1e-6)

    def test_width_scales_down_with_the_downsample_factor(self):
        wide = selection_edges(_square(), 4, downsample_factor=1)
        narrow = selection_edges(_square(), 4, downsample_factor=4)
        self.assertGreater(wide.sum(), narrow.sum())
        self.assertEqual(narrow.sum(), selection_edges(_square(), 1).sum())

    def test_integer_images_are_0_to_255_and_alpha_is_left_alone(self):
        image = np.zeros((12, 12, 4), dtype=np.uint8)
        image[..., 3] = 200
        paint_selection(image, _square(), SelectionStyle())
        self.assertEqual(tuple(image[3, 5]), (255, 176, 0, 200))
        self.assertEqual(tuple(image[6, 6, :3]), (63, 44, 0))

    def test_offset_places_and_clips_the_cell(self):
        image = np.zeros((10, 10, 3), dtype=np.float32)
        labels = _square()  # 12 x 12, larger than the image
        self.assertTrue(paint_selection(image, labels, SelectionStyle(), offset=(-3, -3)))
        np.testing.assert_allclose(image[0, 2], AMBER, atol=1e-6)  # labels[3, 5]
        self.assertFalse(paint_selection(image, labels, SelectionStyle(), offset=(20, 20)))

    def test_nothing_selected_draws_nothing(self):
        image = np.zeros((4, 4, 3), dtype=np.float32)
        self.assertFalse(paint_selection(image, np.zeros((4, 4), dtype=int), SelectionStyle()))
        self.assertFalse(paint_selection(image, None, SelectionStyle()))
        self.assertFalse(np.any(image))


class _Capture:
    def __init__(self):
        self.array = None

    def set_data(self, array):
        self.array = np.array(array, copy=True)

    def get_array(self):
        return self.array


class ImageHighlightTests(unittest.TestCase):
    def _display(self, mask_array, selections):
        display = ImageDisplay(width=16, height=16)
        display.img_display = _Capture()
        viewer = SimpleNamespace(
            _map_mode_active=False,
            _active_map_id=None,
            initialized=False,
            current_downsample_factor=1,
            ui_component=SimpleNamespace(image_selector=SimpleNamespace(value="FOV_A")),
            full_resolution_label_masks={"cells": mask_array},
            _debug=False,
        )
        viewer.image_display = display
        display.main_viewer = viewer
        display.selected_masks_label = set(selections)
        display.combined = np.zeros(mask_array.shape + (3,), dtype=np.float32)
        self.addCleanup(lambda: __import__("matplotlib.pyplot").pyplot.close(display.fig))
        size = mask_array.shape[0]
        limits = (0, size, 0, size, 0, size, 0, size)
        with mock.patch("ueler.viewer.image_display.get_axis_limits_with_padding", return_value=limits):
            display.update_patches()
        return display.img_display.array

    def test_the_image_draws_the_selection_in_the_default_style(self):
        shown = self._display(_square(16, 4, 12, 3), [MaskSelection("FOV_A", "cells", 3)])
        np.testing.assert_allclose(shown[4, 8], AMBER, atol=1e-6)
        np.testing.assert_allclose(shown[8, 8], AMBER * 0.25, atol=1e-6)

    def test_a_cell_id_above_255_is_drawn(self):
        # The old routine cast the labels to uint8, so cell 256 became background.
        shown = self._display(_square(16, 4, 12, 256), [MaskSelection("FOV_A", "cells", 256)])
        np.testing.assert_allclose(shown[4, 8], AMBER, atol=1e-6)


class _Widget(SimpleNamespace):
    pass


class ViewerStyleTests(unittest.TestCase):
    def _viewer(self, **values):
        viewer = ImageMaskViewer.__new__(ImageMaskViewer)
        viewer.predefined_colors = {"White": "#ffffff", "Yellow": "#ffff00", "Red": "#ff0000"}
        viewer.ui_component = SimpleNamespace(
            selection_color_picker=_Widget(value=values.get("color", SELECTION_COLOR)),
            selection_outline_width_slider=_Widget(value=values.get("width", 2)),
            selection_fill_opacity_slider=_Widget(value=values.get("fill", 0.25)),
            plot_base_color_picker=_Widget(value=values.get("base", PLOT_BASE_COLOR)),
            default_mask_color_dropdown=_Widget(value=values.get("mask", "White")),
            mask_color_controls={
                "cells": _Widget(value="White"),
                "nuclei": _Widget(value="Red"),
            },
        )
        viewer.initialized = False
        return viewer

    def test_the_style_is_read_from_the_widgets(self):
        viewer = self._viewer(color="#00FF00", width=4, fill=0.0, base="#123456")
        self.assertEqual(viewer.selection_style, SelectionStyle("#00ff00", 4, 0.0, "#123456"))
        self.assertEqual(viewer.selection_outline_color, "#00ff00")

    def test_a_viewer_without_the_widgets_uses_the_defaults(self):
        viewer = ImageMaskViewer.__new__(ImageMaskViewer)
        viewer.predefined_colors = {"White": "#ffffff"}
        self.assertEqual(viewer.selection_style, SelectionStyle())
        self.assertEqual(viewer.default_mask_color, "White")

    def test_default_mask_colour_moves_only_masks_still_on_the_old_default(self):
        viewer = self._viewer(mask="Yellow")
        self.assertEqual(viewer.default_mask_color, "Yellow")
        viewer.on_default_mask_color_change({"old": "White", "new": "Yellow"})
        controls = viewer.ui_component.mask_color_controls
        self.assertEqual(controls["cells"].value, "Yellow")
        self.assertEqual(controls["nuclei"].value, "Red")  # chosen by hand: kept

    def test_a_style_change_reaches_every_plugin_with_the_previous_colour(self):
        viewer = self._viewer()
        calls = []

        class _Plugin(PluginBase):
            def __init__(self):
                pass

            def on_selection_style_change(self, previous_color=None):
                calls.append(previous_color)

        viewer.SidePlots = SimpleNamespace(a=_Plugin(), b=_Plugin(), not_a_plugin=object())
        viewer.on_selection_style_change()
        self.assertEqual(calls, [SELECTION_COLOR, SELECTION_COLOR])
        viewer.ui_component.selection_color_picker.value = "#00ff00"
        viewer.on_selection_style_change()
        self.assertEqual(calls[-1], SELECTION_COLOR)  # the colour before this change
        viewer.ui_component.selection_color_picker.value = "#0000ff"
        viewer.on_selection_style_change()
        self.assertEqual(calls[-1], "#00ff00")


class _FakeScatter:
    def __init__(self, by=None):
        self.state = {"by": by}
        self.calls = []

    def color(self, **kwargs):
        if not kwargs:
            return dict(self.state)
        self.calls.append(kwargs)
        return None


class PlotStyleTests(unittest.TestCase):
    def _scatter(self, by=None):
        widget = ScatterPlotWidget.__new__(ScatterPlotWidget)
        widget._scatter = _FakeScatter(by)
        widget._color = by
        widget._selected_color = (1, 0, 0, 1)
        widget._point_color = (0, 0, 1, 0.85)
        return widget

    def test_scatter_recolours_selected_and_unselected_points(self):
        widget = self._scatter()
        widget.set_style_colors(selected_color="#ffb000", point_color="#123456")
        call = widget._scatter.calls[-1]
        np.testing.assert_allclose(call["selected"], (1.0, 176 / 255, 0.0, 1.0))
        np.testing.assert_allclose(call["default"], (0x12 / 255, 0x34 / 255, 0x56 / 255, 0.85))

    def test_scatter_coloured_by_a_column_keeps_its_colours(self):
        widget = self._scatter(by="cluster")
        widget.set_style_colors(selected_color="#ffb000", point_color="#123456")
        self.assertEqual(set(widget._scatter.calls[-1]), {"selected"})

    def test_histogram_recolours_its_bars(self):
        plugin = HistogramDisplay.__new__(HistogramDisplay)
        plugin.main_viewer = SimpleNamespace(selection_style=SelectionStyle(color="#00ff00", base_color="#222222"))
        bars = [SimpleNamespace(glyph=SimpleNamespace(fill_color="", line_color="")) for _ in range(3)]
        plugin._style_renderers = [tuple(bars)]
        plugin.on_selection_style_change()
        full, hits, selected = bars
        self.assertEqual(full.glyph.fill_color, "#222222")
        self.assertEqual((hits.glyph.fill_color, hits.glyph.line_color), ("#00ff00", "#00ff00"))
        self.assertEqual(selected.glyph.fill_color, "#00ff00")


if __name__ == "__main__":
    unittest.main()
