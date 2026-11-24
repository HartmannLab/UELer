"""Coarse preview while the view is dragged, sharp frame when the drag ends."""

import types
import unittest
from unittest.mock import patch

from ueler.image_utils import drag_preview_factor
from ueler.viewer import image_display as image_display_module
from ueler.viewer.image_display import (
    ImageDisplay,
    _apply_downsample_factor_for_view,
    _render_if_view_changed,
)


class DragPreviewFactorTests(unittest.TestCase):
    def test_smallest_power_of_two_under_the_limit(self):
        self.assertEqual(drag_preview_factor(2048, 2048, 256), 8)
        self.assertEqual(drag_preview_factor(2048, 1024, 256), 8)
        self.assertEqual(drag_preview_factor(512, 512, 256), 2)
        self.assertEqual(drag_preview_factor(200, 100, 256), 1)

    def test_float_drift_in_the_limits_does_not_change_the_factor(self):
        self.assertEqual(drag_preview_factor(2048.0000001, 2048, 256), 8)

    def test_no_limit_means_no_preview(self):
        for limit in (0, None, -5, "off"):
            self.assertEqual(drag_preview_factor(2048, 2048, limit), 1)


def _display(*, drag=False, downsample=True, preview_px=256, current=1):
    changes = []
    viewer = types.SimpleNamespace(
        initialized=True,
        current_downsample_factor=current,
        drag_preview_px=preview_px,
        ui_component=types.SimpleNamespace(enable_downsample_checkbox=types.SimpleNamespace(value=downsample)),
    )

    def on_change(factor):
        changes.append(factor)
        viewer.current_downsample_factor = factor

    viewer.on_downsample_factor_changed = on_change
    # 2048 px of image on a 1200 px Retina canvas: the sharp factor is 1.
    ax = types.SimpleNamespace(bbox=types.SimpleNamespace(width=1200.0, height=1200.0))
    display = types.SimpleNamespace(main_viewer=viewer, ax=ax, _view_drag_active=drag)
    return display, changes


class FactorDuringDragTests(unittest.TestCase):
    XLIM, YLIM = (0.0, 2048.0), (2048.0, 0.0)

    def test_dragging_renders_the_coarse_preview(self):
        display, changes = _display(drag=True)
        _apply_downsample_factor_for_view(display, self.XLIM, self.YLIM)
        self.assertEqual(changes, [8])
        self.assertTrue(display._coarse_view_rendered)

    def test_without_a_drag_the_factor_is_the_sharp_one(self):
        display, changes = _display(drag=False, current=8)
        _apply_downsample_factor_for_view(display, self.XLIM, self.YLIM)
        self.assertEqual(changes, [1])
        self.assertFalse(display._coarse_view_rendered)

    def test_downsample_off_keeps_native_pixels_while_dragging(self):
        display, changes = _display(drag=True, downsample=False)
        _apply_downsample_factor_for_view(display, self.XLIM, self.YLIM)
        self.assertEqual(changes, [])  # stays at 1
        self.assertFalse(display._coarse_view_rendered)

    def test_preview_can_be_turned_off(self):
        display, changes = _display(drag=True, preview_px=0)
        _apply_downsample_factor_for_view(display, self.XLIM, self.YLIM)
        self.assertEqual(changes, [])
        self.assertFalse(display._coarse_view_rendered)

    def test_preview_never_makes_a_render_sharper(self):
        # Zoomed far out the sharp factor is already coarser than the preview.
        display, changes = _display(drag=True, preview_px=4096, current=1)
        _apply_downsample_factor_for_view(display, self.XLIM, self.YLIM)
        self.assertEqual(changes, [])
        self.assertFalse(display._coarse_view_rendered)


class _Canvas:
    def __init__(self, mode=""):
        self.toolbar = types.SimpleNamespace(mode=mode)
        self.draws = 0

    def draw_idle(self):
        self.draws += 1


def _drag_display(mode="pan/zoom"):
    ax = types.SimpleNamespace(stale=False)
    canvas = _Canvas(mode)
    display = types.SimpleNamespace(
        ax=ax,
        fig=types.SimpleNamespace(canvas=canvas),
        _view_drag_active=False,
        _coarse_view_rendered=False,
        _sharp_render_requested=False,
    )
    return display, canvas


class DragEventTests(unittest.TestCase):
    def test_press_with_the_pan_tool_starts_a_drag(self):
        display, _ = _drag_display("pan/zoom")
        ImageDisplay._on_view_drag_start(display, types.SimpleNamespace(inaxes=display.ax))
        self.assertTrue(display._view_drag_active)

    def test_press_with_the_zoom_tool_or_outside_the_axes_does_not(self):
        display, _ = _drag_display("zoom rect")
        ImageDisplay._on_view_drag_start(display, types.SimpleNamespace(inaxes=display.ax))
        self.assertFalse(display._view_drag_active)
        display, _ = _drag_display("pan/zoom")
        ImageDisplay._on_view_drag_start(display, types.SimpleNamespace(inaxes=None))
        self.assertFalse(display._view_drag_active)

    def test_release_after_a_coarse_frame_asks_for_a_sharp_one(self):
        display, canvas = _drag_display()
        display._view_drag_active = True
        display._coarse_view_rendered = True
        ImageDisplay._on_view_drag_end(display, None)
        self.assertFalse(display._view_drag_active)
        self.assertTrue(display._sharp_render_requested)
        self.assertTrue(display.ax.stale)  # survives a draw request the browser drops
        self.assertEqual(canvas.draws, 1)

    def test_release_without_a_coarse_frame_does_nothing(self):
        display, canvas = _drag_display()
        display._view_drag_active = True
        ImageDisplay._on_view_drag_end(display, None)
        self.assertFalse(display._sharp_render_requested)
        self.assertEqual(canvas.draws, 0)

    def test_leaving_the_figure_without_a_drag_does_nothing(self):
        display, canvas = _drag_display()
        display._coarse_view_rendered = True
        ImageDisplay._on_view_drag_end(display, None)
        self.assertEqual(canvas.draws, 0)


class SharpRenderOnUnchangedViewTests(unittest.TestCase):
    def _display(self):
        ax = types.SimpleNamespace(get_xlim=lambda: (0.0, 100.0), get_ylim=lambda: (100.0, 0.0))
        return types.SimpleNamespace(
            main_viewer=object(),
            ax=ax,
            prev_center_x=50.0,
            prev_center_y=50.0,
            prev_viewport_width=100.0,
            prev_viewport_height=100.0,
            _sharp_render_requested=False,
        )

    def test_unchanged_view_is_not_rendered(self):
        display = self._display()
        with patch.object(image_display_module, "_render_for_view") as render:
            self.assertFalse(_render_if_view_changed(display))
        render.assert_not_called()

    def test_requested_sharp_render_runs_once_on_the_same_view(self):
        display = self._display()
        display._sharp_render_requested = True
        with patch.object(image_display_module, "_render_for_view") as render:
            self.assertTrue(_render_if_view_changed(display))
            self.assertFalse(_render_if_view_changed(display))
        self.assertEqual(render.call_count, 1)
        self.assertFalse(display._sharp_render_requested)


if __name__ == "__main__":
    unittest.main()
