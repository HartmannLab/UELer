"""The view-change render runs before the canvas rasterises (render before draw).

Before, ``ImageDisplay.on_draw`` re-rendered from ``draw_event``, which
matplotlib fires after the figure is rasterised, so the frame sent for a pan or
zoom showed the previous image and a second draw was needed. These tests pin
the new order and the tooltip's redraw-on-change behaviour.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

from ueler.viewer.image_display import ImageDisplay  # noqa: E402
from ueler.viewer.predraw import install_pre_draw_hook  # noqa: E402


class PreDrawHookTests(unittest.TestCase):
    def setUp(self):
        self.fig = plt.figure()
        self.canvas = FigureCanvasAgg(self.fig)

    def tearDown(self):
        plt.close(self.fig)

    def test_hook_runs_before_the_figure_is_rasterised(self):
        order = []
        fig_draw = self.fig.draw
        self.fig.draw = lambda renderer: (order.append("rasterise"), fig_draw(renderer))
        install_pre_draw_hook(self.canvas, lambda: order.append("hook"))
        self.canvas.draw()
        self.assertEqual(order, ["hook", "rasterise"])

    def test_draw_idle_inside_hook_is_suppressed_and_restored(self):
        calls = []
        self.canvas.draw_idle = lambda *a, **k: calls.append("draw_idle")
        install_pre_draw_hook(self.canvas, lambda: self.canvas.draw_idle())
        self.canvas.draw()
        self.assertEqual(calls, [])
        self.canvas.draw_idle()
        self.assertEqual(calls, ["draw_idle"])

    def test_class_draw_idle_is_restored_when_no_instance_override(self):
        install_pre_draw_hook(self.canvas, lambda: None)
        self.canvas.draw()
        self.assertNotIn("draw_idle", vars(self.canvas))

    def test_failing_hook_does_not_block_the_draw(self):
        drawn = []
        fig_draw = self.fig.draw
        self.fig.draw = lambda renderer: (drawn.append(True), fig_draw(renderer))

        def boom():
            raise RuntimeError("render failed")

        install_pre_draw_hook(self.canvas, boom)
        with self.assertLogs("ueler.viewer.predraw", level="ERROR"):
            self.canvas.draw()
        self.assertEqual(drawn, [True])

    def test_multiple_hooks_run_in_order(self):
        order = []
        install_pre_draw_hook(self.canvas, lambda: order.append(1))
        install_pre_draw_hook(self.canvas, lambda: order.append(2))
        self.canvas.draw()
        self.assertEqual(order, [1, 2])


def _viewer_stub(display, order):
    def update_display(_factor):
        order.append("render")
        display.img_display.set_data([[0.5]])
        display.fig.canvas.draw_idle()

    return SimpleNamespace(
        initialized=True,
        _widget_displayed=True,
        current_downsample_factor=1,
        on_downsample_factor_changed=lambda f: None,
        update_display=update_display,
        ui_component=SimpleNamespace(enable_downsample_checkbox=SimpleNamespace(value=False)),
    )


class ImageDisplayRenderOrderTests(unittest.TestCase):
    def setUp(self):
        self.display = ImageDisplay(100, 100)
        self.order = []
        self.display.main_viewer = _viewer_stub(self.display, self.order)
        fig_draw = self.display.fig.draw
        self.display.fig.draw = lambda renderer: (self.order.append("rasterise"), fig_draw(renderer))
        self.display.fig.canvas.draw()  # settle the initial viewport
        self.order.clear()

    def tearDown(self):
        plt.close(self.display.fig)

    def test_view_change_renders_before_rasterising_once(self):
        self.display.ax.set_xlim(10, 60)
        self.display.fig.canvas.draw()
        self.assertEqual(self.order, ["render", "rasterise"])
        self.assertFalse(self.display.fig.stale, "a fresh frame must not leave the figure stale")

    def test_unchanged_view_does_not_render(self):
        self.display.fig.canvas.draw()
        self.assertEqual(self.order, ["rasterise"])


class TooltipRedrawOnChangeTests(unittest.TestCase):
    def setUp(self):
        self.display = ImageDisplay(100, 100)
        self.display.fig.canvas.draw_idle = MagicMock()

    def tearDown(self):
        plt.close(self.display.fig)

    def test_hiding_an_already_hidden_tooltip_does_not_redraw(self):
        self.display._hide_tooltip()
        self.display._hide_tooltip()
        self.display.fig.canvas.draw_idle.assert_not_called()

    def test_showing_the_same_tooltip_twice_redraws_once(self):
        self.display._show_tooltip((5.0, 5.0), "Mask ID: 3")
        self.display._show_tooltip((5.0, 5.0), "Mask ID: 3")
        self.assertEqual(self.display.fig.canvas.draw_idle.call_count, 1)

    def test_tooltip_changes_redraw(self):
        self.display._show_tooltip((5.0, 5.0), "Mask ID: 3")
        self.display._show_tooltip((6.0, 5.0), "Mask ID: 3")
        self.display._show_tooltip((6.0, 5.0), "Mask ID: 4")
        self.display._hide_tooltip()
        self.assertEqual(self.display.fig.canvas.draw_idle.call_count, 4)

    def test_mouse_moving_outside_axes_redraws_only_once(self):
        event = SimpleNamespace(inaxes=None, xdata=None, ydata=None)
        self.display._show_tooltip((5.0, 5.0), "Mask ID: 3")
        self.display.fig.canvas.draw_idle.reset_mock()
        for _ in range(5):
            self.display.on_mouse_move(event)
        self.assertEqual(self.display.fig.canvas.draw_idle.call_count, 1)


if __name__ == "__main__":
    unittest.main()
