"""Regression tests for issue #108 — the heatmap rendered but never appeared.

Root cause (heatmap-specific interaction with the ipympl backend): the plugin built and
laid out its ``sns.clustermap`` figure *inside* the ``with <output>:`` display context and
then called ``plt.show()``. Under ``%matplotlib widget`` (interactive mode) that emits the
canvas on creation and again on ``plt.show()`` — a duplicate/blank ipympl canvas. The Chart
histogram is reliable because it builds *outside* its Output and emits exactly once.

The fix (``_refresh_plot``): build the figure outside any Output context with
``plt.ioff()`` active, emit the interactive canvas exactly once via ``display(fig.canvas)``
inside a *fresh* ``Output``, then swap that Output into ``plot_section.children`` to force a
repaint. In wide mode a synchronous ``canvas.draw()`` plus a deferred single-shot
``draw_idle()`` repaint the reparented footer canvas after the frontend lays it out.

Imports the real ``ueler`` package directly (not the stubbed heatmap test harness), so it
runs cleanly in the project conda environment.
"""

import unittest
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ueler.viewer.plugin.heatmap_layers as heatmap_layers
from ueler.viewer.plugin.heatmap import HeatmapDisplay


class _FakeTimer:
    def __init__(self, interval=0):
        self.interval = interval
        self.single_shot = False
        self.started = False
        self._callback = None

    def add_callback(self, fn, *args):
        self._callback = (fn, args)

    def start(self):
        self.started = True

    def fire(self):
        fn, args = self._callback
        fn(*args)


class _FakeCanvas:
    def __init__(self):
        self.draw_calls = 0
        self.draw_idle_calls = 0
        self.timers = []

    def draw(self):
        self.draw_calls += 1

    def draw_idle(self):
        self.draw_idle_calls += 1

    def new_timer(self, interval=0):
        timer = _FakeTimer(interval)
        self.timers.append(timer)
        return timer


class Issue108HeatmapRefreshTests(unittest.TestCase):
    def setUp(self):
        # Emulate the %matplotlib widget default (interactive mode on).
        self._was_interactive = plt.isinteractive()
        plt.ion()
        self._orig_display = heatmap_layers.display
        self.display_calls = []
        heatmap_layers.display = lambda x: self.display_calls.append(x)

    def tearDown(self):
        heatmap_layers.display = self._orig_display
        if self._was_interactive:
            plt.ion()
        else:
            plt.ioff()

    def _make_heatmap(self, *, wide):
        from ipywidgets import Output, VBox

        heatmap = HeatmapDisplay.__new__(HeatmapDisplay)
        heatmap.adapter = SimpleNamespace(is_wide=lambda: wide)
        heatmap._restoring_plot_section = False
        first_output = Output()
        heatmap.plot_output = first_output
        heatmap.plot_section = VBox([first_output])

        self.generate_calls = 0
        self.interactive_during_build = None
        self.canvas = _FakeCanvas()

        def _fake_generate(figsize_override=None):
            self.generate_calls += 1
            # Record matplotlib's interactive state at build time and publish a fake figure.
            self.interactive_during_build = plt.isinteractive()
            heatmap.data = SimpleNamespace(g=SimpleNamespace(fig=SimpleNamespace(canvas=self.canvas)))

        heatmap.generate_heatmap = _fake_generate
        return heatmap, first_output

    def test_refresh_plot_swaps_fresh_output_into_section(self):
        heatmap, original = self._make_heatmap(wide=False)

        heatmap._refresh_plot()

        self.assertEqual(self.generate_calls, 1)
        self.assertIsNot(heatmap.plot_output, original)
        self.assertEqual(len(heatmap.plot_section.children), 1)
        self.assertIs(heatmap.plot_section.children[0], heatmap.plot_output)

    def test_build_runs_with_interactive_mode_off(self):
        # The figure must be built while non-interactive so ipympl does not auto-emit it.
        heatmap, _ = self._make_heatmap(wide=False)

        heatmap._refresh_plot()

        self.assertIs(self.interactive_during_build, False)
        # Interactive mode is restored afterwards.
        self.assertTrue(plt.isinteractive())

    def test_canvas_emitted_exactly_once(self):
        heatmap, _ = self._make_heatmap(wide=False)

        heatmap._refresh_plot()

        self.assertEqual(len(self.display_calls), 1)
        self.assertIs(self.display_calls[0], self.canvas)

    def test_refresh_plot_swaps_a_new_output_each_time(self):
        heatmap, _ = self._make_heatmap(wide=False)

        heatmap._refresh_plot()
        first = heatmap.plot_output
        heatmap._refresh_plot()
        second = heatmap.plot_output

        self.assertIsNot(first, second)
        self.assertIs(heatmap.plot_section.children[0], second)
        self.assertEqual(self.generate_calls, 2)

    def test_wide_mode_redraws_footer_canvas_now_and_deferred(self):
        heatmap, _ = self._make_heatmap(wide=True)

        heatmap._refresh_plot()

        # Immediate synchronous backstop draw.
        self.assertEqual(self.canvas.draw_calls, 1)
        # A single-shot deferred draw_idle scheduled for after the footer is laid out.
        self.assertEqual(len(self.canvas.timers), 1)
        timer = self.canvas.timers[0]
        self.assertTrue(timer.single_shot and timer.started)
        self.assertEqual(self.canvas.draw_idle_calls, 0)
        timer.fire()
        self.assertEqual(self.canvas.draw_idle_calls, 1)

    def test_vertical_mode_does_not_force_footer_redraw(self):
        heatmap, _ = self._make_heatmap(wide=False)

        heatmap._refresh_plot()

        self.assertEqual(self.canvas.draw_calls, 0)
        self.assertEqual(self.canvas.timers, [])

    def test_ensure_plot_canvas_attached_sets_current_output_as_sole_child(self):
        heatmap, _ = self._make_heatmap(wide=True)
        heatmap.plot_section.children = ()

        heatmap._ensure_plot_canvas_attached()

        self.assertEqual(heatmap.plot_section.children, (heatmap.plot_output,))


if __name__ == "__main__":
    unittest.main()
