"""Regression tests for the ipympl ``'NoneType' object has no attribute 'handle_json'`` crash.

The crash came from pyplot being used on the export plugin's worker thread:
under ``%matplotlib widget`` each ``plt.figure()`` builds an ipympl ``Canvas``
ipywidget, and ``savefig`` on it sets ``canvas.manager = None`` for its whole
duration, so a comm message serviced on the main thread in that window hit the
unguarded ``self.manager.handle_json(content)``.

These tests cover both halves of the fix: the export paths no longer touch
pyplot, and the guard in ``ueler/viewer/ipympl_guard.py`` drops messages that
arrive while the manager is unset.
"""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ueler.viewer.ipympl_guard import install_canvas_message_guard
from ueler.viewer.plugin import export_fovs
from ueler.viewer.plugin.export_fovs import BatchExportPlugin, _agg_figure
from ueler.viewer.scale_bar import ScaleBarSpec

_HAS_IPYMPL = importlib.util.find_spec("ipympl") is not None


def _spec() -> ScaleBarSpec:
    return ScaleBarSpec(pixel_length=16.0, physical_length_um=10.0, label="10 µm")


class AggFigureHelperTests(unittest.TestCase):
    def test_agg_figure_is_not_registered_with_pyplot(self) -> None:
        before = set(plt.get_fignums())
        _agg_figure((2.0, 2.0), 100)
        # Not in Gcf, so it cannot become the active figure and displace
        # whatever the user's own notebook cells are plotting into.
        self.assertEqual(set(plt.get_fignums()), before)

    def test_agg_figure_canvas_is_agg_whatever_the_backend(self) -> None:
        fig = _agg_figure((2.0, 2.0), 100)
        self.assertIsInstance(fig.canvas, FigureCanvasAgg)
        # buffer_rgba is what _render_with_scale_bar reads back; pinning the
        # canvas to Agg is what guarantees it exists.
        self.assertTrue(hasattr(fig.canvas, "buffer_rgba"))


class ExportPathsAvoidPyplotTests(unittest.TestCase):
    """The worker thread must never reach pyplot's global state machine."""

    def setUp(self) -> None:
        self.plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        # Larger than ``dpi`` in both axes: _render_with_scale_bar reshapes the
        # canvas buffer to the input's exact shape, and the figsize floor of 1.0
        # inch makes that mismatch (and silently drop the bar) below that size.
        # Separate, pre-existing defect — see the tracking note.
        self.array = np.zeros((400, 400, 3), dtype=np.uint8)

    def test_module_does_not_import_pyplot(self) -> None:
        self.assertFalse(
            hasattr(export_fovs, "plt"),
            "export_fovs must not hold a pyplot reference — it is what let a "
            "worker-thread figure become an ipympl widget",
        )
        source = Path(export_fovs.__file__).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )
        self.assertNotIn("import pyplot", code)
        self.assertNotIn("import matplotlib.pyplot", code)

    def test_render_with_scale_bar_draws_without_pyplot(self) -> None:
        fignums_before = set(plt.get_fignums())
        with mock.patch.object(plt, "figure") as figure_mock, mock.patch.object(
            plt, "subplots"
        ) as subplots_mock:
            rendered = self.plugin._render_with_scale_bar(self.array, _spec(), 100)
        figure_mock.assert_not_called()
        subplots_mock.assert_not_called()
        self.assertEqual(set(plt.get_fignums()), fignums_before)
        # The bar really was drawn: a black input picks up white pixels.
        self.assertEqual(rendered.shape, self.array.shape)
        self.assertGreater(int(rendered.max()), 0)

    def test_write_pdf_with_scale_bar_saves_without_pyplot(self) -> None:
        fignums_before = set(plt.get_fignums())
        with tempfile.TemporaryDirectory() as folder:
            out = os.path.join(folder, "crop.pdf")
            with mock.patch.object(plt, "figure") as figure_mock:
                self.plugin._write_pdf_with_scale_bar(self.array, out, 100, _spec())
            figure_mock.assert_not_called()
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)
        self.assertEqual(set(plt.get_fignums()), fignums_before)

    def test_write_pdf_propagates_failures(self) -> None:
        """The old implementation's try/finally swallowed nothing; keep that."""
        with self.assertRaises(Exception):
            self.plugin._write_pdf_with_scale_bar(
                self.array, "/nonexistent-dir-xyz/crop.pdf", 100, None
            )

    def test_render_with_scale_bar_returns_input_on_failure(self) -> None:
        with mock.patch.object(
            export_fovs, "add_scale_bar", side_effect=RuntimeError("boom")
        ):
            rendered = self.plugin._render_with_scale_bar(self.array, _spec(), 100)
        self.assertIs(rendered, self.array)


@unittest.skipUnless(_HAS_IPYMPL, "ipympl is not installed")
class CanvasMessageGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        from ipympl.backend_nbagg import Canvas

        self.Canvas = Canvas
        self._original = Canvas.__dict__.get("_handle_message")
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._original is not None:
            self.Canvas._handle_message = self._original
        else:  # pragma: no cover - only if upstream drops the attribute
            delattr(self.Canvas, "_handle_message")

    def test_install_is_idempotent(self) -> None:
        self.assertTrue(install_canvas_message_guard())
        first = self.Canvas._handle_message
        self.assertTrue(install_canvas_message_guard())
        self.assertIs(self.Canvas._handle_message, first)

    def test_message_is_dropped_while_manager_is_unset(self) -> None:
        calls = []
        self.Canvas._handle_message = lambda self, obj, content, buffers: calls.append(content)
        install_canvas_message_guard()

        canvas = SimpleNamespace(manager=None, _closed=False)
        result = self.Canvas._handle_message(canvas, None, {"type": "motion_notify"}, [])

        self.assertIsNone(result)
        self.assertEqual(calls, [], "the message must not reach the upstream handler")

    def test_closing_still_marks_the_canvas_closed(self) -> None:
        self.Canvas._handle_message = lambda self, obj, content, buffers: None
        install_canvas_message_guard()

        canvas = SimpleNamespace(manager=None, _closed=False)
        self.Canvas._handle_message(canvas, None, {"type": "closing"}, [])
        self.assertTrue(canvas._closed)

    def test_messages_pass_through_once_a_manager_exists(self) -> None:
        calls = []
        self.Canvas._handle_message = lambda self, obj, content, buffers: calls.append(content)
        install_canvas_message_guard()

        canvas = SimpleNamespace(manager=object(), _closed=False)
        self.Canvas._handle_message(canvas, None, {"type": "motion_notify"}, [])
        self.assertEqual(calls, [{"type": "motion_notify"}])

    def test_real_canvas_without_a_manager_does_not_raise(self) -> None:
        """The exact shape of the reported crash.

        A ``Canvas`` built outside ``new_figure_manager_given_figure`` has
        ``manager is None`` — the same state ``savefig`` puts a live canvas in —
        and the unguarded handler raises ``AttributeError: 'NoneType' object has
        no attribute 'handle_json'`` here.
        """
        from matplotlib.figure import Figure

        install_canvas_message_guard()
        canvas = self.Canvas(Figure())
        self.assertIsNone(canvas.manager)
        canvas._handle_message(None, {"type": "motion_notify", "x": 1, "y": 1}, [])


@unittest.skipUnless(_HAS_IPYMPL, "ipympl is not installed")
class ViewerInstallsGuardTests(unittest.TestCase):
    def test_main_viewer_installs_the_guard_before_building_figures(self) -> None:
        import inspect

        from ueler.viewer import main_viewer

        source = inspect.getsource(main_viewer.ImageMaskViewer.__init__)
        self.assertIn("install_canvas_message_guard()", source)
        # ipympl binds the handler as a bound method in Canvas.__init__, so the
        # call has to precede the first ImageDisplay.
        self.assertLess(
            source.index("install_canvas_message_guard()"),
            source.index("ImageDisplay("),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
