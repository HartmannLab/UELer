import builtins
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

import pandas as pd


_SCATTER_WIDGET_PATH = (
    Path(__file__).resolve().parents[1]
    / "ueler"
    / "viewer"
    / "plugin"
    / "scatter_widget.py"
)


class ScatterWidgetDependencyTests(unittest.TestCase):
    def test_import_guard_surfaces_anywidget_instructions(self):
        original_import = builtins.__import__

        def mocked_import(name, *args, **kwargs):
            if name == "anywidget":
                raise ImportError("mocked missing anywidget")
            return original_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location(
            "tmp_scatter_widget_guard", str(_SCATTER_WIDGET_PATH)
        )
        module = importlib.util.module_from_spec(spec)

        with mock.patch("builtins.__import__", side_effect=mocked_import):
            with self.assertRaisesRegex(
                ImportError,
                "pip install anywidget",
            ):
                spec.loader.exec_module(module)  # type: ignore[assignment]


class ScatterWidgetColoringTests(unittest.TestCase):
    def test_set_categorical_colors_aligns_with_indices(self):
        from ueler.viewer.plugin.scatter_widget import ScatterPlotWidget

        frame = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=[10, 20, 30])
        widget = ScatterPlotWidget("test", frame, "x", "y")
        categories = pd.Series({10: 1, 30: 2})
        color_map = {1: (1, 0, 0, 1), 2: (0, 1, 0, 1)}

        widget.set_categorical_colors(categories, color_map, default_color=(0, 0, 1, 1))

        scatter = widget.scatter
        encoded = scatter._color_data.loc[frame.index]
        self.assertEqual(encoded.iloc[0], 0)
        self.assertEqual(encoded.iloc[1], 2)
        self.assertEqual(encoded.iloc[2], 1)
        self.assertEqual(scatter._color_map[0], (1.0, 0.0, 0.0, 1.0))
        self.assertEqual(scatter._color_map[1], (0.0, 1.0, 0.0, 1.0))
        self.assertEqual(scatter._color_map[2], (0.0, 0.0, 1.0, 1.0))


class ScatterWidgetCanvasWidthTests(unittest.TestCase):
    def test_explicit_width_is_deterministic_no_container_measurement(self):
        # In the multi-pair matrix the plot gets an explicit pixel width so
        # jscatter does not rely on the frontend measuring the (initially hidden)
        # grid cell — the failure mode that left the canvas narrow and the
        # right-hand y-axis hidden until "reset view" (#118 reply).
        from ueler.viewer.plugin.scatter_widget import ScatterPlotWidget

        frame = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=[10, 20, 30])
        widget = ScatterPlotWidget("test", frame, "x", "y")

        # Defaults to 'auto' (bound to the container).
        self.assertEqual(widget.scatter.width(), "auto")

        widget.set_canvas_width(360)
        self.assertEqual(widget.scatter.width(), 360)
        # The DOM width must NOT be pinned to the plot width, or the right-hand
        # y-axis (drawn outside the plot canvas) is clipped again (#118 reply).
        self.assertIsNone(getattr(widget._jwidget.layout, "width", None))

        widget.set_canvas_width("auto")
        self.assertEqual(widget.scatter.width(), "auto")
        self.assertEqual(getattr(widget._jwidget.layout, "width", None), "100%")


if __name__ == "__main__":
    unittest.main()
