import math
import sys
import types
import unittest
from types import SimpleNamespace


if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")

    def _as_array(data, dtype=None):
        if isinstance(data, (list, tuple)):
            return list(data)
        return [data]

    numpy_stub.float32 = float
    numpy_stub.float64 = float
    numpy_stub.int32 = int
    numpy_stub.int64 = int
    numpy_stub.uint8 = int
    numpy_stub.bool_ = bool
    numpy_stub.newaxis = None
    numpy_stub.ndarray = object
    numpy_stub.array = _as_array

    def _zeros(shape, dtype=None):
        if isinstance(shape, int):
            return [0] * shape
        if not shape:
            return []
        if len(shape) == 1:
            return [0] * shape[0]
        return [_zeros(shape[1:], dtype) for _ in range(shape[0])]

    numpy_stub.zeros = _zeros
    numpy_stub.clip = lambda arr, a_min, a_max: arr
    numpy_stub.isfinite = lambda value: True
    numpy_stub.unique = lambda seq: list(dict.fromkeys(seq))
    numpy_stub.arange = lambda n: list(range(n))
    numpy_stub.floor = math.floor
    numpy_stub.ceil = math.ceil
    numpy_stub.round = lambda value: int(round(value))
    numpy_stub.atleast_3d = lambda value: value
    numpy_stub.max = max
    numpy_stub.min = min
    numpy_stub.percentile = lambda data, q: 0
    numpy_stub.dstack = lambda arrays: arrays[0] if arrays else []

    sys.modules["numpy"] = numpy_stub

if "ipywidgets" not in sys.modules:
    widgets = types.ModuleType("ipywidgets")

    class _Layout:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _Widget:
        def __init__(self, *args, **kwargs):
            children = kwargs.get("children")
            if children is None and args:
                children = args[0]
            if children is None:
                children = tuple()
            self.children = tuple(children)
            self.value = kwargs.get("value")
            self.allowed_tags = kwargs.get("allowed_tags", [])
            self.options = kwargs.get("options", [])
            self.layout = kwargs.get("layout", _Layout())
            self.selected_index = kwargs.get("selected_index")
            self._titles = {}
            self.allow_new = kwargs.get("allow_new", False)
            self.allow_duplicates = kwargs.get("allow_duplicates", False)
            self.description = kwargs.get("description", "")
            self.tooltip = kwargs.get("tooltip", "")
            self.icon = kwargs.get("icon", "")
            self.button_style = kwargs.get("button_style", "")

        def observe(self, *_, **__):
            return None

        def unobserve(self, *_, **__):
            return None

        def on_click(self, *_, **__):
            return None

        def set_title(self, index, title):
            self._titles[index] = title

        def reset(self, *_, **__):
            return None

        def clear_output(self, *_, **__):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    widgets.Layout = _Layout
    for _name in [
        "Accordion",
    "Box",
        "Button",
        "Checkbox",
        "Combobox",
        "ColorPicker",
        "Dropdown",
        "FloatSlider",
        "HBox",
        "HTML",
        "Image",
        "IntText",
        "Output",
        "SelectMultiple",
        "Tab",
        "TagsInput",
        "Text",
        "Textarea",
        "ToggleButtons",
        "VBox",
        "Widget",
    ]:
        setattr(widgets, _name, _Widget)

    sys.modules["ipywidgets"] = widgets

if "IPython.display" not in sys.modules:
    ipy_display = types.ModuleType("IPython.display")

    def _noop_display(*_, **__):
        return None

    ipy_display.display = _noop_display
    sys.modules["IPython.display"] = ipy_display

if "viewer.plugin.chart" not in sys.modules:
    chart_stub = types.ModuleType("viewer.plugin.chart")

    class ChartDisplay:  # pragma: no cover - stubbed for tests
        def __init__(self, *_, **__):
            canvas = types.SimpleNamespace(draw_idle=lambda: None)
            fig = types.SimpleNamespace(canvas=canvas)
            collection = types.SimpleNamespace(set_facecolors=lambda *_: None)
            ax = types.SimpleNamespace(collections=[collection])
            self.data = types.SimpleNamespace(g=types.SimpleNamespace(ax=ax, fig=fig))
            self.ui_component = types.SimpleNamespace(
                y_axis_selector=types.SimpleNamespace(value="None"),
                impose_fov_checkbox=types.SimpleNamespace(value=False),
            )

        def color_points(self, *_):
            return None

    chart_stub.ChartDisplay = ChartDisplay
    sys.modules["viewer.plugin.chart"] = chart_stub

if "viewer.plugin.cell_gallery" not in sys.modules:
    cell_stub = types.ModuleType("viewer.plugin.cell_gallery")

    class CellGalleryDisplay:  # pragma: no cover - stub
        def __init__(self, *_, **__):
            pass

        def set_selected_cells(self, *_):
            return None

    cell_stub.CellGalleryDisplay = CellGalleryDisplay
    sys.modules["viewer.plugin.cell_gallery"] = cell_stub

if "viewer.plugin.heatmap" not in sys.modules:
    heatmap_stub = types.ModuleType("viewer.plugin.heatmap")

    class HeatmapDisplay:  # pragma: no cover - stub for tests
        def __init__(self, *_, **__):
            self.ui = sys.modules["ipywidgets"].VBox()

    heatmap_stub.HeatmapDisplay = HeatmapDisplay
    sys.modules["viewer.plugin.heatmap"] = heatmap_stub

if "viewer.annotation_display" not in sys.modules:
    annotation_stub = types.ModuleType("viewer.annotation_display")

    class AnnotationDisplay:  # pragma: no cover - stub
        def __init__(self, *_, **__):
            self.ui = sys.modules["ipywidgets"].VBox()

    annotation_stub.AnnotationDisplay = AnnotationDisplay
    sys.modules["viewer.annotation_display"] = annotation_stub

from viewer.ui_components import (
    build_wide_plugin_pane,
    collect_wide_plugin_entries,
    update_wide_plugin_panel,
)
from viewer.plugin.plugin_base import PluginBase


widgets = sys.modules["ipywidgets"]


class ToggleFooterPlugin(PluginBase):
    def __init__(self, viewer):
        super().__init__(viewer, 6, 3)
        self.displayed_name = "Heatmap"
        self.main_viewer = viewer
        self.ui = widgets.VBox()
        self.controls = widgets.VBox()
        self.content = widgets.VBox()
        self.horizontal = False

    def wide_panel_layout(self):
        if self.horizontal:
            return {"control": self.controls, "content": self.content}
        return None


class ViewerHarness:
    def __init__(self):
        self.SidePlots = SimpleNamespace()
        self.BottomPlots = SimpleNamespace()
        self.wide_plugin_tab = widgets.Tab(children=tuple(), layout=widgets.Layout())
        self.wide_plugin_tab.selected_index = None
        self.wide_plugin_panel = widgets.VBox(
            [self.wide_plugin_tab],
            layout=widgets.Layout(display="none"),
        )


class WidePanelHelperTests(unittest.TestCase):
    def test_build_wide_plugin_pane_wraps_control_and_content(self):
        control = widgets.VBox()
        content = widgets.VBox()

        pane = build_wide_plugin_pane(control, content)

        self.assertEqual(len(pane.children), 2)
        left_column, right_column = pane.children
        self.assertIn(control, left_column.children)
        self.assertIn(content, right_column.children)

    def test_toggle_moves_plugin_between_side_and_bottom(self):
        viewer = ViewerHarness()
        plugin = ToggleFooterPlugin(viewer)
        viewer.SidePlots.heatmap_output = plugin

        # Start vertical: footer hidden, namespace empty.
        update_wide_plugin_panel(viewer)
        self.assertEqual(len(viewer.wide_plugin_tab.children), 0)
        self.assertEqual(vars(viewer.BottomPlots), {})
        self.assertEqual(viewer.wide_plugin_panel.layout.display, "none")

        # Activate horizontal layout: plugin surfaces in footer.
        plugin.horizontal = True
        update_wide_plugin_panel(viewer)
        self.assertEqual(len(viewer.wide_plugin_tab.children), 1)
        self.assertTrue(hasattr(viewer.BottomPlots, "heatmap_output"))
        self.assertIs(viewer.BottomPlots.heatmap_output, plugin)
        self.assertEqual(viewer.wide_plugin_panel.layout.display, "")

        # Return to vertical: footer clears again.
        plugin.horizontal = False
        update_wide_plugin_panel(viewer)
        self.assertEqual(len(viewer.wide_plugin_tab.children), 0)
        self.assertEqual(vars(viewer.BottomPlots), {})
        self.assertEqual(viewer.wide_plugin_panel.layout.display, "none")

        # collect_wide_plugin_entries mirrors plugin state
        plugin.horizontal = True
        entries = collect_wide_plugin_entries(viewer)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["plugin"], plugin)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
