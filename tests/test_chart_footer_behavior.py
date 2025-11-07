import sys
import types
import unittest
from types import SimpleNamespace

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

# ---------------------------------------------------------------------------
# Optional dependency stubs
# ---------------------------------------------------------------------------
if "anywidget" not in sys.modules:
    sys.modules["anywidget"] = types.ModuleType("anywidget")

if "jscatter" not in sys.modules:
    jscatter_stub = types.ModuleType("jscatter")

    class _StubScatter:
        def __init__(self, *_, **__):
            self.widget = SimpleNamespace(
                layout=None,
                observe=lambda *args, **kwargs: None,
                mouse_mode=None,
            )
            self._selection = []

        def axes(self, *_, **__):
            return None

        def height(self, *_, **__):
            return None

        def size(self, *_, **__):
            return None

        def color(self, *_, **__):
            return None

        def tooltip(self, *_, **__):
            return None

        def show(self, *_, **__):
            return self.widget

        def selection(self, values=None):
            if values is not None:
                self._selection = list(values)
            return self._selection

    def _stub_compose(*entries, **_kwargs):
        return entries

    jscatter_stub.Scatter = _StubScatter
    jscatter_stub.compose = _stub_compose
    sys.modules["jscatter"] = jscatter_stub

if "numpy" not in sys.modules:  # pragma: no cover - stub fallback
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.integer = int
    numpy_stub.ndarray = object
    numpy_stub.greater = staticmethod(lambda a, b: a > b)
    numpy_stub.less = staticmethod(lambda a, b: a < b)
    sys.modules["numpy"] = numpy_stub

if "dask" not in sys.modules:  # pragma: no cover - stub fallback
    dask_stub = types.ModuleType("dask")

    def _noop_delayed(func):
        return func

    dask_stub.delayed = _noop_delayed
    sys.modules["dask"] = dask_stub

if "seaborn_image" not in sys.modules:  # pragma: no cover - stub fallback
    seaborn_stub = types.ModuleType("seaborn_image")
    seaborn_stub.imshow = lambda *_, **__: None
    sys.modules["seaborn_image"] = seaborn_stub

if "tifffile" not in sys.modules:  # pragma: no cover - stub fallback
    tifffile_stub = types.ModuleType("tifffile")

    class _StubTiffFile:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    tifffile_stub.TiffFile = _StubTiffFile
    sys.modules["tifffile"] = tifffile_stub

if "cv2" not in sys.modules:  # pragma: no cover - stub fallback
    cv2_stub = types.ModuleType("cv2")
    sys.modules["cv2"] = cv2_stub

if "skimage" not in sys.modules:  # pragma: no cover - stub fallback
    skimage_stub = types.ModuleType("skimage")
    segmentation_stub = types.ModuleType("skimage.segmentation")
    segmentation_stub.find_boundaries = lambda *_, **__: None
    io_stub = types.ModuleType("skimage.io")
    io_stub.imread = lambda *_, **__: None
    io_stub.imsave = lambda *_, **__: None
    exposure_stub = types.ModuleType("skimage.exposure")
    exposure_stub.rescale_intensity = lambda image, **_: image
    transform_stub = types.ModuleType("skimage.transform")
    transform_stub.resize = lambda *_, **__: None
    color_stub = types.ModuleType("skimage.color")
    color_stub.label2rgb = lambda *_, **__: None
    measure_stub = types.ModuleType("skimage.measure")
    measure_stub.regionprops_table = lambda *_, **__: {}
    draw_stub = types.ModuleType("skimage.draw")
    draw_stub.circle_perimeter = lambda *_, **__: ([], [])
    sys.modules["skimage"] = skimage_stub
    sys.modules["skimage.segmentation"] = segmentation_stub
    sys.modules["skimage.io"] = io_stub
    sys.modules["skimage.exposure"] = exposure_stub
    sys.modules["skimage.transform"] = transform_stub
    sys.modules["skimage.color"] = color_stub
    sys.modules["skimage.measure"] = measure_stub
    sys.modules["skimage.draw"] = draw_stub

try:  # pragma: no cover - prefer real library when available
    import ipywidgets as widgets  # type: ignore
except ImportError:  # pragma: no cover - test stub fallback
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
            self.children = tuple(children) if children is not None else ()
            self.value = kwargs.get("value")
            self.options = kwargs.get("options", [])
            self.layout = kwargs.get("layout", _Layout())
            self.selected_index = kwargs.get("selected_index")
            self.button_style = kwargs.get("button_style", "")
            self.icon = kwargs.get("icon", "")
            self.description = kwargs.get("description", "")
            self.tooltip = kwargs.get("tooltip", "")
            self._titles = {}

        def observe(self, *_, **__):
            return None

        def on_click(self, *_, **__):
            return None

        def set_title(self, index, title):
            self._titles[index] = title

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    for _name in [
        "Accordion",
        "Box",
        "Button",
        "Checkbox",
        "Dropdown",
        "FloatSlider",
        "HBox",
        "HTML",
        "Image",
        "IntSlider",
        "IntText",
        "Layout",
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
    widgets.Layout = _Layout
    sys.modules["ipywidgets"] = widgets

if "matplotlib.pyplot" not in sys.modules:  # pragma: no cover - stub fallback
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = types.ModuleType("matplotlib.pyplot")

    class _Canvas:
        def mpl_connect(self, *_, **__):
            return None

        def draw_idle(self, *_, **__):
            return None

        @property
        def toolbar(self):
            return SimpleNamespace(
                _nav_stack=lambda: SimpleNamespace(push=lambda *args, **kwargs: None)
            )

    class _Figure:
        def __init__(self):
            self.canvas = _Canvas()

        def tight_layout(self):
            return None

    class _Axes:
        def __init__(self, fig):
            self.figure = fig
            self.collections = []

        def hist(self, *_, **__):
            return None

        def set_xlabel(self, *_, **__):
            return None

        def set_ylabel(self, *_, **__):
            return None

        def axvline(self, *_, **__):
            return SimpleNamespace(remove=lambda: None)

    def _subplots(*_, **__):
        fig = _Figure()
        ax = _Axes(fig)
        return fig, ax

    def _show(*_, **__):
        return None

    pyplot_stub.subplots = _subplots
    pyplot_stub.show = _show
    matplotlib_stub.pyplot = pyplot_stub
    font_manager_stub = types.ModuleType("matplotlib.font_manager")
    font_manager_stub.FontProperties = type("FontProperties", (), {"__init__": lambda self, *_, **__: None})
    matplotlib_stub.font_manager = font_manager_stub
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub
    sys.modules["matplotlib.font_manager"] = font_manager_stub

if "matplotlib.font_manager" not in sys.modules:  # pragma: no cover - stub fallback
    font_manager_stub = types.ModuleType("matplotlib.font_manager")
    font_manager_stub.FontProperties = type("FontProperties", (), {"__init__": lambda self, *_, **__: None})
    sys.modules["matplotlib.font_manager"] = font_manager_stub

if "pandas" not in sys.modules:  # pragma: no cover - stub fallback
    pandas_stub = types.ModuleType("pandas")

    class _Series(list):
        def dropna(self):
            return _Series([value for value in self if value is not None])

        def unique(self):
            unique_values = []
            for item in self:
                if item not in unique_values:
                    unique_values.append(item)
            return _Series(unique_values)

        def tolist(self):
            return list(self)

    class _Columns(list):
        def tolist(self):
            return list(self)

    class DataFrame:
        def __init__(self, data):
            self._data = {key: list(values) for key, values in data.items()}
            self.columns = _Columns(list(self._data.keys()))
            length = len(next(iter(self._data.values()), []))
            self.index = list(range(length))

        def copy(self):
            return DataFrame(self._data)

        def __getitem__(self, key):
            return _Series(self._data[key])

        def __contains__(self, key):
            return key in self._data

        def loc(self, *_, **__):  # pragma: no cover - unused in tests
            raise NotImplementedError

    pandas_stub.DataFrame = DataFrame
    pandas_stub.Series = _Series
    pandas_stub.api = SimpleNamespace(
        types=SimpleNamespace(
            is_numeric_dtype=lambda values: all(
                isinstance(value, (int, float)) for value in values
            ),
            is_object_dtype=lambda values: any(
                isinstance(value, str) for value in values
            ),
        )
    )
    sys.modules["pandas"] = pandas_stub

if "IPython.display" not in sys.modules:  # pragma: no cover - stub fallback
    ipy_display = types.ModuleType("IPython.display")

    def _noop_display(*_, **__):
        return None

    ipy_display.display = _noop_display
    sys.modules["IPython.display"] = ipy_display

if "viewer.plugin.cell_gallery" not in sys.modules:  # pragma: no cover - stub fallback
    cell_stub = types.ModuleType("viewer.plugin.cell_gallery")

    class CellGalleryDisplay:  # pragma: no cover - stub
        def __init__(self, *_, **__):
            self.ui = widgets.VBox()

        def set_selected_cells(self, *_):
            return None

    cell_stub.CellGalleryDisplay = CellGalleryDisplay
    sys.modules["viewer.plugin.cell_gallery"] = cell_stub

if "viewer.plugin.heatmap" not in sys.modules:  # pragma: no cover - stub fallback
    heatmap_stub = types.ModuleType("viewer.plugin.heatmap")

    class HeatmapDisplay:  # pragma: no cover - stub
        def __init__(self, *_, **__):
            self.ui = widgets.VBox()

    heatmap_stub.HeatmapDisplay = HeatmapDisplay
    sys.modules["viewer.plugin.heatmap"] = heatmap_stub

if "viewer.annotation_display" not in sys.modules:  # pragma: no cover - stub fallback
    annotation_stub = types.ModuleType("viewer.annotation_display")

    class AnnotationDisplay:  # pragma: no cover - stub
        def __init__(self, *_, **__):
            self.ui = widgets.VBox()

    annotation_stub.AnnotationDisplay = AnnotationDisplay
    sys.modules["viewer.annotation_display"] = annotation_stub

import pandas as pd

from viewer.plugin.chart import ChartDisplay
from viewer.plugin.plugin_base import PluginBase
from viewer.ui_components import update_wide_plugin_panel


class _StubViewer:
    def __init__(self):
        self._debug = False
        self.base_folder = "/tmp"
        self.cell_table = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        self.SidePlots = SimpleNamespace(cell_gallery_output=_StubCellGallery())
        self.BottomPlots = SimpleNamespace()
        self.refresh_calls = 0

    def refresh_bottom_panel(self, ordering=None):
        self.refresh_calls += 1


class _DummyScatter:
    def __init__(self, title):
        self.identifier = title
        self.state = SimpleNamespace(title=title)

    def dispose(self):
        return None


class _StubAdapter:
    def __init__(self, wide=True):
        self._wide = bool(wide)

    def is_wide(self):
        return self._wide

    def set_wide(self, state):
        self._wide = bool(state)


class _StubHeatmap(PluginBase):
    def __init__(self, viewer):
        super().__init__(viewer, 6, 4)
        self.main_viewer = viewer
        self.SidePlots_id = "heatmap_output"
        self.displayed_name = "Heatmap"
        self.adapter = _StubAdapter(wide=True)
        self.plot_output = widgets.Output()
        self.controls_section = widgets.VBox()
        self.plot_section = widgets.VBox(children=(self.plot_output,))
        self.restore_vertical_canvas_calls = 0
        self.restore_footer_canvas_calls = 0

    def wide_panel_layout(self):
        return {
            "title": self.displayed_name,
            "control": self.controls_section,
            "content": self.plot_section,
        }

    def restore_vertical_canvas(self):
        self.restore_vertical_canvas_calls += 1

    def restore_footer_canvas(self):
        self.restore_footer_canvas_calls += 1
        children = getattr(self.plot_section, "children", ())
        if self.plot_output not in children:
            self.plot_section.children = children + (self.plot_output,)


class _StubCellGallery:
    def __init__(self):
        self.received = None

    def set_selected_cells(self, indices):
        self.received = indices


class _ViewerWithFooter(_StubViewer):
    def __init__(self):
        super().__init__()
        self.SidePlots = SimpleNamespace()
        self.BottomPlots = SimpleNamespace()
        self.wide_plugin_tab = widgets.Tab(children=tuple(), layout=widgets.Layout())
        self.wide_plugin_tab.selected_index = None
        self.wide_plugin_panel = widgets.VBox(children=tuple(), layout=widgets.Layout(display="none"))
        self.SidePlots.heatmap_output = _StubHeatmap(self)

    def refresh_bottom_panel(self, ordering=None):
        super().refresh_bottom_panel(ordering)
        update_wide_plugin_panel(self, ordering)


class ChartDisplayFooterTests(unittest.TestCase):
    def setUp(self):
        self.viewer = _StubViewer()
        self.chart = ChartDisplay(self.viewer, width=6, height=4)
        self.viewer.refresh_calls = 0

    def test_selection_does_not_forward_when_unlinked(self):
        gallery = self.viewer.SidePlots.cell_gallery_output
        self.chart.setup_observe()
        self.chart.ui_component.cell_gallery_linked_checkbox.value = False

        self.chart.selected_indices.value = {1}

        self.assertIsNone(gallery.received)

    def test_single_point_click_state_tracks_widget_selection(self):
        self.chart._on_scatter_selection({7}, "widget")
        self.assertEqual(self.chart.single_point_click_state, 1)

        self.chart._on_scatter_selection({7, 8}, "widget")
        self.assertEqual(self.chart.single_point_click_state, 0)

        self.chart._on_scatter_selection(set(), "widget")
        self.assertEqual(self.chart.single_point_click_state, 0)

    def test_single_point_click_state_tracks_external_selection(self):
        self.chart._apply_external_selection({11})
        self.assertEqual(self.chart.single_point_click_state, 1)

        self.chart._apply_external_selection({11, 12})
        self.assertEqual(self.chart.single_point_click_state, 0)

    def test_selection_forwards_to_cell_gallery_when_linked(self):
        gallery = self.viewer.SidePlots.cell_gallery_output
        self.chart.setup_observe()
        self.chart.ui_component.cell_gallery_linked_checkbox.value = True

        indices = {1, 2}
        self.chart.selected_indices.value = indices

        self.assertEqual(gallery.received, indices)

    def test_single_selection_does_not_update_cell_gallery_when_linked(self):
        gallery = self.viewer.SidePlots.cell_gallery_output
        self.chart.setup_observe()
        self.chart.ui_component.cell_gallery_linked_checkbox.value = True

        multi = {1, 2}
        self.chart._on_scatter_selection(multi, "widget")
        self.assertEqual(gallery.received, multi)

        self.chart._on_scatter_selection({3}, "widget")

        self.assertEqual(gallery.received, multi)

    def test_footer_layout_toggles_with_scatter_count(self):
        # Initial state: vertical layout retained
        self.assertEqual(self.chart._section_location, "vertical")
        self.assertEqual(
            list(self.chart.ui.children),
            [self.chart.controls_section, self.chart.plot_section],
        )
        self.assertEqual(self.viewer.refresh_calls, 0)

        # One scatter: stays vertical, no footer refresh
        self.chart._scatter_views["s1"] = _DummyScatter("s1")
        self.chart._sync_panel_location()
        self.assertEqual(self.chart._section_location, "vertical")
        self.assertEqual(self.viewer.refresh_calls, 0)

        # Two scatters: switch to footer with refresh
        self.chart._scatter_views["s2"] = _DummyScatter("s2")
        self.chart._sync_panel_location()
        self.assertEqual(self.chart._section_location, "horizontal")
        self.assertEqual(self.viewer.refresh_calls, 1)
        self.assertEqual(list(self.chart.ui.children), [self.chart._wide_notice])
        layout = self.chart.wide_panel_layout()
        self.assertIsNotNone(layout)
        self.assertIs(layout["control"], self.chart.controls_section)
        self.assertIs(layout["content"], self.chart.plot_section)

        # Subsequent sync does not double refresh when already horizontal
        self.chart._sync_panel_location()
        self.assertEqual(self.viewer.refresh_calls, 1)

        # Drop back to a single scatter: returns to vertical and refreshes
        self.chart._scatter_views.pop("s2")
        self.chart._sync_panel_location()
        self.assertEqual(self.chart._section_location, "vertical")
        self.assertEqual(self.viewer.refresh_calls, 2)
        self.assertEqual(
            list(self.chart.ui.children),
            [self.chart.controls_section, self.chart.plot_section],
        )
        layout = self.chart.wide_panel_layout()
        self.assertIsNone(layout)


class HeatmapFooterPersistenceTests(unittest.TestCase):
    def test_heatmap_survives_chart_refresh(self):
        viewer = _ViewerWithFooter()
        heatmap = viewer.SidePlots.heatmap_output
        chart = ChartDisplay(viewer, width=6, height=4)
        viewer.SidePlots.chart_output = chart

        viewer.refresh_calls = 0
        heatmap.restore_footer_canvas_calls = 0

        viewer.refresh_bottom_panel()
        self.assertIs(viewer.BottomPlots.heatmap_output, heatmap)
        self.assertIn(heatmap.plot_output, getattr(heatmap.plot_section, "children", ()))
        self.assertGreaterEqual(heatmap.restore_footer_canvas_calls, 1)
        self.assertEqual(len(viewer.wide_plugin_tab.children), 1)

        chart._scatter_views["s1"] = _DummyScatter("s1")
        chart._scatter_views["s2"] = _DummyScatter("s2")
        chart._sync_panel_location()

        self.assertIs(viewer.BottomPlots.heatmap_output, heatmap)
        self.assertIn(heatmap.plot_output, getattr(heatmap.plot_section, "children", ()))
        self.assertGreaterEqual(heatmap.restore_footer_canvas_calls, 2)
        self.assertEqual(len(viewer.wide_plugin_tab.children), 2)

        chart._scatter_views.pop("s2")
        chart._sync_panel_location()

        self.assertIs(viewer.BottomPlots.heatmap_output, heatmap)
        self.assertIn(heatmap.plot_output, getattr(heatmap.plot_section, "children", ()))
        self.assertGreaterEqual(heatmap.restore_footer_canvas_calls, 3)
        self.assertEqual(len(viewer.wide_plugin_tab.children), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
