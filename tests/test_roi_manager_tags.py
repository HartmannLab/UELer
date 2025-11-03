import math
import os
import sys
import types
import unittest

from types import SimpleNamespace

# Provide lightweight stubs for heavy optional dependencies used by the plugin.
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = object  # minimal stub for type annotations
    pandas_stub.Series = object
    pandas_stub.isna = lambda value: value is None
    sys.modules["pandas"] = pandas_stub

if "matplotlib" not in sys.modules:
    matplotlib_stub = types.ModuleType("matplotlib")
    sys.modules["matplotlib"] = matplotlib_stub

if "matplotlib.colors" not in sys.modules:
    colors_stub = types.ModuleType("matplotlib.colors")
    colors_stub.to_rgb = lambda value: (1.0, 1.0, 1.0)
    sys.modules["matplotlib.colors"] = colors_stub

if "matplotlib.pyplot" not in sys.modules:
    pyplot_stub = types.ModuleType("matplotlib.pyplot")

    class _Canvas:
        def __init__(self):
            self.toolbar_visible = False
            self.header_visible = False
            self.footer_visible = False

        def mpl_connect(self, *_):
            return 1

    class _Figure:
        def __init__(self):
            self.canvas = _Canvas()

        def subplots_adjust(self, **_):
            return None

    class _Axis:
        def axis(self, *_):
            return None

        def text(self, *_args, **_kwargs):
            return None

        def imshow(self, *_args, **_kwargs):
            return None

        def remove(self):
            return None

    def _build_axes(rows, cols):
        axes = [[_Axis() for _ in range(cols)] for _ in range(rows)]
        return axes if rows > 1 else axes[0][0]

    def subplots(rows, cols, **_):
        fig = _Figure()
        axes = _build_axes(rows, cols)
        return fig, axes

    pyplot_stub.subplots = subplots
    pyplot_stub.show = lambda *_: None
    sys.modules["matplotlib.pyplot"] = pyplot_stub

if "seaborn_image" not in sys.modules:
    sys.modules["seaborn_image"] = types.ModuleType("seaborn_image")

if "dask" not in sys.modules:
    dask_stub = types.ModuleType("dask")

    def _delayed(func):  # pragma: no cover - lightweight stub
        return func

    dask_stub.delayed = _delayed
    sys.modules["dask"] = dask_stub

if "skimage" not in sys.modules:
    skimage_stub = types.ModuleType("skimage")
    sys.modules["skimage"] = skimage_stub

    segmentation_stub = types.ModuleType("skimage.segmentation")
    segmentation_stub.find_boundaries = lambda *args, **kwargs: None
    sys.modules["skimage.segmentation"] = segmentation_stub

    io_stub = types.ModuleType("skimage.io")
    io_stub.imread = lambda *args, **kwargs: None
    io_stub.imsave = lambda *args, **kwargs: None
    sys.modules["skimage.io"] = io_stub

    exposure_stub = types.ModuleType("skimage.exposure")
    exposure_stub.adjust_gamma = lambda image, *_args, **_kwargs: image
    sys.modules["skimage.exposure"] = exposure_stub

    transform_stub = types.ModuleType("skimage.transform")
    transform_stub.resize = lambda *args, **kwargs: None
    sys.modules["skimage.transform"] = transform_stub

    color_stub = types.ModuleType("skimage.color")
    color_stub.rgb2gray = lambda *args, **kwargs: None
    sys.modules["skimage.color"] = color_stub

    measure_stub = types.ModuleType("skimage.measure")
    measure_stub.label = lambda *args, **kwargs: None
    sys.modules["skimage.measure"] = measure_stub

    draw_stub = types.ModuleType("skimage.draw")
    draw_stub.circle_perimeter = lambda *args, **kwargs: ((), ())
    sys.modules["skimage.draw"] = draw_stub

    skimage_stub.segmentation = segmentation_stub
    skimage_stub.io = io_stub
    skimage_stub.exposure = exposure_stub
    skimage_stub.transform = transform_stub
    skimage_stub.color = color_stub
    skimage_stub.measure = measure_stub
    skimage_stub.draw = draw_stub

if "tifffile" not in sys.modules:
    tifffile_stub = types.ModuleType("tifffile")

    class _FakeTiffFile:  # pragma: no cover - stub for imports
        def __init__(self, *_args, **_kwargs):
            self.pages = [self]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def asarray(self):
            return [[0]]

        @property
        def shape(self):
            return (1, 1)

    tifffile_stub.TiffFile = _FakeTiffFile
    sys.modules["tifffile"] = tifffile_stub

if "ipywidgets" not in sys.modules:
    widgets = types.ModuleType("ipywidgets")

    class _Layout:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Widget:
        def __init__(self, *_, **kwargs):
            self.children = kwargs.get("children", ())
            self.value = kwargs.get("value")
            self.allowed_tags = kwargs.get("allowed_tags", [])
            self.allow_duplicates = kwargs.get("allow_duplicates", False)
            self.allow_new = kwargs.get("allow_new", False)
            self.options = kwargs.get("options", [])
            self.description = kwargs.get("description", "")
            self.tooltip = kwargs.get("tooltip", "")
            self.icon = kwargs.get("icon", "")
            self.button_style = kwargs.get("button_style", "")
            self.layout = kwargs.get("layout")
            self.continuous_update = kwargs.get("continuous_update", True)

        def observe(self, *_, **__):
            return None

        def unobserve(self, *_, **__):
            return None

        def on_click(self, *_, **__):
            return None

        def set_title(self, *_, **__):
            return None

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
        "Button",
        "Checkbox",
        "Combobox",
        "ColorPicker",
        "Dropdown",
        "FloatSlider",
        "HBox",
        "HTML",
        "IntText",
        "Output",
        "Tab",
        "TagsInput",
        "Textarea",
        "Text",
        "VBox",
        "Widget",
    ]:
        setattr(widgets, _name, _Widget)

    sys.modules["ipywidgets"] = widgets

from ipywidgets import Layout

from ueler.image_utils import select_downsample_factor  # type: ignore[import-error]
from viewer.plugin.roi_manager_plugin import ROIManagerPlugin


class RestrictiveTagsWidget:
    """Widget stub that enforces membership in allowed_tags when applying values."""

    def __init__(self):
        self._value = ()
        self._allowed_tags = []
        self.allow_new = True
        self.allow_duplicates = False
        self.layout = None
        self._restrict_to_allowed_tags = True

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new):
        new_tuple = tuple(new or ())
        if self._restrict_to_allowed_tags:
            new_tuple = tuple(tag for tag in new_tuple if tag in self._allowed_tags)
        self._value = new_tuple

    @property
    def allowed_tags(self):
        return list(self._allowed_tags)

    @allowed_tags.setter
    def allowed_tags(self, value):
        self._allowed_tags = list(value or [])

    @property
    def restrict_to_allowed_tags(self) -> bool:
        return self._restrict_to_allowed_tags

    @restrict_to_allowed_tags.setter
    def restrict_to_allowed_tags(self, state: bool) -> None:
        self._restrict_to_allowed_tags = bool(state)

    def observe(self, *_, **__):
        return None


class DummyROIManager:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.table = []
        self.observable = SimpleNamespace(add_observer=lambda *_: None)
        self.last_added = None

    def add_roi(self, record):
        record = record.copy()
        record.setdefault("roi_id", "test-roi")
        self.last_added = record
        return record


class DummyMainViewer:
    def __init__(self):
        base_folder = os.getcwd()
        self.base_folder = base_folder
        storage_path = os.path.join(base_folder, "roi_manager.csv")
        self.roi_manager = DummyROIManager(storage_path)
        self.marker_sets = {}
        self.ui_component = SimpleNamespace(
            image_selector=SimpleNamespace(value="FOV1"),
            marker_set_dropdown=SimpleNamespace(value=None),
            channel_selector=SimpleNamespace(value=()),
        )

    def capture_viewport_bounds(self):
        return {
            "x": 0.0,
            "y": 0.0,
            "width": 10.0,
            "height": 10.0,
            "zoom": 1.0,
            "x_min": 0.0,
            "x_max": 10.0,
            "y_min": 0.0,
            "y_max": 10.0,
        }

    def center_on_roi(self, record):
        self.last_center = record


def make_plugin():
    viewer = DummyMainViewer()
    plugin = object.__new__(ROIManagerPlugin)
    plugin.displayed_name = "ROI manager"
    plugin.SidePlots_id = "roi_manager_output"
    plugin.main_viewer = viewer
    plugin.ui_component = SimpleNamespace()
    plugin._selected_roi_id = None
    plugin._suspend_ui_events = False
    plugin.initialized = False
    plugin.STATUS_COLORS = ROIManagerPlugin.STATUS_COLORS
    plugin.CURRENT_MARKER_VALUE = ROIManagerPlugin.CURRENT_MARKER_VALUE
    plugin._suspend_browser_events = False
    plugin._browser_axis_to_roi = {}
    plugin._browser_click_cid = None
    plugin._browser_figure = None
    plugin._browser_current_page = 1
    plugin._browser_total_pages = 1
    plugin._browser_last_signature = None
    plugin._browser_expression_cache = None
    plugin._browser_expression_error = None
    plugin._browser_tag_buttons = {}
    plugin._browser_expression_selection = (0, 0)
    plugin._use_browser_expression_js = False
    plugin._thumbnail_downsample_cache = {}
    plugin.THUMBNAIL_MAX_EDGE = ROIManagerPlugin.THUMBNAIL_MAX_EDGE
    plugin.width = 6
    plugin.height = 3

    plugin._build_widgets()
    return plugin


class FakeArray:
    def __init__(self, height, width):
        self.shape = (height, width)


class ROIManagerTagsTests(unittest.TestCase):
    def test_tags_widget_allows_new_entries(self):
        plugin = make_plugin()
        tags_widget = plugin.ui_component.tags
        self.assertTrue(getattr(tags_widget, "allow_new", False))

    def test_browser_output_widget_scrolls_within_fixed_height(self):
        plugin = make_plugin()
        layout = plugin.ui_component.browser_output.layout
        self.assertEqual(getattr(layout, "height", None), ROIManagerPlugin.BROWSER_SCROLL_HEIGHT)
        self.assertEqual(getattr(layout, "max_height", None), ROIManagerPlugin.BROWSER_SCROLL_HEIGHT)
        self.assertEqual(getattr(layout, "overflow_y", None), "auto")
        self.assertEqual(getattr(layout, "overflow_x", None), "hidden")

    def test_configure_browser_canvas_applies_layout(self):
        plugin = make_plugin()
        layout = Layout()
        canvas = SimpleNamespace(layout=layout)
        fig = SimpleNamespace(canvas=canvas)

        result = plugin._configure_browser_canvas(fig)

        self.assertTrue(result)
        configured_layout = canvas.layout
        self.assertIsNotNone(configured_layout)
        self.assertEqual(getattr(configured_layout, "height", None), plugin.BROWSER_SCROLL_HEIGHT)
        self.assertEqual(getattr(configured_layout, "width", None), "100%")
        self.assertEqual(getattr(configured_layout, "overflow_y", None), "auto")
        self.assertEqual(getattr(configured_layout, "overflow_x", None), "hidden")

    def test_browser_root_layout_can_shrink(self):
        plugin = make_plugin()
        layout = plugin.ui_component.browser_root.layout
        self.assertEqual(getattr(layout, "min_width", None), "0")
        self.assertEqual(plugin.ui_component.browser_root.children[2], plugin.ui_component.browser_pagination)
        self.assertTrue(getattr(ROIManagerPlugin, "_browser_css_injected", False))

    def test_gallery_layout_respects_width_ratio_and_columns(self):
        plugin = make_plugin()
        columns, rows, fig_width, fig_height = plugin._determine_gallery_layout(2)
        self.assertEqual(columns, plugin.BROWSER_COLUMNS)
        self.assertEqual(rows, 1)
        expected_width = plugin.width * plugin.GALLERY_WIDTH_RATIO
        self.assertAlmostEqual(fig_width, expected_width)
        self.assertAlmostEqual(fig_height, fig_width / columns)

    def test_new_tags_extend_allowed_pool(self):
        plugin = make_plugin()
        change = {"name": "value", "new": ("alpha", "beta"), "old": ()}
        plugin._on_tags_value_change(change)
        self.assertIn("alpha", plugin.ui_component.tags.allowed_tags)
        self.assertIn("beta", plugin.ui_component.tags.allowed_tags)

    def test_string_payload_merges_with_existing_tags(self):
        plugin = make_plugin()
        plugin.ui_component.tags.value = ("alpha",)
        change = {"name": "value", "new": "beta", "old": ("alpha",)}
        plugin._on_tags_value_change(change)
        self.assertEqual(tuple(plugin.ui_component.tags.value), ("alpha", "beta"))
        self.assertIn("beta", plugin.ui_component.tags.allowed_tags)

    def test_string_payload_ignores_whitespace_and_duplicates(self):
        plugin = make_plugin()
        plugin.ui_component.tags.value = ("alpha",)
        change = {"name": "value", "new": "  alpha  ", "old": ("alpha",)}
        plugin._on_tags_value_change(change)
        self.assertEqual(tuple(plugin.ui_component.tags.value), ("alpha",))

    def test_new_tag_persists_when_restrictions_apply(self):
        plugin = make_plugin()
        restrictive_widget = RestrictiveTagsWidget()
        # Replace widget after construction to simulate stricter front-end validation
        plugin.ui_component.tags = restrictive_widget
        change = {"name": "value", "new": ("gamma",), "old": ()}
        plugin._on_tags_value_change(change)
        self.assertEqual(tuple(plugin.ui_component.tags.value), ("gamma",))
        self.assertIn("gamma", plugin.ui_component.tags.allowed_tags)

    def test_combobox_entry_creates_new_tag(self):
        plugin = make_plugin()
        change = {"name": "value", "new": "delta", "old": ""}
        plugin._on_tag_entry_change(change)
        self.assertIn("delta", plugin.ui_component.tags.allowed_tags)
        self.assertIn("delta", plugin.ui_component.tag_entry.options)
        self.assertEqual(tuple(plugin.ui_component.tags.value), ("delta",))

    def test_combobox_ignores_duplicate_entries(self):
        plugin = make_plugin()
        plugin._on_tag_entry_change({"name": "value", "new": "epsilon", "old": ""})
        plugin._on_tag_entry_change({"name": "value", "new": "epsilon", "old": ""})
        self.assertEqual(tuple(plugin.ui_component.tags.value), ("epsilon",))

    def test_capture_forwards_custom_tags(self):
        plugin = make_plugin()
        plugin.refresh_roi_table = lambda *_, **__: None
        plugin.set_status = lambda *_, **__: None
        plugin.ui_component.tags.value = ("novel-tag",)

        result = plugin.main_viewer.capture_viewport_bounds()
        self.assertIsNotNone(result)

        plugin._capture_current_view(None)
        recorded = plugin.main_viewer.roi_manager.last_added
        self.assertIsNotNone(recorded)
        self.assertIn("tags", recorded)
        self.assertIn("novel-tag", recorded["tags"])

    def test_browser_expression_compilation(self):
        plugin = make_plugin()
        predicate = plugin._compile_browser_expression("alpha & !beta")
        self.assertIsNotNone(predicate)
        self.assertTrue(predicate(["alpha"]))
        self.assertFalse(predicate(["alpha", "beta"]))
        self.assertIsNone(plugin._browser_expression_error)

        invalid = plugin._compile_browser_expression("alpha & | beta")
        self.assertIsNone(invalid)
        self.assertIsNotNone(plugin._browser_expression_error)

    def test_expression_restores_tail_selection_for_backend_updates(self):
        plugin = make_plugin()
        widget = plugin.ui_component.browser_expression_input

        # Avoid DataFrame-dependent paths for this focused caret test.
        plugin._refresh_browser_gallery = lambda: None

        expression = "(good&hi)|~bad"
        widget.value = expression

        # Simulate a backend-driven value restore while selection is unknown.
        plugin._browser_expression_selection = None
        plugin._on_browser_expression_change({
            "name": "value",
            "new": expression,
        })

        expected_tail = (len(expression), len(expression))
        self.assertEqual(plugin._browser_expression_selection, expected_tail)

        # Inserting an operator should now append at the tail rather than prefixing.
        plugin._insert_browser_expression_snippet("&")
        updated_value = plugin.ui_component.browser_expression_input.value
        self.assertTrue(updated_value.rstrip().endswith("&"))

    def test_expression_insertion_respects_cached_selection(self):
        plugin = make_plugin()
        widget = plugin.ui_component.browser_expression_input

        plugin._refresh_browser_gallery = lambda: None

        widget.value = "alpha beta"
        plugin._browser_expression_selection = (5, 5)

        plugin._insert_browser_expression_snippet("&")

        self.assertEqual(widget.value, "alpha & beta")
        self.assertEqual(plugin._browser_expression_selection, (7, 7))

    def test_expression_insertion_replaces_highlighted_range(self):
        plugin = make_plugin()
        widget = plugin.ui_component.browser_expression_input

        plugin._refresh_browser_gallery = lambda: None

        widget.value = "alpha beta"
        plugin._browser_expression_selection = (6, 10)

        plugin._insert_browser_expression_snippet("gamma")

        self.assertEqual(widget.value, "alpha gamma")
        self.assertEqual(plugin._browser_expression_selection, (11, 11))

    def test_thumbnail_downsample_uses_roi_viewport_dimensions(self):
        plugin = make_plugin()
        arrays = {"ch1": FakeArray(4096, 8192)}
        record = {"roi_id": "roi-large", "x_min": 0, "x_max": 8192, "y_min": 0, "y_max": 4096}

        factor = plugin._resolve_thumbnail_downsample("FOV-L", arrays, record)

        self.assertEqual(factor, 32)
        self.assertLessEqual(math.ceil(8192 / factor), plugin.THUMBNAIL_MAX_EDGE)

        # Cached factor reused for identical ROI metadata.
        arrays["ch1"] = FakeArray(1024, 1024)
        self.assertEqual(plugin._resolve_thumbnail_downsample("FOV-L", arrays, record), factor)

    def test_thumbnail_downsample_defaults_to_unity_for_small_images(self):
        plugin = make_plugin()
        arrays = {"ch1": FakeArray(128, 200)}
        record = {"roi_id": "roi-small", "x_min": 10, "x_max": 210, "y_min": 5, "y_max": 133}

        factor = plugin._resolve_thumbnail_downsample("FOV-small", arrays, record)
        self.assertEqual(factor, 1)

    def test_thumbnail_downsample_falls_back_to_fov_dimensions(self):
        plugin = make_plugin()
        arrays = {"ch1": FakeArray(512, 512)}
        record = {"roi_id": "roi-null"}  # Missing geometry

        factor = plugin._resolve_thumbnail_downsample("FOV-default", arrays, record)
        self.assertEqual(factor, 2)

    def test_select_downsample_factor_clamps_against_allowed_list(self):
        allowed = [1, 2, 4, 8]

        # Large image chooses the largest permitted factor under the baseline.
        factor = select_downsample_factor(1500, 1200, max_dimension=512, allowed_factors=allowed)
        self.assertEqual(factor, 4)

        # Small images stay at the minimum allowed factor.
        self.assertEqual(select_downsample_factor(50, 50, max_dimension=512, allowed_factors=allowed), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
