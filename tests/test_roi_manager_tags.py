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
    plugin._browser_expression_selection = None
    plugin._browser_expression_focused = False
    plugin._browser_expression_widget_bound = False
    plugin._browser_expression_skip_reset = False

    plugin._build_widgets()
    return plugin


class ROIManagerTagsTests(unittest.TestCase):
    def test_tags_widget_allows_new_entries(self):
        plugin = make_plugin()
        tags_widget = plugin.ui_component.tags
        self.assertTrue(getattr(tags_widget, "allow_new", False))

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

    def test_expression_cursor_preserves_last_focused_selection(self):
        plugin = make_plugin()
        widget = plugin.ui_component.browser_expression_input

        # Simulate the user placing the caret at index 4 while the field is focused.
        plugin._on_browser_expression_msg(widget, {
            "event": "selection-change",
            "start": 4,
            "end": 4,
            "focused": True,
        }, None)
        self.assertEqual(plugin._browser_expression_selection, (4, 4))

        # Blur events should not reset the saved caret location back to zero.
        plugin._on_browser_expression_msg(widget, {
            "event": "selection-change",
            "start": 0,
            "end": 0,
            "focused": False,
        }, None)
        self.assertEqual(plugin._browser_expression_selection, (4, 4))

    def test_expression_restores_tail_selection_for_backend_updates(self):
        plugin = make_plugin()
        widget = plugin.ui_component.browser_expression_input

        # Avoid DataFrame-dependent paths for this focused caret test.
        plugin._refresh_browser_gallery = lambda: None

        expression = "(good&hi)|~bad"
        widget.value = expression

        # Simulate a backend-driven value restore while the widget is unfocused.
        plugin._browser_expression_focused = False
        plugin._on_browser_expression_change({
            "name": "value",
            "new": expression,
        })

        expected_tail = (len(expression), len(expression))
        self.assertEqual(plugin._browser_expression_selection, expected_tail)

        # Inserting an operator should now append at the tail rather than prefixing.
        plugin._insert_browser_expression_snippet("&")
        updated_value = plugin.ui_component.browser_expression_input.value
        self.assertTrue(updated_value.endswith("&"))

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
