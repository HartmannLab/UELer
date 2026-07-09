import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

# Stub heavy optional dependencies when running unit tests so the viewer package can load.
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

if "ipywidgets" not in sys.modules:
    widgets = types.ModuleType("ipywidgets")

    class _Layout:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Widget:
        def __init__(self, *_, **kwargs):
            self.children = kwargs.get("children", tuple())
            self.value = kwargs.get("value")
            self.allowed_tags = kwargs.get("allowed_tags", [])
            self.options = kwargs.get("options", [])
            self.description = kwargs.get("description", "")
            self.tooltip = kwargs.get("tooltip", "")
            self.icon = kwargs.get("icon", "")
            self.button_style = kwargs.get("button_style", "")
            self.layout = kwargs.get("layout")
            self.allow_new = kwargs.get("allow_new", False)

        def observe(self, *_, **__):
            return None

        def unobserve(self, *_, **__):
            return None

        def on_click(self, *_, **__):
            return None

        def reset(self, *_, **__):
            return None

        def set_title(self, *_, **__):
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
        "BoundedIntText",
        "Button",
        "Checkbox",
        "ColorPicker",
        "Dropdown",
        "FloatSlider",
        "HTML",
        "HBox",
        "IntText",
        "Label",
        "Output",
        "Tab",
        "TagsInput",
        "Text",
        "VBox",
        "Widget",
    ]:
        setattr(widgets, _name, _Widget)

    sys.modules["ipywidgets"] = widgets

from ueler.viewer.plugin.mask_painter import (
    BORDER_COLOR_MODE_SAME_AS_FILL,
    COLOR_SET_FILE_SUFFIX,
    DEFAULT_COLOR,
    FILL_OPACITY_DEFAULT_PERCENT,
    MaskPainterDisplay,
    colors_match,
    load_registry,
    normalize_hex_color,
    read_color_set_file,
    save_registry,
    serialize_class_color_controls,
    split_default_classes,
    write_color_set_file,
)


class DummyPicker:
    def __init__(self, value):
        self.value = value


def _make_dummy_viewer(base_folder):
    viewer = types.SimpleNamespace()
    viewer.base_folder = str(base_folder)
    viewer.cell_table = None
    viewer._debug = False
    viewer.ui_component = types.SimpleNamespace(
        image_selector=types.SimpleNamespace(value="FOV_001"),
        mask_color_controls={"cell": types.SimpleNamespace(value="Green")},
    )
    viewer.fov_key = "fov"
    viewer.label_key = "label"
    viewer.mask_key = "cell"
    viewer.predefined_colors = {"Green": "#00FF00", "White": "#FFFFFF", "Red": "#FF0000", "Blue": "#0000FF"}
    viewer.image_display = types.SimpleNamespace(set_mask_colors_current_fov=lambda **_: None)
    viewer.get_active_fov = lambda: viewer.ui_component.image_selector.value
    return viewer


class MaskColorPersistenceTests(unittest.TestCase):
    def test_serialize_applies_defaults(self):
        controls = {
            "A": DummyPicker("#FF0000"),
            "B": DummyPicker(""),
            "C": DummyPicker(None),
        }
        class_order = ["A", "B", "C", "D"]

        hidden_cache = {"C": "#00FF00", "D": "#ABCDEF"}
        result = serialize_class_color_controls(
            controls,
            class_order,
            default_color=DEFAULT_COLOR,
            hidden_cache=hidden_cache,
        )

        self.assertEqual(result["A"], "#FF0000")
        self.assertEqual(result["B"], DEFAULT_COLOR)
        self.assertEqual(result["C"], "#00FF00")
        self.assertEqual(result["D"], "#ABCDEF")

    def test_serialize_custom_default(self):
        controls = {
            "A": DummyPicker(""),
        }
        result = serialize_class_color_controls(controls, ["A"], default_color="#123456")
        self.assertEqual(result["A"], "#123456")

    def test_round_trip_save_load(self):
        tmp_dir = self._tmpdir()
        path = tmp_dir / f"example{COLOR_SET_FILE_SUFFIX}"
        payload = {
            "name": "Example",
            "version": "1.0.0",
            "identifier": "marker",
            "default_color": DEFAULT_COLOR,
            "class_order": ["A", "B"],
            "colors": {"A": "#111111", "B": "#222222"},
            "saved_at": "2025-10-01T00:00:00Z",
        }

        write_color_set_file(path, payload)
        self.assertTrue(path.exists())

        loaded = read_color_set_file(path)
        self.assertEqual(loaded["colors"], payload["colors"])
        self.assertEqual(loaded["class_order"], payload["class_order"])
        self.assertEqual(loaded["identifier"], payload["identifier"])
        self.assertEqual(loaded["default_color"], payload["default_color"])

    def test_registry_persistence(self):
        tmp_dir = self._tmpdir()
        records = {
            "Example": {
                "path": str(tmp_dir / f"example{COLOR_SET_FILE_SUFFIX}"),
                "identifier": "marker",
                "last_modified": "2025-10-01T00:00:00Z",
            }
        }

        save_registry(tmp_dir, records)
        reloaded = load_registry(tmp_dir)
        self.assertEqual(reloaded, records)

    def test_color_normalization_helpers(self):
        self.assertEqual(normalize_hex_color("abc"), "#abc")
        self.assertTrue(colors_match("#ABCDEF", "abcdef"))
        non_default, defaulted = split_default_classes([
            "A",
            "B",
            "C",
            "D",
        ], {"A": "#111111", "B": None, "D": "#222222"}, default_color="#111111")
        self.assertEqual(non_default, ["D"])
        self.assertEqual(defaulted, ["A", "B", "C"])

    def test_load_applies_identifier_and_default_color(self):
        tmp_dir = self._tmpdir()

        viewer = _make_dummy_viewer(tmp_dir)
        display = MaskPainterDisplay(viewer, width=400, height=300)

        widgets = sys.modules["ipywidgets"]
        display.ui_component.identifier_dropdown.options = ["cell_type"]
        display.ui_component.identifier_dropdown.value = "cell_type"
        display.current_identifier = "cell_type"
        display.current_classes = ["A", "B"]
        display.class_color_controls = {
            "A": widgets.ColorPicker(description="A", value="#999999"),
            "B": widgets.ColorPicker(description="B", value="#999999"),
        }
        display.hidden_color_cache = {}
        display.selected_classes = ["A", "B"]
        display.ui_component.sorting_items_tagsinput.allowed_tags = list(display.current_classes)
        display.ui_component.sorting_items_tagsinput.value = tuple(display.current_classes)
        display.ui_component.show_all_checkbox.value = False

        payload = {
            "name": "Example",
            "version": "1.0.0",
            "identifier": "cell_type",
            "default_color": "#101010",
            "class_order": ["A", "B"],
            "colors": {"A": "#121212", "B": "#101010"},
            "saved_at": "2025-10-02T00:00:00Z",
        }

        path = tmp_dir / "example.maskcolors.json"
        write_color_set_file(path, payload)

        display._load_color_set(path)

        self.assertEqual(display.default_color, "#101010")
        self.assertEqual(display.ui_component.identifier_dropdown.value, "cell_type")
        self.assertEqual(display.class_color_controls["A"].value, "#121212")
        self.assertEqual(display.class_color_controls["B"].value, "#101010")
        # With the anywidget class-list, all classes remain visible in the list;
        # the hidden_color_cache / selected_classes pruning is no longer used.
        self.assertIn("A", display.ui_component.class_list_widget.class_order)
        self.assertIn("B", display.ui_component.class_list_widget.class_order)
        self.assertEqual(display.ui_component.class_list_widget.class_colors.get("A"), "#121212")
        self.assertEqual(display.ui_component.class_list_widget.class_colors.get("B"), "#101010")

    def test_save_and_load_round_trip_preserves_reply5_fields(self):
        tmp_dir = self._tmpdir()
        viewer = _make_dummy_viewer(tmp_dir)
        display = MaskPainterDisplay(viewer, width=400, height=300)
        widgets = sys.modules["ipywidgets"]

        display.ui_component.identifier_dropdown.options = ["cell_type"]
        display.ui_component.identifier_dropdown.value = "cell_type"
        display.current_identifier = "cell_type"
        display.current_classes = ["B", "C", "A"]
        display.class_color_controls = {
            "A": widgets.ColorPicker(description="A", value="#FF0000"),
            "B": widgets.ColorPicker(description="B", value=DEFAULT_COLOR),
            "C": widgets.ColorPicker(description="C", value="#00FF00"),
        }
        display.class_visible_controls = {
            "A": widgets.Checkbox(value=True),
            "B": widgets.Checkbox(value=False),
            "C": widgets.Checkbox(value=True),
        }
        display.class_mode_controls = {
            "A": widgets.Checkbox(value=True),
            "B": widgets.Checkbox(value=False),
            "C": widgets.Checkbox(value=False),
        }
        display.class_opacity_controls = {
            "A": widgets.BoundedIntText(value=75, min=0, max=100),
            "B": widgets.BoundedIntText(value=40, min=0, max=100),
            "C": widgets.BoundedIntText(value=40, min=0, max=100),
        }
        display._active_classes = ["C", "A"]
        display._linked_fill_classes = {"B", "C"}
        display._linked_opacity_classes = {"B"}
        display.ui_component.only_specified_checkbox.value = True
        display.ui_component.global_fill_checkbox.value = False
        display.ui_component.global_fill_opacity_input.value = 40
        display.ui_component.show_fill_borders_checkbox.value = True
        display.ui_component.border_color_mode_dropdown.value = BORDER_COLOR_MODE_SAME_AS_FILL
        display.ui_component.set_name_input.value = "Example"
        display._push_to_widget()

        display.save_current_color_set(None)

        saved_path = Path(display.registry_records["Example"]["path"])
        payload = read_color_set_file(saved_path)
        self.assertEqual(payload["active_classes"], ["C", "A"])
        self.assertEqual(payload["only_specified"], True)
        self.assertEqual(payload["global_fill"], False)
        self.assertEqual(payload["linked_fill_classes"], ["B", "C"])
        self.assertEqual(payload["linked_opacity_classes"], ["B"])
        self.assertEqual(payload["border_color_mode"], BORDER_COLOR_MODE_SAME_AS_FILL)

        display.class_color_controls["A"].value = DEFAULT_COLOR
        display.class_visible_controls["B"].value = True
        display.class_mode_controls["A"].value = False
        display.class_opacity_controls["A"].value = 5
        display._active_classes = ["A", "B", "C"]
        display._linked_fill_classes = set()
        display._linked_opacity_classes = set()
        display.ui_component.only_specified_checkbox.value = False
        display.ui_component.global_fill_checkbox.value = True
        display.ui_component.global_fill_opacity_input.value = 10
        display.ui_component.show_fill_borders_checkbox.value = False
        display.ui_component.border_color_mode_dropdown.value = "mask_type_color"

        display._load_color_set(saved_path)

        self.assertEqual(display.class_color_controls["A"].value, "#FF0000")
        self.assertEqual(display.class_visible_controls["B"].value, False)
        self.assertEqual(display.class_mode_controls["A"].value, True)
        self.assertEqual(display.class_opacity_controls["A"].value, 75)
        self.assertEqual(display.ui_component.global_fill_checkbox.value, False)
        self.assertEqual(display.ui_component.global_fill_opacity_input.value, 40)
        self.assertEqual(display.ui_component.show_fill_borders_checkbox.value, True)
        self.assertEqual(display.ui_component.border_color_mode_dropdown.value, BORDER_COLOR_MODE_SAME_AS_FILL)
        self.assertEqual(display.ui_component.only_specified_checkbox.value, True)
        self.assertEqual(list(display.ui_component.class_list_widget.class_order), ["C", "A"])
        self.assertEqual(list(display.ui_component.class_list_widget.available_classes), ["B"])
        self.assertEqual(display._linked_fill_classes, {"B", "C"})
        self.assertEqual(display._linked_opacity_classes, {"B"})

    def test_load_old_palette_defaults_new_reply5_fields(self):
        tmp_dir = self._tmpdir()
        viewer = _make_dummy_viewer(tmp_dir)
        display = MaskPainterDisplay(viewer, width=400, height=300)
        widgets = sys.modules["ipywidgets"]

        display.ui_component.identifier_dropdown.options = ["cell_type"]
        display.ui_component.identifier_dropdown.value = "cell_type"
        display.current_identifier = "cell_type"
        display.current_classes = ["A", "B"]
        display.class_color_controls = {
            "A": widgets.ColorPicker(description="A", value="#999999"),
            "B": widgets.ColorPicker(description="B", value="#999999"),
        }
        display.class_visible_controls = {
            "A": widgets.Checkbox(value=True),
            "B": widgets.Checkbox(value=True),
        }
        display.class_mode_controls = {
            "A": widgets.Checkbox(value=False),
            "B": widgets.Checkbox(value=False),
        }
        display.class_opacity_controls = {
            "A": widgets.BoundedIntText(value=FILL_OPACITY_DEFAULT_PERCENT, min=0, max=100),
            "B": widgets.BoundedIntText(value=FILL_OPACITY_DEFAULT_PERCENT, min=0, max=100),
        }
        display._active_classes = ["A", "B"]

        path = tmp_dir / "legacy.maskcolors.json"
        write_color_set_file(path, {
            "name": "Legacy",
            "version": "1.0.0",
            "identifier": "cell_type",
            "default_color": "#101010",
            "class_order": ["A", "B"],
            "colors": {"A": "#121212", "B": "#101010"},
            "saved_at": "2025-10-02T00:00:00Z",
        })

        display._load_color_set(path)

        self.assertEqual(display.ui_component.global_fill_checkbox.value, False)
        self.assertEqual(display.ui_component.global_fill_opacity_input.value, FILL_OPACITY_DEFAULT_PERCENT)
        self.assertEqual(display.ui_component.only_specified_checkbox.value, False)
        self.assertEqual(display._linked_fill_classes, {"A", "B"})
        self.assertEqual(display._linked_opacity_classes, {"A", "B"})

    def _tmpdir(self) -> Path:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
