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
        "Button",
        "Checkbox",
        "ColorPicker",
        "Dropdown",
        "FloatSlider",
        "HBox",
        "IntText",
        "Output",
        "Tab",
        "TagsInput",
        "Text",
        "VBox",
        "Widget",
    ]:
        setattr(widgets, _name, _Widget)

    sys.modules["ipywidgets"] = widgets

from viewer.plugin.mask_painter import (
    COLOR_SET_FILE_SUFFIX,
    DEFAULT_COLOR,
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

        class DummyViewer:
            def __init__(self, base_folder):
                self.base_folder = str(base_folder)
                self.cell_table = None
                self._debug = False
                dummy_widget = sys.modules["ipywidgets"].Widget()
                self.ui_component = types.SimpleNamespace(image_selector=dummy_widget)
                self.fov_key = "fov"
                self.label_key = "label"
                self.mask_key = "mask"
                self.image_display = types.SimpleNamespace(
                    set_mask_colors_current_fov=lambda **_: None
                )

        viewer = DummyViewer(tmp_dir)
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
        self.assertEqual(display.selected_classes, ["A"])
        self.assertIn("B", display.hidden_color_cache)
        self.assertEqual(display.hidden_color_cache["B"], "#101010")

    def _tmpdir(self) -> Path:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
