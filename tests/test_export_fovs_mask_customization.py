"""Tests for Feature 3 of issue #92: export config template save/load/delete."""

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import unittest.mock as mock

# ---------------------------------------------------------------------------
# Minimal stubs for heavy optional dependencies
# ---------------------------------------------------------------------------
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

for _mod in ("skimage", "skimage.exposure", "skimage.transform", "skimage.color",
             "skimage.measure", "skimage.segmentation", "skimage.io",
             "dask", "dask_image", "dask_image.imread", "tifffile",
             "seaborn_image"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

import numpy as np

if "skimage.exposure" in sys.modules:
    sys.modules["skimage.exposure"].adjust_gamma = lambda img, *a, **k: img  # type: ignore[attr-defined]
if "skimage.transform" in sys.modules:
    sys.modules["skimage.transform"].resize = lambda a, s, **k: np.resize(a, s)  # type: ignore[attr-defined]
if "skimage.color" in sys.modules:
    sys.modules["skimage.color"].gray2rgb = lambda a: np.stack([a] * 3, axis=-1)  # type: ignore[attr-defined]
if "skimage.measure" in sys.modules:
    sys.modules["skimage.measure"].label = lambda a: a  # type: ignore[attr-defined]
if "skimage.segmentation" in sys.modules:
    sys.modules["skimage.segmentation"].find_boundaries = lambda a, **k: np.zeros_like(a, dtype=bool)  # type: ignore[attr-defined]

for _mpl in ("matplotlib", "matplotlib.pyplot", "matplotlib.colors", "matplotlib.font_manager",
             "mpl_toolkits", "mpl_toolkits.axes_grid1", "mpl_toolkits.axes_grid1.anchored_artists"):
    if _mpl not in sys.modules:
        sys.modules[_mpl] = types.ModuleType(_mpl)

if not hasattr(sys.modules["matplotlib.colors"], "to_rgb"):
    sys.modules["matplotlib.colors"].to_rgb = lambda v: (1.0, 0.0, 0.0)  # type: ignore[attr-defined]

# ipywidgets stubs
ipywidgets_module = sys.modules.get("ipywidgets")
if ipywidgets_module is None:
    ipywidgets_module = types.ModuleType("ipywidgets")
    sys.modules["ipywidgets"] = ipywidgets_module


if not hasattr(ipywidgets_module, "Widget"):
    class _BaseWidget:
        def __init__(self, *args, **kwargs):
            self.value = kwargs.get("value")
            self.options = kwargs.get("options", ())
            self.description = kwargs.get("description")
            self.layout = kwargs.get("layout")
            self.style = kwargs.get("style")
            self.disabled = kwargs.get("disabled", False)
            self.children = kwargs.get("children", ())
            self.button_style = kwargs.get("button_style", "")
            self.icon = kwargs.get("icon")
            self._observers = []

        def observe(self, callback, names=None):
            self._observers.append(callback)

        def set_title(self, *_args):
            return

        def on_click(self, callback):
            pass

    ipywidgets_module.Widget = _BaseWidget


def _ensure_widget(name):
    if not hasattr(ipywidgets_module, name):
        setattr(ipywidgets_module, name, type(name, (ipywidgets_module.Widget,), {}))


for _wname in ("Accordion", "Button", "Checkbox", "Dropdown", "FloatSlider", "HBox",
               "HTML", "IntProgress", "IntSlider", "IntText", "Layout", "Output",
               "SelectMultiple", "Tab", "Text", "ToggleButtons", "VBox"):
    _ensure_widget(_wname)

if not hasattr(ipywidgets_module, "Layout"):
    class _Layout:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    ipywidgets_module.Layout = _Layout

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
from ueler.viewer.plugin.export_fovs import (  # noqa: E402
    BatchExportPlugin,
    EXPORT_CONFIG_FILE_SUFFIX,
    EXPORT_CONFIG_REGISTRY_FILENAME,
    EXPORT_CONFIG_VERSION,
)


class _StubWidget:
    def __init__(self, value=None):
        self._value = value
        self.disabled = False
        self._observers = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value
        for cb in tuple(self._observers):
            cb({"new": new_value, "owner": self})

    def observe(self, callback, names=None):
        self._observers.append(callback)

    def on_click(self, callback):
        pass


class _StubDropdown(_StubWidget):
    def __init__(self, value=None, options=()):
        super().__init__(value)
        self.options = list(options)


class _StubHTML:
    def __init__(self):
        self.value = ""


class _ViewerStub:
    def __init__(self, base_path: Path) -> None:
        self.base_folder = str(base_path)
        self.marker_sets = {"default": {"selected_channels": (), "channel_settings": {}}}
        self.available_fovs = []
        self.cell_table = None
        self.roi_manager = None
        self.masks_available = True
        self.annotations_available = False
        self.annotation_display_enabled = False
        self.active_annotation_name = None
        self.mask_outline_thickness = 1
        self.current_downsample_factor = 1
        self.predefined_colors = {}
        self.annotation_overlay_alpha = 1.0
        self.annotation_overlay_mode = "combined"
        self.initialized = True
        self._debug = False
        self.pixel_size_nm = 390.0
        self._map_mode_active = False
        setattr(self, "SidePlots", SimpleNamespace())
        self.ui_component = SimpleNamespace(
            mask_display_controls={"MASK": SimpleNamespace(value=True)},
            mask_color_controls={"MASK": SimpleNamespace(value="#ff0000")},
            image_selector=SimpleNamespace(value=None),
        )

    def capture_overlay_snapshot(self, **kwargs):
        from ueler.rendering import OverlaySnapshot
        return OverlaySnapshot(
            include_annotations=kwargs.get("include_annotations", False),
            include_masks=kwargs.get("include_masks", True),
            annotation=None,
            masks=(),
        )

    def get_pixel_size_nm(self):
        return self.pixel_size_nm


class _TestPlugin(BatchExportPlugin):
    """Minimal plugin subclass for testing config templates."""

    def setup_widget_observers(self):
        return

    def _build_widgets(self):
        self.ui_component.include_masks = _StubWidget(True)
        self.ui_component.include_annotations = _StubWidget(False)
        self.ui_component.mask_outline_thickness = _StubWidget(self._mask_outline_thickness)
        self.ui_component.overlay_hint = _StubHTML()
        self.ui_component.masks_only = _StubWidget(False)
        self.ui_component.mask_palette_enabled = _StubWidget(False)
        self.ui_component.mask_palette_dropdown = _StubDropdown(None, [("Current settings", None)])
        self.ui_component.output_path = _StubWidget("/tmp/exports")
        self.ui_component.file_format_dropdown = _StubDropdown("png", [("PNG", "png")])
        self.ui_component.downsample_input = _StubWidget(1)
        self.ui_component.dpi_input = _StubWidget(300)
        self.ui_component.include_scale_bar = _StubWidget(False)
        self.ui_component.scale_bar_ratio = _StubWidget(10.0)
        self.ui_component.marker_set_dropdown = _StubDropdown("default", [("default", "default")])
        self.ui_component.config_name_input = _StubWidget("")
        self.ui_component.config_saved_dropdown = _StubDropdown(None, [("No saved configs", None)])
        self.ui_component.config_save_button = _StubWidget(None)
        self.ui_component.config_load_button = _StubWidget(None)
        self.ui_component.config_delete_button = _StubWidget(None)
        self.ui_component.config_status = _StubHTML()

    def _build_layout(self):
        return

    def _connect_events(self):
        return

    def refresh_marker_options(self):
        return

    def refresh_fov_options(self):
        return

    def refresh_cell_options(self):
        return

    def refresh_roi_options(self):
        return

    def _refresh_mode_availability(self):
        return


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestExportConfigTemplates(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base_path = Path(self._tmp.name)

    def _make_plugin(self) -> _TestPlugin:
        viewer = _ViewerStub(self.base_path)
        plugin = _TestPlugin(viewer, width=320, height=480)
        self.addCleanup(plugin._executor.shutdown, False)
        return plugin

    def test_collect_config_includes_all_expected_fields(self):
        plugin = self._make_plugin()
        payload = plugin._collect_export_config("my config")
        required = {
            "name", "version", "saved_at", "file_format", "downsample", "dpi",
            "include_scale_bar", "scale_bar_ratio", "include_annotations",
            "include_masks", "masks_only", "mask_outline_thickness",
            "mask_palette_enabled", "mask_palette_name", "marker_set",
            "output_path",
        }
        self.assertTrue(required.issubset(payload.keys()), f"Missing keys: {required - payload.keys()}")
        self.assertEqual(payload["name"], "my config")
        self.assertEqual(payload["version"], EXPORT_CONFIG_VERSION)

    def test_save_config_writes_json_file(self):
        plugin = self._make_plugin()
        plugin.ui_component.config_name_input.value = "test save"
        plugin._save_export_config()
        config_dir = self.base_path / ".UELer" / "export_configs"
        json_files = list(config_dir.glob(f"*{EXPORT_CONFIG_FILE_SUFFIX}"))
        self.assertEqual(len(json_files), 1, "Expected exactly one config JSON file")
        data = json.loads(json_files[0].read_text())
        self.assertEqual(data["name"], "test save")
        self.assertEqual(data["version"], EXPORT_CONFIG_VERSION)

    def test_save_config_writes_registry_entry(self):
        plugin = self._make_plugin()
        plugin.ui_component.config_name_input.value = "my export"
        plugin._save_export_config()
        registry_path = self.base_path / ".UELer" / "export_configs" / EXPORT_CONFIG_REGISTRY_FILENAME
        self.assertTrue(registry_path.exists())
        registry = json.loads(registry_path.read_text())
        self.assertIn("my export", registry)

    def test_save_config_requires_name(self):
        plugin = self._make_plugin()
        plugin.ui_component.config_name_input.value = "   "
        plugin._save_export_config()
        config_dir = self.base_path / ".UELer" / "export_configs"
        json_files = list(config_dir.glob(f"*{EXPORT_CONFIG_FILE_SUFFIX}"))
        self.assertEqual(len(json_files), 0, "No file should be written without a name")
        self.assertIn("color:red", plugin.ui_component.config_status.value)

    def test_load_config_restores_widget_values(self):
        plugin = self._make_plugin()
        plugin.ui_component.config_name_input.value = "load test"
        plugin.ui_component.dpi_input.value = 150
        plugin.ui_component.include_scale_bar.value = True
        plugin._save_export_config()

        plugin.ui_component.dpi_input.value = 300
        plugin.ui_component.include_scale_bar.value = False
        saved_name = sorted(plugin._export_config_registry.keys())[0]
        plugin.ui_component.config_saved_dropdown.options = [(saved_name, saved_name)]
        plugin.ui_component.config_saved_dropdown.value = saved_name
        plugin._load_export_config()

        self.assertEqual(plugin.ui_component.dpi_input.value, 150)
        self.assertTrue(plugin.ui_component.include_scale_bar.value)
        self.assertIn("color:green", plugin.ui_component.config_status.value)

    def test_load_config_requires_selection(self):
        plugin = self._make_plugin()
        plugin.ui_component.config_saved_dropdown.value = None
        plugin._load_export_config()
        self.assertIn("color:orange", plugin.ui_component.config_status.value)

    def test_load_config_skips_unknown_palette_name(self):
        plugin = self._make_plugin()
        plugin.ui_component.config_name_input.value = "palette test"
        plugin.ui_component.mask_palette_enabled.value = True
        plugin.ui_component.mask_palette_dropdown.options = [("Current settings", None), ("Saved A", "Saved A")]
        plugin.ui_component.mask_palette_dropdown.value = "Saved A"
        plugin._save_export_config()

        plugin.ui_component.mask_palette_dropdown.options = [("Current settings", None)]
        plugin.ui_component.mask_palette_dropdown.value = None

        saved_name = sorted(plugin._export_config_registry.keys())[0]
        plugin.ui_component.config_saved_dropdown.options = [(saved_name, saved_name)]
        plugin.ui_component.config_saved_dropdown.value = saved_name
        plugin._load_export_config()

        self.assertIsNone(plugin.ui_component.mask_palette_dropdown.value,
                          "Unknown palette name must not be applied")

    def test_apply_config_unknown_marker_set_ignored(self):
        plugin = self._make_plugin()
        payload = plugin._collect_export_config("dummy")
        payload["marker_set"] = "nonexistent_marker_set"
        original_value = plugin.ui_component.marker_set_dropdown.value
        plugin._apply_export_config(payload)
        self.assertEqual(plugin.ui_component.marker_set_dropdown.value, original_value)

    def test_delete_config_removes_file_and_registry(self):
        plugin = self._make_plugin()
        plugin.ui_component.config_name_input.value = "to delete"
        plugin._save_export_config()
        config_dir = self.base_path / ".UELer" / "export_configs"
        self.assertEqual(len(list(config_dir.glob(f"*{EXPORT_CONFIG_FILE_SUFFIX}"))), 1)

        saved_name = sorted(plugin._export_config_registry.keys())[0]
        plugin.ui_component.config_saved_dropdown.options = [(saved_name, saved_name)]
        plugin.ui_component.config_saved_dropdown.value = saved_name
        plugin._delete_export_config()

        self.assertEqual(len(list(config_dir.glob(f"*{EXPORT_CONFIG_FILE_SUFFIX}"))), 0)
        self.assertNotIn(saved_name, plugin._export_config_registry)
        self.assertIn("color:green", plugin.ui_component.config_status.value)

    def test_refresh_config_dropdown_reads_registry(self):
        plugin = self._make_plugin()
        plugin.ui_component.config_name_input.value = "dropdown refresh test"
        plugin._save_export_config()

        plugin._export_config_registry = {}
        plugin._refresh_config_dropdown()

        options = plugin.ui_component.config_saved_dropdown.options
        labels = [lbl for lbl, _ in options] if options and isinstance(options[0], tuple) else list(options)
        self.assertIn("dropdown refresh test", labels)

    def test_save_multiple_configs_all_in_registry(self):
        plugin = self._make_plugin()
        for name in ("alpha", "beta", "gamma"):
            plugin.ui_component.config_name_input.value = name
            plugin._save_export_config()
        self.assertEqual(len(plugin._export_config_registry), 3)

    def test_delete_nonexistent_does_not_crash(self):
        plugin = self._make_plugin()
        plugin.ui_component.config_saved_dropdown.options = []
        plugin.ui_component.config_saved_dropdown.value = None
        plugin._delete_export_config()

    def test_load_config_restores_output_path(self):
        plugin = self._make_plugin()
        plugin.ui_component.output_path.value = "/tmp/my_exports"
        plugin.ui_component.config_name_input.value = "output path test"
        plugin._save_export_config()

        plugin.ui_component.output_path.value = "/tmp/other"
        saved_name = sorted(plugin._export_config_registry.keys())[0]
        plugin.ui_component.config_saved_dropdown.options = [(saved_name, saved_name)]
        plugin.ui_component.config_saved_dropdown.value = saved_name
        plugin._load_export_config()

        self.assertEqual(plugin.ui_component.output_path.value, "/tmp/my_exports")


if __name__ == "__main__":
    unittest.main()
