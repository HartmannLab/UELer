import sys
import types
import unittest
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace

import numpy as np
import unittest.mock as mock

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

if "skimage" not in sys.modules:
    skimage_module = types.ModuleType("skimage")
    sys.modules["skimage"] = skimage_module
else:
    skimage_module = sys.modules["skimage"]

if "skimage.exposure" not in sys.modules:
    exposure_module = types.ModuleType("skimage.exposure")
    exposure_module.adjust_gamma = lambda image, *_args, **_kwargs: image  # type: ignore[attr-defined]
    sys.modules["skimage.exposure"] = exposure_module
    setattr(skimage_module, "exposure", exposure_module)
else:
    setattr(skimage_module, "exposure", sys.modules["skimage.exposure"])

if "skimage.transform" not in sys.modules:
    transform_module = types.ModuleType("skimage.transform")
    transform_module.resize = lambda array, shape, **_kwargs: np.resize(array, shape)  # type: ignore[attr-defined]
    sys.modules["skimage.transform"] = transform_module
    setattr(skimage_module, "transform", transform_module)
else:
    setattr(skimage_module, "transform", sys.modules["skimage.transform"])

if "skimage.color" not in sys.modules:
    color_module = types.ModuleType("skimage.color")
    color_module.gray2rgb = lambda array: np.stack([array] * 3, axis=-1)  # type: ignore[attr-defined]
    sys.modules["skimage.color"] = color_module
    setattr(skimage_module, "color", color_module)
else:
    setattr(skimage_module, "color", sys.modules["skimage.color"])

if "skimage.measure" not in sys.modules:
    measure_module = types.ModuleType("skimage.measure")
    measure_module.label = lambda array: array  # type: ignore[attr-defined]
    sys.modules["skimage.measure"] = measure_module
    setattr(skimage_module, "measure", measure_module)
else:
    setattr(skimage_module, "measure", sys.modules["skimage.measure"])

if "skimage.segmentation" not in sys.modules:
    segmentation_module = types.ModuleType("skimage.segmentation")

    def _find_boundaries_stub(array, mode="inner"):
        return np.zeros_like(array, dtype=bool)

    segmentation_module.find_boundaries = _find_boundaries_stub  # type: ignore[attr-defined]
    sys.modules["skimage.segmentation"] = segmentation_module
    setattr(skimage_module, "segmentation", segmentation_module)
else:
    setattr(skimage_module, "segmentation", sys.modules["skimage.segmentation"])

if "skimage.io" not in sys.modules:
    io_module = types.ModuleType("skimage.io")

    def _imsave_stub(path, array, **_kwargs):
        Path(path).write_bytes(b"stub")

    def _imread_stub(_path):
        return np.ones((4, 4), dtype=np.float32)

    io_module.imsave = _imsave_stub  # type: ignore[attr-defined]
    io_module.imread = _imread_stub  # type: ignore[attr-defined]
    sys.modules["skimage.io"] = io_module
    setattr(skimage_module, "io", io_module)
else:
    setattr(skimage_module, "io", sys.modules["skimage.io"])

if "dask" not in sys.modules:
    dask_module = types.ModuleType("dask")

    def _delayed_stub(func):
        return func

    dask_module.delayed = _delayed_stub  # type: ignore[attr-defined]
    sys.modules["dask"] = dask_module

if "dask_image" not in sys.modules:
    sys.modules["dask_image"] = types.ModuleType("dask_image")

if "dask_image.imread" not in sys.modules:
    imread_module = types.ModuleType("dask_image.imread")

    def _imread_stub(path):
        return np.ones((4, 4), dtype=np.float32)

    imread_module.imread = _imread_stub  # type: ignore[attr-defined]
    sys.modules["dask_image.imread"] = imread_module

if "tifffile" not in sys.modules:
    tifffile_module = types.ModuleType("tifffile")

    class _FakePage:
        def __init__(self, array):
            self._array = array

        def asarray(self):
            return self._array

        @property
        def shape(self):
            return self._array.shape

    class _FakeTiff:
        def __init__(self, path):  # minimal stub returning unity array
            self.pages = [_FakePage(np.ones((4, 4), dtype=np.float32))]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    tifffile_module.TiffFile = _FakeTiff  # type: ignore[attr-defined]
    sys.modules["tifffile"] = tifffile_module

if "seaborn_image" not in sys.modules:
    sys.modules["seaborn_image"] = types.ModuleType("seaborn_image")

if "matplotlib" not in sys.modules:
    matplotlib_module = types.ModuleType("matplotlib")
    sys.modules["matplotlib"] = matplotlib_module

if "matplotlib.pyplot" not in sys.modules:
    pyplot_module = types.ModuleType("matplotlib.pyplot")

    def _subplots_stub(*_args, **_kwargs):
        fig = types.SimpleNamespace(canvas=types.SimpleNamespace(draw_idle=lambda: None))
        ax = types.SimpleNamespace(
            set_xlim=lambda *_args, **_kwargs: None,
            set_ylim=lambda *_args, **_kwargs: None,
            hist=lambda *_args, **_kwargs: None,
            add_artist=lambda *_args, **_kwargs: None,
        )
        return fig, ax

    pyplot_module.subplots = _subplots_stub  # type: ignore[attr-defined]
    pyplot_module.show = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    sys.modules["matplotlib.pyplot"] = pyplot_module

if "matplotlib.colors" not in sys.modules:
    colors_module = types.ModuleType("matplotlib.colors")
    colors_module.to_rgb = lambda value: (1.0, 0.0, 0.0) if value == "Red" else (0.0, 0.0, 1.0)  # type: ignore[attr-defined]
    sys.modules["matplotlib.colors"] = colors_module

if "matplotlib.font_manager" not in sys.modules:
    fm_module = types.ModuleType("matplotlib.font_manager")
    fm_module.FontProperties = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["matplotlib.font_manager"] = fm_module

if "mpl_toolkits" not in sys.modules:
    sys.modules["mpl_toolkits"] = types.ModuleType("mpl_toolkits")

if "mpl_toolkits.axes_grid1" not in sys.modules:
    sys.modules["mpl_toolkits.axes_grid1"] = types.ModuleType("mpl_toolkits.axes_grid1")

if "mpl_toolkits.axes_grid1.anchored_artists" not in sys.modules:
    anchored_module = types.ModuleType("mpl_toolkits.axes_grid1.anchored_artists")

    class _AnchoredSizeBar:
        def __init__(self, *args, **kwargs):  # minimal stub, no behaviour needed for tests
            return None

    anchored_module.AnchoredSizeBar = _AnchoredSizeBar  # type: ignore[attr-defined]
    sys.modules["mpl_toolkits.axes_grid1.anchored_artists"] = anchored_module

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

        def set_title(self, *_args):  # pragma: no cover - tab helper
            return

    ipywidgets_module.Widget = _BaseWidget

if not hasattr(ipywidgets_module, "Layout"):
    class Layout:  # pragma: no cover - minimal stub
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    ipywidgets_module.Layout = Layout

if not hasattr(ipywidgets_module, "Output"):
    class Output(ipywidgets_module.Widget):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def clear_output(self):
            return

    ipywidgets_module.Output = Output

if not hasattr(ipywidgets_module, "Tab"):
    class Tab(ipywidgets_module.Widget):
        pass

    ipywidgets_module.Tab = Tab

if not hasattr(ipywidgets_module, "HTML"):
    class HTML(ipywidgets_module.Widget):
        pass

    ipywidgets_module.HTML = HTML


def _ensure_widget(name):  # pragma: no cover - helper for stubs
    if not hasattr(ipywidgets_module, name):
        setattr(ipywidgets_module, name, type(name, (ipywidgets_module.Widget,), {}))


for widget_name in (
    "Accordion",
    "Button",
    "Checkbox",
    "Dropdown",
    "FloatSlider",
    "HBox",
    "IntProgress",
    "IntSlider",
    "IntText",
    "SelectMultiple",
    "Text",
    "VBox",
):
    _ensure_widget(widget_name)

from ueler.viewer.main_viewer import ImageMaskViewer
from ueler.viewer.plugin.export_fovs import BatchExportPlugin
from ueler.rendering import MaskPainterSnapshot, OverlaySnapshot


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
        for callback in tuple(self._observers):
            callback({"new": new_value, "owner": self})

    def observe(self, callback, names=None):
        self._observers.append(callback)

    def on_click(self, callback):  # button stub
        pass


class _StubDropdown(_StubWidget):
    """Stub dropdown widget with an options list."""

    def __init__(self, value=None, options=()):
        super().__init__(value)
        self.options = list(options)


class _StubHTML:
    def __init__(self):
        self.value = ""


class _StubOutputWidget:
    def __init__(self):
        self.cleared = False

    def clear_output(self):
        self.cleared = True

    def __enter__(self):  # pragma: no cover - supports context usage
        return self

    def __exit__(self, *_args):
        return False


class _BatchExportViewerStub:
    def __init__(self, base_path: Path, *, mask_outline_thickness: int = 3) -> None:
        self.base_folder = str(base_path)
        self.marker_sets = {
            "default": {
                "selected_channels": (),
                "channel_settings": {},
            }
        }
        self.available_fovs = []
        self.cell_table = None
        self.roi_manager = None
        self.masks_available = True
        self.annotations_available = False
        self.annotation_display_enabled = False
        self.active_annotation_name = None
        self.mask_outline_thickness = mask_outline_thickness
        self.current_downsample_factor = 1
        self.predefined_colors = {"Red": "#ff0000"}
        self.annotation_overlay_alpha = 1.0
        self.annotation_overlay_mode = "combined"
        self.side_plots = SimpleNamespace()
        setattr(self, "SidePlots", self.side_plots)
        self.initialized = True
        self._debug = False
        self.pixel_size_nm = 390.0
        self._map_mode_active = False
        self.ui_component = SimpleNamespace(
            mask_display_controls={"MASK": SimpleNamespace(value=True)},
            mask_color_controls={"MASK": SimpleNamespace(value="Red")},
            image_selector=SimpleNamespace(value=None),
        )
        self.apply_overlay_snapshot_to_map_array = lambda image, **kwargs: image

    def get_active_fov(self):
        if self._map_mode_active:
            return None
        return getattr(self.ui_component.image_selector, "value", None) or None

    def capture_overlay_snapshot(self, *, include_masks: bool, include_annotations: bool):  # pragma: no cover - unused in tests
        raise NotImplementedError

    def build_overlay_settings_from_snapshot(self, *args, **kwargs):  # pragma: no cover - unused in tests
        raise NotImplementedError

    def get_pixel_size_nm(self):  # pragma: no cover - used indirectly in tests
        return self.pixel_size_nm


class ExportFOVsBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.base_path = Path(self._tmp_dir.name)
        for name in ("FOV_A", "FOV_B"):
            (self.base_path / name).mkdir(parents=True, exist_ok=True)

    def _make_viewer(self) -> ImageMaskViewer:
        viewer = ImageMaskViewer.__new__(ImageMaskViewer)
        viewer.base_folder = str(self.base_path)
        viewer.available_fovs = ["FOV_A", "FOV_B"]
        viewer.marker_sets = {
            "test": {
                "selected_channels": ("DNA",),
                "channel_settings": {
                    "DNA": {
                        "color": "Red",
                        "contrast_min": 0.0,
                        "contrast_max": 2.0,
                    }
                },
            }
        }
        viewer.predefined_colors = {"Red": "#ff0000", "Blue": "#0000ff"}
        viewer.channel_max_values = {}
        viewer.mask_names = []
        viewer.masks_available = False
        viewer.annotations_available = False
        viewer.annotation_display_enabled = False
        viewer.annotation_overlay_mode = "mask"
        viewer.annotation_overlay_alpha = 0.5
        viewer.active_annotation_name = None
        viewer.mask_cache = {}
        viewer.label_masks_cache = {}
        viewer.annotation_label_cache = {}
        viewer.annotation_cache = {}
        viewer.annotation_palettes = {}
        viewer.annotation_class_ids = {}
        viewer.image_cache = OrderedDict(
            {
                "FOV_A": {"DNA": np.ones((4, 4), dtype=np.float32)},
                "FOV_B": {"DNA": np.ones((4, 4), dtype=np.float32) * 2.0},
            }
        )
        viewer.current_downsample_factor = 1
        viewer.mask_outline_thickness = 1
        viewer.pixel_size_nm = 390.0
        viewer.ui_component = SimpleNamespace(
            image_selector=SimpleNamespace(value="FOV_A"),
            channel_selector=SimpleNamespace(value=("DNA",)),
            color_controls={"DNA": SimpleNamespace(value="Red")},
            contrast_min_controls={"DNA": SimpleNamespace(value=0.0)},
            contrast_max_controls={"DNA": SimpleNamespace(value=2.0)},
            no_image_checkbox=SimpleNamespace(value=False),
            mask_display_controls={},
            mask_color_controls={},
        )
        viewer.image_display = SimpleNamespace(
            fig=SimpleNamespace(
                get_size_inches=lambda: np.array([2.0, 2.0], dtype=np.float32),
                set_size_inches=lambda *_args, **_kwargs: None,
            )
        )

        viewer.update_controls = MethodType(lambda self, _change: None, viewer)
        viewer.update_display = MethodType(lambda self, _factor: None, viewer)
        viewer.load_fov = MethodType(lambda self, _fov, _channels=None: None, viewer)
        viewer._merge_channel_max = MethodType(lambda self, *_args, **_kwargs: None, viewer)
        viewer.get_pixel_size_nm = MethodType(lambda self: getattr(self, "pixel_size_nm", 390.0), viewer)
        return viewer

    def _make_export_plugin(self, viewer: _BatchExportViewerStub | None = None):
        viewer = viewer or _BatchExportViewerStub(self.base_path)

        class _TestBatchExportPlugin(BatchExportPlugin):
            def setup_widget_observers(self):  # pragma: no cover - override UI wiring for tests
                return

            def _build_widgets(self):  # pragma: no cover - provide simplified widgets for tests
                self.ui_component.include_masks = _StubWidget(True)
                self.ui_component.include_annotations = _StubWidget(False)
                self.ui_component.mask_outline_thickness = _StubWidget(self._mask_outline_thickness)
                self.ui_component.overlay_hint = _StubHTML()
                self.ui_component.masks_only = _StubWidget(False)
                self.ui_component.mask_layer_dropdown = _StubDropdown("MASK", [("MASK", "MASK")])
                self.ui_component.mask_color_picker = _StubWidget("#ffffff")
                self.ui_component.mask_alpha_slider = _StubWidget(1.0)
                self.ui_component.mask_palette_enabled = _StubWidget(False)
                self.ui_component.mask_palette_dropdown = _StubDropdown(None, [("Current settings", None)])
                self.ui_component.config_name_input = _StubWidget("")
                self.ui_component.config_saved_dropdown = _StubDropdown(None, [("No saved configs", None)])
                self.ui_component.config_save_button = _StubWidget(None)
                self.ui_component.config_load_button = _StubWidget(None)
                self.ui_component.config_delete_button = _StubWidget(None)
                self.ui_component.config_status = _StubHTML()

            def _build_layout(self):  # pragma: no cover - skipped in tests
                return

            def _connect_events(self):  # pragma: no cover - manually wire slider observer
                self.ui_component.mask_outline_thickness.observe(
                    self._on_mask_outline_thickness_change,
                    names="value",
                )
                self.ui_component.masks_only.observe(
                    lambda _: self._invalidate_overlay_cache(),
                    names="value",
                )

            def refresh_marker_options(self):  # pragma: no cover - not needed for tests
                return

            def refresh_fov_options(self):  # pragma: no cover - not needed for tests
                return

            def refresh_cell_options(self):  # pragma: no cover - not needed for tests
                return

            def refresh_roi_options(self):  # pragma: no cover - not needed for tests
                return

            def _refresh_mode_availability(self):  # pragma: no cover - mode_tabs not built in this stub
                return

        plugin = _TestBatchExportPlugin(viewer, width=320, height=480)
        viewer.SidePlots.export_fovs_output = plugin
        self.addCleanup(plugin._executor.shutdown, False)
        return viewer, plugin

    def _configure_overlays(self, viewer: ImageMaskViewer) -> None:
        mask_array = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        )
        annotation_array = np.array(
            [
                [0, 1, 0],
                [1, 2, 1],
                [0, 1, 0],
            ],
            dtype=np.int32,
        )

        viewer.masks_available = True
        viewer.annotations_available = True
        viewer.annotation_display_enabled = True
        viewer.active_annotation_name = "ANN"
        viewer.annotation_overlay_mode = "combined"
        viewer.annotation_overlay_alpha = 0.4
        viewer.mask_names = ["MASK1"]
        viewer.mask_cache = {"FOV_A": {"MASK1": mask_array}}
        viewer.label_masks_cache = {}
        viewer.annotation_cache = {"FOV_A": {"ANN": annotation_array}}
        viewer.annotation_label_cache = {"FOV_A": {"ANN": {1: annotation_array}}}
        viewer.annotation_palettes = {"ANN": {"1": "#00ff00", "2": "#ff00ff"}}

        viewer.ui_component.mask_display_controls = {
            "MASK1": SimpleNamespace(value=True)
        }
        viewer.ui_component.mask_color_controls = {
            "MASK1": SimpleNamespace(value="Blue")
        }

    def test_capture_overlay_snapshot_and_rebuild(self) -> None:
        viewer = self._make_viewer()
        self._configure_overlays(viewer)
        viewer.ui_component.no_image_checkbox.value = True

        snapshot = viewer.capture_overlay_snapshot(include_annotations=True, include_masks=True)
        self.assertTrue(snapshot.include_annotations)
        self.assertTrue(snapshot.include_masks)
        self.assertTrue(snapshot.skip_image_layer)
        self.assertIsNotNone(snapshot.annotation)
        self.assertEqual(snapshot.masks[0].name, "MASK1")

        annotation_settings, mask_settings = viewer.build_overlay_settings_from_snapshot(
            "FOV_A",
            1,
            snapshot,
        )

        self.assertIsNotNone(annotation_settings)
        self.assertEqual(len(mask_settings), 1)
        self.assertEqual(mask_settings[0].mode, "outline")
        self.assertTrue(mask_settings[0].array.any())
        self.assertEqual(mask_settings[0].outline_thickness, 1)

        # Disable annotations and masks via include flags
        snapshot_disabled = viewer.capture_overlay_snapshot(include_annotations=False, include_masks=False)
        self.assertFalse(snapshot_disabled.include_annotations)
        self.assertFalse(snapshot_disabled.include_masks)
        self.assertTrue(snapshot_disabled.skip_image_layer)

    def test_batch_export_snapshot_strips_painter_when_no_palette_override(self) -> None:
        """Bug 1 regression: live painter must be stripped when no palette override is used."""
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=4)
        viewer_stub.capture_overlay_snapshot = lambda **_kwargs: OverlaySnapshot(
            include_annotations=False,
            include_masks=True,
            annotation=None,
            masks=(),
            mask_painter=MaskPainterSnapshot(
                mask_name="MASK",
                identifier="cell_type",
                active_classes=("Tumor",),
                class_colors={"Tumor": "#00ff00"},
                class_visible={"Tumor": True},
                class_fill={"Tumor": True},
                class_opacity={"Tumor": 50},
                default_color="#ffffff",
                global_fill_opacity=35,
                show_borders_on_filled=True,
                outline_thickness=1,
            ),
        )

        _viewer, plugin = self._make_export_plugin(viewer_stub)
        plugin._mask_outline_thickness = 7

        snapshot = plugin._capture_overlay_snapshot(include_masks=True, include_annotations=False)

        self.assertIsNotNone(snapshot)
        self.assertIsNone(snapshot.mask_painter)

    def test_palette_override_preserves_outline_thickness(self) -> None:
        """Outline thickness must be applied to the painter snapshot built from a saved palette."""
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=1)
        viewer_stub.capture_overlay_snapshot = lambda **_kwargs: OverlaySnapshot(
            include_annotations=False,
            include_masks=True,
            annotation=None,
            masks=(),
            mask_painter=None,
        )
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        plugin._mask_outline_thickness = 7
        payload = self._make_palette_payload("Thick Palette")
        plugin._palette_registry = {
            "Thick Palette": {"path": "/fake/path.json", "saved_at": "2026-01-01T00:00:00Z"}
        }
        with mock.patch(
            "ueler.viewer.palette_store.read_palette_file",
            return_value=payload,
        ), mock.patch(
            "ueler.viewer.plugin.mask_painter._resolve_mask_type_color",
            return_value=None,
        ):
            snapshot = plugin._capture_overlay_snapshot(
                include_masks=True,
                include_annotations=False,
                palette_name="Thick Palette",
            )
        self.assertIsNotNone(snapshot.mask_painter)
        self.assertEqual(snapshot.mask_painter.outline_thickness, 7)

    def test_export_fovs_batch_writes_file(self) -> None:
        viewer = self._make_viewer()
        output_dir = self.base_path / "exports"
        results = viewer.export_fovs_batch(
            "test",
            output_dir=str(output_dir),
            fovs=["FOV_A"],
            show_progress=False,
        )
        self.assertEqual(results, {"FOV_A": True})
        expected_file = output_dir / "FOV_A.png"
        self.assertTrue(expected_file.exists(), "Expected export file was not created")

    def test_export_fovs_batch_missing_channel_reports_error(self) -> None:
        viewer = self._make_viewer()
        viewer.marker_sets["missing"] = {
            "selected_channels": ("RNA",),
            "channel_settings": {
                "RNA": {
                    "color": "Red",
                    "contrast_min": 0.0,
                    "contrast_max": 1.0,
                }
            },
        }
        viewer.ui_component.color_controls["RNA"] = SimpleNamespace(value="Red")
        viewer.ui_component.contrast_min_controls["RNA"] = SimpleNamespace(value=0.0)
        viewer.ui_component.contrast_max_controls["RNA"] = SimpleNamespace(value=1.0)

        output_dir = self.base_path / "exports_missing"
        results = viewer.export_fovs_batch(
            "missing",
            output_dir=str(output_dir),
            fovs=["FOV_A"],
            show_progress=False,
        )
        self.assertIn("FOV_A", results)
        self.assertNotEqual(results["FOV_A"], True)
        self.assertFalse(list(output_dir.glob("*.png")), "Unexpected export file was created")

    def test_batch_export_plugin_defaults_to_viewer_outline(self) -> None:
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=4)
        viewer, plugin = self._make_export_plugin(viewer_stub)
        slider = plugin.ui_component.mask_outline_thickness
        self.assertEqual(viewer.mask_outline_thickness, 4)
        self.assertEqual(slider.value, 4)
        self.assertEqual(plugin._mask_outline_thickness, 4)
        self.assertFalse(plugin._mask_outline_overridden)

    def test_batch_export_plugin_slider_override_is_local(self) -> None:
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=2)
        viewer, plugin = self._make_export_plugin(viewer_stub)
        plugin.ui_component.mask_outline_thickness.value = 7
        self.assertEqual(viewer.mask_outline_thickness, 2)
        self.assertEqual(plugin._mask_outline_thickness, 7)
        self.assertTrue(plugin._mask_outline_overridden)

    def test_batch_export_plugin_syncs_with_viewer_when_not_overridden(self) -> None:
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=3)
        viewer, plugin = self._make_export_plugin(viewer_stub)
        viewer.mask_outline_thickness = 5
        plugin.on_viewer_mask_outline_change(5)
        self.assertEqual(plugin._mask_outline_thickness, 5)
        self.assertFalse(plugin._mask_outline_overridden)
        self.assertEqual(plugin.ui_component.mask_outline_thickness.value, 5)

    def test_batch_export_plugin_ignores_viewer_update_when_overridden(self) -> None:
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=2)
        viewer, plugin = self._make_export_plugin(viewer_stub)
        plugin.ui_component.mask_outline_thickness.value = 6
        self.assertTrue(plugin._mask_outline_overridden)
        viewer.mask_outline_thickness = 3
        plugin.on_viewer_mask_outline_change(3)
        self.assertEqual(plugin._mask_outline_thickness, 6)
        self.assertEqual(plugin.ui_component.mask_outline_thickness.value, 6)
        self.assertTrue(plugin._mask_outline_overridden)
        viewer.mask_outline_thickness = 6
        plugin.on_viewer_mask_outline_change(6)
        self.assertFalse(plugin._mask_outline_overridden)

    def test_resolve_marker_profile_uses_ui_channels_when_marker_set_empty(self) -> None:
        viewer = self._make_viewer()
        viewer.marker_sets["test"]["selected_channels"] = ()
        viewer.marker_sets["test"]["channel_settings"] = {}
        viewer.ui_component.channel_selector = SimpleNamespace(value=("DNA",))
        viewer.ui_component.color_controls["DNA"].value = "Red"
        viewer.ui_component.contrast_min_controls["DNA"].value = 0.0
        viewer.ui_component.contrast_max_controls["DNA"].value = 2.0

        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = viewer
        plugin.ui_component = SimpleNamespace(marker_set_dropdown=SimpleNamespace(value="test"))
        plugin._viewer_pixel_size_nm = viewer.pixel_size_nm
        plugin._mask_outline_thickness = viewer.mask_outline_thickness
        plugin._mask_outline_overridden = False
        plugin._overlay_cache = {}
        plugin._overlay_snapshot = None

        profile = plugin._resolve_marker_profile()

        self.assertEqual(profile.selected_channels, ("DNA",))
        self.assertIn("DNA", profile.channel_settings)
        settings = profile.channel_settings["DNA"]
        self.assertAlmostEqual(settings.contrast_min, 0.0)
        self.assertAlmostEqual(settings.contrast_max, 2.0)

    def test_preview_single_cell_handles_scale_bar_without_error(self) -> None:
        viewer = self._make_viewer()
        viewer.marker_sets["test"]["selected_channels"] = ("DNA",)
        viewer.marker_sets["test"]["channel_settings"] = {
            "DNA": {
                "color": "Red",
                "contrast_min": 0.0,
                "contrast_max": 1.0,
            }
        }
        viewer.ui_component.channel_selector.value = ("DNA",)
        viewer.image_cache["FOV_A"] = {"DNA": np.ones((8, 8), dtype=np.float32)}
        viewer.load_fov = MethodType(lambda self, _fov, _channels=None: None, viewer)
        viewer.capture_overlay_snapshot = MethodType(
            lambda self, include_masks, include_annotations: OverlaySnapshot(
                include_annotations=include_annotations,
                include_masks=include_masks,
                annotation=None,
                masks=(),
            ),
            viewer,
        )
        viewer.build_overlay_settings_from_snapshot = MethodType(
            lambda self, _fov, _downsample, _snapshot: (None, ()),
            viewer,
        )
        viewer.get_pixel_size_nm = MethodType(lambda self: getattr(self, "pixel_size_nm", 390.0), viewer)
        viewer.fov_key = "fov"
        viewer.x_key = "X"
        viewer.y_key = "Y"
        viewer.label_key = "label"

        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = viewer
        plugin._mask_outline_thickness = viewer.mask_outline_thickness
        plugin._viewer_outline_thickness = viewer.mask_outline_thickness
        plugin._mask_outline_overridden = False
        plugin._overlay_cache = {}
        plugin._overlay_snapshot = None
        plugin._notify = lambda *_args, **_kwargs: None
        plugin._cell_records = {
            0: {"fov": "FOV_A", "X": 3.0, "Y": 3.0, "label": "cell0"}
        }
        plugin.ui_component = SimpleNamespace(
            marker_set_dropdown=SimpleNamespace(value="test"),
            cell_selection=_StubWidget((0,)),
            cell_crop_size=_StubWidget(6),
            downsample_input=_StubWidget(1),
            include_scale_bar=_StubWidget(True),
            scale_bar_ratio=_StubWidget(10.0),
            include_masks=_StubWidget(False),
            include_annotations=_StubWidget(False),
            dpi_input=_StubWidget(300),
            cell_preview_output=_StubOutputWidget(),
        )

        mock_spec = SimpleNamespace(physical_length_um=5.0)

        with mock.patch(
            "ueler.viewer.plugin.export_fovs.render_crop_to_array",
            return_value=np.ones((8, 8, 3), dtype=np.float32),
        ), mock.patch.object(
            plugin,
            "_finalise_array",
            return_value=(np.full((8, 8, 3), 128, dtype=np.uint8), mock_spec),
        ) as finalise_mock, mock.patch.object(
            plugin,
            "_render_with_scale_bar",
            side_effect=lambda array, _spec, _dpi: array,
        ) as scale_mock, mock.patch(
            "ueler.viewer.plugin.export_fovs.plt.subplots",
            return_value=(
                SimpleNamespace(),
                SimpleNamespace(imshow=lambda *_args, **_kwargs: None, axis=lambda *_args, **_kwargs: None),
            ),
        ), mock.patch("ueler.viewer.plugin.export_fovs.plt.close"), mock.patch("ueler.viewer.plugin.export_fovs.display"):
            plugin._preview_single_cell()

        finalise_mock.assert_called_once()
        scale_mock.assert_called_once()
        self.assertTrue(plugin.ui_component.cell_preview_output.cleared)

    # ------------------------------------------------------------------
    # Feature 2: masks_only tests
    # ------------------------------------------------------------------
    def _make_plugin_with_snapshot(self, skip_image_layer=False):
        """Helper that wires a viewer stub returning a fixed OverlaySnapshot."""
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=1)
        viewer_stub.capture_overlay_snapshot = lambda **kwargs: OverlaySnapshot(
            include_annotations=kwargs.get("include_annotations", False),
            include_masks=kwargs.get("include_masks", True),
            skip_image_layer=skip_image_layer,
            annotation=None,
            masks=(),
            mask_painter=MaskPainterSnapshot(
                mask_name="MASK",
                identifier="cell_type",
                active_classes=("T",),
                class_colors={"T": "#ff0000"},
                class_visible={"T": True},
                class_fill={"T": False},
                class_opacity={"T": 35},
                default_color="#ffffff",
                global_fill_opacity=35,
                show_borders_on_filled=False,
                outline_thickness=1,
            ),
        )
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        return plugin

    def test_masks_only_sets_skip_image_layer_true(self):
        plugin = self._make_plugin_with_snapshot(skip_image_layer=False)
        plugin.ui_component.masks_only.value = True
        snapshot = plugin._capture_overlay_snapshot(include_masks=True, include_annotations=False)
        self.assertTrue(snapshot.skip_image_layer)

    def test_masks_only_false_overrides_viewer_true(self):
        """Export masks_only=False must override the viewer's skip_image_layer=True."""
        plugin = self._make_plugin_with_snapshot(skip_image_layer=True)
        plugin.ui_component.masks_only.value = False
        snapshot = plugin._capture_overlay_snapshot(include_masks=True, include_annotations=False)
        self.assertFalse(snapshot.skip_image_layer)

    def test_masks_only_false_keeps_viewer_false(self):
        plugin = self._make_plugin_with_snapshot(skip_image_layer=False)
        plugin.ui_component.masks_only.value = False
        snapshot = plugin._capture_overlay_snapshot(include_masks=True, include_annotations=False)
        self.assertFalse(snapshot.skip_image_layer)

    def test_masks_only_invalidates_cache(self):
        plugin = self._make_plugin_with_snapshot()
        plugin._overlay_snapshot = object()
        plugin._overlay_cache = {"key": "value"}
        plugin.ui_component.masks_only.value = True
        self.assertIsNone(plugin._overlay_snapshot)
        self.assertEqual(plugin._overlay_cache, {})

    def test_invalidate_overlay_cache_clears_state(self):
        plugin = self._make_plugin_with_snapshot()
        plugin._overlay_snapshot = object()
        plugin._overlay_cache = {"a": 1, "b": 2}
        plugin._invalidate_overlay_cache()
        self.assertIsNone(plugin._overlay_snapshot)
        self.assertEqual(plugin._overlay_cache, {})

    # ------------------------------------------------------------------
    # Feature 1: palette override tests
    # ------------------------------------------------------------------
    def _make_palette_payload(self, name="Test Palette"):
        return {
            "name": name,
            "version": "1.1.0",
            "identifier": "cell_type",
            "default_color": "#aabbcc",
            "class_order": ["T", "B"],
            "active_classes": ["T", "B"],
            "only_specified": False,
            "colors": {"T": "#ff0000", "B": "#0000ff"},
            "modes": {"T": "fill", "B": "outline"},
            "visible": {"T": True, "B": False},
            "opacities": {"T": 60, "B": 40},
            "global_fill": True,
            "global_fill_opacity": 50,
            "show_fill_borders": True,
            "border_color_mode": "same_as_fill",
            "saved_at": "2026-01-01T00:00:00Z",
        }

    def test_snapshot_from_palette_payload_field_mapping(self):
        viewer_stub = _BatchExportViewerStub(self.base_path)
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        payload = self._make_palette_payload()
        result = plugin._snapshot_from_palette_payload(payload, outline_thickness=3)
        self.assertEqual(result.identifier, "cell_type")
        self.assertEqual(result.class_colors["T"], "#ff0000")
        self.assertEqual(result.class_colors["B"], "#0000ff")
        self.assertTrue(result.class_fill["T"])
        self.assertFalse(result.class_fill["B"])
        self.assertFalse(result.class_visible["B"])
        self.assertEqual(result.global_fill_opacity, 50)
        self.assertTrue(result.show_borders_on_filled)
        self.assertEqual(result.border_color_mode, "same_as_fill")
        self.assertEqual(result.outline_thickness, 3)
        self.assertEqual(result.default_color, "#aabbcc")
        self.assertEqual(result.mask_type_color, "#aabbcc")

    def test_snapshot_from_palette_payload_empty_class_order(self):
        viewer_stub = _BatchExportViewerStub(self.base_path)
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        payload = {"name": "empty", "default_color": "#ffffff"}
        result = plugin._snapshot_from_palette_payload(payload, outline_thickness=1)
        self.assertEqual(result.active_classes, ())
        self.assertEqual(result.class_colors, {})

    def test_palette_override_uses_payload_in_snapshot(self):
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=1)
        viewer_stub.capture_overlay_snapshot = lambda **kwargs: OverlaySnapshot(
            include_annotations=False,
            include_masks=True,
            annotation=None,
            masks=(),
            mask_painter=MaskPainterSnapshot(
                mask_name="MASK",
                identifier="original",
                active_classes=(),
                class_colors={},
                class_visible={},
                class_fill={},
                class_opacity={},
                default_color="#000000",
                global_fill_opacity=35,
                show_borders_on_filled=False,
                outline_thickness=1,
            ),
        )
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        payload = self._make_palette_payload("My Palette")
        plugin._palette_registry = {
            "My Palette": {"path": "/fake/path.json", "saved_at": "2026-01-01T00:00:00Z"}
        }
        with mock.patch(
            "ueler.viewer.palette_store.read_palette_file",
            return_value=payload,
        ):
            snapshot = plugin._capture_overlay_snapshot(
                include_masks=True,
                include_annotations=False,
                palette_name="My Palette",
            )
        self.assertEqual(snapshot.mask_painter.identifier, "cell_type")
        self.assertTrue(snapshot.mask_painter.class_fill.get("T", False))

    def test_palette_override_bad_name_falls_back_to_live_state(self):
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=1)
        live_painter = MaskPainterSnapshot(
            mask_name="MASK",
            identifier="live",
            active_classes=("X",),
            class_colors={"X": "#cccccc"},
            class_visible={"X": True},
            class_fill={"X": False},
            class_opacity={"X": 35},
            default_color="#cccccc",
            global_fill_opacity=35,
            show_borders_on_filled=False,
            outline_thickness=1,
        )
        viewer_stub.capture_overlay_snapshot = lambda **kwargs: OverlaySnapshot(
            include_annotations=False,
            include_masks=True,
            annotation=None,
            masks=(),
            mask_painter=live_painter,
        )
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        plugin._palette_registry = {}
        snapshot = plugin._capture_overlay_snapshot(
            include_masks=True,
            include_annotations=False,
            palette_name="nonexistent",
        )
        self.assertEqual(snapshot.mask_painter.identifier, "live")

    def test_refresh_palette_dropdown_populates_options(self):
        viewer_stub = _BatchExportViewerStub(self.base_path)
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        with mock.patch.object(
            plugin,
            "_load_palette_registry",
            return_value={"Alpha": {"path": "/a"}, "Beta": {"path": "/b"}},
        ):
            plugin._refresh_palette_dropdown()
        options = plugin.ui_component.mask_palette_dropdown.options
        # options are list of (label, value) tuples
        values = [v for _, v in options] if options and isinstance(options[0], tuple) else options
        self.assertIn(None, values)
        self.assertIn("Alpha", values)
        self.assertIn("Beta", values)

    def test_refresh_palette_dropdown_empty_registry(self):
        viewer_stub = _BatchExportViewerStub(self.base_path)
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        with mock.patch.object(plugin, "_load_palette_registry", return_value={}):
            plugin._refresh_palette_dropdown()
        options = plugin.ui_component.mask_palette_dropdown.options
        values = [v for _, v in options] if options and isinstance(options[0], tuple) else options
        self.assertIn(None, values)
        self.assertEqual(len(values), 1)

    def test_snapshot_from_palette_payload_uses_live_mask_type_color(self):
        viewer_stub = _BatchExportViewerStub(self.base_path)
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        payload = self._make_palette_payload()
        result = plugin._snapshot_from_palette_payload(payload, outline_thickness=1, mask_type_color="#abcdef")
        self.assertEqual(result.mask_type_color, "#abcdef")

    def test_snapshot_from_palette_payload_falls_back_to_default_color(self):
        viewer_stub = _BatchExportViewerStub(self.base_path)
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        payload = self._make_palette_payload()
        result = plugin._snapshot_from_palette_payload(payload, outline_thickness=1)
        self.assertEqual(result.mask_type_color, result.default_color)

    def test_palette_override_passes_live_mask_type_color(self):
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=1)
        viewer_stub.capture_overlay_snapshot = lambda **kwargs: OverlaySnapshot(
            include_annotations=False,
            include_masks=True,
            annotation=None,
            masks=(),
            mask_painter=MaskPainterSnapshot(
                mask_name="MASK",
                identifier="live",
                active_classes=(),
                class_colors={},
                class_visible={},
                class_fill={},
                class_opacity={},
                default_color="#000000",
                global_fill_opacity=35,
                show_borders_on_filled=False,
                outline_thickness=1,
                mask_type_color="#abcdef",
            ),
        )
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        payload = self._make_palette_payload("Palette")
        plugin._palette_registry = {
            "Palette": {"path": "/fake/path.json", "saved_at": "2026-01-01T00:00:00Z"}
        }
        with mock.patch(
            "ueler.viewer.palette_store.read_palette_file",
            return_value=payload,
        ):
            snapshot = plugin._capture_overlay_snapshot(
                include_masks=True,
                include_annotations=False,
                palette_name="Palette",
            )
        self.assertEqual(snapshot.mask_painter.mask_type_color, "#abcdef")

    def test_palette_override_reads_left_panel_color_when_painter_snapshot_is_none(self):
        """When mask_painter is None (plugin disabled), mask_type_color comes from left-panel controls."""
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=1)
        # Return a snapshot with mask_painter=None (painter disabled)
        viewer_stub.capture_overlay_snapshot = lambda **kwargs: OverlaySnapshot(
            include_annotations=False,
            include_masks=True,
            annotation=None,
            masks=(),
            mask_painter=None,
        )
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        payload = self._make_palette_payload("My Palette")
        plugin._palette_registry = {
            "My Palette": {"path": "/fake/path.json", "saved_at": "2026-01-01T00:00:00Z"}
        }
        with mock.patch(
            "ueler.viewer.palette_store.read_palette_file",
            return_value=payload,
        ), mock.patch(
            "ueler.viewer.plugin.mask_painter._resolve_mask_type_color",
            return_value="#cc1122",
        ):
            snapshot = plugin._capture_overlay_snapshot(
                include_masks=True,
                include_annotations=False,
                palette_name="My Palette",
            )
        self.assertEqual(snapshot.mask_painter.mask_type_color, "#cc1122")

    # ------------------------------------------------------------------
    # Issue #101 regression tests
    # ------------------------------------------------------------------

    def test_live_painter_stripped_when_no_palette_override(self):
        """Bug 1: live Mask Painter colours must NOT appear when Override is unchecked."""
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=1)
        viewer_stub.capture_overlay_snapshot = lambda **kwargs: OverlaySnapshot(
            include_annotations=False,
            include_masks=True,
            annotation=None,
            masks=(),
            mask_painter=MaskPainterSnapshot(
                mask_name="MASK",
                identifier="live_class",
                active_classes=("T",),
                class_colors={"T": "#ff0000"},
                class_visible={"T": True},
                class_fill={"T": False},
                class_opacity={"T": 35},
                default_color="#ffffff",
                global_fill_opacity=35,
                show_borders_on_filled=False,
                outline_thickness=1,
            ),
        )
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        snapshot = plugin._capture_overlay_snapshot(
            include_masks=True,
            include_annotations=False,
            palette_name=None,
        )
        self.assertIsNone(snapshot.mask_painter)

    def test_export_local_layer_and_color_used_when_no_palette_override(self):
        """Mask layer and colour come from the export-local widgets, not the viewer controls."""
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=2)
        viewer_stub.mask_key = "MASK"
        viewer_stub.capture_overlay_snapshot = lambda **kwargs: OverlaySnapshot(
            include_annotations=False,
            include_masks=True,
            annotation=None,
            masks=(),
            mask_painter=None,
        )
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        plugin.ui_component.mask_color_picker.value = "#ff0000"
        snapshot = plugin._capture_overlay_snapshot(
            include_masks=True,
            include_annotations=False,
            palette_name=None,
        )
        self.assertEqual(len(snapshot.masks), 1)
        self.assertEqual(snapshot.masks[0].name, "MASK")
        self.assertAlmostEqual(snapshot.masks[0].color[0], 1.0)  # red channel
        self.assertAlmostEqual(snapshot.masks[0].color[1], 0.0)  # green channel
        self.assertEqual(snapshot.masks[0].outline_thickness, 2)
        self.assertIsNone(snapshot.mask_painter)

    def test_no_fallback_mask_when_palette_override_is_set(self):
        """When a palette override IS requested, Feature 1 sets mask_painter from
        the palette; no fallback MaskOverlaySnapshot should be injected on top."""
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=1)
        viewer_stub.mask_key = "MASK"
        viewer_stub.capture_overlay_snapshot = lambda **kwargs: OverlaySnapshot(
            include_annotations=False,
            include_masks=True,
            annotation=None,
            masks=(),
            mask_painter=None,
        )
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        payload = self._make_palette_payload("My Palette")
        plugin._palette_registry = {
            "My Palette": {"path": "/fake/path.json", "saved_at": "2026-01-01T00:00:00Z"}
        }
        with mock.patch(
            "ueler.viewer.palette_store.read_palette_file",
            return_value=payload,
        ), mock.patch(
            "ueler.viewer.plugin.mask_painter._resolve_mask_type_color",
            return_value=None,
        ):
            snapshot = plugin._capture_overlay_snapshot(
                include_masks=True,
                include_annotations=False,
                palette_name="My Palette",
            )
        self.assertEqual(snapshot.masks, ())
        self.assertIsNotNone(snapshot.mask_painter)

    # ------------------------------------------------------------------
    # Issue #101 reply: explicit mask layer selector
    # ------------------------------------------------------------------

    def test_refresh_mask_layer_dropdown_populates_from_mask_names(self):
        """Dropdown options are built from viewer.mask_names after refresh."""
        viewer_stub = _BatchExportViewerStub(self.base_path)
        viewer_stub.mask_names = ["MASK", "CELL"]
        viewer_stub.mask_key = "MASK"
        viewer_stub.masks_available = True
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        plugin._refresh_mask_layer_dropdown()
        opts = plugin.ui_component.mask_layer_dropdown.options
        values = [v for _, v in opts]
        self.assertIn("MASK", values)
        self.assertIn("CELL", values)

    def test_refresh_mask_layer_dropdown_defaults_to_mask_key(self):
        """When current dropdown value is not in the new options, it defaults to mask_key."""
        viewer_stub = _BatchExportViewerStub(self.base_path)
        viewer_stub.mask_names = ["CELL"]
        viewer_stub.mask_key = "CELL"
        viewer_stub.masks_available = True
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        plugin.ui_component.mask_layer_dropdown.value = "OLD"
        plugin._refresh_mask_layer_dropdown()
        self.assertEqual(plugin.ui_component.mask_layer_dropdown.value, "CELL")

    def test_capture_snapshot_uses_export_local_layer_dropdown(self):
        """Layer name in snapshot comes from mask_layer_dropdown, not mask_key fallback."""
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=1)
        viewer_stub.capture_overlay_snapshot = lambda **kwargs: OverlaySnapshot(
            include_annotations=False,
            include_masks=True,
            annotation=None,
            masks=(),
            mask_painter=None,
        )
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        plugin.ui_component.mask_layer_dropdown = _StubDropdown("CELL", [("CELL", "CELL")])
        plugin.ui_component.mask_color_picker.value = "#00ff00"
        snapshot = plugin._capture_overlay_snapshot(
            include_masks=True,
            include_annotations=False,
            palette_name=None,
        )
        self.assertEqual(len(snapshot.masks), 1)
        self.assertEqual(snapshot.masks[0].name, "CELL")
        self.assertAlmostEqual(snapshot.masks[0].color[1], 1.0)  # green channel
        self.assertAlmostEqual(snapshot.masks[0].alpha, 1.0)  # default alpha

    def test_capture_snapshot_uses_alpha_slider_value(self):
        """alpha in snapshot comes from mask_alpha_slider widget."""
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=1)
        viewer_stub.capture_overlay_snapshot = lambda **kwargs: OverlaySnapshot(
            include_annotations=False,
            include_masks=True,
            annotation=None,
            masks=(),
            mask_painter=None,
        )
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        plugin.ui_component.mask_layer_dropdown = _StubDropdown("CELL", [("CELL", "CELL")])
        plugin.ui_component.mask_alpha_slider.value = 0.4
        snapshot = plugin._capture_overlay_snapshot(
            include_masks=True,
            include_annotations=False,
            palette_name=None,
        )
        self.assertEqual(len(snapshot.masks), 1)
        self.assertAlmostEqual(snapshot.masks[0].alpha, 0.4)

    def test_capture_snapshot_painter_is_none_without_palette_override(self):
        """mask_painter is always None when no palette override is requested."""
        viewer_stub = _BatchExportViewerStub(self.base_path, mask_outline_thickness=1)
        viewer_stub.capture_overlay_snapshot = lambda **kwargs: OverlaySnapshot(
            include_annotations=False,
            include_masks=True,
            annotation=None,
            masks=(),
            mask_painter=MaskPainterSnapshot(
                mask_name="MASK",
                identifier="live",
                active_classes=(),
                class_colors={},
                class_visible={},
                class_fill={},
                class_opacity={},
                default_color="#ffffff",
                global_fill_opacity=35,
                show_borders_on_filled=False,
                outline_thickness=1,
            ),
        )
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        snapshot = plugin._capture_overlay_snapshot(
            include_masks=True,
            include_annotations=False,
            palette_name=None,
        )
        self.assertIsNone(snapshot.mask_painter)

    def test_config_roundtrip_includes_mask_layer_and_color(self):
        """mask_layer, mask_color, and mask_alpha survive a config save→load roundtrip."""
        viewer_stub = _BatchExportViewerStub(self.base_path)
        viewer_stub.mask_names = ["CELL"]
        viewer_stub.mask_key = "CELL"
        _viewer, plugin = self._make_export_plugin(viewer_stub)
        # Add extra widgets required by _collect_export_config
        plugin.ui_component.file_format_dropdown = _StubWidget("png")
        plugin.ui_component.downsample_input = _StubWidget(1)
        plugin.ui_component.dpi_input = _StubWidget(300)
        plugin.ui_component.include_scale_bar = _StubWidget(False)
        plugin.ui_component.scale_bar_ratio = _StubWidget(10.0)
        plugin.ui_component.include_annotations = _StubWidget(False)
        plugin.ui_component.marker_set_dropdown = _StubDropdown(None, [])
        plugin.ui_component.output_path = _StubWidget("")
        # Set the fields under test
        plugin.ui_component.mask_layer_dropdown = _StubDropdown("CELL", [("CELL", "CELL")])
        plugin.ui_component.mask_color_picker.value = "#aabbcc"
        plugin.ui_component.mask_alpha_slider.value = 0.6

        config = plugin._collect_export_config("test")
        self.assertEqual(config["mask_layer"], "CELL")
        self.assertEqual(config["mask_color"], "#aabbcc")
        self.assertAlmostEqual(config["mask_alpha"], 0.6)

        plugin.ui_component.mask_color_picker.value = "#000000"
        plugin.ui_component.mask_alpha_slider.value = 1.0
        plugin._apply_export_config(config)
        self.assertEqual(plugin.ui_component.mask_color_picker.value, "#aabbcc")
        self.assertAlmostEqual(plugin.ui_component.mask_alpha_slider.value, 0.6)


class TestSimpleViewerModeExport(unittest.TestCase):
    """BatchExportPlugin behaviour when cell_table is None (simple viewer mode)."""

    def _make_plugin(self, cell_table=None):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        viewer = _BatchExportViewerStub(Path(tmp.name))
        viewer.cell_table = cell_table

        from ipywidgets import HTML as _HTML, Tab, VBox

        class _ModePlugin(BatchExportPlugin):
            def setup_widget_observers(self):
                return

            def _build_widgets(self):
                self.ui_component.full_fov_box = VBox([])
                self.ui_component.single_cell_box = VBox([])
                self.ui_component.roi_box = VBox([])
                self.ui_component.mode_tabs = Tab(
                    children=[
                        self.ui_component.full_fov_box,
                        self.ui_component.single_cell_box,
                        self.ui_component.roi_box,
                    ]
                )
                self.ui_component.mode_tabs.selected_index = 0
                self.ui_component.include_masks = _StubWidget(False)
                self.ui_component.include_annotations = _StubWidget(False)
                self.ui_component.mask_outline_thickness = _StubWidget(1)
                self.ui_component.overlay_hint = _StubHTML()
                self.ui_component.masks_only = _StubWidget(False)
                self.ui_component.mask_layer_dropdown = _StubDropdown("MASK", [("MASK", "MASK")])
                self.ui_component.mask_color_picker = _StubWidget("#ffffff")
                self.ui_component.mask_alpha_slider = _StubWidget(1.0)
                self.ui_component.mask_palette_enabled = _StubWidget(False)
                self.ui_component.mask_palette_dropdown = _StubDropdown(None, [("Current settings", None)])
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

        plugin = _ModePlugin(viewer, width=320, height=480)
        self.addCleanup(plugin._executor.shutdown, False)
        return viewer, plugin

    def test_simple_mode_replaces_single_cell_box(self):
        """When cell_table is None, mode_tabs replaces single_cell_box with a notice widget."""
        viewer, plugin = self._make_plugin(cell_table=None)
        children = plugin.ui_component.mode_tabs.children
        self.assertIsNot(children[1], plugin.ui_component.single_cell_box)

    def test_simple_mode_notice_is_html(self):
        """The replacement widget in Single Cells tab is an HTML instance."""
        from ipywidgets import HTML as _HTML
        viewer, plugin = self._make_plugin(cell_table=None)
        children = plugin.ui_component.mode_tabs.children
        self.assertIsInstance(children[1], _HTML)

    def test_simple_mode_full_fov_and_roi_tabs_unaffected(self):
        """Full FOV and ROI boxes are preserved unchanged in simple viewer mode."""
        viewer, plugin = self._make_plugin(cell_table=None)
        children = plugin.ui_component.mode_tabs.children
        self.assertIs(children[0], plugin.ui_component.full_fov_box)
        self.assertIs(children[2], plugin.ui_component.roi_box)

    def test_simple_mode_resets_selected_index_when_on_single_cells_tab(self):
        """If the Single Cells tab is active when cell_table is None, selected_index is reset to 0."""
        viewer, plugin = self._make_plugin(cell_table=None)
        # Manually restore single_cell_box and set active tab to 1, then re-invoke the method
        plugin.ui_component.mode_tabs.children = (
            plugin.ui_component.full_fov_box,
            plugin.ui_component.single_cell_box,
            plugin.ui_component.roi_box,
        )
        plugin.ui_component.mode_tabs.selected_index = 1
        plugin._refresh_mode_availability()
        self.assertEqual(plugin.ui_component.mode_tabs.selected_index, 0)

    def test_simple_mode_leaves_selected_index_unchanged_on_other_tabs(self):
        """selected_index is not changed when the active tab is not Single Cells."""
        viewer, plugin = self._make_plugin(cell_table=None)
        plugin.ui_component.mode_tabs.selected_index = 2  # ROIs tab
        plugin._refresh_mode_availability()
        self.assertEqual(plugin.ui_component.mode_tabs.selected_index, 2)

    def test_full_mode_keeps_single_cell_box(self):
        """When cell_table is not None, mode_tabs retains the original single_cell_box."""
        viewer, plugin = self._make_plugin(cell_table=SimpleNamespace(empty=False))
        children = plugin.ui_component.mode_tabs.children
        self.assertIs(children[1], plugin.ui_component.single_cell_box)

    def test_on_cell_table_change_restores_single_cell_box(self):
        """on_cell_table_change restores single_cell_box after a table is loaded."""
        viewer, plugin = self._make_plugin(cell_table=None)
        # Verify the box was replaced after init
        self.assertIsNot(plugin.ui_component.mode_tabs.children[1], plugin.ui_component.single_cell_box)
        # Simulate cell table being set on the viewer
        viewer.cell_table = SimpleNamespace(empty=False)
        plugin.on_cell_table_change()
        self.assertIs(plugin.ui_component.mode_tabs.children[1], plugin.ui_component.single_cell_box)

    def test_refresh_mode_availability_idempotent_without_table(self):
        """Calling _refresh_mode_availability multiple times with no table is safe."""
        viewer, plugin = self._make_plugin(cell_table=None)
        first_notice = plugin.ui_component.mode_tabs.children[1]
        plugin._refresh_mode_availability()
        # Same notice object should remain (no unnecessary replacement)
        self.assertIs(plugin.ui_component.mode_tabs.children[1], first_notice)

    def test_refresh_mode_availability_idempotent_with_table(self):
        """Calling _refresh_mode_availability multiple times with a table is safe."""
        viewer, plugin = self._make_plugin(cell_table=SimpleNamespace(empty=False))
        plugin._refresh_mode_availability()
        self.assertIs(plugin.ui_component.mode_tabs.children[1], plugin.ui_component.single_cell_box)


class TestMarkerSetDropdownRefresh(unittest.TestCase):
    """on_marker_sets_changed() and update_marker_set() broadcast to BatchExportPlugin."""

    def _make_plugin(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        viewer = _BatchExportViewerStub(Path(tmp.name))
        viewer.marker_sets = {"SetA": {}, "SetB": {}}

        class _Stub(BatchExportPlugin):
            def setup_widget_observers(self):
                return

            def _build_widgets(self):
                from ipywidgets import VBox, Tab
                self.ui_component.full_fov_box = VBox([])
                self.ui_component.single_cell_box = VBox([])
                self.ui_component.roi_box = VBox([])
                self.ui_component.mode_tabs = Tab(
                    children=[
                        self.ui_component.full_fov_box,
                        self.ui_component.single_cell_box,
                        self.ui_component.roi_box,
                    ]
                )
                self.ui_component.mode_tabs.selected_index = 0
                from tests.test_export_fovs_batch import _StubWidget, _StubHTML
                self.ui_component.include_masks = _StubWidget(False)
                self.ui_component.include_annotations = _StubWidget(False)
                self.ui_component.mask_outline_thickness = _StubWidget(1)
                self.ui_component.overlay_hint = _StubHTML()
                self.ui_component.marker_set_dropdown = SimpleNamespace(
                    options=[], value=None
                )

            def _build_layout(self):
                return

            def _connect_events(self):
                return

            def refresh_fov_options(self):
                return

            def refresh_cell_options(self):
                return

            def refresh_roi_options(self):
                return

        plugin = _Stub(viewer, width=320, height=480)
        self.addCleanup(plugin._executor.shutdown, False)
        return viewer, plugin

    def test_on_marker_sets_changed_calls_refresh_marker_options(self):
        """on_marker_sets_changed() delegates to refresh_marker_options()."""
        viewer, plugin = self._make_plugin()
        call_log = []
        plugin.refresh_marker_options = lambda: call_log.append(1)
        plugin.on_marker_sets_changed()
        self.assertEqual(call_log, [1], "refresh_marker_options should be called exactly once")

    def test_on_marker_sets_changed_idempotent(self):
        """Calling on_marker_sets_changed() multiple times is safe."""
        viewer, plugin = self._make_plugin()
        call_log = []
class BatchExportMapModeTests(unittest.TestCase):
    """Tests for BatchExportPlugin map-mode lifecycle hooks."""

    def setUp(self) -> None:
        self._tmp_dir = TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.base_path = Path(self._tmp_dir.name)

    class _StubWidget:
        def __init__(self, value=None):
            self.value = value
            self.disabled = False

    def _make_minimal_plugin(self):
        """Create a minimal BatchExportPlugin with just roi_limit_to_fov wired up."""
        viewer = _BatchExportViewerStub(self.base_path)
        roi_limit = self._StubWidget(False)
        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = viewer
        plugin.ui_component = SimpleNamespace(roi_limit_to_fov=roi_limit)
        plugin._refresh_calls = []
        plugin.refresh_roi_options = lambda: plugin._refresh_calls.append(1)
        return plugin, viewer, roi_limit

    def test_on_map_mode_activate_disables_roi_limit_to_fov(self):
        plugin, viewer, roi_limit = self._make_minimal_plugin()
        roi_limit.value = True
        plugin.on_map_mode_activate()
        self.assertFalse(roi_limit.value)
        self.assertTrue(roi_limit.disabled)
        self.assertTrue(plugin._refresh_calls, "refresh_roi_options should be called")

    def test_on_map_mode_deactivate_reenables_roi_limit_to_fov(self):
        plugin, viewer, roi_limit = self._make_minimal_plugin()
        roi_limit.disabled = True
        plugin.on_map_mode_deactivate()
        self.assertFalse(roi_limit.disabled)
        self.assertTrue(plugin._refresh_calls, "refresh_roi_options should be called")

    def test_on_map_mode_activate_handles_missing_widget_gracefully(self):
        plugin, viewer, _ = self._make_minimal_plugin()
        plugin.ui_component = SimpleNamespace()  # no roi_limit_to_fov
        plugin.on_map_mode_activate()  # must not raise

    def test_on_map_mode_deactivate_handles_missing_widget_gracefully(self):
        plugin, viewer, _ = self._make_minimal_plugin()
        plugin.ui_component = SimpleNamespace()  # no roi_limit_to_fov
        plugin.on_map_mode_deactivate()  # must not raise


        plugin.refresh_marker_options = lambda: call_log.append(1)
        plugin.on_marker_sets_changed()
        plugin.on_marker_sets_changed()
        self.assertEqual(len(call_log), 2)


class BatchExportMapModeTests(unittest.TestCase):
    """Tests for BatchExportPlugin map-mode lifecycle hooks."""

    def setUp(self) -> None:
        self._tmp_dir = TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.base_path = Path(self._tmp_dir.name)

    class _StubWidget:
        def __init__(self, value=None):
            self.value = value
            self.disabled = False

    def _make_minimal_plugin(self):
        """Create a minimal BatchExportPlugin with just roi_limit_to_fov wired up."""
        viewer = _BatchExportViewerStub(self.base_path)
        roi_limit = self._StubWidget(False)
        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = viewer
        plugin.ui_component = SimpleNamespace(roi_limit_to_fov=roi_limit)
        plugin._refresh_calls = []
        plugin.refresh_roi_options = lambda: plugin._refresh_calls.append(1)
        return plugin, viewer, roi_limit

    def test_on_map_mode_activate_disables_roi_limit_to_fov(self):
        plugin, viewer, roi_limit = self._make_minimal_plugin()
        roi_limit.value = True
        plugin.on_map_mode_activate()
        self.assertFalse(roi_limit.value)
        self.assertTrue(roi_limit.disabled)
        self.assertTrue(plugin._refresh_calls, "refresh_roi_options should be called")

    def test_on_map_mode_deactivate_reenables_roi_limit_to_fov(self):
        plugin, viewer, roi_limit = self._make_minimal_plugin()
        roi_limit.disabled = True
        plugin.on_map_mode_deactivate()
        self.assertFalse(roi_limit.disabled)
        self.assertTrue(plugin._refresh_calls, "refresh_roi_options should be called")

    def test_on_map_mode_activate_handles_missing_widget_gracefully(self):
        plugin, viewer, _ = self._make_minimal_plugin()
        plugin.ui_component = SimpleNamespace()  # no roi_limit_to_fov
        plugin.on_map_mode_activate()  # must not raise

    def test_on_map_mode_deactivate_handles_missing_widget_gracefully(self):
        plugin, viewer, _ = self._make_minimal_plugin()
        plugin.ui_component = SimpleNamespace()  # no roi_limit_to_fov
        plugin.on_map_mode_deactivate()  # must not raise


class BatchExportMapROIItemsTests(unittest.TestCase):
    """Tests for map-mode ROI job creation and worker execution in BatchExportPlugin."""

    def setUp(self) -> None:
        self._tmp_dir = TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.base_path = Path(self._tmp_dir.name)

    def _make_plugin_for_build(self, roi_records, selected_ids):
        """Return a minimal BatchExportPlugin for testing _build_roi_items."""
        viewer = _BatchExportViewerStub(self.base_path)
        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = viewer
        plugin._roi_records = roi_records
        plugin.ui_component = SimpleNamespace(
            roi_selection=SimpleNamespace(value=tuple(selected_ids)),
        )
        return plugin

    def _make_marker_profile(self):
        from ueler.viewer.plugin.export_fovs import _MarkerProfile
        from ueler.rendering import ChannelRenderSettings
        return _MarkerProfile(
            name="test",
            selected_channels=("DNA",),
            channel_settings={"DNA": ChannelRenderSettings(color=(1.0, 0.0, 0.0), contrast_min=0.0, contrast_max=1.0)},
        )

    def _make_overlay_snapshot(self, *, skip_image_layer: bool = False):
        return OverlaySnapshot(
            include_annotations=False,
            include_masks=False,
            skip_image_layer=skip_image_layer,
            annotation=None,
            masks=(),
        )

    def test_build_roi_items_skips_unattributed_roi(self):
        """ROI with fov='' and map_id='' is silently skipped."""
        roi_id = "unattributed-roi-0001"
        roi_records = {roi_id: {"fov": "", "map_id": "", "x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10}}
        plugin = self._make_plugin_for_build(roi_records, [roi_id])
        items = plugin._build_roi_items(
            marker_profile=self._make_marker_profile(),
            output_dir=str(self.base_path),
            file_format="png",
            downsample=1,
            dpi=300,
            include_scale_bar=False,
            scale_ratio=10.0,
            pixel_size_nm=390.0,
            overlay_snapshot=self._make_overlay_snapshot(),
        )
        self.assertEqual(len(items), 0)

    def test_build_roi_items_creates_map_roi_item(self):
        """ROI with fov='' and map_id='slide-1' produces one JobItem with map_ prefix filename."""
        roi_id = "map-roi-aabbccddeeff1122"
        roi_records = {
            roi_id: {"fov": "", "map_id": "slide-1", "x_min": 100, "x_max": 200, "y_min": 50, "y_max": 150}
        }
        plugin = self._make_plugin_for_build(roi_records, [roi_id])
        items = plugin._build_roi_items(
            marker_profile=self._make_marker_profile(),
            output_dir=str(self.base_path),
            file_format="png",
            downsample=1,
            dpi=300,
            include_scale_bar=False,
            scale_ratio=10.0,
            pixel_size_nm=390.0,
            overlay_snapshot=self._make_overlay_snapshot(),
        )
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertIn("slide-1", item.output_path)
        self.assertTrue(item.output_path.startswith(str(self.base_path)))
        self.assertTrue(item.output_path.endswith(".png"))
        # Map-mode items use map_ prefix in filename
        fname = Path(item.output_path).name
        self.assertTrue(fname.startswith("map_"), f"Expected 'map_' prefix, got: {fname}")
        self.assertEqual(item.metadata.get("map_id"), "slide-1")

    def test_build_roi_items_single_fov_roi_unaffected(self):
        """ROI with a non-empty fov still creates a JobItem using the old path."""
        roi_id = "fov-roi-aabbccddeeff1122"
        roi_records = {
            roi_id: {"fov": "FOV_A", "map_id": "", "x_min": 10, "x_max": 50, "y_min": 10, "y_max": 50}
        }
        plugin = self._make_plugin_for_build(roi_records, [roi_id])
        items = plugin._build_roi_items(
            marker_profile=self._make_marker_profile(),
            output_dir=str(self.base_path),
            file_format="png",
            downsample=1,
            dpi=300,
            include_scale_bar=False,
            scale_ratio=10.0,
            pixel_size_nm=390.0,
            overlay_snapshot=self._make_overlay_snapshot(),
        )
        self.assertEqual(len(items), 1)
        fname = Path(items[0].output_path).name
        self.assertTrue(fname.startswith("FOV_A_"), f"Expected 'FOV_A_' prefix, got: {fname}")
        self.assertEqual(items[0].metadata.get("fov"), "FOV_A")

    def test_export_map_roi_worker_calls_render_map_region_direct(self):
        """_export_map_roi_worker uses _render_map_region_direct (not layer.render) so
        the marker profile's channel settings are applied — not the live UI widget values."""
        output_file = str(self.base_path / "out_map_roi.png")

        # The stub layer implements the internal helpers used by _render_map_region_direct.
        rendered_tile = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        region_direct_calls = []

        # Patch _render_map_region_direct to record the call and return a known canvas.
        def _fake_render_map_region_direct(
            _self, layer, xmin_um, xmax_um, ymin_um, ymax_um, ds, channels, channel_settings, *, skip_image_layer=False, snapshot=None
        ):
            region_direct_calls.append({
                "xmin_um": xmin_um,
                "xmax_um": xmax_um,
                "ymin_um": ymin_um,
                "ymax_um": ymax_um,
                "ds": ds,
                "channels": channels,
                "channel_settings": channel_settings,
                "skip_image_layer": skip_image_layer,
            })
            return rendered_tile

        class _StubLayer:
            _allowed_downsample = (1, 2, 4)
            def base_pixel_size_um(self): return 0.5
            def map_bounds(self): return (0.0, 500.0, 0.0, 500.0)  # zero-origin

        viewer = _BatchExportViewerStub(self.base_path)
        viewer._active_map_id = "slide-1"
        stub_layer = _StubLayer()
        viewer._get_map_layer = lambda _: stub_layer
        replay_calls = []

        def _replay(image, **kwargs):
            replay_calls.append(kwargs)
            return image

        viewer.apply_overlay_snapshot_to_map_array = _replay

        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = viewer

        roi = {
            "fov": "",
            "map_id": "slide-1",
            "x_min": 100.0,
            "x_max": 300.0,
            "y_min": 50.0,
            "y_max": 200.0,
        }
        profile = self._make_marker_profile()

        write_calls = []

        def _fake_write_image(_self, array, path, fmt, dpi, *, scale_bar_spec=None):
            write_calls.append({"path": path, "fmt": fmt})

        with mock.patch.object(BatchExportPlugin, "_render_map_region_direct", _fake_render_map_region_direct), \
             mock.patch.object(BatchExportPlugin, "_write_image", _fake_write_image):
            result = plugin._export_map_roi_worker(
                roi=roi,
                marker_profile=profile,
                downsample=1,
                file_format="png",
                output_path=output_file,
                dpi=300,
                include_scale_bar=False,
                scale_ratio=10.0,
                overlay_snapshot=self._make_overlay_snapshot(skip_image_layer=True),
            )

        # _render_map_region_direct must be called once with um-space coordinates.
        # With zero-origin bounds (0,0), xmin_um = bounds_min_x + x_min * base_px_um.
        self.assertEqual(len(region_direct_calls), 1)
        call = region_direct_calls[0]
        self.assertAlmostEqual(call["xmin_um"], 0.0 + 100.0 * 0.5)  # bounds_min_x + x_min * base_px_um
        self.assertAlmostEqual(call["xmax_um"], 0.0 + 300.0 * 0.5)
        self.assertAlmostEqual(call["ymin_um"], 0.0 + 50.0 * 0.5)
        self.assertAlmostEqual(call["ymax_um"], 0.0 + 200.0 * 0.5)

        # The marker profile's channel_settings must be forwarded (not UI widget values).
        self.assertIs(call["channel_settings"], profile.channel_settings)
        self.assertTrue(call["skip_image_layer"])

        # _write_image must be called with the output path.
        self.assertEqual(len(write_calls), 1)
        self.assertEqual(write_calls[0]["path"], output_file)
        self.assertEqual(write_calls[0]["fmt"], "png")
        self.assertEqual(len(replay_calls), 1)
        self.assertIs(replay_calls[0]["layer"], stub_layer)

        self.assertEqual(result["output_path"], output_file)

    def test_render_map_region_direct_uses_render_fov_to_array_per_tile(self):
        """_render_map_region_direct calls render_fov_to_array with the supplied
        channel_settings for each visible tile — not the live UI widget values."""
        import math
        from ueler.rendering import ChannelRenderSettings

        channel_settings = {
            "DNA": ChannelRenderSettings(color=(1.0, 0.0, 0.0), contrast_min=0.0, contrast_max=1.0),
        }
        channels = ("DNA",)

        # Minimal fake tile geometry matching the VirtualMapLayer dataclass fields
        from ueler.viewer.virtual_map_layer import MapTileGeometry
        tile = MapTileGeometry(
            name="FOV_A",
            pixel_size_um=0.5,
            width_px=100,
            height_px=100,
            x_min_um=0.0,
            x_max_um=50.0,
            y_min_um=0.0,
            y_max_um=50.0,
        )
        intersection = (5.0, 25.0, 5.0, 25.0)
        region_xy = (10, 50, 10, 50)
        region_ds = (10, 50, 10, 50)
        rendered_tile_array = np.ones((40, 40, 3), dtype=np.float32) * 0.4
        canvas = np.zeros((50, 50, 3), dtype=np.float32)

        class _StubLayer:
            _allowed_downsample = (1,)
            def base_pixel_size_um(self): return 0.5
            def _collect_visible_tiles(self, *_a): return [(tile, intersection)]
            def _allocate_canvas(self, *_a): return canvas
            def _compute_tile_region(self, *_a): return (region_xy, region_ds)
            def _blit_tile(self, *_a, **_kw): return None

        fov_arrays = {"DNA": np.ones((100, 100), dtype=np.float32) * 0.3}

        class _FakeViewer:
            image_cache = {"FOV_A": fov_arrays}
            def load_fov(self, fov_name, channels): pass

        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = _FakeViewer()

        render_calls = []

        def _fake_render_fov_to_array(fov_name, arrays, chans, ch_settings, *, downsample_factor, region_xy, region_ds=None, skip_image_layer=False, masks=None):
            render_calls.append({"fov": fov_name, "channel_settings": ch_settings, "skip_image_layer": skip_image_layer})
            return rendered_tile_array

        with mock.patch("ueler.viewer.plugin.export_fovs.render_fov_to_array", _fake_render_fov_to_array):
            plugin._render_map_region_direct(
                _StubLayer(),
                5.0, 25.0, 5.0, 25.0,
                1,
                channels,
                channel_settings,
                skip_image_layer=True,
            )

        self.assertEqual(len(render_calls), 1)
        self.assertEqual(render_calls[0]["fov"], "FOV_A")
        # The supplied channel_settings are passed through unmodified
        self.assertIs(render_calls[0]["channel_settings"], channel_settings)
        self.assertTrue(render_calls[0]["skip_image_layer"])

    def test_export_map_roi_worker_applies_map_bounds_offset(self):
        """_export_map_roi_worker adds the layer's physical bounds origin to canvas-pixel
        coordinates before passing them to _render_map_region_direct.  This matches the
        behaviour of _render_map_view and is required for maps whose tiles have non-zero
        stage-coordinate origins (e.g. absolute µm positions)."""
        output_file = str(self.base_path / "out_map_roi_offset.png")
        region_direct_calls = []
        rendered_tile = np.ones((10, 10, 3), dtype=np.float32) * 0.5

        def _fake_render(self_, layer, xmin_um, xmax_um, ymin_um, ymax_um, ds, channels, ch_settings, *, skip_image_layer=False, snapshot=None):
            region_direct_calls.append((xmin_um, xmax_um, ymin_um, ymax_um))
            return rendered_tile

        class _StubLayerOffset:
            _allowed_downsample = (1,)
            def base_pixel_size_um(self): return 2.0
            def map_bounds(self): return (1_000.0, 3_000.0, 4_000.0, 6_000.0)  # non-zero origin

        viewer = _BatchExportViewerStub(self.base_path)
        viewer._get_map_layer = lambda _: _StubLayerOffset()

        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = viewer

        roi = {"fov": "", "map_id": "slide-1", "x_min": 50.0, "x_max": 100.0, "y_min": 20.0, "y_max": 70.0}

        with mock.patch.object(BatchExportPlugin, "_render_map_region_direct", _fake_render), \
             mock.patch.object(BatchExportPlugin, "_write_image", lambda *a, **kw: None):
            plugin._export_map_roi_worker(
                roi=roi,
                marker_profile=self._make_marker_profile(),
                downsample=1, file_format="png", output_path=output_file,
                dpi=300, include_scale_bar=False, scale_ratio=10.0,
                overlay_snapshot=self._make_overlay_snapshot(),
            )

        self.assertEqual(len(region_direct_calls), 1)
        xmin_um, xmax_um, ymin_um, ymax_um = region_direct_calls[0]
        # Expected: bounds_min + pixel_coord * base_px_um
        self.assertAlmostEqual(xmin_um, 1_000.0 + 50.0 * 2.0)   # 1100
        self.assertAlmostEqual(xmax_um, 1_000.0 + 100.0 * 2.0)  # 1200
        self.assertAlmostEqual(ymin_um, 4_000.0 + 20.0 * 2.0)   # 4040
        self.assertAlmostEqual(ymax_um, 4_000.0 + 70.0 * 2.0)   # 4140

    def test_export_map_roi_worker_raises_on_empty_roi(self):
        """A ROI with zero extent raises ValueError."""
        class _StubLayer:
            _allowed_downsample = (1,)
            def base_pixel_size_um(self): return 1.0
            def map_bounds(self): return (0.0, 1000.0, 0.0, 1000.0)

        viewer = _BatchExportViewerStub(self.base_path)
        viewer._get_map_layer = lambda _: _StubLayer()
        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = viewer

        with mock.patch.object(BatchExportPlugin, "_write_image", lambda *_a, **_kw: None):
            with self.assertRaises(ValueError, msg="Should raise on zero extent ROI"):
                plugin._export_map_roi_worker(
                    roi={"fov": "", "map_id": "slide-1", "x_min": 5.0, "x_max": 5.0, "y_min": 0.0, "y_max": 10.0},
                    marker_profile=self._make_marker_profile(),
                    downsample=1,
                    file_format="png",
                    output_path=str(self.base_path / "zero.png"),
                    dpi=300,
                    include_scale_bar=False,
                    scale_ratio=10.0,
                    overlay_snapshot=self._make_overlay_snapshot(),
                )


    def test_render_map_region_direct_passes_masks_to_render_fov(self):
        """When the snapshot contains mask outlines, _render_map_region_direct passes
        per-tile MaskRenderSettings to render_fov_to_array so mask outlines are
        drawn at render time — not silently dropped."""
        from ueler.rendering import ChannelRenderSettings
        from ueler.rendering.engine import MaskOverlaySnapshot

        channel_settings = {
            "DNA": ChannelRenderSettings(color=(1.0, 0.0, 0.0), contrast_min=0.0, contrast_max=1.0),
        }
        channels = ("DNA",)
        mask_color = (1.0, 0.0, 0.0)

        from ueler.viewer.virtual_map_layer import MapTileGeometry
        tile = MapTileGeometry(
            name="FOV_A",
            pixel_size_um=0.5,
            width_px=100,
            height_px=100,
            x_min_um=0.0,
            x_max_um=50.0,
            y_min_um=0.0,
            y_max_um=50.0,
        )
        intersection = (5.0, 25.0, 5.0, 25.0)
        region_xy = (10, 50, 10, 50)
        region_ds = (10, 50, 10, 50)
        canvas = np.zeros((50, 50, 3), dtype=np.float32)
        rendered_tile = np.ones((40, 40, 3), dtype=np.float32) * 0.4
        fake_mask = np.ones((100, 100), dtype=np.int32)

        class _StubLayer:
            _allowed_downsample = (1,)
            def base_pixel_size_um(self): return 0.5
            def _collect_visible_tiles(self, *_a): return [(tile, intersection)]
            def _allocate_canvas(self, *_a): return canvas
            def _compute_tile_region(self, *_a): return (region_xy, region_ds)
            def _blit_tile(self, *_a, **_kw): return None

        class _FakeViewer:
            image_cache = {"FOV_A": {"DNA": np.ones((100, 100), dtype=np.float32) * 0.3}}
            def load_fov(self, fov_name, channels): pass
            def _get_mask_array(self, fov_name, mask_name):
                return fake_mask if mask_name == "whole_cell" else None

        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = _FakeViewer()

        snapshot = OverlaySnapshot(
            include_annotations=False,
            include_masks=True,
            skip_image_layer=False,
            annotation=None,
            masks=(MaskOverlaySnapshot(
                name="whole_cell",
                color=mask_color,
                alpha=1.0,
                mode="outline",
                outline_thickness=2,
            ),),
        )

        render_calls = []

        def _fake_render_fov(fov_name, arrays, chans, ch_settings, *, downsample_factor, region_xy, region_ds=None, skip_image_layer=False, masks=None):
            render_calls.append({"fov": fov_name, "masks": masks})
            return rendered_tile

        with mock.patch("ueler.viewer.plugin.export_fovs.render_fov_to_array", _fake_render_fov):
            plugin._render_map_region_direct(
                _StubLayer(),
                5.0, 25.0, 5.0, 25.0,
                1,
                channels,
                channel_settings,
                snapshot=snapshot,
            )

        self.assertEqual(len(render_calls), 1, "render_fov_to_array must be called once per tile")
        tile_masks = render_calls[0]["masks"]
        self.assertIsNotNone(tile_masks, "masks must not be None when snapshot.masks is non-empty")
        self.assertEqual(len(tile_masks), 1, "exactly one MaskRenderSettings expected")
        self.assertEqual(tile_masks[0].color, mask_color)
        self.assertEqual(tile_masks[0].mode, "outline")
        self.assertEqual(tile_masks[0].outline_thickness, 2)

    def test_build_roi_items_routes_nan_fov_map_roi_to_map_worker(self):
        """After CSV reload, fov may be NaN (float). With fov sanitized to '',
        map-mode ROIs must still route to the map-mode worker."""
        roi_id = "nan-fov-roi-aabbccddeeff"
        # Simulate what happens after _ensure_dataframe sanitizes: fov="" (not NaN)
        roi_records = {
            roi_id: {"fov": "", "map_id": "slide-1", "x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100}
        }
        plugin = self._make_plugin_for_build(roi_records, [roi_id])
        items = plugin._build_roi_items(
            marker_profile=self._make_marker_profile(),
            output_dir=str(self.base_path),
            file_format="png",
            downsample=1,
            dpi=300,
            include_scale_bar=False,
            scale_ratio=10.0,
            pixel_size_nm=390.0,
            overlay_snapshot=self._make_overlay_snapshot(),
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].metadata.get("map_id"), "slide-1")
        fname = Path(items[0].output_path).name
        self.assertTrue(fname.startswith("map_"), f"Expected map_ prefix, got: {fname}")

    def test_refresh_roi_options_shows_map_label_for_map_mode_roi(self):
        """Map-mode ROIs (fov='', map_id='slide-1') display [MAP:slide-1] in the label."""
        import pandas as pd
        from ueler.viewer.roi_manager import ROIManager

        viewer = _BatchExportViewerStub(self.base_path)
        viewer.roi_manager = ROIManager(str(self.base_path))

        # Insert a map-mode ROI directly
        viewer.roi_manager.add_roi({
            "fov": "",
            "map_id": "slide-1",
            "x_min": 10.0, "x_max": 50.0, "y_min": 10.0, "y_max": 50.0,
            "marker_set": "test_markers",
        })

        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = viewer
        plugin._roi_records = {}
        plugin.ui_component = SimpleNamespace(
            roi_selection=SimpleNamespace(options=[], value=()),
            roi_limit_to_fov=SimpleNamespace(value=False),
        )

        plugin.refresh_roi_options()

        # The label should include [MAP:slide-1], not empty fov or 'nan'
        options = plugin.ui_component.roi_selection.options
        self.assertEqual(len(options), 1)
        label, _ = options[0]
        self.assertIn("[MAP:slide-1]", label)
        self.assertNotIn("nan", label.lower())

    def test_render_map_region_direct_raises_when_tile_load_fails(self):
        """When image_cache.get returns None on both attempts, RuntimeError is raised."""
        from ueler.viewer.virtual_map_layer import MapTileGeometry

        tile = MapTileGeometry(
            name="MISSING_FOV",
            pixel_size_um=1.0,
            width_px=50,
            height_px=50,
            x_min_um=0.0,
            x_max_um=50.0,
            y_min_um=0.0,
            y_max_um=50.0,
        )
        intersection = (5.0, 45.0, 5.0, 45.0)
        region_xy = (5, 45, 5, 45)
        region_ds = (5, 45, 5, 45)
        canvas = np.zeros((50, 50, 3), dtype=np.float32)

        class _StubLayer:
            _allowed_downsample = (1,)
            def base_pixel_size_um(self): return 1.0
            def _collect_visible_tiles(self, *_a): return [(tile, intersection)]
            def _allocate_canvas(self, *_a): return canvas
            def _compute_tile_region(self, *_a): return (region_xy, region_ds)
            def _blit_tile(self, *_a, **_kw): return None

        class _FakeViewer:
            image_cache = {}  # .get always returns None
            def load_fov(self, fov_name, channels): pass

        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = _FakeViewer()

        channels = ("DNA",)
        from ueler.rendering import ChannelRenderSettings
        channel_settings = {"DNA": ChannelRenderSettings(color=(1.0, 0.0, 0.0), contrast_min=0.0, contrast_max=1.0)}

        with mock.patch("time.sleep"):  # skip actual sleep in tests
            with self.assertRaises(RuntimeError) as ctx:
                plugin._render_map_region_direct(
                    _StubLayer(), 5.0, 45.0, 5.0, 45.0, 1, channels, channel_settings
                )
        self.assertIn("MISSING_FOV", str(ctx.exception))
        self.assertIn("partial image", str(ctx.exception))

    def test_render_map_region_direct_succeeds_on_retry(self):
        """When image_cache.get returns None on the first call but a valid array on retry,
        the export completes without error."""
        from ueler.viewer.virtual_map_layer import MapTileGeometry
        from ueler.rendering import ChannelRenderSettings

        tile = MapTileGeometry(
            name="SLOW_FOV",
            pixel_size_um=1.0,
            width_px=50,
            height_px=50,
            x_min_um=0.0,
            x_max_um=50.0,
            y_min_um=0.0,
            y_max_um=50.0,
        )
        intersection = (5.0, 45.0, 5.0, 45.0)
        region_xy = (5, 45, 5, 45)
        region_ds = (5, 45, 5, 45)
        canvas = np.zeros((50, 50, 3), dtype=np.float32)
        fov_arrays = {"DNA": np.ones((50, 50), dtype=np.float32)}

        class _StubLayer:
            _allowed_downsample = (1,)
            def base_pixel_size_um(self): return 1.0
            def _collect_visible_tiles(self, *_a): return [(tile, intersection)]
            def _allocate_canvas(self, *_a): return canvas
            def _compute_tile_region(self, *_a): return (region_xy, region_ds)
            def _blit_tile(self, *_a, **_kw): return None

        call_count = {"n": 0}

        class _FakeViewer:
            def __init__(self):
                self._cache: dict = {}
            @property
            def image_cache(self):
                return self._cache
            def load_fov(self, fov_name, channels):
                call_count["n"] += 1
                if call_count["n"] >= 2:  # populated on second call
                    self._cache[fov_name] = fov_arrays

        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = _FakeViewer()

        channels = ("DNA",)
        channel_settings = {"DNA": ChannelRenderSettings(color=(1.0, 0.0, 0.0), contrast_min=0.0, contrast_max=1.0)}

        with mock.patch("time.sleep"), \
             mock.patch("ueler.viewer.plugin.export_fovs.render_fov_to_array",
                        return_value=np.zeros((40, 40, 3), dtype=np.float32)):
            result = plugin._render_map_region_direct(
                _StubLayer(), 5.0, 45.0, 5.0, 45.0, 1, channels, channel_settings
            )

        self.assertEqual(call_count["n"], 2)
        self.assertIsNotNone(result)

    def test_render_map_region_direct_canvas_size_correct_for_multi_tile_roi(self):
        """Canvas returned by _render_map_region_direct has correct pixel dimensions."""
        import math as _math
        from ueler.rendering import ChannelRenderSettings
        from ueler.viewer.virtual_map_layer import MapFOVSpec, SlideDescriptor, VirtualMapLayer

        # 3×3 grid of 100×100 µm tiles with 1 µm/px
        specs = tuple(
            MapFOVSpec(
                name=f"FOV_{r}_{c}",
                slide_id="grid",
                center_um=(50.0 + c * 100.0, 50.0 + r * 100.0),
                frame_size_px=(100, 100),
                fov_size_um=100.0,
                metadata={},
            )
            for r in range(3)
            for c in range(3)
        )
        descriptor = SlideDescriptor(
            slide_id="grid",
            source_path=Path("dummy_grid.json"),
            export_datetime=None,
            fovs=specs,
        )

        class _ViewerForLayer:
            def _render_fov_region(self, fov_name, channels, ds, region_xy, region_ds):
                w = max(1, region_ds[1] - region_ds[0])
                h = max(1, region_ds[3] - region_ds[2])
                return np.zeros((h, w, 3), dtype=np.float32)

        layer = VirtualMapLayer(_ViewerForLayer(), descriptor, allowed_downsample=[1, 2])

        fov_arrays = {"DNA": np.ones((100, 100), dtype=np.float32)}

        class _FakeViewer:
            def __init__(self):
                self.image_cache = {f"FOV_{r}_{c}": fov_arrays for r in range(3) for c in range(3)}
            def load_fov(self, *_a): pass

        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = _FakeViewer()

        channels = ("DNA",)
        channel_settings = {"DNA": ChannelRenderSettings(color=(1.0, 0.0, 0.0), contrast_min=0.0, contrast_max=1.0)}

        # ROI covers the full 3×3 grid (0–300 µm × 0–300 µm), ds=2
        with mock.patch("ueler.viewer.plugin.export_fovs.render_fov_to_array",
                        lambda *a, **kw: np.zeros((50, 50, 3), dtype=np.float32)):
            canvas = plugin._render_map_region_direct(
                layer, 0.0, 300.0, 0.0, 300.0, 2, channels, channel_settings
            )

        ds = 2
        pixel_size = 1.0 * ds
        expected_w = _math.ceil(300.0 / pixel_size)
        expected_h = _math.ceil(300.0 / pixel_size)
        self.assertGreaterEqual(canvas.shape[1], expected_w)
        self.assertGreaterEqual(canvas.shape[0], expected_h)

    def test_roi_table_observer_refreshes_batch_export_roi_list(self):
        """When ROI manager adds a new ROI, the batch export's roi_selection is refreshed."""
        from ueler.viewer.roi_manager import ROIManager

        viewer = _BatchExportViewerStub(self.base_path)
        viewer.roi_manager = ROIManager(str(self.base_path))

        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.main_viewer = viewer
        plugin._roi_records = {}
        plugin._executor = None
        plugin._current_job = None
        plugin._current_future = None
        plugin._event_loop = None
        plugin._io_loop = None
        plugin._cell_records = {}
        plugin._cell_filter_snapshot = ""
        plugin._seen_results = set()
        plugin._viewer_outline_thickness = 1
        plugin._mask_outline_thickness = 1
        plugin._mask_outline_overridden = False
        plugin._suspend_outline_widget_callback = False
        plugin._overlay_snapshot = None
        plugin._overlay_cache = {}
        plugin._viewer_pixel_size_nm = 390.0

        # Build minimal UI widgets needed by refresh_roi_options and _connect_events
        plugin.ui_component = SimpleNamespace(
            full_fov_use_all=SimpleNamespace(observe=lambda *a, **kw: None),
            browse_button=SimpleNamespace(on_click=lambda *a: None),
            start_button=SimpleNamespace(on_click=lambda *a: None),
            cancel_button=SimpleNamespace(on_click=lambda *a: None),
            cell_apply_filter=SimpleNamespace(on_click=lambda *a: None),
            cell_preview_button=SimpleNamespace(on_click=lambda *a: None),
            roi_limit_to_fov=SimpleNamespace(value=False, observe=lambda *a, **kw: None),
            roi_selection=SimpleNamespace(options=[], value=()),
            include_masks=SimpleNamespace(observe=lambda *a, **kw: None),
            mask_outline_thickness=SimpleNamespace(observe=lambda *a, **kw: None),
            mask_palette_enabled=SimpleNamespace(observe=lambda *a, **kw: None),
            mask_palette_dropdown=SimpleNamespace(observe=lambda *a, **kw: None, disabled=True),
            masks_only=SimpleNamespace(observe=lambda *a, **kw: None),
            mask_layer_dropdown=SimpleNamespace(observe=lambda *a, **kw: None),
            mask_color_picker=SimpleNamespace(observe=lambda *a, **kw: None),
            mask_alpha_slider=SimpleNamespace(observe=lambda *a, **kw: None),
            config_save_button=SimpleNamespace(on_click=lambda *a: None),
            config_load_button=SimpleNamespace(on_click=lambda *a: None),
            config_delete_button=SimpleNamespace(on_click=lambda *a: None),
        )

        plugin._connect_events()

        # Verify the ROI list is initially empty
        self.assertEqual(len(plugin.ui_component.roi_selection.options), 0)

        # Add a ROI — the observer should auto-refresh the batch export list
        viewer.roi_manager.add_roi({
            "fov": "FOV_A",
            "marker_set": "panel1",
            "x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10,
        })

        # After add_roi, the observer should have triggered refresh_roi_options
        options = plugin.ui_component.roi_selection.options
        self.assertGreaterEqual(len(options), 1, "ROI list should refresh after add_roi")


class EnsureDataframeFovSanitizationTests(unittest.TestCase):
    """Tests for _ensure_dataframe handling of 'fov' column NaN values."""

    def test_fov_nan_sanitized_to_empty_string(self):
        """When fov is NaN (as read from CSV), it should become ''."""
        import pandas as pd
        from ueler.viewer.roi_manager import _ensure_dataframe, ROI_COLUMNS

        df = pd.DataFrame([{
            "roi_id": "test-123",
            "fov": float("nan"),
            "map_id": "slide-1",
            "x": 0, "y": 0, "width": 100, "height": 100, "zoom": 1.0,
            "x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100,
            "marker_set": "panel1",
            "tags": "", "annotation_palette": "", "mask_color_set": "",
            "mask_visibility": "", "comment": "",
            "created_at": "2025-01-01", "updated_at": "2025-01-01",
        }])

        result = _ensure_dataframe(df)
        fov_value = result.iloc[0]["fov"]
        self.assertEqual(fov_value, "", f"Expected empty string, got {fov_value!r}")

    def test_fov_empty_string_preserved(self):
        """When fov is already '', it should stay ''."""
        import pandas as pd
        from ueler.viewer.roi_manager import _ensure_dataframe

        df = pd.DataFrame([{
            "roi_id": "test-456",
            "fov": "",
            "map_id": "slide-1",
            "x": 0, "y": 0, "width": 100, "height": 100, "zoom": 1.0,
            "x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100,
            "marker_set": "panel1",
            "tags": "", "annotation_palette": "", "mask_color_set": "",
            "mask_visibility": "", "comment": "",
            "created_at": "2025-01-01", "updated_at": "2025-01-01",
        }])

        result = _ensure_dataframe(df)
        self.assertEqual(result.iloc[0]["fov"], "")

    def test_fov_normal_value_preserved(self):
        """When fov has a real value like 'FOV_A', it should stay unchanged."""
        import pandas as pd
        from ueler.viewer.roi_manager import _ensure_dataframe

        df = pd.DataFrame([{
            "roi_id": "test-789",
            "fov": "FOV_A",
            "map_id": "",
            "x": 0, "y": 0, "width": 100, "height": 100, "zoom": 1.0,
            "x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100,
            "marker_set": "panel1",
            "tags": "", "annotation_palette": "", "mask_color_set": "",
            "mask_visibility": "", "comment": "",
            "created_at": "2025-01-01", "updated_at": "2025-01-01",
        }])

        result = _ensure_dataframe(df)
        self.assertEqual(result.iloc[0]["fov"], "FOV_A")

    def test_csv_roundtrip_preserves_empty_fov(self):
        """Write a map-mode ROI to CSV and read it back — fov must remain ''."""
        import pandas as pd
        from ueler.viewer.roi_manager import ROIManager

        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        mgr = ROIManager(tmp.name)

        mgr.add_roi({"fov": "", "map_id": "slide-1", "marker_set": "p1",
                      "x_min": 0, "x_max": 10, "y_min": 0, "y_max": 10})

        # Reload from CSV
        mgr2 = ROIManager(tmp.name)
        df = mgr2.list_rois()
        self.assertEqual(len(df), 1)
        fov_val = df.iloc[0]["fov"]
        self.assertEqual(fov_val, "", f"Expected empty fov after CSV roundtrip, got {fov_val!r}")
        self.assertEqual(df.iloc[0]["map_id"], "slide-1")


if __name__ == "__main__":
    unittest.main()
