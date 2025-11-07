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
    "ToggleButtons",
    "VBox",
):
    _ensure_widget(widget_name)

from ueler.viewer.main_viewer import ImageMaskViewer
from ueler.viewer.plugin.export_fovs import BatchExportPlugin
from ueler.rendering import OverlaySnapshot


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
        self.ui_component = SimpleNamespace(
            mask_display_controls={"MASK": SimpleNamespace(value=True)},
            mask_color_controls={"MASK": SimpleNamespace(value="Red")},
            image_selector=SimpleNamespace(value=None),
        )

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

            def _build_layout(self):  # pragma: no cover - skipped in tests
                return

            def _connect_events(self):  # pragma: no cover - manually wire slider observer
                self.ui_component.mask_outline_thickness.observe(
                    self._on_mask_outline_thickness_change,
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

        snapshot = viewer.capture_overlay_snapshot(include_annotations=True, include_masks=True)
        self.assertTrue(snapshot.include_annotations)
        self.assertTrue(snapshot.include_masks)
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


if __name__ == "__main__":
    unittest.main()
