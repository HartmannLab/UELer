import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

# Provide a lightweight dask.array stub when the real dependency is unavailable.
if "dask.array" not in sys.modules:
    class FakeDaskArray:
        def __init__(self, data):
            self.data = np.array(data)
            self.dtype = self.data.dtype
            self.shape = self.data.shape
            self.ndim = self.data.ndim

        def __array__(self):
            return self.data

        def __getitem__(self, key):
            result = self.data[key]
            if isinstance(result, np.ndarray):
                return FakeDaskArray(result)
            return result

        def astype(self, dtype):
            return FakeDaskArray(self.data.astype(dtype))

        def compute(self):
            return self.data

        def rechunk(self, *_args, **_kwargs):
            return self

        def __add__(self, other):
            return FakeDaskArray(self.data + np.asarray(other))

        __radd__ = __add__

        def __sub__(self, other):
            return FakeDaskArray(self.data - np.asarray(other))

        def __rsub__(self, other):
            return FakeDaskArray(np.asarray(other) - self.data)

        def __mul__(self, other):
            return FakeDaskArray(self.data * np.asarray(other))

        __rmul__ = __mul__

        def __truediv__(self, other):
            return FakeDaskArray(self.data / np.asarray(other))

        def __rtruediv__(self, other):
            return FakeDaskArray(np.asarray(other) / self.data)

    def _wrap(value):
        return value if isinstance(value, FakeDaskArray) else FakeDaskArray(value)

    dask_array_stub = types.ModuleType("dask.array")
    dask_array_stub.FakeDaskArray = FakeDaskArray
    dask_array_stub.float32 = np.float32

    def from_array(array, chunks=None):  # pylint: disable=unused-argument
        return _wrap(array)

    def zeros(shape, dtype=np.float32):
        return FakeDaskArray(np.zeros(shape, dtype=dtype))

    def clip(array, a_min, a_max):
        data = array.data if isinstance(array, FakeDaskArray) else np.asarray(array)
        return FakeDaskArray(np.clip(data, a_min, a_max))

    def where(condition, x, y):
        return FakeDaskArray(np.where(condition, np.asarray(x), np.asarray(y)))

    def broadcast_to(array, shape):
        return FakeDaskArray(np.broadcast_to(np.asarray(array), shape))

    def stack(seq, axis=0):
        return FakeDaskArray(np.stack([np.asarray(item) for item in seq], axis=axis))

    def nanpercentile(array, q, axis=None):
        data = np.asarray(array)
        return np.nanpercentile(data, q, axis=axis)

    def nanmax(array, axis=None):
        data = np.asarray(array)
        return np.nanmax(data, axis=axis)

    dask_array_stub.from_array = from_array
    dask_array_stub.zeros = zeros
    dask_array_stub.clip = clip
    dask_array_stub.where = where
    dask_array_stub.broadcast_to = broadcast_to
    dask_array_stub.stack = stack
    dask_array_stub.nanpercentile = nanpercentile
    dask_array_stub.nanmax = nanmax

    sys.modules["dask.array"] = dask_array_stub
    dask_module = types.ModuleType("dask")
    dask_module.array = dask_array_stub
    def _delayed(func):  # pragma: no cover - simple passthrough stub
        def _wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return _wrapper

    dask_module.delayed = _delayed
    sys.modules["dask"] = dask_module

import dask.array as da

if "dask_image.imread" not in sys.modules:
    dask_imread_module = types.ModuleType("dask_image.imread")

    def _imread_placeholder(path):  # pylint: disable=unused-argument
        return FakeDaskArray(np.zeros((1, 1), dtype=np.int32))

    dask_imread_module.imread = _imread_placeholder
    dask_image_pkg = types.ModuleType("dask_image")
    dask_image_pkg.imread = dask_imread_module
    sys.modules["dask_image.imread"] = dask_imread_module
    sys.modules["dask_image"] = dask_image_pkg

# Stub heavy optional dependencies so modules import cleanly during tests.
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")

    def _read_csv_placeholder(*_, **__):  # pylint: disable=unused-argument
        return None

    class _TypeAPI:
        @staticmethod
        def is_float_dtype(_value):
            return False

    pandas_stub.read_csv = _read_csv_placeholder
    pandas_stub.api = types.SimpleNamespace(types=_TypeAPI())
    pandas_stub.DataFrame = object
    sys.modules["pandas"] = pandas_stub

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
    for name in [
        "SelectMultiple",
        "Button",
        "Checkbox",
    "Combobox",
        "ColorPicker",
        "Dropdown",
        "FloatSlider",
        "GridBox",
        "HTML",
        "Image",
        "HBox",
        "Box",
        "IntText",
        "Output",
        "Tab",
        "TagsInput",
        "Text",
    "Textarea",
        "ToggleButtons",
        "VBox",
        "Widget",
        "Accordion",
    ]:
        setattr(widgets, name, _Widget)

    sys.modules["ipywidgets"] = widgets

if "IPython.display" not in sys.modules:
    display_module = types.ModuleType("IPython.display")
    display_module.display = lambda *_, **__: None
    display_module.clear_output = lambda *_, **__: None
    sys.modules["IPython.display"] = display_module

if "seaborn_image" not in sys.modules:
    sys.modules["seaborn_image"] = types.ModuleType("seaborn_image")

if "tifffile" not in sys.modules:
    tifffile_stub = types.ModuleType("tifffile")
    tifffile_stub.imwrite = lambda *_, **__: None
    tifffile_stub.imread = lambda *_, **__: np.zeros((1, 1), dtype=np.int32)
    sys.modules["tifffile"] = tifffile_stub

if "skimage.segmentation" not in sys.modules:
    segmentation_module = types.ModuleType("skimage.segmentation")

    def _find_boundaries_placeholder(array, mode="inner"):  # pylint: disable=unused-argument
        data = np.asarray(array)
        boundaries = np.zeros_like(data, dtype=bool)
        boundaries[1:, :] |= data[1:, :] != data[:-1, :]
        boundaries[:-1, :] |= data[1:, :] != data[:-1, :]
        boundaries[:, 1:] |= data[:, 1:] != data[:, :-1]
        boundaries[:, :-1] |= data[:, 1:] != data[:, :-1]
        return boundaries

    segmentation_module.find_boundaries = _find_boundaries_placeholder
    skimage_pkg = types.ModuleType("skimage")
    skimage_pkg.segmentation = segmentation_module
    io_module = types.ModuleType("skimage.io")
    io_module.imread = lambda *_, **__: np.zeros((1, 1), dtype=np.int32)
    io_module.imsave = lambda *_, **__: None
    exposure_module = types.ModuleType("skimage.exposure")
    exposure_module.rescale_intensity = lambda image, *_, **__: image
    transform_module = types.ModuleType("skimage.transform")
    transform_module.resize = lambda image, output_shape, *_, **__: np.zeros(output_shape, dtype=getattr(image, "dtype", np.float32))
    color_module = types.ModuleType("skimage.color")
    color_module.gray2rgb = lambda image: np.stack([np.asarray(image)] * 3, axis=-1)
    measure_module = types.ModuleType("skimage.measure")
    measure_module.label = lambda array: np.asarray(array, dtype=np.int32)
    skimage_pkg.io = io_module
    skimage_pkg.exposure = exposure_module
    skimage_pkg.transform = transform_module
    skimage_pkg.color = color_module
    skimage_pkg.measure = measure_module
    sys.modules["skimage"] = skimage_pkg
    sys.modules["skimage.segmentation"] = segmentation_module
    sys.modules["skimage.io"] = io_module
    sys.modules["skimage.exposure"] = exposure_module
    sys.modules["skimage.transform"] = transform_module
    sys.modules["skimage.color"] = color_module
    sys.modules["skimage.measure"] = measure_module

if "viewer.plugin.chart" not in sys.modules:
    chart_stub = types.ModuleType("viewer.plugin.chart")

    class _ChartDisplay:  # pragma: no cover - stub
        displayed_name = "Chart"

        def __init__(self, *_, **__):
            self.ui = sys.modules["ipywidgets"].VBox()
            canvas = types.SimpleNamespace(draw_idle=lambda: None)
            fig = types.SimpleNamespace(canvas=canvas)
            ax = types.SimpleNamespace(collections=[])
            self.data = types.SimpleNamespace(g=types.SimpleNamespace(ax=ax, fig=fig))

        def after_all_plugins_loaded(self):
            return None

    chart_stub.ChartDisplay = _ChartDisplay
    sys.modules["viewer.plugin.chart"] = chart_stub

if "viewer.plugin.cell_gallery" not in sys.modules:
    cell_stub = types.ModuleType("viewer.plugin.cell_gallery")

    class _CellGalleryDisplay:  # pragma: no cover - stub
        displayed_name = "Gallery"

        def __init__(self, *_, **__):
            self.ui = sys.modules["ipywidgets"].VBox()

    cell_stub.CellGalleryDisplay = _CellGalleryDisplay
    sys.modules["viewer.plugin.cell_gallery"] = cell_stub

if "viewer.plugin.heatmap" not in sys.modules:
    heatmap_stub = types.ModuleType("viewer.plugin.heatmap")

    class _HeatmapDisplay:  # pragma: no cover - stub
        displayed_name = "Heatmap"

        def __init__(self, *_, **__):
            self.ui = sys.modules["ipywidgets"].VBox()

    heatmap_stub.HeatmapDisplay = _HeatmapDisplay
    sys.modules["viewer.plugin.heatmap"] = heatmap_stub

if "viewer.annotation_display" not in sys.modules:
    annotation_stub = types.ModuleType("viewer.annotation_display")

    class _AnnotationDisplay:  # pragma: no cover - stub
        def __init__(self, *_, **__):
            self.ui = sys.modules["ipywidgets"].VBox()

    annotation_stub.AnnotationDisplay = _AnnotationDisplay
    sys.modules["viewer.annotation_display"] = annotation_stub

from data_loader import load_annotations_for_fov
from viewer.color_palettes import DEFAULT_COLOR, build_discrete_colormap
from viewer.main_viewer import ImageMaskViewer, _unique_annotation_values
from skimage.segmentation import find_boundaries


class AnnotationLoaderTests(unittest.TestCase):
    def test_load_annotations_sets_names_and_dtype(self):
        base_array = np.array([[0, 1], [2, 2]], dtype=np.int32)

        with TemporaryDirectory() as tmp_dir:
            annotations_dir = Path(tmp_dir) / "annotations"
            annotations_dir.mkdir()
            file_path = annotations_dir / "FOV1_example.tif"
            file_path.write_bytes(b"test")

            annotation_names = set()
            with patch("data_loader.imread", return_value=da.from_array(base_array, chunks=base_array.shape)):
                loaded = load_annotations_for_fov("FOV1", str(annotations_dir), annotation_names)

        self.assertIn("example", loaded)
        annotation = loaded["example"]
        self.assertTrue(np.issubdtype(annotation.dtype, np.integer))
        self.assertEqual(annotation.shape, base_array.shape)
        self.assertIn("example", annotation_names)


class PaletteHelperTests(unittest.TestCase):
    def test_build_discrete_colormap_respects_palette(self):
        class_ids = [0, 1, 5]
        palette = {"0": "#000000", "5": "#FFFFFF"}
        colormap = build_discrete_colormap(class_ids, palette)

        self.assertEqual(colormap.shape, (6, 3))
        self.assertTrue(np.allclose(colormap[0], np.array([0.0, 0.0, 0.0], dtype=np.float32)))
        default_rgb = np.array([0xA0 / 255.0] * 3, dtype=np.float32)
        self.assertTrue(np.allclose(colormap[1], default_rgb))
        self.assertTrue(np.allclose(colormap[5], np.array([1.0, 1.0, 1.0], dtype=np.float32)))


class AnnotationUtilityTests(unittest.TestCase):
    def test_unique_annotation_values_handles_numpy_arrays(self):
        array = np.array([[0, 1], [3, 2]], dtype=np.int32)
        values = _unique_annotation_values(array)
        self.assertTrue(np.array_equal(values, np.array([0, 1, 2, 3], dtype=np.int32)))


class AnnotationControlStateTests(unittest.TestCase):
    class _DummyWidget:
        def __init__(self, value=None):
            self.value = value
            self.options = []
            self.disabled = False

    def _build_viewer(self):
        viewer = ImageMaskViewer.__new__(ImageMaskViewer)
        viewer.annotation_display_enabled = True
        viewer.annotation_overlay_mode = "combined"
        viewer.annotation_overlay_alpha = 0.5
        viewer.annotation_label_display_mode = "id"
        viewer.active_annotation_name = None
        viewer.annotation_palette_editor = types.SimpleNamespace(hide=lambda: None)

        selector = self._DummyWidget()
        display_checkbox = self._DummyWidget(value=True)
        overlay_mode = self._DummyWidget(value="combined")
        alpha_slider = self._DummyWidget(value=0.5)
        label_mode = self._DummyWidget(value="id")
        edit_button = self._DummyWidget()
        edit_button.disabled = True

        viewer.ui_component = types.SimpleNamespace(
            annotation_selector=selector,
            annotation_display_checkbox=display_checkbox,
            annotation_overlay_mode=overlay_mode,
            annotation_alpha_slider=alpha_slider,
            annotation_label_mode=label_mode,
            annotation_edit_button=edit_button,
        )
        return viewer, selector, edit_button

    def test_refresh_enables_edit_button_for_spaced_annotation(self):
        viewer, selector, edit_button = self._build_viewer()
        selector.options = [("Simple Segmentation", "Simple Segmentation")]
        selector.value = None

        viewer._refresh_annotation_control_states()

        self.assertEqual(selector.value, "Simple Segmentation")
        self.assertEqual(viewer.active_annotation_name, "Simple Segmentation")
        self.assertFalse(edit_button.disabled)


class AnnotationLayoutTests(unittest.TestCase):
    def _make_viewer(self):
        widgets = sys.modules["ipywidgets"]

        viewer = ImageMaskViewer.__new__(ImageMaskViewer)
        viewer.predefined_colors = {"Red": "#FF0000", "White": "#FFFFFF"}
        viewer.current_downsample_factor = 1
        viewer.annotation_display_enabled = False
        viewer.annotation_overlay_mode = "combined"
        viewer.annotation_overlay_alpha = 0.5
        viewer.annotation_label_display_mode = "id"
        viewer.active_annotation_name = None
        viewer.annotation_palette_editor = types.SimpleNamespace(
            hide=lambda: None,
            layout=types.SimpleNamespace(display="none")
        )
        viewer._control_section_titles = []
        viewer.annotations_available = True
        viewer.masks_available = True
        viewer.mask_names = ["MaskA"]
        viewer.mask_cache = {}
        viewer.label_masks_cache = {}
        viewer.predefined_colors = {"Red": "#FF0000", "White": "#FFFFFF"}

        viewer.ui_component = types.SimpleNamespace()
        viewer.ui_component.channel_selector = types.SimpleNamespace(
            value=("Ch1",), observe=lambda *args, **kwargs: None
        )
        viewer.ui_component.color_controls = {}
        viewer.ui_component.contrast_min_controls = {}
        viewer.ui_component.contrast_max_controls = {}
        viewer.ui_component.mask_display_controls = {}
        viewer.ui_component.mask_color_controls = {}
        viewer.ui_component.channel_controls_box = widgets.VBox()
        viewer.ui_component.mask_controls_box = widgets.VBox()
        viewer.ui_component.annotation_controls_box = widgets.VBox()
        viewer.ui_component.no_channels_label = widgets.HTML()
        viewer.ui_component.no_masks_label = widgets.HTML()
        viewer.ui_component.no_annotations_label = widgets.HTML()
        viewer.ui_component.empty_controls_placeholder = widgets.HTML()
        viewer.ui_component.annotation_controls_header = widgets.HTML()
        viewer.ui_component.annotation_display_checkbox = widgets.Checkbox(value=False, disabled=False)
        viewer.ui_component.annotation_selector = widgets.Dropdown(
            options=[("Simple Segmentation", "Simple Segmentation")],
            value="Simple Segmentation"
        )
        viewer.ui_component.annotation_overlay_mode = widgets.ToggleButtons(
            options=[("Mask outlines", "mask"), ("Fill", "annotation"), ("Both", "combined")],
            value="combined"
        )
        viewer.ui_component.annotation_alpha_slider = widgets.FloatSlider(value=0.5)
        viewer.ui_component.annotation_label_mode = widgets.Dropdown(
            options=[("Class IDs", "id"), ("Labels", "label")],
            value="id"
        )
        viewer.ui_component.annotation_edit_button = widgets.Button(disabled=True)

        accordion = widgets.Accordion()
        accordion.children = tuple()
        accordion.selected_index = 0
        accordion.observe = lambda *args, **kwargs: None
        accordion.set_title = lambda idx, title: None
        viewer.ui_component.control_sections = accordion
        viewer.ui_component.annotation_editor_host = widgets.VBox()

        viewer._get_channel_stats = lambda ch: (1.0, 1.0)
        viewer._calculate_slider_step = lambda max_val: 0.1
        viewer._slider_readout_format = lambda max_val: ".2f"
        viewer.update_display = lambda *_args, **_kwargs: None
        viewer._refresh_annotation_control_states = ImageMaskViewer._refresh_annotation_control_states.__get__(viewer)

        return viewer

    def test_annotations_section_precedes_masks(self):
        viewer = self._make_viewer()

        viewer.update_controls(None)

        sections = viewer.ui_component.control_sections.children
        self.assertEqual(len(sections), 3)
        self.assertIs(sections[0], viewer.ui_component.channel_controls_box)
        self.assertIs(sections[1], viewer.ui_component.annotation_controls_box)
        self.assertIs(sections[2], viewer.ui_component.mask_controls_box)


class AnnotationMetadataTests(unittest.TestCase):
    def test_ensure_metadata_populates_ids_and_palette_for_numpy_arrays(self):
        viewer = ImageMaskViewer.__new__(ImageMaskViewer)
        base_array = np.array([[0, 1], [3, 2]], dtype=np.int32)
        viewer.ui_component = types.SimpleNamespace(image_selector=types.SimpleNamespace(value="FOV1"))
        viewer.annotation_cache = {"FOV1": {"ann": base_array}}
        viewer.annotation_class_ids = {}
        viewer.annotation_palettes = {}
        viewer.annotation_class_labels = {}
        viewer.annotation_palette_editor = types.SimpleNamespace(hide=lambda: None)

        viewer._ensure_annotation_metadata("ann")

        self.assertEqual(viewer.annotation_class_ids.get("ann"), [0, 1, 2, 3])
        palette = viewer.annotation_palettes.get("ann", {})
        self.assertEqual(sorted(palette.keys()), ["0", "1", "2", "3"])
        labels = viewer.annotation_class_labels.get("ann", {})
        self.assertEqual(sorted(labels.keys()), ["0", "1", "2", "3"])


class RenderImageAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.viewer = ImageMaskViewer.__new__(ImageMaskViewer)
        self.viewer.predefined_colors = {"Red": "#FF0000", "White": "#FFFFFF"}
        self.viewer.annotation_display_enabled = True
        self.viewer.annotation_overlay_mode = "annotation"
        self.viewer.annotation_overlay_alpha = 0.5
        self.viewer.active_annotation_name = "ann"
        self.viewer.annotations_available = True
        self.viewer.masks_available = False
        self.viewer.annotation_class_ids = {"ann": [0, 1]}
        self.viewer.annotation_class_labels = {"ann": {"0": "0", "1": "1"}}
        self.viewer.annotation_palettes = {"ann": {"0": "#000000", "1": "#FFFFFF"}}
        self.viewer.annotation_label_cache = {
            "FOV1": {
                "ann": {
                    1: da.from_array(
                        np.array([[0, 1], [1, 0]], dtype=np.int32),
                        chunks=(2, 2),
                    )
                }
            }
        }
        self.viewer.load_fov = lambda *_, **__: None
        self.viewer.image_cache = {
            "FOV1": {
                "chan": da.from_array(np.zeros((2, 2), dtype=np.float32), chunks=(2, 2))
            }
        }
        self.viewer.mask_cache = {"FOV1": {}}
        self.viewer.label_masks_cache = {"FOV1": {}}
        self.viewer.edge_masks_cache = {"FOV1": {}}
        self.viewer.ui_component = types.SimpleNamespace(
            image_selector=types.SimpleNamespace(value="FOV1"),
            color_controls={"chan": types.SimpleNamespace(value="Red")},
            contrast_min_controls={"chan": types.SimpleNamespace(value=0.0)},
            contrast_max_controls={"chan": types.SimpleNamespace(value=1.0)},
            mask_display_controls={},
            mask_color_controls={},
        )

    def test_annotation_fill_mode_blends_colors(self):
        result = self.viewer.render_image(
            ("chan",),
            downsample_factor=1,
            xym=(0, 2, 0, 2),
            xym_ds=(0, 2, 0, 2),
        )
        self.assertEqual(result.shape, (2, 2, 3))
        self.assertTrue(np.allclose(result[0, 0], [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(result[0, 1], [0.5, 0.5, 0.5]))
        self.assertTrue(np.allclose(result[1, 0], [0.5, 0.5, 0.5]))

    def test_mask_mode_skips_annotation_fill(self):
        self.viewer.annotation_overlay_mode = "mask"
        result = self.viewer.render_image(
            ("chan",),
            downsample_factor=1,
            xym=(0, 2, 0, 2),
            xym_ds=(0, 2, 0, 2),
        )
        self.assertTrue(np.allclose(result, np.zeros((2, 2, 3), dtype=np.float32)))

    def test_combined_mode_without_masks_still_blends_fill(self):
        self.viewer.annotation_overlay_mode = "combined"
        self.viewer.masks_available = False
        result = self.viewer.render_image(
            ("chan",),
            downsample_factor=1,
            xym=(0, 2, 0, 2),
            xym_ds=(0, 2, 0, 2),
        )
        self.assertTrue(np.any(result > 0.0))

    def test_combined_mode_adds_mask_edges(self):
        mask_array = np.array([[0, 1], [1, 1]], dtype=np.int32)
        self.viewer.annotation_overlay_mode = "combined"
        self.viewer.masks_available = True
        self.viewer.ui_component.mask_display_controls = {
            "mask": types.SimpleNamespace(value=True)
        }
        self.viewer.ui_component.mask_color_controls = {
            "mask": types.SimpleNamespace(value="White")
        }
        self.viewer.mask_cache = {"FOV1": {"mask": da.from_array(mask_array, chunks=(2, 2))}}
        self.viewer.label_masks_cache = {
            "FOV1": {"mask": {1: da.from_array(mask_array, chunks=(2, 2))}}
        }

        result = self.viewer.render_image(
            ("chan",),
            downsample_factor=1,
            xym=(0, 2, 0, 2),
            xym_ds=(0, 2, 0, 2),
        )

        edge_mask = find_boundaries(mask_array, mode="inner")
        self.assertTrue(np.allclose(result[edge_mask], np.ones((edge_mask.sum(), 3))))
        base_fill = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)
        expected_after_edges = base_fill.copy()
        expected_after_edges[edge_mask] = 1.0
        self.assertTrue(np.allclose(result[..., 0], expected_after_edges))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
