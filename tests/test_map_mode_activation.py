import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import tests.bootstrap  # noqa: F401

# Ensure matplotlib.colors is available before importing the viewer module.
import sys
import types

if "matplotlib.colors" not in sys.modules:  # pragma: no cover - test bootstrap safeguard
    colors_stub = types.ModuleType("matplotlib.colors")

    def _to_rgb(_value):
        return (0.0, 0.0, 0.0)

    colors_stub.to_rgb = _to_rgb  # type: ignore[attr-defined]
    sys.modules["matplotlib.colors"] = colors_stub
    matplotlib_module = sys.modules.get("matplotlib")
    if matplotlib_module is None:
        matplotlib_module = types.ModuleType("matplotlib")
        sys.modules["matplotlib"] = matplotlib_module
    setattr(matplotlib_module, "colors", colors_stub)

if "matplotlib.font_manager" not in sys.modules:  # pragma: no cover - optional dependency stub
    font_manager_stub = types.ModuleType("matplotlib.font_manager")
    font_manager_stub.FontProperties = object  # type: ignore[attr-defined]
    sys.modules["matplotlib.font_manager"] = font_manager_stub
    matplotlib_module = sys.modules.get("matplotlib")
    if matplotlib_module is None:
        matplotlib_module = types.ModuleType("matplotlib")
        sys.modules["matplotlib"] = matplotlib_module
    setattr(matplotlib_module, "font_manager", font_manager_stub)

display_module = sys.modules.get("IPython.display")
if display_module is None:  # pragma: no cover - optional dependency stub
    display_module = types.ModuleType("IPython.display")
    sys.modules["IPython.display"] = display_module

if not hasattr(display_module, "HTML"):
    display_module.HTML = lambda *args, **kwargs: None  # type: ignore[attr-defined]
if not hasattr(display_module, "Javascript"):
    display_module.Javascript = lambda *args, **kwargs: None  # type: ignore[attr-defined]
if not hasattr(display_module, "display"):
    display_module.display = lambda *args, **kwargs: None  # type: ignore[attr-defined]

ipython_module = sys.modules.get("IPython")
if ipython_module is None:
    ipython_module = types.ModuleType("IPython")
    sys.modules["IPython"] = ipython_module
setattr(ipython_module, "display", display_module)

if "cv2" not in sys.modules:  # pragma: no cover - optional dependency stub
    sys.modules["cv2"] = types.ModuleType("cv2")

if "dask" not in sys.modules:  # pragma: no cover - optional dependency stub
    dask_stub = types.ModuleType("dask")

    def _delayed(func):
        return func

    dask_stub.delayed = _delayed  # type: ignore[attr-defined]
    sys.modules["dask"] = dask_stub

if "tifffile" not in sys.modules:  # pragma: no cover - optional dependency stub
    class _FakeTiffFile:
        def __init__(self, *_args, **_kwargs):
            self.pages = [SimpleNamespace(shape=(1, 1), asarray=lambda: np.zeros((1, 1)))]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    tifffile_stub = types.ModuleType("tifffile")
    tifffile_stub.TiffFile = _FakeTiffFile  # type: ignore[attr-defined]
    tifffile_stub.TiffPage = SimpleNamespace  # type: ignore[attr-defined]
    sys.modules["tifffile"] = tifffile_stub

if "skimage" not in sys.modules:  # pragma: no cover - optional dependency stub
    skimage_stub = types.ModuleType("skimage")
    sys.modules["skimage"] = skimage_stub

    segmentation_stub = types.ModuleType("skimage.segmentation")

    def _find_boundaries(array, mode=None):  # noqa: D401
        return np.zeros_like(array, dtype=bool)

    segmentation_stub.find_boundaries = _find_boundaries  # type: ignore[attr-defined]
    sys.modules["skimage.segmentation"] = segmentation_stub
    setattr(skimage_stub, "segmentation", segmentation_stub)

    io_stub = types.ModuleType("skimage.io")
    io_stub.imread = lambda *args, **kwargs: np.zeros((1, 1))  # type: ignore[attr-defined]
    io_stub.imsave = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["skimage.io"] = io_stub
    setattr(skimage_stub, "io", io_stub)

    transform_stub = types.ModuleType("skimage.transform")
    transform_stub.resize = lambda *args, **kwargs: np.zeros((1, 1))  # type: ignore[attr-defined]
    sys.modules["skimage.transform"] = transform_stub
    setattr(skimage_stub, "transform", transform_stub)

    exposure_stub = types.ModuleType("skimage.exposure")
    exposure_stub.adjust_gamma = lambda image, gamma, gain=1.0: image  # type: ignore[attr-defined]
    sys.modules["skimage.exposure"] = exposure_stub
    setattr(skimage_stub, "exposure", exposure_stub)

    color_stub = types.ModuleType("skimage.color")
    color_stub.rgb2gray = lambda *args, **kwargs: np.zeros((1, 1))  # type: ignore[attr-defined]
    sys.modules["skimage.color"] = color_stub
    setattr(skimage_stub, "color", color_stub)

    measure_stub = types.ModuleType("skimage.measure")
    measure_stub.regionprops = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    sys.modules["skimage.measure"] = measure_stub
    setattr(skimage_stub, "measure", measure_stub)

if "seaborn_image" not in sys.modules:  # pragma: no cover - optional dependency stub
    sys.modules["seaborn_image"] = types.ModuleType("seaborn_image")

from ueler.viewer.main_viewer import ImageMaskViewer


class _DummyAxes:
    def __init__(self):
        self._xlim = (0.0, 1.0)
        self._ylim = (0.0, 1.0)

    def set_xlim(self, start, end):
        self._xlim = (start, end)

    def get_xlim(self):
        return self._xlim

    def set_ylim(self, start, end):
        self._ylim = (start, end)

    def get_ylim(self):
        return self._ylim


class _DummyImage:
    def __init__(self):
        self.extent = None
        self.data = None

    def set_extent(self, extent):
        self.extent = extent

    def set_data(self, data):
        self.data = data


class _FakeImageDisplay:
    def __init__(self):
        self.width = 1
        self.height = 1
        self.ax = _DummyAxes()
        self.img_display = _DummyImage()
        self.fig = SimpleNamespace(canvas=SimpleNamespace(draw_idle=lambda: None, toolbar=None))
        self.combined = None


class _DummySelector:
    def __init__(self, value=None):
        self.value = value
        self.disabled = False


class _DummyNavStack:
    def __init__(self):
        self._elements = []


class _DummyToolbar:
    def __init__(self, axes):
        self._axes = axes
        self._nav_stack = _DummyNavStack()

    def push_current(self):
        view = {"xlim": self._axes.get_xlim(), "ylim": self._axes.get_ylim()}
        self._nav_stack._elements.append({self._axes: (view, ())})


class _StubLayer:
    def __init__(self, base_pixel_um, bounds_um):
        self._base_pixel_um = float(base_pixel_um)
        self._bounds_um = bounds_um

    def base_pixel_size_um(self):
        return self._base_pixel_um

    def map_bounds(self):
        return self._bounds_um


class _CaptureLayer(_StubLayer):
    def __init__(self, base_pixel_um, bounds_um):
        super().__init__(base_pixel_um, bounds_um)
        self.viewport_args = None
        self.render_invocations = 0

    def set_viewport(self, xmin_um, xmax_um, ymin_um, ymax_um, *, downsample_factor):
        self.viewport_args = (xmin_um, xmax_um, ymin_um, ymax_um, downsample_factor)

    def render(self, selected_channels):
        self.render_invocations += 1
        return np.ones((1, 1, 3), dtype=np.float32)

    def last_visible_fovs(self):
        return ("FOV_A",)


class MapModeActivationTests(unittest.TestCase):
    def setUp(self):
        self.viewer = ImageMaskViewer.__new__(ImageMaskViewer)
        self.viewer.image_display = _FakeImageDisplay()
        self.viewer.downsample_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.viewer._map_tile_cache = OrderedDict()
        self.viewer._map_tile_cache_capacity = 6
        self.viewer._map_mode_enabled = True
        self.viewer._map_descriptors = {}
        self.viewer._map_layers = {}
        self.viewer._map_mode_active = False
        self.viewer._map_mode_messages = []
        self.viewer._map_pixel_size_nm = None
        self.viewer._map_canvas_size = None
        self.viewer._visible_map_fovs = ()
        self.viewer._active_map_id = None
        self.viewer.current_downsample_factor = 8
        self.viewer.width = 1
        self.viewer.height = 1
        self.viewer.ui_component = SimpleNamespace(
            map_selector=_DummySelector(),
            image_selector=_DummySelector(value="FOV_A"),
        )
        self.viewer.ui_component.channel_selector = _DummySelector(value=("CD3",))
        self.viewer.ui_component.mask_display_controls = {}
        self.viewer.ui_component.mask_color_controls = {}

    def test_set_map_canvas_dimensions_uses_placeholder(self):
        self.viewer._set_map_canvas_dimensions(36_706, 115_751)

        placeholder = self.viewer.image_display.img_display.data
        self.assertIsInstance(placeholder, np.ndarray)
        self.assertEqual(placeholder.shape, (1, 1, 3))
        self.assertEqual(placeholder.dtype, np.float32)
        self.assertEqual(self.viewer.image_display.combined.shape, (1, 1, 3))
        self.assertEqual(self.viewer.width, 36_706)
        self.assertEqual(self.viewer.height, 115_751)
        self.assertEqual(
            self.viewer.image_display.img_display.extent,
            (0, 36_706, 115_751, 0),
        )

    def test_set_map_canvas_dimensions_syncs_toolbar_home_view(self):
        canvas = self.viewer.image_display.fig.canvas
        toolbar = _DummyToolbar(self.viewer.image_display.ax)
        toolbar.push_current()
        canvas.toolbar = toolbar

        stored_view, stored_bboxes = toolbar._nav_stack._elements[0][self.viewer.image_display.ax]
        stored_view["token"] = "keep"

        self.viewer._set_map_canvas_dimensions(36_706, 115_751)

        updated_view, updated_bboxes = toolbar._nav_stack._elements[0][self.viewer.image_display.ax]
        self.assertEqual(updated_view["xlim"], (0, 36_706))
        self.assertEqual(updated_view["ylim"], (115_751, 0))
        self.assertEqual(updated_view["token"], "keep")
        self.assertIs(updated_bboxes, stored_bboxes)

    def test_set_map_canvas_dimensions_seeds_toolbar_when_empty(self):
        canvas = self.viewer.image_display.fig.canvas
        canvas.toolbar = _DummyToolbar(self.viewer.image_display.ax)

        self.viewer._set_map_canvas_dimensions(512, 256)

        elements = canvas.toolbar._nav_stack._elements
        self.assertTrue(elements)
        updated_view, updated_bboxes = elements[0][self.viewer.image_display.ax]
        self.assertEqual(updated_view["xlim"], (0, 512))
        self.assertEqual(updated_view["ylim"], (256, 0))
        self.assertEqual(updated_bboxes, ())

    def test_activate_map_mode_updates_downsample_factor(self):
        layer = _StubLayer(0.5, (0.0, 1_024.0, 0.0, 2_048.0))
        self.viewer._map_descriptors = {"slide-1": object()}

        with patch.object(ImageMaskViewer, "_get_map_layer", lambda self, map_id: layer):
            self.viewer._activate_map_mode("slide-1")

        self.assertTrue(self.viewer._map_mode_active)
        self.assertEqual(self.viewer._map_canvas_size, (2_048, 4_096))
        self.assertEqual(self.viewer._map_pixel_size_nm, 500.0)
        self.assertEqual(self.viewer.current_downsample_factor, 8)
        self.assertTrue(self.viewer.ui_component.image_selector.disabled)
        self.assertFalse(self.viewer.ui_component.map_selector.disabled)
        self.assertEqual(self.viewer.ui_component.map_selector.value, "slide-1")

    def test_render_map_view_offsets_descriptor_bounds(self):
        layer = _CaptureLayer(0.5, (1_000.0, 2_000.0, 4_000.0, 6_000.0))
        self.viewer._map_descriptors = {"slide-1": object()}

        with patch.object(ImageMaskViewer, "_get_map_layer", lambda self, map_id: layer):
            self.viewer._active_map_id = "slide-1"
            combined, visible = self.viewer._render_map_view(("CD3",), 1, (0, 512, 0, 256))

        self.assertIsNotNone(layer.viewport_args)
        xmin_um, xmax_um, ymin_um, ymax_um, downsample = layer.viewport_args
        self.assertEqual(downsample, 1)
        self.assertAlmostEqual(xmin_um, 1_000.0)
        self.assertAlmostEqual(xmax_um, 1_000.0 + 512 * 0.5)
        self.assertAlmostEqual(ymin_um, 4_000.0)
        self.assertAlmostEqual(ymax_um, 4_000.0 + 256 * 0.5)
        self.assertEqual(layer.render_invocations, 1)
        np.testing.assert_allclose(combined, np.ones((1, 1, 3), dtype=np.float32))
        self.assertEqual(visible, ("FOV_A",))

    def test_update_display_preserves_positive_viewport(self):
        captured = {}

        def _capture_viewport(selected_channels, downsample_factor, viewport_pixels):
            captured["viewport"] = viewport_pixels
            return np.zeros((1, 1, 3), dtype=np.float32), ()

        self.viewer._render_map_view = _capture_viewport  # type: ignore[attr-defined]
        self.viewer._map_mode_active = True
        self.viewer._active_map_id = "slide-1"
        self.viewer.width = 4096
        self.viewer.height = 2048
        axes = self.viewer.image_display.ax
        axes.set_xlim(128.0, 128.75)
        axes.set_ylim(64.5, 63.8)

        self.viewer.update_display(downsample_factor=512)

        viewport = captured.get("viewport")
        self.assertIsNotNone(viewport)
        xmin, xmax, ymin, ymax = viewport
        self.assertLess(xmin, xmax)
        self.assertLess(ymin, ymax)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
