import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


if "matplotlib" not in sys.modules:
    sys.modules["matplotlib"] = types.ModuleType("matplotlib")

if "matplotlib.pyplot" not in sys.modules:
    plt_module = types.ModuleType("matplotlib.pyplot")
    plt_module.show = lambda *args, **kwargs: None
    sys.modules["matplotlib.pyplot"] = plt_module

if "matplotlib.colors" not in sys.modules:
    colors_module = types.ModuleType("matplotlib.colors")
    colors_module.to_rgb = lambda value: value
    sys.modules["matplotlib.colors"] = colors_module

if "matplotlib.text" not in sys.modules:
    text_module = types.ModuleType("matplotlib.text")

    class _Annotation:  # pragma: no cover - lightweight stub
        pass

    text_module.Annotation = _Annotation
    sys.modules["matplotlib.text"] = text_module

if "matplotlib.font_manager" not in sys.modules:
    fm_module = types.ModuleType("matplotlib.font_manager")

    class _FontProperties:  # pragma: no cover - stub
        pass

    fm_module.FontProperties = _FontProperties
    sys.modules["matplotlib.font_manager"] = fm_module

if "matplotlib.patches" not in sys.modules:
    patches_module = types.ModuleType("matplotlib.patches")

    class _Polygon:  # pragma: no cover - stub
        def __init__(self, *_args, **_kwargs):
            pass  # lightweight placeholder for matplotlib polygon geometry

    patches_module.Polygon = _Polygon
    sys.modules["matplotlib.patches"] = patches_module

if "matplotlib.widgets" not in sys.modules:
    widgets_module = types.ModuleType("matplotlib.widgets")

    class _RectangleSelector:  # pragma: no cover - stub
        def __init__(self, *_args, **_kwargs):
            pass  # stub widget constructor used only for import-time wiring

    widgets_module.RectangleSelector = _RectangleSelector
    sys.modules["matplotlib.widgets"] = widgets_module

if "matplotlib.backend_bases" not in sys.modules:
    backend_module = types.ModuleType("matplotlib.backend_bases")

    class _MouseButton:  # pragma: no cover - stub
        LEFT = 1
        RIGHT = 3

    backend_module.MouseButton = _MouseButton
    sys.modules["matplotlib.backend_bases"] = backend_module

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

if "skimage" not in sys.modules:
    skimage_pkg = types.ModuleType("skimage")
    skimage_pkg.__path__ = []  # pragma: no cover - package marker
    sys.modules["skimage"] = skimage_pkg

if "skimage.segmentation" not in sys.modules:
    segmentation_module = types.ModuleType("skimage.segmentation")

    def _find_boundaries(*_args, **_kwargs):  # pragma: no cover - stub
        return None

    segmentation_module.find_boundaries = _find_boundaries
    sys.modules["skimage.segmentation"] = segmentation_module

if "skimage.io" not in sys.modules:
    io_module = types.ModuleType("skimage.io")

    def _imread(*_args, **_kwargs):  # pragma: no cover - stub
        return None

    def _imsave(*_args, **_kwargs):  # pragma: no cover - stub
        return None

    io_module.imread = _imread
    io_module.imsave = _imsave
    sys.modules["skimage.io"] = io_module

if "skimage.exposure" not in sys.modules:
    exposure_module = types.ModuleType("skimage.exposure")

    def _rescale_intensity(*_args, **_kwargs):  # pragma: no cover - stub
        return None

    exposure_module.rescale_intensity = _rescale_intensity
    sys.modules["skimage.exposure"] = exposure_module

if "skimage.transform" not in sys.modules:
    transform_module = types.ModuleType("skimage.transform")

    def _resize(image, *_args, **_kwargs):  # pragma: no cover - stub
        return image

    transform_module.resize = _resize
    sys.modules["skimage.transform"] = transform_module

if "skimage.color" not in sys.modules:
    color_module = types.ModuleType("skimage.color")

    def _gray2rgb(image):  # pragma: no cover - stub
        return image

    color_module.gray2rgb = _gray2rgb
    sys.modules["skimage.color"] = color_module

if "skimage.measure" not in sys.modules:
    measure_module = types.ModuleType("skimage.measure")

    def _regionprops(*_args, **_kwargs):  # pragma: no cover - stub
        return []

    measure_module.regionprops = _regionprops
    sys.modules["skimage.measure"] = measure_module

if "dask" not in sys.modules:
    dask_module = types.ModuleType("dask")

    def _delayed(func):  # pragma: no cover - stub
        return func

    dask_module.delayed = _delayed
    sys.modules["dask"] = dask_module

if "seaborn_image" not in sys.modules:
    sys.modules["seaborn_image"] = types.ModuleType("seaborn_image")

if "tifffile" not in sys.modules:
    tifffile_module = types.ModuleType("tifffile")

    def _imwrite(*_args, **_kwargs):  # pragma: no cover - stub
        return None

    tifffile_module.imwrite = _imwrite
    sys.modules["tifffile"] = tifffile_module


from ueler.viewer.main_viewer import ImageMaskViewer
from ueler.viewer.ui_components import uicomponents


class _DummyViewer:
    def __init__(self):
        self.available_fovs = ['FOV_A']
        self._status_image = {'processing': b''}

    def __getattr__(self, name):
        mock = MagicMock(name=name)
        setattr(self, name, mock)
        return mock


class TestCacheSettings(unittest.TestCase):
    def test_cache_size_widget_defaults_to_100_in_advanced_settings(self) -> None:
        viewer = _DummyViewer()
        components = uicomponents(viewer)

        self.assertEqual(components.cache_size_input.value, 100)

        advanced_tab = components.advanced_settings_tabs.children[1]
        self.assertIn(components.cache_size_input, tuple(advanced_tab.children))

    def test_image_mask_viewer_defaults_cache_size_to_100(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            for idx in range(2):
                fov_dir = base_dir / f"FOV_{idx}"
                fov_dir.mkdir(parents=True, exist_ok=True)
                (fov_dir / 'dummy.tiff').write_bytes(b'0')

            def fake_load_fov(self, fov_name, requested_channels=None):
                if fov_name not in self.image_cache:
                    self.image_cache[fov_name] = {'channel': np.zeros((4, 4), dtype=np.uint8)}

            class _StubWidget:
                def __init__(self, value=None):
                    self.value = value
                    self.layout = SimpleNamespace(display='')

                def observe(self, *args, **kwargs):
                    return None

            class _StubSlider(_StubWidget):
                pass

            class _StubHost:
                def __init__(self):
                    self.children = ()
                    self.layout = SimpleNamespace(display='none')

            class _StubUI:
                def __init__(self):
                    self.cache_size_input = _StubWidget(100)
                    self.pixel_size_inttext = _StubWidget(390)
                    self.enable_downsample_checkbox = _StubWidget(True)
                    self.mask_outline_thickness_slider = _StubSlider(1)
                    self.annotation_editor_host = _StubHost()

            def fake_create_widgets(viewer):
                viewer.ui_component = _StubUI()

            with patch.object(ImageMaskViewer, 'load_status_images', lambda self: None), \
                patch.object(ImageMaskViewer, 'load_fov', fake_load_fov), \
                patch.object(ImageMaskViewer, '_initialize_annotation_palette_manager', lambda self: None), \
                patch.object(ImageMaskViewer, '_refresh_annotation_control_states', lambda self: None), \
                patch.object(ImageMaskViewer, 'setup_widget_observers', lambda self: None), \
                patch.object(ImageMaskViewer, 'setup_event_connections', lambda self: None), \
                patch.object(ImageMaskViewer, 'update_marker_set_dropdown', lambda self: None), \
                patch.object(ImageMaskViewer, 'update_controls', lambda self, *_args, **_kwargs: None), \
                patch.object(ImageMaskViewer, 'on_image_change', lambda self, *_args, **_kwargs: None), \
                patch.object(ImageMaskViewer, 'update_display', lambda self, *_args, **_kwargs: None), \
                patch.object(ImageMaskViewer, 'load_widget_states', lambda self, *_args, **_kwargs: None), \
                patch.object(ImageMaskViewer, 'setup_attr_observers', lambda self: None), \
                patch('ueler.viewer.main_viewer.create_widgets', fake_create_widgets), \
                patch('ueler.viewer.main_viewer.ROIManager', lambda *_args, **_kwargs: SimpleNamespace()), \
                patch('ueler.viewer.main_viewer.AnnotationPaletteEditor', lambda *args, **kwargs: SimpleNamespace()), \
                patch('ueler.viewer.main_viewer.ImageDisplay', lambda *args, **kwargs: SimpleNamespace(main_viewer=None)), \
                patch('ueler.viewer.main_viewer.plt.show', lambda: None), \
                patch('ueler.viewer.main_viewer.select_downsample_factor', lambda *_args, **_kwargs: 1):

                viewer = ImageMaskViewer(str(base_dir))

            self.assertEqual(viewer.max_cache_size, 100)


if __name__ == '__main__':
    unittest.main()
