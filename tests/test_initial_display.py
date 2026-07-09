"""Tests for the black-view-on-launch fix (issue #84).

Three root causes are covered:

1. ``load_widget_states`` runs with ``_suspend_display_updates=True``, which
   silently drops the ``update_display`` call inside it.  After the ``finally``
   block an explicit ``update_display`` must fire.

2. In simple-viewer mode (no ``cell_table``), ``on_image_change`` left
   ``channel_selector.value`` empty on startup, so the first render produced a
   solid-black image.

3. ``display()`` now calls ``update_display`` as a backstop after the full
   widget tree is shown.
"""
from __future__ import annotations

import contextlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Lightweight stubs for heavy optional dependencies
# ---------------------------------------------------------------------------

for _mod_name in [
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.colors",
    "matplotlib.text",
    "matplotlib.font_manager",
    "matplotlib.patches",
    "matplotlib.widgets",
    "matplotlib.backend_bases",
    "cv2",
    "skimage",
    "skimage.segmentation",
    "skimage.io",
    "skimage.exposure",
    "skimage.transform",
    "skimage.color",
    "skimage.measure",
    "dask",
    "seaborn_image",
    "tifffile",
]:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        sys.modules[_mod_name] = _stub

# Provide the specific attributes tested code accesses.
_plt = sys.modules["matplotlib.pyplot"]
if not callable(getattr(_plt, "show", None)):
    _plt.show = lambda *_a, **_kw: None
if not callable(getattr(_plt, "subplots", None)):
    _fig_stub = SimpleNamespace(
        canvas=SimpleNamespace(
            header_visible=False,
            draw_idle=lambda: None,
        ),
        tight_layout=lambda: None,
    )
    _ax_stub = SimpleNamespace(
        set_xlim=lambda *_a, **_kw: None,
        set_ylim=lambda *_a, **_kw: None,
        imshow=lambda *_a, **_kw: SimpleNamespace(
            set_data=lambda *_a, **_kw: None,
            set_extent=lambda *_a, **_kw: None,
        ),
        axis=lambda *_a, **_kw: None,
        annotate=lambda *_a, **_kw: SimpleNamespace(set_visible=lambda *_a, **_kw: None),
    )
    _plt.subplots = lambda *_a, **_kw: (_fig_stub, _ax_stub)

_colors = sys.modules["matplotlib.colors"]
if not callable(getattr(_colors, "to_rgb", None)):
    _colors.to_rgb = lambda v: (0.5, 0.5, 0.5)

_text = sys.modules["matplotlib.text"]
if not hasattr(_text, "Annotation"):
    class _Ann:  # pragma: no cover
        pass
    _text.Annotation = _Ann

_patches = sys.modules["matplotlib.patches"]
if not hasattr(_patches, "Polygon"):
    class _Poly:  # pragma: no cover
        def __init__(self, *_a, **_kw):
            pass
    _patches.Polygon = _Poly

_widgets = sys.modules["matplotlib.widgets"]
if not hasattr(_widgets, "RectangleSelector"):
    class _RS:  # pragma: no cover
        def __init__(self, *_a, **_kw):
            pass
    _widgets.RectangleSelector = _RS

_bb = sys.modules["matplotlib.backend_bases"]
if not hasattr(_bb, "MouseButton"):
    class _MB:  # pragma: no cover
        LEFT = 1
        RIGHT = 3
    _bb.MouseButton = _MB

_seg = sys.modules["skimage.segmentation"]
if not callable(getattr(_seg, "find_boundaries", None)):
    _seg.find_boundaries = lambda *_a, **_kw: None

skimage_pkg = sys.modules["skimage"]
if not hasattr(skimage_pkg, "__path__"):
    skimage_pkg.__path__ = []

# ---------------------------------------------------------------------------
# Import the module under test *after* stubs are in place
# ---------------------------------------------------------------------------

from ueler.viewer.main_viewer import ImageMaskViewer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_fov_dir(root: Path, n: int = 2) -> None:
    """Create ``n`` minimal FOV subdirectories each with one dummy TIFF."""
    for i in range(n):
        d = root / f"FOV_{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "channel_DAPI.tiff").write_bytes(b"\x00")


def _fake_load_fov(self, fov_name, requested_channels=None, frame_index=None):
    if fov_name not in self.image_cache:
        arr = np.zeros((8, 8), dtype=np.uint8)
        self.image_cache[fov_name] = {"DAPI": arr}


class _StubWidget:
    def __init__(self, value=None):
        self.value = value
        self.layout = SimpleNamespace(display="", width="", height="")
        self._observers: list = []

    def observe(self, *_a, **_kw):
        return None

    def unobserve(self, *_a, **_kw):
        return None


class _StubTagsInput(_StubWidget):
    def __init__(self, value=()):
        super().__init__(value=value)
        self.allowed_tags: list = []


class _StubUI:
    def __init__(self, channel_options=("DAPI",)):
        self.image_selector = _StubWidget(value="FOV_0")
        self.channel_selector = _StubTagsInput(value=())
        self.channel_selector.allowed_tags = list(channel_options)
        self.color_controls: dict = {}
        self.contrast_min_controls: dict = {}
        self.contrast_max_controls: dict = {}
        self.channel_visibility_controls: dict = {}
        self.mask_display_controls: dict = {}
        self.mask_color_controls: dict = {}
        self.pixel_size_inttext = _StubWidget(390)
        self.mask_outline_thickness_slider = _StubWidget(1)
        self.annotation_editor_host = SimpleNamespace(
            children=(), layout=SimpleNamespace(display="none")
        )
        self.annotation_selector = _StubWidget(value=None)
        self.annotation_display_checkbox = _StubWidget(value=False)
        self.enable_downsample_checkbox = _StubWidget(True)
        self.cache_size_input = _StubWidget(100)


def _noop(self, *_a, **_kw):
    return None


# Context manager that patches everything needed to construct an
# ImageMaskViewer without touching the filesystem or real UI widgets.
def _viewer_patches(create_widgets_side_effect=None):
    stub_img_display = SimpleNamespace(
        main_viewer=None,
        fig=SimpleNamespace(
            canvas=SimpleNamespace(header_visible=False, draw_idle=lambda: None),
            tight_layout=lambda: None,
        ),
        ax=SimpleNamespace(
            set_xlim=lambda *_a, **_kw: None,
            set_ylim=lambda *_a, **_kw: None,
            annotate=lambda *_a, **_kw: SimpleNamespace(
                set_visible=lambda *_a, **_kw: None
            ),
        ),
        img_display=SimpleNamespace(
            set_data=lambda *_a, **_kw: None,
            set_extent=lambda *_a, **_kw: None,
        ),
        height=8,
        width=8,
    )

    def _default_create_widgets(viewer):
        viewer.ui_component = _StubUI()

    cw = create_widgets_side_effect or _default_create_widgets

    return [
        patch.object(ImageMaskViewer, "load_status_images", _noop),
        patch.object(ImageMaskViewer, "_initialize_map_descriptors", _noop),
        patch.object(ImageMaskViewer, "_initialize_annotation_palette_manager", _noop),
        patch.object(ImageMaskViewer, "_refresh_annotation_control_states", _noop),
        patch.object(ImageMaskViewer, "setup_widget_observers", _noop),
        patch.object(ImageMaskViewer, "setup_event_connections", _noop),
        patch.object(ImageMaskViewer, "update_marker_set_dropdown", _noop),
        patch.object(ImageMaskViewer, "update_controls", _noop),
        patch.object(ImageMaskViewer, "setup_attr_observers", _noop),
        patch.object(ImageMaskViewer, "_refresh_map_controls", _noop),
        patch.object(ImageMaskViewer, "load_fov", _fake_load_fov),
        patch("ueler.viewer.main_viewer.create_widgets", cw),
        patch("ueler.viewer.main_viewer.ROIManager", lambda *_a, **_kw: SimpleNamespace()),
        patch(
            "ueler.viewer.main_viewer.AnnotationPaletteEditor",
            lambda *_a, **_kw: SimpleNamespace(),
        ),
        patch(
            "ueler.viewer.main_viewer.ImageDisplay",
            lambda *_a, **_kw: stub_img_display,
        ),
        patch("ueler.viewer.main_viewer.plt.show", lambda: None),
        patch("ueler.viewer.main_viewer.select_downsample_factor", lambda *_a, **_kw: 1),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadWidgetStatesTriggersRender(unittest.TestCase):
    """Phase 1: after load_widget_states completes, update_display is called."""

    def test_update_display_called_after_suspend_released(self):
        """update_display must be invoked once after load_widget_states,
        i.e. after _suspend_display_updates is reset to False."""
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _make_temp_fov_dir(base)

            render_calls: list = []

            def tracking_update_display(self, *_a, **_kw):
                # Only record the call when the flag is NOT suppressed.
                render_calls.append(bool(self._suspend_display_updates))

            def fake_load_widget_states(self, *_a, **_kw):
                # Simulate load_widget_states: flag is True here.
                render_calls.append(("inside_lws", bool(self._suspend_display_updates)))

            extra = [
                patch.object(
                    ImageMaskViewer, "update_display", tracking_update_display
                ),
                patch.object(
                    ImageMaskViewer, "load_widget_states", fake_load_widget_states
                ),
                patch.object(ImageMaskViewer, "on_image_change", _noop),
            ]
            with contextlib.ExitStack() as stack:
                for p in _viewer_patches() + extra:
                    stack.enter_context(p)
                ImageMaskViewer(str(base))

        # At least one render call must have happened with flag=False
        # (i.e. after load_widget_states completed).
        unsuppressed = [v for v in render_calls if v is False]
        self.assertTrue(
            len(unsuppressed) >= 1,
            f"Expected ≥1 unsuppressed render call, got: {render_calls}",
        )

        # The call recorded inside load_widget_states must show flag=True.
        inside = [v for v in render_calls if isinstance(v, tuple) and v[0] == "inside_lws"]
        if inside:
            self.assertTrue(inside[0][1], "Flag should be True inside load_widget_states")

    def test_render_after_lws_fires_even_when_no_saved_state(self):
        """When there is no saved state file load_widget_states is a no-op,
        but update_display must still be called after it returns."""
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _make_temp_fov_dir(base)

            render_count = {"n": 0}

            def counting_update_display(self, *_a, **_kw):
                if not self._suspend_display_updates:
                    render_count["n"] += 1

            extra = [
                patch.object(ImageMaskViewer, "update_display", counting_update_display),
                patch.object(ImageMaskViewer, "load_widget_states", _noop),
                patch.object(ImageMaskViewer, "on_image_change", _noop),
            ]
            with contextlib.ExitStack() as stack:
                for p in _viewer_patches() + extra:
                    stack.enter_context(p)
                ImageMaskViewer(str(base))

        self.assertGreaterEqual(
            render_count["n"],
            1,
            "update_display must be called at least once after load_widget_states",
        )


class TestInitialChannelSelectionSimpleViewer(unittest.TestCase):
    """Phase 2: first channel is selected on startup even in simple viewer mode."""

    def _make_viewer_stub(self, base: Path):
        """Return a minimal viewer stub with uninitialized=False."""
        v = MagicMock(spec=ImageMaskViewer)
        v.initialized = False
        v.cell_table = None
        v.ui_component = _StubUI(channel_options=("DAPI", "CD45"))
        # Populate the cache as load_fov would.
        v.image_cache = {"FOV_0": {"DAPI": np.zeros((4, 4)), "CD45": np.zeros((4, 4))}}
        return v

    def test_first_channel_selected_when_no_cell_table_and_not_initialized(self):
        """on_image_change should default to the first channel during startup."""
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _make_temp_fov_dir(base)

            selected: list = []

            def on_image_change_spy(self, change):
                # Run the real on_image_change but capture channel_selector.value after
                original_on_image_change(self, change)
                selected.append(tuple(self.ui_component.channel_selector.value))

            original_on_image_change = ImageMaskViewer.on_image_change

            extra = [
                patch.object(ImageMaskViewer, "on_image_change", on_image_change_spy),
                patch.object(ImageMaskViewer, "update_display", _noop),
                patch.object(ImageMaskViewer, "load_widget_states", _noop),
            ]
            with contextlib.ExitStack() as stack:
                for p in _viewer_patches() + extra:
                    stack.enter_context(p)
                viewer = ImageMaskViewer(str(base))

        # The channel selection captured during __init__ must be non-empty.
        self.assertTrue(
            any(len(sel) > 0 for sel in selected),
            f"Expected at least one non-empty channel selection during init, got: {selected}",
        )

    def test_channel_selector_empty_after_initialized_fov_switch_no_cell_table(self):
        """After initialization, switching FOV with cell_table=None must leave
        channel_selector empty (existing UX preserved)."""
        # Build a minimal viewer directly in memory (no filesystem needed).
        viewer = MagicMock()
        viewer.initialized = True  # post-init state
        viewer.cell_table = None
        ui = _StubUI(channel_options=("DAPI", "CD45"))
        ui.image_selector.value = "FOV_1"
        viewer.ui_component = ui
        viewer.masks_available = False
        viewer.annotations_available = False
        viewer.channel_names_set = set()
        viewer.SidePlots = SimpleNamespace()
        viewer.image_display = MagicMock()
        viewer.image_display.height = 8
        viewer.image_display.width = 8
        # Prevent the nav-stack update block from unpacking a MagicMock.
        viewer.image_display.fig.canvas.toolbar = None
        # Simulate: no previous channel matches in the new FOV.
        ui.channel_selector.value = ()  # empty → new_selection will be ()

        # Populate image_cache as load_fov would.
        from collections import OrderedDict
        viewer.image_cache = OrderedDict({"FOV_1": {"DAPI": np.zeros((4, 4))}})

        # Run the real method on the mock (binding it explicitly).
        ImageMaskViewer.on_image_change(viewer, {"new": "FOV_1"})

        # For an initialized viewer with no cell_table, value should stay ().
        self.assertEqual(
            viewer.ui_component.channel_selector.value,
            (),
            "After init, simple-viewer FOV switch should leave channel_selector empty",
        )


class TestDisplayBackstopRender(unittest.TestCase):
    """Phase 3: display() calls update_display after the widget tree is shown."""

    def test_display_calls_update_display_as_backstop(self):
        """display() must invoke update_display at least once after display_ui."""
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            _make_temp_fov_dir(base)

            call_order: list[str] = []

            def tracking_display_ui(viewer):
                call_order.append("display_ui")

            def tracking_after_plugins(self):
                call_order.append("after_plugins")

            def tracking_update_display(self, *_a, **_kw):
                call_order.append("update_display")

            extra = [
                patch.object(ImageMaskViewer, "update_display", tracking_update_display),
                patch.object(ImageMaskViewer, "load_widget_states", _noop),
                patch.object(ImageMaskViewer, "on_image_change", _noop),
                patch.object(ImageMaskViewer, "after_all_plugins_loaded", tracking_after_plugins),
                patch("ueler.viewer.main_viewer.display_ui", tracking_display_ui),
            ]
            with contextlib.ExitStack() as stack:
                for p in _viewer_patches() + extra:
                    stack.enter_context(p)
                viewer = ImageMaskViewer(str(base))
                # Now call display() directly to test the backstop.
                viewer.SidePlots = SimpleNamespace()
                viewer.BottomPlots = SimpleNamespace()
                call_order.clear()
                viewer.display()

        # display_ui must appear before update_display in the call order.
        self.assertIn("display_ui", call_order)
        self.assertIn("update_display", call_order)
        self.assertLess(
            call_order.index("display_ui"),
            call_order.index("update_display"),
            "update_display should be called *after* display_ui in display()",
        )


if __name__ == "__main__":
    unittest.main()
