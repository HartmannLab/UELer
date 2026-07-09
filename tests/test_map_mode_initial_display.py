"""Tests for the map-mode black-view fix (follow-up to issue #84).

Four root causes are covered:

1. ``display()`` must flush map tiles synchronously into the canvas buffer
   (via ``update_display`` + ``canvas.draw()``) **before** ``display_ui()``
   sends the widget to the browser.  Without this the browser receives the
   stale 1×1 black placeholder.

2. ``display()`` must re-call ``_sync_navigation_home_view()`` **after**
   ``display_ui()`` so the ipympl toolbar's Home extent is set to the full
   map bounds (not an earlier single-FOV region that ipympl might have
   recorded when the widget was first shown).

3. ``on_draw`` must fire ``update_display`` when the viewport **size** changes
   even if the center is unchanged (scroll-wheel zoom keeps center fixed).
   Previously the short-circuit compared only center, so scroll-wheel zoom
   never reloaded tiles.

4. (Regression guard) ``on_draw`` must still short-circuit when *both* center
   *and* size are unchanged, preserving the memory-safety guard that prevents
   loading all tiles on the first background draw event.

5. ``on_image_change`` must NOT overwrite the map-canvas axis limits with
   single-FOV pixel dimensions.  ``load_cell_table`` calls
   ``_refresh_viewer_state`` which calls ``on_image_change`` unconditionally;
   before this fix that reset ``ax.set_xlim/ylim`` to e.g. ``(0, 1024)``
   inside a 116-tile map whose canvas is ``>10 000`` pixels wide, placing the
   viewport in an empty region and producing a persistent black square view.
"""
from __future__ import annotations

import contextlib
import math
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

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

_plt = sys.modules["matplotlib.pyplot"]
if not callable(getattr(_plt, "show", None)):
    _plt.show = lambda *_a, **_kw: None
if not callable(getattr(_plt, "subplots", None)):
    _fig_stub = SimpleNamespace(
        canvas=SimpleNamespace(
            header_visible=False,
            draw_idle=lambda: None,
            draw=lambda: None,
        ),
        tight_layout=lambda: None,
    )
    _ax_stub = SimpleNamespace(
        set_xlim=lambda *_a, **_kw: None,
        set_ylim=lambda *_a, **_kw: None,
        get_xlim=lambda: (0.0, 100.0),
        get_ylim=lambda: (100.0, 0.0),
        imshow=lambda *_a, **_kw: SimpleNamespace(
            set_data=lambda *_a, **_kw: None,
            set_extent=lambda *_a, **_kw: None,
        ),
        axis=lambda *_a, **_kw: None,
        annotate=lambda *_a, **_kw: SimpleNamespace(
            set_visible=lambda *_a, **_kw: None
        ),
    )
    _plt.subplots = lambda *_a, **_kw: (_fig_stub, _ax_stub)

_colors = sys.modules["matplotlib.colors"]
if not callable(getattr(_colors, "to_rgb", None)):
    _colors.to_rgb = lambda v: (0.5, 0.5, 0.5)

_text = sys.modules["matplotlib.text"]
if not hasattr(_text, "Annotation"):
    class _Ann:
        pass
    _text.Annotation = _Ann

_patches = sys.modules["matplotlib.patches"]
if not hasattr(_patches, "Polygon"):
    class _Poly:
        def __init__(self, *_a, **_kw):
            pass
    _patches.Polygon = _Poly

_widgets = sys.modules["matplotlib.widgets"]
if not hasattr(_widgets, "RectangleSelector"):
    class _RS:
        def __init__(self, *_a, **_kw):
            pass
    _widgets.RectangleSelector = _RS

_bb = sys.modules["matplotlib.backend_bases"]
if not hasattr(_bb, "MouseButton"):
    class _MB:
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
# Imports under test
# ---------------------------------------------------------------------------

from ueler.viewer.image_display import ImageDisplay  # noqa: E402
from ueler.viewer.main_viewer import ImageMaskViewer  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers / stubs
# ---------------------------------------------------------------------------

def _noop(self, *_a, **_kw):
    return None


class _StubWidget:
    def __init__(self, value=None):
        self.value = value
        self.layout = SimpleNamespace(display="", width="", height="")

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
        self.channel_selector = _StubTagsInput(value=("DAPI",))
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
        # map-mode extras
        self.map_mode_toggle = _StubWidget(value=True)
        self.map_selector = _StubWidget(value="map_A")


def _make_stub_image_display():
    """Return a SimpleNamespace that quacks like an ImageDisplay."""
    canvas_stub = SimpleNamespace(
        header_visible=False,
        draw_idle=lambda: None,
        draw=lambda: None,
    )
    fig_stub = SimpleNamespace(
        canvas=canvas_stub,
        tight_layout=lambda: None,
    )
    ax_stub = SimpleNamespace(
        set_xlim=lambda *_a, **_kw: None,
        set_ylim=lambda *_a, **_kw: None,
        get_xlim=lambda: (0.0, 100.0),
        get_ylim=lambda: (100.0, 0.0),
        annotate=lambda *_a, **_kw: SimpleNamespace(
            set_visible=lambda *_a, **_kw: None
        ),
    )
    return SimpleNamespace(
        main_viewer=None,
        fig=fig_stub,
        ax=ax_stub,
        img_display=SimpleNamespace(
            set_data=lambda *_a, **_kw: None,
            set_extent=lambda *_a, **_kw: None,
        ),
        height=100,
        width=100,
        prev_center_x=50.0,
        prev_center_y=50.0,
        prev_viewport_width=100.0,
        prev_viewport_height=100.0,
        combined=None,
    )


def _fake_load_fov(self, fov_name, requested_channels=None, frame_index=None):
    if fov_name not in self.image_cache:
        arr = np.zeros((8, 8), dtype=np.uint8)
        self.image_cache[fov_name] = {"DAPI": arr}


def _viewer_patches(stub_img_display=None, create_widgets_side_effect=None):
    """Return a list of patches required to construct an ImageMaskViewer stub."""
    sid = stub_img_display or _make_stub_image_display()

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
            lambda *_a, **_kw: sid,
        ),
        patch("ueler.viewer.main_viewer.plt.show", lambda: None),
        patch("ueler.viewer.main_viewer.select_downsample_factor", lambda *_a, **_kw: 1),
    ]


# ---------------------------------------------------------------------------
# Tests — Phase 1 & 2: display() map-mode initial render and home-view re-sync
# ---------------------------------------------------------------------------

class TestDisplayMapModePreRender(unittest.TestCase):
    """display() in map mode must flush tiles synchronously before display_ui()
    and re-sync the toolbar Home view after display_ui().
    """

    def _make_map_viewer(self, base_tmpdir):
        """Return a minimal ImageMaskViewer-like mock for testing display()."""
        import tempfile, os
        os.makedirs(os.path.join(base_tmpdir, "FOV_0"), exist_ok=True)
        open(os.path.join(base_tmpdir, "FOV_0", "channel_DAPI.tiff"), "wb").close()

        call_log: list[str] = []
        sid = _make_stub_image_display()

        # Replace canvas.draw with a tracked version
        sid.fig.canvas.draw = lambda: call_log.append("canvas_draw")

        def tracked_update_display(self_, *_a, **_kw):
            call_log.append("update_display")

        def tracked_sync_home(self_):
            call_log.append("sync_home")

        def tracked_display_ui(viewer_):
            call_log.append("display_ui")

        def tracked_after_plugins(self_):
            call_log.append("after_plugins")

        extra = [
            patch.object(ImageMaskViewer, "update_display", tracked_update_display),
            patch.object(ImageMaskViewer, "_sync_navigation_home_view", tracked_sync_home),
            patch.object(ImageMaskViewer, "load_widget_states", _noop),
            patch.object(ImageMaskViewer, "on_image_change", _noop),
            patch.object(ImageMaskViewer, "after_all_plugins_loaded", tracked_after_plugins),
            patch("ueler.viewer.main_viewer.display_ui", tracked_display_ui),
        ]
        return call_log, sid, extra

    def test_update_display_and_canvas_draw_called_before_display_ui_in_map_mode(self):
        """update_display() and canvas.draw() must appear before display_ui() in
        the call log when _map_mode_active is True."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            call_log, sid, extra = self._make_map_viewer(tmpdir)

            with contextlib.ExitStack() as stack:
                for p in _viewer_patches(stub_img_display=sid) + extra:
                    stack.enter_context(p)
                viewer = ImageMaskViewer(tmpdir)
                viewer._map_mode_active = True
                viewer.SidePlots = SimpleNamespace()
                viewer.BottomPlots = SimpleNamespace()
                call_log.clear()
                viewer.display()

        self.assertIn("update_display", call_log, "update_display not called in display()")
        self.assertIn("canvas_draw", call_log, "canvas.draw() not called in display()")
        self.assertIn("display_ui", call_log, "display_ui not called in display()")

        ui_idx = call_log.index("display_ui")
        ud_idx = call_log.index("update_display")
        cd_idx = call_log.index("canvas_draw")

        self.assertLess(
            ud_idx, ui_idx,
            "update_display must be called BEFORE display_ui in map mode"
        )
        self.assertLess(
            cd_idx, ui_idx,
            "canvas.draw() must be called BEFORE display_ui in map mode"
        )

    def test_sync_navigation_home_view_called_after_display_ui_in_map_mode(self):
        """_sync_navigation_home_view() must be called after display_ui() in
        map mode so the toolbar Home extent is patched while fully wired."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            call_log, sid, extra = self._make_map_viewer(tmpdir)

            with contextlib.ExitStack() as stack:
                for p in _viewer_patches(stub_img_display=sid) + extra:
                    stack.enter_context(p)
                viewer = ImageMaskViewer(tmpdir)
                viewer._map_mode_active = True
                viewer.SidePlots = SimpleNamespace()
                viewer.BottomPlots = SimpleNamespace()
                call_log.clear()
                viewer.display()

        self.assertIn("sync_home", call_log, "_sync_navigation_home_view not called")
        self.assertIn("display_ui", call_log, "display_ui not called")

        ui_idx = call_log.index("display_ui")
        sh_idx = call_log.index("sync_home")
        self.assertLess(
            ui_idx, sh_idx,
            "_sync_navigation_home_view must be called AFTER display_ui in map mode"
        )

    def test_display_non_map_mode_does_not_call_canvas_draw_before_display_ui(self):
        """In non-map mode, display() must NOT insert the sync flush — only the
        trailing backstop update_display is expected."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            call_log, sid, extra = self._make_map_viewer(tmpdir)

            with contextlib.ExitStack() as stack:
                for p in _viewer_patches(stub_img_display=sid) + extra:
                    stack.enter_context(p)
                viewer = ImageMaskViewer(tmpdir)
                viewer._map_mode_active = False   # non-map mode
                viewer.SidePlots = SimpleNamespace()
                viewer.BottomPlots = SimpleNamespace()
                call_log.clear()
                viewer.display()

        # canvas_draw must NOT appear before display_ui
        if "canvas_draw" in call_log and "display_ui" in call_log:
            cd_idx = call_log.index("canvas_draw")
            ui_idx = call_log.index("display_ui")
            self.assertGreater(
                cd_idx, ui_idx,
                "canvas.draw() should not precede display_ui in non-map mode"
            )


# ---------------------------------------------------------------------------
# Tests — Phase 3 & 4: on_draw viewport-size tracking
# ---------------------------------------------------------------------------

class TestOnDrawViewportSizeTracking(unittest.TestCase):
    """on_draw must fire update_display when viewport size changes even if the
    center is unchanged, and must still short-circuit when nothing has changed.
    """

    def _make_image_display(self, xlim, ylim):
        """Build a real ImageDisplay-like stub with the live on_draw method."""
        # We test the real on_draw method by binding it to a minimal object
        # instead of constructing a full ImageDisplay (which needs matplotlib).
        viewer = SimpleNamespace(
            initialized=True,
            current_downsample_factor=1,
            on_downsample_factor_changed=lambda *_a, **_kw: None,
            update_display=MagicMock(),
            ui_component=SimpleNamespace(
                enable_downsample_checkbox=SimpleNamespace(value=True)
            ),
        )

        ax = SimpleNamespace(
            get_xlim=lambda: xlim,
            get_ylim=lambda: ylim,
        )

        disp = SimpleNamespace(
            ax=ax,
            main_viewer=viewer,
            prev_center_x=(xlim[0] + xlim[1]) / 2,
            prev_center_y=(ylim[0] + ylim[1]) / 2,
            prev_viewport_width=xlim[1] - xlim[0],
            prev_viewport_height=abs(ylim[0] - ylim[1]),
        )
        return disp, viewer

    def test_on_draw_fires_update_display_when_viewport_width_changes(self):
        """Zooming in/out (center fixed, size changes) must bypass short-circuit."""
        # Initial viewport: center (50, 50), 100×100
        xlim_initial = (0.0, 100.0)
        ylim_initial = (100.0, 0.0)
        disp, viewer = self._make_image_display(xlim_initial, ylim_initial)

        # Simulate scroll-wheel zoom: same center (50, 50) but viewport is now 50×50
        new_xlim = (25.0, 75.0)
        new_ylim = (75.0, 25.0)
        disp.ax.get_xlim = lambda: new_xlim
        disp.ax.get_ylim = lambda: new_ylim

        # Bind and call the real on_draw (skip the decorator)
        ImageDisplay.on_draw(disp, event=None)

        viewer.update_display.assert_called_once_with(viewer.current_downsample_factor)

    def test_on_draw_short_circuits_when_center_and_size_unchanged(self):
        """When neither center nor size has changed, on_draw must short-circuit."""
        xlim = (0.0, 100.0)
        ylim = (100.0, 0.0)
        disp, viewer = self._make_image_display(xlim, ylim)
        # prev_* already seeded to the same values

        ImageDisplay.on_draw(disp, event=None)

        viewer.update_display.assert_not_called()

    def test_on_draw_fires_when_center_changes_but_size_unchanged(self):
        """Panning (center changes, size constant) must also bypass short-circuit."""
        xlim_initial = (0.0, 100.0)
        ylim_initial = (100.0, 0.0)
        disp, viewer = self._make_image_display(xlim_initial, ylim_initial)

        # Pan right: center moves from 50 to 100, same 100-wide viewport
        disp.ax.get_xlim = lambda: (50.0, 150.0)
        disp.ax.get_ylim = lambda: (100.0, 0.0)

        ImageDisplay.on_draw(disp, event=None)

        viewer.update_display.assert_called_once_with(viewer.current_downsample_factor)

    def test_prev_viewport_attrs_updated_after_non_short_circuit(self):
        """After on_draw fires (no short-circuit), prev_viewport_* must be updated."""
        xlim_initial = (0.0, 100.0)
        ylim_initial = (100.0, 0.0)
        disp, viewer = self._make_image_display(xlim_initial, ylim_initial)

        new_xlim = (25.0, 75.0)   # zoomed in: 50-wide
        new_ylim = (75.0, 25.0)   # 50-tall
        disp.ax.get_xlim = lambda: new_xlim
        disp.ax.get_ylim = lambda: new_ylim

        ImageDisplay.on_draw(disp, event=None)

        self.assertAlmostEqual(disp.prev_viewport_width, 50.0)
        self.assertAlmostEqual(disp.prev_viewport_height, 50.0)


# ---------------------------------------------------------------------------
# Test — RC5: on_image_change must not overwrite map-canvas axis limits
# ---------------------------------------------------------------------------

class TestOnImageChangeMapModeGuard(unittest.TestCase):
    """on_image_change must not reset ax.set_xlim/ylim when map mode is active.

    load_cell_table calls _refresh_viewer_state → on_image_change.  Before
    the fix this unconditionally wrote single-FOV pixel dimensions into the
    map canvas, placing the viewport in an empty region of the map (→ black).
    """

    def _make_map_viewer_stub(self, map_width=12000, map_height=9000, fov_size=1024):
        """Return a minimal viewer stub with map mode active."""
        xlim_calls = []
        ylim_calls = []

        ax = SimpleNamespace(
            get_xlim=lambda: (0.0, float(map_width)),
            get_ylim=lambda: (float(map_height), 0.0),
            set_xlim=lambda *a, **kw: xlim_calls.append(a),
            set_ylim=lambda *a, **kw: ylim_calls.append(a),
            annotate=lambda *_a, **_kw: SimpleNamespace(set_visible=lambda *_a, **_kw: None),
        )
        canvas = SimpleNamespace(header_visible=False, draw_idle=lambda: None, toolbar=None)
        fig = SimpleNamespace(canvas=canvas, tight_layout=lambda: None)
        img_display_stub = SimpleNamespace(
            ax=ax,
            fig=fig,
            height=map_height,
            width=map_width,
            clear_patches=lambda: None,
            img_display=SimpleNamespace(
                set_data=lambda *_a, **_kw: None,
                set_extent=lambda *_a, **_kw: None,
            ),
        )

        viewer = MagicMock(spec=ImageMaskViewer)
        viewer._map_mode_active = True
        viewer._grid_display = None
        viewer.initialized = True
        viewer.height = fov_size
        viewer.width = fov_size
        viewer.image_display = img_display_stub
        viewer.masks_available = False
        viewer.annotations_available = False
        viewer.channel_names_set = set()
        viewer.SidePlots = SimpleNamespace()
        viewer.image_cache = {
            "FOV_0": {"DAPI": np.zeros((fov_size, fov_size), dtype=np.uint8)}
        }
        viewer.ui_component = _StubUI(channel_options=("DAPI",))
        viewer.ui_component.image_selector.value = "FOV_0"
        viewer.ui_component.channel_selector.value = ("DAPI",)

        return viewer, img_display_stub, xlim_calls, ylim_calls

    def test_on_image_change_does_not_call_set_xlim_in_map_mode(self):
        """ax.set_xlim must NOT be called when _map_mode_active is True."""
        viewer, _, xlim_calls, ylim_calls = self._make_map_viewer_stub()

        ImageMaskViewer.on_image_change(viewer, {"new": "FOV_0"})

        self.assertEqual(
            xlim_calls, [],
            f"ax.set_xlim was called {len(xlim_calls)} time(s) in map mode; expected 0. "
            f"Calls: {xlim_calls}"
        )
        self.assertEqual(
            ylim_calls, [],
            f"ax.set_ylim was called {len(ylim_calls)} time(s) in map mode; expected 0. "
            f"Calls: {ylim_calls}"
        )

    def test_on_image_change_preserves_map_canvas_dimensions(self):
        """image_display.width/height must stay at map dimensions after on_image_change."""
        map_w, map_h, fov_sz = 12000, 9000, 1024
        viewer, img_display, _, _ = self._make_map_viewer_stub(map_w, map_h, fov_sz)

        ImageMaskViewer.on_image_change(viewer, {"new": "FOV_0"})

        self.assertEqual(
            img_display.width, map_w,
            f"image_display.width was changed from {map_w} to {img_display.width} in map mode"
        )
        self.assertEqual(
            img_display.height, map_h,
            f"image_display.height was changed from {map_h} to {img_display.height} in map mode"
        )

    def test_on_image_change_calls_set_xlim_in_non_map_mode(self):
        """ax.set_xlim IS expected to be called when _map_mode_active is False."""
        viewer, _, xlim_calls, ylim_calls = self._make_map_viewer_stub()
        viewer._map_mode_active = False   # non-map mode

        ImageMaskViewer.on_image_change(viewer, {"new": "FOV_0"})

        self.assertGreater(
            len(xlim_calls), 0,
            "ax.set_xlim should be called in non-map mode"
        )


# ---------------------------------------------------------------------------
# Tests — Phase 6: _map_needs_initial_render flag
# ---------------------------------------------------------------------------

class TestMapNeedsInitialRenderFlag(unittest.TestCase):
    """_set_map_canvas_dimensions must set _map_needs_initial_render=True so
    that the first on_draw after the widget is shown bypasses the short-circuit
    and triggers a real tile render (RC5 fix).
    """

    def _make_on_draw_obj(self, *, center, viewport, needs_initial_render):
        """Build a minimal object that has the real on_draw short-circuit logic."""
        from ueler.viewer.image_display import ImageDisplay

        update_calls: list = []
        viewer = SimpleNamespace(
            initialized=True,
            current_downsample_factor=1,
            on_downsample_factor_changed=lambda *_a, **_kw: None,
            update_display=lambda *_a, **_kw: update_calls.append("update_display"),
            ui_component=SimpleNamespace(
                enable_downsample_checkbox=SimpleNamespace(value=True)
            ),
        )
        cx, cy = center
        vw, vh = viewport
        obj = SimpleNamespace(
            main_viewer=viewer,
            prev_center_x=cx,
            prev_center_y=cy,
            prev_viewport_width=vw,
            prev_viewport_height=vh,
            _map_needs_initial_render=needs_initial_render,
        )
        obj.ax = SimpleNamespace(
            get_xlim=lambda: (cx - vw / 2, cx + vw / 2),
            get_ylim=lambda: (cy + vh / 2, cy - vh / 2),
        )
        return obj, update_calls, ImageDisplay.on_draw

    def test_flag_true_bypasses_short_circuit(self):
        """`on_draw` must not short-circuit when _map_needs_initial_render=True,
        even if center and size are identical to the seeded values."""
        obj, update_calls, on_draw = self._make_on_draw_obj(
            center=(5000.0, 4500.0),
            viewport=(10000.0, 9000.0),
            needs_initial_render=True,
        )
        on_draw(obj, None)
        self.assertIn(
            "update_display", update_calls,
            "on_draw must call update_display when _map_needs_initial_render=True"
        )

    def test_flag_cleared_after_first_draw(self):
        """`_map_needs_initial_render` must be set to False after the first
        on_draw passes through, preventing a second forced render."""
        obj, _, on_draw = self._make_on_draw_obj(
            center=(5000.0, 4500.0),
            viewport=(10000.0, 9000.0),
            needs_initial_render=True,
        )
        on_draw(obj, None)
        self.assertFalse(
            obj._map_needs_initial_render,
            "_map_needs_initial_render must be cleared after the forced render"
        )

    def test_flag_false_short_circuits_as_normal(self):
        """When the flag is False, on_draw still short-circuits on unchanged
        center + size (regression guard for the memory-safety behaviour)."""
        obj, update_calls, on_draw = self._make_on_draw_obj(
            center=(5000.0, 4500.0),
            viewport=(10000.0, 9000.0),
            needs_initial_render=False,
        )
        on_draw(obj, None)
        self.assertNotIn(
            "update_display", update_calls,
            "on_draw must still short-circuit when flag is False and values unchanged"
        )

    def test_set_map_canvas_dimensions_sets_flag(self):
        """`_set_map_canvas_dimensions` must set _map_needs_initial_render=True
        so the first on_draw after map activation renders tiles."""
        import contextlib, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            os.makedirs(os.path.join(tmpdir, "FOV_0"), exist_ok=True)
            open(os.path.join(tmpdir, "FOV_0", "chan.tiff"), "wb").close()

            sid = _make_stub_image_display()
            sid._map_needs_initial_render = False  # start as False

            extra = [
                patch.object(ImageMaskViewer, "load_widget_states", _noop),
                patch.object(ImageMaskViewer, "on_image_change", _noop),
                patch.object(ImageMaskViewer, "update_display", _noop),
                patch.object(ImageMaskViewer, "_sync_navigation_home_view", _noop),
            ]
            with contextlib.ExitStack() as stack:
                for p in _viewer_patches(stub_img_display=sid) + extra:
                    stack.enter_context(p)
                viewer = ImageMaskViewer(tmpdir)
                viewer.image_display = sid
                viewer._set_map_canvas_dimensions(10000, 9000)

        self.assertTrue(
            sid._map_needs_initial_render,
            "_set_map_canvas_dimensions must set _map_needs_initial_render=True"
        )


if __name__ == "__main__":
    unittest.main()
