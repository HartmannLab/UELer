"""Tests for lasso selection in ImageDisplay (single-FOV and map mode)."""

import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np


# ---------------------------------------------------------------------------
# Stub heavy dependencies
# ---------------------------------------------------------------------------

def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


if "cv2" not in sys.modules:
    _stub_module("cv2")

def _get_axis_limits_with_padding(viewer, ds):
    # return (xmin, xmax, ymin, ymax, xmin_ds, xmax_ds, ymin_ds, ymax_ds)
    return (0, viewer._test_width, 0, viewer._test_height, 0, 0, 0, 0)


if "ueler.image_utils" not in sys.modules:
    _iu = _stub_module("ueler.image_utils")
    _iu.calculate_downsample_factor = lambda *a, **kw: 1  # type: ignore[attr-defined]
    _iu.generate_edges = lambda *a, **kw: np.zeros((1, 1))  # type: ignore[attr-defined]
    _iu.get_axis_limits_with_padding = _get_axis_limits_with_padding  # type: ignore[attr-defined]


class _PatchAxisLimitsMixin:
    """Force ``get_axis_limits_with_padding`` to the lightweight stub for the
    duration of each lasso test, restoring it afterward.

    The module-level stub above is only installed when this file imports
    ``ueler.image_utils`` first; in the full suite the real module is already
    loaded, so its real ``get_axis_limits_with_padding`` (which needs viewer
    attributes this file's minimal stub omits) would otherwise be used. Patching
    per-test keeps the tests order-independent without polluting other modules.

    Two references must be patched: ``ueler.image_utils`` (picked up by the
    function-local ``from ... import`` in ``_find_masks_in_lasso_single_fov``)
    and ``ueler.viewer.image_display`` (the name bound at module load, used by
    ``update_patches``).
    """

    _PATCH_TARGETS = ("ueler.image_utils", "ueler.viewer.image_display")

    def setUp(self):
        super().setUp()
        import importlib

        self._orig_axis_limits = {}
        for target in self._PATCH_TARGETS:
            mod = sys.modules.get(target) or importlib.import_module(target)
            if hasattr(mod, "get_axis_limits_with_padding"):
                self._orig_axis_limits[target] = mod.get_axis_limits_with_padding
                mod.get_axis_limits_with_padding = _get_axis_limits_with_padding

    def tearDown(self):
        for target, original in self._orig_axis_limits.items():
            sys.modules[target].get_axis_limits_with_padding = original
        super().tearDown()

# skimage stub
if "skimage" not in sys.modules:
    _skimage = _stub_module("skimage")
    _seg = _stub_module("skimage.segmentation")
    _seg.find_boundaries = lambda mask, **kw: mask  # type: ignore[attr-defined]
    _skimage.segmentation = _seg  # type: ignore[attr-defined]

# matplotlib path (real import, available in test env)
try:
    from matplotlib.path import Path as MplPath  # type: ignore[import]
except Exception:
    # minimal fallback if matplotlib is unavailable
    class MplPath:  # type: ignore[no-redef]
        def __init__(self, verts):
            self._verts = np.array(verts)

        def contains_points(self, points):
            # Naive point-in-polygon using winding number not needed for stubs;
            # just return all-False so tests that call this stub will fail clearly.
            return np.zeros(len(points), dtype=bool)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_display():
    """Return an ImageDisplay instance bypassing matplotlib figure creation."""
    from ueler.viewer.image_display import ImageDisplay

    obj = ImageDisplay.__new__(ImageDisplay)
    obj.width = 10
    obj.height = 10
    obj._roi_selector = None
    obj._roi_callback = None
    obj._lasso_selector = None
    obj._lasso_active = False
    obj._lasso_on_complete = None
    obj.selected_masks_label = set()
    # Mock ax with viewport at origin (xmin=0, ymin=0) — all existing tests use this.
    # y-axis is inverted in map mode (ylim = (height, 0)), so min(ylim) = 0 = ymin_px.
    obj.ax = SimpleNamespace(
        get_xlim=lambda: (0.0, 1000.0),
        get_ylim=lambda: (1000.0, 0.0),
    )
    return obj


def _make_viewer_for_lasso(mask_array: np.ndarray, fov_name: str = "FOV1"):
    """Return a minimal viewer stub wired to a single mask array."""
    viewer = SimpleNamespace()
    viewer._map_mode_active = False
    viewer._active_map_id = None
    viewer._test_width = mask_array.shape[1]
    viewer._test_height = mask_array.shape[0]
    viewer.current_downsample_factor = 1
    viewer.inform_plugins = lambda _method: None

    ui = SimpleNamespace()
    ui.image_selector = SimpleNamespace(value=fov_name)
    viewer.ui_component = ui

    viewer.full_resolution_label_masks = {"whole_cell": mask_array}
    return viewer


def _square_lasso(x0, y0, x1, y1):
    """Return lasso vertices forming a closed rectangle."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


# ---------------------------------------------------------------------------
# Single-FOV lasso tests
# ---------------------------------------------------------------------------

class TestFindMasksInLassoSingleFov(_PatchAxisLimitsMixin, unittest.TestCase):

    def _call(self, image_display, viewer, verts):
        image_display.main_viewer = viewer
        return image_display._find_masks_in_lasso_single_fov(verts)

    def test_cell_fully_inside_lasso_is_selected(self):
        # 10×10 mask: cell 1 occupies rows 2-4, cols 2-4
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[2:5, 2:5] = 1

        display = _make_image_display()
        viewer = _make_viewer_for_lasso(mask)

        # Lasso covers the whole cell with margin
        verts = _square_lasso(1.5, 1.5, 5.5, 5.5)
        result = self._call(display, viewer, verts)

        ids = {sel.mask_id for sel in result}
        self.assertIn(1, ids)

    def test_cell_outside_lasso_not_selected(self):
        # cell 1: rows 2-4, cols 2-4; lasso covers rows 6-8, cols 6-8 only
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[2:5, 2:5] = 1

        display = _make_image_display()
        viewer = _make_viewer_for_lasso(mask)

        verts = _square_lasso(6.0, 6.0, 9.0, 9.0)
        result = self._call(display, viewer, verts)

        ids = {sel.mask_id for sel in result}
        self.assertNotIn(1, ids)

    def test_cell_partially_inside_is_selected(self):
        # cell 1: rows 4-7, cols 4-7; lasso covers only top-left corner (4-5, 4-5)
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[4:8, 4:8] = 1

        display = _make_image_display()
        viewer = _make_viewer_for_lasso(mask)

        # Lasso covers just one pixel of cell 1 (row4, col4)
        verts = _square_lasso(3.5, 3.5, 5.5, 5.5)
        result = self._call(display, viewer, verts)

        ids = {sel.mask_id for sel in result}
        self.assertIn(1, ids)

    def test_two_cells_separate_lasso_selects_one(self):
        # cell 1: rows 1-2, cols 1-2
        # cell 2: rows 6-7, cols 6-7
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[1:3, 1:3] = 1
        mask[6:8, 6:8] = 2

        display = _make_image_display()
        viewer = _make_viewer_for_lasso(mask)

        # Lasso covers only cell 1 region
        verts = _square_lasso(0.5, 0.5, 3.5, 3.5)
        result = self._call(display, viewer, verts)

        ids = {sel.mask_id for sel in result}
        self.assertIn(1, ids)
        self.assertNotIn(2, ids)

    def test_both_cells_selected_when_lasso_covers_both(self):
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[1:3, 1:3] = 1
        mask[6:8, 6:8] = 2

        display = _make_image_display()
        viewer = _make_viewer_for_lasso(mask)

        verts = _square_lasso(0.5, 0.5, 9.5, 9.5)
        result = self._call(display, viewer, verts)

        ids = {sel.mask_id for sel in result}
        self.assertIn(1, ids)
        self.assertIn(2, ids)

    def test_background_pixels_not_selected(self):
        # All background (0) — no selection
        mask = np.zeros((10, 10), dtype=np.int32)

        display = _make_image_display()
        viewer = _make_viewer_for_lasso(mask)

        verts = _square_lasso(0.5, 0.5, 9.5, 9.5)
        result = self._call(display, viewer, verts)

        self.assertEqual(len(result), 0)

    def test_result_fov_and_mask_name_correct(self):
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[3:6, 3:6] = 5

        display = _make_image_display()
        viewer = _make_viewer_for_lasso(mask, fov_name="MyFOV")

        verts = _square_lasso(0.5, 0.5, 9.5, 9.5)
        result = self._call(display, viewer, verts)

        self.assertEqual(len(result), 1)
        sel = next(iter(result))
        self.assertEqual(sel.fov, "MyFOV")
        self.assertEqual(sel.mask, "whole_cell")
        self.assertEqual(sel.mask_id, 5)


# ---------------------------------------------------------------------------
# Map mode lasso tests
# ---------------------------------------------------------------------------

class TestFindMasksInLassoMapMode(_PatchAxisLimitsMixin, unittest.TestCase):
    """Verify that _find_masks_in_lasso_map_mode maps canvas coords to tile masks."""

    def _make_tile_viewport(self, dest_x0, dest_y0, dest_x1, dest_y1,
                             region_xy=(0, 5, 0, 5), downsample_factor=1):
        tvp = SimpleNamespace()
        tvp.dest_x0 = dest_x0
        tvp.dest_y0 = dest_y0
        tvp.dest_x1 = dest_x1
        tvp.dest_y1 = dest_y1
        tvp.region_xy = region_xy        # (x_min_px, x_max_px, y_min_px, y_max_px)
        tvp.downsample_factor = downsample_factor
        return tvp

    def _call(self, image_display, viewer, verts):
        image_display.main_viewer = viewer
        return image_display._find_masks_in_lasso_map_mode(verts)

    def _make_map_viewer(self, mask_array, tile_viewport, fov_name="Tile1"):
        viewer = SimpleNamespace()
        viewer._map_mode_active = True
        viewer._active_map_id = "map1"

        layer = SimpleNamespace()
        layer.last_tile_viewports = lambda: {fov_name: tile_viewport}
        viewer._get_map_layer = lambda map_id: layer

        viewer._selected_mask_names = lambda: ("whole_cell",)
        viewer._get_mask_array = lambda fov, mask: mask_array

        return viewer

    def test_cell_at_tile_origin_selected_by_lasso_at_canvas_origin(self):
        # 5×5 mask; cell 1 occupies rows 1-3, cols 1-3
        mask = np.zeros((5, 5), dtype=np.int32)
        mask[1:4, 1:4] = 1

        # Tile is placed at canvas (10, 20) – (15, 25)
        tvp = self._make_tile_viewport(dest_x0=10, dest_y0=20, dest_x1=15, dest_y1=25,
                                       region_xy=(0, 5, 0, 5))

        viewer = self._make_map_viewer(mask, tvp)
        display = _make_image_display()

        # Lasso in canvas space covers the tile exactly (with margin)
        verts = _square_lasso(9.5, 19.5, 15.5, 25.5)
        result = self._call(display, viewer, verts)

        ids = {sel.mask_id for sel in result}
        self.assertIn(1, ids)

    def test_cell_outside_canvas_lasso_not_selected(self):
        mask = np.zeros((5, 5), dtype=np.int32)
        mask[1:4, 1:4] = 1

        # Tile placed at canvas (50, 50) – (55, 55)
        tvp = self._make_tile_viewport(dest_x0=50, dest_y0=50, dest_x1=55, dest_y1=55,
                                       region_xy=(0, 5, 0, 5))
        viewer = self._make_map_viewer(mask, tvp)
        display = _make_image_display()

        # Lasso at canvas (0, 0) – (10, 10) does not overlap tile
        verts = _square_lasso(0.0, 0.0, 10.0, 10.0)
        result = self._call(display, viewer, verts)

        self.assertEqual(len(result), 0)

    def test_result_fov_name_correct(self):
        mask = np.zeros((5, 5), dtype=np.int32)
        mask[0:5, 0:5] = 7

        tvp = self._make_tile_viewport(dest_x0=0, dest_y0=0, dest_x1=5, dest_y1=5,
                                       region_xy=(0, 5, 0, 5))
        viewer = self._make_map_viewer(mask, tvp, fov_name="Slide_A")
        display = _make_image_display()

        verts = _square_lasso(-0.5, -0.5, 5.5, 5.5)
        result = self._call(display, viewer, verts)

        fovs = {sel.fov for sel in result}
        self.assertIn("Slide_A", fovs)

    def test_downsampled_tile_cell_selected(self):
        # 20×20 full-res mask; cell 1 in rows 4-12, cols 4-12; ds=2 → tile is 10×10
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[4:13, 4:13] = 1

        # tile placed at canvas (0, 0)-(10, 10) with downsample=2
        tvp = self._make_tile_viewport(0, 0, 10, 10,
                                       region_xy=(0, 20, 0, 20),
                                       downsample_factor=2)
        viewer = self._make_map_viewer(mask, tvp)
        display = _make_image_display()

        # With xmin_px=0, ymin_px=0, ds=2:
        #   data_x = 0 + (0 + col) * 2 = col * 2  →  ranges 0..18 for cols 0..9
        # Lasso covers full tile in data coords:
        verts = _square_lasso(-0.5, -0.5, 20.5, 20.5)
        result = self._call(display, viewer, verts)

        self.assertIn(1, {sel.mask_id for sel in result})

    def test_viewport_offset_corrects_canvas_to_data_coords(self):
        # 5×5 mask; cell at rows/cols 1-3; tile at canvas (0,0)–(5,5), ds=1
        # Viewport is panned: xmin=500, ymin=300
        mask = np.zeros((5, 5), dtype=np.int32)
        mask[1:4, 1:4] = 1

        tvp = self._make_tile_viewport(dest_x0=0, dest_y0=0, dest_x1=5, dest_y1=5,
                                       region_xy=(0, 5, 0, 5), downsample_factor=1)
        viewer = self._make_map_viewer(mask, tvp)
        display = _make_image_display()

        # Override ax to simulate panned viewport (xmin=500, ymin=300)
        display.ax = SimpleNamespace(
            get_xlim=lambda: (500.0, 600.0),
            get_ylim=lambda: (400.0, 300.0),   # inverted; min=300=ymin_px
        )

        # data coords: 500 + (0+col)*1 for col 1–3 → 501–503
        #              300 + (0+row)*1 for row 1–3 → 301–303
        verts = _square_lasso(500.5, 300.5, 503.5, 303.5)
        result = self._call(display, viewer, verts)

        self.assertIn(1, {sel.mask_id for sel in result})


# ---------------------------------------------------------------------------
# on_lasso_selected lifecycle tests
# ---------------------------------------------------------------------------

class TestOnLassoSelected(_PatchAxisLimitsMixin, unittest.TestCase):

    def test_lasso_deactivated_after_completion(self):
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[2:5, 2:5] = 1

        display = _make_image_display()
        viewer = _make_viewer_for_lasso(mask)
        display.main_viewer = viewer

        display._lasso_active = True
        display._lasso_on_complete = None

        display._on_lasso_selected(_square_lasso(1.0, 1.0, 6.0, 6.0))

        self.assertFalse(display._lasso_active)
        self.assertIsNone(display._lasso_selector)

    def test_on_complete_callback_called(self):
        mask = np.zeros((10, 10), dtype=np.int32)

        display = _make_image_display()
        viewer = _make_viewer_for_lasso(mask)
        display.main_viewer = viewer

        called = []
        display._lasso_active = True
        display._lasso_on_complete = lambda: called.append(True)

        # Patch update_patches to avoid matplotlib call
        display.update_patches = lambda: None

        display._on_lasso_selected(_square_lasso(0.0, 0.0, 5.0, 5.0))

        self.assertTrue(called)

    def test_empty_verts_completes_gracefully(self):
        display = _make_image_display()
        viewer = _make_viewer_for_lasso(np.zeros((5, 5), dtype=np.int32))
        display.main_viewer = viewer
        display.update_patches = lambda: None

        display._lasso_active = True
        display._on_lasso_selected([])  # too few verts

        self.assertFalse(display._lasso_active)
        self.assertEqual(len(display.selected_masks_label), 0)


if __name__ == "__main__":
    unittest.main()
