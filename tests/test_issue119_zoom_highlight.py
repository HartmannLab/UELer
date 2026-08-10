"""Issue #119 follow-up — the highlight must survive a zoom / pan.

The selection outline is not a Matplotlib artist: ``ImageDisplay.update_patches``
paints white pixels *into* the RGB array and pushes it with ``set_data``.
``update_display`` used to call ``update_patches`` **before** installing the newly
rendered ``combined`` array, so every zoom or pan (which reaches ``update_display``
through ``ImageDisplay.on_draw``) outlined the *previous* viewport's array and then
immediately overwrote it — the highlight flashed and vanished.  The outlines are now
painted after ``set_data(combined)``.

These tests therefore run the real ``ImageMaskViewer.update_display`` *and* the real
``ImageDisplay.update_patches`` against a stub viewer, and assert on the pixels of
the array that ends up on screen — the ordering bug is invisible to a test that only
inspects ``selected_masks_label``.
"""

import unittest
from types import SimpleNamespace

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

import numpy as np

# Module scope on purpose — see tests/test_issue119_selection_across_fov.py: a lazy
# import would run after test_lasso_selection.py installs its ``ueler.image_utils``
# stub, against which ``main_viewer`` cannot be imported.
from ueler.viewer.image_display import ImageDisplay, MaskSelection
from ueler.viewer.main_viewer import ImageMaskViewer


MASK_NAME = "whole_cell"
FOV = "FOV1"
SIZE = 8
WHITE = 1.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _label_mask() -> np.ndarray:
    """An 8x8 label image with cell 1 as a 4x4 block and cell 2 in the corner."""
    mask = np.zeros((SIZE, SIZE), dtype=np.int32)
    mask[1:5, 1:5] = 1
    mask[6:8, 6:8] = 2
    return mask


class _DummyAxes:
    def __init__(self, xlim=(0.0, float(SIZE)), ylim=(float(SIZE), 0.0)):
        self._xlim = xlim
        self._ylim = ylim

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim


class _DummyImgArtist:
    """Records every ``set_data`` so the test can inspect what ended up on screen."""

    def __init__(self):
        self.data_history = []
        self.extents = []

    def set_data(self, data):
        self.data_history.append(np.array(data, copy=True))

    def get_array(self):
        return self.data_history[-1]

    def set_extent(self, extent):
        self.extents.append(extent)

    @property
    def displayed(self):
        return self.data_history[-1]


def _make_display(viewer):
    """A real ``ImageDisplay`` — ``update_patches`` included — on dummy Matplotlib."""
    display = ImageDisplay.__new__(ImageDisplay)
    display.main_viewer = viewer
    display.selected_masks_label = set()
    display.ax = _DummyAxes()
    display.img_display = _DummyImgArtist()
    display.draw_calls = []
    display.fig = SimpleNamespace(
        canvas=SimpleNamespace(draw_idle=lambda: display.draw_calls.append(True))
    )
    display.combined = None
    viewer.image_display = display
    return display


def _make_viewer(*, map_mode=False, selected=(1,)):
    viewer = SimpleNamespace()
    viewer.width = SIZE
    viewer.height = SIZE
    viewer.current_downsample_factor = 1
    viewer.mask_outline_thickness = 1
    viewer.masks_available = True
    viewer.initialized = True
    viewer._suspend_display_updates = False
    viewer._grid_display = None
    viewer._widget_displayed = True
    viewer._map_mode_active = map_mode
    viewer._active_map_id = "map1" if map_mode else None
    viewer._visible_map_fovs = ()
    viewer._last_viewport_px = None
    viewer._debug = False
    viewer.linked_selection_indices = None
    viewer.current_label_masks = {}
    viewer.full_resolution_label_masks = {}
    viewer.label_mask = _label_mask()

    viewer.ui_component = SimpleNamespace(
        channel_selector=SimpleNamespace(value=("CD3",)),
        image_selector=SimpleNamespace(value=FOV),
        mask_display_controls={MASK_NAME: SimpleNamespace(value=True)},
    )

    viewer.render_calls = []
    viewer.informed = []
    viewer.map_highlight_calls = []

    def _render_image(channels, factor, xym, xym_ds):
        viewer.render_calls.append((tuple(channels), factor, xym, xym_ds))
        height = max(1, int(xym_ds[3] - xym_ds[2]))
        width = max(1, int(xym_ds[1] - xym_ds[0]))
        # A dark grey render: distinguishable from both black and the white outline.
        return np.full((height, width, 3), 0.25, dtype=np.float32)

    viewer.render_image = _render_image
    viewer._get_visible_channels = lambda channels: list(channels)
    viewer._refresh_channel_legend = lambda channels: None
    viewer.is_no_image_mode_enabled = lambda: False
    viewer._get_label_mask_at_factor = lambda fov, mask, factor: (
        viewer.label_mask[::factor, ::factor] if mask == MASK_NAME else None
    )
    viewer.update_scale_bar = lambda: None
    viewer.inform_plugins = lambda method: viewer.informed.append(method)
    viewer._render_map_view = lambda channels, factor, bounds: (
        np.full((SIZE, SIZE, 3), 0.25, dtype=np.float32),
        (FOV,),
    )
    viewer._apply_map_painter_overlay = lambda: None
    viewer._update_map_mask_highlights = lambda: viewer.map_highlight_calls.append(True)

    display = _make_display(viewer)
    for mask_id in selected:
        display.selected_masks_label.add(
            MaskSelection(fov=FOV, mask=MASK_NAME, mask_id=int(mask_id))
        )
    return viewer


def _update(viewer, factor=1):
    ImageMaskViewer.update_display(viewer, factor)


def _zoom(viewer, xlim, ylim):
    """Move the viewport the way a Matplotlib zoom/pan does, then redraw."""
    viewer.image_display.ax = _DummyAxes(xlim=xlim, ylim=ylim)


def _white_pixels(array):
    return np.argwhere(np.all(np.asarray(array) >= WHITE, axis=-1))


# ---------------------------------------------------------------------------
# The regression itself
# ---------------------------------------------------------------------------
class ZoomKeepsHighlightTestCase(unittest.TestCase):
    def test_outlines_are_on_the_array_left_on_screen(self):
        """The array installed *last* must carry the outline, not the one before it.

        This is the whole bug: pre-fix the final ``set_data`` was the clean render,
        so the highlight was erased microseconds after being drawn.
        """
        viewer = _make_viewer()

        _update(viewer)

        displayed = viewer.image_display.img_display.displayed
        self.assertTrue(
            _white_pixels(displayed).size,
            msg="no outline pixels on the array left on screen after update_display",
        )

    def test_outline_traces_the_selected_cell(self):
        """The outline sits on cell 1's boundary — not cell 2's, which is unselected."""
        viewer = _make_viewer(selected=(1,))

        _update(viewer)

        outlined = {tuple(rc) for rc in _white_pixels(viewer.image_display.img_display.displayed)}
        # ``find_boundaries(mode="inner")`` marks the ring inside the 4x4 block.
        self.assertIn((1, 1), outlined)
        self.assertIn((4, 4), outlined)
        # The block interior stays untouched, and so does the unselected cell 2.
        self.assertNotIn((2, 2), outlined)
        self.assertNotIn((6, 6), outlined)
        self.assertNotIn((7, 7), outlined)

    def test_clean_render_is_kept_as_the_base_image(self):
        """``image_display.combined`` must stay outline-free.

        ``update_patches`` re-derives the outline from ``combined`` every time, so
        baking outlines into it would make them accumulate and survive a clear.
        """
        viewer = _make_viewer()

        _update(viewer)

        self.assertFalse(
            _white_pixels(viewer.image_display.combined).size,
            msg="outlines were baked into the base image",
        )

    def test_repeated_zooms_keep_the_highlight(self):
        """Zoom in, then out again: every redraw must end with the outline visible."""
        viewer = _make_viewer()
        viewports = [
            ((0.0, 8.0), (8.0, 0.0)),
            ((1.0, 6.0), (6.0, 1.0)),
            ((1.0, 7.0), (7.0, 1.0)),
            ((0.0, 8.0), (8.0, 0.0)),
        ]

        for xlim, ylim in viewports:
            _zoom(viewer, xlim, ylim)
            _update(viewer)
            with self.subTest(xlim=xlim, ylim=ylim):
                self.assertTrue(
                    _white_pixels(viewer.image_display.img_display.displayed).size,
                    msg=f"highlight lost at viewport {xlim} / {ylim}",
                )
                self.assertFalse(
                    _white_pixels(viewer.image_display.combined).size,
                    msg="outlines leaked into the base image across redraws",
                )

    def test_outline_is_aligned_to_the_zoomed_viewport(self):
        """After a zoom the outline is placed in the new array's own coordinates.

        Painting before ``set_data`` also meant ``update_patches`` compared its
        expected region size against the *previous* array, so it could fall back to
        the absolute-offset mapping and land the outline outside the visible slice.
        """
        viewer = _make_viewer()
        _zoom(viewer, (1.0, 6.0), (6.0, 1.0))

        _update(viewer)

        displayed = viewer.image_display.img_display.displayed
        self.assertEqual(displayed.shape[:2], (5, 5))  # the padded 1..6 viewport
        outlined = _white_pixels(displayed)
        self.assertTrue(outlined.size, msg="no outline after zooming in")
        # Cell 1 spans rows/cols 1-4, i.e. 0-3 once the viewport starts at 1.
        self.assertTrue((outlined.max(axis=0) <= 3).all())

    def test_zoom_entirely_inside_a_cell_draws_nothing(self):
        """Not a regression: an outline can only be seen where the border is.

        Zoomed past the cell's border there is no boundary pixel in the viewport, so
        the (correct) result is a bare render.  Pinned so the assertions above are not
        mistaken for "an outline is expected at any zoom level".
        """
        viewer = _make_viewer()
        _zoom(viewer, (2.0, 5.0), (5.0, 2.0))

        _update(viewer)

        self.assertFalse(_white_pixels(viewer.image_display.img_display.displayed).size)

    def test_zoom_at_a_coarser_downsample_factor(self):
        """A zoom-out changes the downsample factor; the outline still lands."""
        viewer = _make_viewer()

        _update(viewer, factor=2)

        self.assertTrue(
            _white_pixels(viewer.image_display.img_display.displayed).size,
            msg="highlight lost when the downsample factor changed",
        )


# ---------------------------------------------------------------------------
# Unchanged behaviour
# ---------------------------------------------------------------------------
class UnchangedBehaviourTestCase(unittest.TestCase):
    def test_without_a_selection_the_render_is_shown_as_is(self):
        viewer = _make_viewer(selected=())

        _update(viewer)

        displayed = viewer.image_display.img_display.displayed
        self.assertFalse(_white_pixels(displayed).size)
        np.testing.assert_allclose(displayed, np.full((SIZE, SIZE, 3), 0.25))

    def test_label_masks_are_still_cached_before_the_repaint(self):
        """``update_patches`` needs the full-resolution masks; they must be loaded."""
        viewer = _make_viewer()

        _update(viewer)

        self.assertIn(MASK_NAME, viewer.full_resolution_label_masks)
        self.assertIn(MASK_NAME, viewer.current_label_masks)

    def test_plugins_and_scale_bar_still_notified(self):
        viewer = _make_viewer()

        _update(viewer)

        self.assertIn("on_mv_update_display", viewer.informed)

    def test_map_mode_uses_its_own_highlight_path(self):
        """In map mode ``update_patches`` must not be the one repainting.

        ``update_patches`` delegates to ``_update_map_mask_highlights`` there, which
        ``update_display`` already calls itself.
        """
        viewer = _make_viewer(map_mode=True)
        calls = []
        viewer.image_display.update_patches = lambda *a, **k: calls.append(a)

        _update(viewer)

        self.assertEqual(calls, [])
        self.assertTrue(viewer.map_highlight_calls)

    def test_suspended_updates_still_short_circuit(self):
        viewer = _make_viewer()
        viewer._suspend_display_updates = True

        _update(viewer)

        self.assertEqual(viewer.render_calls, [])
        self.assertEqual(viewer.image_display.img_display.data_history, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
