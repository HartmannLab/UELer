"""Tests for the per-channel grid display mode (Issue #76)."""

import math
import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np

import tests.bootstrap  # noqa: F401  # registers lightweight stubs

# ---------------------------------------------------------------------------
# Lightweight stubs required before importing ueler modules
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt

from ueler.viewer.channel_grid_view import GridChannelDisplay


# ---------------------------------------------------------------------------
# Shared dummy axes / viewer helpers (mirrors test_map_mode_activation style)
# ---------------------------------------------------------------------------


class _DummyAxes:
    def __init__(self, xlim=(0.0, 100.0), ylim=(100.0, 0.0)):
        self._xlim = xlim
        self._ylim = ylim

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim

    def set_xlim(self, xlim):
        self._xlim = xlim

    def set_ylim(self, ylim):
        self._ylim = ylim


class _DummyImageDisplay:
    def __init__(self):
        self.ax = _DummyAxes()


class _DummyUiComponent:
    def __init__(self):
        self.enable_downsample_checkbox = SimpleNamespace(value=True)


class _DummyViewer:
    """Minimal stub for ImageMaskViewer used by GridChannelDisplay tests."""

    def __init__(self, width=100, height=80):
        self.width = width
        self.height = height
        self.initialized = True
        self.current_downsample_factor = 1
        self.image_display = _DummyImageDisplay()
        self.ui_component = _DummyUiComponent()
        self._update_grid_calls = []

    def on_downsample_factor_changed(self, factor):
        self.current_downsample_factor = factor

    def _update_grid_display(self, factor):
        self._update_grid_calls.append(factor)


# ---------------------------------------------------------------------------
# GridChannelDisplay unit tests
# ---------------------------------------------------------------------------


class GridChannelDisplayTests(unittest.TestCase):
    def setUp(self):
        plt.close("all")
        self.viewer = _DummyViewer(width=100, height=80)
        self.channels = ["A", "B", "C"]
        self.xlim = (0.0, 100.0)
        self.ylim = (80.0, 0.0)

    def tearDown(self):
        plt.close("all")

    # --- subplot count ---

    def test_creates_correct_number_of_img_artists(self):
        grid = GridChannelDisplay(self.viewer, self.channels, self.xlim, self.ylim)
        self.assertEqual(len(grid.img_artists), len(self.channels))

    def test_creates_at_least_n_axes(self):
        grid = GridChannelDisplay(self.viewer, self.channels, self.xlim, self.ylim)
        # Total axes in the grid must be >= channel count (extras are hidden)
        self.assertGreaterEqual(len(grid.axes), len(self.channels))

    def test_single_channel_produces_one_artist(self):
        grid = GridChannelDisplay(self.viewer, ["OnlyCh"], self.xlim, self.ylim)
        self.assertEqual(len(grid.img_artists), 1)

    def test_extra_axes_hidden(self):
        # 3 channels → 2×2 grid → 1 extra axis should be hidden
        grid = GridChannelDisplay(self.viewer, self.channels, self.xlim, self.ylim)
        n = len(self.channels)
        ncols = max(1, math.ceil(math.sqrt(n)))
        nrows = max(1, math.ceil(n / ncols))
        total = nrows * ncols
        visible_count = sum(1 for ax in grid.axes if ax.get_visible())
        self.assertEqual(visible_count, n)
        self.assertEqual(total - visible_count, total - n)

    # --- viewport ---

    def test_get_viewport_returns_initial_limits(self):
        grid = GridChannelDisplay(self.viewer, self.channels, self.xlim, self.ylim)
        xlim, ylim = grid.get_viewport()
        self.assertAlmostEqual(xlim[0], self.xlim[0])
        self.assertAlmostEqual(xlim[1], self.xlim[1])
        self.assertAlmostEqual(ylim[0], self.ylim[0])
        self.assertAlmostEqual(ylim[1], self.ylim[1])

    # --- channel labels ---

    def test_channel_labels_present_in_axes_texts(self):
        grid = GridChannelDisplay(self.viewer, self.channels, self.xlim, self.ylim)
        for i, ch in enumerate(self.channels):
            ax = grid.axes[i]
            texts = [t.get_text() for t in ax.texts]
            self.assertIn(
                ch,
                texts,
                msg=f"Channel label '{ch}' not found in axes[{i}].texts: {texts}",
            )

    # --- update_panes ---

    def test_update_panes_sets_imshow_data(self):
        grid = GridChannelDisplay(self.viewer, ["X", "Y"], self.xlim, self.ylim)
        arrays = {
            "X": np.ones((10, 10, 3), dtype=np.float32),
            "Y": np.zeros((10, 10, 3), dtype=np.float32),
        }
        grid.update_panes(arrays, (0, 10, 0, 10))

        data_x = grid.img_artists[0].get_array()
        data_y = grid.img_artists[1].get_array()
        # After set_data, get_array() returns the supplied array
        np.testing.assert_array_equal(data_x[:, :, 0], np.ones((10, 10)))
        np.testing.assert_array_equal(data_y[:, :, 0], np.zeros((10, 10)))

    def test_update_panes_sets_extent(self):
        grid = GridChannelDisplay(self.viewer, ["A"], self.xlim, self.ylim)
        arr = np.zeros((5, 5, 3), dtype=np.float32)
        region_xy = (10, 50, 20, 70)
        grid.update_panes({"A": arr}, region_xy)
        # extent should be (xmin, xmax, ymax, ymin) = (10, 50, 70, 20) for origin='upper'
        extent = grid.img_artists[0].get_extent()
        self.assertEqual(tuple(extent), (10, 50, 70, 20))

    def test_update_panes_missing_channel_is_ignored(self):
        """update_panes should silently ignore channels absent from the dict."""
        grid = GridChannelDisplay(self.viewer, ["A", "B"], self.xlim, self.ylim)
        # Provide only A; B is missing
        grid.update_panes({"A": np.zeros((5, 5, 3), dtype=np.float32)}, (0, 10, 0, 10))
        # If no exception raised, the test passes

    # --- _on_draw syncs image_display.ax ---

    def test_on_draw_syncs_viewer_image_display_ax(self):
        grid = GridChannelDisplay(self.viewer, ["A"], self.xlim, self.ylim)
        # Simulate a viewport change on the grid's primary axes
        new_xlim = (10.0, 90.0)
        new_ylim = (70.0, 5.0)
        grid.axes[0].set_xlim(new_xlim)
        grid.axes[0].set_ylim(new_ylim)
        # Call _on_draw directly (bypasses the canvas event mechanism)
        grid._on_draw(None)
        self.assertEqual(self.viewer.image_display.ax.get_xlim(), new_xlim)
        self.assertEqual(self.viewer.image_display.ax.get_ylim(), new_ylim)

    def test_on_draw_calls_update_grid_display(self):
        grid = GridChannelDisplay(self.viewer, ["A"], self.xlim, self.ylim)
        # Reset previous center so the change check passes
        grid._prev_cx = None
        grid._prev_cy = None
        grid.axes[0].set_xlim((5.0, 80.0))
        grid.axes[0].set_ylim((78.0, 2.0))
        grid._on_draw(None)
        self.assertGreater(len(self.viewer._update_grid_calls), 0)

    def test_on_draw_noop_when_viewport_unchanged(self):
        grid = GridChannelDisplay(self.viewer, ["A"], self.xlim, self.ylim)
        # First call records the center
        grid._on_draw(None)
        initial_calls = len(self.viewer._update_grid_calls)
        # Second call with same viewport should be skipped
        grid._on_draw(None)
        self.assertEqual(len(self.viewer._update_grid_calls), initial_calls)


# ---------------------------------------------------------------------------
# on_grid_view_toggle integration (uses a heavier mock viewer)
# ---------------------------------------------------------------------------


class _DummyOutput:
    """Minimal ipywidgets.Output substitute."""

    def __init__(self):
        self.layout = SimpleNamespace(display="")
        self._cleared = False
        self._shown = False

    def clear_output(self, wait=False):
        self._cleared = True

    def __enter__(self):
        self._shown = True
        return self

    def __exit__(self, *_):
        return False


class ToggleViewerMixin:
    """Builds a viewer stub suitable for on_grid_view_toggle tests."""

    def _make_toggle_viewer(self, channels=("R", "G", "B")):
        viewer = SimpleNamespace()
        viewer.initialized = True
        viewer.current_downsample_factor = 1
        viewer.width = 64
        viewer.height = 64
        # image_display with real axes
        ax = _DummyAxes()
        viewer.image_display = SimpleNamespace(
            ax=ax,
            fig=SimpleNamespace(
                canvas=SimpleNamespace(draw_idle=lambda: None, toolbar=None)
            ),
        )
        viewer.image_output = _DummyOutput()
        viewer.grid_output = _DummyOutput()
        viewer._grid_display = None
        # ui_component
        visibility_controls = {ch: SimpleNamespace(value=True) for ch in channels}
        viewer.ui_component = SimpleNamespace(
            channel_selector=SimpleNamespace(value=tuple(channels)),
            channel_visibility_controls=visibility_controls,
            enable_downsample_checkbox=SimpleNamespace(value=True),
            image_selector=SimpleNamespace(value="fov1"),
            grid_view_checkbox=SimpleNamespace(value=False, disabled=False),
        )
        # Bind methods from ImageMaskViewer to the stub
        from ueler.viewer.main_viewer import ImageMaskViewer

        viewer._get_visible_channels = lambda chs: ImageMaskViewer._get_visible_channels(
            viewer, chs
        )
        # _update_grid_display is a no-op; we just track calls
        viewer._updates = []
        viewer._update_grid_display = lambda f: viewer._updates.append(f)
        viewer.on_downsample_factor_changed = lambda f: setattr(
            viewer, "current_downsample_factor", f
        )
        viewer.update_display = lambda f: viewer._updates.append(("update_display", f))
        return viewer

    @staticmethod
    def _fire_toggle(viewer, new_value):
        """Call on_grid_view_toggle directly with a fabricated change dict."""
        from ueler.viewer.main_viewer import ImageMaskViewer

        ImageMaskViewer.on_grid_view_toggle(viewer, {"new": new_value})


class GridViewToggleTests(ToggleViewerMixin, unittest.TestCase):
    def test_activate_hides_image_output(self):
        viewer = self._make_toggle_viewer()
        self._fire_toggle(viewer, True)
        self.assertEqual(viewer.image_output.layout.display, "none")

    def test_activate_shows_grid_output(self):
        viewer = self._make_toggle_viewer()
        self._fire_toggle(viewer, True)
        self.assertEqual(viewer.grid_output.layout.display, "")

    def test_activate_creates_grid_display(self):
        viewer = self._make_toggle_viewer()
        self._fire_toggle(viewer, True)
        self.assertIsNotNone(viewer._grid_display)

    def test_activate_grid_display_has_correct_channels(self):
        channels = ("DAPI", "CD3", "CD8")
        viewer = self._make_toggle_viewer(channels=channels)
        self._fire_toggle(viewer, True)
        self.assertEqual(viewer._grid_display.channels, list(channels))

    def test_deactivate_shows_image_output(self):
        viewer = self._make_toggle_viewer()
        self._fire_toggle(viewer, True)
        viewer.image_output.layout.display = "none"  # ensure it was hidden
        self._fire_toggle(viewer, False)
        self.assertEqual(viewer.image_output.layout.display, "")

    def test_deactivate_hides_grid_output(self):
        viewer = self._make_toggle_viewer()
        self._fire_toggle(viewer, True)
        self._fire_toggle(viewer, False)
        self.assertEqual(viewer.grid_output.layout.display, "none")

    def test_deactivate_clears_grid_display(self):
        viewer = self._make_toggle_viewer()
        self._fire_toggle(viewer, True)
        self._fire_toggle(viewer, False)
        self.assertIsNone(viewer._grid_display)

    def test_deactivate_syncs_viewport_to_image_display(self):
        viewer = self._make_toggle_viewer()
        self._fire_toggle(viewer, True)
        # Manually move the grid viewport
        grid = viewer._grid_display
        new_xlim = (5.0, 60.0)
        new_ylim = (55.0, 5.0)
        grid.axes[0].set_xlim(new_xlim)
        grid.axes[0].set_ylim(new_ylim)
        self._fire_toggle(viewer, False)
        self.assertEqual(viewer.image_display.ax.get_xlim(), new_xlim)
        self.assertEqual(viewer.image_display.ax.get_ylim(), new_ylim)

    def test_deactivate_calls_update_display(self):
        viewer = self._make_toggle_viewer()
        self._fire_toggle(viewer, True)
        viewer._updates.clear()
        self._fire_toggle(viewer, False)
        self.assertTrue(
            any(True for item in viewer._updates if isinstance(item, tuple) and item[0] == "update_display"),
            msg="Expected update_display call after deactivating grid mode",
        )


# ---------------------------------------------------------------------------
# update_display routing
# ---------------------------------------------------------------------------


class UpdateDisplayRoutingTests(ToggleViewerMixin, unittest.TestCase):
    """When _grid_display is set, update_display must route to _update_grid_display."""

    def test_update_display_routes_to_grid_when_active(self):
        from ueler.viewer.main_viewer import ImageMaskViewer

        viewer = self._make_toggle_viewer()
        # Activate grid mode so _grid_display is populated
        self._fire_toggle(viewer, True)
        viewer._updates.clear()

        # Call update_display via the unbound method
        ImageMaskViewer.update_display(viewer, 2)

        # _update_grid_display should have been called; update_display should NOT
        # have fallen through to the composited rendering path (which would
        # attempt to access viewer.ui_component.channel_selector etc.).
        self.assertIn(2, viewer._updates)


if __name__ == "__main__":
    unittest.main()
