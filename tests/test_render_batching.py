"""One user action, one render: batching of ``update_display`` requests."""

import types
import unittest
from unittest.mock import patch

import numpy as np

from ueler.viewer.image_display import adopt_current_view
from ueler.viewer.main_viewer import ImageMaskViewer, _batched_display_updates


class _Viewer:
    """Runs the real ``update_display`` guard and records the renders behind it.

    With a grid display set, ``update_display`` hands the render to
    ``_update_grid_display`` right after its guards, which makes that call a
    convenient stand-in for "a render happened".
    """

    update_display = ImageMaskViewer.update_display

    def __init__(self):
        self.renders = []
        self.events = []
        self._grid_display = object()
        self._suspend_display_updates = False
        self.current_downsample_factor = 4

    def _update_grid_display(self, factor):
        self.renders.append(factor)
        self.events.append("render")


class BatchedDisplayUpdatesTests(unittest.TestCase):
    def test_outside_a_batch_every_call_renders(self):
        v = _Viewer()
        v.update_display(2)
        v.update_display(4)
        self.assertEqual(v.renders, [2, 4])

    def test_calls_inside_a_batch_render_once_at_the_last_factor(self):
        v = _Viewer()
        with _batched_display_updates(v):
            v.update_display(2)
            v.update_display(4)
            v.update_display(8)
            self.assertEqual(v.renders, [])
        self.assertEqual(v.renders, [8])

    def test_a_batch_without_requests_does_not_render(self):
        v = _Viewer()
        with _batched_display_updates(v):
            pass
        self.assertEqual(v.renders, [])

    def test_nested_batches_render_once_when_the_outermost_exits(self):
        v = _Viewer()
        with _batched_display_updates(v):
            with _batched_display_updates(v):
                v.update_display(2)
            self.assertEqual(v.renders, [])
            v.update_display(4)
        self.assertEqual(v.renders, [4])

    def test_an_absorbing_block_drops_its_own_requests_only(self):
        v = _Viewer()
        with _batched_display_updates(v, render=False):
            v.update_display(2)
        self.assertEqual(v.renders, [])

        with _batched_display_updates(v):
            v.update_display(4)
            with _batched_display_updates(v, render=False):
                v.update_display(8)
        self.assertEqual(v.renders, [4])

    def test_an_error_skips_the_render_and_resets_the_batch(self):
        v = _Viewer()
        with self.assertRaises(RuntimeError):
            with _batched_display_updates(v):
                v.update_display(2)
                raise RuntimeError("boom")
        self.assertEqual(v.renders, [])
        self.assertEqual(v._display_batch_depth, 0)
        v.update_display(4)
        self.assertEqual(v.renders, [4])

    def test_suspended_updates_are_not_recorded(self):
        v = _Viewer()
        with _batched_display_updates(v):
            v._suspend_display_updates = True
            v.update_display(2)
            v._suspend_display_updates = False
        self.assertEqual(v.renders, [])


class CallerTests(unittest.TestCase):
    def test_update_controls_renders_once(self):
        v = _Viewer()

        def rebuild():
            # A reused slider moved by its new range, then the explicit render.
            v.update_display(v.current_downsample_factor)
            v.update_display(v.current_downsample_factor)

        v._rebuild_controls = rebuild
        ImageMaskViewer.update_controls(v, None)
        self.assertEqual(v.renders, [4])

    def test_fov_switch_renders_once_before_the_highlights(self):
        v = _Viewer()
        v.initialized = True

        def load():
            # The channel selection observer and update_controls both ask.
            v.update_display(4)
            v.update_display(4)

        v._load_selected_fov_and_controls = load
        v._reapply_selection_highlights = lambda: v.events.append("highlights")
        v.inform_plugins = lambda _name: v.events.append("plugins")
        canvas = types.SimpleNamespace(toolbar=None)
        v.image_display = types.SimpleNamespace(ax=object(), fig=types.SimpleNamespace(canvas=canvas))

        ImageMaskViewer.on_image_change.__wrapped__(v, None)
        self.assertEqual(v.events[:2], ["render", "highlights"])
        self.assertEqual(v.events.count("render"), 2)  # the switch, then the grid refresh

    def test_a_render_that_loads_channels_does_not_nest_a_second_render(self):
        v = _Viewer()
        v.image_cache = {"FOV_001": {"ch1": np.zeros((2, 2), dtype=np.uint16)}}
        v.ui_component = types.SimpleNamespace(
            color_controls={"ch1": types.SimpleNamespace(value="Red")},
            contrast_min_controls={"ch1": types.SimpleNamespace(value=0.0)},
            contrast_max_controls={"ch1": types.SimpleNamespace(value=1.0)},
            mask_display_controls={},
            mask_color_controls={},
        )
        v.predefined_colors = {"Red": "#FF0000"}
        v.annotations_available = False
        v.annotation_display_enabled = False
        v.active_annotation_name = None
        v.masks_available = False
        v.image_display = types.SimpleNamespace(selected_cells=[])
        v.is_no_image_mode_enabled = lambda: False
        v._is_mask_painter_enabled = lambda: False
        v._get_mask_painter = lambda: None
        loads = []

        def load_fov(_fov, _channels):
            # First load of a channel clamps its contrast slider, whose
            # observer asks for a render.
            loads.append(_fov)
            v.update_display(1)

        v.load_fov = load_fov
        with patch("ueler.viewer.main_viewer.render_fov_to_array", return_value=np.zeros((2, 2, 3), dtype=np.float32)):
            ImageMaskViewer._compose_fov_image(v, "FOV_001", ("ch1",), 1, (0, 2, 0, 2), (0, 2, 0, 2))
        self.assertEqual(loads, ["FOV_001"])
        self.assertEqual(v.renders, [])
        self.assertEqual(getattr(v, "_display_batch_depth", 0), 0)


class AdoptCurrentViewTests(unittest.TestCase):
    def _display(self, initialized=True):
        factors = []
        viewer = types.SimpleNamespace(
            initialized=initialized,
            current_downsample_factor=1,
            on_downsample_factor_changed=factors.append,
            ui_component=types.SimpleNamespace(enable_downsample_checkbox=types.SimpleNamespace(value=True)),
        )
        ax = types.SimpleNamespace(
            get_xlim=lambda: (0.0, 2048.0),
            get_ylim=lambda: (2048.0, 0.0),
            bbox=types.SimpleNamespace(width=600.0, height=600.0),
        )
        return types.SimpleNamespace(main_viewer=viewer, ax=ax), factors

    def test_records_the_view_and_picks_its_factor(self):
        display, factors = self._display()
        adopt_current_view(display)
        self.assertEqual(
            (display.prev_center_x, display.prev_center_y, display.prev_viewport_width, display.prev_viewport_height),
            (1024.0, 1024.0, 2048.0, 2048.0),
        )
        self.assertEqual(len(factors), 1)
        self.assertGreater(factors[0], 1)  # a 2048 px view on a 600 px canvas is downsampled

    def test_before_start_up_finishes_it_leaves_the_first_draw_alone(self):
        display, factors = self._display(initialized=False)
        adopt_current_view(display)
        self.assertFalse(hasattr(display, "prev_center_x"))
        self.assertEqual(factors, [])


if __name__ == "__main__":
    unittest.main()
