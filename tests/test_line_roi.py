"""Tests for line and polygon ROIs in the ROI manager.

Covers the geometry helpers, their persistence through the ROI CSV, the
interactive polyline editor on the image display, the shape overlay drawn onto
browser thumbnails, and the batch-export include/exclude filter.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ueler.viewer.image_display import ImageDisplay
from ueler.viewer.roi_manager import (
    ROI_COLUMNS,
    ROI_KIND_SHAPE,
    ROIManager,
    build_shape_fields,
    format_roi_label,
    geometry_bounds,
    is_shape_record,
    parse_geometry,
    polyline_length,
    serialize_geometry,
    shape_display_kind,
)


class _Event:
    """Minimal stand-in for a Matplotlib mouse/key event."""

    def __init__(self, inaxes=None, xdata=None, ydata=None, button=None, key=None):
        self.inaxes = inaxes
        self.xdata = xdata
        self.ydata = ydata
        self.button = button
        self.key = key


class GeometryHelperTests(unittest.TestCase):
    """serialize/parse/bounds/length behaviour for shape geometry."""

    def test_serialize_parse_round_trip(self):
        payload = serialize_geometry([[10, 20], [30.128, 20], [50, 60]])
        parsed = parse_geometry(payload)
        self.assertIsNotNone(parsed)
        self.assertFalse(parsed["closed"])
        self.assertEqual(parsed["type"], "polyline")
        # Vertices are stored at two decimals: sub-hundredth-pixel precision is
        # noise, and rounding keeps the CSV cell short.
        self.assertEqual(
            parsed["points"], [(10.0, 20.0), (30.13, 20.0), (50.0, 60.0)]
        )

    def test_closed_shape_round_trips_as_polygon(self):
        parsed = parse_geometry(serialize_geometry([[0, 0], [4, 0], [4, 4]], closed=True))
        self.assertTrue(parsed["closed"])
        self.assertEqual(parsed["type"], "polygon")

    def test_parse_geometry_rejects_unusable_payloads(self):
        for payload in (None, "", "   ", "nan", float("nan"), "{bad json", "[]", "{}", 17):
            with self.subTest(payload=payload):
                self.assertIsNone(parse_geometry(payload))

    def test_parse_geometry_drops_malformed_points(self):
        payload = json.dumps(
            {"closed": False, "points": [[1, 2], ["a", "b"], [3], [4, 5], [float("inf"), 1]]}
        )
        parsed = parse_geometry(payload)
        self.assertEqual(parsed["points"], [(1.0, 2.0), (4.0, 5.0)])

    def test_parse_geometry_accepts_a_dict(self):
        parsed = parse_geometry({"closed": True, "points": [[0, 0], [1, 1], [2, 0]]})
        self.assertTrue(parsed["closed"])
        self.assertEqual(len(parsed["points"]), 3)

    def test_polyline_length_open_and_closed(self):
        self.assertAlmostEqual(polyline_length([[0, 0], [3, 4]]), 5.0)
        self.assertAlmostEqual(polyline_length([[0, 0], [3, 0], [3, 4]], closed=True), 12.0)
        self.assertEqual(polyline_length([[1, 1]]), 0.0)
        self.assertEqual(polyline_length([]), 0.0)

    def test_horizontal_line_still_gets_a_renderable_box(self):
        """A straight line has zero extent on one axis; the renderers reject that."""
        bounds = geometry_bounds([[10, 20], [30, 20]])
        self.assertGreater(bounds["y_max"] - bounds["y_min"], 0.0)
        self.assertGreaterEqual(bounds["y_max"] - bounds["y_min"], 8.0)
        self.assertGreater(bounds["x_max"], bounds["x_min"])

    def test_vertical_line_still_gets_a_renderable_box(self):
        bounds = geometry_bounds([[20, 10], [20, 30]])
        self.assertGreaterEqual(bounds["x_max"] - bounds["x_min"], 8.0)

    def test_single_point_gets_a_renderable_box(self):
        bounds = geometry_bounds([[50, 50]])
        self.assertGreaterEqual(bounds["x_max"] - bounds["x_min"], 8.0)
        self.assertGreaterEqual(bounds["y_max"] - bounds["y_min"], 8.0)

    def test_bounds_never_go_negative(self):
        bounds = geometry_bounds([[0, 0], [2, 2]])
        self.assertGreaterEqual(bounds["x_min"], 0.0)
        self.assertGreaterEqual(bounds["y_min"], 0.0)

    def test_bounds_are_clamped_to_the_canvas(self):
        bounds = geometry_bounds([[98, 98], [99, 99]], limit=(100.0, 100.0))
        self.assertLessEqual(bounds["x_max"], 100.0)
        self.assertLessEqual(bounds["y_max"], 100.0)
        self.assertGreaterEqual(bounds["x_min"], 0.0)

    def test_bounds_centre_and_size_agree_with_the_corners(self):
        bounds = geometry_bounds([[10, 20], [40, 80]])
        self.assertAlmostEqual(bounds["width"], bounds["x_max"] - bounds["x_min"])
        self.assertAlmostEqual(bounds["height"], bounds["y_max"] - bounds["y_min"])
        self.assertAlmostEqual(bounds["x"], (bounds["x_min"] + bounds["x_max"]) / 2.0)
        self.assertAlmostEqual(bounds["y"], (bounds["y_min"] + bounds["y_max"]) / 2.0)

    def test_geometry_bounds_returns_none_without_points(self):
        self.assertIsNone(geometry_bounds([]))
        self.assertIsNone(geometry_bounds("not points"))

    def test_build_shape_fields_produces_kind_geometry_and_box(self):
        fields = build_shape_fields([[10, 20], [30, 20]], closed=False, limit=(512.0, 512.0))
        self.assertEqual(fields["roi_kind"], ROI_KIND_SHAPE)
        self.assertIsNotNone(parse_geometry(fields["geometry"]))
        for key in ("x", "y", "width", "height", "x_min", "x_max", "y_min", "y_max"):
            self.assertIn(key, fields)

    def test_build_shape_fields_returns_none_for_an_empty_gesture(self):
        self.assertIsNone(build_shape_fields([], closed=False))


class ShapeRecordClassificationTests(unittest.TestCase):
    """is_shape_record / shape_display_kind / label decoration."""

    def setUp(self):
        self.geometry = serialize_geometry([[0, 0], [5, 5]])

    def test_explicit_kind_marks_a_shape(self):
        self.assertTrue(is_shape_record({"roi_kind": ROI_KIND_SHAPE, "geometry": self.geometry}))

    def test_geometry_without_a_kind_is_still_a_shape(self):
        """Tolerates a hand-edited CSV that filled geometry but not roi_kind."""
        self.assertTrue(is_shape_record({"roi_kind": "", "geometry": self.geometry}))

    def test_view_records_are_not_shapes(self):
        self.assertFalse(is_shape_record({"roi_kind": "view", "geometry": self.geometry}))
        self.assertFalse(is_shape_record({"roi_kind": "", "geometry": ""}))
        self.assertFalse(is_shape_record({}))
        self.assertFalse(is_shape_record(None))

    def test_display_kind_distinguishes_line_from_polygon(self):
        line = {"roi_kind": ROI_KIND_SHAPE, "geometry": self.geometry}
        polygon = {
            "roi_kind": ROI_KIND_SHAPE,
            "geometry": serialize_geometry([[0, 0], [4, 0], [4, 4]], closed=True),
        }
        self.assertEqual(shape_display_kind(line), "line")
        self.assertEqual(shape_display_kind(polygon), "polygon")
        self.assertEqual(shape_display_kind({"roi_kind": "view"}), "")

    def test_label_names_the_kind_without_breaking_the_suffix(self):
        record = {
            "roi_id": "abcdef1234567890",
            "fov": "FOV1",
            "marker_set": "panel1",
            "tags": "",
            "name": "axis",
            "roi_kind": ROI_KIND_SHAPE,
            "geometry": self.geometry,
        }
        label = format_roi_label(record)
        self.assertIn("line", label)
        self.assertTrue(label.endswith("axis"))

    def test_label_of_a_view_roi_is_unchanged(self):
        record = {
            "roi_id": "abcdef1234567890",
            "fov": "FOV1",
            "marker_set": "panel1",
            "tags": "",
            "name": "",
        }
        self.assertEqual(format_roi_label(record), "FOV1 · panel1 · abcdef12")


class ShapePersistenceTests(unittest.TestCase):
    """Shape ROIs survive the CSV round trip and legacy tables still load."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_geometry_survives_a_save_and_reload(self):
        manager = ROIManager(self.base)
        fields = build_shape_fields([[10, 20], [30.5, 40], [70, 12]], closed=False)
        saved = manager.add_roi({"fov": "FOV1", "name": "axis", **fields})

        reloaded = ROIManager(self.base).get_roi(saved["roi_id"])
        self.assertIsNotNone(reloaded)
        self.assertTrue(is_shape_record(reloaded))
        geometry = parse_geometry(reloaded["geometry"])
        self.assertEqual(
            geometry["points"], [(10.0, 20.0), (30.5, 40.0), (70.0, 12.0)]
        )

    def test_closed_flag_survives_a_save_and_reload(self):
        manager = ROIManager(self.base)
        fields = build_shape_fields([[0, 0], [10, 0], [10, 10]], closed=True)
        saved = manager.add_roi({"fov": "FOV1", **fields})

        reloaded = ROIManager(self.base).get_roi(saved["roi_id"])
        self.assertTrue(parse_geometry(reloaded["geometry"])["closed"])
        self.assertEqual(shape_display_kind(reloaded), "polygon")

    def test_legacy_csv_without_the_new_columns_loads_as_view_rois(self):
        """The columns are new; a table written before them must not break."""
        legacy_columns = [col for col in ROI_COLUMNS if col not in ("roi_kind", "geometry")]
        legacy = pd.DataFrame(
            [{col: ("" if isinstance(col, str) else 0.0) for col in legacy_columns}]
        )
        legacy.loc[0, "roi_id"] = "legacy-1"
        legacy.loc[0, "fov"] = "FOV1"
        os.makedirs(os.path.join(self.base, ".UELer"), exist_ok=True)
        legacy.to_csv(os.path.join(self.base, ".UELer", "roi_manager.csv"), index=False)

        record = ROIManager(self.base).get_roi("legacy-1")
        self.assertIsNotNone(record)
        self.assertEqual(record["roi_kind"], "")
        self.assertEqual(record["geometry"], "")
        self.assertFalse(is_shape_record(record))

    def test_new_columns_are_back_filled_as_text_not_zero(self):
        record = ROIManager(self.base).add_roi({"fov": "FOV1"})
        table = ROIManager(self.base).table
        self.assertEqual(table.loc[0, "geometry"], "")
        self.assertEqual(table.loc[0, "roi_kind"], "")
        self.assertEqual(record["geometry"], "")

    def test_geometry_survives_export_and_import(self):
        manager = ROIManager(self.base)
        fields = build_shape_fields([[1, 2], [3, 4]], closed=False)
        manager.add_roi({"fov": "FOV1", **fields})

        target = os.path.join(self.base, "exported.csv")
        manager.export_to_csv(target)

        fresh = ROIManager(tempfile.mkdtemp(dir=self.base))
        fresh.import_from_csv(target, merge=True)
        shapes = [row for _, row in fresh.table.iterrows() if is_shape_record(row.to_dict())]
        self.assertEqual(len(shapes), 1)
        self.assertEqual(parse_geometry(shapes[0]["geometry"])["points"], [(1.0, 2.0), (3.0, 4.0)])

    def test_a_shape_row_and_a_view_row_coexist(self):
        manager = ROIManager(self.base)
        manager.add_roi({"fov": "FOV1", "x_min": 0.0, "x_max": 10.0})
        manager.add_roi({"fov": "FOV1", **build_shape_fields([[2, 2], [8, 8]])})

        table = ROIManager(self.base).table
        kinds = [is_shape_record(row.to_dict()) for _, row in table.iterrows()]
        self.assertEqual(sorted(kinds), [False, True])


class _EditorHarness:
    """Builds an ImageDisplay carrying only what the polyline code touches.

    ``ImageDisplay.__init__`` needs a full viewer; the polyline editor needs an
    axes, a canvas and its own state, so the instance is assembled directly.
    """

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(100, 0)

        display = ImageDisplay.__new__(ImageDisplay)
        display.fig = self.fig
        display.ax = self.ax
        display.width = 100
        display.height = 100
        display._lasso_active = False
        display._polyline_active = False
        display._polyline_points = []
        display._polyline_closed = False
        display._polyline_cids = []
        display._polyline_artists = []
        display._polyline_undo = []
        display._polyline_redo = []
        display._polyline_on_change = None
        display._polyline_on_finish = None
        display._polyline_drag_index = None
        display._polyline_press = None
        display._polyline_dragged = False
        display._shape_roi_artists = []
        self.display = display

    def close(self):
        plt.close(self.fig)

    def click(self, x, y, button=1):
        """Perform a full press/release click that does not move."""
        from matplotlib.backend_bases import MouseButton

        resolved = MouseButton.LEFT if button == 1 else MouseButton.RIGHT
        event = _Event(inaxes=self.ax, xdata=x, ydata=y, button=resolved)
        self.display._on_polyline_press(event)
        self.display._on_polyline_release(event)

    def drag(self, x0, y0, x1, y1):
        from matplotlib.backend_bases import MouseButton

        press = _Event(inaxes=self.ax, xdata=x0, ydata=y0, button=MouseButton.LEFT)
        self.display._on_polyline_press(press)
        self.display._on_polyline_motion(
            _Event(inaxes=self.ax, xdata=x1, ydata=y1, button=MouseButton.LEFT)
        )
        self.display._on_polyline_release(
            _Event(inaxes=self.ax, xdata=x1, ydata=y1, button=MouseButton.LEFT)
        )

    def key(self, name):
        self.display._on_polyline_key(_Event(key=name))


class PolylineEditorTests(unittest.TestCase):
    """Vertex placement, editing, undo/redo and session lifecycle."""

    def setUp(self):
        self.harness = _EditorHarness()
        self.display = self.harness.display
        self.changes = []
        self.finished = []
        self.display.enable_polyline_editor(
            [], False, self.changes.append, self.finished.append
        )

    def tearDown(self):
        self.harness.close()

    def test_clicks_append_vertices(self):
        self.harness.click(10, 10)
        self.harness.click(40, 40)
        self.harness.click(70, 20)
        self.assertEqual(self.display.polyline_points, [[10, 10], [40, 40], [70, 20]])
        self.assertEqual(len(self.changes), 3)

    def test_dragging_a_vertex_moves_it_without_adding_one(self):
        self.harness.click(10, 10)
        self.harness.click(40, 40)
        self.harness.drag(40, 40, 60, 55)
        self.assertEqual(self.display.polyline_points, [[10, 10], [60, 55]])

    def test_dragging_empty_canvas_adds_nothing(self):
        """A drag that starts off-vertex is a pan gesture, not a new point."""
        self.harness.click(10, 10)
        self.harness.drag(80, 80, 95, 95)
        self.assertEqual(self.display.polyline_points, [[10, 10]])

    def test_right_click_deletes_the_nearest_vertex(self):
        self.harness.click(10, 10)
        self.harness.click(40, 40)
        self.harness.click(70, 20)
        self.harness.click(40, 40, button=3)
        self.assertEqual(self.display.polyline_points, [[10, 10], [70, 20]])

    def test_right_click_far_from_any_vertex_deletes_nothing(self):
        self.harness.click(10, 10)
        self.harness.click(40, 40, button=3)
        self.assertEqual(self.display.polyline_points, [[10, 10]])

    def test_undo_and_redo_step_through_edits(self):
        self.harness.click(10, 10)
        self.harness.click(40, 40)
        self.assertTrue(self.display.undo_polyline())
        self.assertEqual(self.display.polyline_points, [[10, 10]])
        self.assertTrue(self.display.redo_polyline())
        self.assertEqual(self.display.polyline_points, [[10, 10], [40, 40]])

    def test_undo_restores_a_moved_vertex_in_one_step(self):
        self.harness.click(10, 10)
        self.harness.drag(10, 10, 50, 50)
        self.assertTrue(self.display.undo_polyline())
        self.assertEqual(self.display.polyline_points, [[10, 10]])

    def test_undo_on_an_empty_history_reports_failure(self):
        self.assertFalse(self.display.undo_polyline())
        self.assertFalse(self.display.redo_polyline())

    def test_a_click_on_a_vertex_that_does_not_move_leaves_no_undo_step(self):
        self.harness.click(10, 10)
        history = len(self.display._polyline_undo)
        self.harness.click(10, 10)
        self.assertEqual(len(self.display._polyline_undo), history)
        self.assertEqual(self.display.polyline_points, [[10, 10]])

    def test_delete_key_removes_the_last_vertex(self):
        self.harness.click(10, 10)
        self.harness.click(40, 40)
        self.harness.key("delete")
        self.assertEqual(self.display.polyline_points, [[10, 10]])

    def test_enter_finishes_with_the_points(self):
        self.harness.click(10, 10)
        self.harness.click(40, 40)
        self.harness.key("enter")
        self.assertEqual(self.finished, [[[10, 10], [40, 40]]])
        self.assertFalse(self.display._polyline_active)

    def test_escape_cancels_with_none(self):
        self.harness.click(10, 10)
        self.harness.key("escape")
        self.assertEqual(self.finished, [None])
        self.assertFalse(self.display._polyline_active)

    def test_ctrl_z_and_ctrl_y_are_wired(self):
        self.harness.click(10, 10)
        self.harness.click(40, 40)
        self.harness.key("ctrl+z")
        self.assertEqual(self.display.polyline_points, [[10, 10]])
        self.harness.key("ctrl+y")
        self.assertEqual(self.display.polyline_points, [[10, 10], [40, 40]])

    def test_clicks_outside_the_axes_are_ignored(self):
        self.display._on_polyline_press(_Event(inaxes=None, xdata=5, ydata=5, button=1))
        self.assertEqual(self.display.polyline_points, [])

    def test_disable_disconnects_and_clears_artists(self):
        self.harness.click(10, 10)
        self.harness.click(40, 40)
        self.assertTrue(self.display._polyline_artists)
        self.display.disable_polyline_editor()
        self.assertFalse(self.display._polyline_active)
        self.assertEqual(self.display._polyline_artists, [])
        self.assertEqual(self.display._polyline_cids, [])

    def test_editing_prefilled_points_starts_from_them(self):
        self.display.disable_polyline_editor()
        self.display.enable_polyline_editor([[5, 5], [15, 15]], True, None, None)
        self.assertEqual(self.display.polyline_points, [[5, 5], [15, 15]])
        self.assertTrue(self.display._polyline_closed)

    def test_toggling_closed_notifies_and_redraws(self):
        self.harness.click(10, 10)
        self.harness.click(40, 40)
        self.harness.click(40, 10)
        before = len(self.changes)
        self.display.set_polyline_closed(True)
        self.assertTrue(self.display._polyline_closed)
        self.assertEqual(len(self.changes), before + 1)

    def test_a_closed_shape_draws_the_returning_segment(self):
        """The path artist must carry the extra vertex that closes the loop."""
        self.display.disable_polyline_editor()
        self.display.enable_polyline_editor([[0, 0], [10, 0], [10, 10]], True, None, None)
        path_artist = self.display._polyline_artists[0]
        self.assertEqual(len(path_artist.get_xdata()), 4)

    def test_polyline_points_returns_a_copy(self):
        self.harness.click(10, 10)
        snapshot = self.display.polyline_points
        snapshot[0][0] = 999
        self.assertEqual(self.display.polyline_points, [[10, 10]])


class ShapeOverlayArtistTests(unittest.TestCase):
    """Saved shapes are drawn as artists, which survive image redraws."""

    def setUp(self):
        self.harness = _EditorHarness()
        self.display = self.harness.display

    def tearDown(self):
        self.harness.close()

    def test_draw_shape_rois_adds_one_artist_per_shape(self):
        self.display.draw_shape_rois([([[0, 0], [10, 10]], False), ([[20, 20], [30, 30]], False)])
        self.assertEqual(len(self.display._shape_roi_artists), 2)

    def test_redrawing_replaces_rather_than_accumulates(self):
        self.display.draw_shape_rois([([[0, 0], [10, 10]], False)])
        self.display.draw_shape_rois([([[0, 0], [10, 10]], False)])
        self.assertEqual(len(self.display._shape_roi_artists), 1)

    def test_clear_removes_the_overlay(self):
        self.display.draw_shape_rois([([[0, 0], [10, 10]], False)])
        self.display.clear_shape_rois()
        self.assertEqual(self.display._shape_roi_artists, [])

    def test_a_single_point_shape_still_renders(self):
        self.display.draw_shape_rois([([[5, 5]], False)])
        self.assertEqual(len(self.display._shape_roi_artists), 1)

    def test_empty_and_malformed_entries_are_skipped(self):
        self.display.draw_shape_rois([([], False), None, ([[1, 1], [2, 2]], False)])
        self.assertEqual(len(self.display._shape_roi_artists), 1)

    def test_shape_artists_are_not_touched_by_image_data_updates(self):
        """The reason shapes are artists: set_data would erase baked pixels."""
        self.display.draw_shape_rois([([[0, 0], [10, 10]], False)])
        image = self.harness.ax.imshow(np.zeros((20, 20, 3)))
        image.set_data(np.ones((20, 20, 3)))
        self.assertEqual(len(self.display._shape_roi_artists), 1)
        self.assertIn(self.display._shape_roi_artists[0], self.harness.ax.lines)


class ClickGuardTests(unittest.TestCase):
    """Drawing a shape must not also select the cells under the vertices."""

    def setUp(self):
        self.harness = _EditorHarness()
        self.display = self.harness.display
        self.display.main_viewer = SimpleNamespace(
            resolve_mask_hit_at_viewport=self._explode
        )

    def tearDown(self):
        self.harness.close()

    @staticmethod
    def _explode(*_args, **_kwargs):
        raise AssertionError("cell selection ran while drawing a shape")

    def test_mask_selection_is_suppressed_while_drawing(self):
        self.display._polyline_active = True
        self.display.on_mouse_click(_Event(inaxes=self.harness.ax, xdata=5, ydata=5, button=1))

    def test_mask_selection_runs_again_once_drawing_ends(self):
        self.display._polyline_active = False
        with self.assertRaises(AssertionError):
            self.display.on_mouse_click(
                _Event(inaxes=self.harness.ax, xdata=5, ydata=5, button=1)
            )


class ThumbnailShapeOverlayTests(unittest.TestCase):
    """_draw_shape_on_tile stamps the path onto a rendered thumbnail."""

    def setUp(self):
        from ueler.viewer.plugin.roi_manager_plugin import ROIManagerPlugin

        self.plugin = ROIManagerPlugin.__new__(ROIManagerPlugin)

    def _tile(self):
        return np.zeros((40, 40, 3), dtype=np.float32)

    def test_a_horizontal_line_is_stamped_onto_the_tile(self):
        record = {
            "roi_kind": ROI_KIND_SHAPE,
            "geometry": serialize_geometry([[10, 20], [30, 20]]),
            "x_min": 6.0,
            "y_min": 16.0,
        }
        painted = self.plugin._draw_shape_on_tile(self._tile(), record, 1)
        self.assertGreater(painted.sum(), 0.0)
        # y = 20 in image space maps to row 20 - 16 = 4 of the tile.
        self.assertGreater(painted[4, 4:24].sum(), 0.0)

    def test_a_view_roi_tile_is_returned_untouched(self):
        tile = self._tile()
        painted = self.plugin._draw_shape_on_tile(tile, {"roi_kind": "view"}, 1)
        self.assertIs(painted, tile)

    def test_the_source_tile_is_not_modified_in_place(self):
        tile = self._tile()
        record = {
            "roi_kind": ROI_KIND_SHAPE,
            "geometry": serialize_geometry([[10, 20], [30, 20]]),
            "x_min": 6.0,
            "y_min": 16.0,
        }
        self.plugin._draw_shape_on_tile(tile, record, 1)
        self.assertEqual(tile.sum(), 0.0)

    def test_the_downsample_factor_scales_the_path(self):
        record = {
            "roi_kind": ROI_KIND_SHAPE,
            "geometry": serialize_geometry([[0, 0], [80, 0]]),
            "x_min": 0.0,
            "y_min": 0.0,
        }
        painted = self.plugin._draw_shape_on_tile(self._tile(), record, 4)
        self.assertGreater(painted[0, :20].sum(), 0.0)
        self.assertEqual(painted[0, 21:].sum(), 0.0)

    def test_points_outside_the_tile_are_clipped_not_wrapped(self):
        record = {
            "roi_kind": ROI_KIND_SHAPE,
            "geometry": serialize_geometry([[-500, -500], [5, 5]]),
            "x_min": 0.0,
            "y_min": 0.0,
        }
        painted = self.plugin._draw_shape_on_tile(self._tile(), record, 1)
        self.assertGreater(painted.sum(), 0.0)

    def test_a_closed_shape_paints_the_returning_edge(self):
        square = serialize_geometry([[0, 0], [20, 0], [20, 20], [0, 20]], closed=True)
        record = {"roi_kind": ROI_KIND_SHAPE, "geometry": square, "x_min": 0.0, "y_min": 0.0}
        painted = self.plugin._draw_shape_on_tile(self._tile(), record, 1)
        # The closing edge runs down the x = 0 column between the corners.
        self.assertGreater(painted[1:19, 0].sum(), 0.0)


class ExportShapeFilterTests(unittest.TestCase):
    """The batch export ROI list can include or exclude shape ROIs."""

    def setUp(self):
        from ueler.viewer.plugin.export_fovs import BatchExportPlugin

        self._tmp = tempfile.TemporaryDirectory()
        manager = ROIManager(self._tmp.name)
        manager.add_roi({"fov": "FOV1", "name": "view-roi", "x_min": 0.0, "x_max": 20.0})
        manager.add_roi(
            {"fov": "FOV1", "name": "line-roi", **build_shape_fields([[2, 2], [18, 18]])}
        )

        self.plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        self.plugin._roi_records = {}
        self.plugin.main_viewer = SimpleNamespace(
            roi_manager=manager, get_active_fov=lambda: "FOV1"
        )
        self.plugin.ui_component = SimpleNamespace(
            roi_limit_to_fov=SimpleNamespace(value=False),
            roi_include_shapes=SimpleNamespace(value=True),
            roi_selection=SimpleNamespace(options=[], value=()),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _labels(self):
        return [label for label, _ in self.plugin.ui_component.roi_selection.options]

    def test_shapes_are_listed_when_included(self):
        self.plugin.refresh_roi_options()
        labels = self._labels()
        self.assertEqual(len(labels), 2)
        self.assertTrue(any("line-roi" in label for label in labels))

    def test_shapes_are_dropped_when_excluded(self):
        self.plugin.ui_component.roi_include_shapes.value = False
        self.plugin.refresh_roi_options()
        labels = self._labels()
        self.assertEqual(len(labels), 1)
        self.assertTrue(any("view-roi" in label for label in labels))
        self.assertFalse(any("line-roi" in label for label in labels))

    def test_an_excluded_shape_is_not_left_in_the_record_cache(self):
        self.plugin.ui_component.roi_include_shapes.value = False
        self.plugin.refresh_roi_options()
        self.assertTrue(
            all(not is_shape_record(record) for record in self.plugin._roi_records.values())
        )

    def test_an_exported_shape_carries_a_usable_bounding_box(self):
        """Export renders x_min..y_max, so a shape must present a real region."""
        self.plugin.refresh_roi_options()
        shapes = [
            record for record in self.plugin._roi_records.values() if is_shape_record(record)
        ]
        self.assertEqual(len(shapes), 1)
        record = shapes[0]
        self.assertGreater(float(record["x_max"]), float(record["x_min"]))
        self.assertGreater(float(record["y_max"]), float(record["y_min"]))


class PluginShapeHelperTests(unittest.TestCase):
    """Validation, summary text and view-scoped shape lookup in the plugin."""

    def setUp(self):
        from ueler.viewer.plugin.roi_manager_plugin import ROIManagerPlugin

        self._tmp = tempfile.TemporaryDirectory()
        self.manager = ROIManager(self._tmp.name)

        self.plugin = ROIManagerPlugin.__new__(ROIManagerPlugin)
        self.plugin._shape_points = []
        self.plugin._shape_editing = False
        self.plugin._shape_edit_roi_id = None
        self.plugin.ui_component = SimpleNamespace(
            shape_closed_checkbox=SimpleNamespace(value=False),
            shape_show_checkbox=SimpleNamespace(value=True),
            shape_summary=SimpleNamespace(value=""),
        )
        self.plugin.main_viewer = SimpleNamespace(
            roi_manager=self.manager,
            pixel_size_nm=500.0,
            _map_mode_active=False,
            _active_map_id="",
            get_active_fov=lambda: "FOV1",
            image_display=SimpleNamespace(width=512, height=512),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_line_needs_two_vertices_and_a_polygon_three(self):
        self.plugin._shape_points = [[0, 0]]
        self.assertFalse(self.plugin._shape_points_valid())
        self.plugin._shape_points = [[0, 0], [1, 1]]
        self.assertTrue(self.plugin._shape_points_valid())
        self.plugin.ui_component.shape_closed_checkbox.value = True
        self.assertFalse(self.plugin._shape_points_valid())
        self.plugin._shape_points = [[0, 0], [1, 1], [2, 0]]
        self.assertTrue(self.plugin._shape_points_valid())

    def test_the_summary_reports_length_in_pixels_and_microns(self):
        self.plugin._set_shape_summary([[0, 0], [3, 4]], False)
        summary = self.plugin.ui_component.shape_summary.value
        self.assertIn("Line", summary)
        self.assertIn("2 vertices", summary)
        self.assertIn("5.0 px", summary)
        self.assertIn("2.50 µm", summary)

    def test_a_closed_shape_reports_a_perimeter(self):
        self.plugin.ui_component.shape_closed_checkbox.value = True
        self.plugin._set_shape_summary([[0, 0], [3, 0], [3, 4]], True)
        summary = self.plugin.ui_component.shape_summary.value
        self.assertIn("Polygon", summary)
        self.assertIn("perimeter", summary)
        self.assertIn("12.0 px", summary)

    def test_the_summary_omits_microns_without_a_pixel_size(self):
        self.plugin.main_viewer.pixel_size_nm = 0.0
        self.plugin._set_shape_summary([[0, 0], [3, 4]], False)
        self.assertNotIn("µm", self.plugin.ui_component.shape_summary.value)

    def test_map_mode_uses_the_map_pixel_size(self):
        self.plugin.main_viewer._map_mode_active = True
        self.plugin.main_viewer._map_pixel_size_nm = 1000.0
        self.assertEqual(self.plugin._pixel_size_nm(), 1000.0)

    def test_the_canvas_limit_comes_from_the_image_display(self):
        self.assertEqual(self.plugin._canvas_limit(), (512.0, 512.0))
        self.plugin.main_viewer.image_display = SimpleNamespace(width=0, height=0)
        self.assertIsNone(self.plugin._canvas_limit())

    def test_only_shapes_of_the_active_fov_are_returned(self):
        self.manager.add_roi({"fov": "FOV1", **build_shape_fields([[1, 1], [5, 5]])})
        self.manager.add_roi({"fov": "FOV2", **build_shape_fields([[2, 2], [6, 6]])})
        self.manager.add_roi({"fov": "FOV1", "x_min": 0.0, "x_max": 10.0})

        geometries = self.plugin._visible_shape_geometries()
        self.assertEqual(len(geometries), 1)
        self.assertEqual(geometries[0]["points"], [(1.0, 1.0), (5.0, 5.0)])

    def test_map_mode_scopes_shapes_by_map_id(self):
        self.manager.add_roi({"map_id": "MAP1", **build_shape_fields([[1, 1], [5, 5]])})
        self.manager.add_roi({"map_id": "MAP2", **build_shape_fields([[2, 2], [6, 6]])})
        self.plugin.main_viewer._map_mode_active = True
        self.plugin.main_viewer._active_map_id = "MAP1"
        self.plugin.main_viewer.get_active_fov = lambda: None

        geometries = self.plugin._visible_shape_geometries()
        self.assertEqual(len(geometries), 1)
        self.assertEqual(geometries[0]["points"], [(1.0, 1.0), (5.0, 5.0)])

    def test_the_overlay_is_cleared_when_the_checkbox_is_off(self):
        calls = []
        self.plugin.main_viewer.image_display = SimpleNamespace(
            width=512,
            height=512,
            draw_shape_rois=lambda shapes: calls.append(("draw", shapes)),
            clear_shape_rois=lambda: calls.append(("clear", None)),
        )
        self.plugin.ui_component.shape_show_checkbox.value = False
        self.plugin.refresh_shape_overlay()
        self.assertEqual([name for name, _ in calls], ["clear"])

    def test_the_overlay_draws_the_shapes_of_the_current_view(self):
        calls = self._capture_overlay()
        self.manager.add_roi({"fov": "FOV1", **build_shape_fields([[1, 1], [5, 5]])})
        self.plugin.refresh_shape_overlay()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], [([(1.0, 1.0), (5.0, 5.0)], False)])

    def _capture_overlay(self):
        """Record every draw_shape_rois payload, and stub the editor away."""
        calls = []
        self.plugin.main_viewer.image_display = SimpleNamespace(
            width=512,
            height=512,
            draw_shape_rois=calls.append,
            clear_shape_rois=lambda: None,
            finish_polyline=lambda cancel=False: self.plugin._on_shape_edit_finished(
                None if cancel else self.plugin._shape_points
            ),
        )
        return calls


class FinishedShapeStaysVisibleTests(unittest.TestCase):
    """A finished but unsaved shape must not disappear from the canvas.

    Finish removes the editor's own artists, so unless the overlay also draws
    the working copy the shape is invisible until it is saved and read back.
    """

    def setUp(self):
        from ueler.viewer.plugin.roi_manager_plugin import ROIManagerPlugin

        self._tmp = tempfile.TemporaryDirectory()
        self.manager = ROIManager(self._tmp.name)
        self.drawn = []

        self.plugin = ROIManagerPlugin.__new__(ROIManagerPlugin)
        self.plugin._shape_points = []
        self.plugin._shape_editing = False
        self.plugin._shape_edit_roi_id = None
        self.plugin._selected_roi_id = None
        self.plugin.ui_component = SimpleNamespace(
            shape_closed_checkbox=SimpleNamespace(value=False),
            shape_show_checkbox=SimpleNamespace(value=True),
            shape_summary=SimpleNamespace(value=""),
            shape_draw_button=SimpleNamespace(disabled=False),
            shape_edit_button=SimpleNamespace(disabled=False),
            shape_finish_button=SimpleNamespace(disabled=True),
            shape_cancel_button=SimpleNamespace(disabled=True),
            shape_undo_button=SimpleNamespace(disabled=True),
            shape_redo_button=SimpleNamespace(disabled=True),
            shape_save_button=SimpleNamespace(disabled=True),
            name_input=SimpleNamespace(value=""),
            tags=SimpleNamespace(value=()),
            comment=SimpleNamespace(value=""),
            status=SimpleNamespace(value=""),
        )
        self.plugin.main_viewer = SimpleNamespace(
            roi_manager=self.manager,
            pixel_size_nm=0.0,
            current_downsample_factor=1,
            _map_mode_active=False,
            _active_map_id="",
            get_active_fov=lambda: "FOV1",
            image_display=SimpleNamespace(
                width=512,
                height=512,
                draw_shape_rois=self.drawn.append,
                clear_shape_rois=lambda: None,
                finish_polyline=self._finish_polyline,
            ),
        )
        # Only the shape behaviour is under test here.
        self.plugin.refresh_roi_table = lambda *a, **k: None
        self.plugin.set_status = lambda *a, **k: None
        self.plugin._resolve_marker_set_choice = lambda: ""
        self.plugin._get_active_annotation_palette = lambda: ""
        self.plugin._get_active_mask_color_set = lambda: ""
        self.plugin._get_mask_visibility_payload = lambda: ""
        self.plugin._get_mask_painter_payload = lambda: ""

    def tearDown(self):
        self._tmp.cleanup()

    def _finish_polyline(self, cancel=False):
        self.plugin._on_shape_edit_finished(None if cancel else self.plugin._shape_points)

    def test_finishing_an_unsaved_shape_still_draws_it(self):
        self.plugin._shape_editing = True
        self.plugin._shape_points = [[10.0, 10.0], [40.0, 40.0]]
        self.plugin._on_shape_edit_finished(self.plugin._shape_points)

        self.assertTrue(self.drawn)
        self.assertEqual(self.drawn[-1], [([[10.0, 10.0], [40.0, 40.0]], False)])

    def test_the_working_copy_is_not_drawn_while_still_editing(self):
        """During drawing the editor's own artists show it; a second copy would double it."""
        self.plugin._shape_editing = True
        self.plugin._shape_points = [[10.0, 10.0], [40.0, 40.0]]
        self.plugin.refresh_shape_overlay()
        self.assertEqual(self.drawn[-1], [])

    def test_cancelling_removes_the_working_copy_from_the_canvas(self):
        self.plugin._shape_editing = True
        self.plugin._shape_points = [[10.0, 10.0], [40.0, 40.0]]
        self.plugin._on_shape_edit_finished(None)
        self.assertEqual(self.drawn[-1], [])

    def test_editing_a_saved_shape_hides_its_stale_copy(self):
        saved = self.manager.add_roi(
            {"fov": "FOV1", **build_shape_fields([[1, 1], [5, 5]])}
        )
        self.plugin._shape_edit_roi_id = saved["roi_id"]
        self.plugin._shape_points = [[1.0, 1.0], [9.0, 9.0]]
        self.plugin._shape_editing = False
        self.plugin.refresh_shape_overlay()
        # Only the working copy, not the version still stored in the table.
        self.assertEqual(self.drawn[-1], [([[1.0, 1.0], [9.0, 9.0]], False)])

    def test_saving_releases_the_working_copy_so_the_shape_is_drawn_once(self):
        self.plugin._shape_points = [[10.0, 10.0], [40.0, 40.0]]
        self.plugin._on_shape_save(None)

        self.assertEqual(self.plugin._shape_points, [])
        self.assertEqual(self.drawn[-1], [([(10.0, 10.0), (40.0, 40.0)], False)])

    def test_saving_while_still_drawing_finishes_first(self):
        """Forgetting to click Finish must not silently save nothing."""
        self.plugin._shape_editing = True
        self.plugin._shape_points = [[10.0, 10.0], [40.0, 40.0]]
        self.plugin._on_shape_save(None)

        self.assertFalse(self.plugin._shape_editing)
        table = self.manager.table
        self.assertEqual(len(table), 1)
        self.assertTrue(is_shape_record(table.iloc[0].to_dict()))

    def test_the_summary_describes_the_shape_after_saving(self):
        self.plugin._shape_points = [[0.0, 0.0], [3.0, 4.0]]
        self.plugin._on_shape_save(None)
        summary = self.plugin.ui_component.shape_summary.value
        self.assertIn("Saved", summary)
        self.assertIn("5.0 px", summary)

    def test_a_hidden_overlay_stays_hidden_after_finishing(self):
        cleared = []
        self.plugin.main_viewer.image_display.clear_shape_rois = lambda: cleared.append(True)
        self.plugin.ui_component.shape_show_checkbox.value = False
        self.plugin._shape_editing = True
        self.plugin._shape_points = [[10.0, 10.0], [40.0, 40.0]]
        self.plugin._on_shape_edit_finished(self.plugin._shape_points)
        self.assertTrue(cleared)
        self.assertEqual(self.drawn, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
