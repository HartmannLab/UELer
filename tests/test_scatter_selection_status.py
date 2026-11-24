"""The status line above the scatter plots: lasso tip, what a selection reached,
Trace failures and a missing widget front end."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

from tests.test_issue135_follow_main_viewer_selection import (
    _FakeScatterView,
    _make_viewer,
    _selection,
)
from ueler.viewer.plugin.chart import ChartDisplay


class ScatterSelectionStatusTests(unittest.TestCase):
    def setUp(self):
        self.viewer = _make_viewer()
        self.chart = ChartDisplay(self.viewer, width=4, height=3)
        self.chart._sync_mask_highlights_from_selection = lambda _indices: None
        self.chart._scatter_views["scatter-1"] = _FakeScatterView("scatter-1")
        self.ui = self.chart.ui_component

    def status(self):
        return self.chart._selection_status.value

    def test_no_plots_no_line(self):
        self.chart._scatter_views.clear()
        self.chart._update_selection_status(set(), "external")
        self.assertEqual(self.status(), "")

    def test_tip_while_nothing_is_selected(self):
        self.chart._commit_scatter_selection(set(), origin="scatter")
        self.assertIn("lasso", self.status())
        self.assertIn("Shift", self.status())

    def test_unlinked_selection_points_to_linked_plugins(self):
        self.ui.mv_linked_checkbox.value = False
        self.ui.cell_gallery_linked_checkbox.value = False
        self.chart._commit_scatter_selection({0, 1}, origin="scatter")
        self.assertIn("2 cells selected", self.status())
        self.assertIn("Linked plugins", self.status())

    def test_linked_selection_counts_cells_in_the_fov_shown(self):
        self.ui.mv_linked_checkbox.value = True
        self.ui.cell_gallery_linked_checkbox.value = False
        self.chart._commit_scatter_selection({0, 3, 4}, origin="scatter")
        self.assertIn("3 cells selected", self.status())
        self.assertIn("1 of them is in the FOV shown", self.status())

    def test_linked_selection_outside_the_fov_shown_says_so(self):
        self.ui.mv_linked_checkbox.value = True
        self.ui.cell_gallery_linked_checkbox.value = False
        self.chart._commit_scatter_selection({3, 4}, origin="scatter")
        self.assertIn("None of them is in the FOV shown", self.status())

    def test_map_mode_does_not_count_per_fov(self):
        self.viewer._map_mode_active = True
        self.ui.mv_linked_checkbox.value = True
        self.ui.cell_gallery_linked_checkbox.value = False
        self.chart._commit_scatter_selection({3, 4}, origin="scatter")
        self.assertIn("Outlined in the image.", self.status())

    def test_gallery_link_is_named(self):
        self.ui.mv_linked_checkbox.value = False
        self.ui.cell_gallery_linked_checkbox.value = True
        self.chart._commit_scatter_selection({1}, origin="scatter")
        self.assertIn("1 cell selected", self.status())
        self.assertIn("Gallery", self.status())

    def test_selection_followed_from_the_image(self):
        self.ui.follow_mv_checkbox.value = True
        self.viewer.image_display.select(_selection("fov1", 2))
        self.chart.on_selection_change()
        self.assertIn("1 cell selected", self.status())
        self.assertIn("selected in the image", self.status())

    def test_trace_without_an_image_selection_is_reported_in_the_panel(self):
        self.chart.trace_cells(None)
        self.assertIn("Trace found no cells", self.status())

    def test_frontend_problem_replaces_the_tip(self):
        self.viewer.widget_frontend_status = SimpleNamespace(is_problem=True, message="assets <missing>")
        self.chart._update_selection_status(set(), "external")
        self.assertIn("empty boxes", self.status())
        self.assertIn("assets &lt;missing&gt;", self.status())

    def test_healthy_frontend_shows_the_tip(self):
        self.viewer.widget_frontend_status = SimpleNamespace(is_problem=False, message="")
        self.chart._update_selection_status(set(), "external")
        self.assertIn("lasso", self.status())


if __name__ == "__main__":
    unittest.main()
