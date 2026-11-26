"""The viewer's current selection (SelectionHub) and the in-line selection bar.

Every view that makes a selection publishes it to one hub: the image (click,
lasso), the scatter plot, a histogram gate and a heatmap cluster. The bar above
the image shows the latest one, where it already went, and sends it on.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

import pandas as pd

from tests.test_issue135_follow_main_viewer_selection import (
    _FakeScatterView,
    _make_viewer,
    _selection,
)
from ueler.viewer.main_viewer import ImageMaskViewer
from ueler.viewer.plugin import _chart_common
from ueler.viewer.plugin.chart import ChartDisplay
from ueler.viewer.plugin.histogram import HistogramDisplay
from ueler.viewer.selection import CellSelection, SelectionHub
from ueler.viewer.selection_bar import SelectionBar


class SelectionHubTests(unittest.TestCase):
    def setUp(self):
        self.hub = SelectionHub()
        self.seen = []
        self.hub.subscribe(self.seen.append)

    def test_latest_publish_wins(self):
        self.hub.publish({1, 2}, "scatter", {"image"})
        self.hub.publish({3}, "histogram")
        self.assertEqual(self.hub.current, CellSelection(frozenset({3}), "histogram", frozenset()))
        self.assertEqual(len(self.seen), 2)

    def test_empty_publish_clears(self):
        self.hub.publish({1}, "scatter")
        self.hub.publish(set(), "scatter")
        self.assertEqual(len(self.hub.current), 0)
        self.assertEqual(self.hub.current.source, "")

    def test_clear_for_a_source_leaves_another_sources_selection(self):
        self.hub.publish({1}, "histogram")
        self.hub.clear("scatter")
        self.assertEqual(self.hub.current.source, "histogram")
        self.hub.clear("histogram")
        self.assertEqual(len(self.hub.current), 0)

    def test_mark_delivered(self):
        self.hub.publish({1}, "heatmap", {"gallery"})
        self.hub.mark_delivered("image")
        self.assertEqual(self.hub.current.delivered, {"gallery", "image"})

    def test_mark_removed(self):
        self.hub.publish({1}, "heatmap", {"gallery", "image"})
        self.hub.mark_removed("gallery")
        self.assertEqual(self.hub.current.delivered, {"image"})
        self.assertEqual(len(self.hub.current), 1)

    def test_mark_delivered_without_a_selection_does_nothing(self):
        self.hub.mark_delivered("image")
        self.assertEqual(self.seen, [])

    def test_source_label(self):
        self.assertEqual(CellSelection(frozenset({1}), "histogram").source_label, "histogram gate")


class SelectionBarTests(unittest.TestCase):
    def setUp(self):
        self.bar = SelectionBar(SimpleNamespace())

    def test_empty_bar_hides_the_actions(self):
        self.bar.show_selection("", (), False)
        self.assertEqual(self.bar.text.value, "")
        self.assertEqual(self.bar.actions.layout.display, "none")

    def test_summary_and_delivered_targets(self):
        self.bar.show_selection("<b>3 cells</b> · scatter plot", {"gallery"}, True, "plots")
        self.assertIn("3 cells", self.bar.text.value)
        self.assertEqual(self.bar.actions.layout.display, "flex")
        gallery, image, plots = (self.bar.buttons[t] for t in ("gallery", "image", "plots"))
        self.assertEqual((gallery.description, gallery.disabled), ("Gallery ✓", False))
        self.assertIn("Remove", gallery.tooltip)
        self.assertEqual((image.description, image.disabled), ("Image", False))
        self.assertEqual((plots.description, plots.disabled), ("Plots ✓", True))
        self.assertIn("Selected in the plots", plots.tooltip)

    def test_message_overrides_the_summary_until_removed(self):
        self.bar.show_selection("<b>3 cells</b>", (), True)
        self.bar.set_message("No cells inside the loop.", warning=True)
        self.assertIn("No cells inside the loop", self.bar.text.value)
        self.bar.set_message("")
        self.assertIn("3 cells", self.bar.text.value)

    def test_one_line_with_the_full_text_as_title(self):
        self.bar.set_message('Run <code>%matplotlib widget</code> "now"')
        self.assertIn('title="Run %matplotlib widget &quot;now&quot;"', self.bar.text.value)
        self.assertIn("white-space: nowrap", self.bar.text.value)

    def test_buttons_call_their_actions(self):
        calls = []
        for target in ("image", "gallery", "plots", "clear"):
            self.bar.on_action(target, lambda t=target: calls.append(t))
        for target in ("image", "gallery", "plots", "clear"):
            self.bar.trigger(target)
        self.assertEqual(calls, ["image", "gallery", "plots", "clear"])


def _hub_viewer():
    """The issue135 stub viewer with a hub and a selection bar."""
    viewer = _make_viewer()
    viewer.selection_hub = SelectionHub()
    viewer.ui_component.selection_bar = SelectionBar(SimpleNamespace())
    return viewer


class ScatterPublishesTests(unittest.TestCase):
    def setUp(self):
        self.viewer = _hub_viewer()
        self.chart = ChartDisplay(self.viewer, width=4, height=3)
        self.chart._sync_mask_highlights_from_selection = lambda _indices: None
        self.chart._scatter_views["scatter-1"] = _FakeScatterView("scatter-1")
        self.ui = self.chart.ui_component

    def test_lasso_in_the_plot_publishes_with_its_links(self):
        self.ui.mv_linked_checkbox.value = True
        self.ui.cell_gallery_linked_checkbox.value = False
        self.chart._commit_scatter_selection({0, 1}, origin="scatter")
        current = self.viewer.selection_hub.current
        self.assertEqual(current.indices, {0, 1})
        self.assertEqual(current.source, "scatter")
        self.assertEqual(current.delivered, {"image"})

    def test_selection_pushed_from_the_bar_is_not_republished(self):
        self.viewer.selection_hub.publish({4}, "histogram")
        self.chart._commit_scatter_selection({4}, push_highlight=False, origin="hub")
        self.assertEqual(self.viewer.selection_hub.current.source, "histogram")

    def test_clear_selection_button_clears_only_its_own_selection(self):
        self.viewer.selection_hub.publish({4}, "histogram")
        # What the plot's Clear selection button runs.
        self.chart._commit_scatter_selection(set(), origin="scatter")
        self.assertEqual(self.viewer.selection_hub.current.source, "histogram")
        self.chart._commit_scatter_selection({0}, origin="scatter")
        self.chart._commit_scatter_selection(set(), origin="scatter")
        self.assertEqual(len(self.viewer.selection_hub.current), 0)

    def test_plot_line_keeps_only_the_lasso_tip(self):
        self.chart._commit_scatter_selection(set(), origin="scatter")
        self.assertIn("Shift", self.chart._selection_status.value)
        self.chart._commit_scatter_selection({0}, origin="scatter")
        self.assertEqual(self.chart._selection_status.value, "")

    def test_frontend_problem_shown_in_the_plot_line(self):
        self.viewer.widget_frontend_status = SimpleNamespace(is_problem=True, message="assets <missing>")
        self.chart._update_selection_status(set(), "external")
        self.assertIn("empty boxes", self.chart._selection_status.value)
        self.assertIn("assets &lt;missing&gt;", self.chart._selection_status.value)

    def test_trace_failure_goes_to_the_selection_bar(self):
        self.chart.trace_cells(None)
        self.assertIn("Trace found no cells", self.viewer.ui_component.selection_bar.text.value)


class HistogramPublishesTests(unittest.TestCase):
    def setUp(self):
        self.viewer = _hub_viewer()
        self.hist = HistogramDisplay(self.viewer, width=4, height=3)
        self.hist._plot_data = self.viewer.cell_table.copy()
        self.hist._channels = ["intensity", "area"]
        self.hist._refresh_overlays = lambda *a, **k: None
        self.hist._refresh_gate_markers = lambda *a, **k: None

    def test_brushed_gate_publishes(self):
        self.hist.ui_component.mv_linked_checkbox.value = False
        self.hist.ui_component.cell_gallery_linked_checkbox.value = True
        self.hist.handle_range("intensity", 4.0, 8.0)
        current = self.viewer.selection_hub.current
        self.assertEqual(current.source, "histogram")
        self.assertEqual(current.indices, {1, 4})
        self.assertEqual(current.delivered, {"gallery"})

    def test_clear_selection_clears_the_hub(self):
        self.hist.handle_range("intensity", 4.0, 8.0)
        self.hist.clear_selection()
        self.assertEqual(len(self.viewer.selection_hub.current), 0)

    def test_reapplying_after_a_fov_switch_does_not_publish(self):
        self.viewer.selection_hub.publish({0}, "scatter")
        self.hist._gates = {"intensity": ("range", 4.0, 8.0)}
        self.hist.highlight_cells()  # the FOV-switch re-apply: push_to_gallery=False
        self.assertEqual(self.viewer.selection_hub.current.source, "scatter")


class ViewerSummaryAndActionsTests(unittest.TestCase):
    def setUp(self):
        viewer = ImageMaskViewer.__new__(ImageMaskViewer)
        viewer.cell_table = pd.DataFrame({"fov": ["fov1", "fov1", "fov2"], "label": [1, 2, 3]})
        viewer.fov_key = "fov"
        viewer._map_mode_active = False
        viewer.ui_component = SimpleNamespace(
            image_selector=SimpleNamespace(value="fov1"),
            selection_bar=SelectionBar(SimpleNamespace()),
        )
        self.gallery = mock.Mock()
        self.chart = mock.Mock()
        self.histogram = mock.Mock()
        viewer.SidePlots = SimpleNamespace(
            cell_gallery_output=self.gallery, chart_output=self.chart, histogram_output=self.histogram
        )
        viewer.image_display = SimpleNamespace(selected_masks_label=set(), _lasso_active=False, clear_patches=mock.Mock())
        self.viewer = viewer
        self.bar = viewer.ui_component.selection_bar

    def test_summary_names_count_source_and_fov_share(self):
        self.viewer.selection_hub.publish({0, 2}, "histogram", {"gallery"})
        text = self.bar.text.value
        self.assertIn("2 cells", text)
        self.assertIn("histogram gate", text)
        self.assertIn("1 in this FOV, not outlined in the image", text)
        self.assertEqual(self.bar.buttons["gallery"].description, "Gallery ✓")

    def test_outlined_selection_does_not_say_otherwise(self):
        self.viewer.selection_hub.publish({0}, "scatter", {"image"})
        self.assertNotIn("not outlined", self.bar.text.value)

    def test_map_mode_has_no_per_fov_count(self):
        self.viewer._map_mode_active = True
        self.viewer.selection_hub.publish({0, 2}, "scatter")
        self.assertNotIn("in this FOV", self.bar.text.value)

    def test_gallery_button_sends_and_marks(self):
        self.viewer.selection_hub.publish({2, 0}, "histogram")
        self.viewer.send_selection_to_gallery()
        self.gallery.set_selected_cells.assert_called_once_with([0, 2])
        self.assertIn("gallery", self.viewer.selection_hub.current.delivered)

    def test_image_button_outlines_and_marks(self):
        self.viewer.selection_hub.publish({0}, "heatmap", {"gallery"})
        with mock.patch.object(_chart_common, "sync_mask_highlights_from_selection") as sync:
            self.viewer.send_selection_to_image()
        sync.assert_called_once_with(self.viewer, {0}, mask_name=None)
        self.assertIn("image", self.viewer.selection_hub.current.delivered)

    def test_plots_button_selects_without_pushing_back(self):
        self.viewer.selection_hub.publish({0}, "image lasso", {"image"})
        self.viewer.send_selection_to_plots()
        self.chart._commit_scatter_selection.assert_called_once_with({0}, push_highlight=False, origin="hub")
        self.histogram.show_external_selection.assert_called_once_with({0}, push_highlight=False)
        self.assertIn("plots", self.viewer.selection_hub.current.delivered)

    def test_clear_everywhere(self):
        self.viewer.image_display.selected_masks_label = {_selection("fov1", 1)}
        self.viewer.selection_hub.publish({0}, "image click", {"image"})
        self.viewer.clear_selection_everywhere()
        self.viewer.image_display.clear_patches.assert_called_once()
        self.histogram.clear_selection.assert_called_once()
        self.assertEqual(len(self.viewer.selection_hub.current), 0)
        self.assertEqual(self.bar.actions.layout.display, "none")

    def test_second_click_unlinks_the_gallery(self):
        self.viewer.selection_hub.publish({0}, "histogram")
        self.viewer.toggle_selection_target("gallery")
        self.assertEqual(self.bar.buttons["gallery"].description, "Gallery ✓")
        self.viewer.toggle_selection_target("gallery")
        self.assertEqual(self.gallery.set_selected_cells.call_args_list[-1], mock.call([]))
        self.assertNotIn("gallery", self.viewer.selection_hub.current.delivered)
        self.assertEqual(self.bar.buttons["gallery"].description, "Gallery")
        self.assertEqual(len(self.viewer.selection_hub.current), 1)  # the selection itself stays

    def test_second_click_unlinks_the_image(self):
        self.viewer.image_display.selected_masks_label = {_selection("fov1", 1)}
        self.viewer.selection_hub.publish({0}, "histogram", {"image"})
        self.viewer.toggle_selection_target("image")
        self.viewer.image_display.clear_patches.assert_called_once()
        self.assertIn("not outlined in the image", self.bar.text.value)

    def test_second_click_unlinks_the_plots(self):
        self.viewer.selection_hub.publish({0}, "image lasso", {"image", "plots"})
        self.viewer.toggle_selection_target("plots")
        self.chart._commit_scatter_selection.assert_called_once_with(set(), push_highlight=False, origin="hub")
        self.histogram.show_external_selection.assert_called_once_with(set(), push_highlight=False)
        self.assertNotIn("plots", self.viewer.selection_hub.current.delivered)

    def test_the_source_view_cannot_be_unlinked(self):
        self.viewer.selection_hub.publish({0}, "scatter")
        button = self.bar.buttons["plots"]
        self.assertEqual(button.description, "Plots ✓")
        self.assertTrue(button.disabled)
        self.viewer.toggle_selection_target("plots")
        self.chart._commit_scatter_selection.assert_not_called()
        self.assertFalse(self.bar.buttons["image"].disabled)

    def test_image_source_shows_image_as_selected_there(self):
        self.viewer.selection_hub.publish({0}, "image click")
        self.assertTrue(self.bar.buttons["image"].disabled)
        self.assertNotIn("not outlined", self.bar.text.value)

    def test_clear_asks_first(self):
        dialog = mock.Mock()
        self.viewer.ui_component.confirm_dialog = dialog
        self.viewer.selection_hub.publish({0, 2}, "histogram", {"gallery"})
        self.viewer.ask_clear_selection()
        dialog.ask.assert_called_once()
        message, on_confirm = dialog.ask.call_args.args
        self.assertIn("2 cells", message)
        self.assertIn("brings it back", message)  # ◀ restores it
        self.assertEqual(dialog.ask.call_args.kwargs["confirm_label"], "Unselect")
        self.assertEqual(len(self.viewer.selection_hub.current), 2)  # nothing cleared yet
        on_confirm()
        self.assertEqual(len(self.viewer.selection_hub.current), 0)
        self.assertEqual(self.gallery.set_selected_cells.call_args_list[-1], mock.call([]))

    def test_clear_without_a_dialog_clears_directly(self):
        self.viewer.selection_hub.publish({0}, "histogram")
        self.viewer.ask_clear_selection()
        self.assertEqual(len(self.viewer.selection_hub.current), 0)

    def test_clear_with_nothing_selected_does_not_ask(self):
        dialog = mock.Mock()
        self.viewer.ui_component.confirm_dialog = dialog
        self.viewer.ask_clear_selection()
        dialog.ask.assert_not_called()

    def test_image_selection_publishes_its_cell_indices(self):
        self.viewer.label_key = "label"
        self.viewer.image_display.selected_masks_label = {_selection("fov1", 2)}
        self.viewer.publish_image_selection("image lasso")
        current = self.viewer.selection_hub.current
        self.assertEqual(current.indices, {1})
        self.assertEqual(current.source, "image lasso")
        self.assertIn("image", current.delivered)


if __name__ == "__main__":
    unittest.main()
