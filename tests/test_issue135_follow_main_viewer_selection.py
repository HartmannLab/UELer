"""Issue #135 — the cells selected in the main viewer are mapped into the plots.

Selections used to flow one way only: a scatter lasso or a histogram gate could be
pushed *into* the image, but the cells the user picked *in* the image never reached
the plots except through the one-shot **Trace** buttons.  Each plot plugin now
carries a ``Follow main viewer`` checkbox in its *Linked plugins* tab and acts on
the ``on_selection_change`` broadcast ``ImageDisplay`` already emits after a click,
ctrl-click, lasso or clear.

The tests drive the real plugin code against stub viewers.  Two properties matter
throughout and are asserted separately for every plugin:

* the checkbox really gates the behaviour, and
* the receive path never writes back into the viewer (``set_mask_ids``), which
  would replace the user's own selection with its current-FOV projection.

The shared ipywidgets stub ignores ``observe`` registrations, so the checkbox
observers are invoked explicitly; they read the checkbox rather than the change
payload, so this is the same code path a real widget takes.  ``TestHookWiring``
covers the registration itself.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

import pandas as pd

# Imported at module scope on purpose, as in tests/test_issue119_selection_across_fov.py:
# a lazy import inside a test would run after other modules have installed their
# own ``ueler.image_utils`` stub.
from ueler.viewer.image_display import MaskSelection
from ueler.viewer.observable import Observable
from ueler.viewer.plugin import _chart_common
from ueler.viewer.plugin.chart import ChartDisplay
from ueler.viewer.plugin.heatmap import HeatmapDisplay
from ueler.viewer.plugin.histogram import HistogramDisplay
from ueler.viewer.plugin.plugin_base import PluginBase


MASK_NAME = "cells"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _selection(fov, mask_id, mask=MASK_NAME):
    """One entry of ``image_display.selected_masks_label``."""
    return MaskSelection(fov=fov, mask=mask, mask_id=mask_id)


class _FakeImageDisplay:
    """Records every write-back so the tests can assert none happened."""

    def __init__(self):
        self.selected_masks_label: set = set()
        self.set_mask_ids_calls = 0
        self.last_mask_ids: list = []
        self.last_fov_mask_pairs = None

    def set_mask_ids(self, *, mask_name, mask_ids, fov_mask_pairs=None):
        self.set_mask_ids_calls += 1
        viewer = getattr(self, "main_viewer", None)
        if viewer is not None:
            viewer.linked_selection_indices = None
        if fov_mask_pairs is not None:
            self.last_fov_mask_pairs = list(fov_mask_pairs)
            self.last_mask_ids = []
        else:
            self.last_mask_ids = list(mask_ids)
            self.last_fov_mask_pairs = None

    def select(self, *selections):
        """Mimic a click / lasso: replace what the image has selected."""
        self.selected_masks_label = set(selections)


class _FakeCellGallery:
    def __init__(self):
        self.received = None

    def set_selected_cells(self, indices):
        self.received = indices


class _FakeScatterView:
    def __init__(self, identifier):
        self.identifier = identifier
        self.state = SimpleNamespace(title=identifier)
        self.applied: list = []

    def apply_selection(self, indices, *, announce=True):
        normalized = set(indices)
        self.applied.append(normalized)
        return normalized


def _two_fov_table() -> pd.DataFrame:
    """Five cells over two FOVs; labels are unique so a FOV mix-up is visible."""
    return pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov1", "fov2", "fov2"],
            "label": [1, 2, 3, 4, 5],
            "intensity": [1.0, 5.0, 9.0, 3.0, 7.0],
            "area": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cluster": ["A", "A", "B", "C", "B"],
        }
    )


def _make_viewer(cell_table=None) -> SimpleNamespace:
    cell_table = _two_fov_table() if cell_table is None else cell_table
    image_display = _FakeImageDisplay()
    ui_component = SimpleNamespace(image_selector=SimpleNamespace(value="fov1"))
    viewer = SimpleNamespace(
        cell_table=cell_table,
        fov_key="fov",
        label_key="label",
        mask_key=MASK_NAME,
        ui_component=ui_component,
        image_display=image_display,
        SidePlots=SimpleNamespace(cell_gallery_output=_FakeCellGallery()),
        linked_selection_indices=None,
        _debug=False,
        get_active_fov=lambda: ui_component.image_selector.value,
    )
    image_display.main_viewer = viewer
    return viewer


# ---------------------------------------------------------------------------
# The translator
# ---------------------------------------------------------------------------
class ViewerSelectionIndicesTestCase(unittest.TestCase):
    """``(fov, mask_id)`` triples → cell-table row indices."""

    def setUp(self):
        self.viewer = _make_viewer()
        self.image = self.viewer.image_display

    def test_single_fov_selection(self):
        self.image.select(_selection("fov1", 2), _selection("fov1", 3))
        self.assertEqual(
            _chart_common.viewer_selection_indices(self.viewer), {1, 2}
        )

    def test_selection_spanning_several_fovs(self):
        """Map mode selects across FOVs; the same label id lives in both."""
        self.image.select(_selection("fov1", 1), _selection("fov2", 5))
        self.assertEqual(
            _chart_common.viewer_selection_indices(self.viewer), {0, 4}
        )

    def test_label_is_matched_within_its_own_fov(self):
        """Label 4 belongs to fov2 — asking for it in fov1 must find nothing."""
        self.image.select(_selection("fov1", 4))
        self.assertEqual(_chart_common.viewer_selection_indices(self.viewer), set())

    def test_unknown_label_is_dropped(self):
        self.image.select(_selection("fov1", 2), _selection("fov1", 999))
        self.assertEqual(_chart_common.viewer_selection_indices(self.viewer), {1})

    def test_mask_name_is_ignored(self):
        """The triple records the mask that was hit; the cell table keys on label."""
        self.image.select(_selection("fov1", 2, mask="nuclear"))
        self.assertEqual(_chart_common.viewer_selection_indices(self.viewer), {1})

    def test_string_typed_label_column(self):
        table = _two_fov_table()
        table["label"] = table["label"].astype(str)
        viewer = _make_viewer(table)
        viewer.image_display.select(_selection("fov1", 2))
        self.assertEqual(_chart_common.viewer_selection_indices(viewer), {1})

    def test_non_default_index_is_preserved(self):
        table = _two_fov_table()
        table.index = ["c0", "c1", "c2", "c3", "c4"]
        viewer = _make_viewer(table)
        viewer.image_display.select(_selection("fov2", 4))
        self.assertEqual(_chart_common.viewer_selection_indices(viewer), {"c3"})

    def test_empty_selection(self):
        self.assertEqual(_chart_common.viewer_selection_indices(self.viewer), set())

    def test_missing_pieces_yield_an_empty_set(self):
        """It must never raise: ``inform_plugins`` swallows ``AttributeError``."""
        self.assertEqual(
            _chart_common.viewer_selection_indices(SimpleNamespace()), set()
        )

        no_table = _make_viewer()
        no_table.image_display.select(_selection("fov1", 2))
        no_table.cell_table = None
        self.assertEqual(_chart_common.viewer_selection_indices(no_table), set())

        wrong_keys = _make_viewer()
        wrong_keys.image_display.select(_selection("fov1", 2))
        wrong_keys.label_key = "not_a_column"
        self.assertEqual(_chart_common.viewer_selection_indices(wrong_keys), set())


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------
class ScatterFollowTestCase(unittest.TestCase):
    def setUp(self):
        self.viewer = _make_viewer()
        self.image = self.viewer.image_display
        self.chart = ChartDisplay(self.viewer, width=4, height=3)
        self.views = [_FakeScatterView("scatter-1"), _FakeScatterView("scatter-2")]
        for view in self.views:
            self.chart._scatter_views[view.identifier] = view
        self.image.select(_selection("fov1", 2), _selection("fov1", 3))

    def _follow(self, value):
        self.chart.ui_component.follow_mv_checkbox.value = value

    def test_unlinked_plot_ignores_the_image_selection(self):
        self.chart.on_selection_change()
        self.assertEqual(self.views[0].applied, [])
        self.assertEqual(self.chart.selected_indices.value, set())

    def test_linked_plot_selects_the_same_cells_everywhere(self):
        self._follow(True)
        self.chart.on_selection_change()
        for view in self.views:
            self.assertEqual(view.applied, [{1, 2}])
        self.assertEqual(self.chart.selected_indices.value, {1, 2})

    def test_the_viewer_is_never_written_back_to(self):
        """Even with "Main viewer" also ticked — the selection came *from* there."""
        self._follow(True)
        self.chart.ui_component.mv_linked_checkbox.value = True
        self.chart.on_selection_change()
        self.assertEqual(self.image.set_mask_ids_calls, 0)

    def test_a_widget_selection_still_pushes_to_the_viewer(self):
        """The normal (plot-originated) path is unchanged by the new keyword."""
        self.chart.ui_component.mv_linked_checkbox.value = True
        self.chart._commit_scatter_selection({1, 2})
        self.assertEqual(self.image.set_mask_ids_calls, 1)
        self.assertEqual(sorted(self.image.last_mask_ids), [2, 3])

    def test_clearing_the_image_selection_clears_the_plot(self):
        self._follow(True)
        self.chart.on_selection_change()
        self.image.select()  # what clear_patches leaves behind
        self.chart.on_selection_change()
        self.assertEqual(self.views[0].applied[-1], set())
        self.assertEqual(self.chart.selected_indices.value, set())

    def test_ticking_the_box_applies_the_current_selection(self):
        self._follow(True)
        self.chart._on_follow_mv_change(SimpleNamespace(name="value", new=True))
        self.assertEqual(self.chart.selected_indices.value, {1, 2})

    def test_unticking_the_box_leaves_the_plot_alone(self):
        self._follow(True)
        self.chart.on_selection_change()
        self._follow(False)
        self.chart._on_follow_mv_change(SimpleNamespace(name="value", new=False))
        self.assertEqual(self.chart.selected_indices.value, {1, 2})


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
class HistogramFollowTestCase(unittest.TestCase):
    def setUp(self):
        self.viewer = _make_viewer()
        self.image = self.viewer.image_display
        self.hist = HistogramDisplay(self.viewer, width=4, height=3)
        self.hist._plot_data = self.viewer.cell_table.copy()
        self.hist._channels = ["intensity", "area"]
        self.image.select(_selection("fov1", 2), _selection("fov2", 5))

    def _follow(self, value):
        self.hist.ui_component.follow_mv_checkbox.value = value

    def test_unlinked_histogram_ignores_the_image_selection(self):
        self.hist.on_selection_change()
        self.assertEqual(self.hist.selected_indices.value, set())

    def test_linked_histogram_overlays_the_selected_cells(self):
        self._follow(True)
        self.hist.on_selection_change()
        self.assertEqual(self.hist.selected_indices.value, {1, 4})

    def test_the_viewer_is_never_written_back_to(self):
        self._follow(True)
        self.hist.ui_component.mv_linked_checkbox.value = True
        self.hist.on_selection_change()
        self.assertEqual(self.image.set_mask_ids_calls, 0)

    def test_the_local_gate_is_replaced(self):
        """An external selection supersedes the gate terms (#127), as ever."""
        self.hist.set_gate("intensity", ("range", 0.0, 100.0))
        self._follow(True)
        self.hist.on_selection_change()
        self.assertEqual(self.hist._gates, {})
        self.assertEqual(self.hist.selected_indices.value, {1, 4})

    def test_other_plugins_still_push_the_highlight(self):
        """The heatmap → histogram link keeps the default ``push_highlight``."""
        self.hist.ui_component.mv_linked_checkbox.value = True
        self.hist.show_external_selection([1, 2])
        self.assertEqual(self.image.set_mask_ids_calls, 1)

    def test_ticking_the_box_applies_the_current_selection(self):
        self._follow(True)
        self.hist._on_follow_mv_change(SimpleNamespace(name="value", new=True))
        self.assertEqual(self.hist.selected_indices.value, {1, 4})

    def test_unticking_the_box_leaves_the_plot_alone(self):
        self._follow(True)
        self.hist.on_selection_change()
        self._follow(False)
        self.hist._on_follow_mv_change(SimpleNamespace(name="value", new=False))
        self.assertEqual(self.hist.selected_indices.value, {1, 4})


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
class HeatmapFollowTestCase(unittest.TestCase):
    """The clusters the selected cells belong to are highlighted.

    ``HeatmapDisplay`` is built through ``__new__`` — the real ``__init__`` needs a
    full viewer and a rendered clustermap — but the code under test is real:
    ``_map_indices_to_cluster_positions`` and ``_apply_cluster_highlights`` run as
    they do in the app, with only the Matplotlib patch drawing replaced.
    """

    def setUp(self):
        self.viewer = _make_viewer()
        self.image = self.viewer.image_display

        heatmap = HeatmapDisplay.__new__(HeatmapDisplay)
        heatmap.main_viewer = self.viewer
        heatmap.heatmap_data = pd.DataFrame({"marker": [0.0, 1.0, 2.0]},
                                            index=pd.Index(["A", "B", "C"], name="cluster"))
        # Cluster "A" is drawn last, so a position is only right if the display
        # ordering was applied rather than the raw index order.
        heatmap.orientation_state = {
            "cluster_index": pd.Index(["A", "B", "C"]),
            "cluster_order_positions": [1, 2, 0],
        }
        heatmap.ui_component = SimpleNamespace(
            high_level_cluster_dropdown=SimpleNamespace(value="cluster"),
            follow_mv_checkbox=SimpleNamespace(value=False),
        )
        heatmap.data = SimpleNamespace(current_clusters={"index": Observable([])})
        heatmap.highlighted: list = []
        heatmap.highlight_row_colors = lambda positions: heatmap.highlighted.append(
            list(positions)
        )
        self.heatmap = heatmap
        self.image.select(_selection("fov1", 3), _selection("fov2", 5))  # both cluster B

    def _follow(self, value):
        self.heatmap.ui_component.follow_mv_checkbox.value = value

    def test_unlinked_heatmap_ignores_the_image_selection(self):
        self.heatmap.on_selection_change()
        self.assertEqual(self.heatmap.highlighted, [])

    def test_linked_heatmap_highlights_the_selected_cells_cluster(self):
        self._follow(True)
        self.heatmap.on_selection_change()
        # "B" is at index position 1, and the ordering draws that in slot 0 —
        # highlighting slot 1 would mean the display ordering was ignored.
        self.assertEqual(self.heatmap.highlighted, [[0]])
        self.assertEqual(list(self.heatmap.data.current_clusters["index"].value), [0])

    def test_several_clusters_are_all_highlighted(self):
        self.image.select(_selection("fov1", 1), _selection("fov1", 3))  # A and B
        self._follow(True)
        self.heatmap.on_selection_change()
        self.assertEqual(self.heatmap.highlighted, [[0, 2]])

    def test_clearing_the_image_selection_clears_the_highlight(self):
        self._follow(True)
        self.heatmap.on_selection_change()
        self.image.select()
        self.heatmap.on_selection_change()
        self.assertEqual(self.heatmap.highlighted[-1], [])

    def test_the_viewer_is_never_written_back_to(self):
        """Unlike the Trace cluster button, which ends in ``highlight_cells()``."""
        pushed = []
        self.heatmap.highlight_cells = lambda: pushed.append(True)
        self._follow(True)
        self.heatmap.on_selection_change()
        self.assertEqual(pushed, [])
        self.assertEqual(self.image.set_mask_ids_calls, 0)

    def test_an_unplotted_heatmap_is_a_no_op(self):
        self.heatmap.heatmap_data = None
        self._follow(True)
        self.heatmap.on_selection_change()
        self.assertEqual(self.heatmap.highlighted, [])

    def test_ticking_the_box_applies_the_current_selection(self):
        self._follow(True)
        self.heatmap._on_follow_mv_change(SimpleNamespace(name="value", new=True))
        self.assertEqual(self.heatmap.highlighted, [[0]])


# ---------------------------------------------------------------------------
# The hook itself
# ---------------------------------------------------------------------------
class TestHookWiring(unittest.TestCase):
    def test_plugin_base_declares_the_hook(self):
        base = PluginBase.__new__(PluginBase)
        self.assertIsNone(base.on_selection_change())

    def test_every_plot_plugin_overrides_it(self):
        for cls in (ChartDisplay, HistogramDisplay, HeatmapDisplay):
            with self.subTest(plugin=cls.__name__):
                self.assertIsNot(
                    cls.on_selection_change, PluginBase.on_selection_change
                )

    def test_the_checkbox_is_observed(self):
        """The registration the explicit observer calls above stand in for."""
        recorded = []

        class _RecordingCheckbox:
            value = False

            def observe(self, handler, names=None):
                recorded.append((handler, names))

        viewer = _make_viewer()
        chart = ChartDisplay(viewer, width=4, height=3)
        chart.ui_component.follow_mv_checkbox = _RecordingCheckbox()
        chart._wire_events()
        self.assertIn(
            (chart._on_follow_mv_change, "value"),
            [(handler, names) for handler, names in recorded],
        )

    def test_the_broadcast_reaches_the_plugins(self):
        """``inform_plugins`` is what ``ImageDisplay`` calls after every selection."""
        from ueler.viewer.main_viewer import ImageMaskViewer

        called = []

        class _Plugin(PluginBase):
            def __init__(self):
                pass

            def on_selection_change(self):
                called.append(self)

        viewer = SimpleNamespace(SidePlots=SimpleNamespace(a=_Plugin()), _debug=False)
        ImageMaskViewer.inform_plugins(viewer, "on_selection_change")
        self.assertEqual(len(called), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
