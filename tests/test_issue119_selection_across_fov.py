"""Issue #119 — a plot selection must stay highlighted after switching FOV.

The mask highlight is materialised per FOV: ``ImageDisplay.set_mask_ids`` resolves
the selected cell-table rows into ``(fov, mask, mask_id)`` triples for the *active*
FOV and ``update_patches`` draws only the triples matching it.  Switching FOV
therefore left nothing to draw even though the plugin still held the (FOV-
independent) row indices.  The fix records those indices on the viewer as
``linked_selection_indices`` and re-projects them from ``on_image_change``.

The tests below drive the real ``ImageDisplay.set_mask_ids`` /
``_chart_common.sync_mask_highlights_from_selection`` /
``ImageMaskViewer._reapply_selection_highlights`` against stub viewers; only
``update_patches`` is replaced (by a call counter), because what #119 is about is
*which* selections end up in the per-FOV cache, not how Matplotlib paints them.
"""

import unittest
from types import SimpleNamespace

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

import numpy as np
import pandas as pd

# Imported at module scope on purpose: a lazy import inside the tests would run
# after tests/test_lasso_selection.py has installed its module-level
# ``ueler.image_utils`` stub, and ``main_viewer`` cannot be imported against that.
from ueler.viewer.image_display import ImageDisplay, MaskSelection
from ueler.viewer.main_viewer import ImageMaskViewer
from ueler.viewer.plugin import _chart_common


MASK_NAME = "whole_cell"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _cell_table() -> pd.DataFrame:
    """Six cells, three per FOV, labels 1-3 reused in both FOVs.

    Reusing the labels matters: it means a highlight that was *not* re-projected
    cannot accidentally look correct on the next FOV.
    """
    return pd.DataFrame(
        {
            "fov": ["FOV1", "FOV1", "FOV1", "FOV2", "FOV2", "FOV2"],
            "label": [1, 2, 3, 1, 2, 3],
        }
    )


def _label_mask(labels) -> np.ndarray:
    """A tiny label image containing exactly *labels* (one row each)."""
    mask = np.zeros((len(labels) + 1, 4), dtype=np.int32)
    for row, label in enumerate(labels):
        mask[row, :] = int(label)
    return mask


def _make_display(viewer):
    """A real ``ImageDisplay`` with only ``update_patches`` replaced."""
    display = ImageDisplay.__new__(ImageDisplay)
    display.selected_masks_label = set()
    display.main_viewer = viewer
    display.update_patch_calls = []

    def _update_patches(do_not_reset=False):
        display.update_patch_calls.append(do_not_reset)

    display.update_patches = _update_patches
    viewer.image_display = display
    return display


def _make_viewer(fov="FOV1", *, map_mode=False, masks=None):
    viewer = SimpleNamespace()
    viewer.linked_selection_indices = None
    viewer.cell_table = _cell_table()
    viewer.fov_key = "fov"
    viewer.label_key = "label"
    viewer.mask_key = MASK_NAME
    viewer._map_mode_active = map_mode
    viewer._grid_display = None
    viewer.SidePlots = None
    viewer._debug = False
    viewer.ui_component = SimpleNamespace(image_selector=SimpleNamespace(value=fov))
    viewer.inform_plugins = lambda *_a, **_k: None
    viewer._masks = masks if masks is not None else {
        "FOV1": _label_mask([1, 2, 3]),
        "FOV2": _label_mask([1, 2, 3]),
    }
    viewer.get_active_fov = lambda: (
        None if viewer._map_mode_active else (viewer.ui_component.image_selector.value or None)
    )
    _switch_fov(viewer, fov)
    _make_display(viewer)
    return viewer


def _switch_fov(viewer, fov):
    """Mimic what ``update_display`` does on a FOV change.

    ``full_resolution_label_masks`` is rebuilt for the newly active FOV only, which
    is the reason ``set_mask_ids`` can validate ids against the current FOV at all.
    """
    viewer.ui_component.image_selector.value = fov
    viewer.full_resolution_label_masks = {MASK_NAME: viewer._masks[fov]}


def _sync(viewer, indices):
    _chart_common.sync_mask_highlights_from_selection(viewer, set(indices))


def _reapply(viewer):
    ImageMaskViewer._reapply_selection_highlights(viewer)


def _cached(viewer, fov=None):
    """The ``(fov, mask_id)`` pairs currently cached, optionally filtered by FOV."""
    selections = viewer.image_display.selected_masks_label
    if fov is not None:
        selections = [sel for sel in selections if sel.fov == fov]
    return {(sel.fov, sel.mask_id) for sel in selections}


# ---------------------------------------------------------------------------
# The record itself
# ---------------------------------------------------------------------------
class LinkedSelectionRecordTestCase(unittest.TestCase):
    """``sync_mask_highlights_from_selection`` must remember the row indices."""

    def test_selection_is_recorded_fov_independently(self):
        viewer = _make_viewer("FOV1")
        # Rows 0 and 2 are in FOV1, row 4 is in FOV2.
        _sync(viewer, {0, 2, 4})

        # Only FOV1's cells can be outlined right now ...
        self.assertEqual(_cached(viewer), {("FOV1", 1), ("FOV1", 3)})
        # ... but the whole selection is remembered, including the FOV2 row.
        self.assertEqual(viewer.linked_selection_indices, {0, 2, 4})

    def test_record_is_limited_to_rows_that_exist(self):
        viewer = _make_viewer("FOV1")
        _sync(viewer, {0, 999})

        self.assertEqual(viewer.linked_selection_indices, {0})

    def test_clearing_the_plot_selection_drops_the_record(self):
        viewer = _make_viewer("FOV1")
        _sync(viewer, {0, 1})
        _sync(viewer, set())

        # An empty selection short-circuits to ``set_mask_ids([])``, which clears
        # the record — so a later FOV switch falls through to the cutoff/cluster
        # fallback rather than restoring a selection the user just dismissed.
        self.assertIsNone(viewer.linked_selection_indices)
        self.assertEqual(_cached(viewer), set())

    def test_direct_set_mask_ids_clears_the_record(self):
        """A cutoff / cluster ``highlight_cells`` must take the highlight over."""
        viewer = _make_viewer("FOV1")
        _sync(viewer, {0, 2})
        self.assertTrue(viewer.linked_selection_indices)

        viewer.image_display.set_mask_ids(mask_name=MASK_NAME, mask_ids=[2])

        self.assertIsNone(viewer.linked_selection_indices)
        self.assertEqual(_cached(viewer), {("FOV1", 2)})

    def test_clear_patches_clears_the_record(self):
        viewer = _make_viewer("FOV1")
        _sync(viewer, {0, 2})

        viewer.image_display.clear_patches()

        self.assertIsNone(viewer.linked_selection_indices)
        self.assertEqual(_cached(viewer), set())

    def test_lasso_selection_clears_the_record(self):
        """A lasso is a spatial, per-FOV selection — it must not be re-projected."""
        viewer = _make_viewer("FOV1")
        _sync(viewer, {0, 2})

        display = viewer.image_display
        display._lasso_active = True
        display._lasso_on_complete = None
        display._find_masks_in_lasso_single_fov = lambda verts: {
            MaskSelection(fov="FOV1", mask=MASK_NAME, mask_id=2)
        }
        display._on_lasso_selected([(0, 0), (1, 0), (1, 1)])

        self.assertIsNone(viewer.linked_selection_indices)

    def test_click_selection_clears_the_record(self):
        from matplotlib.backend_bases import MouseButton

        viewer = _make_viewer("FOV1")
        _sync(viewer, {0, 2})

        display = viewer.image_display
        display.ax = object()
        display.fig = SimpleNamespace(canvas=SimpleNamespace(toolbar=None))
        display._lasso_active = False
        viewer.resolve_mask_hit_at_viewport = lambda x, y: SimpleNamespace(
            fov_name="FOV1", mask_name=MASK_NAME, mask_id=2
        )

        event = SimpleNamespace(
            inaxes=display.ax, xdata=1.0, ydata=1.0, button=MouseButton.LEFT, key="control"
        )
        display.on_mouse_click(event)

        self.assertIsNone(viewer.linked_selection_indices)


# ---------------------------------------------------------------------------
# The re-projection
# ---------------------------------------------------------------------------
class ReapplyHighlightsTestCase(unittest.TestCase):
    """``_reapply_selection_highlights`` branches on the record."""

    def _fallback_viewer(self, *, heatmap_linked):
        """A viewer whose image_display / plugins are pure recorders."""
        viewer = SimpleNamespace()
        viewer.linked_selection_indices = None
        viewer.calls = []
        viewer.image_display = SimpleNamespace(
            clear_patches=lambda: viewer.calls.append("clear_patches")
        )
        viewer._grid_display = SimpleNamespace(
            clear_patches=lambda: viewer.calls.append("grid_clear_patches")
        )
        histogram = SimpleNamespace(
            highlight_cells=lambda: viewer.calls.append("histogram_highlight")
        )
        heatmap = SimpleNamespace(
            highlight_cells=lambda: viewer.calls.append("heatmap_highlight"),
            ui_component=SimpleNamespace(
                main_viewer_checkbox=SimpleNamespace(value=heatmap_linked)
            ),
        )
        viewer.SidePlots = SimpleNamespace(histogram_output=histogram, heatmap_output=heatmap)
        return viewer

    def test_record_is_reprojected_onto_the_new_fov(self):
        """The issue: the selection survives the switch."""
        viewer = _make_viewer("FOV1")
        _sync(viewer, {0, 2, 4})
        self.assertEqual(_cached(viewer), {("FOV1", 1), ("FOV1", 3)})

        _switch_fov(viewer, "FOV2")
        # This is the pre-fix state: the cache still describes FOV1, so the
        # ``sel.fov == current_fov`` filter in update_patches matches nothing.
        self.assertEqual(_cached(viewer, fov="FOV2"), set())

        _reapply(viewer)

        # Row 4 is FOV2 / label 2 — the only selected cell in the new FOV.
        self.assertEqual(_cached(viewer), {("FOV2", 2)})
        # ... and the record is still intact for the next switch.
        self.assertEqual(viewer.linked_selection_indices, {0, 2, 4})

    def test_switching_back_restores_the_first_fov(self):
        viewer = _make_viewer("FOV1")
        _sync(viewer, {0, 2, 4})

        _switch_fov(viewer, "FOV2")
        _reapply(viewer)
        _switch_fov(viewer, "FOV1")
        _reapply(viewer)

        self.assertEqual(_cached(viewer), {("FOV1", 1), ("FOV1", 3)})

    def test_fov_without_selected_cells_shows_nothing_but_keeps_the_record(self):
        viewer = _make_viewer("FOV1")
        # Rows 0 and 2 are both in FOV1.
        _sync(viewer, {0, 2})

        _switch_fov(viewer, "FOV2")
        _reapply(viewer)

        self.assertEqual(_cached(viewer), set())
        self.assertEqual(viewer.linked_selection_indices, {0, 2})

        _switch_fov(viewer, "FOV1")
        _reapply(viewer)
        self.assertEqual(_cached(viewer), {("FOV1", 1), ("FOV1", 3)})

    def test_no_record_falls_back_to_the_previous_behaviour(self):
        viewer = self._fallback_viewer(heatmap_linked=False)

        _reapply(viewer)

        self.assertEqual(
            viewer.calls, ["clear_patches", "grid_clear_patches", "histogram_highlight"]
        )

    def test_linked_heatmap_keeps_its_patches_and_recomputes(self):
        viewer = self._fallback_viewer(heatmap_linked=True)

        _reapply(viewer)

        self.assertNotIn("clear_patches", viewer.calls)
        self.assertIn("heatmap_highlight", viewer.calls)

    def test_fallback_is_skipped_when_the_record_is_set(self):
        """The re-projection replaces the fallback rather than running alongside it."""
        viewer = self._fallback_viewer(heatmap_linked=False)
        viewer.cell_table = _cell_table()
        viewer.fov_key = "fov"
        viewer.label_key = "label"
        viewer.mask_key = MASK_NAME
        viewer._map_mode_active = False
        viewer.get_active_fov = lambda: "FOV1"
        viewer.full_resolution_label_masks = {MASK_NAME: _label_mask([1, 2, 3])}
        viewer.ui_component = SimpleNamespace(image_selector=SimpleNamespace(value="FOV1"))
        viewer.image_display.set_mask_ids = lambda **kwargs: viewer.calls.append("set_mask_ids")
        viewer.linked_selection_indices = {0}

        _reapply(viewer)

        self.assertEqual(viewer.calls, ["set_mask_ids"])

    def test_no_sideplots_is_tolerated(self):
        viewer = SimpleNamespace(linked_selection_indices=None, SidePlots=None)
        _reapply(viewer)  # must not raise

    def test_missing_attribute_is_tolerated(self):
        """Older pickled/stubbed viewers have no record attribute at all."""
        viewer = SimpleNamespace(SidePlots=None)
        _reapply(viewer)  # must not raise


# ---------------------------------------------------------------------------
# Map mode
# ---------------------------------------------------------------------------
class MapModeTestCase(unittest.TestCase):
    """In map mode every FOV is on screen, so all pairs are cached at once."""

    def test_map_mode_caches_all_fovs_and_still_records(self):
        viewer = _make_viewer("FOV1", map_mode=True)
        _sync(viewer, {0, 2, 4})

        self.assertEqual(
            _cached(viewer), {("FOV1", 1), ("FOV1", 3), ("FOV2", 2)}
        )
        self.assertEqual(viewer.linked_selection_indices, {0, 2, 4})

    def test_leaving_map_mode_reprojects_onto_the_active_fov(self):
        viewer = _make_viewer("FOV1", map_mode=True)
        _sync(viewer, {0, 2, 4})

        viewer._map_mode_active = False
        _switch_fov(viewer, "FOV2")
        _reapply(viewer)

        self.assertEqual(_cached(viewer), {("FOV2", 2)})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
