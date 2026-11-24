"""The hint beside Lasso Select: armed, result, and why a loop selects nothing."""

import unittest
from types import SimpleNamespace

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ueler.viewer.image_display import ImageDisplay, MaskSelection, canvas_is_interactive
from ueler.viewer.main_viewer import ImageMaskViewer


class _InteractiveCanvas(FigureCanvasAgg):
    """Stands in for ipympl's canvas, which subclasses an Agg canvas."""


class _Toggle:
    def __init__(self, viewer):
        self._viewer = viewer
        self._value = False

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new):
        old, self._value = self._value, new
        if old != new:  # what ipywidgets' observe does
            self._viewer.on_lasso_select_toggle({"new": new, "old": old})


class _Display:
    def __init__(self, canvas):
        self.fig = SimpleNamespace(canvas=canvas)
        self._lasso_active = False
        self.enabled = 0

    def enable_lasso_selector(self, on_complete=None):
        self._lasso_active = True
        self.enabled += 1

    def disable_lasso_selector(self):
        self._lasso_active = False


def _viewer(*, interactive=True, masks=("cells",), shown=("cells",)):
    viewer = ImageMaskViewer.__new__(ImageMaskViewer)
    canvas = _InteractiveCanvas(Figure()) if interactive else FigureCanvasAgg(Figure())
    viewer.image_display = _Display(canvas)
    viewer.masks_available = bool(masks)
    viewer.mask_names = list(masks)
    viewer.ui_component = SimpleNamespace(
        lasso_hint=SimpleNamespace(value=""),
        mask_display_controls={name: SimpleNamespace(value=name in shown) for name in masks},
    )
    viewer.ui_component.lasso_select_toggle = _Toggle(viewer)
    return viewer


class CanvasInteractiveTests(unittest.TestCase):
    def test_plain_agg_canvas_is_static(self):
        self.assertFalse(canvas_is_interactive(FigureCanvasAgg(Figure())))

    def test_subclassed_canvas_is_interactive(self):
        self.assertTrue(canvas_is_interactive(_InteractiveCanvas(Figure())))

    def test_ipympl_canvas_is_interactive(self):
        try:
            from ipympl.backend_nbagg import Canvas
        except Exception:  # pragma: no cover - ipympl is a dependency
            self.skipTest("ipympl not importable")
        self.assertTrue(canvas_is_interactive(Canvas(Figure())))


class LassoToggleHintTests(unittest.TestCase):
    def test_static_canvas_releases_the_toggle_and_says_why(self):
        viewer = _viewer(interactive=False)
        viewer.ui_component.lasso_select_toggle.value = True
        self.assertFalse(viewer.ui_component.lasso_select_toggle.value)
        self.assertEqual(viewer.image_display.enabled, 0)
        self.assertIn("%matplotlib widget", viewer.ui_component.lasso_hint.value)

    def test_no_masks_loaded_releases_the_toggle(self):
        viewer = _viewer(masks=(), shown=())
        viewer.ui_component.lasso_select_toggle.value = True
        self.assertFalse(viewer.ui_component.lasso_select_toggle.value)
        self.assertEqual(viewer.image_display.enabled, 0)
        self.assertIn("No segmentation masks", viewer.ui_component.lasso_hint.value)

    def test_no_mask_shown_arms_the_lasso_with_a_warning(self):
        viewer = _viewer(shown=())
        viewer.ui_component.lasso_select_toggle.value = True
        self.assertTrue(viewer.ui_component.lasso_select_toggle.value)
        self.assertEqual(viewer.image_display.enabled, 1)
        self.assertIn("No mask is shown", viewer.ui_component.lasso_hint.value)

    def test_armed_hint_and_cancel_clears_it(self):
        viewer = _viewer()
        viewer.ui_component.lasso_select_toggle.value = True
        self.assertIn("Draw a loop", viewer.ui_component.lasso_hint.value)
        viewer.ui_component.lasso_select_toggle.value = False
        self.assertEqual(viewer.ui_component.lasso_hint.value, "")

    def test_result_survives_the_toggle_release(self):
        viewer = _viewer()
        viewer.ui_component.lasso_select_toggle.value = True
        # The order ImageDisplay uses: release the toggle, then report.
        viewer._on_lasso_complete()
        viewer.report_lasso_result(3, 3)
        self.assertIn("3 cells selected", viewer.ui_component.lasso_hint.value)
        self.assertNotIn("in total", viewer.ui_component.lasso_hint.value)

    def test_result_counts(self):
        viewer = _viewer()
        viewer.report_lasso_result(1, 4)
        self.assertIn("1 cell selected (4 in total)", viewer.ui_component.lasso_hint.value)

    def test_empty_result_with_a_mask_shown(self):
        viewer = _viewer()
        viewer.report_lasso_result(0, 0)
        self.assertIn("No cells inside the loop", viewer.ui_component.lasso_hint.value)

    def test_empty_result_without_a_mask_shown(self):
        viewer = _viewer(shown=())
        viewer.report_lasso_result(0, 0)
        self.assertIn("no mask is shown", viewer.ui_component.lasso_hint.value)


class ImageDisplayReportsTests(unittest.TestCase):
    def _display(self, mask):
        display = ImageDisplay.__new__(ImageDisplay)
        display._lasso_selector = None
        display._lasso_active = True
        display._lasso_on_complete = None
        display.selected_masks_label = {MaskSelection(fov="FOV1", mask="cells", mask_id=9)}
        display.update_patches = lambda *a, **k: None
        display._forget_linked_selection = lambda: None
        display.fig = SimpleNamespace(canvas=SimpleNamespace(new_timer=None))
        reports, hints = [], []
        display.main_viewer = SimpleNamespace(
            _map_mode_active=False,
            _active_map_id=None,
            inform_plugins=lambda _name: None,
            report_lasso_result=lambda added, total: reports.append((added, total)),
            set_lasso_hint=lambda text, **_kw: hints.append(text),
        )
        display._find_masks_in_lasso_single_fov = lambda verts: {
            MaskSelection(fov="FOV1", mask="cells", mask_id=int(i)) for i in np.unique(mask) if i
        }
        return display, reports, hints

    def test_finished_lasso_reports_added_and_total(self):
        display, reports, _ = self._display(np.array([[0, 1], [2, 0]]))
        display._on_lasso_selected([(0, 0), (1, 0), (1, 1)])
        self.assertEqual(reports, [(2, 3)])

    def test_degenerate_lasso_reports_nothing_added(self):
        display, reports, _ = self._display(np.zeros((2, 2)))
        display._on_lasso_selected([(0, 0)])
        self.assertEqual(reports, [(0, 1)])

    def test_clearing_the_selection_clears_the_hint(self):
        display, _, hints = self._display(np.zeros((2, 2)))
        display._lasso_active = False
        display.clear_patches()
        self.assertEqual(hints, [""])

    def test_hint_kept_while_the_lasso_is_armed(self):
        display, _, hints = self._display(np.zeros((2, 2)))
        display.clear_patches()
        self.assertEqual(hints, [])


if __name__ == "__main__":
    unittest.main()
