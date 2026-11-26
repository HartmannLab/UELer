"""Mouse gestures in the image: click, Ctrl-click, Ctrl-drag lasso, right click.

A left click on the background does nothing, so a mis-click keeps the
selection; a right click on the background unselects everything. Ctrl + click
toggles one cell, Ctrl + drag draws a lasso without the Lasso Select toggle,
and a loop around cells that are all selected already unselects them.
Shift + click on a cell goes to the cart handler and leaves the selection alone.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

from matplotlib.backend_bases import MouseButton

from ueler.viewer.image_display import ImageDisplay, MaskSelection

LEFT, RIGHT = MouseButton.LEFT, MouseButton.RIGHT


class _Line:
    def __init__(self, xs, ys):
        self.data = (list(xs), list(ys))
        self.removed = False

    def set_data(self, xs, ys):
        self.data = (list(xs), list(ys))

    def remove(self):
        self.removed = True


class _Axes:
    def __init__(self):
        self.lines = []

    def plot(self, xs, ys, **_kwargs):
        line = _Line(xs, ys)
        self.lines.append(line)
        return (line,)


def _cell(label, fov="FOV1"):
    return MaskSelection(fov=fov, mask="cells", mask_id=label)


class ImageGestureTests(unittest.TestCase):
    def setUp(self):
        self.published = []
        self.cleared = []
        self.reports = []
        # Cell 1 covers x < 10, cell 2 covers 10 <= x < 20; the rest is background.
        self.viewer = SimpleNamespace(
            resolve_mask_hit_at_viewport=self._hit,
            inform_plugins=lambda _name: None,
            publish_image_selection=lambda source, recipe: self.published.append(
                (source, recipe, set(display.selected_masks_label))
            ),
            clear_selection_everywhere=lambda recipe=None: self.cleared.append(recipe),
            report_lasso_result=lambda added, total: self.reports.append((added, total)),
            set_lasso_hint=lambda *a, **k: None,
            linked_selection_indices=None,
            _map_mode_active=False,
        )
        display = ImageDisplay.__new__(ImageDisplay)
        display.main_viewer = self.viewer
        display.ax = _Axes()
        display.fig = SimpleNamespace(canvas=SimpleNamespace(toolbar=None, draw_idle=lambda: None))
        display._lasso_active = False
        display._lasso_selector = None
        display._lasso_on_complete = None
        display.selected_masks_label = set()
        display.update_patches = lambda do_not_reset=False: None
        self.inside = set()
        display._find_masks_in_lasso_single_fov = lambda verts: set(self.inside)
        self.display = display

    @staticmethod
    def _hit(x, y):
        if x < 10:
            return SimpleNamespace(fov_name="FOV1", mask_name="cells", mask_id=1)
        if x < 20:
            return SimpleNamespace(fov_name="FOV1", mask_name="cells", mask_id=2)
        return None

    def _event(self, x, y, button=LEFT, key=None, px=None, **extra):
        px = x * 10 if px is None else px
        return SimpleNamespace(
            inaxes=self.display.ax, xdata=x, ydata=y, button=button, key=key, x=px, y=y * 10, **extra
        )

    def _click(self, x, y=5, button=LEFT, key=None):
        event = self._event(x, y, button, key)
        self.display.on_mouse_click(event)
        self.display.on_mouse_release(event)

    def _ctrl_drag(self, points):
        x0, y0 = points[0]
        self.display.on_mouse_click(self._event(x0, y0, key="control"))
        for x, y in points[1:]:
            self.display.on_mouse_move(self._event(x, y, key="control"))
        self.display.on_mouse_release(self._event(*points[-1], key="control"))

    # -- Shift-click: the active cart ----------------------------------------
    def test_shift_click_on_a_cell_goes_to_the_cart_handler_only(self):
        hits = []
        self.display.shift_click_handler = hits.append
        self.display.selected_masks_label = {_cell(1)}
        self._click(15, key="shift")
        self.assertEqual([h.mask_id for h in hits], [2])
        self.assertEqual(self.display.selected_masks_label, {_cell(1)})
        self.assertEqual(self.published, [])

    def test_shift_click_on_background_or_without_a_handler_does_nothing(self):
        hits = []
        self._click(15, key="shift")  # no handler: no selection either
        self.assertEqual(self.display.selected_masks_label, set())
        self.display.shift_click_handler = hits.append
        self._click(50, key="shift")
        self.assertEqual(hits, [])
        self.assertEqual(self.cleared, [])

    def test_ctrl_shift_click_is_a_ctrl_click(self):
        hits = []
        self.display.shift_click_handler = hits.append
        self._click(5, key="ctrl+shift")
        self.assertEqual(hits, [])
        self.assertEqual(self.display.selected_masks_label, {_cell(1)})

    # -- clicks --------------------------------------------------------------
    def test_left_click_on_background_keeps_the_selection(self):
        self.display.selected_masks_label = {_cell(1)}
        self._click(50)
        self.assertEqual(self.display.selected_masks_label, {_cell(1)})
        self.assertEqual(self.published, [])
        self.assertEqual(self.cleared, [])

    def test_right_click_on_background_unselects_everything(self):
        self.display.selected_masks_label = {_cell(1)}
        self._click(50, button=RIGHT)
        self.assertEqual(self.cleared, [{"action": "right click on background"}])

    def test_right_click_on_a_cell_does_nothing(self):
        self.display.selected_masks_label = {_cell(1)}
        self._click(5, button=RIGHT)
        self.assertEqual(self.cleared, [])
        self.assertEqual(self.display.selected_masks_label, {_cell(1)})

    def test_left_click_on_a_cell_selects_only_it(self):
        self.display.clear_patches = lambda: self.display.selected_masks_label.clear()
        self.display.selected_masks_label = {_cell(1)}
        self._click(15)
        self.assertEqual(self.display.selected_masks_label, {_cell(2)})
        source, recipe, _cells = self.published[-1]
        self.assertEqual((source, recipe["action"], recipe["label"]), ("image click", "select", 2))

    def test_ctrl_click_adds_then_removes_a_cell(self):
        self.display.selected_masks_label = {_cell(1)}
        self._click(15, key="control")
        self.assertEqual(self.display.selected_masks_label, {_cell(1), _cell(2)})
        self.assertEqual(self.published[-1][1]["action"], "add")
        self._click(15, key="control")
        self.assertEqual(self.display.selected_masks_label, {_cell(1)})
        self.assertEqual(self.published[-1][1]["action"], "remove")

    def test_ctrl_from_the_mouse_event_modifiers(self):
        event = self._event(15, 5, modifiers=frozenset({"ctrl"}))
        self.display.on_mouse_click(event)
        self.display.on_mouse_release(event)
        self.assertEqual(self.display.selected_masks_label, {_cell(2)})
        self.assertEqual(self.published[-1][1]["action"], "add")

    def test_ctrl_click_on_background_does_nothing(self):
        self.display.selected_masks_label = {_cell(1)}
        self._click(50, key="control")
        self.assertEqual(self.display.selected_masks_label, {_cell(1)})
        self.assertEqual(self.published, [])

    # -- Ctrl + drag: a lasso ------------------------------------------------
    def test_ctrl_drag_lassoes_without_the_toggle(self):
        self.inside = {_cell(1), _cell(2)}
        self._ctrl_drag([(1, 1), (30, 1), (30, 30), (1, 30)])
        self.assertEqual(self.display.selected_masks_label, {_cell(1), _cell(2)})
        source, recipe, _cells = self.published[-1]
        self.assertEqual((source, recipe["action"], recipe["added"]), ("image lasso", "add", 2))
        self.assertEqual(self.reports[-1], (2, 2))
        self.assertFalse(self.display._lasso_active)
        # The loop was drawn while dragging and is gone afterwards.
        self.assertEqual(len(self.display.ax.lines), 1)
        self.assertTrue(self.display.ax.lines[0].removed)
        self.assertIsNone(self.display._ctrl_gesture)

    def test_a_tiny_ctrl_drag_is_a_click(self):
        event = self._event(15, 5, key="control", px=150)
        self.display.on_mouse_click(event)
        self.display.on_mouse_move(self._event(15.1, 5, key="control", px=152))
        self.display.on_mouse_release(self._event(15.2, 5, key="control", px=153))
        self.assertEqual(self.display.selected_masks_label, {_cell(2)})
        self.assertEqual(self.published[-1][0], "image click")
        self.assertEqual(self.display.ax.lines, [])

    def test_circling_selected_cells_again_unselects_them(self):
        self.display.selected_masks_label = {_cell(1), _cell(2), _cell(3)}
        self.inside = {_cell(1), _cell(2)}
        self._ctrl_drag([(1, 1), (30, 1), (30, 30), (1, 30)])
        self.assertEqual(self.display.selected_masks_label, {_cell(3)})
        recipe = self.published[-1][1]
        self.assertEqual((recipe["action"], recipe["removed"]), ("remove", 2))

    def test_a_loop_reaching_an_unselected_cell_adds(self):
        self.display.selected_masks_label = {_cell(1)}
        self.inside = {_cell(1), _cell(2)}
        self._ctrl_drag([(1, 1), (30, 1), (30, 30), (1, 30)])
        self.assertEqual(self.display.selected_masks_label, {_cell(1), _cell(2)})
        self.assertEqual(self.published[-1][1]["action"], "add")

    def test_the_toggle_lasso_also_unselects_circled_selected_cells(self):
        self.display.selected_masks_label = {_cell(1)}
        self.inside = {_cell(1)}
        self.display._lasso_active = True
        self.display._on_lasso_selected([(0, 0), (9, 0), (9, 9)])
        self.assertEqual(self.display.selected_masks_label, set())
        self.assertEqual(self.published[-1][1]["action"], "remove")


if __name__ == "__main__":
    unittest.main()
