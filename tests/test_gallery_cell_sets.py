"""The Gallery as the editing view of a cell set.

The Gallery names what it shows and counts it, stars and removes cells through
the Cell sets plugin, keeps its cells on Refresh, draws new ones on request,
and saves the cells on screen as a sub-group of the set shown. Rendering is
replaced by a stub, so these tests cover the bookkeeping, not the pixels.
"""
from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

import numpy as np
import pandas as pd

from ueler.viewer.plugin import cell_gallery
from ueler.viewer.plugin.cell_gallery import CellGalleryDisplay, _limit_selection
from ueler.viewer.plugin.cell_sets import CellSetsPlugin
from ueler.viewer.plugin.tile_gallery_widget import parse_action
from ueler.viewer.selection import SelectionHub
from ueler.viewer.selection_bar import SelectionBar
from ueler.viewer.selection_style import SelectionStyle


def _fake_tiles(df, indices, **_kwargs):
    indices = list(indices)
    return [np.zeros((4, 4, 3)) for _ in indices], {}, indices


class LimitSelectionTests(unittest.TestCase):
    def test_same_cells_same_sample_and_a_draw_changes_it(self):
        cells = list(range(100))
        first = _limit_selection(cells, 10)
        self.assertEqual(first, _limit_selection(cells, 10))
        self.assertNotEqual(first, _limit_selection(cells, 10, draw=1))

    def test_a_draw_prefers_cells_not_shown_yet(self):
        cells = list(range(25))
        first = _limit_selection(cells, 10)
        second = _limit_selection(cells, 10, draw=1, exclude=first)
        self.assertFalse(set(first) & set(second))
        third = _limit_selection(cells, 10, draw=2, exclude=set(first) | set(second))
        self.assertEqual(len(third), 10)  # 5 unseen, topped up with seen cells
        self.assertTrue(set(range(25)) - set(first) - set(second) <= set(third))
        again = _limit_selection(cells, 10, draw=3, exclude=set(range(25)))
        self.assertEqual(len(again), 10)  # all seen: start again


class WidgetActionTests(unittest.TestCase):
    def test_parse_action(self):
        self.assertEqual(parse_action("star|12|0"), ("star", "12"))
        self.assertEqual(parse_action("remove|7|3"), ("remove", "7"))
        self.assertEqual(parse_action("12|0"), ("", ""))


class GalleryEditingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(cell_gallery, "create_gallery_tiles", side_effect=_fake_tiles)
        patcher.start()
        self.addCleanup(patcher.stop)
        hub = SelectionHub()
        table = pd.DataFrame(
            {
                "fov": ["fov1"] * 30,
                "label": list(range(1, 31)),
                "X": np.arange(30.0),
                "Y": np.arange(30.0),
            }
        )
        self.viewer = SimpleNamespace(
            cell_table=table,
            fov_key="fov",
            label_key="label",
            x_key="X",
            y_key="Y",
            mask_key="cells",
            mask_names=["cells", "nuclei"],
            mask_outline_thickness=1,
            base_folder=self._tmp.name,
            selection_hub=hub,
            image_display=SimpleNamespace(selected_masks_label=set()),
            inform_plugins=lambda _name: None,
            ui_component=SimpleNamespace(
                selection_bar=SelectionBar(SimpleNamespace()),
                image_selector=SimpleNamespace(value="fov1", options=["fov1"]),
            ),
            get_active_fov=lambda: "fov1",
            send_selection_to_image=lambda: hub.mark_delivered("image"),
            confirm=mock.Mock(return_value=True),
        )
        self.sets = CellSetsPlugin(self.viewer, 6, 3)
        self.gallery = CellGalleryDisplay(self.viewer, 6, 3)
        self.gallery.ui_component.max_displayed_cells_inttext.value = 10
        self.viewer.SidePlots = SimpleNamespace(cell_sets_output=self.sets, cell_gallery_output=self.gallery)
        self.sets.after_all_plugins_loaded()
        self.hub = hub

    def tearDown(self):
        self._tmp.cleanup()

    def _show(self, rows, source="histogram", recipe=None):
        """Publish *rows* and send them to the Gallery, as the bar's Gallery button does."""
        self.hub.publish(set(rows), source, recipe=recipe or {})
        self.hub.mark_delivered("gallery")
        self.gallery.set_selected_cells(sorted(rows))

    def _set(self, name, rows):
        self.hub.publish(set(rows), "scatter")
        self.assertEqual(self.sets.save_selection(name), "")
        return next(m for m in self.sets.store.list_sets() if m["name"] == name)["id"]

    def _show_set(self, set_id):
        self.assertTrue(self.sets.select_set(set_id))
        self.hub.mark_delivered("gallery")
        self.gallery.set_selected_cells(sorted(self.hub.current.indices))

    def _click(self, action, row):
        self._clicks = getattr(self, "_clicks", 0) + 1  # the nonce, as the widget's JavaScript sends it
        self.gallery.gallery.action = f"{action}|{row}|{self._clicks}"

    def test_plain_selection_header_and_no_controls_without_a_cart(self):
        self._show(range(25))
        header = self.gallery.set_controls.header.value
        self.assertIn("Selection: histogram gate", header)
        self.assertIn("showing 10 of 25", header)
        self.assertEqual(list(self.gallery.gallery.controls), [])
        self.assertFalse(self.gallery.set_controls.draw_button.disabled)
        self.assertEqual(self.gallery.set_controls.cart_button.layout.display, "none")

    def test_refresh_keeps_the_cells_and_draw_changes_them(self):
        self._show(range(25))
        first = list(self.gallery._shown)
        self.gallery.refresh_gallery()
        self.assertEqual(self.gallery._shown, first)
        self.gallery.draw_new_cells()
        self.assertFalse(set(first) & set(self.gallery._shown))
        second = list(self.gallery._shown)
        self.gallery.refresh_gallery()
        self.assertEqual(self.gallery._shown, second)

    def test_a_set_offers_star_and_remove_and_shows_stars(self):
        set_id = self._set("CD8", range(12))
        self._show_set(set_id)
        self.assertIn("Set: <b>CD8</b>", self.gallery.set_controls.header.value)
        self.assertEqual(list(self.gallery.gallery.controls), ["star", "remove"])
        row = self.gallery._shown[0]
        self._click("star", row)
        self.assertEqual(self.gallery.gallery.badges, {str(row): ["star"]})
        self.assertIn("★ 1", self.gallery.set_controls.header.value)
        self.assertEqual(self.sets.store.favourites(set_id)["count"], 1)
        self._click("star", row)  # a second click unstars
        self.assertEqual(self.gallery.gallery.badges, {})

    def test_remove_drops_the_tile_and_the_cell_from_the_set(self):
        set_id = self._set("CD8", range(5))
        self._show_set(set_id)
        row = self.gallery._shown[2]
        self._click("remove", row)
        self.assertNotIn(row, self.gallery._shown)
        self.assertNotIn(str(row), [t["id"] for t in self.gallery.gallery.tiles])
        self.assertEqual(self.sets.store.get(set_id)["count"], 4)
        self.assertIn("showing 4 of 4", self.gallery.set_controls.header.value)

    def test_a_set_is_shown_as_it_is_now(self):
        set_id = self._set("cart", range(3))
        self._show_set(set_id)
        self.sets.set_active_cart(set_id)
        self.sets.toggle_cell_in_cart(SimpleNamespace(fov_name="fov1", mask_name="cells", mask_id=20))
        self.assertIn("Cart: <b>cart</b>", self.gallery.set_controls.header.value)
        self.assertIn("of 4", self.gallery.set_controls.header.value)

    def test_stars_only(self):
        set_id = self._set("CD8", range(8))
        self._show_set(set_id)
        for row in self.gallery._shown[:2]:
            self._click("star", row)
        self.gallery.set_controls.stars_only.value = True
        self.gallery._on_stars_only(True)  # the test widgets do not call observers
        self.assertEqual(len(self.gallery._shown), 2)
        self.assertIn("showing 2 of 2", self.gallery.set_controls.header.value)
        self._show(range(4))  # a new selection turns it off
        self.assertFalse(self.gallery.set_controls.stars_only.value)

    def test_with_a_cart_a_star_on_a_plain_selection_goes_to_the_cart(self):
        self.sets.create_cart("cart")
        self._show(range(5))
        self.assertIn("★ goes to the cart", self.gallery.set_controls.header.value)
        self.assertEqual(list(self.gallery.gallery.controls), ["star"])
        self._click("star", 3)
        cart = self.sets.store.get(self.sets.active_cart)
        self.assertEqual(cart["count"], 1)
        self.assertEqual(self.sets.store.favourites(cart["id"])["count"], 1)
        self.assertEqual(self.gallery.gallery.badges, {"3": ["star"]})

    def test_add_shown_to_cart(self):
        self.sets.create_cart("cart")
        self._show(range(25))
        self.assertEqual(self.gallery.set_controls.cart_button.layout.display, "inline-flex")
        self.assertEqual(self.gallery.add_shown_to_cart(), 10)
        self.assertEqual(self.sets.store.get(self.sets.active_cart)["count"], 10)

    def test_save_shown_is_a_sub_group_that_records_its_sample(self):
        set_id = self._set("big", range(25))
        self._show_set(set_id)
        self.gallery.draw_new_cells()
        meta = self.gallery.save_shown_as_set()
        self.assertEqual(meta["container"], set_id)
        self.assertEqual(meta["count"], 10)
        self.assertEqual(meta["name"], "Gallery sample")
        self.assertEqual((meta["sample"]["size"], meta["sample"]["of"], meta["sample"]["draw"]), (10, 25, 1))
        self.assertEqual(meta["sample"]["source_set"], set_id)
        rows = [label for _row, label in self.sets.ui_component.set_list.options]
        self.assertIn(meta["id"], rows)  # listed under its set
        second = self.gallery.save_shown_as_set()
        self.assertEqual(second["name"], "Gallery sample 2")

    def test_save_shown_from_a_plain_selection_is_top_level(self):
        self._show(range(5))
        self.gallery.set_controls.save_name.value = "figure 2"
        meta = self.gallery.save_shown_as_set()
        self.assertIsNone(meta["container"])
        self.assertEqual(meta["name"], "figure 2")

    def test_a_snapshot_is_read_only(self):
        set_id = self._set("CD8", range(4))
        self.hub.publish({0}, "scatter")
        self.sets.star_selection(set_id)
        snap = self.sets.snapshot_set(set_id)
        self._show_set(snap["id"])
        self.assertIn("read-only", self.gallery.set_controls.header.value)
        self.assertEqual(list(self.gallery.gallery.controls), [])
        self.assertEqual(self.gallery.gallery.badges, {"0": ["star"]})

    def test_showing_a_sets_stars_unstars_into_its_set(self):
        set_id = self._set("CD8", range(6))
        self.hub.publish({1, 2}, "scatter")
        self.sets.star_selection(set_id)
        self.sets.select_stars(set_id)
        self.hub.mark_delivered("gallery")
        self.gallery.set_selected_cells(sorted(self.hub.current.indices))
        self.assertIn("★ of <b>CD8</b>", self.gallery.set_controls.header.value)
        self._click("star", 1)
        self.assertEqual(self.gallery._shown, [2])
        self.assertEqual(self.sets.store.get(set_id)["count"], 6)  # unstarred, still in the set

    def test_the_outline_follows_the_selection_colour_until_changed_here(self):
        picker = self.gallery.ui_component.mask_outline_picker
        self.assertEqual(picker.value, SelectionStyle().color)
        self._show(range(5))
        with mock.patch.object(self.gallery, "refresh_gallery") as refresh:
            self.viewer.selection_style = SelectionStyle(color="#00ff00")
            self.gallery.on_selection_style_change(previous_color=SelectionStyle().color)
            self.assertEqual(picker.value, "#00ff00")
            refresh.assert_called_once()
            picker.value = "#ff00ff"  # the user picks another colour in the Gallery
            self.viewer.selection_style = SelectionStyle(color="#0000ff")
            self.gallery.on_selection_style_change(previous_color="#00ff00")
            self.assertEqual(picker.value, "#ff00ff")
            refresh.assert_called_once()


if __name__ == "__main__":
    unittest.main()
