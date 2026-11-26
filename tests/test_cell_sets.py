"""Selection history, saved cell sets, carts, stars, tags and writing to the cell table.

Covers the session log in ``SelectionHub`` (◀ ▶, the cell limit that leaves the
log whole), ``CellSetStore`` (editing, the subset rule, snapshots, ★ favourites)
and the cell-identity helpers, the bar's cart buttons, Shift-click in the image,
and the Cell sets plugin that replaced the Cell Table Editor.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

import numpy as np
import pandas as pd

from ueler.viewer import selection as selection_module
from ueler.viewer.cell_set_store import (
    CellSetStore,
    cells_from_indices,
    indices_from_cells,
    parse_tags,
    write_column,
)
from ueler.viewer.image_display import ImageDisplay, MaskSelection
from ueler.viewer.main_viewer import ImageMaskViewer
from ueler.viewer.plugin.cell_sets import CellSetsPlugin
from ueler.viewer.selection import SelectionHub
from ueler.viewer.selection_bar import SelectionBar


def _table():
    return pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov2", "fov2", "fov3"],
            "label": [1, 2, 1, 3, 7],
            "X": [0.0, 1.0, 2.0, 3.0, 4.0],
            "Y": [0.0, 1.0, 2.0, 3.0, 4.0],
            "CD8": [0.5, 3.0, 2.5, 0.1, 4.0],
        }
    )


def _viewer(table=None):
    return SimpleNamespace(
        cell_table=_table() if table is None else table,
        fov_key="fov",
        label_key="label",
        x_key="X",
        y_key="Y",
        mask_key="cells",
        mask_names=["cells", "nuclei"],
    )


# ---------------------------------------------------------------------------
# The session log
# ---------------------------------------------------------------------------
class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.hub = SelectionHub()

    def test_every_selection_is_a_step_with_its_recipe(self):
        self.hub.publish({1, 2}, "histogram", recipe={"gates": {"CD8": ["range", 1, 2]}})
        self.hub.publish({3}, "scatter")
        steps = self.hub.history
        self.assertEqual([s.step for s in steps], [1, 2])
        self.assertEqual(steps[0].recipe, {"gates": {"CD8": ["range", 1, 2]}})
        self.assertEqual(steps[0].count, 2)
        self.assertEqual(self.hub.current.recipe, {})

    def test_back_and_forward_restore_the_cells(self):
        self.hub.publish({1}, "image click")
        self.hub.publish({2, 3}, "histogram")
        self.assertTrue(self.hub.can_go_back())
        self.assertFalse(self.hub.can_go_forward())
        self.hub.go(-1)
        self.assertEqual(self.hub.current.indices, {1})
        self.assertEqual(self.hub.current.source, "history")
        self.assertEqual(self.hub.current.source_label, "step 1 (image click)")
        self.assertTrue(self.hub.can_go_forward())
        self.hub.go(+1)
        self.assertEqual(self.hub.current.indices, {2, 3})

    def test_moves_are_logged_but_skipped_by_the_arrows(self):
        self.hub.publish({1}, "scatter")
        self.hub.publish({2}, "scatter")
        self.hub.go(-1)
        self.assertEqual(self.hub.history[-1].kind, selection_module.NAVIGATION)
        self.assertEqual(self.hub.history[-1].label, "back to step 1")
        self.assertFalse(self.hub.can_go_back())  # the navigation step is not a place

    def test_a_new_selection_after_going_back_keeps_the_log_whole(self):
        self.hub.publish({1}, "scatter")
        self.hub.publish({2}, "scatter")
        self.hub.go(-1)
        self.hub.publish({5}, "histogram")
        self.assertEqual([s.count for s in self.hub.history if s.kind == "selection"], [1, 1, 1])
        self.assertFalse(self.hub.can_go_forward())

    def test_clearing_is_a_step_and_back_undoes_it(self):
        self.hub.publish({1, 2}, "histogram")
        self.hub.clear()
        self.assertEqual(len(self.hub.current), 0)
        self.hub.go(-1)
        self.assertEqual(self.hub.current.indices, {1, 2})

    def test_only_the_last_steps_keep_cells_but_the_log_is_never_cut(self):
        for i in range(selection_module.KEEP_CELLS + 10):
            self.hub.publish({i}, "scatter")
        self.assertEqual(len(self.hub.history), selection_module.KEEP_CELLS + 10)
        with_cells = [s for s in self.hub.history if s.indices is not None]
        self.assertEqual(len(with_cells), selection_module.KEEP_CELLS)
        self.assertEqual(len(self.hub.trail()), selection_module.KEEP_CELLS + 10)
        for _ in range(selection_module.KEEP_CELLS - 1):
            self.assertIsNotNone(self.hub.go(-1))
        self.assertIsNone(self.hub.go(-1))  # the next step back has no cells

    def test_notes_and_trail_have_no_cells(self):
        self.hub.publish({1, 2}, "histogram")
        self.hub.note("wrote to the cell table", {"column": "celltype", "value": "T"}, count=2)
        trail = self.hub.trail()
        self.assertEqual(trail[-1]["kind"], "event")
        self.assertEqual(trail[-1]["label"], "wrote to the cell table")
        self.assertNotIn("indices", trail[0])


# ---------------------------------------------------------------------------
# Tags, cell identity, writing
# ---------------------------------------------------------------------------
class TagTests(unittest.TestCase):
    def test_tags_are_free_form(self):
        tags, problems = parse_tags(" pilot , celltype:CD8 T,  figure   2 , pilot, ")
        self.assertEqual(tags, ["pilot", "celltype:CD8 T", "figure 2"])
        self.assertEqual(problems, [])

    def test_two_values_for_one_key_are_two_tags(self):
        tags, _ = parse_tags("celltype:A, celltype:B")
        self.assertEqual(tags, ["celltype:A", "celltype:B"])


class CellIdentityTests(unittest.TestCase):
    def test_round_trip_survives_a_reordered_table(self):
        viewer = _viewer()
        cells = cells_from_indices(viewer, {1, 2})
        self.assertEqual(sorted(zip(cells["fov"], cells["label"])), [("fov1", 2), ("fov2", 1)])
        viewer.cell_table = viewer.cell_table.iloc[::-1].reset_index(drop=True)
        indices, missing = indices_from_cells(viewer, cells)
        self.assertEqual(missing, 0)
        self.assertEqual(sorted(viewer.cell_table.loc[sorted(indices), "label"]), [1, 2])

    def test_cells_gone_from_the_table_are_counted(self):
        viewer = _viewer()
        cells = cells_from_indices(viewer, {0, 4})
        viewer.cell_table = viewer.cell_table.iloc[:4]
        indices, missing = indices_from_cells(viewer, cells)
        self.assertEqual(indices, {0})
        self.assertEqual(missing, 1)

    def test_image_mask_layer_is_kept(self):
        cells = cells_from_indices(_viewer(), {0, 1}, {("fov1", 1): "nuclei"})
        masks = dict(zip(cells["label"], cells["mask"]))
        self.assertEqual(masks, {1: "nuclei", 2: "cells"})


class WriteColumnTests(unittest.TestCase):
    def test_new_column(self):
        viewer = _viewer()
        self.assertEqual(write_column(viewer, {0, 2, 99}, "celltype", "T"), 2)
        self.assertEqual(viewer.cell_table["celltype"].tolist(), ["T", "", "T", "", ""])

    def test_numeric_column_becomes_text(self):
        viewer = _viewer()
        write_column(viewer, {0}, "CD8", "high")
        self.assertEqual(viewer.cell_table.loc[0, "CD8"], "high")
        self.assertEqual(viewer.cell_table.loc[1, "CD8"], 3.0)

    def test_linking_columns_are_refused(self):
        with self.assertRaises(ValueError):
            write_column(_viewer(), {0}, "label", "x")


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = CellSetStore(self.root)
        self.cells = cells_from_indices(_viewer(), {0, 1, 2})

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_list_read(self):
        meta = self.store.save_set("T cells", self.cells, tags=["celltype:T"], recipe={"x": 1}, source="histogram",
                                   trail=[{"step": 1}])
        self.assertTrue((self.root / ".UELer").is_dir())
        listed = self.store.list_sets()
        self.assertEqual([m["name"] for m in listed], ["T cells"])
        self.assertEqual(listed[0]["count"], 3)
        self.assertEqual(listed[0]["fovs"], 2)
        self.assertEqual(listed[0]["trail"], [{"step": 1}])
        cells = self.store.read_cells(meta["id"])
        self.assertEqual(sorted(cells["label"].tolist()), [1, 1, 2])

    def test_lives_next_to_the_checkpoints(self):
        from ueler.viewer.checkpoint_store import CheckpointStore

        checkpoints = CheckpointStore(self.root)
        self.assertEqual(self.store.folder.parent, checkpoints._dataset_dir)

    def test_names_are_unique(self):
        self.store.save_set("T cells", self.cells)
        with self.assertRaises(ValueError):
            self.store.save_set("t cells", self.cells)

    def test_update_and_delete(self):
        meta = self.store.save_set("A", self.cells)
        self.store.update_set(meta["id"], name="B", tags=["k:v"], note="hi")
        self.assertEqual(self.store.get(meta["id"])["tags"], ["k:v"])
        self.store.delete_set(meta["id"])
        self.assertEqual(self.store.list_sets(), [])
        self.assertEqual(list(self.store.folder.glob("*.parquet")), [])


def _cells(*triples):
    return pd.DataFrame(list(triples), columns=["fov", "mask", "label"])


class EditableSetTests(unittest.TestCase):
    """Sets as carts: adding, removing, the subset rule, stars and snapshots."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = CellSetStore(Path(self._tmp.name))
        self.cart = self.store.save_set("cart", None, allow_empty=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _labels(self, set_id):
        return sorted(self.store.read_cells(set_id)["label"].tolist())

    def test_an_empty_cart_needs_allow_empty(self):
        with self.assertRaises(ValueError):
            self.store.save_set("x", None)
        self.assertEqual(self.store.get(self.cart["id"])["count"], 0)
        self.assertEqual(len(self.store.read_cells(self.cart["id"])), 0)

    def test_add_and_remove_are_logged_and_counted(self):
        added = self.store.add_cells(self.cart["id"], _cells(("f1", "cells", 1), ("f2", "cells", 2)), {"source": "scatter"})
        self.assertEqual(added, 2)
        self.assertEqual(self.store.add_cells(self.cart["id"], _cells(("f1", "cells", 1), ("f1", "cells", 3))), 1)
        self.assertEqual(self.store.remove_cells(self.cart["id"], _cells(("f2", "cells", 2), ("f9", "cells", 9))), 1)
        meta = self.store.get(self.cart["id"])
        self.assertEqual((meta["count"], meta["fovs"]), (2, 1))
        self.assertEqual([(e["op"], e["count"]) for e in meta["edits"]], [("add", 2), ("add", 1), ("remove", 1)])
        self.assertEqual(meta["edits"][0]["recipe"], {"source": "scatter"})

    def test_the_mask_layer_is_part_of_a_cell(self):
        self.store.add_cells(self.cart["id"], _cells(("f1", "cells", 1)))
        self.assertEqual(self.store.add_cells(self.cart["id"], _cells(("f1", "nuclei", 1))), 1)
        self.assertEqual(self.store.contains(self.cart["id"], _cells(("f1", "nuclei", 1), ("f1", "nuclei", 2))).tolist(), [True, False])

    def test_star_adds_to_the_set_and_unstar_keeps_the_cell(self):
        self.store.add_cells(self.cart["id"], _cells(("f1", "cells", 1)))
        self.assertIsNone(self.store.favourites(self.cart["id"]))
        self.assertEqual(self.store.star(self.cart["id"], _cells(("f1", "cells", 1), ("f1", "cells", 5))), 2)
        stars = self.store.favourites(self.cart["id"])
        self.assertEqual(stars["container"], self.cart["id"])
        self.assertEqual(self._labels(stars["id"]), [1, 5])
        self.assertEqual(self._labels(self.cart["id"]), [1, 5])  # a starred cell is always in the set
        self.assertEqual(self.store.unstar(self.cart["id"], _cells(("f1", "cells", 5))), 1)
        self.assertEqual(self._labels(stars["id"]), [1])
        self.assertEqual(self._labels(self.cart["id"]), [1, 5])

    def test_removing_from_a_set_removes_the_star(self):
        self.store.star(self.cart["id"], _cells(("f1", "cells", 1), ("f1", "cells", 2)))
        self.store.remove_cells(self.cart["id"], _cells(("f1", "cells", 1)))
        self.assertEqual(self._labels(self.store.favourites(self.cart["id"])["id"]), [2])

    def test_every_set_can_have_its_own_favourites(self):
        other = self.store.save_set("other", _cells(("f1", "cells", 1)))
        self.store.star(self.cart["id"], _cells(("f1", "cells", 1)))
        self.store.star(other["id"], _cells(("f1", "cells", 1)))
        self.assertEqual(len([s for s in self.store.list_sets() if s["role"] == "favourites"]), 2)

    def test_a_snapshot_is_frozen_and_keeps_the_stars(self):
        self.store.star(self.cart["id"], _cells(("f1", "cells", 1)))
        self.store.add_cells(self.cart["id"], _cells(("f1", "cells", 2)))
        snap = self.store.snapshot(self.cart["id"], "cart then")
        self.assertTrue(snap["frozen"])
        self.assertEqual(snap["parents"], [self.cart["id"]])
        self.assertEqual(self._labels(snap["id"]), [1, 2])
        self.assertEqual(self._labels(self.store.favourites(snap["id"])["id"]), [1])
        for edit in (self.store.add_cells, self.store.remove_cells, self.store.star):
            with self.assertRaises(ValueError):
                edit(snap["id"], _cells(("f1", "cells", 3)))
        self.store.add_cells(self.cart["id"], _cells(("f1", "cells", 3)))  # the original stays editable
        self.assertEqual(self._labels(snap["id"]), [1, 2])

    def test_deleting_a_set_deletes_its_stars(self):
        self.store.star(self.cart["id"], _cells(("f1", "cells", 1)))
        deleted = self.store.delete_set(self.cart["id"])
        self.assertEqual(len(deleted), 2)
        self.assertEqual(self.store.list_sets(), [])
        self.assertEqual(list(self.store.folder.glob("*.parquet")), [])

    def test_children_names_do_not_clash_with_top_level_names(self):
        self.store.save_set("★ favourites", _cells(("f1", "cells", 1)))
        self.store.star(self.cart["id"], _cells(("f1", "cells", 1)))  # no name clash

    def test_format_1_sets_load_unchanged_and_editable(self):
        old = self.store.save_set("old", _cells(("f1", "cells", 1)), tags=["celltype:T"])
        index_path = self.store.folder / "index.json"
        index = json.loads(index_path.read_text())
        for key in ("container", "role", "frozen", "edits"):
            index["sets"][-1].pop(key)
        index["format"] = 1
        index_path.write_text(json.dumps(index))
        meta = self.store.get(old["id"])
        self.assertEqual((meta["container"], meta["role"], meta["frozen"], meta["edits"]), (None, None, False, []))
        self.assertEqual(meta["tags"], ["celltype:T"])
        self.assertEqual(self.store.add_cells(old["id"], _cells(("f1", "cells", 2))), 1)
        self.assertEqual(json.loads(index_path.read_text())["format"], 2)


# ---------------------------------------------------------------------------
# The bar
# ---------------------------------------------------------------------------
class BarTests(unittest.TestCase):
    def setUp(self):
        self.bar = SelectionBar(SimpleNamespace())

    def test_history_buttons(self):
        self.assertTrue(self.bar.back_button.disabled)
        self.bar.set_history(True, False)
        self.assertFalse(self.bar.back_button.disabled)
        self.assertTrue(self.bar.forward_button.disabled)
        calls = []
        self.bar.on_action("back", lambda: calls.append("back"))
        self.bar.trigger("back")
        self.assertEqual(calls, ["back"])

    def test_cart_buttons_show_only_with_an_active_cart(self):
        self.assertEqual(self.bar.cart_add_button.layout.display, "none")
        self.bar.set_cart("cool CD8")
        self.assertEqual(self.bar.cart_add_button.layout.display, "inline-flex")
        self.assertIn("cool CD8", self.bar.cart_remove_button.tooltip)
        calls = []
        self.bar.on_action("cart_add", lambda: calls.append("add"))
        self.bar.trigger("cart_add")
        self.assertEqual(calls, ["add"])
        self.bar.set_cart(None)
        self.assertEqual(self.bar.cart_remove_button.layout.display, "none")

    def test_save_form_replaces_the_text_and_shows_errors(self):
        self.assertEqual(self.bar.save_button.layout.display, "none")
        answers = iter(["A cell set named 'x' already exists.", ""])
        received = []
        self.bar.on_save(lambda name, tags: received.append((name, tags)) or next(answers))
        self.assertNotEqual(self.bar.save_button.layout.display, "none")
        self.bar.show_selection("<b>2 cells</b>", (), True)
        self.bar.begin_save()
        self.assertEqual(self.bar.save_form.layout.display, "flex")
        self.assertEqual(self.bar.actions.layout.display, "none")
        self.bar.name_input.value = "x"
        self.bar.tags_input.value = "k:v"
        self.bar.submit_save()
        self.assertIn("already exists", self.bar.save_status.value)
        self.assertTrue(self.bar.saving)
        self.bar.submit_save()
        self.assertFalse(self.bar.saving)
        self.assertEqual(self.bar.actions.layout.display, "flex")
        self.assertEqual(received, [("x", "k:v"), ("x", "k:v")])


# ---------------------------------------------------------------------------
# The viewer: ◀ ▶ replace the selection where it was shown
# ---------------------------------------------------------------------------
class StepSelectionTests(unittest.TestCase):
    def setUp(self):
        viewer = ImageMaskViewer.__new__(ImageMaskViewer)
        viewer.cell_table = _table()
        viewer.fov_key = "fov"
        viewer._map_mode_active = False
        viewer.ui_component = SimpleNamespace(
            image_selector=SimpleNamespace(value="fov1"),
            selection_bar=SelectionBar(SimpleNamespace()),
        )
        self.gallery = mock.Mock()
        viewer.SidePlots = SimpleNamespace(cell_gallery_output=self.gallery, chart_output=None, histogram_output=None)
        viewer.image_display = SimpleNamespace(selected_masks_label=set(), _lasso_active=False, clear_patches=mock.Mock())
        self.viewer = viewer

    def test_back_replaces_the_gallery_selection(self):
        hub = self.viewer.selection_hub
        hub.publish({0}, "histogram")
        hub.publish({1, 2}, "heatmap", {"gallery"})
        self.assertFalse(self.viewer.ui_component.selection_bar.back_button.disabled)
        self.viewer.step_selection(-1)
        self.gallery.set_selected_cells.assert_called_with([0])
        self.assertIn("gallery", hub.current.delivered)
        self.assertIn("step 1", self.viewer.ui_component.selection_bar.text.value)

    def test_back_to_an_image_lasso_outlines_it_again(self):
        hub = self.viewer.selection_hub
        hub.publish({0}, "image lasso", {"image"})
        hub.publish({1}, "heatmap", {"gallery"})
        with mock.patch("ueler.viewer.plugin._chart_common.sync_mask_highlights_from_selection") as sync:
            self.viewer.step_selection(-1)
        sync.assert_called_once_with(self.viewer, {0}, mask_name=None)
        self.assertIn("image", hub.current.delivered)

    def test_back_over_a_clear_restores_and_forward_clears_again(self):
        hub = self.viewer.selection_hub
        hub.publish({0, 1}, "heatmap", {"gallery"})
        self.viewer.clear_selection_everywhere()
        self.viewer.step_selection(-1)
        self.assertEqual(hub.current.indices, {0, 1})
        self.viewer.step_selection(+1)
        self.assertEqual(len(hub.current), 0)


class RecipeTests(unittest.TestCase):
    def test_lasso_recipe_keeps_at_most_200_vertices(self):
        display = ImageDisplay.__new__(ImageDisplay)
        display.main_viewer = SimpleNamespace(_map_mode_active=False)
        verts = [(float(i), float(i) / 3) for i in range(1000)]
        recipe = display._lasso_recipe(verts, {MaskSelection("fov1", "cells", 3)})
        self.assertLessEqual(len(recipe["vertices"]), 200)
        self.assertEqual(recipe["fov"], "fov1")
        self.assertEqual(recipe["masks"], ["cells"])
        self.assertEqual(recipe["vertices"][1], [5.0, 1.7])


# ---------------------------------------------------------------------------
# The plugin
# ---------------------------------------------------------------------------
class PluginTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        hub = SelectionHub()
        self.informed = []
        self.viewer = SimpleNamespace(
            **vars(_viewer()),
            base_folder=self._tmp.name,
            selection_hub=hub,
            image_display=SimpleNamespace(selected_masks_label=set()),
            inform_plugins=self.informed.append,
            ui_component=SimpleNamespace(
                selection_bar=SelectionBar(SimpleNamespace()),
                image_selector=SimpleNamespace(value="fov1", options=["fov1", "fov2", "fov3"]),
            ),
            get_active_fov=lambda: self.viewer.ui_component.image_selector.value,
            send_selection_to_image=lambda: hub.mark_delivered("image"),
            confirm=mock.Mock(return_value=True),
        )
        self.plugin = CellSetsPlugin(self.viewer, 6, 3)
        self.plugin.after_all_plugins_loaded()
        self.hub = hub

    def tearDown(self):
        self._tmp.cleanup()

    def test_bar_offers_save_once_the_store_exists(self):
        self.assertNotEqual(self.viewer.ui_component.selection_bar.save_button.layout.display, "none")

    def test_save_keeps_cells_recipe_and_trail(self):
        self.hub.publish({1, 2}, "histogram", recipe={"gates": {"CD8": ["above", 0, 2.0]}})
        self.assertEqual(self.plugin.save_selection("CD8 high", "celltype:CD8"), "")
        meta = self.plugin.store.list_sets()[0]
        self.assertEqual(meta["recipe"], {"gates": {"CD8": ["above", 0, 2.0]}})
        self.assertEqual(meta["tags"], ["celltype:CD8"])
        self.assertEqual(meta["trail"][0]["source"], "histogram")
        self.assertEqual(self.hub.history[-1].label, "saved as set")
        self.assertIn("CD8 high", self.viewer.ui_component.selection_bar.text.value)
        self.assertIn("CD8 high", self.plugin.ui_component.set_list.options[0][0])

    def test_save_reports_an_empty_selection_and_takes_plain_tags(self):
        self.assertIn("Nothing", self.plugin.save_selection("x"))
        self.hub.publish({1}, "scatter")
        self.assertEqual(self.plugin.save_selection("x", "interesting"), "")
        self.assertEqual(self.plugin.store.list_sets()[0]["tags"], ["interesting"])

    def test_write_to_selection_logs_and_offers_the_tag(self):
        self.hub.publish({0, 3}, "scatter")
        ui = self.plugin.ui_component
        ui.column_combo.value = "celltype"
        ui.value_input.value = "B cell"
        self.assertEqual(self.plugin.write_to_selection(), 2)
        self.assertEqual(self.viewer.cell_table.loc[[0, 3], "celltype"].tolist(), ["B cell", "B cell"])
        self.assertIn("on_cell_table_change", self.informed)
        self.assertEqual(self.hub.history[-1].recipe["column"], "celltype")
        self.assertEqual(ui.save_tags.value, "celltype:B cell")

    def test_select_set_moves_to_its_fov_and_outlines_on_its_mask(self):
        self.hub.publish({2, 3}, "image lasso", {"image"})
        self.viewer.image_display.selected_masks_label = {MaskSelection("fov2", "nuclei", 1), MaskSelection("fov2", "nuclei", 3)}
        self.plugin.save_selection("fov2 nuclei")
        set_id = self.plugin.store.list_sets()[0]["id"]
        self.hub.clear()
        self.assertTrue(self.plugin.select_set(set_id))
        self.assertEqual(self.viewer.ui_component.image_selector.value, "fov2")
        current = self.hub.current
        self.assertEqual(current.indices, {2, 3})
        self.assertEqual(current.source_label, "set: fov2 nuclei")
        self.assertEqual(current.recipe["mask"], "nuclei")
        self.assertIn("image", current.delivered)

    def test_view_stays_when_it_shows_part_of_the_set(self):
        self.hub.publish({0, 2, 3}, "scatter")
        self.plugin.save_selection("mixed")
        self.plugin.select_set(self.plugin.store.list_sets()[0]["id"])
        self.assertEqual(self.viewer.ui_component.image_selector.value, "fov1")

    def test_a_set_made_from_a_set_names_its_parent(self):
        self.hub.publish({0}, "scatter")
        self.plugin.save_selection("parent")
        parent = self.plugin.store.list_sets()[0]["id"]
        self.plugin.select_set(parent)
        self.plugin.save_selection("child")
        child = next(m for m in self.plugin.store.list_sets() if m["name"] == "child")
        self.assertEqual(child["parents"], [parent])

    def test_filter_by_tag(self):
        self.hub.publish({0}, "scatter")
        self.plugin.save_selection("a", "region:tumour")
        self.plugin.save_selection("b", "region:stroma")
        self.plugin.ui_component.filter_input.value = "tumour"
        self.plugin.refresh_sets()
        self.assertEqual(len(self.plugin.ui_component.set_list.options), 1)

    def test_delete_asks_first(self):
        self.hub.publish({0}, "scatter")
        self.plugin.save_selection("a")
        set_id = self.plugin.store.list_sets()[0]["id"]
        self.plugin.ask_delete_set(set_id)
        self.assertEqual(len(self.plugin.store.list_sets()), 1)
        message, on_confirm = self.viewer.confirm.call_args.args
        self.assertIn("cannot be undone", message)
        on_confirm()
        self.assertEqual(self.plugin.store.list_sets(), [])

    def test_edit_name_and_tags(self):
        self.hub.publish({0}, "scatter")
        self.plugin.save_selection("a")
        ui = self.plugin.ui_component
        ui.set_list.value = self.plugin.store.list_sets()[0]["id"]
        self.plugin._on_set_picked()
        ui.edit_name.value = "renamed"
        ui.edit_tags.value = "k:v"
        self.plugin._on_update_clicked()
        meta = self.plugin.store.list_sets()[0]
        self.assertEqual((meta["name"], meta["tags"]), ("renamed", ["k:v"]))

    # -- carts, stars, snapshots ---------------------------------------
    def _hit(self, fov, label, mask="cells"):
        return SimpleNamespace(fov_name=fov, mask_name=mask, mask_id=label)

    def test_new_empty_cart_becomes_active_and_shows_in_the_bar(self):
        self.assertEqual(self.plugin.create_cart("cool CD8", "pilot"), "")
        bar = self.viewer.ui_component.selection_bar
        self.assertEqual(bar.cart_add_button.layout.display, "inline-flex")
        self.assertIn("cool CD8", bar.cart_add_button.tooltip)
        cart = self.plugin.store.list_sets()[0]
        self.assertEqual(self.plugin.active_cart, cart["id"])
        self.assertEqual(self.plugin.ui_component.cart_dropdown.value, cart["id"])
        self.assertTrue(self.plugin.ui_component.set_list.options[0][0].startswith("🛒"))

    def test_plus_and_minus_cart_from_the_bar(self):
        self.plugin.create_cart("cart")
        self.hub.publish({0, 1}, "scatter")
        bar = self.viewer.ui_component.selection_bar
        bar.trigger("cart_add")
        cart = self.plugin.store.list_sets()[0]
        self.assertEqual(cart["count"], 2)
        self.assertIn("Added 2 cells", bar.text.value)
        self.assertEqual(cart["edits"][0]["recipe"]["source"], "scatter")
        self.assertEqual(self.hub.history[-1].label, "added to cart")
        self.hub.publish({1, 2}, "scatter")
        self.plugin.add_selection_to_cart()
        self.assertEqual(self.plugin.store.list_sets()[0]["count"], 3)
        self.assertIn("1 already in it", bar.text.value)
        bar.trigger("cart_remove")
        self.assertEqual(self.plugin.store.list_sets()[0]["count"], 1)

    def test_shift_click_toggles_a_cell_in_the_cart(self):
        self.assertIs(self.viewer.image_display.shift_click_handler.__func__, CellSetsPlugin.toggle_cell_in_cart)
        self.assertIsNone(self.plugin.toggle_cell_in_cart(self._hit("fov1", 2)))
        self.assertIn("no", self.viewer.ui_component.selection_bar.text.value.lower())
        self.plugin.create_cart("cart")
        self.assertEqual(self.plugin.toggle_cell_in_cart(self._hit("fov1", 2)), "added")
        self.assertEqual(self.plugin.toggle_cell_in_cart(self._hit("fov1", 2, "nuclei")), "added")
        self.assertEqual(self.plugin.toggle_cell_in_cart(self._hit("fov1", 2)), "removed")
        cells = self.plugin.store.read_cells(self.plugin.active_cart)
        self.assertEqual(cells[["mask", "label"]].values.tolist(), [["nuclei", 2]])
        self.assertEqual(len(self.hub.current), 0)  # the selection is not touched

    def test_star_and_select_stars(self):
        self.hub.publish({0, 1, 2}, "scatter")
        self.plugin.save_selection("set")
        set_id = self.plugin.store.list_sets()[0]["id"]
        self.hub.publish({1, 4}, "scatter")
        self.assertEqual(self.plugin.star_selection(set_id), 2)
        self.assertEqual(self.plugin._entry(set_id)["count"], 4)  # cell 4 joined the set
        self.assertIn("★ 2", self.plugin.ui_component.set_list.options[0][0])
        self.assertEqual(len(self.plugin.ui_component.set_list.options), 1)  # the stars are not a row
        self.assertTrue(self.plugin.select_stars(set_id))
        self.assertEqual(self.hub.current.indices, {1, 4})
        self.assertEqual(self.hub.current.source_label, "set: ★ set")
        self.hub.publish({4}, "scatter")
        self.assertEqual(self.plugin.star_selection(set_id, unstar=True), 1)

    def test_snapshot_is_frozen_and_cannot_be_the_cart(self):
        self.hub.publish({0}, "scatter")
        self.plugin.save_selection("set")
        set_id = self.plugin.store.list_sets()[0]["id"]
        snap = self.plugin.snapshot_set(set_id)
        self.assertTrue(snap["name"].startswith("set (snapshot "))
        self.assertNotIn(snap["id"], [value for _label, value in self.plugin.ui_component.cart_dropdown.options])
        self.plugin.set_active_cart(snap["id"])
        self.assertIsNone(self.plugin.active_cart)
        self.assertTrue(self.plugin.ui_component.star_button.disabled)

    def test_deleting_the_active_cart_clears_it(self):
        self.plugin.create_cart("cart")
        self.plugin._delete_confirmed(self.plugin.active_cart)
        self.assertIsNone(self.plugin.active_cart)
        self.assertEqual(self.viewer.ui_component.selection_bar.cart_add_button.layout.display, "none")


if __name__ == "__main__":
    unittest.main()
