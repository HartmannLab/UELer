"""Selection history, saved cell sets, key:value tags and writing to the cell table.

Covers the session log in ``SelectionHub`` (◀ ▶, the cell limit that leaves the
log whole), ``CellSetStore`` and the cell-identity helpers, and the Cell sets
plugin that replaced the Cell Table Editor.
"""
from __future__ import annotations

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
    def test_parse(self):
        tags, problems = parse_tags(" celltype : CD8 T , region:tumour")
        self.assertEqual(tags, ["celltype:CD8 T", "region:tumour"])
        self.assertEqual(problems, [])

    def test_plain_tags_and_two_values_for_a_key_are_refused(self):
        tags, problems = parse_tags("interesting, celltype:A, celltype:B, :x")
        self.assertEqual(tags, ["celltype:A"])
        self.assertEqual(len(problems), 3)


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

    def test_save_reports_bad_tags_and_empty_selection(self):
        self.assertIn("Nothing", self.plugin.save_selection("x"))
        self.hub.publish({1}, "scatter")
        self.assertIn("key:value", self.plugin.save_selection("x", "interesting"))

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


if __name__ == "__main__":
    unittest.main()
