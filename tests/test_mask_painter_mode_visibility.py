"""Tests for per-class mode (outline/fill) and per-class visibility in MaskPainterDisplay."""

import sys
import os
import types
import unittest
from pathlib import Path

import pandas as pd

from ueler.rendering import get_cell_color, clear_cell_colors


def _make_viewer():
    """Create a minimal mock viewer matching the pattern used across the test suite."""
    cell_table = pd.DataFrame({
        "fov": ["FOV_001", "FOV_001", "FOV_002", "FOV_002"],
        "label": [1, 2, 3, 4],
        "cell_type": ["TypeA", "TypeB", "TypeA", "TypeB"],
    })
    viewer = types.SimpleNamespace()
    viewer.cell_table = cell_table
    viewer.fov_key = "fov"
    viewer.label_key = "label"
    viewer.mask_key = "cell"
    viewer.base_folder = Path.cwd()

    ui_component = types.SimpleNamespace()
    ui_component.image_selector = types.SimpleNamespace()
    ui_component.image_selector.value = "FOV_001"
    viewer.ui_component = ui_component

    image_display = types.SimpleNamespace()
    image_display.set_mask_colors_current_fov = lambda **kwargs: None
    viewer.image_display = image_display

    viewer.get_active_fov = lambda: ui_component.image_selector.value
    return viewer


class TestMaskPainterModeVisibility(unittest.TestCase):
    """Tests for Phase 2-4: per-class visibility toggle and render mode."""

    def setUp(self):
        clear_cell_colors()
        self.viewer = _make_viewer()

    def tearDown(self):
        clear_cell_colors()

    def _make_painter(self):
        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay
        import ipywidgets as W

        painter = MaskPainterDisplay(self.viewer, width=400, height=300)
        painter.ui_component.identifier_dropdown.value = "cell_type"
        painter.current_identifier = "cell_type"
        painter.current_classes = ["TypeA", "TypeB"]
        painter.class_color_controls = {
            "TypeA": W.ColorPicker(description="TypeA", value="#FF0000"),
            "TypeB": W.ColorPicker(description="TypeB", value="#0000FF"),
        }
        from ipywidgets import Checkbox, Layout
        painter.class_visible_controls = {
            "TypeA": Checkbox(value=True, indent=False, layout=Layout(width="30px")),
            "TypeB": Checkbox(value=True, indent=False, layout=Layout(width="30px")),
        }
        painter.class_mode_controls = {
            "TypeA": Checkbox(value=False, description="fill", indent=False, layout=Layout(width="60px")),
            "TypeB": Checkbox(value=False, description="fill", indent=False, layout=Layout(width="60px")),
        }
        painter.ui_component.show_all_checkbox.value = True
        painter.selected_classes = ["TypeA", "TypeB"]
        painter._active_classes = ["TypeA", "TypeB"]
        painter._push_to_widget()
        return painter

    # ------------------------------------------------------------------
    # Phase 2 – hidden classes write "" to registry
    # ------------------------------------------------------------------

    def test_hidden_class_gets_empty_string_sentinel(self):
        """A class whose visibility checkbox is unchecked → '' in registry."""
        painter = self._make_painter()
        # Hide TypeB
        painter.class_visible_controls["TypeB"].value = False

        painter.apply_colors_to_masks(None, notify_cell_gallery=False)

        # TypeA should be colored
        color_a = get_cell_color(self.viewer.get_active_fov(), 1)
        self.assertEqual(color_a, "#FF0000")

        # TypeB should be invisible (empty string)
        color_b = get_cell_color(self.viewer.get_active_fov(), 2)
        self.assertEqual(color_b, "")

    def test_visible_class_stays_colored(self):
        """A class with visibility checkbox ticked keeps its assigned color."""
        painter = self._make_painter()
        painter.apply_colors_to_masks(None, notify_cell_gallery=False)

        color_a = get_cell_color(self.viewer.get_active_fov(), 1)
        self.assertEqual(color_a, "#FF0000")
        color_b = get_cell_color(self.viewer.get_active_fov(), 2)
        self.assertEqual(color_b, "#0000FF")

    # ------------------------------------------------------------------
    # Phase 4 – mode cache
    # ------------------------------------------------------------------

    def test_get_mode_map_returns_outline_by_default(self):
        """get_mode_map_for_fov returns 'outline' for cells registered with outline mode."""
        painter = self._make_painter()
        painter.class_mode_controls["TypeA"].value = False
        painter.class_mode_controls["TypeB"].value = False

        painter.apply_colors_to_masks(None, notify_cell_gallery=False)

        fov = self.viewer.get_active_fov()
        mode_map = painter.get_mode_map_for_fov(fov)
        # All cells should be outline; map may contain entries or be empty
        for mode in mode_map.values():
            self.assertEqual(mode, "outline")

    def test_get_mode_map_returns_fill_for_fill_class(self):
        """Cells of a class set to fill mode appear in mode_map as 'fill'."""
        painter = self._make_painter()
        painter.class_mode_controls["TypeA"].value = True
        painter.class_mode_controls["TypeB"].value = False

        painter.apply_colors_to_masks(None, notify_cell_gallery=False)

        fov = self.viewer.get_active_fov()
        mode_map = painter.get_mode_map_for_fov(fov)
        # mask_id 1 belongs to TypeA
        self.assertEqual(mode_map.get(1), "fill")
        # mask_id 2 belongs to TypeB
        self.assertEqual(mode_map.get(2, "outline"), "outline")

    def test_mode_cache_cleared_on_cell_table_change(self):
        """_cell_mode_cache and _last_applied_class_modes reset when cell table changes."""
        painter = self._make_painter()
        painter.class_mode_controls["TypeA"].value = True
        painter.apply_colors_to_masks(None, notify_cell_gallery=False)

        self.assertTrue(len(painter._cell_mode_cache) > 0)

        painter.on_cell_table_change()
        self.assertEqual(painter._cell_mode_cache, {})
        self.assertEqual(painter._last_applied_class_modes, {})

    def test_mode_change_is_skipped_when_unchanged(self):
        """_last_applied_class_modes caches mode; re-applying same mode is a no-op."""
        painter = self._make_painter()
        painter.class_mode_controls["TypeA"].value = True
        painter.apply_colors_to_masks(None, notify_cell_gallery=False)

        before = dict(painter._cell_mode_cache)
        # Applying again with same mode should not change cache (skip path)
        painter.apply_colors_to_masks(None, notify_cell_gallery=False)
        self.assertEqual(painter._cell_mode_cache, before)

    # ------------------------------------------------------------------
    # _get_visible_classes / _get_hidden_classes consistency
    # ------------------------------------------------------------------

    def test_visible_classes_excludes_unchecked(self):
        painter = self._make_painter()
        painter.class_visible_controls["TypeB"].value = False
        visible = painter._get_visible_classes()
        hidden = painter._get_hidden_classes()
        self.assertIn("TypeA", visible)
        self.assertNotIn("TypeB", visible)
        self.assertIn("TypeB", hidden)

    def test_show_all_true_still_respects_visibility_checkboxes(self):
        """Even with show_all=True, per-class checkbox can hide a class."""
        painter = self._make_painter()
        painter.ui_component.show_all_checkbox.value = True
        painter.class_visible_controls["TypeB"].value = False
        hidden = painter._get_hidden_classes()
        self.assertIn("TypeB", hidden)
        visible = painter._get_visible_classes()
        self.assertNotIn("TypeB", visible)

    # ------------------------------------------------------------------
    # Phase 8 – persistence round-trip
    # ------------------------------------------------------------------

    def test_save_and_load_modes_and_visibility(self):
        """Saving and loading a color set preserves mode and visible flags."""
        import tempfile
        from pathlib import Path
        from ueler.viewer.plugin.mask_painter import (
            write_color_set_file,
            read_color_set_file,
            serialize_class_color_controls,
        )

        painter = self._make_painter()
        painter.class_mode_controls["TypeA"].value = True
        painter.class_mode_controls["TypeB"].value = False
        painter.class_visible_controls["TypeB"].value = False

        with tempfile.TemporaryDirectory() as tmpdir:
            # Manually build and write a payload as save_current_color_set would
            class_order = painter._get_full_class_order()
            color_map = serialize_class_color_controls(
                painter.class_color_controls, class_order, painter.default_color,
                hidden_cache=painter.hidden_color_cache,
            )
            modes_map = {
                cls: ("fill" if getattr(painter.class_mode_controls.get(cls), "value", False) else "outline")
                for cls in class_order
            }
            visible_map = {
                cls: bool(getattr(painter.class_visible_controls.get(cls), "value", True))
                for cls in class_order
            }
            path = Path(tmpdir) / "test_palette.maskcolors.json"
            payload = {
                "name": "test",
                "version": "1",
                "identifier": "cell_type",
                "default_color": painter.default_color,
                "class_order": list(class_order),
                "colors": color_map,
                "modes": modes_map,
                "visible": visible_map,
                "saved_at": "2024-01-01T00:00:00Z",
            }
            write_color_set_file(path, payload)

            # Reset controls before loading
            painter.class_mode_controls["TypeA"].value = False
            painter.class_visible_controls["TypeB"].value = True

            painter._load_color_set(path)

            # Verify modes restored
            self.assertTrue(painter.class_mode_controls["TypeA"].value)   # fill
            self.assertFalse(painter.class_mode_controls["TypeB"].value)  # outline
            # Verify visibility restored
            self.assertFalse(painter.class_visible_controls["TypeB"].value)
            self.assertTrue(painter.class_visible_controls["TypeA"].value)


class TestMaskPainterAddRemoveClass(unittest.TestCase):
    """Tests for Add/Remove class feature in MaskPainterDisplay."""

    def setUp(self):
        clear_cell_colors()
        # Build a viewer with 10 class types
        classes_10 = [f"Type{i}" for i in range(10)]
        rows = []
        for i, cls in enumerate(classes_10):
            rows.append({"fov": "FOV_001", "label": i + 1, "cell_type": cls})
        cell_table = pd.DataFrame(rows)
        viewer = types.SimpleNamespace()
        viewer.cell_table = cell_table
        viewer.fov_key = "fov"
        viewer.label_key = "label"
        viewer.mask_key = "cell"
        viewer.base_folder = Path.cwd()
        ui_component = types.SimpleNamespace()
        ui_component.image_selector = types.SimpleNamespace()
        ui_component.image_selector.value = "FOV_001"
        viewer.ui_component = ui_component
        image_display = types.SimpleNamespace()
        image_display.set_mask_colors_current_fov = lambda **kwargs: None
        viewer.image_display = image_display
        viewer.get_active_fov = lambda: ui_component.image_selector.value
        self.viewer = viewer
        self.classes_10 = classes_10

    def tearDown(self):
        clear_cell_colors()

    def test_initial_selection_limited_to_six(self):
        """On identifier change with >6 classes, class_order has 6 and available_classes has the rest."""
        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay
        painter = MaskPainterDisplay(self.viewer, width=400, height=300)
        painter.ui_component.identifier_dropdown.value = "cell_type"
        # Trigger the change
        painter.on_identifier_change({"new": "cell_type"})

        w = painter.ui_component.class_list_widget
        self.assertEqual(len(w.class_order), 6)
        self.assertEqual(len(w.available_classes), 4)
        self.assertEqual(len(w.class_order) + len(w.available_classes), 10)
        # active + available = full class set
        all_from_widget = set(w.class_order) | set(w.available_classes)
        self.assertEqual(all_from_widget, set(self.classes_10))

    def test_add_class_moves_from_available_to_active(self):
        """_on_add_requested moves a class from available_classes to class_order."""
        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay
        painter = MaskPainterDisplay(self.viewer, width=400, height=300)
        painter.on_identifier_change({"new": "cell_type"})

        w = painter.ui_component.class_list_widget
        before_order = list(w.class_order)
        before_avail = list(w.available_classes)
        self.assertEqual(len(before_order), 6)

        # Request adding the first available class
        cls_to_add = before_avail[0]
        painter._on_add_requested({"new": cls_to_add})

        self.assertIn(cls_to_add, w.class_order)
        self.assertNotIn(cls_to_add, w.available_classes)
        self.assertEqual(len(w.class_order), 7)
        self.assertEqual(len(w.available_classes), 3)

    def test_remove_class_moves_from_active_to_available(self):
        """_on_remove_requested moves a class from class_order to available_classes."""
        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay
        painter = MaskPainterDisplay(self.viewer, width=400, height=300)
        painter.on_identifier_change({"new": "cell_type"})

        w = painter.ui_component.class_list_widget
        cls_to_remove = w.class_order[0]
        painter._on_remove_requested({"new": cls_to_remove})

        self.assertNotIn(cls_to_remove, w.class_order)
        self.assertIn(cls_to_remove, w.available_classes)
        self.assertEqual(len(w.class_order), 5)
        self.assertEqual(len(w.available_classes), 5)

    def test_removed_class_gets_sentinel_in_apply(self):
        """A class removed from the active list receives '' sentinel color when colors are applied."""
        import ipywidgets as W
        from ipywidgets import Checkbox, Layout
        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay
        painter = MaskPainterDisplay(self.viewer, width=400, height=300)
        # Restrict to 2 classes for simplicity
        painter.current_classes = ["TypeA", "TypeB"]
        painter._active_classes = ["TypeA", "TypeB"]
        painter.current_identifier = "cell_type"
        painter.ui_component.identifier_dropdown.value = "cell_type"
        painter.class_color_controls = {
            "TypeA": W.ColorPicker(description="TypeA", value="#FF0000"),
            "TypeB": W.ColorPicker(description="TypeB", value="#0000FF"),
        }
        painter.class_visible_controls = {
            "TypeA": Checkbox(value=True, indent=False, layout=Layout(width="30px")),
            "TypeB": Checkbox(value=True, indent=False, layout=Layout(width="30px")),
        }
        painter.class_mode_controls = {
            "TypeA": Checkbox(value=False, description="fill", indent=False, layout=Layout(width="60px")),
            "TypeB": Checkbox(value=False, description="fill", indent=False, layout=Layout(width="60px")),
        }
        painter._push_to_widget()

        # Remove TypeB
        painter._on_remove_requested({"new": "TypeB"})

        # Apply colors
        painter.apply_colors_to_masks(None, notify_cell_gallery=False)

        # TypeB cells (labels 2, 4 per the cell_table) get "" sentinel
        # Using the real cell_table rows for TypeB:
        typeb_rows = self.viewer.cell_table[self.viewer.cell_table["cell_type"] == "TypeB"]
        for _, row in typeb_rows.iterrows():
            color = get_cell_color(row["fov"], int(row["label"]))
            self.assertEqual(color, "", f"Expected '' for TypeB cell {row['label']}, got {color!r}")


if __name__ == "__main__":
    unittest.main()
