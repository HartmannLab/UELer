"""Tests for per-class mode (outline/fill) and per-class visibility in MaskPainterDisplay."""

import sys
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
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
    ui_component.mask_color_controls = {"cell": types.SimpleNamespace(value="Green")}
    viewer.ui_component = ui_component
    viewer.predefined_colors = {"Green": "#00FF00", "White": "#FFFFFF", "Red": "#FF0000", "Blue": "#0000FF"}

    image_display = types.SimpleNamespace()
    image_display.set_mask_colors_current_fov = lambda **kwargs: None
    viewer.image_display = image_display

    viewer.get_active_fov = lambda: ui_component.image_selector.value
    return viewer


def _make_viewer_with_three_classes():
    cell_table = pd.DataFrame({
        "fov": ["FOV_001", "FOV_001", "FOV_001", "FOV_002"],
        "label": [1, 2, 3, 4],
        "cell_type": ["TypeA", "TypeB", "TypeC", "TypeC"],
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
    ui_component.mask_color_controls = {"cell": types.SimpleNamespace(value="Green")}
    viewer.ui_component = ui_component
    viewer.predefined_colors = {"Green": "#00FF00", "White": "#FFFFFF", "Red": "#FF0000", "Blue": "#0000FF"}

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

    def test_get_effective_opacity_map_returns_fill_opacity_for_fill_class(self):
        """Active visible fill classes should expose a per-cell opacity map."""
        import ipywidgets as W

        painter = self._make_painter()
        painter.class_mode_controls["TypeA"].value = True
        painter.class_opacity_controls = {
            "TypeA": W.BoundedIntText(value=70, min=0, max=100),
            "TypeB": W.BoundedIntText(value=35, min=0, max=100),
        }

        opacity_map = painter.get_effective_opacity_map_for_fov("FOV_001")

        self.assertAlmostEqual(opacity_map[1], 0.70, places=4)
        self.assertNotIn(2, opacity_map)

    def test_global_fill_opacity_updates_only_linked_classes(self):
        """Only classes still linked to the previous global opacity should move with it."""
        import ipywidgets as W

        painter = self._make_painter()
        painter.class_opacity_controls = {
            "TypeA": W.BoundedIntText(value=35, min=0, max=100),
            "TypeB": W.BoundedIntText(value=80, min=0, max=100),
        }
        painter.ui_component.global_fill_opacity_input.value = 35

        painter._on_global_fill_opacity_change({"old": 35, "new": 60})

        self.assertEqual(painter.class_opacity_controls["TypeA"].value, 60)
        self.assertEqual(painter.class_opacity_controls["TypeB"].value, 80)

    def test_capture_snapshot_records_modes_opacity_and_border_state(self):
        """Painter snapshots should preserve the per-class rendering state needed by downstream consumers."""
        import ipywidgets as W

        painter = self._make_painter()
        painter.class_mode_controls["TypeA"].value = True
        painter.class_opacity_controls = {
            "TypeA": W.BoundedIntText(value=55, min=0, max=100),
            "TypeB": W.BoundedIntText(value=35, min=0, max=100),
        }
        painter.ui_component.global_fill_opacity_input.value = 40
        painter.ui_component.show_fill_borders_checkbox.value = True
        self.viewer.mask_outline_thickness = 3

        snapshot = painter.capture_snapshot()

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.identifier, "cell_type")
        self.assertEqual(snapshot.class_fill["TypeA"], True)
        self.assertEqual(snapshot.class_opacity["TypeA"], 55)
        self.assertEqual(snapshot.global_fill_opacity, 40)
        self.assertEqual(snapshot.show_borders_on_filled, True)
        self.assertEqual(snapshot.border_color_mode, "mask_type_color")
        self.assertEqual(snapshot.mask_type_color, "#00ff00")
        self.assertEqual(snapshot.outline_thickness, 3)

    def test_apply_snapshot_restores_saved_state(self):
        """ROI replay should be able to restore a previously captured painter snapshot."""
        from ueler.rendering import MaskPainterSnapshot
        import ipywidgets as W

        self.viewer.update_display = lambda _factor: None
        painter = self._make_painter()
        painter.class_opacity_controls = {
            "TypeA": W.BoundedIntText(value=35, min=0, max=100),
            "TypeB": W.BoundedIntText(value=35, min=0, max=100),
        }

        restored = painter.apply_snapshot(
            MaskPainterSnapshot(
                mask_name="cell",
                identifier="cell_type",
                active_classes=("TypeA",),
                class_colors={"TypeA": "#00FF00", "TypeB": "#0000FF"},
                class_visible={"TypeA": True, "TypeB": False},
                class_fill={"TypeA": True, "TypeB": False},
                class_opacity={"TypeA": 65, "TypeB": 35},
                default_color="#FFFFFF",
                global_fill_opacity=45,
                show_borders_on_filled=True,
                border_color_mode="mask_type_color",
                mask_type_color="#00FF00",
                outline_thickness=2,
            )
        )

        self.assertTrue(restored)
        self.assertEqual(tuple(painter._active_classes), ("TypeA",))
        self.assertEqual(painter.class_color_controls["TypeA"].value, "#00FF00")
        self.assertEqual(painter.class_visible_controls["TypeB"].value, False)
        self.assertEqual(painter.class_mode_controls["TypeA"].value, True)
        self.assertEqual(painter.class_opacity_controls["TypeA"].value, 65)
        self.assertEqual(painter.ui_component.global_fill_opacity_input.value, 45)
        self.assertEqual(painter.ui_component.show_fill_borders_checkbox.value, True)
        self.assertEqual(painter.ui_component.border_color_mode_dropdown.value, "mask_type_color")
        self.assertEqual(self.viewer.ui_component.mask_color_controls["cell"].value, "Green")

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

    def test_mask_painter_starts_disabled(self):
        """Issue #90 requires the mask painter to be disabled by default."""
        painter = self._make_painter()
        self.assertFalse(painter.ui_component.enabled_checkbox.value)

    def test_inactive_class_is_not_treated_as_hidden(self):
        """Inactive classes should remain visible with the default color, not be hidden."""
        from ipywidgets import Checkbox, ColorPicker, Layout

        viewer = _make_viewer_with_three_classes()
        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay

        painter = MaskPainterDisplay(viewer, width=400, height=300)
        painter.ui_component.identifier_dropdown.value = "cell_type"
        painter.current_identifier = "cell_type"
        painter.current_classes = ["TypeA", "TypeB", "TypeC"]
        painter.class_color_controls = {
            "TypeA": ColorPicker(description="TypeA", value="#FF0000"),
            "TypeB": ColorPicker(description="TypeB", value="#0000FF"),
            "TypeC": ColorPicker(description="TypeC", value="#00FF00"),
        }
        painter.class_visible_controls = {
            "TypeA": Checkbox(value=True, indent=False, layout=Layout(width="30px")),
            "TypeB": Checkbox(value=False, indent=False, layout=Layout(width="30px")),
            "TypeC": Checkbox(value=True, indent=False, layout=Layout(width="30px")),
        }
        painter.class_mode_controls = {
            "TypeA": Checkbox(value=False, description="fill", indent=False, layout=Layout(width="60px")),
            "TypeB": Checkbox(value=False, description="fill", indent=False, layout=Layout(width="60px")),
            "TypeC": Checkbox(value=True, description="fill", indent=False, layout=Layout(width="60px")),
        }
        painter._active_classes = ["TypeA", "TypeB"]
        painter.selected_classes = ["TypeA", "TypeB"]
        painter._push_to_widget()

        hidden = painter._get_hidden_classes()

        self.assertIn("TypeB", hidden)
        self.assertNotIn("TypeC", hidden)

    def test_effective_color_map_uses_default_color_for_inactive_classes(self):
        """Current-FOV painter state should keep inactive classes visible in the default color."""
        from ipywidgets import Checkbox, ColorPicker, Layout

        viewer = _make_viewer_with_three_classes()
        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay

        painter = MaskPainterDisplay(viewer, width=400, height=300)
        painter.ui_component.identifier_dropdown.value = "cell_type"
        painter.current_identifier = "cell_type"
        painter.current_classes = ["TypeA", "TypeB", "TypeC"]
        painter.class_color_controls = {
            "TypeA": ColorPicker(description="TypeA", value="#FF0000"),
            "TypeB": ColorPicker(description="TypeB", value="#0000FF"),
            "TypeC": ColorPicker(description="TypeC", value="#00FF00"),
        }
        painter.class_visible_controls = {
            "TypeA": Checkbox(value=True, indent=False, layout=Layout(width="30px")),
            "TypeB": Checkbox(value=False, indent=False, layout=Layout(width="30px")),
            "TypeC": Checkbox(value=True, indent=False, layout=Layout(width="30px")),
        }
        painter.class_mode_controls = {
            "TypeA": Checkbox(value=False, description="fill", indent=False, layout=Layout(width="60px")),
            "TypeB": Checkbox(value=False, description="fill", indent=False, layout=Layout(width="60px")),
            "TypeC": Checkbox(value=True, description="fill", indent=False, layout=Layout(width="60px")),
        }
        painter._active_classes = ["TypeA", "TypeB"]
        painter.selected_classes = ["TypeA", "TypeB"]
        painter._push_to_widget()

        color_map = painter.get_effective_color_map_for_fov("FOV_001")

        self.assertEqual(color_map[1], "#FF0000")
        self.assertEqual(color_map[2], "")
        self.assertEqual(color_map[3], painter.default_color)

    # ------------------------------------------------------------------
    # Phase 8 – persistence round-trip
    # ------------------------------------------------------------------

    def test_save_and_load_modes_visibility_and_opacity(self):
        """Saving and loading a color set preserves mode, visibility, and opacity state."""
        import tempfile
        import ipywidgets as W
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
        painter.class_opacity_controls = {
            "TypeA": W.BoundedIntText(value=70, min=0, max=100),
            "TypeB": W.BoundedIntText(value=20, min=0, max=100),
        }
        painter.ui_component.global_fill_opacity_input.value = 35
        painter.ui_component.show_fill_borders_checkbox.value = True

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
                "opacities": {"TypeA": 70, "TypeB": 20},
                "global_fill_opacity": 35,
                "show_fill_borders": True,
                "saved_at": "2024-01-01T00:00:00Z",
            }
            write_color_set_file(path, payload)

            # Reset controls before loading
            painter.class_mode_controls["TypeA"].value = False
            painter.class_visible_controls["TypeB"].value = True
            painter.class_opacity_controls["TypeA"].value = 35
            painter.class_opacity_controls["TypeB"].value = 35
            painter.ui_component.show_fill_borders_checkbox.value = False

            painter._load_color_set(path)

            # Verify modes restored
            self.assertTrue(painter.class_mode_controls["TypeA"].value)   # fill
            self.assertFalse(painter.class_mode_controls["TypeB"].value)  # outline
            # Verify visibility restored
            self.assertFalse(painter.class_visible_controls["TypeB"].value)
            self.assertTrue(painter.class_visible_controls["TypeA"].value)
            self.assertEqual(painter.class_opacity_controls["TypeA"].value, 70)
            self.assertEqual(painter.class_opacity_controls["TypeB"].value, 20)
            self.assertEqual(painter.ui_component.global_fill_opacity_input.value, 35)
            self.assertTrue(painter.ui_component.show_fill_borders_checkbox.value)


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

    def test_removed_class_uses_default_color_in_apply(self):
        """A class removed from the active list remains visible with the default color."""
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

        # TypeB cells (labels 2, 4 per the cell_table) keep the default color
        typeb_rows = self.viewer.cell_table[self.viewer.cell_table["cell_type"] == "TypeB"]
        for _, row in typeb_rows.iterrows():
            color = get_cell_color(row["fov"], int(row["label"]))
            self.assertEqual(color, painter.default_color)


class TestMaskPainterRenderPath(unittest.TestCase):
    """Tests for issue #90 render-path integration in main_viewer._compose_fov_image."""

    def test_compose_fov_image_uses_current_mask_painter_state(self):
        """Single-FOV redraw must pass explicit current painter state into apply_registry_colors."""
        from ueler.viewer.main_viewer import ImageMaskViewer

        fake_painter = types.SimpleNamespace(
            get_effective_color_map_for_fov=lambda fov: {1: "#FF0000", 2: ""},
            get_effective_border_color_map_for_fov=lambda fov: {1: "#00FF00"},
            get_effective_mode_map_for_fov=lambda fov: {1: "fill"},
            get_effective_opacity_map_for_fov=lambda fov: {1: 0.7},
            get_show_borders_on_filled=lambda: True,
        )
        viewer = types.SimpleNamespace(
            image_cache={"FOV_001": {"ch1": np.zeros((2, 2), dtype=np.uint16)}},
            ui_component=types.SimpleNamespace(
                color_controls={"ch1": types.SimpleNamespace(value="Red")},
                contrast_min_controls={"ch1": types.SimpleNamespace(value=0.0)},
                contrast_max_controls={"ch1": types.SimpleNamespace(value=1.0)},
                mask_display_controls={"cell": types.SimpleNamespace(value=True)},
                mask_color_controls={},
            ),
            predefined_colors={"Red": "#FF0000"},
            annotations_available=False,
            annotation_display_enabled=False,
            active_annotation_name=None,
            masks_available=True,
            mask_key="cell",
            mask_outline_thickness=1,
            label_masks_cache={"FOV_001": {"cell": {1: np.array([[1, 0], [0, 2]], dtype=np.int32)}}},
            image_display=types.SimpleNamespace(selected_cells=[]),
            is_no_image_mode_enabled=lambda: False,
        )
        viewer.load_fov = lambda _fov, _channels: None
        viewer._get_label_mask_at_factor = lambda _fov, _mask, _ds: np.array([[1, 0], [0, 2]], dtype=np.int32)
        viewer._is_mask_painter_enabled = lambda: True
        viewer._get_mask_painter = lambda: fake_painter

        with patch("ueler.viewer.main_viewer.render_fov_to_array", return_value=np.zeros((2, 2, 3), dtype=np.float32)), \
             patch("ueler.viewer.main_viewer.collect_mask_regions", return_value={"cell": np.array([[1, 0], [0, 2]], dtype=np.int32)}), \
             patch("ueler.viewer.main_viewer.apply_registry_colors", side_effect=lambda image, **kwargs: image) as mock_apply:
            ImageMaskViewer._compose_fov_image(
                viewer,
                "FOV_001",
                ("ch1",),
                1,
                (0, 2, 0, 2),
                (0, 2, 0, 2),
            )

        kwargs = mock_apply.call_args.kwargs
        self.assertEqual(kwargs["color_map"], {1: "#FF0000", 2: ""})
        self.assertEqual(kwargs["border_color_map"], {1: "#00FF00"})
        self.assertEqual(kwargs["mode_map"], {1: "fill"})
        self.assertEqual(kwargs["opacity_map"], {1: 0.7})
        self.assertTrue(kwargs["show_borders_on_filled"])

    def test_compose_fov_image_forwards_no_image_mode(self):
        from ueler.viewer.main_viewer import ImageMaskViewer

        viewer = types.SimpleNamespace(
            image_cache={"FOV_001": {"ch1": np.zeros((2, 2), dtype=np.uint16)}},
            ui_component=types.SimpleNamespace(
                color_controls={"ch1": types.SimpleNamespace(value="Red"), "no_image_checkbox": types.SimpleNamespace(value=True)},
                contrast_min_controls={"ch1": types.SimpleNamespace(value=0.0)},
                contrast_max_controls={"ch1": types.SimpleNamespace(value=1.0)},
                mask_display_controls={},
                mask_color_controls={},
                no_image_checkbox=types.SimpleNamespace(value=True),
            ),
            predefined_colors={"Red": "#FF0000"},
            annotations_available=False,
            annotation_display_enabled=False,
            active_annotation_name=None,
            masks_available=False,
            mask_key="cell",
            mask_outline_thickness=1,
            image_display=types.SimpleNamespace(selected_cells=[]),
            is_no_image_mode_enabled=lambda: True,
        )
        viewer.load_fov = lambda _fov, _channels: None
        viewer._get_label_mask_at_factor = lambda _fov, _mask, _ds: None
        viewer._is_mask_painter_enabled = lambda: False
        viewer._get_mask_painter = lambda: None

        with patch("ueler.viewer.main_viewer.render_fov_to_array", return_value=np.zeros((2, 2, 3), dtype=np.float32)) as mock_render:
            ImageMaskViewer._compose_fov_image(
                viewer,
                "FOV_001",
                ("ch1",),
                1,
                (0, 2, 0, 2),
                (0, 2, 0, 2),
            )

        self.assertTrue(mock_render.call_args.kwargs["skip_image_layer"])

    def test_map_state_signature_includes_no_image_mode(self):
        from ueler.viewer.main_viewer import ImageMaskViewer

        viewer = types.SimpleNamespace(
            _map_mode_enabled=True,
            ui_component=types.SimpleNamespace(
                color_controls={"ch1": types.SimpleNamespace(value="Red")},
                contrast_min_controls={"ch1": types.SimpleNamespace(value=0.0)},
                contrast_max_controls={"ch1": types.SimpleNamespace(value=1.0)},
                mask_display_controls={},
                mask_color_controls={},
                no_image_checkbox=types.SimpleNamespace(value=False),
            ),
            predefined_colors={"Red": "#FF0000"},
            masks_available=False,
            annotations_available=False,
            annotation_display_enabled=False,
            active_annotation_name=None,
            annotation_palettes={},
            annotation_class_labels={},
            annotation_overlay_alpha=0.5,
            annotation_overlay_mode="combined",
            annotation_label_display_mode="legend",
            mask_outline_thickness=1,
            image_display=types.SimpleNamespace(selected_cells=[]),
            _is_mask_painter_enabled=lambda: False,
        )
        viewer.is_no_image_mode_enabled = lambda: bool(getattr(viewer.ui_component.no_image_checkbox, "value", False))

        signature_with_image = ImageMaskViewer._map_state_signature(viewer, ("ch1",), 1)
        viewer.ui_component.no_image_checkbox.value = True
        signature_without_image = ImageMaskViewer._map_state_signature(viewer, ("ch1",), 1)

        self.assertNotEqual(signature_with_image, signature_without_image)

    def test_apply_map_painter_overlay_uses_current_mask_painter_state(self):
        """Map-mode redraw must use the effective live painter state, not stale cached registry values."""
        from ueler.viewer.main_viewer import ImageMaskViewer

        fake_painter = types.SimpleNamespace(
            get_effective_color_map_for_fov=lambda fov: {1: "#FF0000"},
            get_effective_border_color_map_for_fov=lambda fov: {1: "#00FF00"},
            get_effective_mode_map_for_fov=lambda fov: {1: "fill"},
            get_effective_opacity_map_for_fov=lambda fov: {1: 0.4},
            get_mode_map_for_fov=lambda fov: {1: "outline"},
            get_opacity_map_for_fov=lambda fov: {1: 0.0},
            get_show_borders_on_filled=lambda: True,
        )

        class _ImageHandle:
            def __init__(self):
                self._data = None

            def set_data(self, data):
                self._data = np.array(data, copy=True)

        image_handle = _ImageHandle()
        viewer = types.SimpleNamespace(
            _map_mode_active=True,
            _active_map_id="slide-1",
            _is_mask_painter_enabled=lambda: True,
            _get_mask_painter=lambda: fake_painter,
            _get_map_layer=lambda _map_id: types.SimpleNamespace(
                last_tile_viewports=lambda: {
                    "FOV_001": types.SimpleNamespace(
                        region_xy=(0, 2, 0, 2),
                        downsample_factor=1,
                        dest_x0=0,
                        dest_x1=2,
                        dest_y0=0,
                        dest_y1=2,
                    )
                }
            ),
            _get_mask_array=lambda _fov, _mask: np.array([[1, 0], [0, 0]], dtype=np.int32),
            mask_key="cell",
            mask_outline_thickness=1,
            image_display=types.SimpleNamespace(
                _materialize_combined=lambda: np.zeros((2, 2, 3), dtype=np.float32),
                combined=None,
                img_display=image_handle,
                fig=types.SimpleNamespace(canvas=types.SimpleNamespace(draw_idle=lambda: None)),
            ),
        )

        with patch("ueler.viewer.main_viewer.get_all_cell_colors_for_fov", return_value={1: "#0000FF"}), \
             patch("ueler.viewer.main_viewer.apply_registry_colors", side_effect=lambda image, **kwargs: image) as mock_apply:
            ImageMaskViewer._apply_map_painter_overlay(viewer)

        kwargs = mock_apply.call_args.kwargs
        self.assertEqual(kwargs["color_map"], {1: "#FF0000"})
        self.assertEqual(kwargs["border_color_map"], {1: "#00FF00"})
        self.assertEqual(kwargs["mode_map"], {1: "fill"})
        self.assertEqual(kwargs["opacity_map"], {1: 0.4})
        self.assertTrue(kwargs["show_borders_on_filled"])


class TestMaskPainterOnlySpecified(unittest.TestCase):
    """Tests for the 'Only specified' toggle in MaskPainterDisplay."""

    def setUp(self):
        clear_cell_colors()
        self.viewer = _make_viewer()

    def tearDown(self):
        clear_cell_colors()

    def _make_painter_with_custom_colors(self):
        """Painter with TypeA customized and TypeB still at default color."""
        import ipywidgets as W
        from ipywidgets import Checkbox, Layout
        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay
        painter = MaskPainterDisplay(self.viewer, width=400, height=300)
        painter.ui_component.identifier_dropdown.value = "cell_type"
        painter.current_identifier = "cell_type"
        painter.current_classes = ["TypeA", "TypeB"]
        painter.class_color_controls = {
            "TypeA": W.ColorPicker(description="TypeA", value="#FF0000"),  # custom
            "TypeB": W.ColorPicker(description="TypeB", value=painter.default_color),  # default
        }
        painter.class_visible_controls = {
            "TypeA": Checkbox(value=True, indent=False, layout=Layout(width="30px")),
            "TypeB": Checkbox(value=True, indent=False, layout=Layout(width="30px")),
        }
        painter.class_mode_controls = {
            "TypeA": Checkbox(value=False, description="fill", indent=False, layout=Layout(width="60px")),
            "TypeB": Checkbox(value=False, description="fill", indent=False, layout=Layout(width="60px")),
        }
        painter._active_classes = ["TypeA", "TypeB"]
        painter._push_to_widget()
        return painter

    def test_only_specified_on_filters_default_color_classes(self):
        """Enabling 'Only specified' removes classes at default color from class_order."""
        painter = self._make_painter_with_custom_colors()
        w = painter.ui_component.class_list_widget
        # Before toggle: both classes active
        self.assertIn("TypeB", w.class_order)

        # Call handler directly (bootstrap stub Checkbox.observe() is a no-op)
        painter._on_only_specified_toggle({"new": True})

        self.assertIn("TypeA", w.class_order)
        self.assertNotIn("TypeB", w.class_order)
        self.assertIn("TypeB", w.available_classes)

    def test_only_specified_off_restores_all_classes(self):
        """Disabling 'Only specified' restores all current_classes to active."""
        painter = self._make_painter_with_custom_colors()
        w = painter.ui_component.class_list_widget

        # Turn on then off (call handler directly; bootstrap stub Checkbox.observe() is a no-op)
        painter._on_only_specified_toggle({"new": True})
        painter._on_only_specified_toggle({"new": False})

        self.assertIn("TypeA", w.class_order)
        self.assertIn("TypeB", w.class_order)
        self.assertEqual(len(w.available_classes), 0)

    def test_only_specified_when_all_custom_shows_all(self):
        """If all classes have custom colors, 'Only specified' keeps all classes active."""
        import ipywidgets as W
        from ipywidgets import Checkbox, Layout
        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay
        painter = MaskPainterDisplay(self.viewer, width=400, height=300)
        painter.current_classes = ["TypeA", "TypeB"]
        painter.current_identifier = "cell_type"
        painter.class_color_controls = {
            "TypeA": W.ColorPicker(description="TypeA", value="#FF0000"),
            "TypeB": W.ColorPicker(description="TypeB", value="#00FF00"),
        }
        painter.class_visible_controls = {
            "TypeA": Checkbox(value=True, indent=False, layout=Layout(width="30px")),
            "TypeB": Checkbox(value=True, indent=False, layout=Layout(width="30px")),
        }
        painter.class_mode_controls = {
            "TypeA": Checkbox(value=False, description="fill", indent=False, layout=Layout(width="60px")),
            "TypeB": Checkbox(value=False, description="fill", indent=False, layout=Layout(width="60px")),
        }
        painter._active_classes = ["TypeA", "TypeB"]
        painter._push_to_widget()

        # Call handler directly (bootstrap stub Checkbox.observe() is a no-op)
        painter._on_only_specified_toggle({"new": True})

        w = painter.ui_component.class_list_widget
        self.assertIn("TypeA", w.class_order)
        self.assertIn("TypeB", w.class_order)
        self.assertEqual(len(w.available_classes), 0)

    def test_only_specified_filtered_class_uses_default_color_on_apply(self):
        """Classes removed from the active list still render with the default color."""
        painter = self._make_painter_with_custom_colors()
        # Call handler directly (bootstrap stub Checkbox.observe() is a no-op)
        painter._on_only_specified_toggle({"new": True})

        painter.apply_colors_to_masks(None, notify_cell_gallery=False)

        # TypeA (custom color) should be colored
        color_a = get_cell_color("FOV_001", 1)
        self.assertEqual(color_a, "#FF0000")
        # TypeB (default color, filtered out) should still use the default color
        color_b = get_cell_color("FOV_001", 2)
        self.assertEqual(color_b, painter.default_color)


if __name__ == "__main__":
    unittest.main()
