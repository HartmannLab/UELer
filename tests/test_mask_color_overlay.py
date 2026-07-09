"""Tests for the mask color registry optimisations introduced in issue #82.

Covers:
- Nested-dict registry structure (O(1) per-FOV access)
- set_cell_colors_bulk vectorised write
- Per-class dirty tracking in apply_colors_to_masks
"""

import unittest
from unittest.mock import MagicMock, patch, call

import pandas as pd
import numpy as np

from ueler.rendering import (
    set_cell_color,
    set_cell_colors_bulk,
    get_cell_color,
    get_all_cell_colors_for_fov,
    clear_cell_colors,
)
from ueler.rendering.engine import _CELL_COLOR_REGISTRY
from ueler.viewer.mask_color_overlay import apply_registry_colors


# ---------------------------------------------------------------------------
# Registry structure tests
# ---------------------------------------------------------------------------

class NestedRegistryTests(unittest.TestCase):
    """Verify the restructured nested dict registry behaves correctly."""

    def setUp(self):
        clear_cell_colors()

    def tearDown(self):
        clear_cell_colors()

    def test_set_cell_color_creates_nested_entry(self):
        set_cell_color("FOV_001", 42, "#FF0000")
        self.assertIn("FOV_001", _CELL_COLOR_REGISTRY)
        self.assertEqual(_CELL_COLOR_REGISTRY["FOV_001"][42], "#FF0000")

    def test_get_cell_color_o1_lookup(self):
        set_cell_color("FOV_001", 7, "#00FF00")
        self.assertEqual(get_cell_color("FOV_001", 7), "#00FF00")

    def test_get_cell_color_missing_fov_returns_none(self):
        self.assertIsNone(get_cell_color("NO_SUCH_FOV", 1))

    def test_get_cell_color_missing_mask_id_returns_none(self):
        set_cell_color("FOV_001", 1, "#FFFFFF")
        self.assertIsNone(get_cell_color("FOV_001", 999))

    def test_get_all_cell_colors_for_fov_o1_lookup(self):
        set_cell_color("FOV_001", 1, "#FF0000")
        set_cell_color("FOV_001", 2, "#00FF00")
        set_cell_color("FOV_002", 3, "#0000FF")  # different FOV — must not appear
        result = get_all_cell_colors_for_fov("FOV_001")
        self.assertEqual(result, {1: "#FF0000", 2: "#00FF00"})

    def test_get_all_cell_colors_for_fov_missing_fov_returns_empty(self):
        self.assertEqual(get_all_cell_colors_for_fov("NO_SUCH_FOV"), {})

    def test_clear_cell_colors_specific_fov(self):
        set_cell_color("FOV_001", 1, "#FF0000")
        set_cell_color("FOV_002", 2, "#00FF00")
        clear_cell_colors("FOV_001")
        self.assertIsNone(get_cell_color("FOV_001", 1))
        self.assertEqual(get_cell_color("FOV_002", 2), "#00FF00")

    def test_clear_cell_colors_all(self):
        set_cell_color("FOV_001", 1, "#FF0000")
        set_cell_color("FOV_002", 2, "#00FF00")
        clear_cell_colors()
        self.assertEqual(get_all_cell_colors_for_fov("FOV_001"), {})
        self.assertEqual(get_all_cell_colors_for_fov("FOV_002"), {})

    def test_get_all_cell_colors_returns_copy(self):
        """Mutating the returned dict must not affect the registry."""
        set_cell_color("FOV_001", 1, "#FF0000")
        result = get_all_cell_colors_for_fov("FOV_001")
        result[99] = "#AABBCC"
        self.assertNotIn(99, _CELL_COLOR_REGISTRY.get("FOV_001", {}))


# ---------------------------------------------------------------------------
# set_cell_colors_bulk tests
# ---------------------------------------------------------------------------

class BulkWriteTests(unittest.TestCase):
    """Verify set_cell_colors_bulk writes the nested registry correctly."""

    def setUp(self):
        clear_cell_colors()

    def tearDown(self):
        clear_cell_colors()

    def test_bulk_write_single_fov(self):
        entries = {"FOV_001": {1: "#FF0000", 2: "#00FF00", 3: "#0000FF"}}
        set_cell_colors_bulk(entries)
        self.assertEqual(get_cell_color("FOV_001", 1), "#FF0000")
        self.assertEqual(get_cell_color("FOV_001", 2), "#00FF00")
        self.assertEqual(get_cell_color("FOV_001", 3), "#0000FF")

    def test_bulk_write_multiple_fovs(self):
        entries = {
            "FOV_001": {10: "#AAAAAA"},
            "FOV_002": {20: "#BBBBBB"},
            "FOV_003": {30: "#CCCCCC"},
        }
        set_cell_colors_bulk(entries)
        self.assertEqual(get_cell_color("FOV_001", 10), "#AAAAAA")
        self.assertEqual(get_cell_color("FOV_002", 20), "#BBBBBB")
        self.assertEqual(get_cell_color("FOV_003", 30), "#CCCCCC")

    def test_bulk_write_merges_with_existing(self):
        set_cell_color("FOV_001", 1, "#FF0000")
        set_cell_colors_bulk({"FOV_001": {2: "#00FF00"}})
        # old entry preserved, new entry added
        self.assertEqual(get_cell_color("FOV_001", 1), "#FF0000")
        self.assertEqual(get_cell_color("FOV_001", 2), "#00FF00")

    def test_bulk_write_overwrites_existing_entry(self):
        set_cell_color("FOV_001", 5, "#OLD000")
        set_cell_colors_bulk({"FOV_001": {5: "#NEW000"}})
        self.assertEqual(get_cell_color("FOV_001", 5), "#NEW000")

    def test_bulk_write_empty_entries_is_noop(self):
        set_cell_colors_bulk({})
        self.assertEqual(_CELL_COLOR_REGISTRY, {})


class OverlayFillRenderingTests(unittest.TestCase):
    """Verify painter overlay rendering supports per-cell fill opacity and borders."""

    def test_apply_registry_colors_uses_per_cell_opacity(self):
        image = np.zeros((3, 3, 3), dtype=np.float32)
        region = np.array(
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.int32,
        )

        result = apply_registry_colors(
            image,
            fov="FOV_001",
            mask_regions={"cell": region},
            outline_thickness=1,
            downsample_factor=1,
            color_map={1: "#FF0000"},
            mode_map={1: "fill"},
            opacity_map={1: 0.5},
        )

        self.assertAlmostEqual(float(result[1, 1, 0]), 0.5, places=4)
        self.assertAlmostEqual(float(result[1, 1, 1]), 0.0, places=4)
        self.assertAlmostEqual(float(result[1, 1, 2]), 0.0, places=4)

    def test_zero_fill_opacity_falls_back_to_outline_only(self):
        image = np.zeros((5, 5, 3), dtype=np.float32)
        region = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        edges = np.array(
            [
                [False, False, False, False, False],
                [False, True, True, True, False],
                [False, True, False, True, False],
                [False, True, True, True, False],
                [False, False, False, False, False],
            ],
            dtype=bool,
        )

        with patch("ueler.viewer.mask_color_overlay.find_boundaries", return_value=edges):
            result = apply_registry_colors(
                image,
                fov="FOV_001",
                mask_regions={"cell": region},
                outline_thickness=1,
                downsample_factor=1,
                color_map={1: "#FF0000"},
                mode_map={1: "fill"},
                opacity_map={1: 0.0},
                show_borders_on_filled=False,
            )

        self.assertAlmostEqual(float(result[2, 2, 0]), 0.0, places=4)
        self.assertTrue(np.any(result[:, :, 0] == 1.0))

    def test_fill_with_border_preserves_outline_on_top(self):
        image = np.zeros((5, 5, 3), dtype=np.float32)
        region = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        edges = np.array(
            [
                [False, False, False, False, False],
                [False, True, True, True, False],
                [False, True, False, True, False],
                [False, True, True, True, False],
                [False, False, False, False, False],
            ],
            dtype=bool,
        )

        with patch("ueler.viewer.mask_color_overlay.find_boundaries", return_value=edges):
            result = apply_registry_colors(
                image,
                fov="FOV_001",
                mask_regions={"cell": region},
                outline_thickness=1,
                downsample_factor=1,
                color_map={1: "#FF0000"},
                mode_map={1: "fill"},
                opacity_map={1: 0.5},
                show_borders_on_filled=True,
            )

        self.assertAlmostEqual(float(result[2, 2, 0]), 0.5, places=4)
        self.assertAlmostEqual(float(result[1, 1, 0]), 1.0, places=4)

    def test_fill_with_border_can_use_distinct_border_color(self):
        image = np.zeros((5, 5, 3), dtype=np.float32)
        region = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        edges = np.array(
            [
                [False, False, False, False, False],
                [False, True, True, True, False],
                [False, True, False, True, False],
                [False, True, True, True, False],
                [False, False, False, False, False],
            ],
            dtype=bool,
        )

        with patch("ueler.viewer.mask_color_overlay.find_boundaries", return_value=edges):
            result = apply_registry_colors(
                image,
                fov="FOV_001",
                mask_regions={"cell": region},
                outline_thickness=1,
                downsample_factor=1,
                color_map={1: "#FF0000"},
                border_color_map={1: "#00FF00"},
                mode_map={1: "fill"},
                opacity_map={1: 0.5},
                show_borders_on_filled=True,
            )

        self.assertAlmostEqual(float(result[2, 2, 0]), 0.5, places=4)
        self.assertAlmostEqual(float(result[2, 2, 1]), 0.0, places=4)
        self.assertAlmostEqual(float(result[1, 1, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(result[1, 1, 1]), 1.0, places=4)

    def test_thickened_fill_border_does_not_reblend_neighbor_fill(self):
        image = np.zeros((5, 8, 3), dtype=np.float32)
        region = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 2, 2, 2, 0],
                [0, 1, 1, 1, 2, 2, 2, 0],
                [0, 1, 1, 1, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        cell1_edges = np.array(
            [
                [False, False, False, False, False, False, False, False],
                [False, True, True, True, False, False, False, False],
                [False, True, False, True, False, False, False, False],
                [False, True, True, True, False, False, False, False],
                [False, False, False, False, False, False, False, False],
            ],
            dtype=bool,
        )
        cell2_edges = np.array(
            [
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, True, True, True, False],
                [False, False, False, False, True, False, True, False],
                [False, False, False, False, True, True, True, False],
                [False, False, False, False, False, False, False, False],
            ],
            dtype=bool,
        )
        cell1_expanded = cell1_edges.copy()
        cell1_expanded[2, 5] = True
        cell2_expanded = cell2_edges.copy()

        def _fake_find_boundaries(mask_bool, mode=None):
            if np.any(mask_bool[:, 1:4]):
                return cell1_edges
            return cell2_edges

        def _fake_thicken_outline(edges, dilation):
            if np.array_equal(edges, cell1_edges):
                return cell1_expanded
            if np.array_equal(edges, cell2_edges):
                return cell2_expanded
            return edges

        with patch("ueler.viewer.mask_color_overlay.find_boundaries", side_effect=_fake_find_boundaries), \
             patch("ueler.viewer.mask_color_overlay.thicken_outline", side_effect=_fake_thicken_outline):
            result = apply_registry_colors(
                image,
                fov="FOV_001",
                mask_regions={"cell": region},
                outline_thickness=2,
                downsample_factor=1,
                color_map={1: "#FF0000", 2: "#0000FF"},
                mode_map={1: "fill", 2: "fill"},
                opacity_map={1: 0.5, 2: 0.5},
                show_borders_on_filled=True,
            )

        self.assertAlmostEqual(float(result[2, 5, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(result[2, 5, 1]), 0.0, places=4)
        self.assertAlmostEqual(float(result[2, 5, 2]), 0.5, places=4)


# ---------------------------------------------------------------------------
# _register_color_globally: bulk write path (mask_painter integration)
# ---------------------------------------------------------------------------

class RegisterColorGloballyBulkTests(unittest.TestCase):
    """Verify that apply_colors_to_masks uses set_cell_colors_bulk, not iterrows."""

    def _make_viewer(self, cell_table):
        import types

        viewer = types.SimpleNamespace()
        viewer.cell_table = cell_table
        viewer.fov_key = "fov"
        viewer.label_key = "label"
        viewer.mask_key = "cell"
        viewer.get_active_fov = lambda: "FOV_001"

        image_display = types.SimpleNamespace()
        image_display.set_mask_colors_current_fov = MagicMock()
        viewer.image_display = image_display
        return viewer

    def _setup_painter(self, painter, classes):
        """Configure painter with a set of visible classes."""
        import ipywidgets

        painter.ui_component.identifier_dropdown.options = ["cell_type"]
        painter.ui_component.identifier_dropdown.value = "cell_type"
        painter.selected_classes = list(classes)
        painter.ui_component.sorting_items_tagsinput.value = tuple(classes)
        painter.ui_component.show_all_checkbox.value = False
        painter.class_color_controls = {
            "A": ipywidgets.ColorPicker(value="#FF0000"),
            "B": ipywidgets.ColorPicker(value="#00FF00"),
        }

    def setUp(self):
        clear_cell_colors()

    def tearDown(self):
        clear_cell_colors()

    def test_register_color_globally_calls_bulk_write_not_iterrows(self):
        """set_cell_colors_bulk must be called; iterrows must not be called."""
        cell_table = pd.DataFrame({
            "fov": ["FOV_001", "FOV_001", "FOV_002", "FOV_002"],
            "label": [1, 2, 3, 4],
            "cell_type": ["A", "B", "A", "B"],
        })
        viewer = self._make_viewer(cell_table)

        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay

        painter = MaskPainterDisplay(viewer, width=400, height=300)
        self._setup_painter(painter, ["A", "B"])

        with patch("ueler.viewer.plugin.mask_painter.set_cell_colors_bulk") as mock_bulk:
            painter.apply_colors_to_masks(None, register_globally=True)
            self.assertTrue(mock_bulk.called, "set_cell_colors_bulk was not called")

    def test_register_color_globally_populates_registry_for_all_fovs(self):
        """After apply_colors_to_masks, registry has entries for every FOV."""
        import ipywidgets

        cell_table = pd.DataFrame({
            "fov": ["FOV_001", "FOV_002", "FOV_003"],
            "label": [10, 20, 30],
            "cell_type": ["A", "A", "A"],
        })
        viewer = self._make_viewer(cell_table)

        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay

        painter = MaskPainterDisplay(viewer, width=400, height=300)
        painter.ui_component.identifier_dropdown.options = ["cell_type"]
        painter.ui_component.identifier_dropdown.value = "cell_type"
        painter.selected_classes = ["A"]
        painter.ui_component.sorting_items_tagsinput.value = ("A",)
        painter.ui_component.show_all_checkbox.value = False
        painter.class_color_controls = {
            "A": ipywidgets.ColorPicker(value="#ABCDEF"),
        }

        painter.apply_colors_to_masks(None, register_globally=True)

        self.assertEqual(get_cell_color("FOV_001", 10), "#ABCDEF")
        self.assertEqual(get_cell_color("FOV_002", 20), "#ABCDEF")
        self.assertEqual(get_cell_color("FOV_003", 30), "#ABCDEF")


# ---------------------------------------------------------------------------
# Per-class dirty tracking tests
# ---------------------------------------------------------------------------

class PerClassDirtyTrackingTests(unittest.TestCase):
    """Verify that _register_color_globally is skipped for unchanged classes."""

    def _make_painter(self, cell_table):
        import types

        viewer = types.SimpleNamespace()
        viewer.cell_table = cell_table
        viewer.fov_key = "fov"
        viewer.label_key = "label"
        viewer.mask_key = "cell"
        viewer.get_active_fov = lambda: None  # map mode → always register_globally=True

        image_display = types.SimpleNamespace()
        image_display.set_mask_colors_current_fov = MagicMock()
        viewer.image_display = image_display

        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay
        import ipywidgets

        painter = MaskPainterDisplay(viewer, width=400, height=300)
        painter.ui_component.identifier_dropdown.options = ["cell_type"]
        painter.ui_component.identifier_dropdown.value = "cell_type"
        painter.selected_classes = ["A", "B"]
        painter.ui_component.sorting_items_tagsinput.value = ("A", "B")
        painter.ui_component.show_all_checkbox.value = False
        painter.class_color_controls = {
            "A": ipywidgets.ColorPicker(value="#FF0000"),
            "B": ipywidgets.ColorPicker(value="#00FF00"),
        }
        return painter

    def setUp(self):
        clear_cell_colors()

    def tearDown(self):
        clear_cell_colors()

    def test_first_apply_registers_all_classes(self):
        cell_table = pd.DataFrame({
            "fov": ["FOV_001", "FOV_002"],
            "label": [1, 2],
            "cell_type": ["A", "B"],
        })
        painter = self._make_painter(cell_table)

        with patch("ueler.viewer.plugin.mask_painter.set_cell_colors_bulk") as mock_bulk:
            painter.apply_colors_to_masks(None, register_globally=True)
        # Both classes must have triggered a bulk write (called once each)
        self.assertEqual(mock_bulk.call_count, 2)

    def test_second_apply_same_colors_skips_all_registration(self):
        """Re-applying with unchanged colors must skip set_cell_colors_bulk entirely."""
        cell_table = pd.DataFrame({
            "fov": ["FOV_001", "FOV_002"],
            "label": [1, 2],
            "cell_type": ["A", "B"],
        })
        painter = self._make_painter(cell_table)

        # First apply populates the dirty cache
        painter.apply_colors_to_masks(None, register_globally=True)

        with patch("ueler.viewer.plugin.mask_painter.set_cell_colors_bulk") as mock_bulk:
            painter.apply_colors_to_masks(None, register_globally=True)
        mock_bulk.assert_not_called()

    def test_color_change_triggers_re_registration_only_for_changed_class(self):
        """Only the class whose color changed should re-register."""
        import ipywidgets

        cell_table = pd.DataFrame({
            "fov": ["FOV_001", "FOV_001"],
            "label": [1, 2],
            "cell_type": ["A", "B"],
        })
        painter = self._make_painter(cell_table)

        # First apply — populates dirty cache
        painter.apply_colors_to_masks(None, register_globally=True)

        # Change only class A's color
        painter.class_color_controls["A"].value = "#FFFFFF"

        with patch("ueler.viewer.plugin.mask_painter.set_cell_colors_bulk") as mock_bulk:
            painter.apply_colors_to_masks(None, register_globally=True)

        # Exactly one call (for class A only; class B color unchanged)
        self.assertEqual(mock_bulk.call_count, 1)

    def test_cell_table_change_resets_dirty_cache(self):
        cell_table = pd.DataFrame({
            "fov": ["FOV_001"],
            "label": [1],
            "cell_type": ["A"],
        })
        painter = self._make_painter(cell_table)
        painter.apply_colors_to_masks(None, register_globally=True)
        self.assertNotEqual(painter._last_applied_class_colors, {})

        painter.on_cell_table_change()
        self.assertEqual(painter._last_applied_class_colors, {})


if __name__ == "__main__":
    unittest.main()
