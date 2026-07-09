"""Tests for CellTableEditorPlugin."""

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stub heavy / optional dependencies before importing the plugin
# ---------------------------------------------------------------------------

def _stub_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


for _name in ("ipywidgets",):
    if _name not in sys.modules:
        _stub_module(_name)

# ipywidgets widget stubs
import ipywidgets as _ipy  # noqa: E402 – may be real or stub

for _cls in ("Button", "Combobox", "HTML", "HBox", "Layout", "Text", "VBox", "Widget"):
    if not hasattr(_ipy, _cls):
        _stub = type(_cls, (), {"__init__": lambda self, *a, **kw: None,
                                "on_click": lambda self, *a, **kw: None,
                                "observe": lambda self, *a, **kw: None})
        setattr(_ipy, _cls, _stub)

# layout_utils stub
if "ueler.viewer.layout_utils" not in sys.modules:
    lu = _stub_module("ueler.viewer.layout_utils")
    lu.column_block_layout = lambda **kw: None  # type: ignore[attr-defined]
    lu.content_widget_layout = lambda **kw: None  # type: ignore[attr-defined]

# plugin_base stub
if "ueler.viewer.plugin.plugin_base" not in sys.modules:
    _pb_mod = types.ModuleType("ueler.viewer.plugin.plugin_base")

    class _PluginBase:
        def __init__(self, viewer, width, height):
            self.viewer = viewer
            self.width = width
            self.height = height
            self.SidePlots_id = ""
            self.displayed_name = ""
            self.initialized = False

        def setup_widget_observers(self):
            pass

    _pb_mod.PluginBase = _PluginBase  # type: ignore[attr-defined]
    sys.modules["ueler.viewer.plugin.plugin_base"] = _pb_mod

if "ueler.viewer.plugin" not in sys.modules:
    sys.modules["ueler.viewer.plugin"] = types.ModuleType("ueler.viewer.plugin")

from ueler.viewer.plugin.cell_table_editor import CellTableEditorPlugin  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_viewer(cell_table: pd.DataFrame | None = None):
    viewer = SimpleNamespace()
    viewer.fov_key = "fov"
    viewer.x_key = "X"
    viewer.y_key = "Y"
    viewer.label_key = "label"
    viewer.mask_key = "whole_cell"
    viewer.cell_table = cell_table
    viewer.base_folder = Path.cwd()
    viewer._debug = False

    image_display = SimpleNamespace()
    image_display.selected_masks_label = set()
    viewer.image_display = image_display

    _informed = []
    viewer.inform_plugins = lambda event: _informed.append(event)
    viewer._informed = _informed

    return viewer


def _make_plugin(viewer):
    plugin = CellTableEditorPlugin.__new__(CellTableEditorPlugin)
    # bypass full __init__ to avoid widget rendering
    plugin.viewer = viewer
    plugin.main_viewer = viewer
    plugin.width = 6
    plugin.height = 3
    plugin.displayed_name = "Cell Table Editor"
    plugin.SidePlots_id = "cell_table_editor_output"
    plugin.initialized = False
    plugin.ui_component = SimpleNamespace()

    # minimal widget stubs with value attributes
    plugin.ui_component.column_combo = SimpleNamespace(value="", options=[])
    plugin.ui_component.value_input = SimpleNamespace(value="")
    plugin.ui_component.apply_btn = SimpleNamespace(disabled=True)
    plugin.ui_component.status_label = SimpleNamespace(value="")

    return plugin


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCellTableEditorApply(unittest.TestCase):

    def _make_ct(self):
        return pd.DataFrame({
            "fov": ["FOV1", "FOV1", "FOV2", "FOV2"],
            "label": [10, 20, 10, 30],
            "X": [1.0, 2.0, 3.0, 4.0],
            "Y": [1.0, 2.0, 3.0, 4.0],
        })

    def test_apply_updates_correct_row(self):
        ct = self._make_ct()
        viewer = _make_viewer(ct)
        plugin = _make_plugin(viewer)

        # Select cell with fov=FOV1, mask_id=10
        from ueler.viewer.image_display import MaskSelection
        viewer.image_display.selected_masks_label = {
            MaskSelection(fov="FOV1", mask="whole_cell", mask_id=10)
        }
        plugin.ui_component.column_combo.value = "region"
        plugin.ui_component.value_input.value = "area_A"

        plugin._on_apply_clicked(None)

        # Row 0 should be updated
        self.assertEqual(ct.loc[0, "region"], "area_A")
        # Row 1 (FOV1/20) should not be touched
        self.assertEqual(ct.loc[1, "region"], "")
        # Row 2 (FOV2/10) should not be touched
        self.assertEqual(ct.loc[2, "region"], "")

    def test_apply_notifies_plugins(self):
        ct = self._make_ct()
        viewer = _make_viewer(ct)
        plugin = _make_plugin(viewer)

        from ueler.viewer.image_display import MaskSelection
        viewer.image_display.selected_masks_label = {
            MaskSelection(fov="FOV1", mask="whole_cell", mask_id=10)
        }
        plugin.ui_component.column_combo.value = "my_label"
        plugin.ui_component.value_input.value = "X"

        plugin._on_apply_clicked(None)

        self.assertIn("on_cell_table_change", viewer._informed)

    def test_new_column_initialized_for_all_rows(self):
        ct = self._make_ct()
        viewer = _make_viewer(ct)
        plugin = _make_plugin(viewer)

        from ueler.viewer.image_display import MaskSelection
        viewer.image_display.selected_masks_label = {
            MaskSelection(fov="FOV1", mask="whole_cell", mask_id=10)
        }
        plugin.ui_component.column_combo.value = "brand_new"
        plugin.ui_component.value_input.value = "val"

        plugin._on_apply_clicked(None)

        # Column exists for all rows; unselected rows have default ""
        self.assertIn("brand_new", ct.columns)
        self.assertEqual(ct.loc[1, "brand_new"], "")
        self.assertEqual(ct.loc[2, "brand_new"], "")

    def test_multi_selection_updates_multiple_rows(self):
        ct = self._make_ct()
        viewer = _make_viewer(ct)
        plugin = _make_plugin(viewer)

        from ueler.viewer.image_display import MaskSelection
        viewer.image_display.selected_masks_label = {
            MaskSelection(fov="FOV1", mask="whole_cell", mask_id=10),
            MaskSelection(fov="FOV2", mask="whole_cell", mask_id=30),
        }
        plugin.ui_component.column_combo.value = "group"
        plugin.ui_component.value_input.value = "G1"

        plugin._on_apply_clicked(None)

        self.assertEqual(ct.loc[0, "group"], "G1")  # FOV1/10
        self.assertEqual(ct.loc[3, "group"], "G1")  # FOV2/30
        self.assertEqual(ct.loc[1, "group"], "")    # untouched

    def test_empty_column_name_shows_error(self):
        ct = self._make_ct()
        viewer = _make_viewer(ct)
        plugin = _make_plugin(viewer)

        from ueler.viewer.image_display import MaskSelection
        viewer.image_display.selected_masks_label = {
            MaskSelection(fov="FOV1", mask="whole_cell", mask_id=10)
        }
        plugin.ui_component.column_combo.value = "   "
        plugin.ui_component.value_input.value = "X"

        plugin._on_apply_clicked(None)

        self.assertIn("column name", plugin.ui_component.status_label.value.lower())
        self.assertNotIn("on_cell_table_change", viewer._informed)

    def test_no_selection_shows_error(self):
        ct = self._make_ct()
        viewer = _make_viewer(ct)
        plugin = _make_plugin(viewer)

        viewer.image_display.selected_masks_label = set()
        plugin.ui_component.column_combo.value = "region"
        plugin.ui_component.value_input.value = "X"

        plugin._on_apply_clicked(None)

        self.assertIn("selected", plugin.ui_component.status_label.value.lower())
        self.assertNotIn("on_cell_table_change", viewer._informed)

    def test_get_column_options_excludes_system_keys(self):
        ct = pd.DataFrame({
            "fov": ["F"], "whole_cell": [1], "X": [0.0], "Y": [0.0],
            "label": [1], "extra": ["E"],
        })
        viewer = _make_viewer(ct)
        plugin = _make_plugin(viewer)

        options = plugin._get_column_options()
        self.assertIn("extra", options)
        self.assertNotIn("fov", options)
        self.assertNotIn("whole_cell", options)
        self.assertNotIn("X", options)
        self.assertNotIn("Y", options)
        self.assertNotIn("label", options)

    def test_on_cell_table_change_refreshes_options(self):
        ct = self._make_ct()
        viewer = _make_viewer(ct)
        plugin = _make_plugin(viewer)

        # Before: no extra column
        plugin.on_cell_table_change()
        opts_before = list(plugin.ui_component.column_combo.options)

        # Add column to cell table and refresh
        ct["new_col"] = ""
        plugin.on_cell_table_change()
        opts_after = list(plugin.ui_component.column_combo.options)

        self.assertNotIn("new_col", opts_before)
        self.assertIn("new_col", opts_after)


if __name__ == "__main__":
    unittest.main()
