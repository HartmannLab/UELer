"""Test that mask painter registers colors for all FOVs, not just the current one.

This test addresses issue #56: painted cell mask colors should be available
in the gallery regardless of which FOV is loaded in the main viewer.
"""

import sys
import types
import unittest
from pathlib import Path

import pandas as pd

from ueler.rendering import get_cell_color, clear_cell_colors


class TestPaintedColorsAllFovs(unittest.TestCase):
    """Verify that mask painter registers colors for all FOVs simultaneously."""

    def setUp(self):
        """Set up mock viewer with multiple FOVs."""
        clear_cell_colors()
        
        # Create a mock cell table with cells from multiple FOVs
        self.cell_table = pd.DataFrame({
            'fov': ['FOV_001', 'FOV_001', 'FOV_002', 'FOV_002', 'FOV_003'],
            'label': [1, 2, 3, 4, 5],
            'cell_type': ['TypeA', 'TypeB', 'TypeA', 'TypeB', 'TypeA'],
        })
        
        # Create a mock main viewer
        self.viewer = self._create_mock_viewer()

    def tearDown(self):
        """Clean up color registry."""
        clear_cell_colors()

    def _create_mock_viewer(self):
        """Create a minimal mock viewer for testing."""
        viewer = types.SimpleNamespace()
        viewer.cell_table = self.cell_table
        viewer.fov_key = 'fov'
        viewer.label_key = 'label'
        viewer.mask_key = 'cell'
        viewer.base_folder = Path.cwd()
        
        # Mock UI component with image selector
        ui_component = types.SimpleNamespace()
        ui_component.image_selector = types.SimpleNamespace()
        ui_component.image_selector.value = 'FOV_001'  # Currently viewing FOV_001
        viewer.ui_component = ui_component
        
        # Mock image display for set_mask_colors_current_fov
        image_display = types.SimpleNamespace()
        image_display.set_mask_colors_current_fov = lambda **kwargs: None
        viewer.image_display = image_display
        
        return viewer

    def test_colors_registered_for_all_fovs(self):
        """Test that applying colors registers them for all FOVs, not just current."""
        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay
        import ipywidgets
        
        # Create mask painter plugin
        painter = MaskPainterDisplay(self.viewer, width=400, height=300)
        
        # Set up identifier and classes
        painter.ui_component.identifier_dropdown.options = ['cell_type']
        painter.ui_component.identifier_dropdown.value = 'cell_type'
        painter.current_identifier = 'cell_type'
        painter.current_classes = ['TypeA', 'TypeB']
        
        # Create color pickers for each class
        painter.class_color_controls = {
            'TypeA': ipywidgets.ColorPicker(description='TypeA', value='#FF0000'),  # Red
            'TypeB': ipywidgets.ColorPicker(description='TypeB', value='#00FF00'),  # Green
        }
        
        # Set up visible classes
        painter.ui_component.sorting_items_tagsinput.allowed_tags = ['TypeA', 'TypeB']
        painter.ui_component.sorting_items_tagsinput.value = ('TypeA', 'TypeB')
        painter.selected_classes = ['TypeA', 'TypeB']
        painter.ui_component.show_all_checkbox.value = False
        
        # Apply colors while viewing FOV_001
        painter.apply_colors_to_masks(None, notify_cell_gallery=False)
        
        # Verify that colors are registered for ALL FOVs, not just FOV_001
        # TypeA cells: FOV_001/1, FOV_002/3, FOV_003/5
        self.assertEqual(get_cell_color('FOV_001', 1), '#FF0000', 
                        "TypeA in FOV_001 should be red")
        self.assertEqual(get_cell_color('FOV_002', 3), '#FF0000',
                        "TypeA in FOV_002 should be red (not just current FOV)")
        self.assertEqual(get_cell_color('FOV_003', 5), '#FF0000',
                        "TypeA in FOV_003 should be red (not just current FOV)")
        
        # TypeB cells: FOV_001/2, FOV_002/4
        self.assertEqual(get_cell_color('FOV_001', 2), '#00FF00',
                        "TypeB in FOV_001 should be green")
        self.assertEqual(get_cell_color('FOV_002', 4), '#00FF00',
                        "TypeB in FOV_002 should be green (not just current FOV)")

    def test_colors_available_before_loading_fov(self):
        """Test that gallery can access colors before loading the FOV in viewer."""
        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay
        import ipywidgets
        
        # Create mask painter plugin
        painter = MaskPainterDisplay(self.viewer, width=400, height=300)
        
        # Set up identifier and classes
        painter.ui_component.identifier_dropdown.value = 'cell_type'
        painter.current_identifier = 'cell_type'
        painter.current_classes = ['TypeA', 'TypeB']
        painter.class_color_controls = {
            'TypeA': ipywidgets.ColorPicker(description='TypeA', value='#0000FF'),  # Blue
            'TypeB': ipywidgets.ColorPicker(description='TypeB', value='#FFFF00'),  # Yellow
        }
        painter.selected_classes = ['TypeA', 'TypeB']
        painter.ui_component.sorting_items_tagsinput.value = ('TypeA', 'TypeB')
        painter.ui_component.show_all_checkbox.value = False
        
        # Apply colors while viewing FOV_001
        self.viewer.ui_component.image_selector.value = 'FOV_001'
        painter.apply_colors_to_masks(None, notify_cell_gallery=False)
        
        # Now switch to FOV_003 (which has never been loaded)
        self.viewer.ui_component.image_selector.value = 'FOV_003'
        
        # Color for FOV_003/5 should still be available (it was registered globally)
        self.assertEqual(get_cell_color('FOV_003', 5), '#0000FF',
                        "Color should be available for FOV_003 even though it was never loaded")

    def test_hidden_classes_registered_for_all_fovs(self):
        """Test that hidden classes (default color) are also registered globally."""
        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay
        import ipywidgets
        
        # Create mask painter plugin
        painter = MaskPainterDisplay(self.viewer, width=400, height=300)
        painter.default_color = '#808080'  # Gray
        
        # Set up identifier with only TypeA visible (TypeB hidden)
        painter.ui_component.identifier_dropdown.value = 'cell_type'
        painter.current_identifier = 'cell_type'
        painter.current_classes = ['TypeA', 'TypeB']
        painter.class_color_controls = {
            'TypeA': ipywidgets.ColorPicker(description='TypeA', value='#FF00FF'),  # Magenta
            'TypeB': ipywidgets.ColorPicker(description='TypeB', value='#00FFFF'),  # Cyan (but hidden)
        }
        painter.selected_classes = ['TypeA']  # Only TypeA is selected (visible)
        painter.ui_component.sorting_items_tagsinput.value = ('TypeA',)
        painter.ui_component.show_all_checkbox.value = False
        
        # Apply colors
        painter.apply_colors_to_masks(None, notify_cell_gallery=False)
        
        # Verify TypeA is colored across all FOVs
        self.assertEqual(get_cell_color('FOV_001', 1), '#FF00FF')
        self.assertEqual(get_cell_color('FOV_002', 3), '#FF00FF')
        self.assertEqual(get_cell_color('FOV_003', 5), '#FF00FF')
        
        # Verify TypeB (hidden) gets default color across all FOVs
        self.assertEqual(get_cell_color('FOV_001', 2), '#808080')
        self.assertEqual(get_cell_color('FOV_002', 4), '#808080')


if __name__ == '__main__':
    unittest.main()
