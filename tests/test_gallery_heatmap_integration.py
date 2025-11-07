"""Test cell gallery integration with heatmap plugin."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Add bootstrap for matplotlib stubs
sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
import bootstrap  # noqa

from ueler.viewer.plugin.cell_gallery import CellGalleryDisplay


class TestHeatmapIntegration(unittest.TestCase):
    """Test that heatmap can trigger cell gallery updates."""
    
    def setUp(self):
        """Create mock viewer and gallery instance."""
        self.mock_viewer = MagicMock()
        self.mock_viewer.cell_table = MagicMock()
        self.mock_viewer.mask_outline_thickness = 2
        self.mock_viewer.channel_settings = {}
        
        # Create gallery instance
        self.gallery = CellGalleryDisplay(self.mock_viewer, 400, 600)
        
    def test_set_selected_cells_triggers_plot(self):
        """Verify set_selected_cells updates the observable and triggers rendering."""
        # Mock the plot method to track if it's called
        plot_called = []
        original_plot = self.gallery.plot_gellery
        
        def mock_plot():
            plot_called.append(True)
        
        self.gallery.plot_gellery = mock_plot
        
        # Set selected cells (as heatmap would do)
        test_indices = [1, 2, 3, 4, 5]
        self.gallery.set_selected_cells(test_indices)
        
        # Verify the observable was updated
        self.assertEqual(self.gallery.data.selected_cells.value, test_indices)
        
        # Verify plot was triggered
        self.assertTrue(len(plot_called) > 0, "plot_gellery should have been called")
        
        # Restore
        self.gallery.plot_gellery = original_plot
    
    def test_observable_notifies_on_value_change(self):
        """Verify Observable notifies observers when value changes."""
        notifications = []
        
        def observer(value):
            notifications.append(value)
        
        self.gallery.data.selected_cells.add_observer(observer)
        
        # Change value
        test_indices = [10, 20, 30]
        self.gallery.data.selected_cells.value = test_indices
        
        # Verify notification
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0], test_indices)
    
    def test_random_sampling_with_large_selection(self):
        """Verify random sampling works when selection exceeds max."""
        from ueler.viewer.plugin.cell_gallery import _limit_selection
        
        # Create large selection
        large_selection = list(range(200))
        max_cells = 50
        
        # Apply limit
        result = _limit_selection(large_selection, max_cells)
        
        # Verify results
        self.assertEqual(len(result), max_cells, "Should limit to max_cells")
        self.assertTrue(all(i in large_selection for i in result), "All sampled indices should be from original")
        self.assertEqual(len(set(result)), len(result), "No duplicates allowed")
        self.assertEqual(result, sorted(result), "Result should be sorted")


if __name__ == "__main__":
    unittest.main()
