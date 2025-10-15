# viewer/cell_gallery.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from ipywidgets import (SelectMultiple, FloatSlider, Dropdown, VBox, Output,TagsInput,
                        Checkbox, IntText, Text, Button, HBox, Layout,IntSlider, Tab, RadioButtons,HTML)
from IPython.display import display
from collections import OrderedDict
import matplotlib.font_manager as fm
from image_utils import process_single_crop, estimate_color_range, color_one_image
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import LassoSelector
from viewer.observable import Observable
from viewer.plugin.plugin_base import PluginBase
import os
import pickle
from pyFlowSOM import map_data_to_nodes, som
from viewer.decorators import update_status_bar
from scipy.spatial import KDTree

class RunFlowsom(PluginBase):
    def __init__(self, main_viewer, width, height):
        super().__init__(main_viewer, width, height)
        self.SidePlots_id = "region_output"
        self.displayed_name = "Region annotation"
        self.main_viewer = main_viewer
        self.width = width
        self.height = height
        self.ALLOWED_STEP_SIZES = [1, 5, 10, 20, 50, 100]

        self.ui_component = UiComponent(self)
        self.data = Data()

        # Initialize history stacks
        self.undo_stack = []
        self.redo_stack = []
        self.coords = [] # List of coordinates
        self.points = [] # List of points on the plot
        self.selected = [] # List of selected point indices
        self.step_size_index = 0
        self.point_size = 3
        self.save_history()

        self.mouse_press_pos = None
        self.is_dragging = False

        self.plot_output = Output()

        # Initialize drag detection
        self.drag_threshold = 5  # pixels

        # Initialize LassoSelector once
        self.ls = LassoSelector(self.main_viewer.image_display.ax, self.onselect, button=[1])  # Left-click for lasso

        self.step_size = self.ALLOWED_STEP_SIZES[self.step_size_index]

        self.initiate_ui()
        self.setup_widget_observers()

        # Register observer
        # self.data.current_clusters["index"].add_observer(self.update_ui_components)

        # Always run this at the end of __init__
        # self.load_widget_states(os.path.join(self.main_viewer.base_folder, ".UELer", f'{self.displayed_name}_widget_states.json'))

        self.load_points_to_coords()
        self.initialized = True

    def on_cell_table_change(self):
        pass

    def enable_draw(self, event):
        self.cid_press = self.main_viewer.image_display.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.main_viewer.image_display.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.main_viewer.image_display.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key = self.main_viewer.image_display.fig.canvas.mpl_connect('key_press_event', self.on_key)
    
    def update_kdtree(self):
        """Update the KDTree with current coordinates."""
        if self.coords:
            self.kdtree = KDTree(self.coords)
        else:
            self.kdtree = None

    def save_history(self):
        """Save the current state of coordinates to the undo stack and clear the redo stack."""
        self.undo_stack.append(self.coords.copy())
        self.redo_stack.clear()
    
    def onselect(self, verts):
        """Handle selection of points using LassoSelector."""
        path = Path(verts)
        self.selected = []
        # Reset all points to red
        for point in self.points:
            point.set_color('red')
        # Select new points and set them to orange
        for i, coord in enumerate(self.coords):
            if path.contains_point(coord):
                self.selected.append(i)
                self.points[i].set_color('orange')
                # annotate the selected points with their index
                self.main_viewer.image_display.ax.annotate(str(i), (coord[0], coord[1]), color='orange')
        self.main_viewer.image_display.fig.canvas.draw()
        print("Selected points:", [self.coords[i] for i in self.selected])
        self.save_history()

    def on_press(self, event):
        """Handle mouse press events to detect dragging."""
        if event.inaxes != self.main_viewer.image_display.ax:
            return
        self.mouse_press_pos = (event.x, event.y)
        self.is_dragging = False

    def on_motion(self, event):
        """Handle mouse motion to determine if dragging is occurring."""
        if self.mouse_press_pos is None:
            return
        if event.inaxes != self.main_viewer.image_display.ax:
            return
        dx = event.x - self.mouse_press_pos[0]
        dy = event.y - self.mouse_press_pos[1]
        distance = (dx**2 + dy**2)**0.5
        if distance > self.drag_threshold:
            self.is_dragging = True
            
    def on_release(self, event):
        """Handle mouse release events to add points if it was a click or unselect all on right-click."""
        if event.inaxes != self.main_viewer.image_display.ax or self.mouse_press_pos is None:
            self.mouse_press_pos = None
            self.is_dragging = False
            return
        if event.button == 1 and not self.is_dragging:  # Left-click release without drag
            click_coord = (event.xdata, event.ydata)
            self.coords.append(click_coord)
            point, = self.main_viewer.image_display.ax.plot(event.xdata, event.ydata, 'ro', markersize=self.point_size)
            self.points.append(point)
            self.main_viewer.image_display.fig.canvas.draw()
            print("Added point:", click_coord)
            self.save_history()
            self.auto_save_coords()  # Add this line
        elif event.button == 3:  # Right-click
            self.unselect_all_points()
        self.mouse_press_pos = None
        self.is_dragging = False
        self.update_kdtree()
    
    def unselect_all_points(self):
        """Unselect all points and reset their colors to red."""
        if self.selected:
            # Reset point colors
            for i in self.selected:
                self.points[i].set_color('red')
            
            # Clear text annotations
            for txt in self.main_viewer.image_display.ax.texts:
                # Only remove annotations for point indices, not other text like step size
                if txt != self.step_text:
                    txt.remove()
            
            self.selected.clear()
            self.main_viewer.image_display.fig.canvas.draw()
            print("Unselected all points")
            self.save_history()
    
    def on_key(self, event):
        """Handle key press events for various actions."""
        if event.key == '=':
            self.step_size_index = min(self.step_size_index + 1, len(self.ALLOWED_STEP_SIZES) - 1)
            self.step_size = self.ALLOWED_STEP_SIZES[self.step_size_index]
            self.update_step_text()
            print(f"Step size increased to {self.step_size}")
            return
        elif event.key == '-':
            self.step_size_index = max(self.step_size_index - 1, 0)
            self.step_size = self.ALLOWED_STEP_SIZES[self.step_size_index]
            self.update_step_text()
            print(f"Step size decreased to {self.step_size}")
            return
        elif event.key == 'c':
            self.duplicate_selected_points()
            return
        elif event.key in ['ctrl+z', 'meta+z']:
            self.undo()
            return
        elif event.key in ['ctrl+y', 'meta+y']:
            self.redo()
            return
        elif event.key == 'delete':
            self.delete_selected_points()
            return
        
        if not self.selected:
            return
        
        move_map = {
            'up': (0, self.step_size),
            'down': (0, -self.step_size),
            'left': (self.step_size, 0),
            'right': (-self.step_size, 0)
        }
        
        if event.key in move_map:
            dx, dy = move_map[event.key]
            for i in self.selected:
                x, y = self.coords[i]
                new_x, new_y = x + dx, y + dy
                self.coords[i] = (new_x, new_y)
            self.refresh_points()
            print(f"Moved points {event.key}")
            self.save_history()
            self.auto_save_coords()  # Add this line
    def undo(self):
        """Undo the last action by reverting to the previous state."""
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.coords = self.undo_stack[-1].copy()
            self.refresh_points()
            self.selected.clear()
            print("Undo performed.")
        else:
            print("Nothing to undo.")

    def redo(self):
        """Redo the last undone action by restoring the next state."""
        if self.redo_stack:
            self.undo_stack.append(self.redo_stack.pop())
            self.coords = self.undo_stack[-1].copy()
            self.refresh_points()
            self.selected.clear()
            print("Redo performed.")
        else:
            print("Nothing to redo.")
    
    def delete_all_points(self, event):
        """Delete all points on the plot."""
        if self.ui_component.confirm_delete_checkbox.value:
            self.coords.clear()
            for point in self.points:
                point.remove()
            self.points.clear()
            self.selected.clear()
            self.main_viewer.image_display.fig.canvas.draw()
            print("Deleted all points")
            self.save_history()
            self.auto_save_coords()  # Add this line
            self.update_kdtree()

            # set the confrim delete checkbox to False
            self.ui_component.confirm_delete_checkbox.value = False
        else:
            print("Please confirm deletion first.")

    def load_points_to_coords(self):
        """Load points from a CSV file to the current field of view."""
        current_fov = self.main_viewer.ui_component.image_selector.value
        load_path = os.path.join(self.ui_component.folder_input.value, f"{current_fov}.csv")
        if not os.path.exists(load_path):
            print(f"No file found at {load_path}")
            return
        df = pd.read_csv(load_path)
        self.coords = df[['x', 'y']].values.tolist()
        self.refresh_points()
        print(f"Loaded {len(self.coords)} points from {load_path}")
        self.save_history()
        self.update_kdtree()

    # def save_to_disk(self, event):
    #     """Export the current coordinates to a CSV file."""
    #     if not self.coords:
    #         print("No points to export.")
    #         return
    #     current_fov = self.main_viewer.ui_component.image_selector.value
    #     export_path = os.path.join(self.ui_component.folder_input.value, f"{current_fov}.csv")
        
    #     # conver to DataFrame
    #     df = pd.DataFrame(self.coords, columns=['x', 'y'])
    #     df.to_csv(export_path, index=False)
    #     print(f"Exported {len(self.coords)} points to {export_path}")
        
    def auto_save_coords(self):
        """Automatically save coordinates after any modification."""
        if not self.coords:
            return
        current_fov = self.main_viewer.ui_component.image_selector.value
        export_path = os.path.join(self.ui_component.folder_input.value, f"{current_fov}.csv")
        
        # convert to DataFrame
        df = pd.DataFrame(self.coords, columns=['x', 'y'])
        df.to_csv(export_path, index=False)

    def on_press(self, event):
        """Handle mouse press events to detect dragging."""
        if event.inaxes != self.main_viewer.image_display.ax:
            return
        self.mouse_press_pos = (event.x, event.y)
        self.is_dragging = False

    def delete_selected_points(self):
        """Delete the currently selected points."""
        if self.selected:
            for i in sorted(self.selected, reverse=True):
                self.coords.pop(i)
                self.points[i].remove()
                self.points.pop(i)
            self.selected.clear()
            self.main_viewer.image_display.fig.canvas.draw()
            print("Deleted selected points")
            self.save_history()
            self.auto_save_coords()  # Add this line
        self.update_kdtree()

    def refresh_points(self):
        """Refresh and redraw all points on the axes without resetting the zoom."""
        # Save the current view limits
        xlim = self.main_viewer.image_display.ax.get_xlim()
        ylim = self.main_viewer.image_display.ax.get_ylim()
        
        # Update the image data if it has changed
        self.main_viewer.image_display.img_display.set_data(self.main_viewer.image_display.img_display.get_array().copy())
        
        # Remove existing points
        for point in self.points:
            point.remove()
        self.points.clear()
        
        # Redraw all points
        for coord in self.coords:
            point, = self.main_viewer.image_display.ax.plot(coord[0], coord[1], 'ro', markersize=self.point_size)
            self.points.append(point)

        # Redraw selected points in orange
        for i in self.selected:
            self.points[i].set_color('orange')
        
        # Update the step text
        self.update_step_text()
        
        # Redraw the canvas
        self.main_viewer.image_display.fig.canvas.draw()
    
    def on_fov_change(self):
        """Handle changes to the current field of view."""
        self.coords.clear()
        self.refresh_points()
        self.load_points_to_coords()
        self.refresh_points()
    
    def update_step_text(self):
        """Update the step size text displayed on the plot."""
        if hasattr(self, 'step_text'):
            self.step_text.set_text(f'Step size: {self.step_size}')
        else:
            self.step_text = self.main_viewer.image_display.ax.text(
                0.02, 0.98, f'Step size: {self.step_size}', 
                transform=self.main_viewer.image_display.ax.transAxes,
                backgroundcolor='white',
                verticalalignment='top'
            )
        self.main_viewer.image_display.fig.canvas.draw()

    @update_status_bar
    def fit_spline(self, b):
        pass
    
    def initiate_ui(self):
        deletion_box = HBox([
            self.ui_component.confirm_delete_checkbox,
            self.ui_component.delete_button
            ])
        self.ui = VBox([
            self.ui_component.folder_input,
            self.ui_component.draw_button,
            deletion_box
            ])

class UiComponent:
    def __init__(self, parent):
        self.folder_input = Text(
            description="Folder",
            layout=Layout(width='auto')
        )

        self.draw_button = Button(
            description="Draw",
            layout=Layout(width='auto')
        )
        self.draw_button.on_click(parent.enable_draw)

        self.confirm_delete_checkbox = Checkbox(
            description="Confirm delete",
            value=False,
            layout=Layout(width='auto')
        )

        self.delete_button = Button(
            description="Delete all",
            layout=Layout(width='auto')
        )
        self.delete_button.on_click(parent.delete_all_points)

class Data:
    def __init__(self):
        self.points = {}

class linked_controls:
    def __init__(self):
        pass