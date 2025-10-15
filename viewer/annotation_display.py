# viewer/annotation_display.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from ipywidgets import (SelectMultiple, FloatSlider, Dropdown, VBox, Output,
                        Checkbox, IntText, Text, Button, HBox, Layout,IntSlider, Tab, RadioButtons)
from IPython.display import display
from collections import OrderedDict
import matplotlib.font_manager as fm
from image_utils import process_single_crop, estimate_color_range, color_one_image
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backend_bases import MouseButton
from viewer.observable import Observable



class AnnotationDisplay:
    def __init__(self, main_viewer, width, height):
        self.SidePlots_id = "annotation_display_output"
        self.displayed_name = "Cell annotation"
        self.main_viewer = main_viewer
        self.width = width
        self.height = height

        self.ui_component = UiComponent()
        self.data = Data()

        # Get columns that are of integer or string types
        if self.main_viewer.cell_table is not None:
            cell_table = self.main_viewer.cell_table
            label_columns = cell_table.select_dtypes(include=['int', 'int64', 'object']).columns.tolist()

            # Create checkboxes for each label column
            self.ui_component.label_checkboxes = []
            for col in label_columns:
                checkbox = Checkbox(
                    value=False,
                    description=col,
                    disabled=False
                )
                self.ui_component.label_checkboxes.append(checkbox)
                checkbox.observe(self.on_label_checkbox_change, names='value')
        else:
            self.ui_component.label_checkboxes = []

        self.initiate_ui()


    def initiate_ui(self):
        # Create a VBox for label checkboxes
        label_selection_box = VBox(self.ui_component.label_checkboxes, layout=Layout(flex_flow='column wrap'))

        # Update the UI to include the label selection box
        self.ui = VBox([
            label_selection_box,
            ],
          layout=Layout(max_height=f'{self.height}in')
          )
        
    def on_label_checkbox_change(self, change):
        # Update the list of selected labels
        selected_labels = [cb.description for cb in self.ui_component.label_checkboxes if cb.value]
        self.main_viewer.selected_tooltip_labels = selected_labels

    
class UiComponent:
    def __init__(self):
        pass

class Data:
    def __init__(self):
        pass

class linked_controls:
    def __init__(self):
        pass