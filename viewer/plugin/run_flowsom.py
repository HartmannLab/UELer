# viewer/cell_gallery.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
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
from viewer.observable import Observable
from viewer.plugin.plugin_base import PluginBase
import os
import pickle
from pyFlowSOM import map_data_to_nodes, som
from viewer.decorators import update_status_bar

class RunFlowsom(PluginBase):
    def __init__(self, main_viewer, width, height):
        super().__init__(main_viewer, width, height)
        self.SidePlots_id = "flowsom_output"
        self.displayed_name = "FlowSOM"
        self.main_viewer = main_viewer
        self.width = width
        self.height = height

        self.ui_component = UiComponent(self)
        self.data = Data()
        self.plot_output = Output()

        self.initiate_ui()
        self.setup_widget_observers()

        # Register observer
        # self.data.current_clusters["index"].add_observer(self.update_ui_components)

        # Always run this at the end of __init__
        # self.load_widget_states(os.path.join(self.main_viewer.base_folder, ".UELer", f'{self.displayed_name}_widget_states.json'))
        self.initialized = True

    # def after_all_plugins_loaded(self):
    #     # Add observer to monitor changes to selected_indices
    #     pass

    def on_cell_table_change(self):
        # Update the channel_selector options
        self.ui_component.channel_selector.allowed_tags = self.main_viewer.cell_table.columns.tolist()
        self.ui_component.subset_on_dropdown.options = self.main_viewer.cell_table.select_dtypes(include=['int', 'int64', 'object']).columns.tolist()

    @update_status_bar
    def run_flowsom(self, b):
        # First, subset the data based on the selected high-level clusters
        subset_on = self.ui_component.subset_on_dropdown.value
        subset = list(self.ui_component.subset_selector.value)
        df = self.main_viewer.cell_table
        if subset:
            in_subset = df[subset_on].isin(subset)
            df_src = df.copy()
            df = df[in_subset].copy()
        
        # Get the selected channels
        channels_for_flowsom = self.ui_component.channel_selector.value
        
        # Assuming df and channels_for_flowsom are defined
        df_SOM = df.loc[:, channels_for_flowsom]

        # Get the parameters
        xdim = self.ui_component.xdim_input.value
        ydim = self.ui_component.ydim_input.value
        rlen = self.ui_component.rlen_input.value
        seed = self.ui_component.seed_input.value

        # Train SOM
        flowsom_array = df_SOM.to_numpy()
        node_output = som(flowsom_array, xdim=xdim, ydim=ydim, rlen=rlen, seed=seed)
        clusters, dists = map_data_to_nodes(node_output, flowsom_array)

        # Save the clusters in the cell_table
        column_name_text = self.ui_component.column_name_text.value
        df[column_name_text] = clusters

        if subset:
            # Update the original cell_table
            df_src.loc[in_subset, column_name_text] = clusters
            
            # Convert the column to int and keeps NaN
            df_src[column_name_text] = df_src[column_name_text].astype("Int64")
            df = df_src
        # df[column_name_text] = df[column_name_text].astype("category")

        # Update the cell_table
        self.main_viewer.cell_table = df
        self.main_viewer.inform_plugins("on_cell_table_change")

        print(f"FlowSOM clustering completed. The labels are saved in the column {column_name_text}")
    
    def on_subset_on_dropdown_change(self, change):
        selected_clusters = change['new']  # Get the selected clusters
        if selected_clusters:
            # Filter the cell_table based on the selected high-level clusters
            filtered_fovs = self.main_viewer.cell_table[selected_clusters].unique()
            # Update the subset_selector options
            self.ui_component.subset_selector.options = filtered_fovs
        else:
            # If no cluster is selected, show no options in the subset_selector
            self.ui_component.subset_selector.options = []

    def initiate_ui(self):
        # Define the layout for the HBox
        hbox_layout = Layout(
            display='flex',
            flex_flow='row',
            align_items='center',
            width='100%',
            justify_content='flex-start',
            gap='10px'  # Adjust gap as needed
        )
        setup = VBox([
            HBox([
                VBox([
                    self.ui_component.channel_selector_text,
                    self.ui_component.channel_selector,
                    self.ui_component.column_name_text
                    ], layout=Layout(width='50%', overflow='hidden')),
                VBox([
                    self.ui_component.subset_on_dropdown,
                    self.ui_component.subset_selector
                    ], layout=Layout(width='50%', overflow='hidden')),
                ]),
            HBox([self.ui_component.xdim_input, self.ui_component.ydim_input, self.ui_component.rlen_input, self.ui_component.seed_input], layout = hbox_layout),
            HBox([self.ui_component.run_button])
        ])
       
        self.ui = VBox([
            setup,
            VBox([self.plot_output], layout=Layout(height_max='400px'))
        ])

class UiComponent:
    def __init__(self, parent):
        max_widget_width = '100px'

        widget_style = {'description_width': 'auto'}

        self.SidePlots = dict()

        self.channel_selector_text = HTML(
            value='Channels:',
        )

        self.channel_selector = TagsInput(
            value = parent.main_viewer.cell_table.columns[0],
            allowed_tags=parent.main_viewer.cell_table.columns.tolist(),  # This will be updated later
            description='Channels:',
            allow_duplicates=False,
            style={'description_width': 'auto'},
            layout=Layout(width='100%'),
        )
        cluster_columns = parent.main_viewer.cell_table.select_dtypes(include=['int', 'int64', 'object']).columns.tolist()
        self.column_name_text = Text(
            value="FlowSOM_cluster",
            description='Save as:',
            style={'description_width': 'auto'},
            tooltip = "Name of the column in the cell table to save the FlowSOM cluster",
            layout=Layout(width='99%')
        )
        self.subset_on_dropdown = Dropdown(
            options=cluster_columns,
            description='Subset on:',
            style={'description_width': 'auto'},
            layout=Layout(width='99%')
        )
        self.subset_on_dropdown.observe(parent.on_subset_on_dropdown_change, names='value')
        self.subset_selector = SelectMultiple(
            options=[],  # This will be updated later
            description='Subset:',
            disabled=False,
            style={'description_width': 'auto'},
            layout=Layout(width='99%')
        )
        # Widgets for interactivity
        self.xdim_input = IntText(
            value=10,
            description='xdim:',
            style = widget_style,
            layout=Layout(
                flex='1 1 auto', width='auto'
                )
        )
        self.xdim_input.layout.max_width = max_widget_width
        self.ydim_input = IntText(
            value=10,
            description='ydim:',
            style = widget_style,
            layout=Layout(
                flex='1 1 auto', width='auto'
                )
        )
        self.ydim_input.layout.max_width = max_widget_width
        self.rlen_input = IntText(
            value=10,
            description='rlen:',
            style = widget_style,
            layout=Layout(
                flex='1 1 auto', width='auto'
                )
        )
        self.rlen_input.layout.max_width = max_widget_width
        self.seed_input = IntText(
            value=42,
            description='seed:',
            style = widget_style,
            layout=Layout(
                flex='1 1 auto', width='auto'
                )
        )
        self.seed_input.layout.max_width = max_widget_width
        self.run_button = Button(
            description='Run',
            disabled=False,
            button_style='',
            tooltip='Run FlowSOM clustering',
            icon='play'
        )
        self.run_button.on_click(parent.run_flowsom)

class Data:
    def __init__(self):
        self.current_clusters = {
        "index": Observable([])
    }

class linked_controls:
    def __init__(self):
        pass