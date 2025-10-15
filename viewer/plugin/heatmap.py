# viewer/cell_gallery.py

from ipywidgets import (SelectMultiple, FloatSlider, Dropdown, VBox, Output, TagsInput,
                        Checkbox, IntText, Text, Button, HBox, Layout, IntSlider, Tab, RadioButtons, HTML)
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
from viewer.observable import Observable
from viewer.plugin.plugin_base import PluginBase
from viewer.plugin.heatmap_adapter import HeatmapModeAdapter
from viewer.plugin.heatmap_layers import DataLayer, InteractionLayer, DisplayLayer

class HeatmapDisplay(DataLayer, InteractionLayer, DisplayLayer, PluginBase):
    def __init__(self, main_viewer, width, height):
        super().__init__(main_viewer, width, height)
        self.SidePlots_id = "heatmap_output"
        self.displayed_name = "Heatmap"
        self.main_viewer = main_viewer
        self.width = width
        self.height = height

        self.adapter = HeatmapModeAdapter(mode="vertical")

        self.ui_component = UiComponent(self)
        self.data = Data()
        self.plot_output = Output()

        self.orientation_state = {
            "horizontal": False,
            "view": None,
            "cluster_axis": 0,
            "marker_axis": 1,
            "cluster_index": None,
            "marker_index": None,
            "cluster_order_positions": [],
            "marker_order_positions": [],
            "cluster_leaves": [],
            "marker_leaves": []
        }

        self.initiate_ui()
        self.setup_widget_observers()
        self._reset_selection_cache()
        self.initialized = True
        # Ensure layout reflects the starting orientation before observers fire.
        self._sync_panel_location()

class UiComponent:
    def __init__(self, parent):
        self.channel_selector_text = HTML(
            value='Channels:',
        )

        self.channel_selector = TagsInput(
            value = parent.main_viewer.cell_table.columns[0],
            allowed_tags=parent.main_viewer.cell_table.columns.tolist(),  # This will be updated later
            description='Channels:',
            allow_duplicates=False,
            style={'description_width': 'auto'},
            layout=Layout(width='100%')
        )
        cluster_columns = parent.main_viewer.cell_table.select_dtypes(include=['int', 'int64', 'object']).columns.tolist()
        self.high_level_cluster_dropdown = Dropdown(
            options=cluster_columns,
            description='Class:',
            style={'description_width': 'auto'},
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
        self.cluster_method_dropdown = Dropdown(
            options=['single', 'complete', 'average', 'ward'],
            value='average',
            description='Linkage:',
        )
        self.distance_metric_dropdown = Dropdown(
            options=['euclidean', 'cityblock', 'cosine', 'correlation'],
            value='euclidean',
            description='Metric:',
        )
        self.horizontal_layout_checkbox = Checkbox(
            value=False,
            description='Horizontal layout',
            disabled=False,
            indent=False
        )
        self.horizontal_layout_checkbox.observe(parent.on_orientation_toggle, names='value')
        self.plot_button = Button(
            description='Plot',
            disabled=False,
            button_style='',
            tooltip='Plot the heatmap with dendrogram',
            icon='line-chart'
        )
        self.plot_button.on_click(parent.plot_heatmap)

        self.lock_cutoff_button = Checkbox(
            value=False,
            description='Lock Cutoff',
            disabled=False,
            indent=False
        )

        self.cluster_id_text = IntText(
            value=None,
            description='Meta-cluster ID:',
            disabled=True,
            style={'description_width': 'auto'}
        )
        # self.cluster_id_text.observe(self.on_cluster_id_change, names='value')

        self.cluster_id_apply_button = Button(
            description='Apply',
            disabled=True,
            button_style='',
            tooltip='Apply the cluster ID',
            icon='check'
        )

        self.cluster_id_apply_button.on_click(parent.apply_new_cluster_id)

        self.main_viewer_checkbox = Checkbox(
            value=False,
            description='Main viewer',
            disabled=False,
            indent=False
        )
        
        self.chart_checkbox = Checkbox(
            value=False,
            description='Chart',
            disabled=False,
            indent=False
        )

        self.cell_gallery_checkbox = Checkbox(
            value=False,
            description='Cell Gallery',
            disabled=False,
            indent=False
        )
        self.current_fov_checkbox = Checkbox(
            value=False,
            description='Current FOV',
            disabled=False,
            indent=False
        )
        
        self.column_name_text = Text(
            value='new_cluster',
            placeholder='Enter column name',
            description='Column Name:',
            disabled=False,
            style={'description_width': 'auto'}
        )

        self.save_to_cell_table_button = Button(
            description='Save to Cell Table',
            disabled=False,
            button_style='',
            tooltip='Save the current cluster labels to the cell table',
            icon='save'
        )

        self.save_to_cell_table_button.on_click(parent.save_to_cell_table)

        self.overwrite_checkbox = Checkbox(
            value=False,
            description='Overwrite',
            disabled=False,
            indent=False
        )

        # Widgets for tracing cell clusters
        self.trace_cluster_button = Button(
            description='Cluster',
            disabled=False,
            button_style='',
            tooltip='Trace the cluster of the selected cell',
            icon='search'
        )

        self.trace_cluster_button.on_click(parent.trace_cluster)

        self.trace_metacluster_button = Button(
            description='Meta-cluster',
            disabled=False,
            button_style='',
            tooltip='Trace the meta-cluster of the selected cell',
            icon='search'
        )

        self.trace_cluster_button.on_click(parent.trace_metacluster)

class Data:
    def __init__(self):
        self.current_clusters = {
            "index": Observable([])
        }
        self.cluster_colors = {}
        self.meta_cluster_colors = {}
        self.g = None

class linked_controls:
    def __init__(self):
        # Legacy widget linkage placeholder kept for interface parity.
        pass