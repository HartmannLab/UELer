# ui_components.py
from ipywidgets import (SelectMultiple, FloatSlider, Dropdown, Box, VBox, Output, Image,
                        Checkbox, IntText, Text, Button, HBox, Accordion, Layout, Tab,
                        TagsInput, HTML, ToggleButtons)
from IPython.display import display
from .plugin.chart import ChartDisplay
from .plugin.cell_gallery import CellGalleryDisplay
from .plugin.heatmap import HeatmapDisplay
from .annotation_display import AnnotationDisplay
from types import SimpleNamespace

from viewer.plugin.plugin_base import PluginBase


def build_wide_plugin_pane(control=None, content=None):
    """Compose the standard left/right layout used in the footer tabs."""
    if control is None and content is None:
        return VBox(children=tuple(), layout=Layout(width='100%'))

    if control is None:
        return VBox(children=(content,), layout=Layout(width='100%', overflow_y='auto'))

    if content is None:
        return VBox(children=(control,), layout=Layout(width='100%', overflow_y='auto'))

    control_box = VBox(children=(control,), layout=Layout(width='320px', flex='0 0 320px', overflow_y='auto', gap='8px'))
    content_box = VBox(children=(content,), layout=Layout(flex='1 1 auto', overflow='auto', min_height='360px'))
    return HBox(children=(control_box, content_box), layout=Layout(width='100%', gap='12px', align_items='stretch'))


def collect_wide_plugin_entries(viewer):
    """Inspect active plugins and gather footer-ready layouts."""
    entries = []
    sideplots = getattr(viewer, 'SidePlots', None)
    if sideplots is None:
        return entries

    for attr_name in dir(sideplots):
        plugin = getattr(sideplots, attr_name)
        if not isinstance(plugin, PluginBase):
            continue
        layout_provider = getattr(plugin, 'wide_panel_layout', None)
        if layout_provider is None:
            continue
        layout = layout_provider()
        if not layout:
            continue
        control = layout.get('control') if isinstance(layout, dict) else None
        content = layout.get('content') if isinstance(layout, dict) else None
        title = layout.get('title') if isinstance(layout, dict) else None
        entries.append({
            'attr': attr_name,
            'plugin': plugin,
            'title': title or plugin.displayed_name or attr_name,
            'control': control,
            'content': content
        })

    return entries


def update_wide_plugin_panel(viewer, ordering=None):
    if not hasattr(viewer, 'wide_plugin_tab') or not hasattr(viewer, 'wide_plugin_panel'):
        return

    entries = collect_wide_plugin_entries(viewer)

    bottom_ns = getattr(viewer, 'BottomPlots', None)
    if bottom_ns is None:
        bottom_ns = SimpleNamespace()
        viewer.BottomPlots = bottom_ns

    for attr in list(vars(bottom_ns).keys()):
        delattr(bottom_ns, attr)

    if not entries:
        viewer.wide_plugin_tab.children = []
        viewer.wide_plugin_panel.layout.display = 'none'
        return

    key = ordering or (lambda entry: entry['title'].lower())
    entries = sorted(entries, key=key)

    tab_children = []
    for entry in entries:
        pane = build_wide_plugin_pane(entry.get('control'), entry.get('content'))
        tab_children.append(pane)
        setattr(bottom_ns, entry['attr'], entry['plugin'])

    viewer.wide_plugin_tab.children = tab_children
    for idx, entry in enumerate(entries):
        viewer.wide_plugin_tab.set_title(idx, entry['title'])

    selected = getattr(viewer.wide_plugin_tab, 'selected_index', None)
    if viewer.wide_plugin_tab.children and selected is None:
        viewer.wide_plugin_tab.selected_index = 0

    viewer.wide_plugin_panel.layout.display = ''

    sideplots = getattr(viewer, 'SidePlots', None)
    heatmap_plugin = getattr(sideplots, 'heatmap_output', None) if sideplots else None
    if isinstance(heatmap_plugin, PluginBase):
        restore = getattr(heatmap_plugin, 'restore_vertical_canvas', None)
        if callable(restore):
            restore()

def create_widgets(viewer):
    viewer.ui_component = uicomponents(viewer)

def display_ui(viewer):
    """Display the main UI."""
    marker_set_widgets = VBox([
        VBox([viewer.ui_component.marker_set_dropdown, viewer.ui_component.marker_set_name_input]),
        HBox([
            VBox([viewer.ui_component.load_marker_set_button, viewer.ui_component.save_marker_set_button]),
            VBox([viewer.ui_component.update_marker_set_button, viewer.ui_component.delete_marker_set_button])
        ]),
        viewer.ui_component.delete_confirmation_checkbox
    ])
    # Add a new output widget for charts
    viewer.BottomPlots = BottomPlots()
    if viewer.cell_table is not None:
        viewer.SidePlots = SidePlots()
        viewer.SidePlots.annotation_display_output = AnnotationDisplay(viewer,6,3)
        viewer.dynamically_load_plugins()

        # Dynamically create Accordion children
        accordion_children = []
        for attr_name in dir(viewer.SidePlots):
            attr = getattr(viewer.SidePlots, attr_name)
            if hasattr(attr, 'ui') and hasattr(attr, 'displayed_name'):
                accordion_children.append(
                    Accordion(
                        children=[attr.ui],
                        titles=(attr.displayed_name,),
                        layout=Layout(width='6in')
                    )
                )

        viewer.side_plot = VBox(accordion_children)
    else:
        viewer.side_plot = Output()

    viewer.wide_plugin_tab = Tab(children=[], layout=Layout(width='100%'))
    viewer.wide_plugin_panel = VBox(
        [viewer.wide_plugin_tab],
        layout=Layout(
            width='100%',
            margin='12px 0 0 0',
            border='1px solid var(--jp-border-color2, #cccccc)',
            padding='8px',
            display='none'
        )
    )

    # Now wrap the container in a Box that makes it scrollable as a whole
    control_panel_stack = VBox([
        viewer.ui_component.control_sections,
        viewer.ui_component.annotation_editor_host
    ], layout=Layout(gap='8px'))

    top_part_widgets = VBox([
        viewer.ui_component.cache_size_input,
        viewer.ui_component.image_selector,
        viewer.ui_component.channel_selector_text,
        viewer.ui_component.channel_selector,
        marker_set_widgets,
        control_panel_stack,
        VBox([viewer.ui_component.advanced_settings_accordion])
    ])

    left_panel_children = [top_part_widgets]
    left_panel_children.append(
        HBox([HTML(value="Status:"), viewer.ui_component.status_bar])
    )

    left_panel = VBox(
        left_panel_children,
        layout=Layout(width='350px', overflow_y='auto', gap='10px')
    )

    ui = HBox([
        left_panel,
        viewer.image_output,
        viewer.side_plot  # Add the chart output widget to the right
    ])

    root = VBox([ui, viewer.wide_plugin_panel], layout=Layout(width='100%'))

    if hasattr(viewer, 'refresh_bottom_panel'):
        viewer.refresh_bottom_panel()

    display(root)

class SidePlots:
    def __init__(self):
        pass


class BottomPlots:
    def __init__(self):
        pass

class uicomponents:
    def __init__(self, viewer):
        """Initialize and return all UI widgets."""
        # Initialize cache size input
        self.cache_size_input = IntText(
            value=3,
            description='Cache Size:',
            disabled=False
        )
        self.cache_size_input.observe(viewer.on_cache_size_change, names='value')

        # Initialize image selector dropdown
        self.image_selector = Dropdown(
            options=viewer.available_fovs,
            value=viewer.available_fovs[0],
            description='Select Image:',
            disabled=False
        )
        self.image_selector.observe(viewer.on_image_change, names='value')

        self.channel_selector_text = HTML(value='Channels:')

        # Initialize channel selector
        self.channel_selector = TagsInput(
            allowed_tags=[],  # This will be updated later
            description='Channels:',
            disabled=False,
            layout=Layout(width='100%')
        )
        self.channel_selector.observe(viewer.update_controls, names='value')
        self.channel_selector.observe(viewer.on_channel_selection_change, names='value')

        # Containers for channel, mask, and annotation controls
        self.channel_controls_box = VBox(
            layout=Layout(
                width='100%',
                overflow_y='auto',
                gap='6px',
                padding='4px 0'
            )
        )
        self.mask_controls_box = VBox(
            layout=Layout(
                width='100%',
                overflow_y='auto',
                gap='6px',
                padding='4px 0'
            )
        )

        self.no_channels_label = HTML(value='<i>No channels selected.</i>')
        self.no_masks_label = HTML(value='<i>No masks available for this FOV.</i>')
        self.no_annotations_label = HTML(value='<i>No annotations detected.</i>')
        self.empty_controls_placeholder = HTML(value='<i>No viewer controls are available.</i>')

        self.control_sections = Accordion(
            children=(self.channel_controls_box,),
            layout=Layout(width='100%', max_height='640px')
        )
        self.control_sections.set_title(0, 'Channels')

        self.annotation_editor_host = VBox(
            layout=Layout(width='100%', padding='8px 0 0 0')
        )

        # Initialize markerset widgets
        self.marker_set_dropdown = Dropdown(
            options=[],  # Will be populated with marker set names
            value=None,
            description='Marker Set:',
            disabled=False
        )
        self.marker_set_name_input = Text(
            value='',
            placeholder='Enter marker set name',
            description='Set Name:',
            disabled=False
        )

        self.load_marker_set_button = Button(
            description='Load Marker Set',
            disabled=False,
            button_style='',
            tooltip='Load the selected marker set',
            icon='folder-open'
        )
        self.load_marker_set_button.on_click(viewer.load_marker_set)

        self.save_marker_set_button = Button(
            description='Save Marker Set',
            disabled=False,
            button_style='',
            tooltip='Save the current configuration as a new marker set',
            icon='save'
        )
        self.save_marker_set_button.on_click(viewer.save_marker_set)

        self.update_marker_set_button = Button(
            description='Update Marker Set',
            disabled=False,
            button_style='',
            tooltip='Update the selected marker set with current settings',
            icon='refresh'
        )
        self.update_marker_set_button.on_click(viewer.update_marker_set)

        self.delete_marker_set_button = Button(
            description='Delete Marker Set',
            disabled=False,
            button_style='danger',
            tooltip='Delete the selected marker set',
            icon='trash'
        )
        self.delete_marker_set_button.on_click(viewer.delete_marker_set)

        self.delete_confirmation_checkbox = Checkbox(
            value=False,
            description='Confirm Deletion',
            disabled=False
        )
        # Initialize enable downsample checkbox
        self.pixel_size_inttext = IntText(
            value=390,
            description='Pixel Size (nm):',
            disabled=False,
            style={'description_width': 'auto'}
        )
        self.pixel_size_inttext.observe(viewer.on_pixel_size_change, names='value')
        self.enable_downsample_checkbox = Checkbox(
            value=True,
            description='Downsample',
            disabled=False,
            style={'description_width': 'auto'}
        )

        self.x_key = Text(
            value='centroid-1',
            description='X key:',
            disabled=False
        )
        self.x_key.observe(viewer.on_key_change, names='value')

        self.y_key = Text(
            value='centroid-0',
            description='Y key:',
            disabled=False
        )
        self.y_key.observe(viewer.on_key_change, names='value')

        self.label_key = Text(
            value='label',
            description='Label key:',
            disabled=False
        )
        self.label_key.observe(viewer.on_key_change, names='value')
        
        self.mask_key = Text(
            value='whole_cell',
            description='Mask key:',
            disabled=False
        )
        self.mask_key.observe(viewer.on_key_change, names='value')

        self.fov_key = Text(
            value='fov',
            description='Fov key:',
            disabled=False
        )
        self.fov_key.observe(viewer.on_key_change, names='value')

        identifiers_VBox = VBox([
            self.x_key,
            self.y_key,
            self.mask_key,
            self.label_key,
            self.fov_key,
            ])
        
        main_viewer_VBox = VBox([
            self.pixel_size_inttext,
            self.enable_downsample_checkbox
            ])

        self.advanced_settings_tabs = Tab(
            children=[identifiers_VBox, main_viewer_VBox],
            titles=('Data mapping','Advanced Settings')
        )

        self.advanced_settings_accordion = Accordion(
            children=[self.advanced_settings_tabs],
            titles=('Advanced Settings',),
            selected_index=None
        )

        self.status_bar = Image(
            value=viewer._status_image["processing"],
            format='gif',
            width=225,
            height=30,
        )

        # Initialize dictionaries to hold controls for each channel and mask
        self.color_controls = {}
        self.contrast_min_controls = {}
        self.contrast_max_controls = {}
        self.mask_color_controls = {}
        self.mask_display_controls = {}

        # Annotation controls (initially disabled until annotations are detected)
        self.annotation_controls_header = HTML(value='<b>Annotations</b>')
        self.annotation_controls_box = VBox(
            layout=Layout(
                width='100%',
                overflow_y='auto',
                gap='6px',
                padding='4px 0'
            )
        )

        self.annotation_display_checkbox = Checkbox(
            value=False,
            description='Show annotation',
            disabled=True,
            layout=Layout(width='auto'),
            style={'description_width': 'auto'}
        )
        self.annotation_display_checkbox.observe(viewer.on_annotation_toggle, names='value')

        self.annotation_selector = Dropdown(
            options=[],
            value=None,
            description='Annotation:',
            disabled=True,
            layout=Layout(width='100%'),
            style={'description_width': 'auto'}
        )
        self.annotation_selector.observe(viewer.on_annotation_selection_change, names='value')

        self.annotation_overlay_mode = ToggleButtons(
            options=[
                ('Mask outlines', 'mask'),
                ('Annotation fill', 'annotation'),
                ('Fill + mask edges', 'combined')
            ],
            value='combined',
            description='Overlay mode:',
            disabled=True,
            layout=Layout(width='100%'),
            style={'description_width': 'auto'}
        )
        self.annotation_overlay_mode.observe(viewer.on_annotation_overlay_mode_change, names='value')

        self.annotation_alpha_slider = FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            description='Fill alpha:',
            disabled=True,
            continuous_update=False,
            layout=Layout(width='100%'),
            style={'description_width': 'auto'}
        )
        self.annotation_alpha_slider.observe(viewer.on_annotation_alpha_change, names='value')

        self.annotation_label_mode = Dropdown(
            options=[('Class IDs', 'id'), ('Labels', 'label')],
            value='id',
            description='Legend labels:',
            disabled=True,
            layout=Layout(width='100%'),
            style={'description_width': 'auto'}
        )
        self.annotation_label_mode.observe(viewer.on_annotation_label_mode_change, names='value')

        self.annotation_edit_button = Button(
            description='Edit palette…',
            disabled=True,
            tooltip='Adjust colors for annotation classes',
            icon='paint-brush',
            layout=Layout(width='auto')
        )
        self.annotation_edit_button.on_click(viewer.on_edit_annotation_palette)
