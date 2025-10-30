# ui_components.py
import ipywidgets as widgets
from IPython.display import display
from types import SimpleNamespace

_widget_attr = getattr(widgets, "Widget", None)
if not isinstance(_widget_attr, type):  # pragma: no cover - ensure baseline widget API exists
    class _FallbackWidget:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            children = kwargs.get("children")
            if children is None and args:
                children = args[0]
            self.children = tuple(children or ())
            self.value = kwargs.get("value")
            self.options = list(kwargs.get("options", ()))
            self.description = kwargs.get("description", "")
            self.disabled = kwargs.get("disabled", False)
            self.layout = kwargs.get("layout")

        def observe(self, *_args, **_kwargs):
            return None

        def on_click(self, *_args, **_kwargs):  # pragma: no cover - parity with ipywidgets API
            return None

        def set_title(self, *_args, **_kwargs):  # pragma: no cover - accordion/tab helper
            return None

    setattr(widgets, "Widget", _FallbackWidget)

Widget = getattr(widgets, "Widget")

Layout = getattr(widgets, "Layout", None)
if Layout is None:  # pragma: no cover - fallback for stripped stubs
    try:
        from ipywidgets.widgets.widget_layout import Layout  # type: ignore
    except Exception:  # pragma: no cover - minimal stand-in when submodule missing
        class Layout:  # type: ignore[override]
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

setattr(widgets, "Layout", Layout)


class _FallbackOutput(Widget):  # pragma: no cover - minimal output widget stand-in
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = ()

    def clear_output(self, *_args, **_kwargs):
        self.outputs = ()


def _ensure_widget_class(name, *, default=None):
    value = getattr(widgets, name, None)
    if isinstance(value, type):
        return value
    resolved = default or Widget
    if not isinstance(resolved, type):  # pragma: no cover - defensive guard
        resolved = Widget
    setattr(widgets, name, resolved)
    return resolved


SelectMultiple = _ensure_widget_class("SelectMultiple")
FloatSlider = _ensure_widget_class("FloatSlider")
Dropdown = _ensure_widget_class("Dropdown")
VBox = _ensure_widget_class("VBox")
Output = _ensure_widget_class("Output", default=_FallbackOutput)
Checkbox = _ensure_widget_class("Checkbox")
IntText = _ensure_widget_class("IntText")
Text = _ensure_widget_class("Text")
Button = _ensure_widget_class("Button")
HBox = _ensure_widget_class("HBox")
Accordion = _ensure_widget_class("Accordion")
Tab = _ensure_widget_class("Tab")
HTML = _ensure_widget_class("HTML")
ToggleButtons = _ensure_widget_class("ToggleButtons")
IntSlider = _ensure_widget_class("IntSlider")
TagsInput = _ensure_widget_class("TagsInput")
Image = _ensure_widget_class("Image")

from .plugin.chart import ChartDisplay  # type: ignore[import-error]
from .plugin.cell_gallery import CellGalleryDisplay  # type: ignore[import-error]
from .plugin.heatmap import HeatmapDisplay  # type: ignore[import-error]
from .plugin.plugin_base import PluginBase  # type: ignore[import-error]
from .annotation_display import AnnotationDisplay  # type: ignore[import-error]


def build_wide_plugin_pane(control=None, content=None):
    """Compose the standard left/right layout used in the footer tabs."""
    if control is None and content is None:
        return VBox(children=(), layout=Layout(width='100%'))

    if control is None:
        return VBox(children=(content,), layout=Layout(width='100%', overflow_y='auto'))

    if content is None:
        return VBox(children=(control,), layout=Layout(width='100%', overflow_y='auto'))

    control_box = VBox(children=(control,), layout=Layout(width='6in', flex='0 0 6in', overflow_y='auto', gap='8px'))
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


def _ensure_bottom_namespace(viewer):
    bottom_ns = getattr(viewer, 'BottomPlots', None)
    if bottom_ns is None:
        bottom_ns = SimpleNamespace()
        viewer.BottomPlots = bottom_ns
    return bottom_ns


def _ensure_pane_cache(viewer):
    pane_cache = getattr(viewer, '_wide_plugin_panes', None)
    if not isinstance(pane_cache, dict):
        pane_cache = {}
        viewer._wide_plugin_panes = pane_cache
    return pane_cache


def _cleanup_bottom_state(bottom_ns, pane_cache, active_attrs):
    for attr in {name for name in vars(bottom_ns) if name not in active_attrs}:
        delattr(bottom_ns, attr)
    for attr in [name for name in pane_cache if name not in active_attrs]:
        pane_cache.pop(attr, None)


def _clear_wide_panel(viewer):
    viewer.wide_plugin_tab.children = ()
    viewer.wide_plugin_tab.selected_index = None
    viewer.wide_plugin_panel.layout.display = 'none'


def _resolve_wide_pane(entry, pane_cache):
    plugin = entry['plugin']
    attr = entry['attr']
    token_getter = getattr(plugin, 'wide_panel_cache_token', None)
    token = token_getter() if callable(token_getter) else None
    cached_pane, cached_token = pane_cache.get(attr, (None, None))
    if cached_pane is None or cached_token != token:
        pane = build_wide_plugin_pane(entry.get('control'), entry.get('content'))
        pane_cache[attr] = (pane, token)
        return pane, False
    return cached_pane, True


def _apply_wide_panel(viewer, entries, tab_children):
    previous_selection = getattr(viewer.wide_plugin_tab, 'selected_index', None)
    viewer.wide_plugin_tab.children = tuple(tab_children)
    for idx, entry in enumerate(entries):
        viewer.wide_plugin_tab.set_title(idx, entry['title'])
    if viewer.wide_plugin_tab.children:
        if previous_selection is not None and previous_selection < len(viewer.wide_plugin_tab.children):
            viewer.wide_plugin_tab.selected_index = previous_selection
        elif getattr(viewer.wide_plugin_tab, 'selected_index', None) is None:
            viewer.wide_plugin_tab.selected_index = 0
    else:
        viewer.wide_plugin_tab.selected_index = None
    viewer.wide_plugin_panel.layout.display = ''


def _restore_heatmap(viewer):
    sideplots = getattr(viewer, 'SidePlots', None)
    heatmap_plugin = getattr(sideplots, 'heatmap_output', None) if sideplots else None
    if isinstance(heatmap_plugin, PluginBase):
        restore_footer = getattr(heatmap_plugin, 'restore_footer_canvas', None)
        if callable(restore_footer):
            restore_footer()
        restore_vertical = getattr(heatmap_plugin, 'restore_vertical_canvas', None)
        if callable(restore_vertical):
            restore_vertical()


def _trigger_cached_plugin_refresh(plugin):
    refresh = getattr(plugin, 'request_cached_wide_panel_refresh', None)
    if not callable(refresh):
        return
    try:
        refresh()
    except Exception:  # pragma: no cover - best effort logging only
        logger = getattr(plugin, 'logger', None)
        log_exception = getattr(logger, 'exception', None) if logger else None
        if callable(log_exception):
            log_exception('Failed to refresh cached wide panel for %s', getattr(plugin, 'displayed_name', plugin))


def update_wide_plugin_panel(viewer, ordering=None):
    debug_enabled = getattr(viewer, '_debug', False)
    if not hasattr(viewer, 'wide_plugin_tab') or not hasattr(viewer, 'wide_plugin_panel'):
        if debug_enabled:
            print("[wide-plugin] viewer missing wide_plugin_tab or wide_plugin_panel; skipping update")
        return

    entries = collect_wide_plugin_entries(viewer)
    if debug_enabled:
        print(f"[wide-plugin] update requested with {len(entries)} entries")
    bottom_ns = _ensure_bottom_namespace(viewer)
    pane_cache = _ensure_pane_cache(viewer)

    active_attrs = {entry['attr'] for entry in entries}
    _cleanup_bottom_state(bottom_ns, pane_cache, active_attrs)

    if not entries:
        if debug_enabled:
            print('[wide-plugin] no entries found; hiding footer panel')
        _clear_wide_panel(viewer)
        return

    key = ordering or (lambda entry: entry['title'].lower())
    entries = sorted(entries, key=key)

    tab_children = []
    refresh_queue = []
    for entry in entries:
        pane, reused = _resolve_wide_pane(entry, pane_cache)
        tab_children.append(pane)
        setattr(bottom_ns, entry['attr'], entry['plugin'])
        if reused:
            refresh_queue.append(entry['plugin'])
            if debug_enabled:
                print(f"[wide-plugin] reused pane for {entry['attr']}")
        elif debug_enabled:
            print(f"[wide-plugin] rebuilt pane for {entry['attr']}")

    _apply_wide_panel(viewer, entries, tab_children)
    _restore_heatmap(viewer)
    for plugin in refresh_queue:
        _trigger_cached_plugin_refresh(plugin)
    if debug_enabled and refresh_queue:
        refreshed = ', '.join(getattr(plugin, 'displayed_name', repr(plugin)) for plugin in refresh_queue)
        print(f'[wide-plugin] triggered cached refresh for: {refreshed}')


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
        if hasattr(viewer, "setup_attr_observers"):
            viewer.setup_attr_observers()

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
    ], layout=Layout(width='100%', gap='8px'))

    top_part_widgets = VBox([
        viewer.ui_component.cache_size_input,
        viewer.ui_component.image_selector,
        viewer.ui_component.channel_selector_text,
        viewer.ui_component.channel_selector,
        marker_set_widgets,
        control_panel_stack,
        VBox([viewer.ui_component.advanced_settings_accordion])
    ], layout=Layout(width='100%', overflow_x='hidden'))

    left_panel_children = [top_part_widgets]
    left_panel_children.append(
        HBox([HTML(value="Status:"), viewer.ui_component.status_bar])
    )

    left_panel = VBox(
        left_panel_children,
    layout=Layout(width='350px', overflow_x='hidden', overflow_y='auto', gap='10px')
    )

    ui = HBox([
        left_panel,
        viewer.image_output,
        viewer.side_plot  # Add the chart output widget to the right
    ])

    root = VBox([ui, viewer.wide_plugin_panel], layout=Layout(width='100%', max_width='100%', gap='12px'))

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

        self.mask_outline_thickness_slider = IntSlider(
            value=1,
            min=1,
            max=10,
            step=1,
            description='Mask outline px:',
            layout=Layout(width='100%'),
            style={'description_width': '150px'},
            continuous_update=False,
        )

        self.no_channels_label = HTML(value='<i>No channels selected.</i>')
        self.no_masks_label = HTML(value='<i>No masks available for this FOV.</i>')
        self.no_annotations_label = HTML(value='<i>No annotations detected.</i>')
        self.empty_controls_placeholder = HTML(value='<i>No viewer controls are available.</i>')

        self.control_sections = Accordion(
            children=(self.channel_controls_box,),
            layout=Layout(width='98%', max_height='640px')
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
