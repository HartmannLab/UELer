import os
import json
from ipywidgets import Widget


class PluginBase:
    def __init__(self, viewer, width, height):
        self.viewer = viewer
        self.width = width
        self.height = height
        self.SidePlots_id = ""
        self.displayed_name = ""
        self.initialized = False
        # Common plugin behavior
    
    def initiate_ui(self):
        pass    

    def wide_panel_layout(self):
        """
        Optional hook for footer-wide layout support.

        Returns:
            dict | None: A mapping with optional keys ``control`` and ``content``
            pointing to ipywidget instances for the left control column and the
            main content area respectively. Returning ``None`` keeps the plugin
            rendered exclusively in the SidePlots accordion.
        """
        return None

    def wide_panel_cache_token(self):
        """Provide an identifier used to decide when a cached footer pane must be rebuilt."""
        return None

    def after_all_plugins_loaded(self):
        widget_states_path = os.path.join(self.main_viewer.base_folder, ".UELer", f'{self.displayed_name}_widget_states.json')
        self.load_widget_states(widget_states_path)

    def on_mv_update_display(self):
        pass

    def on_cell_table_change(self):
        pass

    def on_fov_change(self):
        pass

    def on_widget_value_change(self, change):
        """Callback function to handle widget value changes."""
        if self.initialized:
            widget_states_path = os.path.join(self.main_viewer.base_folder, ".UELer", f'{self.displayed_name}_widget_states.json')
            self.save_widget_states(widget_states_path)
    
    def setup_widget_observers(self):
        """Set up observers on all widgets in ui_component."""

        def observe_widget(widget):
            if isinstance(widget, Widget):
                if hasattr(widget, 'observe') and hasattr(widget, 'value'):
                    widget.observe(self.on_widget_value_change, names='value')
                # If the widget is a container, recurse into its children
                if hasattr(widget, 'children'):
                    for child in widget.children:
                        observe_widget(child)

        ui_attrs = vars(self.ui_component)
        for attr_name, attr_value in ui_attrs.items():
            observe_widget(attr_value)

    def save_widget_states(self, file_path):
        """Save the current state of all widgets to a JSON file."""
        state = {}

        # Iterate over attributes of self.ui_component
        ui_attrs = vars(self.ui_component)
        for attr_name, attr_value in ui_attrs.items():
            if isinstance(attr_value, Widget):
                # Save the value of the widget if it has a 'value' attribute
                if hasattr(attr_value, 'value'):
                    state[attr_name] = attr_value.value
                else:
                    # Skip widgets without a 'value' attribute
                    pass
            elif isinstance(attr_value, dict):
                # Assume it's a dictionary of widgets
                state[attr_name] = {}
                for key, widget in attr_value.items():
                    if isinstance(widget, Widget):
                        if hasattr(widget, 'value'):
                            state[attr_name][key] = widget.value
                        else:
                            # Skip widgets without a 'value' attribute
                            pass
                    else:
                        # Handle non-widget items if any
                        state[attr_name][key] = widget
            else:
                # Handle other attribute types if necessary
                pass

        # Save additional viewer attributes
        # if applicable

        # Custom serializer to handle non-JSON serializable types
        def default_serializer(o):
            try:
                return o.item()  # For numpy types
            except Exception:
                raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4, default=default_serializer)
        
        # Save the state to a JSON file
        if self.main_viewer._debug:
            print(f"{self.displayed_name} widget states saved to {file_path}")

    def load_widget_states(self, file_path):
        """Load the state of all widgets from a JSON file."""
        if not os.path.exists(file_path):
            print(f"No widget states file found at {file_path}")
            return

        with open(file_path, 'r') as f:
            state = json.load(f)

        # Iterate over the saved state and restore widget values
        for attr_name, value in state.items():
            if hasattr(self.ui_component, attr_name):
                # Get the attribute but don't modify it directly
                widget = getattr(self.ui_component, attr_name)
                if self.main_viewer._debug:
                    print(f"Restoring state for widget {attr_name}")
                try:
                    if isinstance(widget, Widget):
                        if hasattr(widget, 'value'):
                            if hasattr(widget, 'options'):
                                if value in widget.options:
                                    # Set the value through the widget instance
                                    setattr(widget, 'value', value)
                                    if self.main_viewer._debug:
                                        print(f" Value {value} set for widget {attr_name}")
                                else:
                                    if self.main_viewer._debug:
                                        print(f" Value {value} not found in widget options")
                            else:
                                # Set the value through the widget instance
                                setattr(widget, 'value', value)
                                if self.main_viewer._debug:
                                    print(f" Value {value} set for widget {attr_name}")
                        else:
                            if self.main_viewer._debug:
                                print(f" Widget {attr_name} has no 'value' attribute")
                    elif isinstance(widget, dict):
                        for key, widget_value in value.items():
                            if key in widget:
                                sub_widget = widget[key]
                                if isinstance(sub_widget, Widget) and hasattr(sub_widget, 'value'):
                                    setattr(sub_widget, 'value', widget_value)
                                    if self.main_viewer._debug:
                                        print(f" Value {widget_value} set for widget {key} in {attr_name}")
                                else:
                                    if self.main_viewer._debug:
                                        print(f" Widget {key} in {attr_name} is not a valid widget")
                    else:
                        if self.main_viewer._debug:
                            print(f"Attribute {attr_name} is not a valid widget")
                except Exception as e:
                    if self.main_viewer._debug:
                        print(f"Error setting value for widget {attr_name}: {e}")
            else:
                if self.main_viewer._debug:
                    print(f"Attribute {attr_name} not found in ui_component")
                
        
        print(f"\n{self.displayed_name} widget states loaded from {file_path}")