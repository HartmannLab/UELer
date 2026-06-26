"""Base class shared by viewer plugin implementations."""

from __future__ import annotations

import json
import logging
import os
from ipywidgets import Widget

_logger = logging.getLogger(__name__)


class PluginBase:
    """Common plugin behaviour reused across viewer extensions."""

    def __init__(self, viewer, width, height):
        self.viewer = viewer
        self.width = width
        self.height = height
        self.SidePlots_id = ""  # NOSONAR - legacy public attribute name
        self.displayed_name = ""
        self.initialized = False
        # Common plugin behavior

    def initiate_ui(self):
        """Hook for subclasses to allocate their widgets."""

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
        widget_states_path = os.path.join(self.main_viewer.base_folder, ".UELer", f"{self.displayed_name}_widget_states.json")
        self.load_widget_states(widget_states_path)

    def on_mv_update_display(self):
        """Hook invoked when the main viewer refreshes its display."""

        pass

    def on_cell_table_change(self):
        """Hook invoked when the cell table changes."""

        pass

    def on_fov_change(self):
        """Hook invoked when the active field-of-view changes."""

        pass

    def on_map_mode_activate(self):
        """Hook invoked when map mode is activated or the active map changes."""

        pass

    def on_map_mode_deactivate(self):
        """Hook invoked when map mode is deactivated."""

        pass

    def on_no_image_toggle(self):
        """Hook invoked when the viewer toggles image-layer rendering."""

        pass

    def on_widget_value_change(self, change):  # NOSONAR - legacy signature
        """Callback function to handle widget value changes."""
        if self.initialized:
            widget_states_path = os.path.join(self.main_viewer.base_folder, ".UELer", f"{self.displayed_name}_widget_states.json")
            self.save_widget_states(widget_states_path)

    def setup_widget_observers(self):
        """Set up observers on all widgets in ui_component."""

        def observe_widget(widget):
            if isinstance(widget, Widget):
                if hasattr(widget, 'observe') and hasattr(widget, 'value'):
                    widget.observe(self.on_widget_value_change, names='value')
                if hasattr(widget, 'children'):
                    for child in widget.children:
                        observe_widget(child)

        ui_attrs = vars(self.ui_component)
        for attr_name, attr_value in ui_attrs.items():
            observe_widget(attr_value)

    def save_widget_states(self, file_path):  # NOSONAR - legacy complexity
        """Save the current state of all widgets to a JSON file."""
        state = {}

        ui_attrs = vars(self.ui_component)
        for attr_name, attr_value in ui_attrs.items():
            if isinstance(attr_value, Widget):
                if hasattr(attr_value, 'value'):
                    state[attr_name] = attr_value.value
            elif isinstance(attr_value, dict):
                state[attr_name] = {}
                for key, widget in attr_value.items():
                    if isinstance(widget, Widget):
                        if hasattr(widget, 'value'):
                            state[attr_name][key] = widget.value
                        else:
                            state[attr_name][key] = widget
                    else:
                        state[attr_name][key] = widget

        def default_serializer(o):
            try:
                return o.item()
            except Exception as exc:  # pragma: no cover - defensive guard
                raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable') from exc

        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4, default=default_serializer)

        _logger.debug("%s widget states saved to %s", self.displayed_name, file_path)

    def load_widget_states(self, file_path):  # NOSONAR - legacy complexity
        """Load the state of all widgets from a JSON file."""
        if not os.path.exists(file_path):
            _logger.debug("No widget states file found at %s", file_path)
            return

        with open(file_path, 'r') as f:
            state = json.load(f)

        for attr_name, value in state.items():
            if hasattr(self.ui_component, attr_name):
                widget = getattr(self.ui_component, attr_name)
                _logger.debug("Restoring state for widget %s", attr_name)
                try:
                    if isinstance(widget, Widget):
                        if hasattr(widget, 'value'):
                            if hasattr(widget, 'options'):
                                if value in widget.options:
                                    setattr(widget, 'value', value)
                                    _logger.debug(" Value %s set for widget %s", value, attr_name)
                                else:
                                    _logger.debug(" Value %s not found in widget options", value)
                            else:
                                setattr(widget, 'value', value)
                                _logger.debug(" Value %s set for widget %s", value, attr_name)
                        else:
                            _logger.debug(" Widget %s has no 'value' attribute", attr_name)
                    elif isinstance(widget, dict):
                        for key, widget_value in value.items():
                            if key in widget:
                                sub_widget = widget[key]
                                if isinstance(sub_widget, Widget) and hasattr(sub_widget, 'value'):
                                    setattr(sub_widget, 'value', widget_value)
                                    _logger.debug(" Value %s set for widget %s in %s", widget_value, key, attr_name)
                                else:
                                    _logger.debug(" Widget %s in %s is not a valid widget", key, attr_name)
                    else:
                        _logger.debug("Attribute %s is not a valid widget", attr_name)
                except Exception as e:  # pragma: no cover - defensive guard
                    _logger.debug("Error setting value for widget %s: %s", attr_name, e)
            else:
                _logger.debug("Attribute %s not found in ui_component", attr_name)

        _logger.debug("%s widget states loaded from %s", self.displayed_name, file_path)


__all__ = ["PluginBase"]
