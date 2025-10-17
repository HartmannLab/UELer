"""Placeholder Batch Export plugin exposed via the packaged namespace."""

from __future__ import annotations

from ipywidgets import HTML, Layout, VBox

from .plugin_base import PluginBase

PLACEHOLDER_MESSAGE = "Batch export will be available soon."


class RunFlowsom(PluginBase):
    """Minimal stub so the Batch export tab communicates its pending status."""

    def __init__(self, main_viewer, width, height):
        super().__init__(main_viewer, width, height)
        self.SidePlots_id = "batch_export_output"
        self.displayed_name = "Batch export"
        self.main_viewer = main_viewer
        self.width = width
        self.height = height

        self._message = HTML(
            value=f"<b>{PLACEHOLDER_MESSAGE}</b>",
            layout=Layout(width="100%", padding="12px"),
        )
        self.ui = VBox([self._message], layout=Layout(width="100%"))

        self.initialized = True

    def initiate_ui(self):
        """Kept for PluginBase compatibility; no controls to assemble yet."""

        return None

    def setup_widget_observers(self):
        """No observers required until the feature ships."""

        return None


__all__ = ["RunFlowsom", "PLACEHOLDER_MESSAGE"]
