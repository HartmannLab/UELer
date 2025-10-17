"""Go To plugin implementation under the packaged namespace."""

from __future__ import annotations

from collections import OrderedDict

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import (
    Button,
    Checkbox,
    Dropdown,
    FloatSlider,
    HBox,
    IntSlider,
    IntText,
    Layout,
    Output,
    SelectMultiple,
    VBox,
)
from matplotlib.text import Annotation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from ueler.image_utils import color_one_image, estimate_color_range, process_single_crop
from ueler.viewer.decorators import update_status_bar
from ueler.viewer.observable import Observable

from .plugin_base import PluginBase


class goTo(PluginBase):  # NOSONAR - legacy class name kept for backwards compatibility
    def __init__(self, main_viewer, width, height):
        super().__init__(main_viewer, width, height)
        self.SidePlots_id = "go_to_output"
        self.displayed_name = "Go to"
        self.main_viewer = main_viewer
        self.width = width
        self.height = height

        self.ui_component = UiComponent()

        narrow_layout = Layout(width="150px")

        self.plot_output = Output(layout=Layout(max_height="300px", overflow_y="auto"))

        self.ui_component.fov_dropdown = Dropdown(
            options=self.main_viewer.ui_component.image_selector.options,
            value=self.main_viewer.ui_component.image_selector.value,
            description="FOV:",
            style={"description_width": "auto"},
            layout=narrow_layout,
        )

        self.ui_component.cell_ID_text = IntText(
            value=0,
            description="Cell ID:",
            style={"description_width": "auto"},
            layout=narrow_layout,
        )

        self.ui_component.zoom_width = IntSlider(
            value=150,
            min=50,
            max=500,
            step=10,
            description="Width (pixel):",
            continuous_update=False,
            style={"description_width": "auto"},
        )

        self.ui_component.go_to_button = Button(
            description="Go to",
            disabled=False,
            button_style="",
            tooltip="Go to the specified cell or location",
            icon="map-marker",
        )

        self.ui_component.go_to_button.on_click(self.go_to_on_click)

        self.initiate_ui()

    def go_to_on_click(self, _button):
        df = self.main_viewer.cell_table
        label_key = self.main_viewer.label_key
        fov_key = self.main_viewer.fov_key
        x_key = self.main_viewer.x_key
        y_key = self.main_viewer.y_key

        l_specified_cell = (
            (df[fov_key] == self.ui_component.fov_dropdown.value)
            & (df[label_key] == self.ui_component.cell_ID_text.value)
        )

        crop_width = self.ui_component.zoom_width.value

        x_vals = df.loc[l_specified_cell, x_key]
        y_vals = df.loc[l_specified_cell, y_key]

        if x_vals.empty or y_vals.empty:
            print("Cell not found.")
            return

        x = x_vals.iloc[0]
        y = y_vals.iloc[0]
        fov = self.ui_component.fov_dropdown.value
        self.main_viewer.ui_component.image_selector.value = fov
        self.main_viewer.image_display.ax.set_xlim(x - crop_width / 2, x + crop_width / 2)
        self.main_viewer.image_display.ax.set_ylim(y - crop_width / 2, y + crop_width / 2)

        self.main_viewer.image_display.ax.figure.canvas.draw()
        self.main_viewer.image_display.update_display()

    def initiate_ui(self):
        controls = VBox(
            [
                HBox(
                    [
                        self.ui_component.zoom_width,
                        self.ui_component.fov_dropdown,
                        self.ui_component.cell_ID_text,
                    ]
                ),
                HBox([self.ui_component.go_to_button]),
            ]
        )
        ui = VBox([controls, VBox([self.plot_output], layout=Layout(max_height="400px"))])
        self.ui = ui


class UiComponent:
    def __init__(self):
        """Placeholder container for widget references."""

        pass


__all__ = ["goTo", "UiComponent"]
