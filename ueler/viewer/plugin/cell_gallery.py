"""Cell gallery plugin implementation for the packaged namespace."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Button, HBox, IntSlider, IntText, Layout, Output, VBox

from ueler.image_utils import color_one_image, estimate_color_range, process_single_crop
from ueler.viewer.decorators import update_status_bar
from ueler.viewer.observable import Observable

from .plugin_base import PluginBase


class CellGalleryDisplay(PluginBase):
    def __init__(self, main_viewer, width, height):
        super().__init__(main_viewer, width, height)
        self.SidePlots_id = "cell_gallery_output"
        self.displayed_name = "Gallery"
        self.main_viewer = main_viewer
        self.width = width
        self.height = height

        self.data = Data()

        self.ui_component = UiComponent()

        self.ui_component.cutout_size_slider = IntSlider(
            value=150,
            min=50,
            max=500,
            step=50,
            description="Cutout Size (pixel):",
            continuous_update=False,
        )

        self.plot_output = Output(layout=Layout(max_height="300px", overflow_y="auto"))

        self.ui_component.max_displayed_cells_inttext = IntText(
            value=20,
            description="Max Displayed Cells:",
            style={"description_width": "auto"},
        )

        self.ui_component.refresh_button = Button(
            description="Refresh",
            disabled=False,
            button_style="",
            tooltip="Refresh the gallery",
            icon="refresh",
        )

        self.ui_component.refresh_button.on_click(self.refresh_gallery)

        self.initiate_ui()

        self.data.selected_cells.add_observer(self.on_selected_indices_change)

    def initiate_ui(self):
        controls = VBox(
            [
                HBox([
                    self.ui_component.cutout_size_slider,
                    self.ui_component.refresh_button,
                ]),
                HBox([self.ui_component.max_displayed_cells_inttext]),
            ]
        )
        ui = VBox([controls, VBox([self.plot_output], layout=Layout(height="400px"))])
        self.ui = ui

    @update_status_bar
    def plot_gellery(self):  # noqa: N802 - keep legacy method name
        crop_width = round(self.ui_component.cutout_size_slider.value)
        canvas, n, _, _, ind = create_gallery(
            self.main_viewer.cell_table,
            self.data.selected_cells.value,
            self.main_viewer.marker2display(),
            self.main_viewer.base_folder,
            crop_width=crop_width,
            color_range=self.main_viewer.get_color_range(),
            mask_suffix=self.main_viewer.mask_key,
            mask_source=self.main_viewer.masks_folder,
            max_displayed_cells=self.ui_component.max_displayed_cells_inttext.value,
            fov_key=self.main_viewer.fov_key,
            x_key=self.main_viewer.x_key,
            y_key=self.main_viewer.y_key,
            image_viewer=self.main_viewer,
        )
        self.data.displayed_cells = ind
        m = 5

        self.data.n = n
        self.data.m = m

        with self.plot_output:
            self.plot_output.clear_output()

            min_rows = 3
            displayed_rows = max(n, min_rows)
            fig_height = displayed_rows

            fig, ax = plt.subplots(figsize=(self.width * 0.9, fig_height))
            ax.imshow(canvas)
            ax.axis("off")

            fig.canvas.header_visible = False
            fig.tight_layout()
            plt.show(fig)

            self.data.fig = fig
            self.data.ax = ax
            self.mask_id_annotation = self._create_annotation()

            def on_click(event):
                if event.inaxes == ax:
                    spacing = 5
                    col = int(event.xdata // (crop_width + spacing))
                    row = int(event.ydata // (crop_width + spacing))

                    if row < n and col < m:
                        clicked_index = row * m + col
                        df = self.main_viewer.cell_table
                        fov_key = self.main_viewer.fov_key
                        x_key = self.main_viewer.x_key
                        y_key = self.main_viewer.y_key
                        clicked_cell_ind = ind[clicked_index]
                        x = df.iloc[clicked_cell_ind][x_key]
                        y = df.iloc[clicked_cell_ind][y_key]
                        fov = df.iloc[clicked_cell_ind][fov_key]
                        self.main_viewer.ui_component.image_selector.value = fov
                        self.main_viewer.image_display.ax.set_xlim(
                            x - crop_width / 2, x + crop_width / 2
                        )
                        self.main_viewer.image_display.ax.set_ylim(
                            y - crop_width / 2, y + crop_width / 2
                        )

            self.data.fig.canvas.mpl_connect("button_press_event", on_click)
            self.data.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

            self.hover_timer = self.data.fig.canvas.new_timer(interval=300)
            self.hover_timer.single_shot = True
            self.hover_timer.add_callback(self.process_hover_event)
            self.hover_timer.start()

    def on_mouse_move(self, event):
        if event.inaxes != self.data.ax:
            return

        if self.hover_timer is not None:
            self.hover_timer.stop()

        self.last_hover_event = event

        self.hover_timer = self.data.fig.canvas.new_timer(interval=300)
        self.hover_timer.single_shot = True
        self.hover_timer.add_callback(self.process_hover_event)
        self.hover_timer.start()

    def process_hover_event(self):
        event = self.last_hover_event
        spacing = 5
        crop_width = round(self.ui_component.cutout_size_slider.value)
        col = int(event.xdata // (crop_width + spacing))
        row = int(event.ydata // (crop_width + spacing))

        m = self.data.m
        n = self.data.n
        ind = self.data.displayed_cells

        if row < n and col < m:
            clicked_index = row * m + col
            df = self.main_viewer.cell_table
            fov_key = self.main_viewer.fov_key
            label_key = self.main_viewer.label_key
            clicked_cell_ind = ind[clicked_index]
            fov = df.iloc[clicked_cell_ind][fov_key]
            label = df.iloc[clicked_cell_ind][label_key]

            tooltip_text = f"{fov}: {label}"
            self.mask_id_annotation.xy = (event.xdata, event.ydata)
            self.mask_id_annotation.set_text(tooltip_text)
            self.mask_id_annotation.set_visible(True)
            self.data.fig.canvas.draw_idle()

    def _create_annotation(self):
        return self.data.ax.annotate(
            "",
            xy=(0, 0),
            xycoords="data",
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=12,
            color="yellow",
            bbox={"boxstyle": "round,pad=0.3", "fc": "black", "ec": "yellow", "lw": 1},
            arrowprops={"arrowstyle": "->"},
            visible=False,
        )

    def refresh_gallery(self, _button):
        self.plot_gellery()

    def on_selected_indices_change(self, _selected_indices):
        print("on_selected_indices_change() triggered")
        self.plot_gellery()

    def set_selected_cells(self, row_indices):
        self.data.selected_cells.value = row_indices


class UiComponent:
    def __init__(self):
        """Placeholder container for UI widget references."""

        pass


def create_gallery(
    df,
    ind,
    marker2display,
    file_source,
    crop_width=100,
    color_range=None,
    mask_suffix=None,
    mask_source="",
    max_displayed_cells=np.inf,
    fov_key="fov",
    x_key="X",
    y_key="Y",
    image_viewer=None,
):
    images = []
    if isinstance(ind, set):
        ind = list(ind)

    if len(ind) > max_displayed_cells:
        print(
            f"{len(ind)} cells selected, which exceeds the maximum number of cells to display."
        )
        seed = int(abs(np.sum(ind))) % (2**32)
        rng = np.random.default_rng(seed=seed)
        sample_size = int(max_displayed_cells)
        ind = rng.choice(list(ind), sample_size, replace=False).tolist()
        print(f"Randomly sampled {sample_size} cells for display.")

    for idx in ind:
        fov = df[fov_key][idx]
        crop_position = np.round([df.iloc[idx][y_key], df.iloc[idx][x_key]]).astype(int)

        img = process_single_crop(
            fov,
            marker2display,
            crop_position,
            crop_width,
            file_source,
            mask_suffix=mask_suffix,
            mask_source=mask_source,
            mask_ID=df.label[idx],
            image_viewer=image_viewer,
        )
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        images.append(img)

    m = 5
    n = (len(images) + m - 1) // m
    spacing = 5

    img_height, img_width, num_channels = images[0].shape
    canvas_height = n * img_height + (n - 1) * spacing
    canvas_width = m * img_width + (m - 1) * spacing

    canvas = np.zeros((canvas_height, canvas_width, num_channels), dtype=images[0].dtype)

    for idx, img in enumerate(images):
        row = idx // m
        col = idx % m
        start_row = row * (img_height + spacing)
        start_col = col * (img_width + spacing)
        canvas[start_row:start_row + img_height, start_col:start_col + img_width, :] = img

    if mask_suffix is not None:
        canvas_center_mask = canvas[:, :, -1]
        canvas = canvas[:, :, :-1]
        canvas_mask = canvas[:, :, -1]
        canvas = canvas[:, :, :-1]

    if color_range is None:
        color_range = estimate_color_range(canvas, marker2display.keys(), pt=95, est_lb=False)

    canvas = color_one_image(canvas, marker2display, color_range)

    if mask_suffix is not None:
        mask_indices = canvas_mask > 0
        center_mask_indices = canvas_center_mask > 0

        canvas[mask_indices] = [125, 125, 125]
        canvas[center_mask_indices] = [255, 255, 255]

    return canvas, n, m, color_range, ind


class Data:
    def __init__(self):
        """Hold mutable state shared across gallery callbacks."""

        self.selected_cells = Observable([])


__all__ = [
    "CellGalleryDisplay",
    "UiComponent",
    "create_gallery",
    "Data",
]
