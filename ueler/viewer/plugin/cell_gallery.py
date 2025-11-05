"""Cell gallery plugin implementation for the packaged namespace."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Button, Checkbox, ColorPicker, HBox, IntSlider, IntText, Layout, Output, VBox
from matplotlib.colors import to_rgb
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ueler.rendering import (
    AnnotationRenderSettings,
    ChannelRenderSettings,
    MaskRenderSettings,
    OverlaySnapshot,
    get_cell_color,
    render_crop_to_array,
)
from ueler.viewer.decorators import update_status_bar
from ueler.viewer.observable import Observable

from .plugin_base import PluginBase


GRID_COLUMNS = 5
GRID_SPACING = 5
DEFAULT_CUTOUT_SIZE = 150
MIN_CUTOUT_SIZE = 50
MAX_CUTOUT_SIZE = 500
CUTOUT_STEP = 50
DEFAULT_MAX_DISPLAYED_CELLS = 20
MAX_DOWNSAMPLE_FACTOR = 8
DEFAULT_MASK_OUTLINE_THICKNESS = 1
MAX_MASK_OUTLINE_THICKNESS = 8
HOVER_DELAY_MS = 300


class CellGalleryDisplay(PluginBase):
    def __init__(self, main_viewer, width, height):
        super().__init__(main_viewer, width, height)
        self.SidePlots_id = "cell_gallery_output"
        self.displayed_name = "Gallery"
        self.main_viewer = main_viewer
        self.width = width
        self.height = height

        self.data = Data()
        self.hover_timer = None
        self.last_hover_event = None
        self._skip_next_fov_refresh = False

        self.ui_component = UiComponent()

        self.ui_component.cutout_size_slider = IntSlider(
            value=DEFAULT_CUTOUT_SIZE,
            min=MIN_CUTOUT_SIZE,
            max=MAX_CUTOUT_SIZE,
            step=CUTOUT_STEP,
            description="Cutout Size (px):",
            continuous_update=False,
        )

        self.ui_component.max_displayed_cells_inttext = IntText(
            value=DEFAULT_MAX_DISPLAYED_CELLS,
            description="Max Displayed Cells:",
            style={"description_width": "auto"},
        )

        self.ui_component.downsample_slider = IntSlider(
            value=1,
            min=1,
            max=MAX_DOWNSAMPLE_FACTOR,
            step=1,
            description="Downsample",
            continuous_update=False,
        )

        thickness_default = int(
            getattr(main_viewer, "mask_outline_thickness", DEFAULT_MASK_OUTLINE_THICKNESS)
        )
        thickness_default = max(1, min(thickness_default, MAX_MASK_OUTLINE_THICKNESS))

        self.ui_component.mask_outline_thickness_slider = IntSlider(
            value=thickness_default,
            min=1,
            max=MAX_MASK_OUTLINE_THICKNESS,
            step=1,
            description="Outline px",
            continuous_update=False,
        )

        default_outline_colour = getattr(main_viewer, "selection_outline_color", "#FFFFFF")
        if not isinstance(default_outline_colour, str):
            default_outline_colour = "#FFFFFF"

        self.ui_component.mask_outline_picker = ColorPicker(
            value=default_outline_colour,
            description="Mask colour",
            style={"description_width": "auto"},
        )

        self.ui_component.use_uniform_color_checkbox = Checkbox(
            value=False,
            description="Use uniform color",
            tooltip="When enabled, all masks use the same color. When disabled, painted mask colors from mask painter are shown.",
            style={"description_width": "auto"},
        )

        self.plot_output = Output(layout=Layout(max_height="300px", overflow_y="auto"))

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
                HBox(
                    [
                        self.ui_component.cutout_size_slider,
                        self.ui_component.downsample_slider,
                        self.ui_component.refresh_button,
                    ]
                ),
                HBox([self.ui_component.max_displayed_cells_inttext]),
                HBox(
                    [
                        self.ui_component.mask_outline_picker,
                        self.ui_component.mask_outline_thickness_slider,
                    ]
                ),
                HBox([self.ui_component.use_uniform_color_checkbox]),
            ]
        )
        self.ui = VBox([controls, VBox([self.plot_output], layout=Layout(height="400px"))])

    def _collect_ui_values(self) -> _UiValues:
        crop_width = int(self.ui_component.cutout_size_slider.value)

        try:
            max_cells = float(self.ui_component.max_displayed_cells_inttext.value)
        except (TypeError, ValueError):
            max_cells = np.inf
        if max_cells <= 0:
            max_cells = np.inf

        downsample_factor = max(1, int(self.ui_component.downsample_slider.value))
        highlight_colour = self.ui_component.mask_outline_picker.value
        outline_thickness = int(self.ui_component.mask_outline_thickness_slider.value)
        use_uniform_color = bool(self.ui_component.use_uniform_color_checkbox.value)

        return _UiValues(
            crop_width=crop_width,
            max_cells=max_cells,
            downsample_factor=downsample_factor,
            highlight_colour=highlight_colour,
            outline_thickness=outline_thickness,
            use_uniform_color=use_uniform_color,
        )

    def _show_empty_message(self, message: str = "No cells selected."):
        with self.plot_output:
            self.plot_output.clear_output()
            print(message)

    def _update_tile_metadata(self, canvas: np.ndarray, rows: int, columns: int):
        self.data.n = rows
        self.data.m = columns if columns else GRID_COLUMNS

        if rows > 0:
            tile_height = (canvas.shape[0] - max(rows - 1, 0) * GRID_SPACING) // rows
        else:
            tile_height = canvas.shape[0]
        if self.data.m > 0:
            tile_width = (canvas.shape[1] - max(self.data.m - 1, 0) * GRID_SPACING) // self.data.m
        else:
            tile_width = canvas.shape[1]

        self.data.tile_height = max(1, tile_height)
        self.data.tile_width = max(1, tile_width)

    def _draw_gallery(
        self,
        canvas: np.ndarray,
        rows: int,
        displayed_indices: Sequence[int],
        crop_width: int,
    ):
        with self.plot_output:
            self.plot_output.clear_output()

            min_rows = 3
            fig_height = max(rows, min_rows)

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
                if event.inaxes != ax or event.xdata is None or event.ydata is None:
                    return

                col = int(event.xdata // (self.data.tile_width + GRID_SPACING))
                row = int(event.ydata // (self.data.tile_height + GRID_SPACING))

                if row < rows and col < GRID_COLUMNS:
                    clicked_index = row * GRID_COLUMNS + col
                    if clicked_index >= len(displayed_indices):
                        return
                    df = self.main_viewer.cell_table
                    fov_key = self.main_viewer.fov_key
                    x_key = self.main_viewer.x_key
                    y_key = self.main_viewer.y_key
                    cell_index = displayed_indices[clicked_index]
                    x = df.iloc[cell_index][x_key]
                    y = df.iloc[cell_index][y_key]
                    fov = df.iloc[cell_index][fov_key]
                    image_selector = getattr(
                        self.main_viewer.ui_component, "image_selector", None
                    )
                    if image_selector is None:
                        return

                    prior_fov = getattr(image_selector, "value", None)
                    should_skip_refresh = prior_fov != fov
                    self._skip_next_fov_refresh = should_skip_refresh

                    try:
                        image_selector.value = fov
                        self.main_viewer.image_display.ax.set_xlim(
                            x - crop_width / 2, x + crop_width / 2
                        )
                        self.main_viewer.image_display.ax.set_ylim(
                            y - crop_width / 2, y + crop_width / 2
                        )
                    finally:
                        if not should_skip_refresh:
                            self._skip_next_fov_refresh = False

            fig.canvas.mpl_connect("button_press_event", on_click)
            fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

            self.hover_timer = fig.canvas.new_timer(interval=HOVER_DELAY_MS)
            self.hover_timer.single_shot = True
            self.hover_timer.add_callback(self.process_hover_event)
            self.hover_timer.start()

    @update_status_bar
    def plot_gellery(self):  # noqa: N802 - preserve legacy method name
        selected_indices = list(self.data.selected_cells.value or [])
        ui_values = self._collect_ui_values()

        overlay_snapshot = self._capture_overlay_snapshot()

        canvas, n, m, _, displayed_indices = create_gallery(
            self.main_viewer.cell_table,
            selected_indices,
            viewer=self.main_viewer,
            crop_width=ui_values.crop_width,
            max_displayed_cells=ui_values.max_cells,
            overlay_snapshot=overlay_snapshot,
            downsample_factor=ui_values.downsample_factor,
            label_key=self.main_viewer.label_key,
            highlight_outline_color=ui_values.highlight_colour,
            mask_outline_thickness=ui_values.outline_thickness,
            use_uniform_color=ui_values.use_uniform_color,
        )

        if not displayed_indices:
            self._show_empty_message()
            return

        self.data.displayed_cells = displayed_indices
        self._update_tile_metadata(canvas, n, m)
        self._draw_gallery(canvas, n, displayed_indices, ui_values.crop_width)

    def on_mouse_move(self, event):
        if self.data.ax is None or event.inaxes != self.data.ax:
            return

        if self.hover_timer is not None:
            self.hover_timer.stop()

        self.last_hover_event = event

        self.hover_timer = self.data.fig.canvas.new_timer(interval=HOVER_DELAY_MS)
        self.hover_timer.single_shot = True
        self.hover_timer.add_callback(self.process_hover_event)
        self.hover_timer.start()

    def process_hover_event(self):
        event = self.last_hover_event
        if event is None or event.xdata is None or event.ydata is None:
            return

        tile_width = self.data.tile_width
        tile_height = self.data.tile_height
        if tile_width <= 0 or tile_height <= 0:
            return

        col = int(event.xdata // (tile_width + GRID_SPACING))
        row = int(event.ydata // (tile_height + GRID_SPACING))

        ind = self.data.displayed_cells
        if row < self.data.n and col < GRID_COLUMNS:
            clicked_index = row * GRID_COLUMNS + col
            if clicked_index >= len(ind):
                return
            df = self.main_viewer.cell_table
            fov_key = self.main_viewer.fov_key
            label_key = self.main_viewer.label_key
            cell_index = ind[clicked_index]
            fov = df.iloc[cell_index][fov_key]
            label = df.iloc[cell_index][label_key]

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

    def _capture_overlay_snapshot(self) -> Optional[OverlaySnapshot]:
        capture = getattr(self.main_viewer, "capture_overlay_snapshot", None)
        if callable(capture):
            try:
                return capture()
            except Exception as exc:  # pragma: no cover - non-critical
                print(f"Failed to capture overlay snapshot: {exc}")
        return None

    def refresh_gallery(self, _button=None):
        self.plot_gellery()

    def on_selected_indices_change(self, _selected_indices):
        self.plot_gellery()

    def set_selected_cells(self, row_indices):
        self.data.selected_cells.value = list(row_indices)

    def on_fov_change(self) -> None:  # type: ignore[override]
        side_plots = getattr(self.main_viewer, "SidePlots", None)
        chart_plugin = getattr(side_plots, "chart_output", None) if side_plots else None
        if chart_plugin is not None and hasattr(chart_plugin, "single_point_click_state"):
            state = getattr(chart_plugin, "single_point_click_state", 0)
            chart_plugin.single_point_click_state = 0
            if state == 1:
                return

        if self._skip_next_fov_refresh:
            self._skip_next_fov_refresh = False
            return

        selected = getattr(self.data.selected_cells, "value", None)
        if selected:
            self.plot_gellery()

    def on_viewer_mask_outline_change(self, thickness: int) -> None:
        """Handle mask outline thickness changes from the main viewer.
        
        This callback is invoked when the main viewer's mask outline thickness
        slider is adjusted, ensuring the cell gallery stays synchronized.
        """
        try:
            thickness = max(1, min(thickness, MAX_MASK_OUTLINE_THICKNESS))
        except (TypeError, ValueError):
            thickness = DEFAULT_MASK_OUTLINE_THICKNESS
        
        # Update the cell gallery's thickness slider to match
        if self.ui_component.mask_outline_thickness_slider is not None:
            self.ui_component.mask_outline_thickness_slider.value = thickness
        
        # Refresh the gallery if there are selected cells
        selected = getattr(self.data.selected_cells, "value", None)
        if selected:
            self.plot_gellery()

    def on_mask_painter_change(self) -> None:
        """Handle mask painter color application changes.
        
        This callback is invoked when the mask painter applies new colors to masks,
        ensuring the cell gallery reflects the updated mask colors immediately.
        """
        # Refresh the gallery if there are selected cells
        selected = getattr(self.data.selected_cells, "value", None)
        if selected:
            self.plot_gellery()


class UiComponent:
    def __init__(self):
        """Placeholder container for UI widget references."""
        self.cutout_size_slider: IntSlider | None = None
        self.max_displayed_cells_inttext: IntText | None = None
        self.downsample_slider: IntSlider | None = None
        self.mask_outline_picker: ColorPicker | None = None
        self.mask_outline_thickness_slider: IntSlider | None = None
        self.use_uniform_color_checkbox: Checkbox | None = None
        self.refresh_button: Button | None = None


class Data:
    def __init__(self):
        """Hold mutable state shared across gallery callbacks."""

        self.selected_cells = Observable([])
        self.displayed_cells: List[int] = []
        self.n = 0
        self.m = 0
        self.tile_width = 0
        self.tile_height = 0
        self.fig = None
        self.ax = None


@dataclass
class _RenderContext:
    viewer: any
    selected_channels: Tuple[str, ...]
    channel_settings: Mapping[str, ChannelRenderSettings]
    crop_width: int
    downsample_factor: int
    overlay_snapshot: Optional[OverlaySnapshot]
    overlay_cache: Dict[
        Tuple[str, int],
        Tuple[Optional[AnnotationRenderSettings], Tuple[MaskRenderSettings, ...]],
    ]
    label_key: str
    highlight_rgb: Tuple[float, float, float]
    outline_thickness: int
    use_uniform_color: bool
    mask_name: Optional[str]
    fov_key: str
    x_key: str
    y_key: str


@dataclass
class _UiValues:
    crop_width: int
    max_cells: float
    downsample_factor: int
    highlight_colour: str
    outline_thickness: int
    use_uniform_color: bool


def _resolve_mask_array(viewer, fov_name: str, mask_name: Optional[str]) -> Optional[np.ndarray]:
    if viewer is None or not mask_name:
        return None

    mask_cache = getattr(viewer, "mask_cache", {})
    masks_for_fov = mask_cache.get(fov_name)
    if not masks_for_fov:
        return None

    mask_array = masks_for_fov.get(mask_name)
    if mask_array is None:
        return None

    try:
        return mask_array.compute()
    except AttributeError:
        return np.asarray(mask_array)


def _limit_selection(indices, max_displayed: float) -> List[int]:
    if isinstance(indices, set):
        indices = list(indices)

    if max_displayed is None or np.isinf(max_displayed) or len(indices) <= max_displayed:
        return list(indices)

    print(f"{len(indices)} cells selected, which exceeds the maximum number of cells to display.")
    safe_indices = np.asarray(indices, dtype=int)
    seed = int(abs(int(safe_indices.sum()))) % (2**32)
    rng = np.random.default_rng(seed=seed)
    sample_size = max(1, int(max_displayed))
    sampled = rng.choice(safe_indices, sample_size, replace=False)
    print(f"Randomly sampled {sample_size} cells for display.")
    return sorted(int(i) for i in sampled.tolist())


def _build_channel_settings(
    marker_to_display: Mapping[str, Tuple[float, float, float]],
    color_range: Mapping[str, Sequence[float]],
) -> Dict[str, ChannelRenderSettings]:
    settings: Dict[str, ChannelRenderSettings] = {}
    for channel, colour in marker_to_display.items():
        range_entry = color_range.get(channel)
        if range_entry is None or len(range_entry) < 2:
            raise KeyError(f"Channel '{channel}' missing contrast range settings.")
        colour_rgb = tuple(np.clip(np.asarray(colour, dtype=float), 0.0, 1.0).tolist())
        settings[channel] = ChannelRenderSettings(
            color=colour_rgb,
            contrast_min=float(range_entry[0]),
            contrast_max=float(range_entry[1]),
        )
    return settings


def _resolve_overlay_settings(
    viewer,
    fov_name: str,
    downsample_factor: int,
    snapshot: Optional[OverlaySnapshot],
    cache: Dict[Tuple[str, int], Tuple[Optional[AnnotationRenderSettings], Tuple[MaskRenderSettings, ...]]],
) -> Tuple[Optional[AnnotationRenderSettings], Tuple[MaskRenderSettings, ...]]:
    if snapshot is None:
        return None, ()

    cache_key = (fov_name, downsample_factor)
    if cache_key not in cache:
        cache[cache_key] = viewer.build_overlay_settings_from_snapshot(
            fov_name,
            downsample_factor,
            snapshot,
        )
    return cache[cache_key]


def _compose_canvas(images: Sequence[np.ndarray], columns: int, spacing: int) -> Tuple[np.ndarray, int, int]:
    if not images:
        return np.zeros((0, 0, 3), dtype=np.float32), 0, 0

    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    channels = [img.shape[2] if img.ndim == 3 else 1 for img in images]

    if len(set(channels)) != 1:
        raise ValueError("All images must have the same number of channels to compose a gallery.")

    max_height = max(heights)
    max_width = max(widths)
    num_channels = channels[0]

    rows = (len(images) + columns - 1) // columns
    canvas_height = rows * max_height + max(rows - 1, 0) * spacing
    canvas_width = columns * max_width + max(columns - 1, 0) * spacing

    canvas = np.zeros((canvas_height, canvas_width, num_channels), dtype=np.float32)

    for index, img in enumerate(images):
        if img.ndim == 2:
            img_to_place = img[:, :, np.newaxis]
        else:
            img_to_place = img

        img_height, img_width = img_to_place.shape[:2]

        row = index // columns
        col = index % columns

        slot_row = row * (max_height + spacing)
        slot_col = col * (max_width + spacing)

        # Center the tile within its slot when the rendered size is smaller than the grid slot.
        row_offset = slot_row + max((max_height - img_height) // 2, 0)
        col_offset = slot_col + max((max_width - img_width) // 2, 0)

        if img_to_place.shape[2] != num_channels:
            raise ValueError("Gallery tiles must share the same channel depth.")

        canvas[
            row_offset : row_offset + img_height,
            col_offset : col_offset + img_width,
            : img_to_place.shape[2],
        ] = img_to_place

    return canvas, rows, columns


def _extract_mask_id(record, label_key: str):
    if hasattr(record, "get"):
        mask_id = record.get(label_key)
    else:
        mask_id = record[label_key]

    if mask_id is None or (isinstance(mask_id, float) and np.isnan(mask_id)):
        return None
    return mask_id


def _render_tile_for_index(df, index: int, context: _RenderContext):
    record = df.iloc[index]
    fov = record[context.fov_key]
    center_xy = (float(record[context.x_key]), float(record[context.y_key]))
    mask_id = _extract_mask_id(record, context.label_key)

    viewer = context.viewer
    viewer.load_fov(fov, context.selected_channels)
    channel_arrays = viewer.image_cache[fov]

    annotation_settings, mask_settings = _resolve_overlay_settings(
        viewer,
        fov,
        context.downsample_factor,
        context.overlay_snapshot,
        context.overlay_cache,
    )

    masks = list(mask_settings)
    
    # Determine which color to use for this cell's mask outline
    cell_color = None
    if context.use_uniform_color:
        # Use the uniform color from the color picker
        cell_color = context.highlight_rgb
    else:
        # Try to get the painted color from the centralized color registry
        painted_color = get_cell_color(fov, mask_id)
        if painted_color:
            try:
                from matplotlib.colors import to_rgb
                cell_color = to_rgb(painted_color)
            except (ValueError, TypeError):
                pass
    
    # Apply mask outline if we have a valid color and mask
    if cell_color is not None and context.mask_name and mask_id is not None:
        mask_array = _resolve_mask_array(viewer, fov, context.mask_name)
        if mask_array is not None and np.any(mask_array == mask_id):
            masks.append(
                MaskRenderSettings(
                    array=mask_array == mask_id,
                    color=cell_color,
                    mode="outline",
                    outline_thickness=context.outline_thickness,
                    downsample_factor=context.downsample_factor,
                )
            )

    tile = render_crop_to_array(
        fov,
        channel_arrays,
        context.selected_channels,
        context.channel_settings,
        center_xy=center_xy,
        size_px=context.crop_width,
        downsample_factor=context.downsample_factor,
        annotation=annotation_settings,
        masks=tuple(masks),
    )
    return np.clip(tile, 0.0, 1.0)


def create_gallery(
    df,
    selected_indices,
    *,
    viewer,
    crop_width: int = 100,
    max_displayed_cells: float = np.inf,
    overlay_snapshot: Optional[OverlaySnapshot] = None,
    downsample_factor: int = 1,
    label_key: str = "label",
    highlight_outline_color: Union[str, Tuple[float, float, float]] = "#FFFFFF",
    mask_outline_thickness: int = 1,
    use_uniform_color: bool = False,
):
    indices = _limit_selection(selected_indices, max_displayed_cells)

    if not indices:
        empty = np.zeros((crop_width, crop_width, 3), dtype=np.float32)
        return empty, 0, GRID_COLUMNS, {}, []

    if viewer is None:
        raise ValueError("viewer must be provided to render gallery tiles.")

    marker_to_display = viewer.marker2display()
    color_range = dict(viewer.get_color_range())
    if not marker_to_display:
        empty = np.zeros((crop_width, crop_width, 3), dtype=np.float32)
        return empty, 0, GRID_COLUMNS, color_range, []

    channel_settings = _build_channel_settings(marker_to_display, color_range)
    selected_channels = tuple(channel_settings.keys())

    highlight_rgb = to_rgb(highlight_outline_color)
    outline_thickness = max(1, int(mask_outline_thickness))
    mask_name = getattr(viewer, "mask_key", None)

    fov_key = getattr(viewer, "fov_key", "fov")
    x_key = getattr(viewer, "x_key", "X")
    y_key = getattr(viewer, "y_key", "Y")

    overlay_cache: Dict[
        Tuple[str, int],
        Tuple[Optional[AnnotationRenderSettings], Tuple[MaskRenderSettings, ...]],
    ] = {}
    context = _RenderContext(
        viewer=viewer,
        selected_channels=selected_channels,
        channel_settings=channel_settings,
        crop_width=crop_width,
        downsample_factor=downsample_factor,
        overlay_snapshot=overlay_snapshot,
        overlay_cache=overlay_cache,
        label_key=label_key,
        highlight_rgb=highlight_rgb,
        outline_thickness=outline_thickness,
        use_uniform_color=use_uniform_color,
        mask_name=mask_name,
        fov_key=fov_key,
        x_key=x_key,
        y_key=y_key,
    )
    images: List[np.ndarray] = []
    displayed_indices: List[int] = []

    for idx in indices:
        tile = _render_tile_for_index(df, idx, context)
        if tile is None:
            continue
        images.append(tile)
        displayed_indices.append(idx)

    if not images:
        empty = np.zeros((crop_width, crop_width, 3), dtype=np.float32)
        return empty, 0, GRID_COLUMNS, color_range, []

    canvas, rows, cols = _compose_canvas(images, GRID_COLUMNS, GRID_SPACING)
    return canvas, rows, cols, color_range, displayed_indices


__all__ = [
    "CellGalleryDisplay",
    "UiComponent",
    "create_gallery",
    "Data",
]
