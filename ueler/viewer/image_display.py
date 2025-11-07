# viewer/image_display.py

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
from matplotlib.text import Annotation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.colors import to_rgb
from ueler.image_utils import (
    calculate_downsample_factor,
    generate_edges,
    get_axis_limits_with_padding,
)
from ueler.rendering.engine import scale_outline_thickness, thicken_outline
from matplotlib.patches import Polygon
from matplotlib.widgets import RectangleSelector
# from skimage.measure import find_contours
from skimage.segmentation import find_boundaries
import cv2
import math
from matplotlib.backend_bases import MouseButton
from ueler.viewer.decorators import update_status_bar
from .tooltip_utils import format_tooltip_value


class ImageDisplay:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.hover_timer = None
        self.last_hover_event = None
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(0, self.width)
        # self.ax.set_ylim(self.height, 0)  # Invert y-axis for image orientation
        self.img_display = self.ax.imshow(
            np.zeros((self.height, self.width, 3), dtype=np.float32),
            extent=(0, self.width, 0, self.height),
            origin='upper'
        )
        self.ax.axis("off")
        self.scalebar = None
        self.mask_id_annotation = self._create_annotation()
        self._setup_event_connections()
        self.selected_masks_label = set() # Set to track selected mask labels
        self.fig.canvas.header_visible = False
        self.fig.tight_layout()
        self.selected_mask_label = set()  # For storing mask IDs to display
        self._roi_selector = None
        self._roi_callback = None

    def _materialize_combined(self):
        """Return a copy of the combined image as a NumPy array, if available."""
        data = getattr(self, "combined", None)
        if data is None:
            return None

        if hasattr(data, "compute"):
            try:
                return data.copy().compute()
            except AttributeError:
                # Some dask arrays may not implement copy(); fall back to compute first
                return np.array(data.compute(), copy=True)

        return np.array(data, copy=True)

    def update_scale_bar(
        self,
        spec,
        *,
        color: str = "white",
        font_size: float = 12.0,
        data_pixel_ratio: float = 1.0,
    ) -> None:
        """Update the anchored scale bar artist for the current axes."""

        if hasattr(self, "scalebar") and self.scalebar is not None:
            try:
                self.scalebar.remove()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            self.scalebar = None

        if spec is None:
            self.fig.canvas.draw_idle()
            return

        try:
            from ueler.viewer.scale_bar import add_scale_bar

            ratio = 1.0
            try:
                ratio = float(data_pixel_ratio)
            except (TypeError, ValueError):
                ratio = 1.0
            if not np.isfinite(ratio) or ratio <= 0.0:
                ratio = 1.0

            adjusted_spec = spec
            if not math.isclose(ratio, 1.0):
                try:
                    adjusted_spec = replace(spec, pixel_length=spec.pixel_length * ratio)
                except Exception:
                    from ueler.viewer.scale_bar import ScaleBarSpec  # local import avoids cycle in tests

                    adjusted_spec = ScaleBarSpec(
                        pixel_length=spec.pixel_length * ratio,
                        physical_length_um=spec.physical_length_um,
                        label=spec.label,
                    )

            self.scalebar = add_scale_bar(
                self.ax,
                adjusted_spec,
                color=color,
                font_size=font_size,
            )
        except Exception:  # pragma: no cover - fallback when Matplotlib back-end is mocked
            self.scalebar = None
        self.fig.canvas.draw_idle()

    def _create_annotation(self):
        return self.ax.annotate(
            "",
            xy=(0, 0),
            xycoords="data",
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=12,
            color='yellow',
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="yellow", lw=1),
            arrowprops=dict(arrowstyle="->"),
            visible=False
        )

    def _setup_event_connections(self):
        self.fig.canvas.callbacks.connect('draw_event', self.on_draw)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)

    @update_status_bar
    def on_draw(self, event):
        current_center_x = (self.ax.get_xlim()[0] + self.ax.get_xlim()[1]) / 2
        current_center_y = (self.ax.get_ylim()[0] + self.ax.get_ylim()[1]) / 2

        if hasattr(self, "prev_center_x") and hasattr(self, "prev_center_y"):
            if math.isclose(current_center_x, self.prev_center_x) and math.isclose(current_center_y, self.prev_center_y):
                return

        self.prev_center_x = current_center_x
        self.prev_center_y = current_center_y
        
        """Adjust the downsample factor based on the zoom level."""
        if self.main_viewer.initialized:
            # Get the range width of the x and y axis
            range_width_x = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
            range_width_y = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]

            new_downsample_factor = calculate_downsample_factor(
                np.abs(range_width_x), np.abs(range_width_y), not self.main_viewer.ui_component.enable_downsample_checkbox.value
            )
        else:
            new_downsample_factor = 8

        if new_downsample_factor != self.main_viewer.current_downsample_factor:
            self.main_viewer.current_downsample_factor = new_downsample_factor
        
        if self.main_viewer.initialized:
            self.main_viewer.update_display(self.main_viewer.current_downsample_factor)

    def on_mouse_move(self, event):
        if event.inaxes != self.ax:
            self.mask_id_annotation.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            self.mask_id_annotation.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        # Cancel any existing timer
        if self.hover_timer is not None:
            self.hover_timer.stop()

        # Store the event
        self.last_hover_event = event

        # Start a new timer
        self.hover_timer = self.fig.canvas.new_timer(interval=300)  # 100 milliseconds
        self.hover_timer.single_shot = True
        self.hover_timer.add_callback(self.process_hover_event)
        self.hover_timer.start()

    def process_hover_event(self):
        event = self.last_hover_event
        x, y = event.xdata, event.ydata

        # Convert to integer indices
        ix = int(x)
        iy = int(y)

        # Adjust for downsample factor and image extent
        downsample_factor = self.main_viewer.current_downsample_factor
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        # Correct for inverted axes
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if xmin > xmax:
            xmin, xmax = xmax, xmin

        xmin = int(max(xmin, 0))
        xmax = int(min(xmax, self.width))
        ymin = int(max(ymin, 0))
        ymax = int(min(ymax, self.height))

        ix_ds = (ix - xmin) // downsample_factor
        iy_ds = (iy - ymin) // downsample_factor

        mask_id = 0
        found_mask = False

        # Use a more efficient search method for masks
        for mask_name, label_mask in self.main_viewer.current_label_masks.items():
            if 0 <= iy_ds < label_mask.shape[0] and 0 <= ix_ds < label_mask.shape[1]:
                mask_id = label_mask[iy_ds, ix_ds].compute()
                if mask_id != 0:
                    # Cache the filtered DataFrame
                    if not hasattr(self, '_cached_cell_data') or self._cached_cell_data_key != (mask_name, mask_id):
                        self._cached_cell_data = self.main_viewer.cell_table[
                            (self.main_viewer.cell_table['fov'] == self.main_viewer.ui_component.image_selector.value) &
                            (self.main_viewer.cell_table['label'] == mask_id)
                        ]
                        self._cached_cell_data_key = (mask_name, mask_id)

                    cell_data = self._cached_cell_data

                    # Construct tooltip text
                    tooltip_text = f"{mask_name} ID: {mask_id}"
                    if not cell_data.empty:
                        for channel in self.main_viewer.ui_component.channel_selector.value:
                            if channel in cell_data.columns:
                                value = cell_data[channel].iloc[0]
                                tooltip_text += f"\n{channel}: {format_tooltip_value(value)}"
                        for label in getattr(self.main_viewer, 'selected_tooltip_labels', []):
                            if label in cell_data.columns:
                                value = cell_data[label].iloc[0]
                                tooltip_text += f"\n{label}: {format_tooltip_value(value)}"

                    # Update annotation
                    self.mask_id_annotation.xy = (x, y)
                    self.mask_id_annotation.set_text(tooltip_text)
                    self.mask_id_annotation.set_visible(True)
                    self.fig.canvas.draw_idle()
                    found_mask = True
                    break  # Stop after finding the first mask

        if not found_mask:
            self.mask_id_annotation.set_visible(False)
            self.fig.canvas.draw_idle()

    def on_mouse_click(self, event):
        """Handle mouse click events to select/unselect masks."""
        if event.inaxes != self.ax:
            print("Mouse click outside axes")
            return

        # Check if any navigation tool is active
        if self.fig.canvas.toolbar is not None and self.fig.canvas.toolbar.mode != '':
            # A navigation tool (e.g., zoom or pan) is active; ignore the click
            print("Navigation tool active")
            return

        # Get mouse event coordinates
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            print("Mouse click outside data area")
            return

        # Convert to integer indices
        ix = int(x)
        iy = int(y)

        # Ensure indices are within bounds
        ix = min(max(ix, 0), self.width - 1)
        iy = min(max(iy, 0), self.height - 1)

        if event.button == MouseButton.LEFT: # Ctrl + left click
            if event.key != 'control':
                self.clear_patches()
            for mask_name, label_mask_full in self.main_viewer.full_resolution_label_masks.items():
                if 0 <= iy < self.height and 0 <= ix < self.width:
                    mask_id = label_mask_full[iy, ix].compute()
                    if mask_id != 0:
                        key = (mask_name, mask_id)
                        if key in self.selected_masks_label:
                            # Deselect the mask
                            self.selected_masks_label.discard(key)
                        else:
                            # Store the full-resolution binary mask of the selected mask_id
                            # mask_binary = (label_mask_full == mask_id)
                            self.selected_masks_label.add(key)
                        self.update_patches()
                        break  # Stop after processing the first mask found
        elif event.button == 3:  # Right click
            # Clear all selections
            self.clear_patches()

    def clear_patches(self):
        self.selected_masks_label.clear()
        self.update_patches()

    def update_patches(self, do_not_reset=False):
        """Update the display of selected mask patches (contour lines)."""
        if getattr(self, '_in_on_draw', False):
            # Skip updating patches if already handling a draw event
            return

        # Adjust for downsample factor
        downsample_factor = self.main_viewer.current_downsample_factor

        xmin, xmax, ymin, ymax, xmin_ds, xmax_ds, ymin_ds, ymax_ds = get_axis_limits_with_padding(self.main_viewer, downsample_factor)
        
        # Loop through selected masks
        selected_mask_visible_ds = None
        for mask_name, label_mask_full in self.main_viewer.full_resolution_label_masks.items():
            mask_visible_ds = label_mask_full[ymin:ymax:downsample_factor, xmin:xmax:downsample_factor].compute()
            selected_mask_visible_ds = np.zeros_like(mask_visible_ds)
            for mask_name, mask_id in self.selected_masks_label:
                print(f"Processing mask {mask_name} ID {mask_id}")
                selected_mask_visible_ds[mask_visible_ds == mask_id] = mask_id

        # If selected_mask_full_visible is defined
        if selected_mask_visible_ds is not None:
            if self.selected_masks_label:
                mask_binary_ds = selected_mask_visible_ds.astype(np.uint8)

                outline_thickness = scale_outline_thickness(
                    getattr(self.main_viewer, "mask_outline_thickness", 1),
                    downsample_factor,
                )

                edge_mask = find_boundaries(mask_binary_ds, mode="inner")
                if outline_thickness > 1:
                    edge_mask = thicken_outline(edge_mask, outline_thickness - 1)
                
                if do_not_reset:
                    combined = self.img_display.get_array().copy()
                else:
                    combined = self._materialize_combined()
                    if combined is None:
                        return

                # Highlight selected cells in white
                combined[edge_mask] = [1, 1, 1]
                self.img_display.set_data(combined)

                if self.main_viewer._debug:
                    print("Redrawing canvas")
                self.fig.canvas.draw_idle()
            else:
                # No cells selected - just refresh to show painted colors if painter is enabled
                if not do_not_reset:
                    combined = self._materialize_combined()
                    if combined is not None:
                        self.img_display.set_data(combined)
                        self.fig.canvas.draw_idle()

    def set_mask_ids(self, mask_name, mask_ids):
        """
        Set the mask IDs to display and update the display.

        Parameters:
            mask_name (str): The name of the mask to select IDs from.
            mask_ids (list or int): The mask ID(s) to display. Can be a single int or a list of ints.
        """
        # if mask_ids is empty

        if not mask_ids:
            self.selected_masks_label.clear()
            self.update_patches()
            return

        # Ensure mask_ids is a set of integers
        if isinstance(mask_ids, int):
            mask_ids = {mask_ids}
        else:
            mask_ids = set(mask_ids)

        # Clear previous selections
        self.selected_masks_label.clear()

        # Get the full-resolution label mask
        label_mask_full = self.main_viewer.full_resolution_label_masks.get(mask_name)
        if label_mask_full is None:
            print(f"Mask '{mask_name}' not found.")
            return

        # Select the specified masks
        unique_label_full = np.unique(label_mask_full).compute()
        for mask_id in mask_ids:
            if mask_id not in unique_label_full:
                continue
            key = (mask_name, mask_id)
            self.selected_masks_label.add(key)

        self.update_patches()
    
    def set_mask_colors_current_fov(self, mask_name, mask_ids, color=None, cummulative = False):
        cdf = self.main_viewer.current_downsample_factor

        xmin, xmax, ymin, ymax, xmin_ds, xmax_ds, ymin_ds, ymax_ds = get_axis_limits_with_padding(self.main_viewer, cdf)
        
        fov_name = self.main_viewer.ui_component.image_selector.value
        # xmin, xmax, ymin, ymax, xmin_ds, xmax_ds, ymin_ds, ymax_ds = get_axis_limits_with_padding(self, cdf)
        combined = self._materialize_combined()
        if combined is None:
            return
        color_rgb = np.array(to_rgb(color), dtype=np.float32)
        # Overlay masks
        selected_masks = [mask_name for mask_name, cb in self.main_viewer.ui_component.mask_display_controls.items() if cb.value]
        if self.main_viewer._debug:
            print(f"color_rgb: {color_rgb}")
        if selected_masks:
            if self.main_viewer.ui_component.image_selector.value in self.main_viewer.mask_cache:
                if mask_name in selected_masks:
                    if mask_name in self.main_viewer.mask_cache[self.main_viewer.ui_component.image_selector.value]:
                        # If selected_mask_full_visible is defined
                        mask_label_ds = self.main_viewer.label_masks_cache[fov_name][mask_name][cdf]
                        mask_label_ds = mask_label_ds[ymin_ds:ymax_ds, xmin_ds:xmax_ds]

                        # In the `selected_mask_label_ds`, Keep only labels in mask_ids
                        mask_label_ds = np.where(np.isin(mask_label_ds, mask_ids), mask_label_ds, 0)
                        print(f"sum(mask_label_ds): {np.sum(mask_label_ds)}")

                        # Find contours in the downsampled mask
                        edge_mask = generate_edges(
                            mask_label_ds.compute(),
                            thickness=getattr(self.main_viewer, "mask_outline_thickness", 1),
                            downsample=cdf,
                        )
                        if cummulative:
                            combined = self.img_display.get_array().copy()
                        else:
                            combined = self._materialize_combined()
                            if combined is None:
                                return

                        # print the type of edge_mask
                        print(f"edge_mask: {type(edge_mask)}")
                        combined[edge_mask.compute()] = color_rgb
                        self.img_display.set_data(combined)

                        self.fig.canvas.draw_idle()
                        if self.main_viewer._debug:
                            print("Redrawing canvas")
                    else:
                        print(f"Mask '{mask_name}' not found in FOV '{self.main_viewer.ui_component.image_selector.value}'.")
            else:
                print(f"Masks not loaded for FOV '{self.main_viewer.ui_component.image_selector.value}'.")
            # self.update_patches(do_not_reset=True)

    def update_image(self, combined, extent):
        self.img_display.set_data(combined)
        self.img_display.set_extent(extent)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # ROI selection helpers
    # ------------------------------------------------------------------
    def enable_roi_selector(self, on_complete):
        """Enable a rectangle selector to capture ROI bounds."""
        self.disable_roi_selector()
        self._roi_callback = on_complete
        self._roi_selector = RectangleSelector(
            self.ax,
            self._on_roi_selected,
            button=[1],
            interactive=False,
            useblit=True,
            spancoords='data'
        )
        self._roi_selector.set_active(True)

    def disable_roi_selector(self):
        if self._roi_selector is not None:
            try:
                self._roi_selector.set_active(False)
            except Exception:
                pass
            self._roi_selector.disconnect_events()
            self._roi_selector = None
        self._roi_callback = None

    def _on_roi_selected(self, eclick, erelease):
        if self._roi_callback is None:
            return

        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata

        if None in (x0, y0, x1, y1):
            self.disable_roi_selector()
            self._roi_callback(None)
            return

        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])

        # Clamp to image bounds
        x_min = max(0.0, min(self.width, x_min))
        x_max = max(0.0, min(self.width, x_max))
        y_min = max(0.0, min(self.height, y_min))
        y_max = max(0.0, min(self.height, y_max))

        if x_max - x_min <= 0 or y_max - y_min <= 0:
            self.disable_roi_selector()
            self._roi_callback(None)
            return

        bounds = {
            "x_min": float(x_min),
            "x_max": float(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
        }

        callback = self._roi_callback
        self.disable_roi_selector()
        callback(bounds)
        