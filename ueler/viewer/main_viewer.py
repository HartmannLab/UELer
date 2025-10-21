# viewer.py

import logging
import os
import numpy as np
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.text import Annotation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import cv2
from collections import OrderedDict
from types import SimpleNamespace
from ipywidgets import IntText, Output, Dropdown, FloatSlider, Checkbox, Button, VBox, HBox, Layout, Widget
from IPython.display import display
import pandas as pd
from skimage.segmentation import find_boundaries
# Import modules
from ueler.constants import PREDEFINED_COLORS, DOWNSAMPLE_FACTORS, UICOMPNENTS_SKIP
from ueler.data_loader import (
    load_annotations_for_fov,
    load_channel_struct_fov,
    load_images_for_fov,
    load_masks_for_fov,
    load_one_channel_fov,
    merge_channel_max,
)
from ueler.image_utils import (
    calculate_downsample_factor,
    generate_edges,
    get_axis_limits_with_padding,
)
from ueler.viewer.images import load_asset_bytes
from .ui_components import create_widgets, display_ui, update_wide_plugin_panel
from .roi_manager import ROIManager
from ueler.viewer.plugin.roi_manager_plugin import ROIManagerPlugin
import json

from .image_display import ImageDisplay
import importlib
from ueler.viewer.plugin.plugin_base import PluginBase
from ueler.viewer.annotation_palette_editor import AnnotationPaletteEditor
from ueler.viewer.color_palettes import (
    DEFAULT_COLOR,
    apply_color_defaults,
    build_discrete_colormap,
    merge_palette_updates,
)
from ueler.viewer.rendering import (
    AnnotationRenderSettings,
    ChannelRenderSettings,
    MaskRenderSettings,
    render_fov_to_array,
)
from ueler.export import ExportJob
# viewer/main_viewer.py

from weakref import WeakKeyDictionary

from skimage.io import imsave
# from dask.distributed import LocalCluster, Client

# # Create a LocalCluster with a memory limit of 4 GB per worker
# cluster = LocalCluster(memory_limit='8GB')

# # Connect to the cluster
# client = Client(cluster)

# main_viewer.py

__all__ = ["ImageMaskViewer"]


def _unique_annotation_values(array):
    def export_fovs_batch(
        self,
        marker_set_name,
        output_dir=None,
        fovs=None,
        surfix=None,
        file_format="png",
        downsample_factor=None,
        dpi=300,
        figure_size=None,
        show_progress=True,
    ):
        """Export multiple FOVs using a specified marker set."""

        import time
        from IPython.display import clear_output

        logger = logging.getLogger(__name__)
        results: Dict[str, Any] = {}

        if marker_set_name not in self.marker_sets:
            raise ValueError(f"Marker set '{marker_set_name}' not found.")

        if output_dir is None:
            output_dir = os.path.join(self.base_folder, f"exported_{marker_set_name}")
        os.makedirs(output_dir, exist_ok=True)

        if fovs is None:
            fovs = list(self.available_fovs)
        elif isinstance(fovs, str):
            fovs = [fovs]

        if downsample_factor is None:
            downsample_factor = self.current_downsample_factor

        current_state = {
            "fov": self.ui_component.image_selector.value,
            "channels": self.ui_component.channel_selector.value,
            "downsample_factor": self.current_downsample_factor,
            "channel_settings": {},
            "mask_settings": {},
        }

        for ch in current_state["channels"]:
            if ch in self.ui_component.color_controls:
                current_state["channel_settings"][ch] = {
                    "color": self.ui_component.color_controls[ch].value,
                    "contrast_min": self.ui_component.contrast_min_controls[ch].value,
                    "contrast_max": self.ui_component.contrast_max_controls[ch].value,
                }

        if self.masks_available:
            for mask_name in self.mask_names:
                if mask_name in self.ui_component.mask_display_controls:
                    current_state["mask_settings"][mask_name] = {
                        "visible": self.ui_component.mask_display_controls[mask_name].value,
                        "color": (
                            self.ui_component.mask_color_controls[mask_name].value
                            if mask_name in self.ui_component.mask_color_controls
                            else None
                        ),
                    }

        original_figsize = self.image_display.fig.get_size_inches()
        total_fovs = len(fovs)

        try:
            marker_set = self.marker_sets[marker_set_name]
            selected_channels = tuple(marker_set["selected_channels"])
            channel_settings = marker_set["channel_settings"]

            self.ui_component.channel_selector.value = tuple(selected_channels)
            self.update_controls(None)

            for ch in selected_channels:
                if ch in self.ui_component.color_controls and ch in channel_settings:
                    self.ui_component.color_controls[ch].value = channel_settings[ch]["color"]
                    self.ui_component.contrast_min_controls[ch].value = channel_settings[ch]["contrast_min"]
                    self.ui_component.contrast_max_controls[ch].value = channel_settings[ch]["contrast_max"]

            if figure_size is not None:
                self.image_display.fig.set_size_inches(figure_size)

            start_time = time.time()
            items = [
                {"id": fov, "fov": fov, "index": index, "total": total_fovs}
                for index, fov in enumerate(fovs, start=1)
            ]

            def _export_worker(payload: Dict[str, Any]) -> str:
                index = payload["index"]
                total = payload["total"]
                fov = payload["fov"]

                if show_progress:
                    clear_output(wait=True)
                    print(f"Exporting FOV {index}/{total}: {fov}")
                    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

                temp_current_fov = self.ui_component.image_selector.value
                try:
                    self.load_fov(fov, selected_channels)
                    fov_images = self.image_cache[fov]

                    missing_channels = [
                        ch for ch in selected_channels if ch not in fov_images or fov_images[ch] is None
                    ]
                    if missing_channels:
                        raise ValueError(
                            f"Missing or failed to load channels: {', '.join(missing_channels)}"
                        )

                    first_channel = next(
                        (fov_images[ch] for ch in selected_channels if fov_images[ch] is not None),
                        None,
                    )
                    if first_channel is None:
                        raise ValueError("No valid channels found in this FOV")

                    height, width = first_channel.shape
                    self.ui_component.image_selector.value = fov

                    downsample_factor_adjusted = min(
                        downsample_factor,
                        min(max(1, width // 2), max(1, height // 2)),
                    )
                    if downsample_factor_adjusted < 1:
                        downsample_factor_adjusted = 1

                    if show_progress:
                        print(f"Using downsample factor: {downsample_factor_adjusted}")

                    xmin, xmax, ymin, ymax = 0, width, 0, height
                    xmin_ds = xmin // downsample_factor_adjusted
                    xmax_ds = xmin_ds + (xmax - xmin) // downsample_factor_adjusted
                    ymin_ds = ymin // downsample_factor_adjusted
                    ymax_ds = ymin_ds + (ymax - ymin) // downsample_factor_adjusted

                    if show_progress:
                        print(
                            "Rendering image: size "
                            f"{xmax - xmin}x{ymax - ymin} → {xmax_ds - xmin_ds}x{ymax_ds - ymin_ds}"
                        )

                    combined = self.render_image(
                        selected_channels,
                        downsample_factor_adjusted,
                        (xmin, xmax, ymin, ymax),
                        (xmin_ds, xmax_ds, ymin_ds, ymax_ds),
                    )

                    img_array = np.asarray(combined, dtype=np.float32)
                    img_array = np.clip(img_array, 0.0, 1.0)
                    img_array = (img_array * 255).astype(np.uint8)

                    if img_array.shape[0] < 2 or img_array.shape[1] < 2:
                        raise ValueError(f"Image too small: {img_array.shape}")

                    filename = (
                        f"{fov}_{surfix}.{file_format}"
                        if surfix is not None
                        else f"{fov}.{file_format}"
                    )
                    output_path = os.path.join(output_dir, filename)
                    imsave(output_path, img_array)

                    if show_progress:
                        print(f"Successfully exported {fov} to {output_path}")

                    return output_path
                finally:
                    self.ui_component.image_selector.value = temp_current_fov

            job = ExportJob(
                mode="full_fov",
                items=items,
                output_dir=output_dir,
                file_format=file_format,
                worker=_export_worker,
                overrides={
                    "marker_set": marker_set_name,
                    "downsample_factor": downsample_factor,
                    "dpi": dpi,
                    "figure_size": figure_size,
                    "surfix": surfix,
                },
                logger=logger,
            )

            job.start()
            snapshot = job.status()

            for entry in snapshot["items"]:
                fov_name = entry["item"]["fov"]
                status_value = entry["status"]
                if status_value == "succeeded":
                    results[fov_name] = True
                elif status_value == "failed":
                    error = entry.get("error") or {}
                    error_type = error.get("type") or "Error"
                    message = error.get("message") or ""
                    combined_message = f"{error_type}: {message}".strip()
                    results[fov_name] = combined_message or error_type
                elif status_value == "cancelled":
                    results[fov_name] = "Cancelled"
                else:
                    results[fov_name] = status_value

            if show_progress:
                clear_output(wait=True)
                successful = sum(1 for value in results.values() if value is True)
                print(f"Export complete: {successful}/{total_fovs} FOVs exported successfully")
                failures = [(name, value) for name, value in results.items() if value is not True]
                if failures:
                    print("The following FOVs had errors:")
                    for name, error in failures:
                        print(f"  {name}: {error}")

            logger.info(
                "Batch export finished (marker_set=%s, state=%s, succeeded=%d, failed=%d, cancelled=%d)",
                marker_set_name,
                snapshot["state"],
                sum(1 for entry in snapshot["items"] if entry["status"] == "succeeded"),
                sum(1 for entry in snapshot["items"] if entry["status"] == "failed"),
                sum(1 for entry in snapshot["items"] if entry["status"] == "cancelled"),
            )

        finally:
            self.image_display.fig.set_size_inches(original_figsize)

            self.ui_component.image_selector.value = current_state["fov"]
            self.ui_component.channel_selector.value = current_state["channels"]
            self.update_controls(None)

            for ch, settings in current_state["channel_settings"].items():
                if ch in self.ui_component.color_controls:
                    self.ui_component.color_controls[ch].value = settings["color"]
                    self.ui_component.contrast_min_controls[ch].value = settings["contrast_min"]
                    self.ui_component.contrast_max_controls[ch].value = settings["contrast_max"]

            if self.masks_available:
                for mask_name, settings in current_state["mask_settings"].items():
                    if mask_name in self.ui_component.mask_display_controls:
                        self.ui_component.mask_display_controls[mask_name].value = settings["visible"]
                    if (
                        mask_name in self.ui_component.mask_color_controls
                        and settings["color"] is not None
                    ):
                        self.ui_component.mask_color_controls[mask_name].value = settings["color"]

            self.update_display(current_state["downsample_factor"])

        return results


    # ------------------------------------------------------------------
    # Backwards-compatibility shim
    # If the concrete ImageMaskViewer class was accidentally removed during a
    # refactor we still want `from ueler.viewer.main_viewer import ImageMaskViewer`
    # to work. Create a lightweight class and attach any top-level functions
    # that look like instance methods (first parameter named 'self').
    try:
        import inspect

        class ImageMaskViewer:  # pragma: no cover - compatibility shim
            """Small compatibility class exposing module-level functions as methods."""
            pass

        for _name, _obj in list(globals().items()):
            if inspect.isfunction(_obj):
                try:
                    _sig = inspect.signature(_obj)
                except (ValueError, TypeError):
                    continue
                _params = list(_sig.parameters.keys())
                if _params and _params[0] == "self":
                    setattr(ImageMaskViewer, _name, _obj)

        # Ensure the public API contains the symbol
        if "ImageMaskViewer" not in globals().get("__all__", []):
            __all__.append("ImageMaskViewer")
    except Exception:
        # If anything goes wrong here we intentionally fail silently so the
        # original refactor errors remain visible to developers.
        pass

        # Load masks only if masks are available
        if self.masks_available:
            # Load masks if not in cache
            if fov_name not in self.mask_cache:
                # Load masks for FOV
                masks_dict = load_masks_for_fov(fov_name, self.masks_folder, self.mask_names_set)
                if masks_dict:
                    self.mask_cache[fov_name] = masks_dict
                    self.mask_cache.move_to_end(fov_name)

                    # Precompute edge masks and downsampled label masks
                    self.edge_masks_cache[fov_name] = {}
                    self.label_masks_cache[fov_name] = {}
                    for mask_name, mask in masks_dict.items():
                        # Check if the mask is valid and has non-zero dimensions
                        if mask is None or mask.size == 0:
                            print(f"Warning: Mask '{mask_name}' in FOV '{fov_name}' is empty or invalid.")
                            continue  # Skip this mask

                        self.edge_masks_cache[fov_name][mask_name] = {}
                        self.label_masks_cache[fov_name][mask_name] = {}
                        for factor in self.downsample_factors:
                            # Ensure the downsampled dimensions are at least 1
                            new_width = max(1, mask.shape[1] // factor)
                            new_height = max(1, mask.shape[0] // factor)

                            # Downsample label mask
                            downsampled_mask = mask.astype(np.float32)[::factor,::factor].astype(mask.dtype)
                            self.label_masks_cache[fov_name][mask_name][factor] = downsampled_mask

                            # # Generate edge mask
                            # edge_mask = generate_edges(downsampled_mask)
                            # self.edge_masks_cache[fov_name][mask_name][factor] = edge_mask
                else:
                    self.mask_cache[fov_name] = {}
                    self.edge_masks_cache[fov_name] = {}
                    self.label_masks_cache[fov_name] = {}

                # Remove least recently used if cache exceeds max size
                while len(self.mask_cache) > self.max_cache_size:
                    removed_fov, _ = self.mask_cache.popitem(last=False)
                    self.edge_masks_cache.pop(removed_fov, None)
                    self.label_masks_cache.pop(removed_fov, None)
                    print(f"Removed FOV '{removed_fov}' from mask cache.")
            else:
                # Move the accessed FOV to the end to mark it as recently used
                self.mask_cache.move_to_end(fov_name)

            # Update global mask names set
            self.mask_names = sorted(self.mask_names_set)

        # Load annotations if available
        if self.annotations_available:
            if fov_name not in self.annotation_cache:
                annotation_dict = load_annotations_for_fov(
                    fov_name,
                    self.annotations_folder,
                    self.annotation_names_set,
                )
                self.annotation_cache[fov_name] = annotation_dict or {}
                self.annotation_label_cache[fov_name] = {}
                self.annotation_names_set.update(annotation_dict.keys())

                for annotation_name, annotation_array in annotation_dict.items():
                    self.annotation_label_cache[fov_name][annotation_name] = {}
                    for factor in self.downsample_factors:
                        ds = annotation_array[::factor, ::factor]
                        self.annotation_label_cache[fov_name][annotation_name][factor] = ds

                    sample_values = _unique_annotation_values(annotation_array)
                    if sample_values.size == 0:
                        sample_values = np.array([0], dtype=np.int32)
                    unique_ids = sorted({int(v) for v in sample_values})

                    existing_ids = self.annotation_class_ids.get(annotation_name, [])
                    merged_ids = sorted(set(existing_ids) | set(unique_ids))
                    self.annotation_class_ids[annotation_name] = merged_ids

                    existing_palette = self.annotation_palettes.get(annotation_name, {})
                    palette = apply_color_defaults(merged_ids, existing_palette)
                    self.annotation_palettes[annotation_name] = dict(palette)

                    existing_labels = self.annotation_class_labels.get(annotation_name, {})
                    for class_id in merged_ids:
                        key = str(class_id)
                        if key not in existing_labels:
                            existing_labels[key] = str(class_id)
                    self.annotation_class_labels[annotation_name] = existing_labels
                self.annotation_names = sorted(self.annotation_names_set)
            else:
                self.annotation_cache.move_to_end(fov_name)

            while len(self.annotation_cache) > self.max_cache_size:
                removed_fov, _ = self.annotation_cache.popitem(last=False)
                self.annotation_label_cache.pop(removed_fov, None)

    def on_image_change(self, change):
        previous_channels = self.ui_component.channel_selector.value  # Keep previous channel selections

        # Load the selected FOV
        self.load_fov(self.ui_component.image_selector.value)

        # Update image dimensions
        fov_images = self.image_cache[self.ui_component.image_selector.value]
        first_channel_image = next(iter(fov_images.values()))
        self.height, self.width = first_channel_image.shape
        self.image_display.height = self.height
        self.image_display.width = self.width

        # Update axis limits
        self.image_display.ax.set_xlim(0, self.width)
        self.image_display.ax.set_ylim(self.height, 0)

        # Update channel names
        new_channel_options = list(fov_images.keys())
        self.channel_names_set.update(new_channel_options)
        self.ui_component.channel_selector.allowed_tags = new_channel_options

        # Keep previous selections if they exist in the new image
        new_selection = tuple(ch for ch in previous_channels if ch in new_channel_options)
        if new_selection:
            self.ui_component.channel_selector.value = new_selection
        elif self.cell_table is not None:
            self.ui_component.channel_selector.value = (new_channel_options[0],)
        else:
            self.ui_component.channel_selector.value = ()

        # Remove controls for channels that no longer exist
        channels_to_remove = [ch for ch in self.ui_component.color_controls if ch not in new_channel_options]
        for ch in channels_to_remove:
            del self.ui_component.color_controls[ch]
            del self.ui_component.contrast_min_controls[ch]
            del self.ui_component.contrast_max_controls[ch]

        # Update mask controls only if masks are available
        if self.masks_available:
            if self.ui_component.image_selector.value in self.mask_cache:
                available_masks = self.mask_cache[self.ui_component.image_selector.value].keys()
            else:
                available_masks = []

            # Update mask display controls
            for mask_name in self.mask_names:
                if mask_name not in available_masks:
                    # Disable mask controls if mask not available in current FOV
                    if mask_name in self.ui_component.mask_display_controls:
                        self.ui_component.mask_display_controls[mask_name].disabled = True
                    if mask_name in self.ui_component.mask_color_controls:
                        self.ui_component.mask_color_controls[mask_name].disabled = True
                else:
                    if mask_name in self.ui_component.mask_display_controls:
                        self.ui_component.mask_display_controls[mask_name].disabled = False
                    if mask_name in self.ui_component.mask_color_controls:
                        self.ui_component.mask_color_controls[mask_name].disabled = False

        if self.annotations_available:
            selector = getattr(self.ui_component, "annotation_selector", None)
            display_toggle = getattr(self.ui_component, "annotation_display_checkbox", None)
            available_annotation_names = []
            if self.ui_component.image_selector.value in self.annotation_cache:
                available_annotation_names = sorted(self.annotation_cache[self.ui_component.image_selector.value].keys())

            option_pairs = [(name, name) for name in available_annotation_names]

            if selector is not None:
                selector.options = option_pairs
                if available_annotation_names:
                    if self.active_annotation_name not in available_annotation_names:
                        self.active_annotation_name = available_annotation_names[0]
                    selector.value = self.active_annotation_name
                else:
                    selector.value = None
            if display_toggle is not None:
                has_options = bool(available_annotation_names)
                display_toggle.disabled = not has_options
                if not has_options:
                    display_toggle.value = False
                    self.annotation_display_enabled = False

            self._refresh_annotation_control_states()

        self.update_controls(None)

        if self.initialized: 
            if self.SidePlots:
                self.SidePlots.chart_output.highlight_cells()
                if self.SidePlots.heatmap_output.ui_component.main_viewer_checkbox.value:
                    self.SidePlots.heatmap_output.highlight_cells()
                else:
                    self.image_display.clear_patches()

        ax = self.image_display.ax

        toolbar = getattr(self.image_display.fig.canvas, "toolbar", None)
        nav_stack = getattr(toolbar, "_nav_stack", None) if toolbar is not None else None
        elements = getattr(nav_stack, "_elements", None) if nav_stack is not None else None

        if elements:
            nav_stack_element = elements[0]
            try:
                existing_view, existing_bboxes = nav_stack_element[ax]
            except KeyError:
                existing_view, existing_bboxes = ({}, ())
            existing_view = dict(existing_view)
            existing_view['xlim'] = ax.get_xlim()
            existing_view['ylim'] = ax.get_ylim()
            nav_stack_element[ax] = (existing_view, existing_bboxes)
        elif toolbar is not None and hasattr(toolbar, "push_current"):
            toolbar.push_current()

        self.inform_plugins('on_fov_change')

    def on_channel_selection_change(self, change):
        self.update_controls(None)
        self.setup_widget_observers()
        # Optional: Decide whether to clear the marker set selection
        # self.marker_set_dropdown.value = None
    
    def on_pixel_size_change(self, change):
        image_display = self.image_display
        ax = image_display.ax
        # Remove existing scalebar if it exists
        if hasattr(image_display, 'scalebar') and image_display.scalebar:
            image_display.scalebar.remove()
        
        # Determine new length and label
        if change['new'] == 390:
            new_length = 102.6
            new_label = '40 µm'
        elif change['new'] == 195:
            new_length = 51.3
            new_label = '20 µm'
        elif change['new'] == 98:
            new_length = 102
            new_label = '10 µm'
        
        # Create and add the new scalebar
        fontprops = fm.FontProperties(size=12)
        image_display.scalebar = AnchoredSizeBar(
            ax.transData,
            new_length,
            new_label,
            'lower right',
            pad=0.5,
            color='white',
            frameon=False,
            size_vertical=2,
            fontproperties=fontprops
        )
        ax.add_artist(image_display.scalebar)
        image_display.fig.canvas.draw()
    
    def on_key_change(self, change):
        # dynamically assign the new values to x_key, y_key, label_key, and mask_key
        # based on the selected key
        owner = change['owner']
        key_name = owner.description

        if key_name == 'X key:':
            key_name = 'x_key'
        elif key_name == 'Y key:':
            key_name = 'y_key'
        elif key_name == 'Label key:':
            key_name = 'label_key'
        elif key_name == 'Mask key:':
            key_name = 'mask_key'
        elif key_name == 'Fov key:':
            key_name = 'fov_key'

        new_key = change['new']
        setattr(self, key_name, new_key)
        print(f"{key_name} set to '{new_key}'.")


    def _get_channel_stats(self, channel):
        stats = self.channel_max_values.get(channel)
        if isinstance(stats, dict):
            display_max = stats.get("display_max")
            dtype_max = stats.get("dtype_max")
        else:
            display_max = stats
            dtype_max = None

        if not isinstance(display_max, (int, float)) or not np.isfinite(display_max) or display_max <= 0:
            display_max = None

        if dtype_max is None or not np.isfinite(dtype_max) or dtype_max <= 0:
            dtype_max = None

        if display_max is None:
            display_max = 65535.0

        if dtype_max is None:
            dtype_max = 65535.0

        display_max = min(display_max, dtype_max) if dtype_max else display_max

        return float(display_max), float(dtype_max)

    @staticmethod
    def _calculate_slider_step(max_value):
        if max_value <= 0:
            return 0.01
        if max_value <= 1.0:
            return max(max_value / 100.0, 0.001)
        return max(1.0, max_value / 1000.0)

    @staticmethod
    def _slider_readout_format(max_value):
        if max_value is None or not np.isfinite(max_value):
            return '0.2e'
        if max_value == 0:
            return '0.2e'
        threshold = 1.0
        if abs(max_value) < threshold:
            return '0.2e'
        if abs(max_value) >= 1e4:
            return '0.2e'
        return '0.2f'

    def _sync_channel_controls(self, channel):
        factor_max_value = 1.2
        if not hasattr(self, "ui_component") or self.ui_component is None:
            return

        if not hasattr(self.ui_component, "contrast_min_controls"):
            return

        max_value, _ = self._get_channel_stats(channel)
        max_value *= factor_max_value
        step_size = self._calculate_slider_step(max_value)
        readout_format = self._slider_readout_format(max_value)

        min_slider = self.ui_component.contrast_min_controls.get(channel)
        max_slider = self.ui_component.contrast_max_controls.get(channel)

        if min_slider is not None:
            min_slider.max = max_value
            min_slider.step = step_size
            min_slider.readout_format = readout_format
            if min_slider.value > max_value:
                min_slider.value = max_value

        if max_slider is not None:
            max_slider.max = max_value
            max_slider.step = step_size
            max_slider.readout_format = readout_format
            lower_bound = min_slider.value if min_slider is not None else 0.0
            if max_slider.value < lower_bound:
                max_slider.value = lower_bound
            elif max_slider.value > max_value:
                max_slider.value = max_value

    def _merge_channel_max(self, channel, display_max, dtype_max=None, sync=True):
        if channel is None or display_max is None:
            return

        try:
            display_value = float(display_max)
        except (TypeError, ValueError):
            return

        if not np.isfinite(display_value):
            return

        if dtype_max is None:
            prior = self.channel_max_values.get(channel)
            if isinstance(prior, dict):
                dtype_candidate = prior.get("dtype_max")
            elif prior is not None:
                try:
                    dtype_candidate = float(prior)
                except (TypeError, ValueError):
                    dtype_candidate = None
            else:
                dtype_candidate = None
        else:
            dtype_candidate = dtype_max

        try:
            dtype_value = float(dtype_candidate) if dtype_candidate is not None else display_value
        except (TypeError, ValueError):
            dtype_value = display_value

        if not np.isfinite(dtype_value):
            dtype_value = display_value

        dtype_value = max(dtype_value, display_value)

        if merge_channel_max(channel, self.channel_max_values, display_value, dtype_value) and sync:
            self._sync_channel_controls(channel)

    def update_controls(self, change):
        """Create widgets dynamically based on selected channels and masks, and attach update callbacks."""
        channel_widgets = []

        for channel in self.ui_component.channel_selector.value:
            max_value, _ = self._get_channel_stats(channel)
            step_size = self._calculate_slider_step(max_value)
            readout_format = self._slider_readout_format(max_value)

            # Reuse existing controls if they exist
            if channel in self.ui_component.color_controls:
                color_dropdown = self.ui_component.color_controls[channel]
            else:
                color_dropdown = Dropdown(
                    options=list(self.predefined_colors.keys()),
                    value="Red",
                    description=f"Color {channel}",
                    disabled=False
                )
                color_dropdown.observe(lambda change, ch=channel: self.update_display(self.current_downsample_factor), names='value')
                self.ui_component.color_controls[channel] = color_dropdown

            if channel in self.ui_component.contrast_min_controls:
                contrast_min_slider = self.ui_component.contrast_min_controls[channel]
                contrast_min_slider.max = max_value
                contrast_min_slider.step = step_size
                contrast_min_slider.readout_format = readout_format
                contrast_min_slider.value = min(contrast_min_slider.value, max_value)
            else:
                contrast_min_slider = FloatSlider(
                    value=0.0,
                    min=0.0,
                    max=max_value,
                    step=step_size,
                    description=f"Min {channel}",
                    continuous_update=False,
                    readout_format=readout_format
                )
                contrast_min_slider.observe(lambda change, ch=channel: self.update_display(self.current_downsample_factor), names='value')
                self.ui_component.contrast_min_controls[channel] = contrast_min_slider

            if channel in self.ui_component.contrast_max_controls:
                contrast_max_slider = self.ui_component.contrast_max_controls[channel]
                contrast_max_slider.max = max_value
                contrast_max_slider.step = step_size
                contrast_max_slider.readout_format = readout_format
                contrast_max_slider.value = max(min(contrast_max_slider.value, max_value), contrast_min_slider.value)
            else:
                contrast_max_slider = FloatSlider(
                    value=max_value,
                    min=0.0,
                    max=max_value,
                    step=step_size,
                    description=f"Max {channel}",
                    continuous_update=False,
                    readout_format=readout_format
                )
                contrast_max_slider.observe(lambda change, ch=channel: self.update_display(self.current_downsample_factor), names='value')
                self.ui_component.contrast_max_controls[channel] = contrast_max_slider

            channel_widgets.extend([color_dropdown, contrast_min_slider, contrast_max_slider])

        if channel_widgets:
            self.ui_component.channel_controls_box.children = tuple(channel_widgets)
        else:
            self.ui_component.channel_controls_box.children = (
                self.ui_component.no_channels_label,
            )

        mask_widgets = []
        if self.masks_available:
            for mask_name in self.mask_names:
                # Checkbox to display/hide mask
                if mask_name in self.ui_component.mask_display_controls:
                    display_checkbox = self.ui_component.mask_display_controls[mask_name]
                else:
                    display_checkbox = Checkbox(
                        value=False,
                        description='',
                        disabled=False,
                        layout=Layout(width='auto'),
                        style={'description_width': 'auto'}
                    )
                    display_checkbox.observe(lambda change, mn=mask_name: self.update_display(self.current_downsample_factor), names='value')
                    self.ui_component.mask_display_controls[mask_name] = display_checkbox

                # Dropdown to select mask color
                if mask_name in self.ui_component.mask_color_controls:
                    mask_color_dropdown = self.ui_component.mask_color_controls[mask_name]
                else:
                    mask_color_dropdown = Dropdown(
                        options=list(self.predefined_colors.keys()),
                        value="White",
                        description=f"Mask {mask_name}",
                        disabled=False,
                        layout=Layout(width='250px'),
                        style={'description_width': '150px'}
                    )
                    mask_color_dropdown.observe(lambda change, mn=mask_name: self.update_display(self.current_downsample_factor), names='value')
                    self.ui_component.mask_color_controls[mask_name] = mask_color_dropdown

                mask_control = HBox(
                    [mask_color_dropdown, display_checkbox],
                    layout=Layout(
                        display='flex',
                        align_items='flex-start',
                        justify_content='flex-start',
                        gap='10px',
                        min_height='30px',
                        overflow='hidden'
                    )
                )
                mask_widgets.append(mask_control)

        if mask_widgets:
            self.ui_component.mask_controls_box.children = tuple(mask_widgets)
        elif self.masks_available:
            self.ui_component.mask_controls_box.children = (
                self.ui_component.no_masks_label,
            )
        else:
            self.ui_component.mask_controls_box.children = tuple()

        annotation_widgets = []
        if self.annotations_available:
            annotation_widgets = [
                self.ui_component.annotation_controls_header,
                self.ui_component.annotation_display_checkbox,
                self.ui_component.annotation_selector,
                self.ui_component.annotation_overlay_mode,
                self.ui_component.annotation_alpha_slider,
                self.ui_component.annotation_label_mode,
                self.ui_component.annotation_edit_button,
            ]
            self.ui_component.annotation_controls_box.children = tuple(annotation_widgets)
        else:
            self.ui_component.annotation_controls_box.children = (
                self.ui_component.no_annotations_label,
            )

        accordion = self.ui_component.control_sections
        previous_index = getattr(accordion, "selected_index", None)
        previous_title = None
        if (
            previous_index is not None
            and 0 <= previous_index < len(getattr(self, "_control_section_titles", []))
        ):
            previous_title = self._control_section_titles[previous_index]

        section_children = []
        section_titles = []

        if self.ui_component.channel_controls_box.children:
            section_children.append(self.ui_component.channel_controls_box)
            section_titles.append("Channels")

        if self.annotations_available:
            if not annotation_widgets:
                self.ui_component.annotation_controls_box.children = (
                    self.ui_component.no_annotations_label,
                )
            section_children.append(self.ui_component.annotation_controls_box)
            section_titles.append("Annotations")

        if self.masks_available:
            if not mask_widgets:
                self.ui_component.mask_controls_box.children = (
                    self.ui_component.no_masks_label,
                )
            section_children.append(self.ui_component.mask_controls_box)
            section_titles.append("Masks")

        if not section_children:
            section_children = (
                VBox([self.ui_component.empty_controls_placeholder], layout=Layout(width="100%")),
            )
            section_titles = ["Controls"]

        accordion.children = tuple(section_children)
        for idx, title in enumerate(section_titles):
            accordion.set_title(idx, title)

        self._control_section_titles = list(section_titles)

        if section_children:
            if previous_title and previous_title in section_titles:
                accordion.selected_index = section_titles.index(previous_title)
            elif previous_index is not None and previous_index < len(section_children):
                accordion.selected_index = previous_index
            else:
                accordion.selected_index = 0
        else:
            accordion.selected_index = None

        self._refresh_annotation_control_states()

        self.update_display(self.current_downsample_factor)

    def _refresh_annotation_control_states(self):
        if not hasattr(self, "ui_component"):
            return

        selector = getattr(self.ui_component, "annotation_selector", None)
        display_checkbox = getattr(self.ui_component, "annotation_display_checkbox", None)
        overlay_mode_control = getattr(self.ui_component, "annotation_overlay_mode", None)
        alpha_slider = getattr(self.ui_component, "annotation_alpha_slider", None)
        label_mode_control = getattr(self.ui_component, "annotation_label_mode", None)
        edit_button = getattr(self.ui_component, "annotation_edit_button", None)

        if selector is None or display_checkbox is None:
            return

        raw_options = list(selector.options)
        option_values = []
        for item in raw_options:
            if isinstance(item, dict):
                label_candidate = item.get("label")
                value = item.get("value", label_candidate)
            elif isinstance(item, (tuple, list)):
                if not item:
                    continue
                if len(item) == 1:
                    value = item[0]
                else:
                    value = item[1]
            else:
                value = item
            if value is None:
                continue
            option_values.append(value)

        has_annotations = bool(option_values)
        current_selection = selector.value if has_annotations else None
        if current_selection not in option_values:
            current_selection = None

        if display_checkbox.value != self.annotation_display_enabled:
            display_checkbox.value = self.annotation_display_enabled if has_annotations else False

        display_checkbox.disabled = not has_annotations
        selector.disabled = not has_annotations

        if not has_annotations:
            self.annotation_display_enabled = False
            self.active_annotation_name = None
            self.annotation_palette_editor.hide()

        if has_annotations:
            preferred = self.active_annotation_name if self.active_annotation_name in option_values else option_values[0]
            if selector.value != preferred:
                selector.value = preferred
            current_selection = selector.value
            if current_selection is not None:
                self.active_annotation_name = current_selection
        else:
            current_selection = None

        if overlay_mode_control is not None:
            overlay_mode_control.disabled = not has_annotations
            if overlay_mode_control.value != self.annotation_overlay_mode:
                overlay_mode_control.value = self.annotation_overlay_mode

        if alpha_slider is not None:
            alpha_slider.disabled = not (self.annotation_display_enabled and current_selection is not None)
            if alpha_slider.value != self.annotation_overlay_alpha:
                alpha_slider.value = float(self.annotation_overlay_alpha)

        if label_mode_control is not None:
            label_mode_control.disabled = not has_annotations
            if label_mode_control.value != self.annotation_label_display_mode:
                label_mode_control.value = self.annotation_label_display_mode

        if edit_button is not None:
            edit_button.disabled = current_selection is None

    def on_annotation_toggle(self, change):
        self.annotation_display_enabled = bool(change.get("new"))
        self._refresh_annotation_control_states()
        self.update_display(self.current_downsample_factor)

    def on_annotation_selection_change(self, change):
        new_annotation = change.get("new") if isinstance(change, dict) else None
        if new_annotation == self.active_annotation_name:
            return
        self.active_annotation_name = new_annotation
        self._ensure_annotation_metadata(new_annotation)
        self._refresh_annotation_control_states()
        if self.annotation_display_enabled and new_annotation:
            self.update_display(self.current_downsample_factor)

    def on_annotation_overlay_mode_change(self, change):
        new_mode = change.get("new") if isinstance(change, dict) else None
        if not new_mode or new_mode == self.annotation_overlay_mode:
            return
        self.annotation_overlay_mode = new_mode
        if self.annotation_display_enabled or new_mode == "mask":
            self.update_display(self.current_downsample_factor)

    def on_annotation_alpha_change(self, change):
        new_alpha = change.get("new") if isinstance(change, dict) else None
        if new_alpha is None:
            return
        try:
            alpha_value = float(new_alpha)
        except (TypeError, ValueError):
            return
        if np.isfinite(alpha_value):
            self.annotation_overlay_alpha = np.clip(alpha_value, 0.0, 1.0)
            if self.annotation_display_enabled:
                self.update_display(self.current_downsample_factor)

    def on_annotation_label_mode_change(self, change):
        new_mode = change.get("new") if isinstance(change, dict) else None
        if new_mode in {"id", "label"}:
            self.annotation_label_display_mode = new_mode
            self._refresh_annotation_control_states()

    def on_edit_annotation_palette(self, _button):
        if not self.active_annotation_name:
            return
        self._ensure_annotation_metadata(self.active_annotation_name)
        class_ids = self.annotation_class_ids.get(self.active_annotation_name, [])
        palette = self.annotation_palettes.get(self.active_annotation_name, {})
        labels = self.annotation_class_labels.get(self.active_annotation_name, {})
        self.annotation_palette_editor.load(self.active_annotation_name, class_ids, palette, labels)

    def apply_annotation_palette_changes(self, annotation_name, palette_updates, label_updates):
        palette = self.annotation_palettes.setdefault(annotation_name, {})
        merge_palette_updates(palette, palette_updates)

        labels = self.annotation_class_labels.setdefault(annotation_name, {})
        for key, value in label_updates.items():
            labels[str(key)] = value

        self._refresh_annotation_control_states()
        if annotation_name == self.active_annotation_name and self.annotation_display_enabled:
            self.update_display(self.current_downsample_factor)

    def close_annotation_editor(self):
        self._refresh_annotation_control_states()

    def _ensure_annotation_metadata(self, annotation_name):
        if not annotation_name:
            return

        existing_ids = self.annotation_class_ids.get(annotation_name)
        if existing_ids:
            return

        fov_annotations = self.annotation_cache.get(self.ui_component.image_selector.value, {})
        annotation_array = fov_annotations.get(annotation_name)
        if annotation_array is None:
            return

        sample_values = _unique_annotation_values(annotation_array)
        if sample_values.size == 0:
            sample_values = np.array([0], dtype=np.int32)

        merged_ids = sorted({int(v) for v in sample_values} or {0})
        palette = apply_color_defaults(merged_ids, self.annotation_palettes.get(annotation_name, {}))
        self.annotation_class_ids[annotation_name] = merged_ids
        self.annotation_palettes[annotation_name] = dict(palette)

        labels = self.annotation_class_labels.get(annotation_name, {})
        for class_id in merged_ids:
            key = str(class_id)
            labels.setdefault(key, key)
        self.annotation_class_labels[annotation_name] = labels

    # ------------------------------------------------------------------
    # ROI helpers for plugin integration
    # ------------------------------------------------------------------
    def capture_viewport_bounds(self):
        ax = getattr(self.image_display, "ax", None)
        if ax is None:
            return None

        x_limits = sorted(ax.get_xlim())
        y_limits = sorted(ax.get_ylim())
        width = x_limits[1] - x_limits[0]
        height = y_limits[1] - y_limits[0]
        if width <= 0 or height <= 0:
            return None

        return {
            "x": sum(x_limits) / 2.0,
            "y": sum(y_limits) / 2.0,
            "width": width,
            "height": height,
            "zoom": float(self.current_downsample_factor),
            "x_min": x_limits[0],
            "x_max": x_limits[1],
            "y_min": y_limits[0],
            "y_max": y_limits[1],
        }

    def center_on_roi(self, record):
        target_fov = record.get("fov")
        current_selector = getattr(self.ui_component, "image_selector", None)
        if target_fov and current_selector is not None and target_fov != current_selector.value:
            current_selector.value = target_fov

        x_min = record.get("x_min", 0.0)
        x_max = record.get("x_max", self.width)
        y_min = record.get("y_min", 0.0)
        y_max = record.get("y_max", self.height)

        ax = getattr(self.image_display, "ax", None)
        if ax is None:
            return
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        self.image_display.fig.canvas.draw_idle()
        self.update_display(self.current_downsample_factor)

    def render_image(self, selected_channels, downsample_factor, xym, xym_ds):
        """Render a composite image for the active viewer selection."""

        current_fov = self.ui_component.image_selector.value
        self.load_fov(current_fov, selected_channels)
        fov_images = self.image_cache[current_fov]

        if xym is not None:
            xmin, xmax, ymin, ymax = (int(v) for v in xym)
        else:
            first_channel = next(iter(fov_images.values()), None)
            if first_channel is None:
                raise ValueError(f"FOV '{current_fov}' has no loaded channels")
            shape = getattr(first_channel, "shape", None)
            if not shape or len(shape) < 2:
                raise ValueError("Unable to determine image bounds for rendering")
            ymax, xmax = int(shape[0]), int(shape[1])
            xmin, ymin = 0, 0

        region_xy = (xmin, xmax, ymin, ymax)

        if xym_ds is not None:
            xmin_ds, xmax_ds, ymin_ds, ymax_ds = (int(v) for v in xym_ds)
        else:
            xmin_ds = xmin // downsample_factor
            xmax_ds = max(xmin_ds + 1, xmax // downsample_factor)
            ymin_ds = ymin // downsample_factor
            ymax_ds = max(ymin_ds + 1, ymax // downsample_factor)

        region_ds = (xmin_ds, xmax_ds, ymin_ds, ymax_ds)

        if not selected_channels:
            height = max(1, ymax_ds - ymin_ds)
            width = max(1, xmax_ds - xmin_ds)
            return np.zeros((height, width, 3), dtype=np.float32)

        controls = self.ui_component
        channel_settings = {}
        for ch in selected_channels:
            color_key = controls.color_controls[ch].value
            color_value = self.predefined_colors.get(color_key, color_key)
            color_rgb = to_rgb(color_value)
            channel_settings[ch] = ChannelRenderSettings(
                color=color_rgb,
                contrast_min=controls.contrast_min_controls[ch].value,
                contrast_max=controls.contrast_max_controls[ch].value,
            )

        annotation_settings = None
        if (
            self.annotations_available
            and self.annotation_display_enabled
            and self.active_annotation_name is not None
        ):
            annotation_layers = self.annotation_label_cache.get(current_fov, {})
            annotation_entry = annotation_layers.get(self.active_annotation_name, {})
            annotation_ds = annotation_entry.get(downsample_factor)
            if annotation_ds is None and annotation_entry:
                base_array = annotation_entry.get(1)
                if base_array is not None:
                    annotation_ds = base_array[::downsample_factor, ::downsample_factor]
            if annotation_ds is not None:
                class_ids = self.annotation_class_ids.get(self.active_annotation_name)
                if not class_ids:
                    cache_source = self.annotation_cache.get(current_fov, {}).get(
                        self.active_annotation_name
                    )
                    if cache_source is not None:
                        class_ids = list(_unique_annotation_values(cache_source))
                palette = self.annotation_palettes.get(self.active_annotation_name, {})
                colormap = build_discrete_colormap(class_ids or [0], palette)
                annotation_settings = AnnotationRenderSettings(
                    array=annotation_ds,
                    colormap=colormap,
                    alpha=self.annotation_overlay_alpha,
                    mode=self.annotation_overlay_mode,
                )

        mask_settings = []
        if self.masks_available:
            selected_masks = [
                mask_name for mask_name, cb in controls.mask_display_controls.items() if cb.value
            ]
            for mask_name in selected_masks:
                label_dict = self.label_masks_cache.get(current_fov, {}).get(mask_name)
                if not label_dict:
                    continue
                label_mask_ds = label_dict.get(downsample_factor)
                if label_mask_ds is None:
                    continue
                try:
                    mask_array = label_mask_ds.compute()
                except AttributeError:
                    mask_array = np.asarray(label_mask_ds)
                if mask_array.size == 0:
                    continue
                edge_mask = find_boundaries(mask_array, mode="inner")
                color_key = controls.mask_color_controls[mask_name].value
                color_value = self.predefined_colors.get(color_key, color_key)
                mask_settings.append(
                    MaskRenderSettings(
                        array=edge_mask.astype(bool, copy=False),
                        color=to_rgb(color_value),
                    )
                )

        return render_fov_to_array(
            current_fov,
            fov_images,
            selected_channels,
            channel_settings,
            downsample_factor=downsample_factor,
            region_xy=region_xy,
            region_ds=region_ds,
            annotation=annotation_settings,
            masks=mask_settings,
        )

    def update_display(self, downsample_factor=8):
        """Update the display with the current downsample factor and visible area."""
        # Get visible region coordinates
        xmin, xmax, ymin, ymax, xmin_ds, xmax_ds, ymin_ds, ymax_ds = get_axis_limits_with_padding(self, downsample_factor)

        # Wrap the limits in tuples
        xym = (xmin, xmax, ymin, ymax)
        xym_r = (xmin, xmax, ymax, ymin)
        xym_ds = (xmin_ds, xmax_ds, ymin_ds, ymax_ds)
        
        # Get selected channels
        selected_channels = list(self.ui_component.channel_selector.value)
        
        if not selected_channels:
            # If no channels selected, display black image
            self.image_display.img_display.set_data(np.zeros((xym_ds[3] - xym_ds[2], xym_ds[1] - xym_ds[0], 3), dtype=np.float32))
            self.image_display.img_display.set_extent(xym_r)
            self.image_display.fig.canvas.draw_idle()
            return
        
        # Render the image
        combined = self.render_image(
            selected_channels, 
            downsample_factor,
            xym,
            xym_ds
        )

        # Update the displayed image
        self.image_display.img_display.set_data(combined)
        self.image_display.combined = combined
        self.image_display.img_display.set_extent(xym_r)
        self.image_display.fig.canvas.draw_idle()

        # Update the current label masks for interaction only if masks are available
        if self.masks_available:
            self.current_label_masks = {}
            self.full_resolution_label_masks = {}
            selected_masks = [mask_name for mask_name, cb in self.ui_component.mask_display_controls.items() if cb.value]
            for mask_name in selected_masks:
                if mask_name in self.label_masks_cache[self.ui_component.image_selector.value]:
                    label_mask_dict = self.label_masks_cache[self.ui_component.image_selector.value][mask_name]
                    # Get the full-resolution mask (downsample factor of 1)
                    label_mask_full = label_mask_dict[1]
                    # Get the downsampled mask for current view
                    label_mask_ds = label_mask_dict[downsample_factor]
                    self.current_label_masks[mask_name] = label_mask_ds[ymin_ds:ymax_ds, xmin_ds:xmax_ds]
                    self.full_resolution_label_masks[mask_name] = label_mask_full

            self.image_display.update_patches()
        
        self.inform_plugins('on_mv_update_display')

        # Update the displayed image
        self.image_display.img_display.set_data(combined)
        self.image_display.combined = combined
        self.image_display.img_display.set_extent(xym_r)  # Adjust extent to match axis limits
        self.image_display.fig.canvas.draw_idle()

        # Update the current label masks for interaction only if masks are available
        if self.masks_available:
            self.current_label_masks = {}
            self.full_resolution_label_masks = {}
            for mask_name in selected_masks:
                if mask_name in self.label_masks_cache[self.ui_component.image_selector.value]:
                    label_mask_dict = self.label_masks_cache[self.ui_component.image_selector.value][mask_name]
                    # Get the full-resolution mask (downsample factor of 1)
                    label_mask_full = label_mask_dict[1]
                    # Get the downsampled mask for current view
                    label_mask_ds = label_mask_dict[downsample_factor]
                    self.current_label_masks[mask_name] = label_mask_ds[ymin_ds:ymax_ds, xmin_ds:xmax_ds]
                    self.full_resolution_label_masks[mask_name] = label_mask_full

            self.image_display.update_patches()
        
        self.inform_plugins('on_mv_update_display')
    
    def update_keys(self, *args):
        # Update key attributes based on the loaded widget values.
        if hasattr(self.ui_component, 'x_key'):
            self.x_key = self.ui_component.x_key.value
        if hasattr(self.ui_component, 'y_key'):
            self.y_key = self.ui_component.y_key.value
        if hasattr(self.ui_component, 'label_key'):
            self.label_key = self.ui_component.label_key.value
        if hasattr(self.ui_component, 'mask_key'):
            self.mask_key = self.ui_component.mask_key.value
        if hasattr(self.ui_component, 'fov_key'):
            self.fov_key = self.ui_component.fov_key.value
    
    def inform_plugins(self, method_name):
        '''
        Inform all plugins about the method call.

        loop through all the attributes of self.SidePlots, call the `method_name` method
        '''
        # Check if the SidePlots attribute exists
        if not hasattr(self, 'SidePlots'):
            return
        for attr_name in dir(self.SidePlots):
            attr = getattr(self.SidePlots, attr_name)
            if isinstance(attr, PluginBase):
                try :
                    getattr(attr, method_name)()
                except AttributeError:
                    if self._debug:
                        print(f"Skipping {attr_name}")


    def save_marker_set(self, button):
        set_name = self.ui_component.marker_set_name_input.value.strip()
        if not set_name:
            print("Please enter a valid marker set name.")
            return
        if set_name in self.marker_sets:
            print(f"A marker set named '{set_name}' already exists.")
            return

        # Capture current settings
        selected_channels = list(self.ui_component.channel_selector.value)
        if not selected_channels:
            print("No channels selected to save.")
            return

        channel_settings = {}
        for ch in selected_channels:
            channel_settings[ch] = {
                'color': self.ui_component.color_controls[ch].value,
                'contrast_min': self.ui_component.contrast_min_controls[ch].value,
                'contrast_max': self.ui_component.contrast_max_controls[ch].value
            }

        # Save the marker set
        self.marker_sets[set_name] = {
            'selected_channels': selected_channels,
            'channel_settings': channel_settings
        }
        # Update the marker set dropdown
        self.update_marker_set_dropdown()
        print(f"Marker set '{set_name}' saved.")
        # clear the marker_set_name_input value
        self.ui_component.marker_set_name_input.value = ""

    def update_marker_set(self, button):
        set_name = self.ui_component.marker_set_dropdown.value
        if not set_name:
            print("No marker set selected to update.")
            return

        # Capture current settings
        selected_channels = list(self.ui_component.channel_selector.value)
        if not selected_channels:
            print("No channels selected to save.")
            return

        channel_settings = {}
        for ch in selected_channels:
            channel_settings[ch] = {
                'color': self.ui_component.color_controls[ch].value,
                'contrast_min': self.ui_component.contrast_min_controls[ch].value,
                'contrast_max': self.ui_component.contrast_max_controls[ch].value
            }

        # Update the marker set
        self.marker_sets[set_name] = {
            'selected_channels': selected_channels,
            'channel_settings': channel_settings
        }
        print(f"Marker set '{set_name}' updated.")

    def delete_marker_set(self, button):
        set_name = self.ui_component.marker_set_dropdown.value
        if not set_name:
            print("No marker set selected to delete.")
            return

        # Check if deletion is confirmed
        if not self.ui_component.delete_confirmation_checkbox.value:
            print("Please check 'Confirm Deletion' to delete the marker set.")
            return

        del self.marker_sets[set_name]
        # Update the marker set dropdown
        self.update_marker_set_dropdown()
        self.ui_component.delete_confirmation_checkbox.value = False  # Reset the checkbox
        print(f"Marker set '{set_name}' deleted.")

    def load_marker_set(self, button):
        set_name = self.ui_component.marker_set_dropdown.value
        if not set_name:
            print("No marker set selected to load.")
            return
        if set_name not in self.marker_sets:
            print(f"Marker set '{set_name}' not found.")
            return

        marker_set = self.marker_sets[set_name]
        selected_channels = marker_set['selected_channels']
        channel_settings = marker_set['channel_settings']

        # Ensure cached maxima accommodate saved settings before controls refresh
        for ch in selected_channels:
            settings = channel_settings.get(ch, {})
            saved_max = settings.get('contrast_max')
            if saved_max is not None:
                self._merge_channel_max(ch, saved_max, sync=False)

        # Update channel selector
        self.ui_component.channel_selector.value = tuple(selected_channels)

        # Update controls
        self.update_controls(None)

        for ch in selected_channels:
            settings = channel_settings.get(ch, {})

            # Ensure slider ranges expand to cover any stored maxima
            self._sync_channel_controls(ch)

            color_control = self.ui_component.color_controls.get(ch) if hasattr(self.ui_component, 'color_controls') else None
            min_control = self.ui_component.contrast_min_controls.get(ch) if hasattr(self.ui_component, 'contrast_min_controls') else None
            max_control = self.ui_component.contrast_max_controls.get(ch) if hasattr(self.ui_component, 'contrast_max_controls') else None

            if color_control is not None and 'color' in settings:
                color_control.value = settings['color']
            if min_control is not None and 'contrast_min' in settings:
                min_control.value = settings['contrast_min']
            if max_control is not None and 'contrast_max' in settings:
                max_control.value = settings['contrast_max']

            if 'contrast_max' in settings:
                self._merge_channel_max(ch, settings['contrast_max'])

        # Update display
        self.update_display(self.current_downsample_factor)
        print(f"Marker set '{set_name}' loaded.")

    def update_marker_set_dropdown(self):
        if self.marker_sets:
            self.ui_component.marker_set_dropdown.options = sorted(self.marker_sets.keys())
            self.ui_component.marker_set_dropdown.value = None
        else:
            self.ui_component.marker_set_dropdown.options = []
            self.ui_component.marker_set_dropdown.value = None
        self.inform_plugins('on_marker_sets_changed')

    def display(self):
        """Display the main UI."""
        # Initialize the mask ID annotation
        self.mask_id_annotation = self.image_display.ax.annotate(
            "",
            xy=(0, 0),
            xycoords="data",
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=12,
            color='yellow',
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="yellow", lw=1),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            visible=False
        )

        # Display the main UI
        display_ui(self)
        self.after_all_plugins_loaded()

    def refresh_bottom_panel(self, ordering=None):
        debug_enabled = getattr(self, '_debug', False)
        if debug_enabled:
            print("Refreshing bottom panel...")
        update_wide_plugin_panel(self, ordering)

    def save_widget_states(self, file_path):
        """Save the current state of all widgets to a JSON file."""
        ALLOWED_WIDGETS = {'marker_set_dropdown'}
        state = {}

        # Iterate over attributes of self.ui_component
        ui_attrs = vars(self.ui_component)
        for attr_name, attr_value in ui_attrs.items():
            if attr_name in UICOMPNENTS_SKIP and attr_name not in ALLOWED_WIDGETS:
                continue
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
        state['marker_sets'] = self.marker_sets
        
        if self.masks_available:
            state['mask_names'] = self.mask_names

        state['control_sections_selected_index'] = getattr(
            self.ui_component.control_sections, 'selected_index', None
        )

        # Save the state to a JSON file
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4)
        
        if self._debug:
            print(f"Widget states saved to {file_path}")

    def load_widget_states(self, file_path):
        """Load the state of all widgets from a JSON file."""
        # When the json file does not exist, do nothing
        if not os.path.exists(file_path):
            # this is the first time the viewer is run
            return

        with open(file_path, 'r') as f:
            state = json.load(f)

        selected_section_index = state.get('control_sections_selected_index')

        # Iterate over the saved state and restore widget values
        for attr_name, value in state.items():
            if attr_name == 'control_sections_selected_index':
                continue
            # Skip attributes that are in the UICOMPNENTS_SKIP list
            if attr_name in UICOMPNENTS_SKIP:
                continue
            # Check if the ui_component has an attribute with the name attr_name
            elif hasattr(self.ui_component, attr_name):
                # Get the attribute from the ui_component
                attr = getattr(self.ui_component, attr_name)
                # Check if the attribute is an instance of Widget
                if isinstance(attr, Widget):
                    # If the widget has a 'value' attribute, set it to the saved value
                    if hasattr(attr, 'value'):
                        attr.value = value
                    else:
                        # Skip widgets without a 'value' attribute
                        pass
                # Check if the attribute is a dictionary
                elif isinstance(attr, dict):
                    # Iterate over the dictionary items
                    for key, widget_value in value.items():
                        # If the key exists in the attribute dictionary
                        if key in attr:
                            # Get the widget from the dictionary
                            widget = attr[key]
                            # If the widget is an instance of Widget and has a 'value' attribute, set it to the saved value
                            if isinstance(widget, Widget) and hasattr(widget, 'value'):
                                widget.value = widget_value
                else:
                    # Handle other attribute types if necessary
                    pass
            # Special handling for 'marker_sets' attribute
            elif attr_name == 'marker_sets':
                self.marker_sets = value
            # Special handling for 'mask_names' attribute
            elif attr_name == 'mask_names':
                self.mask_names = value

        # Update the marker set dropdown and controls
        self.update_marker_set_dropdown()
        self.update_controls(None)
        self.update_display(self.current_downsample_factor)
        self.update_keys(None)

        accordion = getattr(self.ui_component, 'control_sections', None)
        if accordion is not None:
            if selected_section_index is None:
                accordion.selected_index = None
            elif (
                isinstance(selected_section_index, int)
                and 0 <= selected_section_index < len(accordion.children)
            ):
                accordion.selected_index = selected_section_index

        if self._debug:
            print(f"Widget states loaded from {file_path}")
        self.inform_plugins('on_marker_sets_changed')
        self.inform_plugins('refresh_roi_table')
    
    def load_status_images(self):
        self._status_image["processing"] = load_asset_bytes("loading.gif")
        self._status_image["ready"] = load_asset_bytes("ready.png")

    def load_cell_table_from_path(self, file_path):
        """Load the cell table from a CSV file and convert columns with integer values (allowing NA) to integer."""
        df = pd.read_csv(file_path)
        
        for col in df.columns:
            # Check if column is numeric and float type (NA causes float type)
            if pd.api.types.is_float_dtype(df[col]):
                # Convert non-null values to int and compare to the original values
                if df[col].dropna().apply(float.is_integer).all():
                    # Use nullable integer dtype so NAs are preserved
                    df[col] = df[col].astype("Int64")
        
        self.cell_table = df

    def set_cell_table(self, cell_table):
        """Load the cell table from a DataFrame."""
        self.cell_table = cell_table

    def marker2display(self):
        """
        Generate a dictionary mapping selected markers to their RGB color vectors.

        Returns:
            dict: A dictionary where keys are marker names and values are RGB tuples.
        """
        marker_to_color = {}
        for marker in self.ui_component.channel_selector.value:
            color_value = self.ui_component.color_controls[marker].value  # Can be hex code or color name
            try:
                # Convert color to RGB tuple (values between 0 and 1)
                color_rgb = to_rgb(color_value)
                marker_to_color[marker] = color_rgb
            except ValueError as e:
                print(f"Invalid color value for marker '{marker}': {color_value}")
                # Handle the error as needed (e.g., default to black or raise an exception)
                marker_to_color[marker] = (0, 0, 0)  # Default to black
        return marker_to_color
    
    def get_color_range(self):
        """
        Generate a dictionary mapping selected markers to their color ranges.

        Returns:
            dict: A dictionary where keys are marker names and values are ranges [min max].
        """
        color_ranges = {}
        for marker in self.ui_component.channel_selector.value:
            color_range = [self.ui_component.contrast_min_controls[marker].value,
                            self.ui_component.contrast_max_controls[marker].value]
            color_ranges[marker] = color_range
        return color_ranges
    
    def on_widget_value_change(self, change):
        """Callback function to handle widget value changes."""
        if self.initialized:
            os.makedirs(os.path.join(self.base_folder, '.UELer'), exist_ok=True)
            widget_states_path = os.path.join(self.base_folder, '.UELer', 'widget_states.json')
            self.save_widget_states(widget_states_path)
    
    def on_save_marker_set_click(self, button):
        return self.on_widget_value_change(button)

    def _on_control_sections_change(self, change):
        self.on_widget_value_change(change)

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
            if attr_name in UICOMPNENTS_SKIP:
                continue
            observe_widget(attr_value)

        if hasattr(self.ui_component, "control_sections") and hasattr(self.ui_component.control_sections, "observe"):
            self.ui_component.control_sections.observe(self._on_control_sections_change, names="selected_index")

        # Add observer for save_marker_set_button
        self.ui_component.save_marker_set_button.on_click(self.on_save_marker_set_click)

    def setup_attr_observers(self):
        def _setup(container):
            for attr in vars(container).values():
                if hasattr(attr, 'setup_observe'):
                    attr.setup_observe()

        _setup(self.SidePlots)
        _setup(self.BottomPlots)

    def export_fovs_batch(self, marker_set_name, output_dir=None, fovs=None, surfix = None, file_format='png', 
                          downsample_factor=None, dpi=300, figure_size=None, show_progress=True):
        """
        Export multiple FOVs using a specified marker set.
        
        Parameters:
        -----------
        marker_set_name : str
            Name of the marker set to use for visualization.
        output_dir : str, optional
            Directory to save the exported images. If None, creates a subdirectory in the base folder.
        fovs : list, optional
            List of FOV names to export. If None, exports all available FOVs.
        file_format : str, optional
            File format for the exported images ('png', 'jpg', 'tif', 'pdf'). Default is 'png'.
        downsample_factor : int, optional
            Downsample factor to use. If None, uses the current downsample factor.
        dpi : int, optional
            Resolution for the exported images. Default is 300.
        figure_size : tuple, optional
            Size of the figure in inches (width, height). If None, uses the current figure size.
        show_progress : bool, optional
            Whether to show progress information. Default is True.
            
        Returns:
        --------
        dict
            Dictionary mapping FOV names to export status (True for success, error message for failure).
        """
        import time
        from IPython.display import clear_output
        
        # Validate marker set
        if marker_set_name not in self.marker_sets:
            raise ValueError(f"Marker set '{marker_set_name}' not found.")
        
        # Set output directory
        if output_dir is None:
            output_dir = os.path.join(self.base_folder, f"exported_{marker_set_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine FOVs to export
        if fovs is None:
            fovs = self.available_fovs
        elif isinstance(fovs, str):
            fovs = [fovs]  # Convert single FOV string to list
        
        # Set downsample factor
        if downsample_factor is None:
            downsample_factor = self.current_downsample_factor
        
        # Save current state
        current_state = {
            'fov': self.ui_component.image_selector.value,
            'channels': self.ui_component.channel_selector.value,
            'downsample_factor': self.current_downsample_factor,
            'channel_settings': {},
            'mask_settings': {},
        }
        
        # Save channel settings
        for ch in current_state['channels']:
            if ch in self.ui_component.color_controls:
                current_state['channel_settings'][ch] = {
                    'color': self.ui_component.color_controls[ch].value,
                    'contrast_min': self.ui_component.contrast_min_controls[ch].value,
                    'contrast_max': self.ui_component.contrast_max_controls[ch].value
                }
        
        # Save mask settings if masks are available
        if self.masks_available:
            for mask_name in self.mask_names:
                if mask_name in self.ui_component.mask_display_controls:
                    current_state['mask_settings'][mask_name] = {
                        'visible': self.ui_component.mask_display_controls[mask_name].value,
                        'color': self.ui_component.mask_color_controls[mask_name].value if mask_name in self.ui_component.mask_color_controls else None
                    }
        
        # Store original figure size
        original_figsize = self.image_display.fig.get_size_inches()
        
        # Results dictionary
        results = {}
        total_fovs = len(fovs)
        
        try:
            # Load marker set
            marker_set = self.marker_sets[marker_set_name]
            selected_channels = marker_set['selected_channels']
            channel_settings = marker_set['channel_settings']
            
            # Update channel selector (needed for render_image to use correct settings)
            self.ui_component.channel_selector.value = tuple(selected_channels)
            
            # Update controls
            self.update_controls(None)
            
            # Apply channel settings
            for ch in selected_channels:
                if ch in self.ui_component.color_controls and ch in channel_settings:
                    self.ui_component.color_controls[ch].value = channel_settings[ch]['color']
                    self.ui_component.contrast_min_controls[ch].value = channel_settings[ch]['contrast_min']
                    self.ui_component.contrast_max_controls[ch].value = channel_settings[ch]['contrast_max']
            
            # Set figure size if specified
            if figure_size is not None:
                self.image_display.fig.set_size_inches(figure_size)
            
            # Process each FOV
            start_time = time.time()
            current_fov = self.ui_component.image_selector.value  # Remember current FOV
            
            for i, fov in enumerate(fovs):
                try:
                    if show_progress:
                        clear_output(wait=True)
                        print(f"Exporting FOV {i+1}/{total_fovs}: {fov}")
                        print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
                    
                    # Load the FOV data without changing the UI selection
                    self.load_fov(fov, selected_channels)
                    
                    # Get image dimensions for this FOV
                    fov_images = self.image_cache[fov]
                    
                    # Verify that all channels exist and were loaded successfully
                    missing_channels = []
                    for ch in selected_channels:
                        if ch not in fov_images or fov_images[ch] is None:
                            missing_channels.append(ch)
                    
                    if missing_channels:
                        raise ValueError(f"Missing or failed to load channels: {', '.join(missing_channels)}")
                    
                    # Now safely get first image for dimensions
                    first_channel = None
                    for ch in selected_channels:
                        if fov_images[ch] is not None:
                            first_channel = ch
                            break
                            
                    if first_channel is None:
                        raise ValueError("No valid channels found in this FOV")
                        
                    first_channel_image = fov_images[first_channel]
                    height, width = first_channel_image.shape
                    
                    # Temporarily set the internal current FOV for render_image
                    temp_current_fov = self.ui_component.image_selector.value
                    self.ui_component.image_selector.value = fov
                    
                    # Calculate the view limits
                    xmin, xmax = 0, width
                    ymin, ymax = 0, height
                    
                    # Ensure downsample factor is appropriate for image size
                    downsample_factor_adjusted = min(downsample_factor, min(width//2, height//2))
                    if downsample_factor_adjusted < 1:
                        downsample_factor_adjusted = 1
                    
                    if show_progress:
                        print(f"Using downsample factor: {downsample_factor_adjusted}")
                    
                    xmin_ds = xmin // downsample_factor_adjusted 
                    xmax_ds = xmin_ds + (xmax - xmin) // downsample_factor_adjusted
                    ymin_ds = ymin // downsample_factor_adjusted
                    ymax_ds = ymin_ds + (ymax - ymin) // downsample_factor_adjusted
                    
                    # Directly render the image
                    xym = (xmin, xmax, ymin, ymax)
                    xym_ds = (xmin_ds, xmax_ds, ymin_ds, ymax_ds)
                    
                    if show_progress:
                        print(f"Rendering image: size {xmax-xmin}x{ymax-ymin} → {xmax_ds-xmin_ds}x{ymax_ds-ymin_ds}")
                    
                    combined = self.render_image(selected_channels, downsample_factor_adjusted, xym, xym_ds)

                    # Convert to uint8 for reliable saving
                    img_array = np.asarray(combined, dtype=np.float32)
                    img_array = np.clip(img_array, 0.0, 1.0)
                    img_array = (img_array * 255).astype(np.uint8)
                    
                    # Check image dimensions
                    if img_array.shape[0] < 2 or img_array.shape[1] < 2:
                        raise ValueError(f"Image too small: {img_array.shape}")
                    
                    # Save the combined image
                    if surfix is not None:
                        output_path = os.path.join(output_dir, f"{fov}_{surfix}.{file_format}")
                    else:
                        output_path = os.path.join(output_dir, f"{fov}.{file_format}")
                    imsave(output_path, img_array)
                    
                    results[fov] = True
                    
                    if show_progress:
                        print(f"Successfully exported {fov} to {output_path}")
                    
                    # Restore the original UI selection
                    self.ui_component.image_selector.value = temp_current_fov
                    
                except Exception as e:
                    results[fov] = str(e)
                    if show_progress:
                        print(f"Error exporting {fov}: {e}")
                        # Print more detailed error information
                        import traceback
                        traceback.print_exc()
        
        finally:
            # Restore original figure size
            self.image_display.fig.set_size_inches(original_figsize)
            
            # Restore original state
            self.ui_component.image_selector.value = current_state['fov']
            self.ui_component.channel_selector.value = current_state['channels']
            self.update_controls(None)
            
            # Restore channel settings
            for ch, settings in current_state['channel_settings'].items():
                if ch in self.ui_component.color_controls:
                    self.ui_component.color_controls[ch].value = settings['color']
                    self.ui_component.contrast_min_controls[ch].value = settings['contrast_min']
                    self.ui_component.contrast_max_controls[ch].value = settings['contrast_max']
            
            # Restore mask settings if masks are available
            if self.masks_available:
                for mask_name, settings in current_state['mask_settings'].items():
                    if mask_name in self.ui_component.mask_display_controls:
                        self.ui_component.mask_display_controls[mask_name].value = settings['visible']
                    if mask_name in self.ui_component.mask_color_controls and settings['color'] is not None:
                        self.ui_component.mask_color_controls[mask_name].value = settings['color']
            
            # Restore display with original downsample factor
            self.update_display(current_state['downsample_factor'])
            
            if show_progress:
                clear_output(wait=True)
                successful = sum(1 for result in results.values() if result is True)
                print(f"Export complete: {successful}/{total_fovs} FOVs exported successfully")
                if successful < total_fovs:
                    print("The following FOVs had errors:")
                    for fov, result in results.items():
                        if result is not True:
                            print(f"  {fov}: {result}")
        
        return results