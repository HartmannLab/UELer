# viewer.py

import logging
import math
import os
import glob
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.text import Annotation
import cv2
from collections import OrderedDict
from types import SimpleNamespace
from ipywidgets import IntText, Output, Dropdown, FloatSlider, Checkbox, Button, VBox, HBox, Layout, Widget
from IPython.display import display
import pandas as pd
# Import modules
from ueler.constants import PREDEFINED_COLORS, DOWNSAMPLE_FACTORS, UICOMPNENTS_SKIP
from ueler.data_loader import (
    load_annotations_for_fov,
    load_channel_struct_fov,
    load_images_for_fov,
    load_masks_for_fov,
    load_one_channel_fov,
    merge_channel_max,
    OMEFovWrapper,
)
from ueler.image_utils import (
    calculate_downsample_factor,
    generate_edges,
    get_axis_limits_with_padding,
    select_downsample_factor,
)
from ueler.viewer.images import load_asset_bytes
from ueler.viewer.scale_bar import compute_scale_bar_spec, effective_pixel_size_nm
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
    normalize_hex_color,
)
from ueler.viewer.mask_color_overlay import apply_registry_colors, collect_mask_regions
from ueler.rendering import (
    AnnotationOverlaySnapshot,
    AnnotationRenderSettings,
    ChannelRenderSettings,
    MaskOverlaySnapshot,
    MaskRenderSettings,
    OverlaySnapshot,
    render_fov_to_array,
)
from ueler.export.job import Job, JobItem
# viewer/main_viewer.py

from ueler.viewer.palette_store import (
    PaletteStoreError,
    load_registry as load_palette_registry,
    read_palette_file,
    resolve_palette_path,
    save_registry as save_palette_registry,
    write_palette_file,
)
from ueler.viewer.map_descriptor_loader import MapDescriptorLoader
from .tooltip_utils import resolve_cell_record
from .virtual_map_layer import MapPixelHit, VirtualMapLayer

from weakref import WeakKeyDictionary

from skimage.io import imsave
from skimage.segmentation import find_boundaries
from ueler.rendering.engine import scale_outline_thickness, thicken_outline
# from dask.distributed import LocalCluster, Client

# # Create a LocalCluster with a memory limit of 4 GB per worker
# cluster = LocalCluster(memory_limit='8GB')

# # Connect to the cluster
# client = Client(cluster)

# main_viewer.py

__all__ = ["ImageMaskViewer"]


logger = logging.getLogger(__name__)


ANNOTATION_PALETTE_FILE_SUFFIX = ".pixelannotations.json"
ANNOTATION_PALETTE_VERSION = "1.0.0"
ANNOTATION_REGISTRY_FILENAME = "pixel_annotation_sets_index.json"
ANNOTATION_PALETTE_FOLDER_NAME = "pixel_annotation_palettes"

MAP_DESCRIPTOR_RELATIVE_PATH = Path(".UELer") / "maps"
_MAP_MODE_FLAG = os.getenv("ENABLE_MAP_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MapCellPosition:
    """Stitched-map coordinates for a single cell."""

    x_px: float
    y_px: float
    pixel_scale: float


@dataclass(frozen=True)
class MapPixelLocalization:
    """Reverse map lookup describing the owning FOV for a map pixel."""

    map_id: str
    fov_name: str
    local_x_px: float
    local_y_px: float
    pixel_scale: float


@dataclass(frozen=True)
class MaskHit:
    """Mask lookup result for hover/click interactions."""

    fov_name: str
    mask_name: str
    mask_id: int
    local_x_px: float
    local_y_px: float
    map_id: Optional[str] = None
    cell_record: Optional[pd.Series] = None


def _unique_annotation_values(array):
    """Return the sorted unique values from an annotation array."""
    try:
        data = array.compute()
    except AttributeError:
        data = np.asarray(array)
    except Exception:
        return np.array([], dtype=np.int32)

    try:
        values = np.unique(np.asarray(data))
    except Exception:
        return np.array([], dtype=np.int32)

    finite_values = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.inexact) else values
    if finite_values.size == 0:
        return np.array([], dtype=np.int32)
    return finite_values.astype(np.int32, copy=False)

class ImageMaskViewer:
    def __init__(self, base_folder, masks_folder=None, annotations_folder=None):
        self.initialized = False
        self.base_folder = base_folder
        self.masks_folder = masks_folder
        if annotations_folder is None:
            candidate = os.path.join(self.base_folder, "annotations")
            self.annotations_folder = candidate if os.path.isdir(candidate) else None
        else:
            self.annotations_folder = annotations_folder

        # Add this flag
        self.masks_available = self.masks_folder is not None
        self.annotations_available = self.annotations_folder is not None

        # This is the surfix for the mask to be linked
        self.mask_key = None

        # Initialize the cell table
        self.cell_table = None

        # Map mode scaffolding (feature-flagged)
        self._map_mode_enabled = _MAP_MODE_FLAG
        self._map_mode_messages = []
        self._map_descriptors = {}
        self._map_fov_lookup: Dict[str, str] = {}
        self._map_tile_cache_capacity = 6
        self._map_tile_cache = OrderedDict()
        self._map_layers = {}
        self._active_map_id = None
        self._map_mode_active = False
        self._visible_map_fovs = ()
        self._map_pixel_size_nm = None
        self._map_canvas_size = None
        self._last_viewport_px: Optional[Tuple[float, float, float, float, int]] = None

        # Specifiy the keys for the x, y and label columns in the cell table
        self.x_key = "X"
        self.y_key = "Y"
        self.label_key = "label"
        self.mask_key = "mask"
        self.fov_key = "fov"

        self._debug = False

        # Initialize variables
        self._fov_mode = "folder"
        
        # Get potential FOV directories (excluding hidden ones like .UELer)
        subdirs = [d for d in os.listdir(self.base_folder) 
                   if os.path.isdir(os.path.join(self.base_folder, d)) 
                   and not d.startswith(".")]
        
        # Check which of these are actually valid FOVs
        valid_fov_folders = [fov for fov in subdirs if self._has_tiff_files(fov)]
        valid_fov_folders.sort()

        ome_files = glob.glob(os.path.join(self.base_folder, "*.ome.tif")) + glob.glob(os.path.join(self.base_folder, "*.ome.tiff"))
        
        if valid_fov_folders:
             self._fov_mode = "folder"
             self.available_fovs = valid_fov_folders
        elif ome_files:
             self._fov_mode = "ome-tiff"
             self.available_fovs = []
             for f in ome_files:
                 name = os.path.basename(f)
                 if name.endswith(".ome.tif"):
                     name = name[:-8]
                 elif name.endswith(".ome.tiff"):
                     name = name[:-9]
                 else:
                     name = os.path.splitext(name)[0]
                 self.available_fovs.append(name)
             self.available_fovs.sort()
        else:
             self.available_fovs = []

        if not self.available_fovs:
            raise ValueError("No FOVs found in the base folder.")

        if self._map_mode_enabled:
            self._initialize_map_descriptors()

        # Initialize caches and other variables
        self.max_cache_size = 100

        # Initialize current downsample factor
        self.downsample_factors = DOWNSAMPLE_FACTORS
        self.current_downsample_factor = 8  # Initial downsample factor

        self.image_cache = OrderedDict()
        self.mask_cache = OrderedDict()
        self.edge_masks_cache = {}
        self.label_masks_cache = {}
        self.annotation_cache = OrderedDict()
        self.annotation_label_cache = {}
        self.annotation_names_set = set()
        self.annotation_names = []
        self.annotation_palettes = {}
        self.annotation_class_labels = {}
        self.annotation_class_ids = {}
        self.annotation_palette_folder: Optional[Path] = None
        self.annotation_palette_registry_records: Dict[str, Dict[str, str]] = {}
        self.active_annotation_palette_set_name: Optional[str] = None
        self.annotation_display_enabled = False
        self.active_annotation_name: Optional[str] = None
        self.annotation_overlay_mode = "combined"
        self.annotation_label_display_mode = "id"
        self.annotation_overlay_alpha = 0.5
        self.mask_outline_thickness = 1
        self._suspend_mask_outline_sync = False
        self._control_section_titles = []
        self.pixel_size_nm = 390.0
        self.scale_bar_fraction = 0.1
        self._suspend_pixel_size_observer = False

        self.channel_names_set = set()
        self.channel_max_values = {}
        self.mask_names_set = set()
        self.predefined_colors = PREDEFINED_COLORS
        self._status_image = {}

        # ROI management
        self.roi_manager = ROIManager(self.base_folder)
        self.roi_plugin: Optional[ROIManagerPlugin] = None
        self.active_marker_set_name: Optional[str] = None

        # Initialize marker sets
        self.marker_sets = {}

        # Initialize current_label_masks
        self.current_label_masks = {}
        self.full_resolution_label_masks = {}

        # Load the status images
        self.load_status_images()

        # Load the first FOV and set image dimensions
        initial_fov = self.available_fovs[0]
        self.load_fov(initial_fov)
        fov_images = self.image_cache[initial_fov]
        if isinstance(fov_images, OMEFovWrapper):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[viewer] Using OME metadata for %s dimensions; reported shape=%s",
                    initial_fov,
                    fov_images.shape,
                )
            self.height, self.width = fov_images.shape  # full-res Y/X from OME metadata
        else:
            try:
                first_channel_name, first_channel_image = next(iter(fov_images.items()))
            except StopIteration as exc:
                raise ValueError(f"No channel data found for FOV '{initial_fov}'.") from exc
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[viewer] Sampling channel %s for %s dimensions; this loads data with shape=%s",
                    first_channel_name,
                    initial_fov,
                    getattr(first_channel_image, "shape", None),
                )
            self.height, self.width = first_channel_image.shape

        # Calculate the downsample factor based on image size
        initial_factor = select_downsample_factor(
            self.width,
            self.height,
            max_dimension=512,
            allowed_factors=self.downsample_factors,
        )
        self.on_downsample_factor_changed(initial_factor)

        # Initialize image output and image display
        self.image_output = Output()
        with self.image_output:
            self.image_display = ImageDisplay(self.width, self.height)
            self.image_display.main_viewer = self
            plt.show()

        # Initialize widgets
        create_widgets(self)
        self._refresh_map_controls()
        pixel_widget = getattr(self.ui_component, "pixel_size_inttext", None)
        if pixel_widget is not None:
            try:
                self.pixel_size_nm = max(1.0, float(pixel_widget.value))
            except (TypeError, ValueError):
                self.pixel_size_nm = 390.0
                self._suspend_pixel_size_observer = True
                try:
                    pixel_widget.value = int(self.pixel_size_nm)
                finally:
                    self._suspend_pixel_size_observer = False
        mask_slider = getattr(self.ui_component, "mask_outline_thickness_slider", None)
        if mask_slider is not None:
            mask_slider.observe(self._on_mask_outline_slider_change, names="value")
            self._set_mask_outline_slider_value(self.mask_outline_thickness)
        self.annotation_palette_editor = AnnotationPaletteEditor(
            self.apply_annotation_palette_changes,
            self.close_annotation_editor,
        )
        self.ui_component.annotation_editor_host.children = (self.annotation_palette_editor,)
        self.ui_component.annotation_editor_host.layout.display = ""
        self._initialize_annotation_palette_manager()
        self._refresh_annotation_control_states()

        # Setup widget observers
        self.setup_widget_observers()

        # Setup event connections
        self.setup_event_connections()

        # Initial setup
        self.update_marker_set_dropdown()
        self.update_controls(None)
        self.on_image_change(None)
        self.update_display(self.current_downsample_factor)

        self.SidePlots = SimpleNamespace()
        self.BottomPlots = SimpleNamespace()

        self.load_widget_states(os.path.join(self.base_folder, ".UELer", 'widget_states.json'))

        # Setup attribute observers
        self.setup_attr_observers()

        self.initialized = True
    
    def _has_tiff_files(self, fov_name):
        """Check if a directory contains .tif or .tiff files, including in 'rescaled' subdirectory."""
        import glob
        
        fov_path = os.path.join(self.base_folder, fov_name)
        
        # Check for 'rescaled' subfolder first
        rescaled_path = os.path.join(fov_path, 'rescaled')
        if os.path.isdir(rescaled_path):
            channel_folder = rescaled_path
        else:
            channel_folder = fov_path
        
        # Get list of TIFF files in the channel folder
        tiff_files = glob.glob(os.path.join(channel_folder, '*.tiff'))
        tiff_files += glob.glob(os.path.join(channel_folder, '*.tif'))
        
        return bool(tiff_files)

    def _initialize_map_descriptors(self) -> None:
        """Load map descriptors when map mode is enabled."""

        descriptor_root = Path(self.base_folder) / MAP_DESCRIPTOR_RELATIVE_PATH
        loader = MapDescriptorLoader()
        result = loader.load_from_directory(descriptor_root)
        self._map_descriptors = result.slides
        self._map_fov_lookup.clear()
        self._map_layers.clear()
        self._map_tile_cache.clear()
        for map_id, descriptor in self._map_descriptors.items():
            for spec in getattr(descriptor, "fovs", ()):  # type: ignore[attr-defined]
                try:
                    name = str(getattr(spec, "name"))
                except Exception:
                    continue
                if not name:
                    continue
                if name in self._map_fov_lookup and self._map_fov_lookup[name] != map_id:
                    self._record_map_mode_warning(
                        f"FOV '{name}' appears in multiple map descriptors; using '{self._map_fov_lookup[name]}'"
                    )
                    continue
                self._map_fov_lookup[name] = map_id
        if hasattr(self, "ui_component"):
            self._refresh_map_controls()

        for message in result.warnings:
            self._record_map_mode_warning(message)
        for message in result.errors:
            self._record_map_mode_warning(message)

    def _record_map_mode_warning(self, message: str) -> None:
        text = f"[Map mode] {message}"
        if text in self._map_mode_messages:
            return
        self._map_mode_messages.append(text)
        logger.warning(text)
        try:
            print(f"Map mode warning: {message}")
        except Exception:  # pragma: no cover - console output best effort
            pass

    def _refresh_map_controls(self) -> None:
        ui = getattr(self, "ui_component", None)
        if ui is None:
            return

        toggle = getattr(ui, "map_mode_toggle", None)
        selector = getattr(ui, "map_selector", None)
        summary = getattr(ui, "map_summary", None)
        container = getattr(ui, "map_controls_box", None)

        has_maps = bool(self._map_descriptors)
        if container is not None:
            container.layout.display = "" if has_maps else "none"

        if toggle is not None:
            toggle.disabled = not (self._map_mode_enabled and has_maps)
            if not has_maps:
                toggle.value = False

        options = []
        if has_maps:
            for map_id, descriptor in sorted(self._map_descriptors.items()):
                tiles = len(descriptor.fovs)
                label = f"{map_id} ({tiles} tile{'s' if tiles != 1 else ''})"
                options.append((label, map_id))

        selected_value = None
        if selector is not None:
            selector.options = options
            if options:
                values = [value for _label, value in options]
                if self._active_map_id in values:
                    selected_value = self._active_map_id
                else:
                    selected_value = values[0]
                if selector.value != selected_value:
                    selector.value = selected_value
            else:
                selector.value = None
            selector.disabled = not (self._map_mode_active or (has_maps and bool(options) and toggle is not None and toggle.value))
            selected_value = selector.value

        if summary is not None:
            summary.value = self._compose_map_summary(selected_value)

    def _compose_map_summary(self, map_id: Optional[str]) -> str:
        if not map_id:
            return ""
        descriptor = self._map_descriptors.get(map_id)
        if descriptor is None:
            return ""
        try:
            layer = self._get_map_layer(map_id)
        except Exception:
            tiles = len(descriptor.fovs)
            return f"<span style='font-size:90%'>Tiles: {tiles}</span>"

        width_px, height_px, base_pixel = self._compute_map_canvas_dimensions(layer)
        bounds = layer.map_bounds()
        width_um = max(0.0, bounds[1] - bounds[0])
        height_um = max(0.0, bounds[3] - bounds[2])
        tiles = len(descriptor.fovs)
        return (
            "<span style='font-size:90%'>Tiles: {tiles} | Canvas: {width_px}x{height_px}px | "
            "Span: {width_um:.0f}x{height_um:.0f} um</span>"
        ).format(
            tiles=tiles,
            width_px=width_px,
            height_px=height_px,
            width_um=width_um,
            height_um=height_um,
        )

    def _compute_map_canvas_dimensions(self, layer) -> Tuple[int, int, float]:
        bounds = layer.map_bounds()
        base_pixel = max(1e-9, float(layer.base_pixel_size_um()))
        width_um = max(0.0, bounds[1] - bounds[0])
        height_um = max(0.0, bounds[3] - bounds[2])
        width_px = max(1, int(math.ceil(width_um / base_pixel)))
        height_px = max(1, int(math.ceil(height_um / base_pixel)))
        return width_px, height_px, base_pixel

    def _select_default_map_id(self) -> Optional[str]:
        if not self._map_descriptors:
            return None
        return sorted(self._map_descriptors.keys())[0]

    def _map_id_for_fov(self, fov_name: Optional[str]) -> Optional[str]:
        if not fov_name:
            return None
        return self._map_fov_lookup.get(str(fov_name))

    def _set_map_canvas_dimensions(self, width_px: int, height_px: int) -> None:
        self.width = width_px
        self.height = height_px
        display = getattr(self, "image_display", None)
        if display is None:
            return
        display.width = width_px
        display.height = height_px
        display.ax.set_xlim(0, width_px)
        display.ax.set_ylim(height_px, 0)
        try:
            display.img_display.set_extent((0, width_px, height_px, 0))
            placeholder = np.zeros((1, 1, 3), dtype=np.float32)
            display.img_display.set_data(placeholder)
            display.combined = placeholder
        except Exception:
            pass
        self._sync_navigation_home_view()

    def _sync_canvas_to_current_fov(self) -> None:
        selector = getattr(self.ui_component, "image_selector", None)
        if selector is None:
            return
        fov_name = selector.value
        if not fov_name:
            return
        channels = tuple(getattr(self.ui_component.channel_selector, "value", ()))
        try:
            self.load_fov(fov_name, channels)
        except Exception:
            pass
        fov_images = self.image_cache.get(fov_name, {})
        first_channel = next(iter(fov_images.values()), None)
        if first_channel is None:
            return
        height, width = first_channel.shape
        self._set_map_canvas_dimensions(width, height)

    def _sync_navigation_home_view(self) -> None:
        display = getattr(self, "image_display", None)
        if display is None:
            return
        ax = getattr(display, "ax", None)
        if ax is None:
            return
        fig = getattr(display, "fig", None)
        canvas = getattr(fig, "canvas", None) if fig is not None else None
        toolbar = getattr(canvas, "toolbar", None) if canvas is not None else None
        if toolbar is None:
            return
        nav_stack = getattr(toolbar, "_nav_stack", None)
        if nav_stack is None and hasattr(toolbar, "push_current"):
            try:
                toolbar.push_current()
            except Exception:
                nav_stack = None
            else:
                nav_stack = getattr(toolbar, "_nav_stack", None)
        elements = getattr(nav_stack, "_elements", None) if nav_stack is not None else None
        if (not elements) and hasattr(toolbar, "push_current"):
            try:
                toolbar.push_current()
            except Exception:
                elements = None
            else:
                nav_stack = getattr(toolbar, "_nav_stack", None)
                elements = getattr(nav_stack, "_elements", None) if nav_stack is not None else None
        if not elements:
            return
        try:
            nav_entry = elements[0]
        except Exception:
            return
        if not hasattr(nav_entry, "get"):
            return
        record = nav_entry.get(ax)
        if record is None:
            view_limits = {"xlim": ax.get_xlim(), "ylim": ax.get_ylim()}
            nav_entry[ax] = (view_limits, ())
            return
        existing_view, existing_bboxes = record
        new_view = dict(existing_view)
        new_view["xlim"] = ax.get_xlim()
        new_view["ylim"] = ax.get_ylim()
        nav_entry[ax] = (new_view, existing_bboxes)

    def on_map_mode_toggle(self, change):
        new_value = bool(change.get("new"))
        if new_value == self._map_mode_active:
            return
        if new_value:
            map_id = getattr(self.ui_component, "map_selector", None)
            candidate = map_id.value if map_id is not None else None
            if not candidate:
                candidate = self._select_default_map_id()
                if map_id is not None:
                    map_id.value = candidate
            if candidate is None:
                toggle = getattr(self.ui_component, "map_mode_toggle", None)
                if toggle is not None:
                    toggle.value = False
                return
            self._activate_map_mode(candidate)
            self._refresh_map_controls()
        else:
            self._deactivate_map_mode()
            self._refresh_map_controls()

        self.update_display(self.current_downsample_factor)
        self.inform_plugins('on_fov_change')

    def on_map_selector_change(self, change):
        new_value = change.get("new") if isinstance(change, dict) else None
        if not new_value:
            return
        if new_value not in self._map_descriptors:
            return
        self._active_map_id = new_value
        if self._map_mode_active:
            self._activate_map_mode(new_value)
            self._refresh_map_controls()
            self.update_display(self.current_downsample_factor)
            self.inform_plugins('on_fov_change')
        else:
            self._refresh_map_controls()

    def _activate_map_mode(self, map_id: str) -> None:
        if map_id not in self._map_descriptors:
            raise KeyError(map_id)
        layer = self._get_map_layer(map_id)
        width_px, height_px, base_pixel = self._compute_map_canvas_dimensions(layer)
        self._map_mode_active = True
        self._active_map_id = map_id
        self._map_pixel_size_nm = base_pixel * 1000.0
        self._map_canvas_size = (width_px, height_px)
        selector = getattr(self.ui_component, "map_selector", None)
        if selector is not None and selector.value != map_id:
            selector.value = map_id
        image_selector = getattr(self.ui_component, "image_selector", None)
        if image_selector is not None:
            image_selector.disabled = True
        if selector is not None:
            selector.disabled = False
        self._set_map_canvas_dimensions(width_px, height_px)
        try:
            factor = select_downsample_factor(
                width_px,
                height_px,
                allowed_factors=getattr(self, "downsample_factors", DOWNSAMPLE_FACTORS),
                minimum=1,
            )
            self.on_downsample_factor_changed(factor)
        except Exception:
            pass

    def _deactivate_map_mode(self) -> None:
        self._map_mode_active = False
        self._map_pixel_size_nm = None
        self._map_canvas_size = None
        self._visible_map_fovs = ()
        image_selector = getattr(self.ui_component, "image_selector", None)
        if image_selector is not None:
            image_selector.disabled = False
        map_selector = getattr(self.ui_component, "map_selector", None)
        if map_selector is not None:
            map_selector.disabled = True
        self._sync_canvas_to_current_fov()

    def _render_map_view(
        self,
        selected_channels: Sequence[str],
        downsample_factor: int,
        viewport_pixels: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        if not self._active_map_id:
            return np.zeros((1, 1, 3), dtype=np.float32), ()
        layer = self._get_map_layer(self._active_map_id)
        base_pixel = layer.base_pixel_size_um()
        bounds = layer.map_bounds()
        try:
            bounds_min_x = float(bounds[0])
            bounds_min_y = float(bounds[2])
        except (TypeError, ValueError, IndexError):
            bounds_min_x = 0.0
            bounds_min_y = 0.0

        xmin, xmax, ymin, ymax = viewport_pixels
        xmin_um = bounds_min_x + xmin * base_pixel
        xmax_um = bounds_min_x + xmax * base_pixel
        ymin_um = bounds_min_y + ymin * base_pixel
        ymax_um = bounds_min_y + ymax * base_pixel
        layer.set_viewport(
            xmin_um,
            xmax_um,
            ymin_um,
            ymax_um,
            downsample_factor=downsample_factor,
        )
        combined = layer.render(tuple(selected_channels))
        return combined, layer.last_visible_fovs()
    def _get_map_layer(self, map_id: str) -> VirtualMapLayer:
        if not self._map_mode_enabled:
            raise RuntimeError("Map mode is disabled.")
        descriptor = self._map_descriptors.get(map_id)
        if descriptor is None:
            raise KeyError(map_id)
        layer = self._map_layers.get(map_id)
        if layer is None:
            allowed = tuple(getattr(self, "downsample_factors", DOWNSAMPLE_FACTORS))
            layer = VirtualMapLayer(
                self,
                descriptor,
                allowed_downsample=allowed,
                cache=self._map_tile_cache,
                cache_capacity=self._map_tile_cache_capacity,
            )
            self._map_layers[map_id] = layer
        return layer

    def _invalidate_map_tiles_for_fov(self, fov_name: str) -> None:
        if not self._map_layers:
            return
        for layer in self._map_layers.values():
            try:
                layer.invalidate_for_fov(fov_name)
            except Exception:
                if self._debug:
                    print(f"[viewer] Failed to invalidate map cache for {fov_name}")

    def _map_state_signature(
        self,
        selected_channels: Tuple[str, ...],
        downsample_factor: int,
    ):
        if not self._map_mode_enabled:
            return None
        controls = getattr(self, "ui_component", None)
        if controls is None:
            return None

        try:
            outline = int(self.mask_outline_thickness)
        except Exception:
            outline = 1

        channel_entries = []
        color_controls = getattr(controls, "color_controls", {})
        contrast_min_controls = getattr(controls, "contrast_min_controls", {})
        contrast_max_controls = getattr(controls, "contrast_max_controls", {})
        for channel in selected_channels:
            color_widget = color_controls.get(channel)
            color_key = color_widget.value if color_widget is not None else None
            if isinstance(color_key, str):
                color_value = self.predefined_colors.get(color_key, color_key)
            else:
                color_value = color_key
            min_widget = contrast_min_controls.get(channel)
            max_widget = contrast_max_controls.get(channel)
            try:
                min_val = float(min_widget.value)
            except Exception:
                min_val = 0.0
            try:
                max_val = float(max_widget.value)
            except Exception:
                max_val = 0.0
            channel_entries.append(
                (channel, color_value, round(min_val, 6), round(max_val, 6))
            )

        mask_entries = []
        mask_display_controls = getattr(controls, "mask_display_controls", {})
        mask_color_controls = getattr(controls, "mask_color_controls", {})
        if self.masks_available and mask_display_controls:
            for mask_name, checkbox in mask_display_controls.items():
                if not bool(getattr(checkbox, "value", False)):
                    continue
                color_widget = mask_color_controls.get(mask_name)
                color_key = color_widget.value if color_widget is not None else None
                if isinstance(color_key, str):
                    color_value = self.predefined_colors.get(color_key, color_key)
                else:
                    color_value = color_key
                mask_entries.append((mask_name, color_value))
        mask_signature = tuple(sorted(mask_entries))

        annotation_signature = None
        if (
            self.annotations_available
            and self.annotation_display_enabled
            and self.active_annotation_name
        ):
            name = self.active_annotation_name
            palette = tuple(
                sorted((str(k), str(v)) for k, v in self.annotation_palettes.get(name, {}).items())
            )
            labels = tuple(
                sorted((str(k), str(v)) for k, v in self.annotation_class_labels.get(name, {}).items())
            )
            try:
                alpha = round(float(self.annotation_overlay_alpha), 6)
            except Exception:
                alpha = 0.5
            annotation_signature = (
                name,
                alpha,
                str(self.annotation_overlay_mode),
                str(self.annotation_label_display_mode),
                palette,
                labels,
            )

        display = getattr(self, "image_display", None)
        if display is not None:
            selected_cells = getattr(display, "selected_cells", ())
            try:
                painter_selected = tuple(sorted(selected_cells))
            except Exception:
                painter_selected = ()
        else:
            painter_selected = ()

        painter_signature = (bool(self._is_mask_painter_enabled()), painter_selected)

        return (
            "v1",
            int(downsample_factor),
            tuple(channel_entries),
            mask_signature,
            annotation_signature,
            outline,
            painter_signature,
        )
    
    def resolve_cell_map_position(
        self,
        fov_name: str,
        x_value: float,
        y_value: float,
        *,
        map_id: Optional[str] = None,
    ) -> Optional[MapCellPosition]:
        """Translate FOV-local cell coordinates to stitched-map pixels."""

        if map_id is None:
            map_id = self._active_map_id
        if not (self._map_mode_enabled and map_id and fov_name):
            return None

        try:
            layer = self._get_map_layer(map_id)
        except Exception:
            return None

        geometry_getter = getattr(layer, "tile_geometry", None)
        if not callable(geometry_getter):
            return None

        tile = geometry_getter(str(fov_name))
        if tile is None:
            return None

        base_pixel_um = float(layer.base_pixel_size_um())
        if not math.isfinite(base_pixel_um) or base_pixel_um <= 0.0:
            return None

        try:
            pixel_size_um = float(tile.pixel_size_um)
        except Exception:
            return None
        if not math.isfinite(pixel_size_um) or pixel_size_um <= 0.0:
            return None

        try:
            bounds = layer.map_bounds()
            bounds_min_x = float(bounds[0])
            bounds_min_y = float(bounds[2])
        except (TypeError, ValueError, IndexError):
            bounds_min_x = 0.0
            bounds_min_y = 0.0

        try:
            local_x = float(x_value)
            local_y = float(y_value)
        except (TypeError, ValueError):
            return None

        offset_x_um = float(getattr(tile, "x_min_um", 0.0)) - bounds_min_x
        offset_y_um = float(getattr(tile, "y_min_um", 0.0)) - bounds_min_y

        map_x_um = offset_x_um + local_x * pixel_size_um
        map_y_um = offset_y_um + local_y * pixel_size_um

        x_px = map_x_um / base_pixel_um
        y_px = map_y_um / base_pixel_um
        if not (math.isfinite(x_px) and math.isfinite(y_px)):
            return None

        pixel_scale = pixel_size_um / base_pixel_um if base_pixel_um > 0 else 1.0
        if not math.isfinite(pixel_scale) or pixel_scale <= 0.0:
            pixel_scale = 1.0

        return MapCellPosition(x_px=x_px, y_px=y_px, pixel_scale=pixel_scale)

    def resolve_map_pixel_to_fov(
        self,
        x_value: float,
        y_value: float,
        *,
        map_id: Optional[str] = None,
    ) -> Optional[MapPixelLocalization]:
        if map_id is None:
            map_id = self._active_map_id
        if not (self._map_mode_enabled and map_id):
            return None

        try:
            layer = self._get_map_layer(map_id)
        except Exception:
            return None

        hit = layer.localize_map_pixel(x_value, y_value)
        if hit is None:
            return None

        try:
            map_id_str = str(map_id)
            fov_name = str(hit.fov_name)
            local_x_px = float(hit.local_x_px)
            local_y_px = float(hit.local_y_px)
            pixel_scale = float(hit.pixel_scale)
        except Exception:
            return None

        return MapPixelLocalization(
            map_id=map_id_str,
            fov_name=fov_name,
            local_x_px=local_x_px,
            local_y_px=local_y_px,
            pixel_scale=pixel_scale,
        )

    def _selected_mask_names(self) -> Tuple[str, ...]:
        controls = getattr(self.ui_component, "mask_display_controls", {})
        if not isinstance(controls, dict):
            return ()
        selected = []
        for name, checkbox in controls.items():
            try:
                if bool(getattr(checkbox, "value", False)):
                    selected.append(str(name))
            except Exception:
                continue
        return tuple(selected)

    def _ensure_mask_cache(self, fov_name: str) -> None:
        if not self.masks_available:
            return
        if fov_name in self.mask_cache:
            return
        try:
            self.load_fov(fov_name)
        except Exception:
            if self._debug:
                print(f"[viewer] Failed to prime mask cache for {fov_name}")

    def _get_mask_array(self, fov_name: str, mask_name: str):
        self._ensure_mask_cache(fov_name)
        label_entry = self.label_masks_cache.get(fov_name, {}).get(mask_name)
        mask_candidate = None
        if isinstance(label_entry, dict):
            mask_candidate = label_entry.get(1)
        if mask_candidate is None:
            mask_candidate = self.mask_cache.get(fov_name, {}).get(mask_name)
        if mask_candidate is None:
            return None
        try:
            return mask_candidate.compute()
        except AttributeError:
            return np.asarray(mask_candidate)
        except Exception:
            return None

    def _lookup_mask_hit(
        self,
        fov_name: str,
        local_x: float,
        local_y: float,
        *,
        map_id: Optional[str] = None,
    ) -> Optional[MaskHit]:
        if not self.masks_available:
            return None

        mask_names = self._selected_mask_names()
        if not mask_names:
            return None

        try:
            ix = int(math.floor(local_x))
            iy = int(math.floor(local_y))
        except (TypeError, ValueError):
            return None
        if ix < 0 or iy < 0:
            return None

        for mask_name in mask_names:
            mask_array = self._get_mask_array(fov_name, mask_name)
            if mask_array is None or mask_array.size == 0:
                continue
            if iy >= mask_array.shape[0] or ix >= mask_array.shape[1]:
                continue
            try:
                mask_value = int(mask_array[iy, ix])
            except Exception:
                continue
            if mask_value == 0:
                continue

            cell_row = None
            if self.cell_table is not None:
                try:
                    cell_row = resolve_cell_record(
                        self.cell_table,
                        fov_value=fov_name,
                        mask_name=mask_name,
                        mask_id=mask_value,
                        fov_key=self.fov_key,
                        label_key=self.label_key,
                        mask_key=self.mask_key,
                    )
                except Exception:
                    cell_row = None

            return MaskHit(
                fov_name=str(fov_name),
                mask_name=str(mask_name),
                mask_id=int(mask_value),
                local_x_px=float(local_x),
                local_y_px=float(local_y),
                map_id=map_id,
                cell_record=cell_row,
            )

        return None

    def resolve_mask_hit_at_viewport(self, x_px: float, y_px: float) -> Optional[MaskHit]:
        if not self.masks_available:
            return None

        if self._map_mode_active and self._active_map_id:
            localization = self.resolve_map_pixel_to_fov(x_px, y_px, map_id=self._active_map_id)
            if localization is None:
                return None
            return self._lookup_mask_hit(
                localization.fov_name,
                localization.local_x_px,
                localization.local_y_px,
                map_id=localization.map_id,
            )

        selector = getattr(self.ui_component, "image_selector", None)
        fov_name = selector.value if selector is not None else None
        if not fov_name:
            return None
        return self._lookup_mask_hit(fov_name, x_px, y_px)

    def _update_map_mask_highlights(self) -> None:
        display = getattr(self, "image_display", None)
        if display is None:
            return

        base_image = display._materialize_combined()
        if base_image is None:
            return

        selections = getattr(display, "selected_masks_label", None)
        if not (self._map_mode_active and self._active_map_id and selections):
            display.img_display.set_data(base_image)
            display.fig.canvas.draw_idle()
            return

        try:
            layer = self._get_map_layer(self._active_map_id)
        except Exception:
            display.img_display.set_data(base_image)
            display.fig.canvas.draw_idle()
            return

        tile_viewports = layer.last_tile_viewports()
        if not tile_viewports:
            display.img_display.set_data(base_image)
            display.fig.canvas.draw_idle()
            return

        overlay = np.array(base_image, copy=True)
        applied = False

        for selection in selections:
            fov_name = getattr(selection, "fov", None)
            mask_name = getattr(selection, "mask", None)
            mask_id = getattr(selection, "mask_id", None)
            if not (fov_name and mask_name and isinstance(mask_id, int)):
                continue

            viewport_info = tile_viewports.get(str(fov_name))
            if viewport_info is None:
                continue

            region_xy = getattr(viewport_info, "region_xy", None)
            region_ds = getattr(viewport_info, "region_ds", None)
            if region_xy is None or region_ds is None:
                continue

            mask_array = self._get_mask_array(str(fov_name), str(mask_name))
            if mask_array is None or mask_array.size == 0:
                continue

            try:
                x_min_px, x_max_px, y_min_px, y_max_px = (int(region_xy[0]), int(region_xy[1]), int(region_xy[2]), int(region_xy[3]))
            except Exception:
                continue

            if x_min_px >= x_max_px or y_min_px >= y_max_px:
                continue

            try:
                mask_region = mask_array[y_min_px:y_max_px, x_min_px:x_max_px]
            except Exception:
                continue
            if mask_region.size == 0:
                continue

            downsample = max(1, int(getattr(viewport_info, "downsample_factor", 1)))
            mask_region = mask_region[::downsample, ::downsample]
            if mask_region.size == 0:
                continue

            try:
                mask_binary = mask_region == mask_id
            except Exception:
                continue
            if not np.any(mask_binary):
                continue

            try:
                edges = find_boundaries(mask_binary, mode="inner")
            except Exception:
                edges = mask_binary
            if not np.any(edges):
                edges = mask_binary

            outline_thickness = scale_outline_thickness(
                getattr(self, "mask_outline_thickness", 1),
                downsample,
            )
            if outline_thickness > 1:
                try:
                    edges = thicken_outline(edges, outline_thickness - 1)
                except Exception:
                    pass

            dest_x0 = max(0, int(getattr(viewport_info, "dest_x0", 0)))
            dest_x1 = max(dest_x0, int(getattr(viewport_info, "dest_x1", dest_x0)))
            dest_y0 = max(0, int(getattr(viewport_info, "dest_y0", 0)))
            dest_y1 = max(dest_y0, int(getattr(viewport_info, "dest_y1", dest_y0)))

            if dest_x0 >= dest_x1 or dest_y0 >= dest_y1:
                continue

            section_view = overlay[dest_y0:dest_y1, dest_x0:dest_x1]
            if section_view.size == 0:
                continue

            if section_view.shape[0] != edges.shape[0] or section_view.shape[1] != edges.shape[1]:
                rows = min(section_view.shape[0], edges.shape[0])
                cols = min(section_view.shape[1], edges.shape[1])
                if rows <= 0 or cols <= 0:
                    continue
                section_view = section_view[:rows, :cols]
                edges = edges[:rows, :cols]

            section_view[edges] = [1.0, 1.0, 1.0]
            applied = True

        if applied:
            display.img_display.set_data(overlay)
        else:
            display.img_display.set_data(base_image)
        display.fig.canvas.draw_idle()
    def after_all_plugins_loaded(self):
        # loop through all the attributes of self.SidePlots, call the `after_all_plugins_loaded`` method
        for attr_name in dir(self.SidePlots):
            attr = getattr(self.SidePlots, attr_name)
            if isinstance(attr, PluginBase):
                if self._debug:
                    print(f"Calling after_all_plugins_loaded for {attr_name}")
                attr.after_all_plugins_loaded()
            else:
                if self._debug:
                    print(f"Skipping {attr_name}")


    def dynamically_load_plugins(self):
        plugin_dir = os.path.join(os.path.dirname(__file__), 'plugin')
        for filename in os.listdir(plugin_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                if self._debug:
                    print(f"Loading plugin: {filename}")
                module_name = filename[:-3]
                module_path = f"ueler.viewer.plugin.{module_name}"
                module = importlib.import_module(module_path)

                for attr_name in dir(module):
                    plugin_candidate = getattr(module, attr_name)
                    if (
                        isinstance(plugin_candidate, type)
                        and issubclass(plugin_candidate, PluginBase)
                        and plugin_candidate is not PluginBase
                    ):
                        instance = plugin_candidate(self, 6, 3)
                        self.SidePlots.__dict__[f"{module_name}_output"] = instance
                        if module_name == 'roi_manager_plugin':
                            self.roi_plugin = instance
                        

    def setup_event_connections(self):
        # Handle cache size changes
        self.ui_component.cache_size_input.observe(self.on_cache_size_change, names='value')

    def on_downsample_factor_changed(self, new_factor: int) -> None:
        try:
            factor = max(1, int(new_factor))
        except (TypeError, ValueError):
            factor = 1

        if factor == getattr(self, "current_downsample_factor", factor):
            self.current_downsample_factor = factor
            return

        self.current_downsample_factor = factor
        if self._fov_mode != "ome-tiff":
            return

        for resource in self.image_cache.values():
            if isinstance(resource, OMEFovWrapper):
                resource.set_downsample_factor(factor)

    def _close_image_resource(self, resource) -> None:
        close_fn = getattr(resource, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                if self._debug:
                    print(f"[viewer] Failed to close image resource: {resource}")

    def on_cache_size_change(self, change):
        """Handle changes to the cache size input."""
        self.max_cache_size = self.ui_component.cache_size_input.value
        print(f"Cache size set to {self.max_cache_size}.")

        # Trim caches if necessary
        while len(self.image_cache) > self.max_cache_size:
            removed_fov, resource = self.image_cache.popitem(last=False)
            self._close_image_resource(resource)
            print(f"Removed FOV '{removed_fov}' from image cache due to cache size limit.")

        while len(self.mask_cache) > self.max_cache_size:
            removed_fov, _ = self.mask_cache.popitem(last=False)
            self.edge_masks_cache.pop(removed_fov, None)
            self.label_masks_cache.pop(removed_fov, None)
            print(f"Removed FOV '{removed_fov}' from mask cache due to cache size limit.")

        while len(self.annotation_cache) > self.max_cache_size:
            removed_fov, _ = self.annotation_cache.popitem(last=False)
            self.annotation_label_cache.pop(removed_fov, None)
            print(f"Removed FOV '{removed_fov}' from annotation cache due to cache size limit.")

    def _ensure_channel_max_computed(self, fov_name, channel_name):
        if channel_name in self.channel_max_values:
            return

        fov_images = self.image_cache.get(fov_name)
        if isinstance(fov_images, OMEFovWrapper):
            arr = fov_images[channel_name]
            if arr is None:
                return
            try:
                ch_max = arr.max().compute()
            except Exception:
                ch_max = 0
            
            dtype = getattr(arr, "dtype", None)
            if dtype is not None and np.issubdtype(dtype, np.integer):
                dtype_limit = float(np.iinfo(dtype).max)
            elif dtype is not None and np.issubdtype(dtype, np.floating):
                dtype_limit = float(np.finfo(dtype).max)
            else:
                dtype_limit = float(ch_max) if ch_max is not None else 65535.0
            
            display_max = float(ch_max)
            merge_channel_max(channel_name, self.channel_max_values, display_max, dtype_limit)

    def load_fov(self, fov_name, requested_channels=None):
        """Load images and masks for a FOV into the cache."""
        
        if self._fov_mode == "ome-tiff" and fov_name not in self.image_cache:
             candidates = glob.glob(os.path.join(self.base_folder, f"{fov_name}.ome.tif*"))
             if candidates:
                 path = candidates[0]
                 wrapper = OMEFovWrapper(path, ds_factor=self.current_downsample_factor)
                 self.image_cache[fov_name] = wrapper
                 
                 # We do NOT compute max for all channels here anymore to avoid memory spikes.
                 # Instead, we compute it on demand in _ensure_channel_max_computed.
                 
                 self.image_cache.move_to_end(fov_name)

        # Load images if not in cache
        if fov_name not in self.image_cache:
            # Load channel structures
            channels = load_channel_struct_fov(fov_name, self.base_folder)
            # Add to cache
            self.image_cache[fov_name] = channels

        # If the requested channels are not provided, take the first channel
        if requested_channels is None:
            channel_list = list(self.image_cache[fov_name].keys())
            requested_channels = (channel_list[0],)

        for ch in requested_channels:
            if isinstance(self.image_cache[fov_name], OMEFovWrapper):
                self._ensure_channel_max_computed(fov_name, ch)
            # for self.image_cache[fov_name][ch] is None, load the image
            elif self.image_cache[fov_name][ch] is None:
                # Load images for FOV
                self.image_cache[fov_name][ch] = load_one_channel_fov(fov_name, self.base_folder, self.channel_max_values, ch)
                self._sync_channel_controls(ch)

        self.image_cache.move_to_end(fov_name)

        # Remove least recently used if cache exceeds max size
        while len(self.image_cache) > self.max_cache_size:
            removed_fov, resource = self.image_cache.popitem(last=False)
            self._close_image_resource(resource)
            print(f"Removed FOV '{removed_fov}' from image cache.")
            self._invalidate_map_tiles_for_fov(removed_fov)

        else:
            # Move the accessed FOV to the end to mark it as recently used
            self.image_cache.move_to_end(fov_name)

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
        fov_name = self.ui_component.image_selector.value
        fov_images = self.image_cache[fov_name]
        if isinstance(fov_images, OMEFovWrapper):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[viewer] Using OME metadata for %s dimensions on image change; reported shape=%s",
                    fov_name,
                    fov_images.shape,
                )
            self.height, self.width = fov_images.shape
        else:
            try:
                first_channel_name, first_channel_image = next(iter(fov_images.items()))
            except StopIteration as exc:
                raise ValueError(f"No channel data found for FOV '{fov_name}'.") from exc
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[viewer] Sampling channel %s for %s dimensions during image change; shape=%s",
                    first_channel_name,
                    fov_name,
                    getattr(first_channel_image, "shape", None),
                )
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
                chart_plugin = getattr(self.SidePlots, "chart_output", None)
                heatmap_plugin = getattr(self.SidePlots, "heatmap_output", None)
                heatmap_checkbox = (
                    getattr(getattr(heatmap_plugin, "ui_component", None), "main_viewer_checkbox", None)
                    if heatmap_plugin is not None
                    else None
                )
                heatmap_linked = bool(heatmap_checkbox and heatmap_checkbox.value)

                if not heatmap_linked:
                    self.image_display.clear_patches()

                if chart_plugin is not None:
                    chart_plugin.highlight_cells()

                if heatmap_linked and heatmap_plugin is not None:
                    heatmap_plugin.highlight_cells()

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
        if self._suspend_pixel_size_observer:
            return

        new_value = change.get("new") if isinstance(change, dict) else None
        try:
            new_nm = float(new_value)
        except (TypeError, ValueError):
            self._restore_pixel_size_widget()
            return

        if new_nm <= 0:
            self._restore_pixel_size_widget()
            return

        if abs(new_nm - self.pixel_size_nm) < 1e-6:
            return

        self.pixel_size_nm = new_nm
        self.update_scale_bar()
        self._notify_plugins_pixel_size_changed(new_nm)
    
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

    def _restore_pixel_size_widget(self) -> None:
        widget = getattr(self.ui_component, "pixel_size_inttext", None)
        if widget is None:
            return
        self._suspend_pixel_size_observer = True
        try:
            widget.value = int(round(max(self.pixel_size_nm, 1.0)))
        finally:
            self._suspend_pixel_size_observer = False

    def _notify_plugins_pixel_size_changed(self, pixel_nm: float) -> None:
        sideplots = getattr(self, "SidePlots", None)
        plugin = getattr(sideplots, "export_fovs_output", None) if sideplots is not None else None
        if plugin is None or not hasattr(plugin, "on_viewer_pixel_size_change"):
            return
        try:
            plugin.on_viewer_pixel_size_change(pixel_nm)
        except Exception:
            if self._debug:
                print("[viewer] Plugin notification for pixel size change failed")

    def get_pixel_size_nm(self) -> float:
        if self._map_mode_active and self._map_pixel_size_nm is not None:
            try:
                return max(1.0, float(self._map_pixel_size_nm))
            except (TypeError, ValueError):
                pass
        return max(1.0, float(getattr(self, "pixel_size_nm", 390.0)))

    def update_scale_bar(self) -> None:
        display = getattr(self, "image_display", None)
        if display is None or not hasattr(display, "update_scale_bar"):
            return

        pixel_nm = self.get_pixel_size_nm()
        combined = getattr(display, "combined", None)
        width = None
        if combined is not None and hasattr(combined, "shape"):
            try:
                width = int(combined.shape[1])
            except Exception:
                width = None
        if not width and hasattr(self, "width"):
            try:
                width = int(self.width)
            except Exception:
                width = None

        if not width or width <= 0:
            display.update_scale_bar(None)
            return

        downsample = max(1, int(getattr(self, "current_downsample_factor", 1)))
        fraction = float(getattr(self, "scale_bar_fraction", 0.1))
        try:
            effective_nm = effective_pixel_size_nm(pixel_nm, downsample)
            spec = compute_scale_bar_spec(
                image_width_px=width,
                pixel_size_nm=effective_nm,
                max_fraction=max(0.01, min(fraction, 0.5)),
            )
        except ValueError:
            spec = None

        display.update_scale_bar(
            spec,
            color="white",
            font_size=12.0,
            data_pixel_ratio=float(downsample),
        )


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

        # Ensure max values are computed for selected channels
        fov_name = self.ui_component.image_selector.value
        for channel in self.ui_component.channel_selector.value:
            self._ensure_channel_max_computed(fov_name, channel)

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
            outline_slider = getattr(self.ui_component, "mask_outline_thickness_slider", None)
            if outline_slider is not None:
                outline_slider.disabled = not bool(self.mask_names)
                self._set_mask_outline_slider_value(self.mask_outline_thickness)
                mask_widgets.append(outline_slider)
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
                self.ui_component.annotation_alpha_slider,
                self.ui_component.annotation_label_mode,
                self.ui_component.annotation_edit_button,
                self.ui_component.annotation_palette_tab,
                self.ui_component.annotation_palette_feedback,
            ]
            self.ui_component.annotation_palette_tab.layout.display = ""
            self.ui_component.annotation_controls_box.children = tuple(annotation_widgets)
        else:
            self.ui_component.annotation_palette_tab.layout.display = "none"
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

        section_children = [self.ui_component.channel_section_panel]
        section_titles = ["Channels"]

        if self.masks_available:
            if not mask_widgets:
                self.ui_component.mask_controls_box.children = (
                    self.ui_component.no_masks_label,
                )
            section_children.append(self.ui_component.mask_controls_box)
            section_titles.append("Masks")

        if self.annotations_available:
            if not annotation_widgets:
                self.ui_component.annotation_controls_box.children = (
                    self.ui_component.no_annotations_label,
                )
            section_children.append(self.ui_component.annotation_controls_box)
            section_titles.append("Pixel annotations")

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

    # ------------------------------------------------------------------
    # Annotation palette persistence
    # ------------------------------------------------------------------
    def _initialize_annotation_palette_manager(self) -> None:
        ui = getattr(self, "ui_component", None)
        if ui is None:
            return

        ui.annotation_palette_save_button.on_click(self.save_active_annotation_palette)
        ui.annotation_palette_load_button.on_click(self.load_annotation_palette_from_ui)
        ui.annotation_palette_change_folder_button.on_click(self._on_annotation_palette_change_folder_clicked)
        ui.annotation_palette_manual_apply.on_click(self._apply_annotation_palette_manual_folder)
        ui.annotation_palette_manual_cancel.on_click(self._cancel_annotation_palette_manual_folder)
        ui.annotation_palette_apply_saved_button.on_click(self.apply_saved_annotation_palette)
        ui.annotation_palette_overwrite_button.on_click(self.overwrite_saved_annotation_palette)
        ui.annotation_palette_delete_button.on_click(self.delete_saved_annotation_palette)

        folder = self._determine_annotation_palette_folder()
        if folder is None:
            folder = Path.cwd()
            ui.annotation_palette_manual_box.layout.display = "block"
            self._log_annotation_palette(
                "Pixel annotation palette directory unavailable; enter a custom location.",
                error=True,
                clear=True,
            )
        else:
            ui.annotation_palette_manual_box.layout.display = "none"

        self.annotation_palette_folder = folder.resolve()
        ui.annotation_palette_folder_display.value = str(self.annotation_palette_folder)
        self._load_annotation_palette_registry(self.annotation_palette_folder)

    def _determine_annotation_palette_folder(self) -> Optional[Path]:
        base_folder = getattr(self, "base_folder", None)
        if not base_folder:
            return None
        base_path = Path(base_folder).expanduser()
        target = base_path / ".UELer" / ANNOTATION_PALETTE_FOLDER_NAME
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - fallback for restricted environments
            return None
        return target

    def _load_annotation_palette_registry(self, folder: Path) -> None:
        folder = folder.expanduser().resolve()
        self.annotation_palette_folder = folder
        ui = self.ui_component
        ui.annotation_palette_folder_display.value = str(folder)
        try:
            records = load_palette_registry(folder, ANNOTATION_REGISTRY_FILENAME)
        except PaletteStoreError as err:
            self._log_annotation_palette(f"Failed to load palette registry: {err}", error=True, clear=True)
            records = {}
        self.annotation_palette_registry_records = records
        for record in self.annotation_palette_registry_records.values():
            record.setdefault("folder", str(folder))
        self._refresh_annotation_palette_dropdown()
        load_picker = getattr(ui, "annotation_palette_load_picker", None)
        if load_picker is not None:
            try:
                load_picker.reset(path=str(folder))
            except Exception:  # pragma: no cover - FileChooser may not expose reset
                pass

    def _refresh_annotation_palette_dropdown(self) -> None:
        dropdown = self.ui_component.annotation_palette_saved_sets_dropdown
        if not self.annotation_palette_registry_records:
            dropdown.options = [("No saved sets", "")]
            dropdown.value = ""
            return

        entries = []
        for name in sorted(self.annotation_palette_registry_records.keys()):
            record = self.annotation_palette_registry_records.get(name, {})
            annotation = (record.get("annotation") or "").strip()
            label = f"{name} ({annotation})" if annotation else name
            entries.append((label, name))
        dropdown.options = [("Select a set", "")] + entries
        dropdown.value = ""

    def _log_annotation_palette(self, message: str, *, error: bool = False, clear: bool = False) -> None:
        output = getattr(self.ui_component, "annotation_palette_feedback", None)
        if output is None:
            return
        with output:
            if clear:
                output.clear_output(wait=True)
            prefix = "⚠️ " if error else ""
            print(f"{prefix}{message}")

    def save_active_annotation_palette(self, _button) -> None:
        try:
            ui = self.ui_component
            name = ui.annotation_palette_name_input.value.strip()
            if not name:
                raise PaletteStoreError("Please provide a name for the palette.")
            if self.annotation_palette_folder is None:
                raise PaletteStoreError("Select a palette folder before saving.")
            annotation = (self.active_annotation_name or '').strip()
            if not annotation:
                raise PaletteStoreError("Select an annotation before saving a palette.")

            self._ensure_annotation_metadata(annotation)
            class_ids = list(self.annotation_class_ids.get(annotation, []))
            if not class_ids:
                raise PaletteStoreError(f"No classes detected for annotation '{annotation}'.")

            palette = apply_color_defaults(class_ids, self.annotation_palettes.get(annotation, {}))
            labels = dict(self.annotation_class_labels.get(annotation, {}))
            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            payload = {
                "name": name,
                "version": ANNOTATION_PALETTE_VERSION,
                "annotation": annotation,
                "class_ids": list(class_ids),
                "default_color": DEFAULT_COLOR,
                "colors": dict(palette),
                "labels": labels,
                "label_mode": self.annotation_label_display_mode,
                "saved_at": timestamp,
            }

            folder = self.annotation_palette_folder
            path = resolve_palette_path(
                folder,
                name,
                ANNOTATION_PALETTE_FILE_SUFFIX,
                default_slug="pixel-annotations",
            )
            write_palette_file(path, payload)
            self._update_annotation_palette_registry(folder, name, path, annotation, timestamp)
            self.active_annotation_palette_set_name = name
            self._log_annotation_palette(
                f"Saved palette '{name}' for annotation '{annotation}'.",
                clear=True,
            )
        except Exception as err:  # pylint: disable=broad-except
            self._log_annotation_palette(f"Failed to save palette: {err}", error=True, clear=True)

    def _update_annotation_palette_registry(
        self,
        folder: Path,
        name: str,
        path: Path,
        annotation: str,
        timestamp: str,
    ) -> None:
        try:
            records = load_palette_registry(folder, ANNOTATION_REGISTRY_FILENAME)
        except PaletteStoreError:
            records = {}
        records[name] = {
            "name": name,
            "path": str(path.resolve()),
            "annotation": annotation,
            "last_modified": timestamp,
            "folder": str(folder.resolve()),
        }
        save_palette_registry(folder, ANNOTATION_REGISTRY_FILENAME, records)
        if (
            self.annotation_palette_folder is not None
            and folder.resolve() == self.annotation_palette_folder.resolve()
        ):
            self.annotation_palette_registry_records = records
            self._refresh_annotation_palette_dropdown()

    def get_active_annotation_palette_set(self) -> str:
        return self.active_annotation_palette_set_name or ""

    def resolve_annotation_palette_snapshot(self, name: str) -> Optional[AnnotationOverlaySnapshot]:
        candidate = (name or "").strip()
        if not candidate:
            return None
        record = self.annotation_palette_registry_records.get(candidate)
        if record is None:
            return None
        path = Path(record.get("path", "")).expanduser()
        if not path.exists():
            return None
        try:
            payload = read_palette_file(path)
        except Exception:  # pragma: no cover - IO errors
            return None

        annotation = str(payload.get("annotation") or "").strip()
        if not annotation:
            return None

        class_ids_payload = payload.get("class_ids", [])
        try:
            class_ids = [int(value) for value in class_ids_payload]
        except Exception:  # pragma: no cover - invalid payload
            class_ids = []
        if not class_ids:
            return None

        colors_payload = payload.get("colors", {})
        if not isinstance(colors_payload, dict):
            colors_payload = {}
        normalized_colors = {
            str(key): normalize_hex_color(value) or DEFAULT_COLOR
            for key, value in colors_payload.items()
        }

        palette = apply_color_defaults(class_ids, normalized_colors)
        return AnnotationOverlaySnapshot(
            name=annotation,
            alpha=float(payload.get("alpha", self.annotation_overlay_alpha)),
            mode=str(payload.get("mode", self.annotation_overlay_mode) or "combined"),
            palette=dict(palette),
        )

    def apply_annotation_palette_set(self, name: str) -> bool:
        if not name:
            return False

        record = self.annotation_palette_registry_records.get(name)
        if record is None:
            return False

        path = Path(record.get("path", "")).expanduser()
        if not path.exists():
            return False

        try:
            self._load_annotation_palette(path)
        except Exception:  # pragma: no cover - delegated error reporting
            return False

        self.active_annotation_palette_set_name = name
        dropdown = getattr(self.ui_component, "annotation_palette_saved_sets_dropdown", None)
        if dropdown is not None:
            try:
                option_values = [value if isinstance(value, str) else value[1] for value in getattr(dropdown, "options", [])]
                if name in option_values:
                    dropdown.value = name
            except Exception:
                pass
        return True

    def get_mask_visibility_state(self) -> Dict[str, bool]:
        controls = getattr(self.ui_component, "mask_display_controls", {})
        if not isinstance(controls, dict):
            return {}
        state: Dict[str, bool] = {}
        for name, checkbox in controls.items():
            try:
                state[name] = bool(getattr(checkbox, "value", False))
            except Exception:  # pragma: no cover - widget access guard
                continue
        return state

    def apply_mask_visibility_state(self, state: Mapping[str, object]) -> bool:
        if not state:
            return False

        controls = getattr(self.ui_component, "mask_display_controls", {})
        if not isinstance(controls, dict) or not controls:
            return False

        matched_any = False
        updated = False
        for name, desired in state.items():
            checkbox = controls.get(name)
            if checkbox is None:
                continue
            matched_any = True
            desired_value = bool(desired)
            try:
                current_value = bool(getattr(checkbox, "value", False))
            except Exception:  # pragma: no cover - widget guard
                current_value = False
            if current_value != desired_value:
                checkbox.value = desired_value
                updated = True

        if updated:
            try:
                self.update_display(self.current_downsample_factor)
            except Exception:  # pragma: no cover - downstream update guard
                pass

        return matched_any or updated

    def _get_annotation_palette_load_path(self) -> Optional[Path]:
        picker = getattr(self.ui_component, "annotation_palette_load_picker", None)
        if picker is not None:
            selected = getattr(picker, "selected", None)
            if selected:
                return Path(selected).expanduser()
        path_input = getattr(self.ui_component, "annotation_palette_load_path_input", None)
        if path_input is not None:
            value = path_input.value.strip()
            if value:
                return Path(value).expanduser()
        return None

    def load_annotation_palette_from_ui(self, _button) -> None:
        try:
            path = self._get_annotation_palette_load_path()
            if path is None:
                raise PaletteStoreError("Select a palette file to load.")
            self._load_annotation_palette(path)
        except Exception as err:  # pylint: disable=broad-except
            self._log_annotation_palette(f"Failed to load palette: {err}", error=True, clear=True)

    def _load_annotation_palette(self, path: Path) -> None:
        if not path.exists():
            raise PaletteStoreError(f"Palette file '{path}' was not found.")

        payload = read_palette_file(path)
        annotation = str(payload.get("annotation") or "").strip()
        if not annotation:
            raise PaletteStoreError("Palette file does not specify an annotation name.")

        if annotation not in self.annotation_names_set:
            raise PaletteStoreError(
                f"Annotation '{annotation}' is not available in the current dataset."
            )

        class_ids_payload = payload.get("class_ids", [])
        class_ids: List[int] = []
        for item in class_ids_payload:
            try:
                class_ids.append(int(item))
            except (TypeError, ValueError) as err:
                raise PaletteStoreError("Palette file contains invalid class identifiers.") from err

        colors_payload = payload.get("colors", {})
        if not isinstance(colors_payload, dict):
            raise PaletteStoreError("Palette file stores colors in an unexpected format.")
        normalized_colors = {
            str(key): normalize_hex_color(value) or DEFAULT_COLOR
            for key, value in colors_payload.items()
        }

        if not class_ids:
            inferred_ids = []
            for key in normalized_colors.keys():
                try:
                    inferred_ids.append(int(key))
                except (TypeError, ValueError):
                    continue
            class_ids = sorted(set(inferred_ids))

        if not class_ids:
            raise PaletteStoreError("Palette file does not declare any class identifiers.")

        palette = apply_color_defaults(class_ids, normalized_colors)
        labels_payload = payload.get("labels", {})
        labels = {str(key): str(value) for key, value in labels_payload.items()} if isinstance(labels_payload, dict) else {}

        label_mode = payload.get("label_mode")
        if isinstance(label_mode, str) and label_mode in {"id", "label"}:
            self.annotation_label_display_mode = label_mode

        self.annotation_class_ids[annotation] = list(class_ids)
        self.annotation_palettes[annotation] = dict(palette)
        if labels:
            self.annotation_class_labels[annotation] = labels

        selector = self.ui_component.annotation_selector
        selector.value = annotation
        self.active_annotation_name = annotation

        self._refresh_annotation_control_states()
        if self.annotation_display_enabled:
            self.update_display(self.current_downsample_factor)

        palette_name = str(payload.get("name", path.stem))
        self.active_annotation_palette_set_name = palette_name

        self._log_annotation_palette(
            f"Loaded palette '{payload.get('name', path.stem)}' for annotation '{annotation}'.",
            clear=True,
        )

    def _get_selected_annotation_palette_record(self) -> Optional[Dict[str, str]]:
        dropdown = self.ui_component.annotation_palette_saved_sets_dropdown
        name = dropdown.value
        if not name:
            return None
        return self.annotation_palette_registry_records.get(name)

    def apply_saved_annotation_palette(self, _button) -> None:
        try:
            record = self._get_selected_annotation_palette_record()
            if record is None:
                raise PaletteStoreError("Select a saved palette to apply.")
            path = Path(record.get("path", "")).expanduser()
            if not path.exists():
                raise PaletteStoreError(f"Palette file '{path}' was not found.")
            self._load_annotation_palette(path)
            dropdown = self.ui_component.annotation_palette_saved_sets_dropdown
            selected_name = dropdown.value or ""
            if not selected_name:
                selected_name = record.get("annotation") or path.stem
            self.active_annotation_palette_set_name = selected_name
        except Exception as err:  # pylint: disable=broad-except
            self._log_annotation_palette(f"Failed to apply palette: {err}", error=True, clear=True)

    def overwrite_saved_annotation_palette(self, _button) -> None:
        try:
            record = self._get_selected_annotation_palette_record()
            if record is None:
                raise PaletteStoreError("Select a saved palette to overwrite.")
            annotation = (self.active_annotation_name or '').strip()
            if not annotation:
                raise PaletteStoreError("Select an annotation before overwriting a palette.")

            self._ensure_annotation_metadata(annotation)
            class_ids = list(self.annotation_class_ids.get(annotation, []))
            if not class_ids:
                raise PaletteStoreError(f"No classes detected for annotation '{annotation}'.")

            palette = apply_color_defaults(class_ids, self.annotation_palettes.get(annotation, {}))
            labels = dict(self.annotation_class_labels.get(annotation, {}))
            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            path = Path(record.get("path", "")).expanduser()
            if not str(path):
                raise PaletteStoreError("Saved palette path is missing.")

            payload = {
                "name": self.ui_component.annotation_palette_saved_sets_dropdown.value,
                "version": ANNOTATION_PALETTE_VERSION,
                "annotation": annotation,
                "class_ids": list(class_ids),
                "default_color": DEFAULT_COLOR,
                "colors": dict(palette),
                "labels": labels,
                "label_mode": self.annotation_label_display_mode,
                "saved_at": timestamp,
            }

            write_palette_file(path, payload)
            folder = Path(record.get("folder", path.parent)).expanduser()
            name = self.ui_component.annotation_palette_saved_sets_dropdown.value
            self._update_annotation_palette_registry(folder, name, path, annotation, timestamp)
            self._log_annotation_palette("Palette overwritten successfully.", clear=True)
        except Exception as err:  # pylint: disable=broad-except
            self._log_annotation_palette(f"Failed to overwrite palette: {err}", error=True, clear=True)

    def delete_saved_annotation_palette(self, _button) -> None:
        try:
            record = self._get_selected_annotation_palette_record()
            if record is None:
                raise PaletteStoreError("Select a saved palette to delete.")

            path = Path(record.get("path", "")).expanduser()
            if path.exists():
                path.unlink()

            folder = Path(record.get("folder", self.annotation_palette_folder or path.parent)).expanduser()
            records = load_palette_registry(folder, ANNOTATION_REGISTRY_FILENAME)
            name = self.ui_component.annotation_palette_saved_sets_dropdown.value
            records.pop(name, None)
            save_palette_registry(folder, ANNOTATION_REGISTRY_FILENAME, records)

            if (
                self.annotation_palette_folder is not None
                and folder.resolve() == self.annotation_palette_folder.resolve()
            ):
                self.annotation_palette_registry_records = records
                self._refresh_annotation_palette_dropdown()

            self._log_annotation_palette("Deleted saved palette.", clear=True)
        except Exception as err:  # pylint: disable=broad-except
            self._log_annotation_palette(f"Failed to delete palette: {err}", error=True, clear=True)

    def _on_annotation_palette_change_folder_clicked(self, _button) -> None:
        self.ui_component.annotation_palette_manual_box.layout.display = "block"

    def _apply_annotation_palette_manual_folder(self, _button) -> None:
        raw_value = self.ui_component.annotation_palette_manual_input.value.strip()
        if not raw_value:
            self._log_annotation_palette("Enter a folder path to use for palettes.", error=True, clear=True)
            return

        folder = Path(raw_value).expanduser()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception as err:  # pylint: disable=broad-except
            self._log_annotation_palette(f"Could not use folder '{folder}': {err}", error=True, clear=True)
            return

        self.ui_component.annotation_palette_manual_input.value = ""
        self.ui_component.annotation_palette_manual_box.layout.display = "none"
        self._load_annotation_palette_registry(folder)
        self._log_annotation_palette(f"Using custom palette directory {folder}", clear=True)

    def _cancel_annotation_palette_manual_folder(self, _button) -> None:
        self.ui_component.annotation_palette_manual_input.value = ""
        self.ui_component.annotation_palette_manual_box.layout.display = "none"

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

    def focus_on_cell(
        self,
        fov_name: str,
        x_value: float,
        y_value: float,
        *,
        radius: float = 100.0,
    ) -> None:
        """Center the viewer on a cell, respecting stitched map offsets."""

        display = getattr(self, "image_display", None)
        if display is None:
            return

        map_id_for_focus = None
        map_switched = False
        target_map_id = self._map_id_for_fov(fov_name) if self._map_mode_enabled else None
        if self._map_mode_active and target_map_id:
            if self._active_map_id != target_map_id:
                try:
                    self._activate_map_mode(target_map_id)
                except Exception:
                    target_map_id = None
                else:
                    self._refresh_map_controls()
                    map_switched = True
            map_id_for_focus = target_map_id
        elif self._map_mode_active and self._active_map_id:
            map_id_for_focus = self._active_map_id

        if map_switched:
            try:
                self.update_display(self.current_downsample_factor)
            except Exception:
                if self._debug:
                    print("[viewer] Failed to refresh display after map switch")

        ax = getattr(display, "ax", None)
        if ax is None:
            return

        fig = getattr(display, "fig", None)
        canvas = getattr(fig, "canvas", None) if fig is not None else None
        toolbar = getattr(canvas, "toolbar", None) if canvas is not None else None
        nav_stack = getattr(toolbar, "_nav_stack", None) if toolbar is not None else None

        previous_view = None
        if nav_stack is not None:
            try:
                previous_view = nav_stack()
            except Exception:
                previous_view = None

        inverted_axes = False
        try:
            y_limits = ax.get_ylim()
            inverted_axes = y_limits[0] > y_limits[1]
        except Exception:
            inverted_axes = False

        radius = max(1.0, float(radius))

        if map_id_for_focus and self._map_mode_active:
            position = self.resolve_cell_map_position(
                fov_name,
                x_value,
                y_value,
                map_id=map_id_for_focus,
            )
            if position is not None:
                scale = max(1.0, float(position.pixel_scale))
                half = max(1.0, radius * scale)
                center_x = float(position.x_px)
                center_y = float(position.y_px)
                ax.set_xlim(center_x - half, center_x + half)
                if inverted_axes:
                    ax.set_ylim(center_y + half, center_y - half)
                else:
                    ax.set_ylim(center_y - half, center_y + half)
                if nav_stack is not None and previous_view is not None:
                    try:
                        nav_stack.push(previous_view)
                    except Exception:
                        pass
                if canvas is not None:
                    try:
                        canvas.draw_idle()
                    except Exception:
                        pass
                return

        image_selector = getattr(self.ui_component, "image_selector", None)
        if image_selector is not None and fov_name and image_selector.value != fov_name:
            image_selector.value = fov_name

        try:
            center_x = float(x_value)
            center_y = float(y_value)
        except (TypeError, ValueError):
            return

        half = radius
        ax.set_xlim(center_x - half, center_x + half)
        if inverted_axes:
            ax.set_ylim(center_y + half, center_y - half)
        else:
            ax.set_ylim(center_y - half, center_y + half)

        if nav_stack is not None and previous_view is not None:
            try:
                nav_stack.push(previous_view)
            except Exception:
                pass

        if canvas is not None:
            try:
                canvas.draw_idle()
            except Exception:
                pass

    def _compose_fov_image(
        self,
        fov_name: str,
        selected_channels: Tuple[str, ...],
        downsample_factor: int,
        region_xy: Tuple[int, int, int, int],
        region_ds: Tuple[int, int, int, int],
    ) -> np.ndarray:
        if not selected_channels:
            height = max(1, region_ds[3] - region_ds[2])
            width = max(1, region_ds[1] - region_ds[0])
            return np.zeros((height, width, 3), dtype=np.float32)

        self.load_fov(fov_name, selected_channels)
        fov_images = self.image_cache[fov_name]

        controls = self.ui_component
        channel_settings = {}
        for ch in selected_channels:
            control = controls.color_controls.get(ch)
            color_key = control.value if control is not None else "Red"
            color_value = self.predefined_colors.get(color_key, color_key)
            color_rgb = to_rgb(color_value)
            min_control = controls.contrast_min_controls.get(ch)
            max_control = controls.contrast_max_controls.get(ch)
            channel_settings[ch] = ChannelRenderSettings(
                color=color_rgb,
                contrast_min=min_control.value if min_control is not None else 0.0,
                contrast_max=max_control.value if max_control is not None else 65535.0,
            )

        annotation_settings = None
        if (
            self.annotations_available
            and self.annotation_display_enabled
            and self.active_annotation_name is not None
        ):
            annotation_layers = self.annotation_label_cache.get(fov_name, {})
            annotation_entry = annotation_layers.get(self.active_annotation_name, {})
            annotation_ds = annotation_entry.get(downsample_factor)
            if annotation_ds is None and annotation_entry:
                base_array = annotation_entry.get(1)
                if base_array is not None:
                    annotation_ds = base_array[::downsample_factor, ::downsample_factor]
            if annotation_ds is not None:
                class_ids = self.annotation_class_ids.get(self.active_annotation_name)
                if not class_ids:
                    cache_source = self.annotation_cache.get(fov_name, {}).get(
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
        mask_regions = {}
        if self.masks_available:
            selected_masks = [
                mask_name
                for mask_name, cb in controls.mask_display_controls.items()
                if getattr(cb, "value", False)
            ]
            for mask_name in selected_masks:
                label_dict = self.label_masks_cache.get(fov_name, {}).get(mask_name)
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
                color_control = controls.mask_color_controls.get(mask_name)
                color_key = color_control.value if color_control is not None else "White"
                color_value = self.predefined_colors.get(color_key, color_key)
                mask_settings.append(
                    MaskRenderSettings(
                        array=mask_array,
                        color=to_rgb(color_value),
                        mode="outline",
                        outline_thickness=int(self.mask_outline_thickness),
                        downsample_factor=downsample_factor,
                    )
                )

            label_cache = self.label_masks_cache.get(fov_name, {})
            mask_regions = collect_mask_regions(
                label_cache,
                selected_masks,
                downsample_factor,
                region_ds,
            )

        combined = render_fov_to_array(
            fov_name,
            fov_images,
            selected_channels,
            channel_settings,
            downsample_factor=downsample_factor,
            region_xy=region_xy,
            region_ds=region_ds,
            annotation=annotation_settings,
            masks=mask_settings,
        )

        painter_enabled = self._is_mask_painter_enabled()
        if painter_enabled and mask_regions:
            excluded = (
                set(self.image_display.selected_cells)
                if hasattr(self.image_display, "selected_cells")
                else set()
            )
            combined = apply_registry_colors(
                combined,
                fov=fov_name,
                mask_regions=mask_regions,
                outline_thickness=int(self.mask_outline_thickness),
                downsample_factor=downsample_factor,
                exclude_ids=excluded,
            )

        return combined

    def render_image(self, selected_channels, downsample_factor, xym, xym_ds):
        """Render a composite image for the active viewer selection."""

        current_fov = self.ui_component.image_selector.value

        if xym is not None:
            xmin, xmax, ymin, ymax = (int(v) for v in xym)
        else:
            self.load_fov(current_fov, tuple(selected_channels))
            fov_images = self.image_cache.get(current_fov, {})
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

        combined = self._compose_fov_image(
            current_fov,
            tuple(selected_channels),
            downsample_factor,
            region_xy,
            region_ds,
        )

        return combined

    def _render_fov_region(
        self,
        fov_name: str,
        selected_channels: Tuple[str, ...],
        downsample_factor: int,
        region_xy: Tuple[int, int, int, int],
        region_ds: Tuple[int, int, int, int],
    ) -> np.ndarray:
        return self._compose_fov_image(
            fov_name,
            tuple(selected_channels),
            downsample_factor,
            region_xy,
            region_ds,
        )

    # ------------------------------------------------------------------
    # Mask painter integration
    # ------------------------------------------------------------------
    def _is_mask_painter_enabled(self) -> bool:
        """Check if the mask painter plugin is enabled."""
        sideplots = getattr(self, "SidePlots", None)
        if sideplots is None:
            return False
        
        painter = getattr(sideplots, "mask_painter_output", None)
        if painter is None:
            return False
        
        ui = getattr(painter, "ui_component", None)
        if ui is None:
            return False
        
        checkbox = getattr(ui, "enabled_checkbox", None)
        if checkbox is None:
            return False
        
        return bool(getattr(checkbox, "value", False))

    # ------------------------------------------------------------------
    # Export overlay helpers
    # ------------------------------------------------------------------
    def capture_overlay_snapshot(
        self,
        *,
        include_annotations: bool = True,
        include_masks: bool = True,
    ) -> OverlaySnapshot:
        """Capture the viewer's current overlay configuration for exports."""

        use_annotations = bool(include_annotations and self.annotations_available and self.annotation_display_enabled)
        annotation_snapshot: Optional[AnnotationOverlaySnapshot] = None

        if use_annotations and self.active_annotation_name:
            name = self.active_annotation_name
            palette = dict(self.annotation_palettes.get(name, {}))
            annotation_snapshot = AnnotationOverlaySnapshot(
                name=name,
                alpha=float(self.annotation_overlay_alpha),
                mode=str(self.annotation_overlay_mode),
                palette=palette,
            )

        mask_snapshots: list[MaskOverlaySnapshot] = []
        use_masks = bool(include_masks and self.masks_available)
        if use_masks:
            mask_controls = getattr(self.ui_component, "mask_display_controls", {})
            color_controls = getattr(self.ui_component, "mask_color_controls", {})
            for mask_name, checkbox in mask_controls.items():
                if not checkbox.value:
                    continue
                color_widget = color_controls.get(mask_name)
                color_key = color_widget.value if color_widget is not None else None
                if isinstance(color_key, str):
                    colour_value = self.predefined_colors.get(color_key, color_key)
                else:
                    colour_value = self.predefined_colors.get(str(color_key), "#FFFFFF")
                mask_snapshots.append(
                    MaskOverlaySnapshot(
                        name=mask_name,
                        color=to_rgb(colour_value),
                        alpha=1.0,
                        mode="outline",
                        outline_thickness=int(self.mask_outline_thickness),
                    )
                )

        return OverlaySnapshot(
            include_annotations=use_annotations,
            include_masks=use_masks,
            annotation=annotation_snapshot,
            masks=tuple(mask_snapshots),
        )

    def build_overlay_settings_from_snapshot(
        self,
        fov_name: str,
        downsample_factor: int,
        snapshot: OverlaySnapshot,
    ) -> tuple[Optional[AnnotationRenderSettings], tuple[MaskRenderSettings, ...]]:
        """Reconstruct render settings for overlays based on a captured snapshot."""

        annotation_settings: Optional[AnnotationRenderSettings] = None
        mask_settings: list[MaskRenderSettings] = []

        if snapshot.include_annotations and snapshot.annotation:
            name = snapshot.annotation.name
            annotation_ds = None
            label_cache = self.annotation_label_cache.get(fov_name, {}).get(name)
            if label_cache:
                annotation_ds = label_cache.get(downsample_factor)
            if annotation_ds is None:
                raw_annotation = self.annotation_cache.get(fov_name, {}).get(name)
                if raw_annotation is not None:
                    annotation_ds = raw_annotation[::downsample_factor, ::downsample_factor]

            if annotation_ds is not None:
                try:
                    annotation_array = annotation_ds.compute()
                except AttributeError:
                    annotation_array = np.asarray(annotation_ds)

                class_ids = _unique_annotation_values(annotation_array)
                if class_ids.size == 0:
                    class_ids = np.array([0], dtype=np.int32)
                colormap = build_discrete_colormap(class_ids, snapshot.annotation.palette)
                annotation_settings = AnnotationRenderSettings(
                    array=annotation_array,
                    colormap=colormap,
                    alpha=float(snapshot.annotation.alpha),
                    mode=str(snapshot.annotation.mode),
                )

        if snapshot.include_masks and snapshot.masks:
            mask_cache = self.label_masks_cache.get(fov_name, {})
            for mask_snapshot in snapshot.masks:
                ds_cache = mask_cache.get(mask_snapshot.name)
                mask_ds = None
                if ds_cache:
                    mask_ds = ds_cache.get(downsample_factor)
                if mask_ds is None:
                    raw_mask = self.mask_cache.get(fov_name, {}).get(mask_snapshot.name)
                    if raw_mask is not None:
                        mask_ds = raw_mask[::downsample_factor, ::downsample_factor]
                if mask_ds is None:
                    continue

                try:
                    mask_array = mask_ds.compute()
                except AttributeError:
                    mask_array = np.asarray(mask_ds)

                if mask_array.size == 0:
                    continue

                mask_settings.append(
                    MaskRenderSettings(
                        array=mask_array,
                        color=mask_snapshot.color,
                        alpha=float(mask_snapshot.alpha),
                        mode=str(mask_snapshot.mode),
                        outline_thickness=int(getattr(mask_snapshot, "outline_thickness", 1)),
                        downsample_factor=downsample_factor,
                    )
                )

        return annotation_settings, tuple(mask_settings)

    def update_display(self, downsample_factor=8):
        """Update the display with the current downsample factor and visible area."""
        # Get visible region coordinates
        xmin, xmax, ymin, ymax, xmin_ds, xmax_ds, ymin_ds, ymax_ds = get_axis_limits_with_padding(self, downsample_factor)

        # Wrap the limits in tuples
        xym = (xmin, xmax, ymin, ymax)
        xym_r = (xmin, xmax, ymax, ymin)
        xym_ds = (xmin_ds, xmax_ds, ymin_ds, ymax_ds)
        try:
            self._last_viewport_px = (float(xmin), float(xmax), float(ymin), float(ymax), int(downsample_factor))
        except Exception:
            self._last_viewport_px = None
        
        # Get selected channels
        selected_channels = list(self.ui_component.channel_selector.value)
        channel_tuple = tuple(selected_channels)

        if not selected_channels:
            # If no channels selected, display black image sized to the current viewport
            viewport_height = max(1, int(xym_ds[3] - xym_ds[2]))
            viewport_width = max(1, int(xym_ds[1] - xym_ds[0]))
            blank = np.zeros((viewport_height, viewport_width, 3), dtype=np.float32)
            self.image_display.img_display.set_data(blank)
            self.image_display.img_display.set_extent(xym_r)
            self.image_display.combined = blank
            self.image_display.fig.canvas.draw_idle()
            return

        visible_fovs: Tuple[str, ...] = ()
        if self._map_mode_active and self._active_map_id:
            combined, visible_fovs = self._render_map_view(
                channel_tuple,
                downsample_factor,
                (int(xym[0]), int(xym[1]), int(xym[2]), int(xym[3])),
            )
            self.current_label_masks = {}
            self.full_resolution_label_masks = {}
        else:
            combined = self.render_image(
                selected_channels,
                downsample_factor,
                xym,
                xym_ds,
            )
            selector = getattr(self.ui_component, "image_selector", None)
            current_fov = selector.value if selector is not None else None
            if current_fov:
                visible_fovs = (str(current_fov),)
            if self.masks_available:
                self.current_label_masks = {}
                self.full_resolution_label_masks = {}
                selected_masks = [
                    mask_name
                    for mask_name, cb in self.ui_component.mask_display_controls.items()
                    if getattr(cb, "value", False)
                ]
                for mask_name in selected_masks:
                    label_cache = self.label_masks_cache.get(self.ui_component.image_selector.value, {})
                    label_mask_dict = label_cache.get(mask_name)
                    if not label_mask_dict:
                        continue
                    if 1 in label_mask_dict:
                        label_mask_full = label_mask_dict[1]
                        self.full_resolution_label_masks[mask_name] = label_mask_full
                    label_mask_ds = label_mask_dict.get(downsample_factor)
                    if label_mask_ds is None:
                        continue
                    self.current_label_masks[mask_name] = label_mask_ds[ymin_ds:ymax_ds, xmin_ds:xmax_ds]

                if hasattr(self.image_display, "update_patches"):
                    self.image_display.update_patches()

        # Update the displayed image
        self.image_display.img_display.set_data(combined)
        self.image_display.combined = combined
        self.image_display.img_display.set_extent(xym_r)
        self.image_display.fig.canvas.draw_idle()

        self._visible_map_fovs = visible_fovs

        self.inform_plugins('on_mv_update_display')
        self.update_scale_bar()
        if self._map_mode_active and self._active_map_id:
            try:
                self._update_map_mask_highlights()
            except Exception:
                if self._debug:
                    print("[viewer] Failed to refresh map mask highlights")

    # ------------------------------------------------------------------
    # Mask outline controls
    # ------------------------------------------------------------------
    def _set_mask_outline_slider_value(self, thickness: int) -> None:
        slider = getattr(self.ui_component, "mask_outline_thickness_slider", None)
        if slider is None:
            return
        try:
            thickness = max(1, int(thickness))
        except (TypeError, ValueError):
            thickness = 1
        if int(slider.value) == thickness:
            return
        self._suspend_mask_outline_sync = True
        try:
            slider.value = thickness
        finally:
            self._suspend_mask_outline_sync = False

    def _on_mask_outline_slider_change(self, change) -> None:
        if self._suspend_mask_outline_sync:
            return
        try:
            thickness = max(1, int(change.get("new", 1)))
        except (TypeError, ValueError):
            thickness = 1
        if thickness == int(getattr(self, "mask_outline_thickness", 1)):
            return
        self.mask_outline_thickness = thickness
        if getattr(self, "initialized", False):
            try:
                self.update_display(self.current_downsample_factor)
            except Exception:
                if self._debug:
                    print("[viewer] Failed to refresh display after mask outline update")
        self._notify_plugins_mask_outline_changed(thickness)

    def _notify_plugins_mask_outline_changed(self, thickness: int) -> None:
        sideplots = getattr(self, "SidePlots", None)
        if sideplots is None:
            return
        
        # Notify export plugin
        plugin = getattr(sideplots, "export_fovs_output", None)
        if hasattr(plugin, "on_viewer_mask_outline_change"):
            try:
                plugin.on_viewer_mask_outline_change(thickness)
            except Exception:
                if self._debug:
                    print("[viewer] Plugin notification for mask outline change failed")
        
        # Notify cell gallery plugin
        cell_gallery_plugin = getattr(sideplots, "cell_gallery_output", None)
        if hasattr(cell_gallery_plugin, "on_viewer_mask_outline_change"):
            try:
                cell_gallery_plugin.on_viewer_mask_outline_change(thickness)
            except Exception:
                if self._debug:
                    print("[viewer] Cell gallery notification for mask outline change failed")

    def on_plugin_mask_outline_change(self, thickness: int) -> None:
        try:
            thickness = max(1, int(thickness))
        except (TypeError, ValueError):
            thickness = 1
        if thickness == int(getattr(self, "mask_outline_thickness", 1)):
            return
        self.mask_outline_thickness = thickness
        self._set_mask_outline_slider_value(thickness)
        if getattr(self, "initialized", False):
            try:
                self.update_display(self.current_downsample_factor)
            except Exception:
                if self._debug:
                    print("[viewer] Failed to refresh display after plugin mask outline change")
        self._notify_plugins_mask_outline_changed(thickness)
        
    
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

    def _apply_marker_set(self, set_name: str, *, silent: bool = False) -> bool:
        if not set_name:
            if not silent:
                print("Please provide a valid marker set name.")
            return False

        marker_set = self.marker_sets.get(set_name)
        if marker_set is None:
            if not silent:
                print(f"Marker set '{set_name}' not found.")
            return False

        selected_channels = tuple(marker_set.get('selected_channels', ()))
        if not selected_channels:
            if not silent:
                print(f"Marker set '{set_name}' has no channels saved.")
            return False

        channel_settings = marker_set.get('channel_settings', {}) or {}

        for ch in selected_channels:
            settings = channel_settings.get(ch, {}) if isinstance(channel_settings, dict) else {}
            saved_max = settings.get('contrast_max') if isinstance(settings, dict) else None
            if saved_max is not None:
                self._merge_channel_max(ch, saved_max, sync=False)

        selector = getattr(self.ui_component, 'channel_selector', None)
        if selector is None:
            if not silent:
                print("Channel selector widget unavailable; cannot load marker set.")
            return False

        selector.value = selected_channels
        self.update_controls(None)

        color_controls = getattr(self.ui_component, 'color_controls', {}) or {}
        min_controls = getattr(self.ui_component, 'contrast_min_controls', {}) or {}
        max_controls = getattr(self.ui_component, 'contrast_max_controls', {}) or {}

        for ch in selected_channels:
            settings = channel_settings.get(ch, {}) if isinstance(channel_settings, dict) else {}

            self._sync_channel_controls(ch)

            color_control = color_controls.get(ch)
            min_control = min_controls.get(ch)
            max_control = max_controls.get(ch)

            if color_control is not None and isinstance(settings, dict) and 'color' in settings:
                color_control.value = settings['color']
            if min_control is not None and isinstance(settings, dict) and 'contrast_min' in settings:
                min_control.value = settings['contrast_min']
            if max_control is not None and isinstance(settings, dict) and 'contrast_max' in settings:
                max_control.value = settings['contrast_max']

            if isinstance(settings, dict) and 'contrast_max' in settings:
                self._merge_channel_max(ch, settings['contrast_max'])

        try:
            self.update_display(self.current_downsample_factor)
        except Exception:  # pragma: no cover - rendering safeguards
            if not silent:
                print("Marker set applied, but display refresh failed.")

        self.active_marker_set_name = set_name
        if not silent:
            print(f"Marker set '{set_name}' loaded.")
        return True

    def load_marker_set(self, button):
        set_name = self.ui_component.marker_set_dropdown.value
        if not set_name:
            print("No marker set selected to load.")
            return
        self._apply_marker_set(set_name, silent=False)

    def apply_marker_set_by_name(self, set_name: str) -> bool:
        return self._apply_marker_set(set_name, silent=True)

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

    def export_fovs_batch(self, marker_set_name, output_dir=None, fovs=None, surfix=None,
                          file_format='png', downsample_factor=None, dpi=300,
                          figure_size=None, show_progress=True):
        """
        Export multiple FOVs using a specified marker set.

        Parameters
        ----------
        marker_set_name : str
            Name of the marker set to use for visualisation.
        output_dir : str, optional
            Directory to save the exported images. If ``None``, creates a
            subdirectory in the base folder.
        fovs : list, optional
            List of FOV names to export. If ``None``, exports all available FOVs.
        file_format : str, optional
            File format for the exported images (``'png'``, ``'jpg'``, ``'tif'``,
            ``'pdf'``). Default is ``'png'``.
        downsample_factor : int, optional
            Downsample factor to use. If ``None``, uses the current downsample
            factor from the viewer.
        dpi : int, optional
            Resolution for the exported images. Default is 300.
        figure_size : tuple, optional
            Size of the figure in inches (width, height). If ``None``, uses the
            current figure size.
        show_progress : bool, optional
            Whether to log progress information. Default is ``True``.

        Returns
        -------
        dict
            Dictionary mapping FOV names to export status (``True`` for
            success, error message string for failure).
        """

        if marker_set_name not in self.marker_sets:
            raise ValueError(f"Marker set '{marker_set_name}' not found.")

        if output_dir is None:
            output_dir = os.path.join(self.base_folder, f"exported_{marker_set_name}")
        os.makedirs(output_dir, exist_ok=True)

        if fovs is None:
            fovs = list(self.available_fovs)
        elif isinstance(fovs, str):
            fovs = [fovs]
        else:
            fovs = list(fovs)

        requested_downsample = (
            int(downsample_factor)
            if downsample_factor is not None
            else int(self.current_downsample_factor)
        )
        if requested_downsample < 1:
            requested_downsample = 1

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

        marker_set = self.marker_sets[marker_set_name]
        selected_channels = tuple(marker_set["selected_channels"])
        channel_settings = marker_set["channel_settings"]

        self.ui_component.channel_selector.value = selected_channels
        self.update_controls(None)

        for ch in selected_channels:
            if ch in self.ui_component.color_controls and ch in channel_settings:
                self.ui_component.color_controls[ch].value = channel_settings[ch]["color"]
                self.ui_component.contrast_min_controls[ch].value = channel_settings[ch]["contrast_min"]
                self.ui_component.contrast_max_controls[ch].value = channel_settings[ch]["contrast_max"]

        if figure_size is not None:
            self.image_display.fig.set_size_inches(figure_size)

        overrides = {
            "downsample_factor": requested_downsample,
            "dpi": dpi,
            "figure_size": figure_size,
            "surfix": surfix,
        }

        def progress_callback(status):
            if not show_progress:
                return
            logger.info(
                "Export progress %d/%d (succeeded=%d, failed=%d)",
                status.completed,
                status.total,
                status.succeeded,
                status.failed,
            )

        job_items = []

        for fov in fovs:
            if surfix is not None:
                output_path = os.path.join(output_dir, f"{fov}_{surfix}.{file_format}")
            else:
                output_path = os.path.join(output_dir, f"{fov}.{file_format}")

            metadata = {
                "fov": fov,
                "marker_set": marker_set_name,
                "mode": "full_fov",
                "file_format": file_format,
            }

            def make_worker(fov_name=fov, output_path_str=output_path):
                def worker():
                    self.load_fov(fov_name, selected_channels)
                    fov_images = self.image_cache[fov_name]

                    missing = [ch for ch in selected_channels if fov_images.get(ch) is None]
                    available_channels = [ch for ch in selected_channels if fov_images.get(ch) is not None]
                    if missing and available_channels:
                        # Log and continue with the subset of available channels instead of failing the whole FOV.
                        logger.warning(
                            "FOV '%s': missing channels %s — exporting with available channels %s",
                            fov_name,
                            ", ".join(missing),
                            ", ".join(available_channels),
                        )
                    if not available_channels:
                        raise ValueError("No valid channels found in this FOV")

                    first_channel = next((fov_images[ch] for ch in available_channels), None)
                    if first_channel is None:
                        raise ValueError("No valid channels found in this FOV")

                    height, width = first_channel.shape
                    self.ui_component.image_selector.value = fov_name

                    xmin, xmax = 0, width
                    ymin, ymax = 0, height

                    width_cap = max(1, width // 2)
                    height_cap = max(1, height // 2)
                    max_downsample = max(1, min(width_cap, height_cap))
                    local_downsample = max(1, min(requested_downsample, max_downsample))

                    xmin_ds = xmin // local_downsample
                    xmax_ds = xmin_ds + max(1, (xmax - xmin) // local_downsample)
                    ymin_ds = ymin // local_downsample
                    ymax_ds = ymin_ds + max(1, (ymax - ymin) // local_downsample)

                    xym = (xmin, xmax, ymin, ymax)
                    xym_ds = (xmin_ds, xmax_ds, ymin_ds, ymax_ds)

                    combined = self.render_image(selected_channels, local_downsample, xym, xym_ds)
                    img_array = np.asarray(combined, dtype=np.float32)
                    img_array = np.clip(img_array, 0.0, 1.0)
                    img_array = (img_array * 255).astype(np.uint8)

                    if img_array.shape[0] < 2 or img_array.shape[1] < 2:
                        raise ValueError(f"Image too small: {img_array.shape}")

                    imsave(output_path_str, img_array)
                    return {"output_path": output_path_str}

                return worker

            job_items.append(
                JobItem(
                    item_id=fov,
                    execute=make_worker(),
                    output_path=output_path,
                    metadata=metadata,
                )
            )

        job = Job(
            mode="full_fov",
            items=job_items,
            marker_set=marker_set_name,
            output_dir=output_dir,
            file_format=file_format,
            overrides=overrides,
            progress_callback=progress_callback if show_progress else None,
        )

        status = None
        try:
            status = job.start()
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

        if status is None:
            status = job.status()

        failures = [item_id for item_id, result in status.results.items() if not result.ok]
        if show_progress:
            if failures:
                logger.info(
                    "Export complete with failures (%d/%d succeeded). Failures: %s",
                    status.succeeded,
                    status.total,
                    ", ".join(failures),
                )
            else:
                logger.info(
                    "Export complete: %d/%d items written to %s",
                    status.succeeded,
                    status.total,
                    output_dir,
                )

        legacy_results = {}
        for item_id, result in status.results.items():
            legacy_results[item_id] = True if result.ok else (result.error or "Export failed")

        return legacy_results