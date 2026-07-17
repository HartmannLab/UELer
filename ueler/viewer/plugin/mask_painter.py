# viewer/plugin/mask_painter.py
from __future__ import annotations

import importlib
import logging
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from ipywidgets import (
    Accordion,
    BoundedFloatText,
    BoundedIntText,
    Button,
    Checkbox,
    ColorPicker,
    Dropdown,
    FloatText,
    HTML,
    HBox,
    Label,
    Layout,
    Output,
    Tab,
    TagsInput,
    Text,
    ToggleButtons,
    VBox,
)

_FileChooserModule = None
try:  # pragma: no cover - optional dependency
    _FileChooserModule = importlib.import_module("ipyfilechooser")
except Exception:  # pragma: no cover - executed when ipyfilechooser is absent
    _FileChooserModule = None

FileChooser = getattr(_FileChooserModule, "FileChooser", None)

from ueler.viewer.decorators import update_status_bar
from ueler.viewer.plugin.plugin_base import PluginBase
from ueler.viewer.plugin.mask_class_list_widget import MaskClassListWidget
from ueler.viewer.color_palettes import DEFAULT_COLOR, colors_match, normalize_hex_color
from ueler.viewer.palette_store import (
    PaletteStoreError,
    load_registry as load_palette_registry,
    read_palette_file,
    resolve_palette_path,
    save_registry as save_palette_registry,
    slugify_name as shared_slugify_name,
    write_palette_file,
)
from ueler.rendering import MaskPainterSnapshot, set_cell_color, get_cell_color, clear_cell_colors, set_cell_colors_bulk

_logger = logging.getLogger(__name__)
COLOR_SET_FILE_SUFFIX = ".maskcolors.json"
REGISTRY_FILENAME = "mask_color_sets_index.json"
COLOR_SET_VERSION = "1.2.0"
FILL_OPACITY_DEFAULT_PERCENT = 35
BORDER_COLOR_MODE_MASK_TYPE = "mask_type_color"
BORDER_COLOR_MODE_SAME_AS_FILL = "same_as_fill"


def _normalise_opacity_percent(value: object, default: int = FILL_OPACITY_DEFAULT_PERCENT) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    return max(0, min(100, int(round(numeric))))


def _opacity_percent_to_alpha(value: object, default: int = FILL_OPACITY_DEFAULT_PERCENT) -> float:
    return _normalise_opacity_percent(value, default) / 100.0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _resolve_mask_type_color(main_viewer, mask_name: Optional[str], fallback: str = DEFAULT_COLOR) -> str:
    if not mask_name:
        return fallback

    controls = getattr(getattr(main_viewer, "ui_component", None), "mask_color_controls", {}) or {}
    widget = controls.get(str(mask_name))
    raw_value = getattr(widget, "value", None)
    if not raw_value:
        return fallback

    predefined = getattr(main_viewer, "predefined_colors", {}) or {}
    resolved = predefined.get(raw_value, raw_value)
    return normalize_hex_color(resolved) or resolved or fallback


def _resolve_color_dropdown_option(main_viewer, stored_value: object) -> Optional[str]:
    if stored_value is None:
        return None

    value = str(stored_value).strip()
    if not value:
        return None

    predefined = getattr(main_viewer, "predefined_colors", {}) or {}
    if value in predefined:
        return value

    normalized = normalize_hex_color(value)
    if normalized is None:
        return None

    for option, colour_hex in predefined.items():
        if normalize_hex_color(colour_hex) == normalized:
            return option
    return None


class ColorSetError(PaletteStoreError):
    """Raised when saving or loading a color set fails."""


def slugify_name(name: str) -> str:
    return shared_slugify_name(name, default_slug="mask-colors")


def serialize_class_color_controls(
    class_color_controls: Mapping[str, object],
    class_order: Sequence[str],
    default_color: str = DEFAULT_COLOR,
    hidden_cache: Optional[Mapping[str, str]] = None,
) -> "OrderedDict[str, str]":
    """Collect colors from the UI controls with guaranteed defaults."""

    serialized: "OrderedDict[str, str]" = OrderedDict()
    seen = set()
    for cls in class_order:
        key = str(cls)
        widget = class_color_controls.get(key)
        value = getattr(widget, "value", None)
        if hidden_cache and key in hidden_cache:
            value = hidden_cache[key]
        serialized[key] = value or default_color
        seen.add(key)

    for key, widget in class_color_controls.items():
        if key in seen:
            continue
        value = getattr(widget, "value", None)
        if hidden_cache and key in hidden_cache:
            value = hidden_cache[key]
        serialized[key] = value or default_color

    return serialized


def split_default_classes(
    class_keys: Sequence[str],
    color_map: Mapping[str, str],
    default_color: str,
) -> Tuple[List[str], List[str]]:
    non_default: List[str] = []
    defaulted: List[str] = []
    for key in class_keys:
        color = color_map.get(key)
        if color is None:
            color = default_color
        if colors_match(color, default_color):
            defaulted.append(key)
        else:
            non_default.append(key)
    return non_default, defaulted


CONTINUOUS_COLORMAPS: Tuple[str, ...] = (
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "coolwarm",
    "RdBu_r",
    "turbo",
    "Greys",
    "hot",
)


def _get_colormap(name: str):
    """Return a matplotlib colormap by name, compatible across versions."""
    try:  # matplotlib >= 3.5
        import matplotlib
        return matplotlib.colormaps[name]
    except Exception:  # pragma: no cover - fallback for older matplotlib
        from matplotlib import cm
        return cm.get_cmap(name)


def _arcsinh_transform(values: np.ndarray, cofactor: float) -> np.ndarray:
    """Apply an arcsinh(value / cofactor) transform, guarding the cofactor."""
    factor = float(cofactor)
    if not np.isfinite(factor) or factor <= 0:
        factor = 1.0
    return np.arcsinh(values / factor)


def resolve_continuous_range(
    values,
    arcsinh: bool = False,
    cofactor: float = 5.0,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> Tuple[float, float]:
    """Resolve a (vmin, vmax) range for continuous coloring.

    The range is computed in the (optionally arcsinh-transformed) value space
    using percentiles. Intended to be computed once over the *whole* column so
    a given value maps to the same color in every FOV. Handles the all-NaN and
    constant-column edge cases without raising.
    """
    arr = np.asarray(values, dtype=float)
    if arcsinh:
        arr = _arcsinh_transform(arr, cofactor)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, lo_pct))
    vmax = float(np.nanpercentile(finite, hi_pct))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        # Constant or degenerate column: widen so Normalize has non-zero width.
        base = vmin if np.isfinite(vmin) else 0.0
        return base, base + 1.0
    return vmin, vmax


def compute_continuous_colors(
    values,
    mask_ids,
    *,
    colormap: str,
    vmin: float,
    vmax: float,
    arcsinh: bool = False,
    cofactor: float = 5.0,
) -> Dict[int, str]:
    """Map continuous values to per-cell hex colors via a matplotlib colormap.

    Returns ``{mask_id: hex}``. NaN/non-finite values are skipped (omitted from
    the dict) so those cells fall through to the base mask, mirroring how
    hidden/inactive cells behave in the categorical path.
    """
    from matplotlib.colors import Normalize

    arr = np.asarray(values, dtype=float)
    ids = np.asarray(mask_ids)
    if arr.size == 0:
        return {}
    if arcsinh:
        arr = _arcsinh_transform(arr, cofactor)

    lo = float(vmin)
    hi = float(vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return {}

    norm = Normalize(vmin=lo, vmax=hi, clip=True)
    cmap = _get_colormap(colormap)
    # Sample the colormap only for finite cells, then build hex strings
    # vectorized (avoids a per-cell matplotlib ``to_hex`` call, which is the
    # dominant cost for large FOVs / map mode).
    rgba = cmap(norm(arr[finite_mask]))
    rgb = (np.clip(rgba[:, :3], 0.0, 1.0) * 255).round().astype(int)
    hexes = ["#{:02x}{:02x}{:02x}".format(r, g, b) for r, g, b in rgb]
    finite_ids = ids[finite_mask]
    return {int(mid): hx for mid, hx in zip(finite_ids, hexes)}


def build_painter_state_maps_for_fov(
    *,
    cell_table,
    fov_key: str,
    label_key: str,
    fov: Optional[str],
    identifier: Optional[str],
    active_classes: Sequence[str],
    class_colors: Mapping[str, str],
    class_visible: Mapping[str, bool],
    class_fill: Mapping[str, bool],
    class_opacity: Mapping[str, int],
    default_color: str,
    global_fill: bool = False,
    global_fill_opacity: int,
    border_color_mode: str = BORDER_COLOR_MODE_MASK_TYPE,
    mask_type_color: str = DEFAULT_COLOR,
    continuous: Optional[Mapping[str, object]] = None,
) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str], Dict[int, float]]:
    """Resolve effective painter color/mode/opacity maps for one FOV.

    When ``continuous`` is provided it takes precedence over the categorical
    branch: cells are colored by mapping a numeric column through a colormap
    using the pre-resolved global ``vmin``/``vmax`` carried in the spec.
    """
    if continuous is not None:
        column = continuous.get("column")
        if cell_table is None or not fov or not column or column not in cell_table.columns:
            return {}, {}, {}, {}
        rows = cell_table.loc[cell_table[fov_key] == fov, [label_key, column]]
        if rows.empty:
            return {}, {}, {}, {}
        color_map = compute_continuous_colors(
            rows[column].to_numpy(),
            rows[label_key].to_numpy(),
            colormap=str(continuous.get("colormap", "viridis")),
            vmin=float(continuous.get("vmin", 0.0)),
            vmax=float(continuous.get("vmax", 1.0)),
            arcsinh=bool(continuous.get("arcsinh", False)),
            cofactor=float(continuous.get("cofactor", 5.0)),
        )
        mode = "fill" if bool(continuous.get("fill", True)) else "outline"
        mode_map = {mid: mode for mid in color_map}
        if mode == "fill":
            alpha = _opacity_percent_to_alpha(
                continuous.get("opacity", 100), _normalise_opacity_percent(continuous.get("opacity", 100))
            )
            opacity_map = {mid: alpha for mid in color_map}
        else:
            opacity_map = {}
        return color_map, {}, mode_map, opacity_map

    if cell_table is None or not fov or not identifier or identifier not in cell_table.columns:
        return {}, {}, {}, {}

    current_rows = cell_table.loc[
        cell_table[fov_key] == fov,
        [label_key, identifier],
    ]
    if current_rows.empty:
        return {}, {}, {}, {}

    active_set = {str(cls) for cls in active_classes}
    hidden_set = {
        cls for cls in active_set
        if not bool(class_visible.get(cls, True))
    }
    default_percent = _normalise_opacity_percent(global_fill_opacity)

    color_map: Dict[int, str] = {}
    border_map: Dict[int, str] = {}
    mode_map: Dict[int, str] = {}
    opacity_map: Dict[int, float] = {}

    resolved_border_mode = str(border_color_mode or BORDER_COLOR_MODE_MASK_TYPE)
    resolved_mask_type_color = normalize_hex_color(mask_type_color) or mask_type_color or default_color
    default_fill_enabled = bool(global_fill)

    for _, row in current_rows.iterrows():
        cls_key = str(row[identifier])
        mask_id = int(row[label_key])
        if cls_key in hidden_set:
            color_map[mask_id] = ""
            continue
        if cls_key not in active_set:
            color_map[mask_id] = default_color
            if default_fill_enabled:
                mode_map[mask_id] = "fill"
                border_map[mask_id] = (
                    default_color
                    if resolved_border_mode == BORDER_COLOR_MODE_SAME_AS_FILL
                    else resolved_mask_type_color
                )
                opacity_map[mask_id] = _opacity_percent_to_alpha(default_percent, default_percent)
            else:
                mode_map[mask_id] = "outline"
            continue

        class_color = class_colors.get(cls_key, default_color) or default_color
        color_map[mask_id] = class_color
        if bool(class_fill.get(cls_key, False)):
            mode_map[mask_id] = "fill"
            border_map[mask_id] = (
                class_color
                if resolved_border_mode == BORDER_COLOR_MODE_SAME_AS_FILL
                else resolved_mask_type_color
            )
            opacity_map[mask_id] = _opacity_percent_to_alpha(
                class_opacity.get(cls_key, default_percent),
                default_percent,
            )
        else:
            mode_map[mask_id] = "outline"

    return color_map, border_map, mode_map, opacity_map


def write_color_set_file(path: Path, payload: Mapping[str, object]) -> Path:
    return write_palette_file(path, payload)


def read_color_set_file(path: Path) -> Dict[str, object]:
    return read_palette_file(path)


def load_registry(folder: Path) -> Dict[str, Dict[str, str]]:
    return load_palette_registry(folder, REGISTRY_FILENAME)


def save_registry(folder: Path, records: Mapping[str, Mapping[str, str]]) -> Path:
    return save_palette_registry(folder, REGISTRY_FILENAME, records)


def apply_color_map_to_controls(
    color_map: Mapping[str, str],
    class_color_controls: Mapping[str, ColorPicker],
    default_color: str = DEFAULT_COLOR,
) -> None:
    for key, widget in class_color_controls.items():
        widget.value = color_map.get(key, default_color)


class MaskPainterDisplay(PluginBase):
    def __init__(self, main_viewer, width, height):
        super().__init__(main_viewer, width, height)
        self.SidePlots_id = "mask_painter_output"
        self.displayed_name = "Mask painter"
        self.main_viewer = main_viewer
        self.width = width
        self.height = height

        self.ui_component = UiComponent()
        self.default_color = normalize_hex_color(DEFAULT_COLOR) or DEFAULT_COLOR
        self.ui_component.default_color_picker.value = self.default_color
        self.ui_component.default_color_picker.observe(self.on_default_color_change, names="value")
        self.class_color_controls: Dict[str, ColorPicker] = {}
        self.class_visible_controls: Dict[str, Checkbox] = {}
        self.class_mode_controls: Dict[str, Checkbox] = {}
        self.class_opacity_controls: Dict[str, BoundedIntText] = {}
        self._mode_control_class_keys: Dict[int, str] = {}
        self._opacity_control_class_keys: Dict[int, str] = {}
        self._linked_fill_classes: set[str] = set()
        self._linked_opacity_classes: set[str] = set()
        self.current_classes: List[str] = []
        self._active_classes: List[str] = []
        self.current_identifier: Optional[str] = None
        self.selected_classes: List[str] = []
        self.hidden_color_cache: Dict[str, str] = {}
        self.registry_records: Dict[str, Dict[str, str]] = {}
        self.active_color_set_name = ""
        self._cell_mode_cache: Dict[str, Dict[int, str]] = {}  # fov -> {mask_id: mode}
        self._cell_opacity_cache: Dict[str, Dict[int, float]] = {}  # fov -> {mask_id: alpha}
        self._last_applied_class_modes: Dict[str, str] = {}
        self._last_applied_class_opacities: Dict[str, float] = {}
        self._syncing: bool = False  # guard against anywidget ↔ ipywidget sync loops
        
        # Track last applied state to avoid unnecessary re-application
        self._last_applied_fov: Optional[str] = None
        self._last_applied_identifier: Optional[str] = None
        self._last_applied_classes: Optional[set] = None
        self._last_applied_class_colors: dict[str, str] = {}
        self._state_maps_cache: Dict[str, tuple] = {}  # fov -> (color_map, border_map, mode_map, opacity_map)
        # Cached continuous auto-range: ((column, arcsinh, cofactor), (vmin, vmax)).
        # Keeps the full-column percentile off the render hot path (issue #115 reply).
        self._continuous_range_cache: Optional[Tuple[tuple, Tuple[float, float]]] = None

        storage_folder = self._determine_storage_folder()
        if storage_folder is None:
            storage_folder = Path.cwd()
            self._log(
                "Mask palette directory unavailable; using manual override path. Click 'Change location' to adjust.",
                error=True,
                clear=True,
            )
            self.ui_component.show_manual_folder_box()
        self.registry_folder = storage_folder.resolve()
        self.ui_component.set_folder_path(self.registry_folder)

        self._migrate_legacy_palettes(self.registry_folder)

        self._initialise_identifier_options()
        self._initialise_continuous_options()

        self.ui_component.update_button.on_click(self.apply_colors_to_masks)
        self.ui_component.identifier_dropdown.observe(self.on_identifier_change, names="value")
        self.ui_component.sorting_items_tagsinput.observe(self.on_sorting_items_change, names="value")
        self.ui_component.show_all_checkbox.observe(self.on_show_all_toggle, names="value")
        self.ui_component.save_button.on_click(self.save_current_color_set)
        self.ui_component.load_button.on_click(self.load_color_set_from_ui)
        self.ui_component.change_folder_button.on_click(self.on_change_folder_clicked)
        self.ui_component.manual_folder_apply.on_click(self.apply_manual_folder_override)
        self.ui_component.manual_folder_cancel.on_click(self.cancel_manual_folder_override)
        self.ui_component.apply_saved_button.on_click(self.apply_saved_color_set)
        self.ui_component.overwrite_saved_button.on_click(self.overwrite_saved_color_set)
        self.ui_component.delete_saved_button.on_click(self.delete_saved_color_set)
        self.ui_component.set_name_input.observe(self._refresh_save_button_state, names="value")
        self.ui_component.identifier_dropdown.observe(self._refresh_save_button_state, names="value")
        self.ui_component.only_specified_checkbox.observe(self._on_only_specified_toggle, names="value")
        self.ui_component.enabled_checkbox.observe(self._on_enabled_toggle, names="value")
        self.ui_component.global_fill_checkbox.observe(self._on_global_fill_toggle, names="value")
        self.ui_component.global_fill_opacity_input.observe(self._on_global_fill_opacity_change, names="value")
        self.ui_component.show_fill_borders_checkbox.observe(self._on_fill_border_toggle, names="value")
        self.ui_component.border_color_mode_dropdown.observe(self._on_fill_border_toggle, names="value")

        # Wire continuous-coloring controls (issue #115)
        self.ui_component.color_mode_toggle.observe(self._on_color_mode_change, names="value")
        self.ui_component.continuous_column_dropdown.observe(self._on_continuous_column_change, names="value")
        self.ui_component.colormap_dropdown.observe(self._on_continuous_param_change, names="value")
        self.ui_component.arcsinh_checkbox.observe(self._on_continuous_param_change, names="value")
        self.ui_component.arcsinh_cofactor_input.observe(self._on_continuous_param_change, names="value")
        self.ui_component.continuous_opacity_input.observe(self._on_continuous_param_change, names="value")
        self.ui_component.continuous_fill_checkbox.observe(self._on_continuous_param_change, names="value")
        self.ui_component.auto_range_checkbox.observe(self._on_auto_range_toggle, names="value")
        self.ui_component.vmin_input.observe(self._on_continuous_range_edit, names="value")
        self.ui_component.vmax_input.observe(self._on_continuous_range_edit, names="value")
        self.ui_component.autorange_button.on_click(self._recompute_continuous_range)

        # Wire anywidget class-list traitlet changes → Python state
        _w = self.ui_component.class_list_widget
        _w.observe(self._pull_from_widget, names=["class_order", "class_colors", "class_visible", "class_fill", "class_opacity"])
        _w.observe(self._on_add_requested, names=["add_requested"])
        _w.observe(self._on_remove_requested, names=["remove_requested"])

        self._load_registry_for_folder(self.registry_folder)
        self._refresh_save_button_state()
        self.initiate_ui()

    def _initialise_identifier_options(self) -> None:
        identifier_options: Iterable[str] = []
        if self.main_viewer.cell_table is not None:
            cell_table = self.main_viewer.cell_table
            identifier_options = cell_table.select_dtypes(include=["int", "int64", "object", "bool"]).columns.tolist()

        self.ui_component.identifier_dropdown.options = list(identifier_options)

    def _initialise_continuous_options(self) -> None:
        """Populate the continuous-value dropdown with the float columns."""
        continuous_options: Iterable[str] = []
        if self.main_viewer.cell_table is not None:
            cell_table = self.main_viewer.cell_table
            continuous_options = cell_table.select_dtypes(
                include=["float", "float32", "float64"]
            ).columns.tolist()
        self.ui_component.continuous_column_dropdown.options = list(continuous_options)

    # ------------------------------------------------------------------
    # Identifier and class handling
    # ------------------------------------------------------------------
    def on_identifier_change(self, change):
        identifier = change["new"]
        self.current_identifier = identifier
        if not identifier or self.main_viewer.cell_table is None:
            self._log("Select a valid identifier before editing colors.", clear=True)
            return

        column = self.main_viewer.cell_table[identifier]
        classes = column.dropna().astype(str).unique().tolist()
        classes.sort()
        self.current_classes = classes
        self._active_classes = classes[:min(len(classes), 6)]

        self.class_color_controls.clear()
        self.class_visible_controls.clear()
        self.class_mode_controls.clear()
        self.class_opacity_controls.clear()
        self._mode_control_class_keys.clear()
        self._opacity_control_class_keys.clear()
        self._linked_fill_classes.clear()
        self._linked_opacity_classes.clear()
        for cls in classes:
            picker = ColorPicker(description=str(cls), value=self.default_color, layout=Layout(width="200px"))
            self.class_color_controls[cls] = picker
            vis_cb = Checkbox(
                value=True,
                description="",
                indent=False,
                layout=Layout(width="30px"),
                tooltip=f"Show/hide {cls}",
            )
            vis_cb.observe(self._on_visibility_or_mode_change, names="value")
            self.class_visible_controls[cls] = vis_cb
            mode_cb = Checkbox(
                value=False,
                description="fill",
                indent=False,
                layout=Layout(width="60px"),
                tooltip=f"Render {cls} as filled (unchecked = outline)",
            )
            mode_cb.observe(self._on_visibility_or_mode_change, names="value")
            mode_cb.observe(self._on_class_mode_change, names="value")
            self.class_mode_controls[cls] = mode_cb
            self._mode_control_class_keys[id(mode_cb)] = cls
            opacity_input = BoundedIntText(
                value=_normalise_opacity_percent(self.ui_component.global_fill_opacity_input.value),
                min=0,
                max=100,
                step=1,
            )
            opacity_input.observe(self._on_visibility_or_mode_change, names="value")
            opacity_input.observe(self._on_class_opacity_change, names="value")
            self.class_opacity_controls[cls] = opacity_input
            self._opacity_control_class_keys[id(opacity_input)] = cls

        self._linked_fill_classes = set(classes)
        self._linked_opacity_classes = set(classes)

        # Sync state to the anywidget list
        self._push_to_widget()
        self._log(f"Identifier set to '{identifier}' with {len(classes)} classes.", clear=True)

    def on_sorting_items_change(self, change):
        # Legacy observer; kept for backward compatibility with test setups.
        # The anywidget handles ordering; this is a no-op when the widget is active.
        pass

    def on_show_all_toggle(self, change):
        # Legacy observer; kept for backward compatibility with test setups.
        pass

    def on_default_color_change(self, change):
        self._set_default_color(change.get("new"), update_ui=False)

    def _handle_selection_transition(self, new_order: Sequence[str]) -> None:
        # Legacy helper; kept for backward compatibility with test setups.
        self.selected_classes = list(new_order)

    def _refresh_color_picker_display(self, preferred_order: Optional[Sequence[str]] = None) -> None:
        # With the anywidget, display order is controlled via _push_to_widget().
        self._push_to_widget()

    def _set_default_color(self, new_color: Optional[str], update_ui: bool = True) -> None:
        normalized = normalize_hex_color(new_color) or DEFAULT_COLOR
        old_color = getattr(self, "default_color", DEFAULT_COLOR)
        self.default_color = normalized

        # Update any picker that was still showing the old default color
        for picker in self.class_color_controls.values():
            if colors_match(picker.value, old_color):
                picker.value = normalized

        if update_ui:
            picker = self.ui_component.default_color_picker
            picker.unobserve(self.on_default_color_change, names="value")
            try:
                picker.value = normalized
            finally:
                picker.observe(self.on_default_color_change, names="value")

        self._push_to_widget()

    def _get_active_classes(self) -> List[str]:
        """Return the classes currently surfaced in the UI, preserving order."""
        order = list(self.ui_component.class_list_widget.class_order)
        if order:
            return [key for key in order if key in self.class_color_controls]
        if self.ui_component.show_all_checkbox.value:
            ti_order = list(self.ui_component.sorting_items_tagsinput.value)
            return ti_order + [
                key for key in self.class_color_controls.keys()
                if key not in set(ti_order)
            ]
        fallback = list(self.selected_classes or self.ui_component.sorting_items_tagsinput.value)
        return [key for key in fallback if key in self.class_color_controls]

    def _get_visible_classes(self) -> List[str]:
        """Return classes that are currently visible (vis checkbox ticked), in display order."""
        order = self._get_active_classes()
        return [
            key for key in order
            if (
                self.class_visible_controls.get(key) is None
                or self.class_visible_controls[key].value
            )
        ]

    def _get_hidden_classes(self) -> List[str]:
        """Return active classes whose visibility checkbox is unchecked."""
        visible = set(self._get_visible_classes())
        return [key for key in self._get_active_classes() if key not in visible]

    def _get_inactive_classes(self) -> List[str]:
        """Return classes present in the dataset but not currently surfaced in the UI."""
        active = set(self._get_active_classes())
        return [key for key in self.current_classes if key not in active]

    def _get_customized_first_class_order(self) -> List[str]:
        color_map = {
            cls: getattr(widget, "value", self.default_color)
            for cls, widget in self.class_color_controls.items()
        }
        customized, defaulted = split_default_classes(
            self.current_classes,
            color_map,
            self.default_color,
        )
        return [
            cls for cls in (customized + defaulted)
            if cls in self.class_color_controls
        ]

    def _resolve_tracked_class_key(self, change, control_map: Mapping[int, str]) -> Optional[str]:
        owner = change.get("owner") if isinstance(change, dict) else None
        if owner is None:
            return None
        return control_map.get(id(owner))

    def _set_linked_classes_from_state(
        self,
        *,
        linked_fill_classes: Optional[Sequence[str]] = None,
        linked_opacity_classes: Optional[Sequence[str]] = None,
    ) -> None:
        class_keys = {str(cls) for cls in self.class_color_controls.keys()}
        global_fill = bool(getattr(self.ui_component.global_fill_checkbox, "value", False))
        global_opacity = _normalise_opacity_percent(getattr(self.ui_component.global_fill_opacity_input, "value", FILL_OPACITY_DEFAULT_PERCENT))

        if linked_fill_classes is None:
            self._linked_fill_classes = {
                cls for cls in class_keys
                if bool(getattr(self.class_mode_controls.get(cls), "value", False)) == global_fill
            }
        else:
            self._linked_fill_classes = {str(cls) for cls in linked_fill_classes if str(cls) in class_keys}

        if linked_opacity_classes is None:
            self._linked_opacity_classes = {
                cls for cls in class_keys
                if _normalise_opacity_percent(getattr(self.class_opacity_controls.get(cls), "value", global_opacity), global_opacity) == global_opacity
            }
        else:
            self._linked_opacity_classes = {str(cls) for cls in linked_opacity_classes if str(cls) in class_keys}

    def _build_color_set_payload(self, name: str, timestamp: str) -> Dict[str, object]:
        class_order = self._get_full_class_order()
        color_map = serialize_class_color_controls(
            self.class_color_controls,
            class_order,
            self.default_color,
            hidden_cache=self.hidden_color_cache,
        )
        modes_map = {
            cls: ("fill" if getattr(self.class_mode_controls.get(cls), "value", False) else "outline")
            for cls in class_order
        }
        visible_map = {
            cls: bool(getattr(self.class_visible_controls.get(cls), "value", True))
            for cls in class_order
        }
        return {
            "name": name,
            "version": COLOR_SET_VERSION,
            "identifier": self.current_identifier,
            "default_color": self.default_color,
            "class_order": list(class_order),
            "active_classes": list(self._get_active_classes()),
            "only_specified": bool(self.ui_component.only_specified_checkbox.value),
            "colors": color_map,
            "modes": modes_map,
            "visible": visible_map,
            "opacities": {
                cls: _normalise_opacity_percent(getattr(self.class_opacity_controls.get(cls), "value", FILL_OPACITY_DEFAULT_PERCENT))
                for cls in class_order
            },
            "linked_fill_classes": [cls for cls in class_order if cls in self._linked_fill_classes],
            "linked_opacity_classes": [cls for cls in class_order if cls in self._linked_opacity_classes],
            "global_fill": bool(self.ui_component.global_fill_checkbox.value),
            "global_fill_opacity": _normalise_opacity_percent(self.ui_component.global_fill_opacity_input.value),
            "show_fill_borders": bool(self.ui_component.show_fill_borders_checkbox.value),
            "border_color_mode": self.get_border_color_mode(),
            "color_mode": str(self.ui_component.color_mode_toggle.value or "categorical"),
            "continuous": {
                "column": str(self.ui_component.continuous_column_dropdown.value or ""),
                "colormap": str(self.ui_component.colormap_dropdown.value or "viridis"),
                "vmin": _safe_float(self.ui_component.vmin_input.value, 0.0),
                "vmax": _safe_float(self.ui_component.vmax_input.value, 1.0),
                "arcsinh": bool(self.ui_component.arcsinh_checkbox.value),
                "cofactor": _safe_float(self.ui_component.arcsinh_cofactor_input.value, 5.0),
                "auto_range": bool(self.ui_component.auto_range_checkbox.value),
                "opacity": _normalise_opacity_percent(self.ui_component.continuous_opacity_input.value),
                "fill": bool(self.ui_component.continuous_fill_checkbox.value),
            },
            "saved_at": timestamp,
        }

    def get_effective_color_map_for_fov(self, fov: Optional[str] = None) -> Dict[int, str]:
        """Build the current single-FOV painter color map directly from UI state."""
        current_fov = fov or self.main_viewer.get_active_fov()
        color_map, _, _, _ = build_painter_state_maps_for_fov(
            cell_table=self.main_viewer.cell_table,
            fov_key=self.main_viewer.fov_key,
            label_key=self.main_viewer.label_key,
            fov=current_fov,
            identifier=self.ui_component.identifier_dropdown.value,
            active_classes=self._get_active_classes(),
            class_colors={cls: getattr(widget, "value", self.default_color) for cls, widget in self.class_color_controls.items()},
            class_visible={cls: bool(getattr(widget, "value", True)) for cls, widget in self.class_visible_controls.items()},
            class_fill={cls: bool(getattr(widget, "value", False)) for cls, widget in self.class_mode_controls.items()},
            class_opacity={cls: _normalise_opacity_percent(getattr(widget, "value", FILL_OPACITY_DEFAULT_PERCENT)) for cls, widget in self.class_opacity_controls.items()},
            default_color=self.default_color,
            global_fill=bool(self.ui_component.global_fill_checkbox.value),
            global_fill_opacity=_normalise_opacity_percent(self.ui_component.global_fill_opacity_input.value),
            border_color_mode=self.get_border_color_mode(),
            mask_type_color=self.get_mask_type_color(),
            continuous=self._active_continuous_spec(),
        )
        return color_map

    def get_effective_border_color_map_for_fov(self, fov: Optional[str] = None) -> Dict[int, str]:
        """Build the current single-FOV painter border color map directly from UI state."""
        current_fov = fov or self.main_viewer.get_active_fov()
        _, border_map, _, _ = build_painter_state_maps_for_fov(
            cell_table=self.main_viewer.cell_table,
            fov_key=self.main_viewer.fov_key,
            label_key=self.main_viewer.label_key,
            fov=current_fov,
            identifier=self.ui_component.identifier_dropdown.value,
            active_classes=self._get_active_classes(),
            class_colors={cls: getattr(widget, "value", self.default_color) for cls, widget in self.class_color_controls.items()},
            class_visible={cls: bool(getattr(widget, "value", True)) for cls, widget in self.class_visible_controls.items()},
            class_fill={cls: bool(getattr(widget, "value", False)) for cls, widget in self.class_mode_controls.items()},
            class_opacity={cls: _normalise_opacity_percent(getattr(widget, "value", FILL_OPACITY_DEFAULT_PERCENT)) for cls, widget in self.class_opacity_controls.items()},
            default_color=self.default_color,
            global_fill=bool(self.ui_component.global_fill_checkbox.value),
            global_fill_opacity=_normalise_opacity_percent(self.ui_component.global_fill_opacity_input.value),
            border_color_mode=self.get_border_color_mode(),
            mask_type_color=self.get_mask_type_color(),
            continuous=self._active_continuous_spec(),
        )
        return border_map

    def get_effective_mode_map_for_fov(self, fov: Optional[str] = None) -> Dict[int, str]:
        """Build the current single-FOV painter mode map directly from UI state."""
        current_fov = fov or self.main_viewer.get_active_fov()
        _, _, mode_map, _ = build_painter_state_maps_for_fov(
            cell_table=self.main_viewer.cell_table,
            fov_key=self.main_viewer.fov_key,
            label_key=self.main_viewer.label_key,
            fov=current_fov,
            identifier=self.ui_component.identifier_dropdown.value,
            active_classes=self._get_active_classes(),
            class_colors={cls: getattr(widget, "value", self.default_color) for cls, widget in self.class_color_controls.items()},
            class_visible={cls: bool(getattr(widget, "value", True)) for cls, widget in self.class_visible_controls.items()},
            class_fill={cls: bool(getattr(widget, "value", False)) for cls, widget in self.class_mode_controls.items()},
            class_opacity={cls: _normalise_opacity_percent(getattr(widget, "value", FILL_OPACITY_DEFAULT_PERCENT)) for cls, widget in self.class_opacity_controls.items()},
            default_color=self.default_color,
            global_fill=bool(self.ui_component.global_fill_checkbox.value),
            global_fill_opacity=_normalise_opacity_percent(self.ui_component.global_fill_opacity_input.value),
            border_color_mode=self.get_border_color_mode(),
            mask_type_color=self.get_mask_type_color(),
            continuous=self._active_continuous_spec(),
        )
        return mode_map

    def get_effective_opacity_map_for_fov(self, fov: Optional[str] = None) -> Dict[int, float]:
        """Build the current single-FOV painter opacity map directly from UI state."""
        current_fov = fov or self.main_viewer.get_active_fov()
        _, _, _, opacity_map = build_painter_state_maps_for_fov(
            cell_table=self.main_viewer.cell_table,
            fov_key=self.main_viewer.fov_key,
            label_key=self.main_viewer.label_key,
            fov=current_fov,
            identifier=self.ui_component.identifier_dropdown.value,
            active_classes=self._get_active_classes(),
            class_colors={cls: getattr(widget, "value", self.default_color) for cls, widget in self.class_color_controls.items()},
            class_visible={cls: bool(getattr(widget, "value", True)) for cls, widget in self.class_visible_controls.items()},
            class_fill={cls: bool(getattr(widget, "value", False)) for cls, widget in self.class_mode_controls.items()},
            class_opacity={cls: _normalise_opacity_percent(getattr(widget, "value", FILL_OPACITY_DEFAULT_PERCENT)) for cls, widget in self.class_opacity_controls.items()},
            default_color=self.default_color,
            global_fill=bool(self.ui_component.global_fill_checkbox.value),
            global_fill_opacity=_normalise_opacity_percent(self.ui_component.global_fill_opacity_input.value),
            border_color_mode=self.get_border_color_mode(),
            mask_type_color=self.get_mask_type_color(),
            continuous=self._active_continuous_spec(),
        )
        return opacity_map

    def _invalidate_state_maps_cache(self) -> None:
        """Clear the per-FOV state-maps cache. Call whenever painter UI state changes."""
        self._state_maps_cache.clear()

    def get_effective_state_maps_for_fov(
        self, fov: Optional[str] = None
    ) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str], Dict[int, float]]:
        """Return (color_map, border_map, mode_map, opacity_map) from a single cell-table pass.

        Results are cached per FOV and invalidated by _invalidate_state_maps_cache() whenever
        painter UI state changes, so repeated renders of the same FOV (pan/zoom) are O(1).
        """
        current_fov = fov or self.main_viewer.get_active_fov()
        cached = self._state_maps_cache.get(str(current_fov) if current_fov else "")
        if cached is not None:
            return cached
        result = build_painter_state_maps_for_fov(
            cell_table=self.main_viewer.cell_table,
            fov_key=self.main_viewer.fov_key,
            label_key=self.main_viewer.label_key,
            fov=current_fov,
            identifier=self.ui_component.identifier_dropdown.value,
            active_classes=self._get_active_classes(),
            class_colors={cls: getattr(widget, "value", self.default_color) for cls, widget in self.class_color_controls.items()},
            class_visible={cls: bool(getattr(widget, "value", True)) for cls, widget in self.class_visible_controls.items()},
            class_fill={cls: bool(getattr(widget, "value", False)) for cls, widget in self.class_mode_controls.items()},
            class_opacity={cls: _normalise_opacity_percent(getattr(widget, "value", FILL_OPACITY_DEFAULT_PERCENT)) for cls, widget in self.class_opacity_controls.items()},
            default_color=self.default_color,
            global_fill=bool(self.ui_component.global_fill_checkbox.value),
            global_fill_opacity=_normalise_opacity_percent(self.ui_component.global_fill_opacity_input.value),
            border_color_mode=self.get_border_color_mode(),
            mask_type_color=self.get_mask_type_color(),
            continuous=self._active_continuous_spec(),
        )
        self._state_maps_cache[str(current_fov) if current_fov else ""] = result
        return result

    def get_opacity_map_for_fov(self, fov: str) -> Dict[int, float]:
        return dict(self._cell_opacity_cache.get(str(fov), {}))

    def get_show_borders_on_filled(self) -> bool:
        return bool(getattr(self.ui_component.show_fill_borders_checkbox, "value", False))

    def get_border_color_mode(self) -> str:
        value = getattr(self.ui_component.border_color_mode_dropdown, "value", BORDER_COLOR_MODE_MASK_TYPE)
        if value == BORDER_COLOR_MODE_SAME_AS_FILL:
            return BORDER_COLOR_MODE_SAME_AS_FILL
        return BORDER_COLOR_MODE_MASK_TYPE

    def get_mask_type_color(self) -> str:
        return _resolve_mask_type_color(
            self.main_viewer,
            getattr(self.main_viewer, "mask_key", None),
            fallback=self.default_color,
        )

    def capture_snapshot(self) -> Optional[MaskPainterSnapshot]:
        color_mode = self.ui_component.color_mode_toggle.value
        identifier = self.ui_component.identifier_dropdown.value
        continuous_column = self.ui_component.continuous_column_dropdown.value
        # Continuous mode has no per-class controls; guard on the value column instead.
        if color_mode == "continuous":
            if not continuous_column:
                return None
        elif not identifier or not self.class_color_controls:
            return None

        mask_type_color = self.get_mask_type_color()
        arcsinh = bool(self.ui_component.arcsinh_checkbox.value)
        cofactor = _safe_float(self.ui_component.arcsinh_cofactor_input.value, 5.0)
        vmin = _safe_float(self.ui_component.vmin_input.value, 0.0)
        vmax = _safe_float(self.ui_component.vmax_input.value, 1.0)
        if color_mode == "continuous" and continuous_column:
            spec = self._build_continuous_spec()
            if spec is not None:
                vmin, vmax = float(spec["vmin"]), float(spec["vmax"])

        return MaskPainterSnapshot(
            mask_name=str(getattr(self.main_viewer, "mask_key", "") or ""),
            identifier=str(identifier or ""),
            active_classes=tuple(str(cls) for cls in self._get_active_classes()),
            class_colors={cls: getattr(widget, "value", self.default_color) or self.default_color for cls, widget in self.class_color_controls.items()},
            class_visible={cls: bool(getattr(widget, "value", True)) for cls, widget in self.class_visible_controls.items()},
            class_fill={cls: bool(getattr(widget, "value", False)) for cls, widget in self.class_mode_controls.items()},
            class_opacity={cls: _normalise_opacity_percent(getattr(widget, "value", FILL_OPACITY_DEFAULT_PERCENT)) for cls, widget in self.class_opacity_controls.items()},
            default_color=self.default_color,
            global_fill=bool(self.ui_component.global_fill_checkbox.value),
            global_fill_opacity=_normalise_opacity_percent(self.ui_component.global_fill_opacity_input.value),
            show_borders_on_filled=bool(self.ui_component.show_fill_borders_checkbox.value),
            border_color_mode=self.get_border_color_mode(),
            mask_type_color=mask_type_color,
            outline_thickness=int(getattr(self.main_viewer, "mask_outline_thickness", 1)),
            color_mode=str(color_mode),
            continuous_column=str(continuous_column or ""),
            colormap=str(self.ui_component.colormap_dropdown.value or "viridis"),
            vmin=vmin,
            vmax=vmax,
            arcsinh=arcsinh,
            arcsinh_cofactor=cofactor,
            continuous_opacity=_normalise_opacity_percent(self.ui_component.continuous_opacity_input.value),
            continuous_fill=bool(self.ui_component.continuous_fill_checkbox.value),
        )

    def apply_snapshot(self, snapshot: Optional[MaskPainterSnapshot]) -> bool:
        if snapshot is None:
            return False

        if str(getattr(snapshot, "color_mode", "categorical")) == "continuous":
            return self._apply_continuous_snapshot(snapshot)

        identifier = str(getattr(snapshot, "identifier", "") or "")
        if not identifier:
            return False

        options = tuple(getattr(self.ui_component.identifier_dropdown, "options", ()) or ())
        if identifier not in options:
            return False

        self.default_color = getattr(snapshot, "default_color", self.default_color) or self.default_color
        if self.ui_component.identifier_dropdown.value != identifier:
            self.ui_component.identifier_dropdown.value = identifier
        elif not self.class_color_controls:
            self.on_identifier_change({"new": identifier})

        if not self.class_color_controls:
            return False

        active_set = {str(cls) for cls in getattr(snapshot, "active_classes", ())}
        available = tuple(
            cls for cls in self.current_classes
            if str(cls) in active_set
        )

        self._syncing = True
        try:
            class_selector = getattr(self.ui_component, "class_selector", None)
            if class_selector is not None:
                class_selector.value = available
            self.ui_component.global_fill_checkbox.value = bool(
                getattr(snapshot, "global_fill", False)
            )
            self.ui_component.global_fill_opacity_input.value = _normalise_opacity_percent(
                getattr(snapshot, "global_fill_opacity", FILL_OPACITY_DEFAULT_PERCENT)
            )
            self.ui_component.show_fill_borders_checkbox.value = bool(
                getattr(snapshot, "show_borders_on_filled", False)
            )
            self.ui_component.border_color_mode_dropdown.value = str(
                getattr(snapshot, "border_color_mode", BORDER_COLOR_MODE_MASK_TYPE) or BORDER_COLOR_MODE_MASK_TYPE
            )

            mask_name = str(getattr(snapshot, "mask_name", "") or getattr(self.main_viewer, "mask_key", "") or "")
            if mask_name:
                color_controls = getattr(getattr(self.main_viewer, "ui_component", None), "mask_color_controls", {}) or {}
                mask_color_control = color_controls.get(mask_name)
                mask_option = _resolve_color_dropdown_option(self.main_viewer, getattr(snapshot, "mask_type_color", ""))
                if mask_color_control is not None and mask_option is not None:
                    mask_color_control.value = mask_option

            for cls in self.current_classes:
                cls_key = str(cls)
                color_picker = self.class_color_controls.get(cls_key)
                if color_picker is not None:
                    color_picker.value = snapshot.class_colors.get(cls_key, self.default_color) or self.default_color
                visible_control = self.class_visible_controls.get(cls_key)
                if visible_control is not None:
                    visible_control.value = bool(snapshot.class_visible.get(cls_key, True))
                mode_control = self.class_mode_controls.get(cls_key)
                if mode_control is not None:
                    mode_control.value = bool(snapshot.class_fill.get(cls_key, False))
                opacity_control = self.class_opacity_controls.get(cls_key)
                if opacity_control is not None:
                    opacity_control.value = _normalise_opacity_percent(
                        snapshot.class_opacity.get(cls_key, FILL_OPACITY_DEFAULT_PERCENT)
                    )

            self._active_classes = [cls for cls in self.current_classes if str(cls) in active_set]
        finally:
            self._syncing = False

        self._push_to_widget()
        self._last_applied_class_opacities = {
            cls: float(getattr(widget, "value", FILL_OPACITY_DEFAULT_PERCENT))
            for cls, widget in self.class_opacity_controls.items()
        }
        self._cell_opacity_cache.clear()
        self.apply_colors_to_masks(None)
        return True

    def _apply_continuous_snapshot(self, snapshot: MaskPainterSnapshot) -> bool:
        """Restore continuous-mode widgets from a snapshot and re-apply colors."""
        column = str(getattr(snapshot, "continuous_column", "") or "")
        if not column:
            return False
        options = tuple(getattr(self.ui_component.continuous_column_dropdown, "options", ()) or ())
        if column not in options:
            return False

        self._syncing = True
        try:
            self.ui_component.color_mode_toggle.value = "continuous"
            self.ui_component.continuous_column_dropdown.value = column
            colormap = str(getattr(snapshot, "colormap", "viridis") or "viridis")
            if colormap in tuple(self.ui_component.colormap_dropdown.options or ()):
                self.ui_component.colormap_dropdown.value = colormap
            self.ui_component.arcsinh_checkbox.value = bool(getattr(snapshot, "arcsinh", False))
            self.ui_component.arcsinh_cofactor_input.value = float(getattr(snapshot, "arcsinh_cofactor", 5.0))
            # Snapshot carries a concrete resolved range → pin it via manual mode.
            self.ui_component.auto_range_checkbox.value = False
            self.ui_component.vmin_input.disabled = False
            self.ui_component.vmax_input.disabled = False
            self._set_range_fields(float(getattr(snapshot, "vmin", 0.0)), float(getattr(snapshot, "vmax", 1.0)))
            self.ui_component.continuous_opacity_input.value = _normalise_opacity_percent(
                getattr(snapshot, "continuous_opacity", 100)
            )
            self.ui_component.continuous_fill_checkbox.value = bool(getattr(snapshot, "continuous_fill", True))
        finally:
            self._syncing = False

        self.ui_component.categorical_layout.layout.display = "none"
        self.ui_component.continuous_layout.layout.display = "block"
        self._invalidate_state_maps_cache()
        self.apply_colors_to_masks(None)
        return True

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_current_color_set(self, _):
        try:
            folder = self.registry_folder
            name = self.ui_component.set_name_input.value.strip()
            if not name:
                raise ColorSetError("Please provide a name for the color set.")
            if self.ui_component.color_mode_toggle.value == "continuous":
                if not self.ui_component.continuous_column_dropdown.value:
                    raise ColorSetError("Select a continuous value column before saving colors.")
            elif not self.class_color_controls:
                raise ColorSetError("Select an identifier before saving colors.")

            timestamp = datetime.utcnow().isoformat() + "Z"
            payload = self._build_color_set_payload(name, timestamp)

            target_path = self._resolve_color_set_path(folder, name)
            write_color_set_file(target_path, payload)
            self._update_registry(folder, name, target_path, self.current_identifier, timestamp)
            self._log(f"Saved color set '{name}' to {target_path}", clear=True)
            self._set_active_color_set_name(name, force=True)
        except Exception as err:  # pylint: disable=broad-except
            self._log(f"Failed to save color set: {err}", error=True, clear=True)

    def load_color_set_from_ui(self, _):
        try:
            path = self._get_load_path()
            if path is None:
                raise ColorSetError("Select a color set file to load.")
            self._load_color_set(path)
        except Exception as err:  # pylint: disable=broad-except
            self._log(f"Failed to load color set: {err}", error=True, clear=True)

    def apply_saved_color_set(self, _):
        try:
            record = self._get_selected_registry_record()
            if record is None:
                raise ColorSetError("Select a saved color set from the Manage tab.")
            path = Path(record["path"])
            identifier = (record.get("identifier") or "").strip()
            if identifier:
                self._ensure_identifier_available(identifier)
            self._load_color_set(path)
            self._set_active_color_set_name(self.ui_component.saved_sets_dropdown.value)
        except Exception as err:  # pylint: disable=broad-except
            self._log(f"Failed to apply saved color set: {err}", error=True, clear=True)

    def overwrite_saved_color_set(self, _):
        try:
            record = self._get_selected_registry_record()
            if record is None:
                raise ColorSetError("Select a saved color set to overwrite.")
            name = self.ui_component.saved_sets_dropdown.value
            folder = Path(record.get("folder", self.registry_folder))
            path = Path(record["path"])

            if self.ui_component.color_mode_toggle.value == "continuous":
                if not self.ui_component.continuous_column_dropdown.value:
                    raise ColorSetError("Select a continuous value column before overwriting a color set.")
            elif not self.class_color_controls:
                raise ColorSetError("Select an identifier before overwriting a color set.")

            timestamp = datetime.utcnow().isoformat() + "Z"
            payload = self._build_color_set_payload(name, timestamp)

            write_color_set_file(path, payload)
            self._update_registry(folder, name, path, self.current_identifier, timestamp)
            self._log(f"Overwrote color set '{name}'.", clear=True)
            self._set_active_color_set_name(name, force=True)
        except Exception as err:  # pylint: disable=broad-except
            self._log(f"Failed to overwrite color set: {err}", error=True, clear=True)

    def delete_saved_color_set(self, _):
        try:
            record = self._get_selected_registry_record()
            if record is None:
                raise ColorSetError("Select a saved color set to delete.")
            name = self.ui_component.saved_sets_dropdown.value
            path = Path(record["path"])
            if path.exists():
                path.unlink()
            folder = Path(record.get("folder", self.registry_folder))
            records = load_registry(folder)
            records.pop(name, None)
            save_registry(folder, records)
            if folder.resolve() == self.registry_folder.resolve():
                self.registry_records = records
                self._refresh_saved_sets_dropdown()
            if self.active_color_set_name == name:
                self._set_active_color_set_name(None)
            self._log(f"Deleted color set '{name}'.", clear=True)
        except Exception as err:  # pylint: disable=broad-except
            self._log(f"Failed to delete color set: {err}", error=True, clear=True)

    def _resolve_color_set_path(self, folder: Path, name: str) -> Path:
        return resolve_palette_path(
            folder,
            name,
            COLOR_SET_FILE_SUFFIX,
            default_slug="mask-colors",
        )

    def _get_full_class_order(self) -> List[str]:
        if self.current_classes:
            return list(self.current_classes)
        return list(self.class_color_controls.keys())

    def _update_registry(self, folder: Path, name: str, path: Path, identifier: Optional[str], timestamp: str) -> None:
        folder = folder.resolve()
        records = load_registry(folder)
        records[name] = {
            "path": str(path.resolve()),
            "identifier": identifier or "",
            "last_modified": timestamp,
            "folder": str(folder.resolve()),
        }
        save_registry(folder, records)
        if folder.resolve() == self.registry_folder.resolve():
            self.registry_records = records
            self._refresh_saved_sets_dropdown()

    def _set_active_color_set_name(self, name: Optional[str], *, force: bool = False) -> None:
        candidate = (str(name).strip() if name else "")
        if not candidate:
            self.active_color_set_name = ""
            return
        if force or candidate in self.registry_records:
            self.active_color_set_name = candidate
        else:
            self.active_color_set_name = ""

    def get_active_color_set_name(self) -> str:
        return self.active_color_set_name or ""

    def resolve_saved_color_map(self, name: str) -> Optional[Tuple[Dict[str, str], str]]:
        candidate = (name or "").strip()
        if not candidate:
            return None
        record = self.registry_records.get(candidate)
        if record is None:
            return None
        path = Path(record.get("path", "")).expanduser()
        if not path.exists():
            return None
        try:
            payload = read_color_set_file(path)
        except Exception:  # pragma: no cover - IO errors
            return None

        colors_payload = payload.get("colors", {})
        if not isinstance(colors_payload, dict):
            return None

        default_color = normalize_hex_color(payload.get("default_color", self.default_color)) or self.default_color
        color_map: Dict[str, str] = {}
        for key, value in colors_payload.items():
            color_map[str(key)] = normalize_hex_color(value) or default_color
        return color_map, default_color

    def apply_color_set_by_name(self, name: str) -> bool:
        candidate = (name or "").strip()
        if not candidate:
            return False
        record = self.registry_records.get(candidate)
        if record is None:
            return False

        path = Path(record.get("path", ""))
        try:
            path = path.expanduser()
        except AttributeError:  # pragma: no cover - defensive
            path = Path(record.get("path", ""))
        if not path.exists():
            return False

        try:
            self._load_color_set(path)
        except Exception:  # pragma: no cover - downstream errors
            return False

        self._set_active_color_set_name(candidate)
        return True

    def _load_color_set(self, path: Path) -> None:
        if not path.exists():
            raise ColorSetError(f"Color set file '{path}' was not found.")
        payload = read_color_set_file(path)

        if str(payload.get("color_mode") or "categorical") == "continuous":
            self._apply_continuous_payload(payload, path)
            return

        color_map = payload.get("colors", {})
        default_color = payload.get("default_color", self.default_color)
        saved_identifier = (payload.get("identifier") or "").strip()
        if saved_identifier:
            self._ensure_identifier_available(saved_identifier)
        self._set_default_color(default_color, update_ui=True)

        apply_color_map_to_controls(color_map, self.class_color_controls, self.default_color)

        incoming_classes = [str(cls) for cls in payload.get("class_order", [])]
        if not incoming_classes:
            incoming_classes = [str(cls) for cls in color_map.keys()]
        if not incoming_classes:
            incoming_classes = list(self.class_color_controls.keys())

        # Build a deduplicated order that respects the saved order first,
        # then appends any extra classes from the current controls.
        ordered_unique: List[str] = []
        seen: set = set()
        for cls in incoming_classes:
            if cls not in seen:
                ordered_unique.append(cls)
                seen.add(cls)
        for cls in self.class_color_controls.keys():
            if cls not in seen:
                ordered_unique.append(cls)
                seen.add(cls)
        self._syncing = True
        try:
            self.current_classes = ordered_unique
            active_classes_payload = [str(cls) for cls in payload.get("active_classes", [])]
            self._active_classes = [cls for cls in active_classes_payload if cls in self.class_color_controls]
            if not self._active_classes:
                self._active_classes = [cls for cls in ordered_unique if cls in self.class_color_controls]

            # Restore per-class modes
            modes_payload = payload.get("modes", {})
            if isinstance(modes_payload, dict):
                for cls, mode in modes_payload.items():
                    cb = self.class_mode_controls.get(str(cls))
                    if cb is not None and mode in ("outline", "fill"):
                        cb.value = (mode == "fill")

            # Restore per-class visibility
            visible_payload = payload.get("visible", {})
            if isinstance(visible_payload, dict):
                for cls, vis in visible_payload.items():
                    cb = self.class_visible_controls.get(str(cls))
                    if cb is not None:
                        cb.value = bool(vis)

            self.ui_component.global_fill_checkbox.value = bool(payload.get("global_fill", False))
            self.ui_component.global_fill_opacity_input.value = _normalise_opacity_percent(
                payload.get("global_fill_opacity", self.ui_component.global_fill_opacity_input.value)
            )
            self.ui_component.show_fill_borders_checkbox.value = bool(payload.get("show_fill_borders", False))
            self.ui_component.border_color_mode_dropdown.value = str(
                payload.get("border_color_mode", BORDER_COLOR_MODE_MASK_TYPE) or BORDER_COLOR_MODE_MASK_TYPE
            )

            opacity_payload = payload.get("opacities", {})
            if isinstance(opacity_payload, dict):
                for cls, opacity in opacity_payload.items():
                    widget = self.class_opacity_controls.get(str(cls))
                    if widget is not None:
                        widget.value = _normalise_opacity_percent(opacity)

            self._set_linked_classes_from_state(
                linked_fill_classes=payload.get("linked_fill_classes"),
                linked_opacity_classes=payload.get("linked_opacity_classes"),
            )

            only_specified = bool(payload.get("only_specified", False))
            self.ui_component.only_specified_checkbox.value = only_specified
            if only_specified and not active_classes_payload:
                non_default, _ = split_default_classes(
                    self.current_classes,
                    {cls: getattr(widget, "value", self.default_color) for cls, widget in self.class_color_controls.items()},
                    self.default_color,
                )
                self._active_classes = [cls for cls in non_default if cls in self.class_color_controls]
        finally:
            self._syncing = False

        self._push_to_widget()
        self._log(f"Loaded color set '{payload.get('name', path.stem)}' from {path}.", clear=True)
        if self.ui_component.enabled_checkbox.value:
            self.apply_colors_to_masks(None)

    def _apply_continuous_payload(self, payload: Mapping[str, object], path: Path) -> None:
        """Restore a saved continuous-coloring palette."""
        cont = dict(payload.get("continuous") or {})
        column = str(cont.get("column") or "")
        self._syncing = True
        try:
            self.ui_component.color_mode_toggle.value = "continuous"
            options = tuple(self.ui_component.continuous_column_dropdown.options or ())
            if column and column in options:
                self.ui_component.continuous_column_dropdown.value = column
            elif column:
                self._log(
                    f"Continuous column '{column}' from palette is not in the current dataset.",
                    error=True,
                )
            colormap = str(cont.get("colormap") or "viridis")
            if colormap in tuple(self.ui_component.colormap_dropdown.options or ()):
                self.ui_component.colormap_dropdown.value = colormap
            self.ui_component.arcsinh_checkbox.value = bool(cont.get("arcsinh", False))
            self.ui_component.arcsinh_cofactor_input.value = float(cont.get("cofactor", 5.0) or 5.0)
            auto_range = bool(cont.get("auto_range", True))
            self.ui_component.auto_range_checkbox.value = auto_range
            self.ui_component.vmin_input.disabled = auto_range
            self.ui_component.vmax_input.disabled = auto_range
            if not auto_range:
                self._set_range_fields(float(cont.get("vmin", 0.0) or 0.0), float(cont.get("vmax", 1.0)))
            self.ui_component.continuous_opacity_input.value = _normalise_opacity_percent(cont.get("opacity", 100))
            self.ui_component.continuous_fill_checkbox.value = bool(cont.get("fill", True))
        finally:
            self._syncing = False

        self.ui_component.categorical_layout.layout.display = "none"
        self.ui_component.continuous_layout.layout.display = "block"
        self._invalidate_state_maps_cache()
        self._render_colorbar()
        self._log(f"Loaded continuous color set '{payload.get('name', path.stem)}' from {path}.", clear=True)
        if self.ui_component.enabled_checkbox.value:
            self.apply_colors_to_masks(None)

    def _ensure_identifier_available(self, identifier: str) -> None:
        if not identifier:
            return
        dropdown = self.ui_component.identifier_dropdown
        options = list(dropdown.options)
        if identifier not in options:
            self._log(
                f"Identifier '{identifier}' from palette is not available in the current dataset.",
                error=True,
            )
            return
        if dropdown.value != identifier:
            dropdown.value = identifier
        self.current_identifier = identifier

    def _get_load_path(self) -> Optional[Path]:
        if FileChooser and self.ui_component.load_picker is not None:
            selected = getattr(self.ui_component.load_picker, "selected", None)
            if selected:
                return Path(selected).expanduser()
        if self.ui_component.load_path_input is not None:
            value = self.ui_component.load_path_input.value.strip()
            if value:
                return Path(value).expanduser()
        return None

    def _get_selected_registry_record(self) -> Optional[Dict[str, str]]:
        name = self.ui_component.saved_sets_dropdown.value
        if not name:
            return None
        return self.registry_records.get(name)

    # ------------------------------------------------------------------
    # Viewer integration
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Continuous coloring (issue #115)
    # ------------------------------------------------------------------
    def _active_continuous_spec(self) -> Optional[Dict[str, object]]:
        """Return the continuous spec when the painter is in continuous mode, else None."""
        if self.ui_component.color_mode_toggle.value != "continuous":
            return None
        return self._build_continuous_spec()

    def _build_continuous_spec(self) -> Optional[Dict[str, object]]:
        """Read the continuous widgets into a spec dict with a resolved global range.

        This is on the render hot path (called for every
        ``build_painter_state_maps_for_fov``), so it must be cheap: the auto range
        is read from ``_continuous_range_cache`` (computed once per
        column/arcsinh/cofactor by the param-change handlers) and no ipywidget
        values are written here.
        """
        column = self.ui_component.continuous_column_dropdown.value
        if not column or self.main_viewer.cell_table is None:
            return None
        if column not in self.main_viewer.cell_table.columns:
            return None
        arcsinh = bool(self.ui_component.arcsinh_checkbox.value)
        cofactor = _safe_float(self.ui_component.arcsinh_cofactor_input.value, 5.0)
        vmin, vmax = self._current_range(column, arcsinh, cofactor)
        return {
            "column": str(column),
            "colormap": str(self.ui_component.colormap_dropdown.value or "viridis"),
            "vmin": float(vmin),
            "vmax": float(vmax),
            "arcsinh": arcsinh,
            "cofactor": cofactor,
            "opacity": _normalise_opacity_percent(self.ui_component.continuous_opacity_input.value),
            "fill": bool(self.ui_component.continuous_fill_checkbox.value),
        }

    def _compute_auto_range(self, column, arcsinh: bool, cofactor: float) -> Tuple[float, float]:
        values = self.main_viewer.cell_table[column].to_numpy()
        return resolve_continuous_range(values, arcsinh=arcsinh, cofactor=cofactor)

    def _current_range(self, column, arcsinh: bool, cofactor: float) -> Tuple[float, float]:
        """Return the active (vmin, vmax) without touching widgets.

        Manual mode reads the vmin/vmax fields directly. Auto mode returns the
        cached global percentile range, computing it only on a cache miss (key =
        column + arcsinh + cofactor); the cache is refreshed by the param-change
        handlers and cleared on cell-table reload.
        """
        if not self.ui_component.auto_range_checkbox.value:
            return (
                _safe_float(self.ui_component.vmin_input.value, 0.0),
                _safe_float(self.ui_component.vmax_input.value, 1.0),
            )
        key = (str(column), bool(arcsinh), float(cofactor))
        cached = self._continuous_range_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        rng = self._compute_auto_range(column, arcsinh, cofactor)
        self._continuous_range_cache = (key, rng)
        return rng

    def _refresh_auto_range_fields(self) -> None:
        """Recompute the global auto-range and reflect it in the vmin/vmax fields.

        Called only from user-facing handlers (column/arcsinh/cofactor/auto changes,
        recompute button) — never on the render hot path.
        """
        column = self.ui_component.continuous_column_dropdown.value
        if (
            not column
            or self.main_viewer.cell_table is None
            or column not in self.main_viewer.cell_table.columns
            or not self.ui_component.auto_range_checkbox.value
        ):
            return
        arcsinh = bool(self.ui_component.arcsinh_checkbox.value)
        cofactor = _safe_float(self.ui_component.arcsinh_cofactor_input.value, 5.0)
        rng = self._compute_auto_range(column, arcsinh, cofactor)
        self._continuous_range_cache = ((str(column), arcsinh, float(cofactor)), rng)
        self._set_range_fields(*rng)

    def _set_range_fields(self, vmin: float, vmax: float) -> None:
        for widget, value in ((self.ui_component.vmin_input, vmin), (self.ui_component.vmax_input, vmax)):
            widget.unobserve(self._on_continuous_range_edit, names="value")
            try:
                widget.value = float(value)
            finally:
                widget.observe(self._on_continuous_range_edit, names="value")

    def _refresh_continuous_display(self) -> None:
        """Recompute continuous colors for the current FOV and refresh the viewer."""
        self._invalidate_state_maps_cache()
        self._render_colorbar()
        if not self.ui_component.enabled_checkbox.value:
            return
        map_mode = self.main_viewer.get_active_fov() is None
        self.apply_colors_to_masks(None, notify_cell_gallery=False, register_globally=map_mode)

    def _on_color_mode_change(self, change):
        continuous = change.get("new") == "continuous"
        self.ui_component.categorical_layout.layout.display = "none" if continuous else "block"
        self.ui_component.continuous_layout.layout.display = "block" if continuous else "none"
        self._refresh_save_button_state()
        self._invalidate_state_maps_cache()
        if continuous:
            self._refresh_auto_range_fields()
            self._render_colorbar()
        if self.ui_component.enabled_checkbox.value:
            update_display = getattr(self.main_viewer, "update_display", None)
            current_ds = getattr(self.main_viewer, "current_downsample_factor", 1)
            if callable(update_display):
                update_display(current_ds)

    def _on_continuous_column_change(self, _change):
        self._refresh_save_button_state()
        # New column → recompute the global auto-range (no-op in manual mode).
        self._refresh_auto_range_fields()
        self._refresh_continuous_display()

    def _on_continuous_param_change(self, _change):
        if self._syncing:
            return
        # arcsinh / cofactor changes affect the transformed range when auto is on.
        self._refresh_auto_range_fields()
        self._refresh_continuous_display()

    def _on_auto_range_toggle(self, change):
        auto = bool(change.get("new", True))
        self.ui_component.vmin_input.disabled = auto
        self.ui_component.vmax_input.disabled = auto
        if auto:
            self._refresh_auto_range_fields()
        self._refresh_continuous_display()

    def _on_continuous_range_edit(self, _change):
        if self._syncing or self.ui_component.auto_range_checkbox.value:
            return
        self._refresh_continuous_display()

    def _recompute_continuous_range(self, _):
        column = self.ui_component.continuous_column_dropdown.value
        if not column or self.main_viewer.cell_table is None or column not in self.main_viewer.cell_table.columns:
            self._log("Select a continuous value column first.", error=True, clear=True)
            return
        self._refresh_auto_range_fields()
        self._refresh_continuous_display()

    def _render_colorbar(self) -> None:
        """Draw a horizontal colorbar legend as a static PNG in the output widget.

        Rendered with a detached Agg ``Figure`` (not ``pyplot``) so it never
        creates an interactive/``ipympl`` canvas widget — doing so crashed under
        the notebook's ``%matplotlib widget`` backend (issue #115 reply) and was
        needlessly slow. The result is a plain PNG shown via ``IPython.display``.
        """
        column = self.ui_component.continuous_column_dropdown.value
        if not column:
            return
        try:
            import io
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib import cm
            from matplotlib.colors import Normalize
            from IPython.display import Image as _IPyImage, display as _display
        except Exception:  # pragma: no cover - matplotlib/IPython always present here
            return
        arcsinh = bool(self.ui_component.arcsinh_checkbox.value)
        vmin = _safe_float(self.ui_component.vmin_input.value, 0.0)
        vmax = _safe_float(self.ui_component.vmax_input.value, 1.0)
        if vmax <= vmin:
            vmax = vmin + 1.0
        try:
            fig = Figure(figsize=(4, 0.5))
            FigureCanvasAgg(fig)
            ax = fig.add_axes([0.04, 0.5, 0.92, 0.3])
            mappable = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=self.ui_component.colormap_dropdown.value or "viridis")
            fig.colorbar(mappable, cax=ax, orientation="horizontal")
            label = f"arcsinh({column} / {_safe_float(self.ui_component.arcsinh_cofactor_input.value, 5.0):g})" if arcsinh else str(column)
            ax.set_title(label, fontsize=7)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        except Exception:  # pragma: no cover - defensive; colorbar is non-critical
            return
        png = buf.getvalue()
        with self.ui_component.colorbar_output:
            self.ui_component.colorbar_output.clear_output(wait=True)
            _display(_IPyImage(data=png))

    def _apply_continuous_colors_to_masks(self, *, notify_cell_gallery: bool = True, register_globally: bool = True):
        """Apply continuous (gradient) colors to masks.

        Continuous colors are computed **on demand per FOV** by
        ``build_painter_state_maps_for_fov`` for every consumer (the live
        single-FOV display and map overlay via ``get_effective_state_maps_for_fov``;
        the cell gallery, batch export, and ROI thumbnails via
        ``resolve_mask_painter_snapshot_for_fov``). So there is no need to
        pre-register a color for every cell in every FOV — doing so was the cause
        of the multi-minute stall on Apply (issue #115 reply). We only clear any
        stale registry colors, invalidate the per-FOV state-maps cache, and refresh
        the current FOV. The ``register_globally`` argument is retained for
        signature parity with the categorical path but is intentionally unused here.
        """
        if self.main_viewer.cell_table is None:
            self._log("Cell table is not available.", error=True, clear=True)
            return
        spec = self._build_continuous_spec()
        if spec is None:
            self._log("Select a continuous value column to color.", error=True, clear=True)
            return

        cell_table = self.main_viewer.cell_table
        column = spec["column"]
        if not cell_table[column].notna().any():
            self._log(f"Column '{column}' has no finite values to color.", error=True, clear=True)
            return

        # Clear any per-cell colors left over from a previous categorical apply so
        # they cannot bleed through the cell gallery's per-cell registry fallback
        # for cells the continuous map omits (e.g. NaN-valued cells).
        clear_cell_colors()
        self._invalidate_state_maps_cache()

        self._log("Masks updated with continuous colors.")
        self._render_colorbar()

        if notify_cell_gallery:
            self._notify_cell_gallery_update()

        if self.ui_component.enabled_checkbox.value:
            update_display = getattr(self.main_viewer, "update_display", None)
            current_ds = getattr(self.main_viewer, "current_downsample_factor", 1)
            if callable(update_display):
                update_display(current_ds)

    @update_status_bar
    def apply_colors_to_masks(self, _, *, notify_cell_gallery: bool = True, register_globally: bool = True):
        """Apply colors to masks.

        Parameters
        ----------
        notify_cell_gallery : bool
            Whether to notify the cell gallery of color changes
        register_globally : bool
            Whether to register colors globally for all FOVs (slow for large datasets).
            Set to False when called automatically on display updates.
        """
        self._invalidate_state_maps_cache()

        if self.ui_component.color_mode_toggle.value == "continuous":
            self._apply_continuous_colors_to_masks(
                notify_cell_gallery=notify_cell_gallery,
                register_globally=register_globally,
            )
            return

        identifier = self.ui_component.identifier_dropdown.value
        if not identifier:
            self._log("No identifier selected.", error=True, clear=True)
            return

        if self.main_viewer.cell_table is None:
            self._log("Cell table is not available.", error=True, clear=True)
            return

        visible_classes = self._get_visible_classes()
        if not visible_classes and not self.ui_component.show_all_checkbox.value:
            self._log("Select at least one class to color.", error=True, clear=True)
            return

        current_fov = self.main_viewer.get_active_fov()

        column = self.main_viewer.cell_table[identifier]
        dtype = column.dtype
        def _convert(value: str):
            if np.issubdtype(dtype, np.integer):
                return int(value)
            if np.issubdtype(dtype, np.floating):
                return float(value)
            return value

        converted_visible = [(cls, _convert(cls)) for cls in visible_classes]
        hidden_classes = self._get_hidden_classes()
        converted_hidden = [(cls, _convert(cls)) for cls in hidden_classes]
        inactive_classes = self._get_inactive_classes()
        converted_inactive = [(cls, _convert(cls)) for cls in inactive_classes]

        def _apply_color_to_current_fov(cls_value, color):
            """Apply color to masks in the currently loaded FOV (viewer display only)."""
            mask_ids = self.main_viewer.cell_table.loc[
                (self.main_viewer.cell_table[self.main_viewer.fov_key] == current_fov)
                & (self.main_viewer.cell_table[identifier] == cls_value),
                self.main_viewer.label_key,
            ].tolist()

            if not mask_ids:
                return

            self.main_viewer.image_display.set_mask_colors_current_fov(
                mask_name=self.main_viewer.mask_key,
                mask_ids=mask_ids,
                color=color,
                cummulative=True,
            )

        def _register_color_globally(cls_str, cls_value, color):
            """Register color for all cells of this class across all FOVs (for gallery).

            Uses a vectorized bulk-write instead of iterrows() to avoid O(N)
            Python-level per-cell overhead.
            """
            if self._last_applied_class_colors.get(cls_str) == color:
                # Color unchanged since last registration — registry is already
                # correct for this class; skip the expensive bulk write.
                return
            matching_cells = self.main_viewer.cell_table.loc[
                self.main_viewer.cell_table[identifier] == cls_value,
                [self.main_viewer.fov_key, self.main_viewer.label_key],
            ]
            fov_arr = matching_cells[self.main_viewer.fov_key].to_numpy()
            mid_arr = matching_cells[self.main_viewer.label_key].to_numpy()
            entries: dict[str, dict[int, str]] = {}
            for fov, mid in zip(fov_arr, mid_arr):
                entries.setdefault(fov, {})[int(mid)] = color
            set_cell_colors_bulk(entries)
            self._last_applied_class_colors[cls_str] = color

        def _register_mode_globally(cls_str, cls_value, mode):
            """Register render mode for all cells of this class across all FOVs."""
            if self._last_applied_class_modes.get(cls_str) == mode:
                return
            matching_cells = self.main_viewer.cell_table.loc[
                self.main_viewer.cell_table[identifier] == cls_value,
                [self.main_viewer.fov_key, self.main_viewer.label_key],
            ]
            fov_arr = matching_cells[self.main_viewer.fov_key].to_numpy()
            mid_arr = matching_cells[self.main_viewer.label_key].to_numpy()
            for fov, mid in zip(fov_arr, mid_arr):
                self._cell_mode_cache.setdefault(str(fov), {})[int(mid)] = mode
            self._last_applied_class_modes[cls_str] = mode

        def _register_opacity_globally(cls_str, cls_value, alpha):
            """Register fill alpha for all cells of this class across all FOVs."""
            alpha = max(0.0, min(1.0, float(alpha)))
            if self._last_applied_class_opacities.get(cls_str) == alpha:
                return
            matching_cells = self.main_viewer.cell_table.loc[
                self.main_viewer.cell_table[identifier] == cls_value,
                [self.main_viewer.fov_key, self.main_viewer.label_key],
            ]
            fov_arr = matching_cells[self.main_viewer.fov_key].to_numpy()
            mid_arr = matching_cells[self.main_viewer.label_key].to_numpy()
            for fov, mid in zip(fov_arr, mid_arr):
                self._cell_opacity_cache.setdefault(str(fov), {})[int(mid)] = alpha
            self._last_applied_class_opacities[cls_str] = alpha

        # Apply colors to visible classes
        for cls_str, cls_value in reversed(converted_visible):
            picker = self.class_color_controls.get(cls_str)
            if picker is None:
                self._log(f"Class '{cls_str}' not found in controls; skipping.", error=True)
                continue

            color = picker.value
            mode_cb = self.class_mode_controls.get(cls_str)
            mode = ("fill" if mode_cb.value else "outline") if mode_cb is not None else "outline"
            opacity_control = self.class_opacity_controls.get(cls_str)
            opacity_alpha = _opacity_percent_to_alpha(
                getattr(opacity_control, "value", self.ui_component.global_fill_opacity_input.value),
                _normalise_opacity_percent(self.ui_component.global_fill_opacity_input.value),
            )
            if current_fov:
                _apply_color_to_current_fov(cls_value, color)
            if register_globally:
                _register_color_globally(cls_str, cls_value, color)
                _register_mode_globally(cls_str, cls_value, mode)
                _register_opacity_globally(cls_str, cls_value, opacity_alpha if mode == "fill" else 0.0)

        # Apply empty-string sentinel to hidden classes so they are invisible
        if hidden_classes:
            for cls_str, cls_value in converted_hidden:
                if current_fov:
                    _apply_color_to_current_fov(cls_value, "")
                if register_globally:
                    _register_color_globally(cls_str, cls_value, "")
                    _register_mode_globally(cls_str, cls_value, "outline")
                    _register_opacity_globally(cls_str, cls_value, 0.0)

        # Inactive classes stay visible with the default color.
        if inactive_classes:
            inactive_mode = "fill" if bool(self.ui_component.global_fill_checkbox.value) else "outline"
            inactive_alpha = (
                _opacity_percent_to_alpha(
                    self.ui_component.global_fill_opacity_input.value,
                    _normalise_opacity_percent(self.ui_component.global_fill_opacity_input.value),
                )
                if inactive_mode == "fill"
                else 0.0
            )
            for cls_str, cls_value in converted_inactive:
                if register_globally:
                    _register_color_globally(cls_str, cls_value, self.default_color)
                    _register_mode_globally(cls_str, cls_value, inactive_mode)
                    _register_opacity_globally(cls_str, cls_value, inactive_alpha)

        self._log("Masks updated with class-based colors.")
        
        # Track the applied state
        self._last_applied_fov = current_fov
        self._last_applied_identifier = identifier
        self._last_applied_classes = set(visible_classes)

        if notify_cell_gallery:
            self._notify_cell_gallery_update()

        if self.ui_component.enabled_checkbox.value:
            update_display = getattr(self.main_viewer, "update_display", None)
            current_ds = getattr(self.main_viewer, "current_downsample_factor", 1)
            if callable(update_display):
                update_display(current_ds)

    def _on_visibility_or_mode_change(self, _change):
        """Called when any per-class visibility, mode, or opacity control changes.

        Forces ``apply_colors_to_masks`` to re-run so hidden classes immediately
        receive the ``""`` sentinel and visible classes get their current color/mode.
        Resets the dirty-skip caches so the change is not skipped.
        """
        if self._syncing or not self.ui_component.enabled_checkbox.value:
            return
        # Invalidate caches so apply_colors_to_masks doesn't skip the re-write
        self._last_applied_class_colors.clear()
        self._last_applied_class_modes.clear()
        self._last_applied_class_opacities.clear()
        map_mode = self.main_viewer.get_active_fov() is None
        self.apply_colors_to_masks(
            None,
            notify_cell_gallery=False,
            register_globally=map_mode,
        )

    def _on_class_mode_change(self, change):
        if self._syncing:
            return
        cls_key = self._resolve_tracked_class_key(change, self._mode_control_class_keys)
        if cls_key is None:
            return
        new_value = bool(change.get("new", getattr(change.get("owner"), "value", False)))
        if new_value == bool(self.ui_component.global_fill_checkbox.value):
            self._linked_fill_classes.add(cls_key)
        else:
            self._linked_fill_classes.discard(cls_key)

    def _on_class_opacity_change(self, change):
        if self._syncing:
            return
        cls_key = self._resolve_tracked_class_key(change, self._opacity_control_class_keys)
        if cls_key is None:
            return
        new_value = _normalise_opacity_percent(
            change.get("new", getattr(change.get("owner"), "value", FILL_OPACITY_DEFAULT_PERCENT)),
            _normalise_opacity_percent(self.ui_component.global_fill_opacity_input.value),
        )
        if new_value == _normalise_opacity_percent(self.ui_component.global_fill_opacity_input.value):
            self._linked_opacity_classes.add(cls_key)
        else:
            self._linked_opacity_classes.discard(cls_key)

    def _on_global_fill_toggle(self, change):
        if self._syncing:
            return
        old = bool(change.get("old", False))
        new = bool(change.get("new", old))
        if old == new:
            return
        self._push_to_widget()
        if self.ui_component.enabled_checkbox.value:
            self._last_applied_class_modes.clear()
            self._last_applied_class_opacities.clear()
            map_mode = self.main_viewer.get_active_fov() is None
            self.apply_colors_to_masks(
                None,
                notify_cell_gallery=False,
                register_globally=map_mode,
            )

    def _on_global_fill_opacity_change(self, change):
        if self._syncing:
            return
        old = _normalise_opacity_percent(change.get("old", FILL_OPACITY_DEFAULT_PERCENT))
        new = _normalise_opacity_percent(change.get("new", old))
        if old == new:
            return
        self._syncing = True
        try:
            for widget in self.class_opacity_controls.values():
                if widget is None:
                    continue
                if _normalise_opacity_percent(getattr(widget, "value", old), old) != old:
                    continue
                widget.value = new
            self._push_to_widget()
        finally:
            self._syncing = False
        if self.ui_component.enabled_checkbox.value:
            self._last_applied_class_opacities.clear()
            map_mode = self.main_viewer.get_active_fov() is None
            self.apply_colors_to_masks(
                None,
                notify_cell_gallery=False,
                register_globally=map_mode,
            )

    def _on_fill_border_toggle(self, _change):
        if not self.ui_component.enabled_checkbox.value:
            return
        update_display = getattr(self.main_viewer, "update_display", None)
        current_ds = getattr(self.main_viewer, "current_downsample_factor", 1)
        if callable(update_display):
            update_display(current_ds)

    def _on_enabled_toggle(self, change):
        """Refresh the viewer immediately when the mask painter is enabled or disabled."""
        self._last_applied_fov = None
        self._last_applied_identifier = None
        self._last_applied_classes = None
        self._last_applied_class_colors.clear()
        self._last_applied_class_modes.clear()
        self._last_applied_class_opacities.clear()

        if change.get("new"):
            map_mode = self.main_viewer.get_active_fov() is None
            self.apply_colors_to_masks(
                None,
                notify_cell_gallery=False,
                register_globally=map_mode,
            )

        update_display = getattr(self.main_viewer, "update_display", None)
        current_ds = getattr(self.main_viewer, "current_downsample_factor", 1)
        if callable(update_display):
            update_display(current_ds)

    def on_cell_table_change(self):
        self._initialise_identifier_options()
        self._initialise_continuous_options()
        self._active_classes = []
        # A new cell table may reuse a column name with different data → drop the
        # cached continuous auto-range so it is recomputed on next use.
        self._continuous_range_cache = None
        # Clear tracking state when cell table changes
        self._last_applied_fov = None
        self._last_applied_identifier = None
        self._last_applied_classes = None
        self._last_applied_class_colors = {}
        self._last_applied_class_modes = {}
        self._last_applied_class_opacities = {}
        self._cell_mode_cache = {}
        self._cell_opacity_cache = {}
        self._state_maps_cache = {}

    def on_mv_update_display(self):
        """Handle main viewer display updates.
        
        Only re-apply colors if:
        1. Mask painter is enabled
        2. AND one of the following has changed:
           - Current FOV
           - Selected identifier
           - Set of visible classes
           
        Note: This only applies colors to the current FOV for display purposes.
        Global registration (for gallery) is skipped to avoid performance issues.
        """
        if not self.ui_component.enabled_checkbox.value:
            return
        
        current_fov = self.main_viewer.get_active_fov()
        current_identifier = self.ui_component.identifier_dropdown.value
        current_classes = set(self._get_visible_classes()) if current_identifier else set()
        
        # Check if state has actually changed
        state_changed = (
            self._last_applied_fov != current_fov
            or self._last_applied_identifier != current_identifier
            or self._last_applied_classes != current_classes
        )
        
        if state_changed:
            # In map mode (current_fov is None), register globally so the
            # _apply_map_painter_overlay method can read from the registry.
            # In single-FOV mode, skip global registration for performance.
            map_mode = current_fov is None
            self.apply_colors_to_masks(
                None,
                notify_cell_gallery=False,
                register_globally=map_mode,
            )

    def _notify_cell_gallery_update(self) -> None:
        """Notify the cell gallery plugin that mask painter has applied color changes."""
        sideplots = getattr(self.main_viewer, "SidePlots", None)
        if sideplots is None:
            return
        
        cell_gallery_plugin = getattr(sideplots, "cell_gallery_output", None)
        if hasattr(cell_gallery_plugin, "on_mask_painter_change"):
            try:
                cell_gallery_plugin.on_mask_painter_change()
            except Exception as exc:
                _logger.debug("[mask_painter] Failed to notify cell gallery: %s", exc)

    def get_cell_color(self, fov: str, mask_id: int) -> Optional[str]:
        """Get the painted color for a specific cell ID in a specific FOV.
        
        This method now delegates to the centralized rendering engine registry.
        
        Returns None if no color has been painted for this cell.
        """
        return get_cell_color(fov, mask_id)

    def get_mode_map_for_fov(self, fov: str) -> Dict[int, str]:
        """Return the per-cell render mode mapping for a given FOV.

        Returns a dict mapping ``mask_id -> "outline" | "fill"`` for cells
        that have been assigned a mode via ``apply_colors_to_masks``.
        Cells absent from the returned dict should default to ``"outline"``.
        """
        return dict(self._cell_mode_cache.get(str(fov), {}))

    def initiate_ui(self):
        row1 = HBox([
            self.ui_component.enabled_checkbox,
            self.ui_component.identifier_dropdown,
        ])
        row2 = HBox([
            self.ui_component.update_button,
            self.ui_component.apply_saved_button,
            self.ui_component.saved_sets_dropdown,
        ])
        self.ui_component.control_panel = VBox([row1, row2])
        self.ui = VBox(
            [
                self.ui_component.control_panel,
                HTML("<hr style='margin:4px 0'>"),
                self.ui_component.colors_layout,
                self.ui_component.palette_accordion,
                self.ui_component.feedback_label,
            ],
            layout=Layout(max_height="600px", overflow_y="auto"),
        )

    # ------------------------------------------------------------------
    # Folder management and logging
    # ------------------------------------------------------------------
    def on_change_folder_clicked(self, _):
        self.ui_component.toggle_manual_folder_box(True)

    def apply_manual_folder_override(self, _):
        raw_value = self.ui_component.manual_folder_input.value.strip()
        if not raw_value:
            self._log("Enter a folder path to use for mask palettes.", error=True, clear=True)
            return

        folder = Path(raw_value).expanduser()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception as err:  # pylint: disable=broad-except
            self._log(f"Could not use folder '{folder}': {err}", error=True, clear=True)
            return

        self.registry_folder = folder.resolve()
        self.ui_component.set_folder_path(self.registry_folder)
        self._load_registry_for_folder(self.registry_folder)
        self.ui_component.toggle_manual_folder_box(False)
        self._log(f"Using custom palette directory {self.registry_folder}", clear=True)

    def cancel_manual_folder_override(self, _):
        self.ui_component.manual_folder_input.value = ""
        self.ui_component.toggle_manual_folder_box(False)

    def _refresh_save_button_state(self, _=None) -> None:
        """Enable Save set button only when a name and a value source are set."""
        name_val = getattr(self.ui_component.set_name_input, "value", None) or ""
        has_name = bool(str(name_val).strip())
        if getattr(self.ui_component.color_mode_toggle, "value", "categorical") == "continuous":
            has_source = bool(getattr(self.ui_component.continuous_column_dropdown, "value", None))
        else:
            has_source = bool(getattr(self.ui_component.identifier_dropdown, "value", None))
        self.ui_component.save_button.disabled = not (has_name and has_source)

    def _push_to_widget(self, _=None) -> None:
        """Push the current Python-side state (colors, visibility, order) to the anywidget."""
        if self._syncing or not self.class_color_controls:
            return
        w = self.ui_component.class_list_widget
        active = list(self._active_classes) if self._active_classes else self._get_full_class_order()
        active_set = set(active)
        available = [cls for cls in self.current_classes if cls not in active_set]
        colors = {cls: (p.value or self.default_color) for cls, p in self.class_color_controls.items() if cls in active_set}
        visible = {cls: cb.value for cls, cb in self.class_visible_controls.items() if cls in active_set}
        fill = {cls: cb.value for cls, cb in self.class_mode_controls.items() if cls in active_set}
        opacity = {
            cls: _normalise_opacity_percent(getattr(widget, "value", FILL_OPACITY_DEFAULT_PERCENT))
            for cls, widget in self.class_opacity_controls.items()
            if cls in active_set
        }
        self._syncing = True
        try:
            w.class_order = active
            w.class_colors = colors
            w.class_visible = visible
            w.class_fill = fill
            w.class_opacity = opacity
            w.default_color = self.default_color
            w.available_classes = available
        finally:
            self._syncing = False

    def _pull_from_widget(self, change) -> None:
        """Pull anywidget traitlet changes back into the Python-side ipywidgets."""
        if self._syncing:
            return
        w = self.ui_component.class_list_widget
        name = change["name"]
        self._syncing = True
        try:
            if name == "class_order":
                self._active_classes = list(w.class_order)
            elif name == "class_colors":
                for cls, color in w.class_colors.items():
                    picker = self.class_color_controls.get(cls)
                    if picker is not None and picker.value != color:
                        picker.value = color
            elif name == "class_visible":
                for cls, vis in w.class_visible.items():
                    cb = self.class_visible_controls.get(cls)
                    if cb is not None and cb.value != bool(vis):
                        cb.value = bool(vis)
            elif name == "class_fill":
                for cls, fill_val in w.class_fill.items():
                    cb = self.class_mode_controls.get(cls)
                    if cb is not None and cb.value != bool(fill_val):
                        cb.value = bool(fill_val)
            elif name == "class_opacity":
                for cls, opacity in w.class_opacity.items():
                    widget = self.class_opacity_controls.get(cls)
                    next_value = _normalise_opacity_percent(opacity)
                    if widget is not None and widget.value != next_value:
                        widget.value = next_value
        finally:
            self._syncing = False
        # Visibility or fill changes need an immediate re-apply when painter is active
        if name in ("class_visible", "class_fill", "class_opacity") and self.ui_component.enabled_checkbox.value:
            self._last_applied_class_colors.clear()
            self._last_applied_class_modes.clear()
            self._last_applied_class_opacities.clear()
            map_mode = self.main_viewer.get_active_fov() is None
            self.apply_colors_to_masks(None, notify_cell_gallery=False, register_globally=map_mode)

    def _on_add_requested(self, change) -> None:
        """Called when the JS Add button signals that a class should be made active."""
        cls = change["new"]
        if not cls:
            return
        w = self.ui_component.class_list_widget
        if cls not in self.current_classes or cls in self._active_classes:
            # Invalid or already active — just clear the signal
            self._syncing = True
            try:
                w.add_requested = ""
            finally:
                self._syncing = False
            return
        self._active_classes.append(cls)
        self._push_to_widget()
        self._syncing = True
        try:
            w.add_requested = ""
        finally:
            self._syncing = False

    def _on_remove_requested(self, change) -> None:
        """Called when the JS × button signals that a class should be removed from active."""
        cls = change["new"]
        if not cls:
            return
        w = self.ui_component.class_list_widget
        if cls not in self._active_classes:
            # Not active — just clear the signal
            self._syncing = True
            try:
                w.remove_requested = ""
            finally:
                self._syncing = False
            return
        self._active_classes.remove(cls)
        self._push_to_widget()
        self._syncing = True
        try:
            w.remove_requested = ""
        finally:
            self._syncing = False

    def _on_only_specified_toggle(self, change) -> None:
        """Recompute active classes based on the 'Only specified' checkbox.

        When ON  → keep only classes whose color differs from the default color.
        When OFF → restore all classes in ``current_classes`` to active.
        """
        if self._syncing:
            return
        ordered_classes = self._get_customized_first_class_order()
        if change["new"]:
            non_default, _ = split_default_classes(
                ordered_classes,
                {cls: getattr(widget, "value", self.default_color) for cls, widget in self.class_color_controls.items()},
                self.default_color,
            )
            self._active_classes = [cls for cls in non_default if cls in self.class_color_controls]
        else:
            self._active_classes = ordered_classes
        self._push_to_widget()

    def _log(self, message: str, error: bool = False, clear: bool = False) -> None:
        if error:
            self.ui_component.feedback_label.value = f'<span style="color:orange">⚠️ {message}</span>'
            _logger.warning(message)
        else:
            self.ui_component.feedback_label.value = f'<span style="color:green">✓ {message}</span>'
            _logger.info(message)

    def _determine_storage_folder(self) -> Optional[Path]:
        base_folder = getattr(self.main_viewer, "base_folder", None)
        if not base_folder:
            return None
        base_path = Path(base_folder).expanduser()
        target = base_path / ".UELer"
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception as err:  # pylint: disable=broad-except
            self._log(f"Unable to prepare palette folder at {target}: {err}", error=True, clear=True)
            return None
        return target

    def _migrate_legacy_palettes(self, target_folder: Path) -> None:
        try:
            target_folder.mkdir(parents=True, exist_ok=True)
        except Exception as err:  # pylint: disable=broad-except
            self._log(f"Unable to access palette directory {target_folder}: {err}", error=True)
            return

        candidates = []
        previous_default = Path.cwd()
        if previous_default.resolve() != target_folder.resolve():
            candidates.append(previous_default)

        for source in candidates:
            registry_path = source / REGISTRY_FILENAME
            color_files = list(source.glob(f"*{COLOR_SET_FILE_SUFFIX}"))
            if not registry_path.exists() and not color_files:
                continue

            for file_path in color_files:
                destination = target_folder / file_path.name
                if destination.exists():
                    continue
                try:
                    shutil.move(str(file_path), str(destination))
                except Exception as err:  # pylint: disable=broad-except
                    self._log(f"Failed to migrate {file_path}: {err}", error=True)

            if registry_path.exists():
                try:
                    records = load_registry(source)
                except Exception as err:  # pylint: disable=broad-except
                    self._log(f"Failed to read legacy registry {registry_path}: {err}", error=True)
                    continue

                updated: Dict[str, Dict[str, str]] = {}
                for name, record in records.items():
                    existing_name = Path(record.get("path", "")).name or f"{slugify_name(name)}{COLOR_SET_FILE_SUFFIX}"
                    destination = target_folder / existing_name
                    if not destination.exists():
                        continue
                    updated[name] = {
                        **record,
                        "path": str(destination.resolve()),
                        "folder": str(target_folder.resolve()),
                    }

                if updated:
                    save_registry(target_folder, updated)

                try:
                    backup_path = registry_path.with_suffix(registry_path.suffix + ".bak")
                    registry_path.rename(backup_path)
                except Exception:  # pylint: disable=broad-except
                    pass

    def _load_registry_for_folder(self, folder: Path) -> None:
        folder = folder.expanduser().resolve()
        self.registry_folder = folder
        self.ui_component.set_folder_path(folder)
        self.registry_records = load_registry(folder)
        for record in self.registry_records.values():
            record.setdefault("folder", str(folder))
        self._refresh_saved_sets_dropdown()
        if FileChooser and self.ui_component.load_picker is not None:
            try:
                self.ui_component.load_picker.reset(path=str(folder))
            except Exception:  # pylint: disable=broad-except
                pass

    def _refresh_saved_sets_dropdown(self) -> None:
        if not self.registry_records:
            self.ui_component.saved_sets_dropdown.options = [("No saved sets", "")]
            self.ui_component.saved_sets_dropdown.value = ""
            self._set_active_color_set_name(None)
            return

        option_entries = []
        for name in sorted(self.registry_records.keys()):
            record = self.registry_records.get(name, {})
            identifier = (record.get("identifier") or "").strip()
            label = f"{name} ({identifier})" if identifier else name
            option_entries.append((label, name))
        self.ui_component.saved_sets_dropdown.options = [("Select a set", "")] + option_entries
        self.ui_component.saved_sets_dropdown.value = ""
        if self.active_color_set_name and self.active_color_set_name not in self.registry_records:
            self._set_active_color_set_name(None)


class UiComponent:
    def __init__(self):
        self.SidePlots: Dict[str, object] = {}

        self.identifier_dropdown = Dropdown(description="Identifier:")
        self.update_button = Button(description="Update Colors", button_style="primary")
        self.enabled_checkbox = Checkbox(value=False, description="Enable", indent=False, tooltip="Enable mask painter")

        # Keep TagsInput and show_all_checkbox for backward compatibility with tests
        # that manipulate them directly; they are no longer rendered in the main UI.
        self.sorting_items_tagsinput = TagsInput(
            value=tuple(), allowed_tags=[], description="", allow_duplicates=False,
            layout=Layout(width="100%"),
        )
        self.show_all_checkbox = Checkbox(
            description="Show all", value=False, indent=False, layout=Layout(width="auto"),
        )
        self.default_color_picker = ColorPicker(
            description="Default:", value=DEFAULT_COLOR, layout=Layout(width="auto"),
        )
        self.global_fill_opacity_input = BoundedIntText(
            description="Opacity (%):",
            value=FILL_OPACITY_DEFAULT_PERCENT,
            min=0,
            max=100,
            step=1,
        )
        self.global_fill_opacity_input.layout.width = "95px"
        self.global_fill_opacity_input.layout.min_width = "95px"
        self.global_fill_checkbox = Checkbox(
            value=False,
            description="Global fill",
            indent=False,
            layout=Layout(width="auto"),
            tooltip="Apply fill mode to classes still inheriting the global fill setting",
        )
        self.border_color_mode_dropdown = Dropdown(
            options=[
                ("Mask color", BORDER_COLOR_MODE_MASK_TYPE),
                ("Same as fill", BORDER_COLOR_MODE_SAME_AS_FILL),
            ],
            value=BORDER_COLOR_MODE_MASK_TYPE,
            description="Border color:",
            layout=Layout(width="auto"),
        )
        self.show_fill_borders_checkbox = Checkbox(
            value=False,
            description="Borders on filled",
            indent=False,
            layout=Layout(width="auto"),
            tooltip="Render a border on top of filled masks",
        )
        # Kept for compatibility; no longer the primary class-row container.
        self.color_picker_box = VBox([], layout=Layout(overflow_y="auto", max_height="300px"))

        # The anywidget renders drag-sortable rows with color picker + vis/fill checkboxes.
        self.class_list_widget = MaskClassListWidget()

        self.only_specified_checkbox = Checkbox(
            value=False,
            description="Only specified",
            indent=False,
            layout=Layout(width="auto"),
            tooltip="Show only classes whose color differs from the default color",
        )

        # --- Coloring mode: categorical (default) vs continuous (issue #115) ---
        self.color_mode_toggle = ToggleButtons(
            options=[("Categorical", "categorical"), ("Continuous", "continuous")],
            value="categorical",
            tooltip="Color masks by discrete classes or by a continuous variable",
        )

        # Categorical controls (existing UI), grouped so it can be hidden as a unit.
        self.categorical_layout = VBox([
            HBox([self.default_color_picker, self.only_specified_checkbox]),
            HBox(
                [
                    self.global_fill_checkbox,
                    HTML("<div style='width:12px'></div>"),
                    self.global_fill_opacity_input,
                    self.show_fill_borders_checkbox,
                ],
                layout=Layout(align_items="center"),
            ),
            HBox([self.border_color_mode_dropdown]),
            HTML("<hr style='margin:4px 0'>"),
            self.class_list_widget,
        ])

        # Continuous controls (issue #115).
        self.continuous_column_dropdown = Dropdown(description="Value:", layout=Layout(width="auto"))
        self.colormap_dropdown = Dropdown(
            description="Colormap:",
            options=list(CONTINUOUS_COLORMAPS),
            value="viridis",
            layout=Layout(width="auto"),
        )
        self.arcsinh_checkbox = Checkbox(
            value=False, description="arcsinh transform", indent=False, layout=Layout(width="auto"),
        )
        self.arcsinh_cofactor_input = BoundedFloatText(
            value=5.0, min=1e-6, max=1e6, step=1.0, description="cofactor:", layout=Layout(width="160px"),
        )
        self.auto_range_checkbox = Checkbox(
            value=True, description="Auto range (1–99 pct)", indent=False, layout=Layout(width="auto"),
        )
        self.vmin_input = FloatText(description="vmin:", disabled=True, layout=Layout(width="150px"))
        self.vmax_input = FloatText(description="vmax:", disabled=True, layout=Layout(width="150px"))
        self.autorange_button = Button(description="Recompute range", icon="refresh")
        self.continuous_opacity_input = BoundedIntText(
            description="Opacity (%):", value=100, min=0, max=100, step=1, layout=Layout(width="130px"),
        )
        self.continuous_fill_checkbox = Checkbox(
            value=True, description="Fill (unchecked = outline)", indent=False, layout=Layout(width="auto"),
        )
        self.colorbar_output = Output(layout=Layout(height="70px"))

        self.continuous_layout = VBox(
            [
                self.continuous_column_dropdown,
                self.colormap_dropdown,
                HBox([self.arcsinh_checkbox, self.arcsinh_cofactor_input], layout=Layout(align_items="center")),
                HBox([self.auto_range_checkbox, self.autorange_button], layout=Layout(align_items="center")),
                HBox([self.vmin_input, self.vmax_input]),
                HBox([self.continuous_fill_checkbox, self.continuous_opacity_input], layout=Layout(align_items="center")),
                self.colorbar_output,
            ],
            layout=Layout(display="none"),
        )

        self.colors_layout = VBox([
            HBox([self.color_mode_toggle]),
            self.categorical_layout,
            self.continuous_layout,
        ])

        self.set_name_input = Text(description="Name:", placeholder="My palette")
        self.save_button = Button(description="Save set", icon="save", disabled=True)

        self.folder_display = Text(description="Folder:", value="", disabled=True, layout=Layout(width="70%"))
        self.change_folder_button = Button(description="Change location", icon="folder-open")
        folder_header = HBox([self.folder_display, self.change_folder_button])

        self.manual_folder_input = Text(description="Override:", placeholder="path/to/folder")
        self.manual_folder_apply = Button(description="Use override", button_style="info")
        self.manual_folder_cancel = Button(description="Cancel", button_style="warning")
        self.manual_folder_box = VBox(
            [
                self.manual_folder_input,
                HBox([self.manual_folder_apply, self.manual_folder_cancel]),
            ],
            layout=Layout(display="none", padding="0.5rem"),
        )

        save_box_children = [folder_header, self.manual_folder_box, self.set_name_input, self.save_button]
        self.save_box = VBox(save_box_children)

        default_folder = str(Path.cwd())
        if FileChooser is not None:
            self.load_picker = FileChooser(default_folder, filter_pattern=f"*{COLOR_SET_FILE_SUFFIX}")
            self.load_path_input = None
        else:
            self.load_picker = None
            self.load_path_input = Text(description="File:", placeholder=f"path/to/file{COLOR_SET_FILE_SUFFIX}")
        self.load_button = Button(description="Load file", icon="upload")

        load_box_children = []
        if self.load_picker is not None:
            load_box_children.append(self.load_picker)
        if self.load_path_input is not None:
            load_box_children.append(self.load_path_input)
        load_box_children.append(self.load_button)
        self.load_box = VBox(load_box_children)

        self.saved_sets_dropdown = Dropdown(description="", options=[("No saved sets", "")], layout=Layout(width="auto"))
        self.apply_saved_button = Button(description="Apply set")
        self.overwrite_saved_button = Button(description="Overwrite")
        self.delete_saved_button = Button(description="Delete", button_style="danger")

        manage_box_children = [
            self.saved_sets_dropdown,
            HBox([self.apply_saved_button, self.overwrite_saved_button, self.delete_saved_button]),
        ]
        self.manage_box = VBox(manage_box_children)

        self.color_set_tab = Tab(children=[self.save_box, self.load_box, self.manage_box])
        self.color_set_tab.set_title(0, "Save")
        self.color_set_tab.set_title(1, "Load")
        self.color_set_tab.set_title(2, "Manage")

        self.palette_accordion = Accordion(children=[self.color_set_tab], selected_index=None)
        self.palette_accordion.set_title(0, "Palette: Save / Load / Manage")

        self.feedback_label = HTML(value="")

        self.control_panel = None

    def set_folder_path(self, path: Path) -> None:
        self.folder_display.value = str(path)

    def toggle_manual_folder_box(self, visible: bool) -> None:
        self.manual_folder_box.layout.display = "block" if visible else "none"

    def show_manual_folder_box(self) -> None:
        self.toggle_manual_folder_box(True)