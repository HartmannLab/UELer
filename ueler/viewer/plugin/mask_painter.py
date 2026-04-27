# viewer/plugin/mask_painter.py
from __future__ import annotations

import importlib
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from ipywidgets import (
    Accordion,
    Button,
    Checkbox,
    ColorPicker,
    Dropdown,
    HTML,
    HBox,
    Label,
    Layout,
    Tab,
    TagsInput,
    Text,
    VBox,
)

_FileChooserModule = None
try:  # pragma: no cover - optional dependency
    _FileChooserModule = importlib.import_module("ipyfilechooser")
except Exception:  # pragma: no cover - executed when ipyfilechooser is absent
    _FileChooserModule = None

FileChooser = getattr(_FileChooserModule, "FileChooser", None)

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
from ueler.rendering import set_cell_color, get_cell_color, clear_cell_colors, set_cell_colors_bulk
COLOR_SET_FILE_SUFFIX = ".maskcolors.json"
REGISTRY_FILENAME = "mask_color_sets_index.json"
COLOR_SET_VERSION = "1.0.0"


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
        self.current_classes: List[str] = []
        self.current_identifier: Optional[str] = None
        self.selected_classes: List[str] = []
        self.hidden_color_cache: Dict[str, str] = {}
        self.registry_records: Dict[str, Dict[str, str]] = {}
        self.active_color_set_name = ""
        self._cell_mode_cache: Dict[str, Dict[int, str]] = {}  # fov -> {mask_id: mode}
        self._last_applied_class_modes: Dict[str, str] = {}
        self._syncing: bool = False  # guard against anywidget ↔ ipywidget sync loops
        
        # Track last applied state to avoid unnecessary re-application
        self._last_applied_fov: Optional[str] = None
        self._last_applied_identifier: Optional[str] = None
        self._last_applied_classes: Optional[set] = None
        self._last_applied_class_colors: dict[str, str] = {}

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

        # Wire anywidget class-list traitlet changes → Python state
        _w = self.ui_component.class_list_widget
        _w.observe(self._pull_from_widget, names=["class_order", "class_colors", "class_visible", "class_fill"])

        self._load_registry_for_folder(self.registry_folder)
        self._refresh_save_button_state()
        self.initiate_ui()

    def _initialise_identifier_options(self) -> None:
        identifier_options: Iterable[str] = []
        if self.main_viewer.cell_table is not None:
            cell_table = self.main_viewer.cell_table
            identifier_options = cell_table.select_dtypes(include=["int", "int64", "object", "bool"]).columns.tolist()

        self.ui_component.identifier_dropdown.options = list(identifier_options)

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

        self.class_color_controls.clear()
        self.class_visible_controls.clear()
        self.class_mode_controls.clear()
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
            self.class_mode_controls[cls] = mode_cb

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

    def _get_visible_classes(self) -> List[str]:
        """Return classes that are currently visible (vis checkbox ticked), in display order."""
        # Use the anywidget order when available; fall back to selected_classes or
        # sorting_items_tagsinput for test setups that bypass the anywidget.
        order = list(self.ui_component.class_list_widget.class_order)
        if not order:
            if self.ui_component.show_all_checkbox.value:
                ti_order = list(self.ui_component.sorting_items_tagsinput.value)
                order = ti_order + [
                    k for k in self.class_color_controls if k not in set(ti_order)
                ]
            else:
                order = list(self.selected_classes or self.ui_component.sorting_items_tagsinput.value)
        # Filter by per-class visibility checkbox
        return [
            key for key in order
            if key in self.class_color_controls
            and (
                self.class_visible_controls.get(key) is None
                or self.class_visible_controls[key].value
            )
        ]

    def _get_hidden_classes(self) -> List[str]:
        """Return classes that are either not selected or have their checkbox unchecked."""
        if self.ui_component.show_all_checkbox.value:
            # Only checkbox-hidden classes
            return [
                key for key, cb in self.class_visible_controls.items()
                if not cb.value
            ]
        visible = set(self._get_visible_classes())
        return [key for key in self.class_color_controls.keys() if key not in visible]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_current_color_set(self, _):
        try:
            folder = self.registry_folder
            name = self.ui_component.set_name_input.value.strip()
            if not name:
                raise ColorSetError("Please provide a name for the color set.")
            if not self.class_color_controls:
                raise ColorSetError("Select an identifier before saving colors.")

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
            timestamp = datetime.utcnow().isoformat() + "Z"
            identifier = self.current_identifier

            payload = {
                "name": name,
                "version": COLOR_SET_VERSION,
                "identifier": identifier,
                "default_color": self.default_color,
                "class_order": list(class_order),
                "colors": color_map,
                "modes": modes_map,
                "visible": visible_map,
                "saved_at": timestamp,
            }

            target_path = self._resolve_color_set_path(folder, name)
            write_color_set_file(target_path, payload)
            self._update_registry(folder, name, target_path, identifier, timestamp)
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

            if not self.class_color_controls:
                raise ColorSetError("Select an identifier before overwriting a color set.")

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
            timestamp = datetime.utcnow().isoformat() + "Z"
            identifier = self.current_identifier

            payload = {
                "name": name,
                "version": COLOR_SET_VERSION,
                "identifier": identifier,
                "default_color": self.default_color,
                "class_order": list(class_order),
                "colors": color_map,
                "modes": modes_map,
                "visible": visible_map,
                "saved_at": timestamp,
            }

            write_color_set_file(path, payload)
            self._update_registry(folder, name, path, identifier, timestamp)
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
        self.current_classes = ordered_unique

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

        self._push_to_widget()
        self._log(f"Loaded color set '{payload.get('name', path.stem)}' from {path}.", clear=True)
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

        # Apply colors to visible classes
        for cls_str, cls_value in reversed(converted_visible):
            picker = self.class_color_controls.get(cls_str)
            if picker is None:
                self._log(f"Class '{cls_str}' not found in controls; skipping.", error=True)
                continue

            color = picker.value
            mode_cb = self.class_mode_controls.get(cls_str)
            mode = ("fill" if mode_cb.value else "outline") if mode_cb is not None else "outline"
            if current_fov:
                _apply_color_to_current_fov(cls_value, color)
            if register_globally:
                _register_color_globally(cls_str, cls_value, color)
                _register_mode_globally(cls_str, cls_value, mode)

        # Apply empty-string sentinel to hidden classes so they are invisible
        if hidden_classes:
            for cls_str, cls_value in converted_hidden:
                if current_fov:
                    _apply_color_to_current_fov(cls_value, "")
                if register_globally:
                    _register_color_globally(cls_str, cls_value, "")

        self._log("Masks updated with class-based colors.")
        
        # Track the applied state
        self._last_applied_fov = current_fov
        self._last_applied_identifier = identifier
        self._last_applied_classes = set(visible_classes)

        if notify_cell_gallery:
            self._notify_cell_gallery_update()

    def _on_visibility_or_mode_change(self, _change):
        """Called when any per-class visibility checkbox or mode ToggleButtons changes.

        Forces ``apply_colors_to_masks`` to re-run so hidden classes immediately
        receive the ``""`` sentinel and visible classes get their current color/mode.
        Resets the dirty-skip caches so the change is not skipped.
        """
        if not self.ui_component.enabled_checkbox.value:
            return
        # Invalidate caches so apply_colors_to_masks doesn't skip the re-write
        self._last_applied_class_colors.clear()
        self._last_applied_class_modes.clear()
        map_mode = self.main_viewer.get_active_fov() is None
        self.apply_colors_to_masks(
            None,
            notify_cell_gallery=False,
            register_globally=map_mode,
        )

    def on_cell_table_change(self):
        self._initialise_identifier_options()
        # Clear tracking state when cell table changes
        self._last_applied_fov = None
        self._last_applied_identifier = None
        self._last_applied_classes = None
        self._last_applied_class_colors = {}
        self._last_applied_class_modes = {}
        self._cell_mode_cache = {}

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
                if getattr(self.main_viewer, "_debug", False):
                    print(f"[mask_painter] Failed to notify cell gallery: {exc}")

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
        """Enable Save set button only when both a name and an identifier are set."""
        name_val = getattr(self.ui_component.set_name_input, "value", None) or ""
        has_name = bool(str(name_val).strip())
        has_identifier = bool(getattr(self.ui_component.identifier_dropdown, "value", None))
        self.ui_component.save_button.disabled = not (has_name and has_identifier)

    def _push_to_widget(self, _=None) -> None:
        """Push the current Python-side state (colors, visibility, order) to the anywidget."""
        if self._syncing or not self.class_color_controls:
            return
        w = self.ui_component.class_list_widget
        order = self._get_full_class_order()
        colors = {cls: (p.value or self.default_color) for cls, p in self.class_color_controls.items()}
        visible = {cls: cb.value for cls, cb in self.class_visible_controls.items()}
        fill = {cls: cb.value for cls, cb in self.class_mode_controls.items()}
        self._syncing = True
        try:
            w.class_order = order
            w.class_colors = colors
            w.class_visible = visible
            w.class_fill = fill
            w.default_color = self.default_color
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
                self.current_classes = list(w.class_order)
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
        finally:
            self._syncing = False
        # Visibility or fill changes need an immediate re-apply when painter is active
        if name in ("class_visible", "class_fill") and self.ui_component.enabled_checkbox.value:
            self._last_applied_class_colors.clear()
            self._last_applied_class_modes.clear()
            map_mode = self.main_viewer.get_active_fov() is None
            self.apply_colors_to_masks(None, notify_cell_gallery=False, register_globally=map_mode)

    def _log(self, message: str, error: bool = False, clear: bool = False) -> None:
        if error:
            self.ui_component.feedback_label.value = f'<span style="color:orange">⚠️ {message}</span>'
        else:
            self.ui_component.feedback_label.value = f'<span style="color:green">✓ {message}</span>'

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
        self.enabled_checkbox = Checkbox(value=True, description="Enable", indent=False, tooltip="Enable mask painter")

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
        # Kept for compatibility; no longer the primary class-row container.
        self.color_picker_box = VBox([], layout=Layout(overflow_y="auto", max_height="300px"))

        # The anywidget renders drag-sortable rows with color picker + vis/fill checkboxes.
        self.class_list_widget = MaskClassListWidget()

        self.colors_layout = VBox([
            HBox([self.default_color_picker]),
            HTML("<hr style='margin:4px 0'>"),
            self.class_list_widget,
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