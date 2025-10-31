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
    Button,
    Checkbox,
    ColorPicker,
    Dropdown,
    HBox,
    Layout,
    Output,
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
        self.current_classes: List[str] = []
        self.current_identifier: Optional[str] = None
        self.selected_classes: List[str] = []
        self.hidden_color_cache: Dict[str, str] = {}
        self.registry_records: Dict[str, Dict[str, str]] = {}
        self.active_color_set_name = ""

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

        self._load_registry_for_folder(self.registry_folder)
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
        color_picker_widgets: List[ColorPicker] = []
        for cls in classes:
            picker = ColorPicker(description=str(cls), value=self.default_color, layout=Layout(width="auto"))
            self.class_color_controls[cls] = picker
            color_picker_widgets.append(picker)

        self.ui_component.color_picker_box.children = tuple(color_picker_widgets)
        allowed_tags = list(classes)
        self.ui_component.sorting_items_tagsinput.allowed_tags = allowed_tags
        default_selection = tuple(classes[: min(len(classes), 6)]) if classes else tuple()
        self.ui_component.sorting_items_tagsinput.value = default_selection
        self._log(f"Identifier set to '{identifier}' with {len(classes)} classes.", clear=True)

    def on_sorting_items_change(self, change):
        if not self.class_color_controls:
            return

        new_order = [str(tag) for tag in change["new"]]
        if self.ui_component.show_all_checkbox.value:
            self.selected_classes = list(new_order)
            self._refresh_color_picker_display(preferred_order=new_order)
            return

        self._handle_selection_transition(new_order)
        self._refresh_color_picker_display()

    def on_show_all_toggle(self, change):
        if not self.class_color_controls:
            return

        if change["new"]:
            for key, color in list(self.hidden_color_cache.items()):
                picker = self.class_color_controls.get(key)
                if picker is not None:
                    picker.value = color
            self.hidden_color_cache.clear()
            self.selected_classes = list(self.class_color_controls.keys())
            self._refresh_color_picker_display()
        else:
            current_selection = [str(tag) for tag in self.ui_component.sorting_items_tagsinput.value]
            self._handle_selection_transition(current_selection)
            self._refresh_color_picker_display()

    def on_default_color_change(self, change):
        self._set_default_color(change.get("new"), update_ui=False)

    def _handle_selection_transition(self, new_order: Sequence[str]) -> None:
        new_set = set(new_order)
        old_set = set(self.selected_classes)

        # Cache colors for classes leaving the selection
        for key in old_set - new_set:
            picker = self.class_color_controls.get(key)
            if picker is None:
                continue
            self.hidden_color_cache[key] = picker.value or self.hidden_color_cache.get(key, self.default_color)
            picker.value = self.default_color

        # Restore cached colors for classes returning to the selection
        for key in new_set - old_set:
            picker = self.class_color_controls.get(key)
            if picker is None:
                continue
            cached = self.hidden_color_cache.pop(key, None)
            if cached:
                picker.value = cached

        self.selected_classes = list(new_order)

    def _refresh_color_picker_display(self, preferred_order: Optional[Sequence[str]] = None) -> None:
        if not self.class_color_controls:
            self.ui_component.color_picker_box.children = tuple()
            return

        if self.ui_component.show_all_checkbox.value:
            order = list(preferred_order or self.ui_component.sorting_items_tagsinput.value)
            visible_keys = [key for key in order if key in self.class_color_controls]
            visible_keys.extend(
                key for key in self.class_color_controls.keys() if key not in visible_keys
            )
        else:
            visible_keys = list(self.selected_classes)

        children = tuple(
            self.class_color_controls[key]
            for key in visible_keys
            if key in self.class_color_controls
        )
        self.ui_component.color_picker_box.children = children

    def _set_default_color(self, new_color: Optional[str], update_ui: bool = True) -> None:
        normalized = normalize_hex_color(new_color) or DEFAULT_COLOR
        old_color = getattr(self, "default_color", DEFAULT_COLOR)
        if colors_match(normalized, old_color):
            if update_ui and self.ui_component.default_color_picker.value != normalized:
                picker = self.ui_component.default_color_picker
                picker.unobserve(self.on_default_color_change, names="value")
                try:
                    picker.value = normalized
                finally:
                    picker.observe(self.on_default_color_change, names="value")
            self.default_color = normalized
            return

        self.default_color = normalized
        for picker in self.class_color_controls.values():
            if colors_match(picker.value, old_color):
                picker.value = normalized
        for key, value in list(self.hidden_color_cache.items()):
            if colors_match(value, old_color):
                self.hidden_color_cache[key] = normalized

        if not self.ui_component.show_all_checkbox.value:
            defaulted = []
            for key in list(self.selected_classes):
                picker = self.class_color_controls.get(key)
                if picker is not None and colors_match(picker.value, normalized):
                    defaulted.append(key)
            if defaulted:
                for key in defaulted:
                    picker_widget = self.class_color_controls.get(key)
                    if picker_widget is not None:
                        picker_widget.value = normalized
                    self.hidden_color_cache[key] = normalized
                self.selected_classes = [key for key in self.selected_classes if key not in defaulted]
                tags_input = self.ui_component.sorting_items_tagsinput
                tags_input.unobserve(self.on_sorting_items_change, names="value")
                try:
                    tags_input.value = tuple(self.selected_classes)
                finally:
                    tags_input.observe(self.on_sorting_items_change, names="value")

        if update_ui:
            picker = self.ui_component.default_color_picker
            picker.unobserve(self.on_default_color_change, names="value")
            try:
                picker.value = normalized
            finally:
                picker.observe(self.on_default_color_change, names="value")

        self._refresh_color_picker_display()

    def _get_visible_classes(self) -> List[str]:
        if self.ui_component.show_all_checkbox.value:
            order = list(self.ui_component.sorting_items_tagsinput.value)
            if not order:
                order = list(self.class_color_controls.keys())
            remainder = [
                key for key in self.class_color_controls.keys() if key not in order
            ]
            return order + remainder
        return list(self.selected_classes or self.ui_component.sorting_items_tagsinput.value)

    def _get_hidden_classes(self) -> List[str]:
        if self.ui_component.show_all_checkbox.value:
            return []
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
            timestamp = datetime.utcnow().isoformat() + "Z"
            identifier = self.current_identifier

            payload = {
                "name": name,
                "version": COLOR_SET_VERSION,
                "identifier": identifier,
                "default_color": self.default_color,
                "class_order": list(class_order),
                "colors": color_map,
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
            timestamp = datetime.utcnow().isoformat() + "Z"
            identifier = self.current_identifier

            payload = {
                "name": name,
                "version": COLOR_SET_VERSION,
                "identifier": identifier,
                "default_color": self.default_color,
                "class_order": list(class_order),
                "colors": color_map,
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

        ordered_unique = []
        seen = set()
        for cls in incoming_classes:
            if cls not in seen:
                ordered_unique.append(cls)
                seen.add(cls)
        for cls in self.class_color_controls.keys():
            if cls not in seen:
                ordered_unique.append(cls)
                seen.add(cls)

        current_colors = {
            key: self.class_color_controls[key].value
            for key in ordered_unique
            if key in self.class_color_controls
        }
        non_default_classes, defaulted_classes = split_default_classes(
            ordered_unique,
            current_colors,
            self.default_color,
        )

        allowed = list(self.ui_component.sorting_items_tagsinput.allowed_tags)
        combined_allowed = sorted(set(allowed) | set(ordered_unique))
        self.ui_component.sorting_items_tagsinput.allowed_tags = combined_allowed

        selection = [cls for cls in non_default_classes if cls in combined_allowed]
        tags_input = self.ui_component.sorting_items_tagsinput
        tags_input.unobserve(self.on_sorting_items_change, names="value")
        try:
            tags_input.value = tuple(selection)
        finally:
            tags_input.observe(self.on_sorting_items_change, names="value")

        if self.ui_component.show_all_checkbox.value:
            self.hidden_color_cache.clear()
            self.selected_classes = list(selection)
        else:
            self._handle_selection_transition(selection)
            for key in defaulted_classes:
                picker = self.class_color_controls.get(key)
                if picker is not None:
                    picker.value = self.default_color

        self._refresh_color_picker_display()
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
    def apply_colors_to_masks(self, _):
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

        current_fov = self.main_viewer.ui_component.image_selector.value

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

        def _apply_color(cls_value, color):
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

        for cls_str, cls_value in reversed(converted_visible):
            picker = self.class_color_controls.get(cls_str)
            if picker is None:
                self._log(f"Class '{cls_str}' not found in controls; skipping.", error=True)
                continue
            _apply_color(cls_value, picker.value)

        if hidden_classes:
            for cls_str, cls_value in converted_hidden:
                _apply_color(cls_value, self.default_color)

        self._log("Masks updated with class-based colors.")

    def on_cell_table_change(self):
        self._initialise_identifier_options()

    def on_mv_update_display(self):
        if self.ui_component.enabled_checkbox.value:
            self.apply_colors_to_masks(None)

    def initiate_ui(self):
        controls = HBox([
            self.ui_component.identifier_dropdown,
            self.ui_component.update_button,
            self.ui_component.enabled_checkbox,
        ])

        self.ui_component.control_panel = controls
        self.ui = VBox(
            [
                controls,
                self.ui_component.colors_layout,
                self.ui_component.color_set_tab,
                self.ui_component.feedback_output,
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

    def _log(self, message: str, error: bool = False, clear: bool = False) -> None:
        with self.ui_component.feedback_output:
            if clear:
                self.ui_component.feedback_output.clear_output(wait=True)
            prefix = "⚠️ " if error else ""
            print(f"{prefix}{message}")

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
        self.update_button = Button(description="Update Colors")
        self.enabled_checkbox = Checkbox(value=True, description="Enable", tooltip="Enable mask painter")

        self.sorting_items_tagsinput = TagsInput(value=tuple(), allowed_tags=[], description="Items:", allow_duplicates=False)
        self.show_all_checkbox = Checkbox(description="Show all classes", value=False)
        self.default_color_picker = ColorPicker(description="Default color", value=DEFAULT_COLOR, layout=Layout(width="auto"))
        self.color_picker_box = VBox([], layout=Layout(width="70%"))

        self.sorting_container = VBox(
            [self.sorting_items_tagsinput, self.show_all_checkbox, self.default_color_picker],
            layout=Layout(width="30%", overflow_x="auto"),
        )
        self.colors_layout = HBox([
            self.sorting_container,
            self.color_picker_box,
        ])

        self.set_name_input = Text(description="Name:", placeholder="My palette")
        self.save_button = Button(description="Save set", icon="save")

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

        self.saved_sets_dropdown = Dropdown(description="Saved sets:", options=[("No saved sets", "")])
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

        self.feedback_output = Output(layout=Layout(max_height="120px", overflow_y="auto"))

        self.control_panel = None

    def set_folder_path(self, path: Path) -> None:
        self.folder_display.value = str(path)

    def toggle_manual_folder_box(self, visible: bool) -> None:
        self.manual_folder_box.layout.display = "block" if visible else "none"

    def show_manual_folder_box(self) -> None:
        self.toggle_manual_folder_box(True)