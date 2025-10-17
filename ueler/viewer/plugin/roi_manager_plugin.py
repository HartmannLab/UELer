import os
from types import SimpleNamespace
from typing import Optional

import pandas as pd
from ipywidgets import (
    Accordion,
    Button,
    Checkbox,
    Combobox,
    Dropdown,
    HBox,
    HTML,
    Layout,
    TagsInput,
    Text,
    Textarea,
    VBox,
)

from .plugin_base import PluginBase


class ROIManagerPlugin(PluginBase):
    """Plugin that encapsulates the ROI manager UI and interactions."""

    CURRENT_MARKER_VALUE = "__current__"

    STATUS_COLORS = {
        "info": "#424242",
        "success": "#2e7d32",
        "warning": "#f9a825",
        "error": "#c62828",
    }

    def __init__(self, main_viewer, width: int = 6, height: int = 3) -> None:
        super().__init__(main_viewer, width, height)
        self.displayed_name = "ROI manager"
        self.SidePlots_id = "roi_manager_output"
        self.main_viewer = main_viewer
        self.ui_component = SimpleNamespace()
        self._selected_roi_id: Optional[str] = None
        self._suspend_ui_events = False

        self._build_widgets()
        self._build_layout()
        self._connect_events()

        # Populate controls with current data
        self.refresh_marker_options()
        self.refresh_roi_table()

        # Subscribe to ROI manager updates
        self.main_viewer.roi_manager.observable.add_observer(self._on_roi_table_change)

        self.setup_widget_observers()
        self.initialized = True

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        full_width = Layout(width="100%")
        button_layout = Layout(width="auto", flex="1 1 auto")
        style_auto = {"description_width": "auto"}

        self.ui_component.roi_table = Dropdown(
            options=[("—", None)],
            value=None,
            description="Saved ROI:",
            layout=full_width,
            style=style_auto,
        )

        self.ui_component.limit_to_fov_checkbox = Checkbox(
            value=True,
            description="Only current FOV",
            indent=False,
            layout=Layout(width="auto"),
        )

        self.ui_component.capture_button = Button(
            description="Capture view",
            icon="camera",
            button_style="primary",
            layout=button_layout,
        )
        self.ui_component.update_button = Button(
            description="Update",
            icon="refresh",
            button_style="",
            layout=button_layout,
        )
        self.ui_component.delete_button = Button(
            description="Delete",
            icon="trash",
            button_style="danger",
            layout=button_layout,
        )

        self.ui_component.center_button = Button(
            description="Center",
            icon="crosshairs",
            button_style="",
            layout=button_layout,
        )

        self.ui_component.marker_dropdown = Dropdown(
            options=[("Current set", self.CURRENT_MARKER_VALUE), ("None", "")],
            value=self.CURRENT_MARKER_VALUE,
            description="Marker set:",
            layout=full_width,
            style=style_auto,
        )

        self.ui_component.tag_entry = Combobox(
            value="",
            options=[],
            placeholder="Type or select a tag",
            ensure_option=False,
            description="Add tag:",
            layout=full_width,
            style=style_auto,
        )
        if hasattr(self.ui_component.tag_entry, "continuous_update"):
            try:
                self.ui_component.tag_entry.continuous_update = False
            except Exception:  # pragma: no cover - defensive for older widgets
                pass

        self.ui_component.tags = TagsInput(
            value=(),
            allowed_tags=[],
            description="Tags:",
            allow_duplicates=False,
            allow_new=True,
            layout=full_width,
        )
        if hasattr(self.ui_component.tags, "allow_new"):
            try:
                self.ui_component.tags.allow_new = True
            except Exception:  # pragma: no cover - widget implementations may reject assignment
                pass
        else:  # pragma: no cover - fallback for older widgets lacking the trait
            setattr(self.ui_component.tags, "allow_new", True)
        if hasattr(self.ui_component.tags, "restrict_to_allowed_tags"):
            try:
                self.ui_component.tags.restrict_to_allowed_tags = False
            except Exception:  # pragma: no cover - defensive for older widgets
                pass

        self.ui_component.comment = Textarea(
            value="",
            placeholder="Add a comment",
            description="Comment:",
            layout=full_width,
            style=style_auto,
        )

        default_path = os.path.relpath(
            self.main_viewer.roi_manager.storage_path, self.main_viewer.base_folder
        )

        self.ui_component.path = Text(
            value=default_path,
            description="File path:",
            layout=full_width,
            style=style_auto,
        )

        self.ui_component.export_button = Button(
            description="Export", icon="download", layout=button_layout
        )
        self.ui_component.import_button = Button(
            description="Import", icon="upload", layout=button_layout
        )

        self.ui_component.status = HTML(value="")

    def _build_layout(self) -> None:
        actions_primary = HBox(
            [
                self.ui_component.capture_button,
                self.ui_component.update_button,
                self.ui_component.center_button,
            ],
            layout=Layout(gap="6px"),
        )

        actions_secondary = HBox(
            [self.ui_component.export_button, self.ui_component.import_button],
            layout=Layout(gap="6px"),
        )

        metadata_box = VBox(
            [
                self.ui_component.marker_dropdown,
                self.ui_component.tag_entry,
                self.ui_component.tags,
                self.ui_component.comment,
            ],
            layout=Layout(gap="6px"),
        )

        file_box = VBox(
            [self.ui_component.path, actions_secondary],
            layout=Layout(gap="6px"),
        )

        header = HTML("<strong>ROI manager</strong>")

        content = VBox(
            [
                header,
                HBox(
                    [self.ui_component.roi_table, self.ui_component.limit_to_fov_checkbox],
                    layout=Layout(align_items="center", gap="8px"),
                ),
                actions_primary,
                metadata_box,
                file_box,
                self.ui_component.status,
            ],
            layout=Layout(gap="10px"),
        )

        self.panel = Accordion(
            children=[content],
            titles=("ROI manager",),
            selected_index=None,
            layout=Layout(width="100%"),
        )
        self.ui = content

    def _connect_events(self) -> None:
        self.ui_component.roi_table.observe(self._on_selection_change, names="value")
        self.ui_component.limit_to_fov_checkbox.observe(
            lambda *_: self.refresh_roi_table(), names="value"
        )
        self.ui_component.capture_button.on_click(self._capture_current_view)
        self.ui_component.update_button.on_click(self._update_selected_roi)
        self.ui_component.delete_button.on_click(self._delete_selected_roi)
        self.ui_component.export_button.on_click(self._export_rois)
        self.ui_component.import_button.on_click(self._import_rois)
        self.ui_component.center_button.on_click(self._center_on_selected_roi)
        self.ui_component.tags.observe(self._on_tags_value_change, names="value")
        self.ui_component.tag_entry.observe(self._on_tag_entry_change, names="value")

    # ------------------------------------------------------------------
    # Event handlers and helpers
    # ------------------------------------------------------------------
    def _on_roi_table_change(self, df):  # pragma: no cover - observer callback
        self.refresh_roi_table(df)

    def _on_selection_change(self, change):
        if self._suspend_ui_events or change.get("name") != "value":
            return
        roi_id = change.get("new")
        if roi_id == self._selected_roi_id:
            return
        self._selected_roi_id = roi_id
        if roi_id:
            record = self.main_viewer.roi_manager.get_roi(roi_id)
            if record:
                self._populate_fields(record)
                return
        self._clear_fields()

    def refresh_marker_options(self) -> None:
        options = [("Current set", self.CURRENT_MARKER_VALUE), ("None", "")]
        marker_sets = sorted(self.main_viewer.marker_sets.keys())
        options.extend((name, name) for name in marker_sets)

        current_value = self.ui_component.marker_dropdown.value
        valid_values = {value for _, value in options}
        if current_value not in valid_values:
            current_value = self.CURRENT_MARKER_VALUE

        self.ui_component.marker_dropdown.options = options
        self.ui_component.marker_dropdown.value = current_value

    def _update_available_tags(self, table: pd.DataFrame) -> None:
        tag_pool: set[str] = set()
        if table is not None and not table.empty and "tags" in table.columns:
            for raw_tags in table["tags"].astype(str):
                if not raw_tags or raw_tags.lower() == "nan":
                    continue
                for tag in raw_tags.split(","):
                    cleaned = tag.strip()
                    if cleaned:
                        tag_pool.add(cleaned)

        tag_pool.update(self.ui_component.tags.value)
        allowed = sorted(tag_pool)
        if list(self.ui_component.tags.allowed_tags) != allowed:
            self.ui_component.tags.allowed_tags = allowed
            self._sync_tag_entry_options()

    def refresh_roi_table(self, df: Optional[pd.DataFrame] = None) -> None:
        source_table = self.main_viewer.roi_manager.table
        self._update_available_tags(source_table)

        df = df.copy() if df is not None else source_table
        if df is None:
            df = pd.DataFrame()

        if self.ui_component.limit_to_fov_checkbox.value:
            current_fov = getattr(self.main_viewer.ui_component.image_selector, "value", None)
            if current_fov:
                df = df[df["fov"] == current_fov]

        df = df.copy()
        if not df.empty and "created_at" in df.columns:
            df = df.sort_values("created_at", ascending=False)

        options = [("—", None)]
        for _, row in df.iterrows():
            options.append((self._format_roi_label(row), row.get("roi_id")))

        valid_ids = {value for _, value in options if value is not None}
        previous = self.ui_component.roi_table.value
        selection = self._selected_roi_id or previous
        if selection not in valid_ids:
            selection = None

        self._suspend_ui_events = True
        try:
            self.ui_component.roi_table.options = options
            self.ui_component.roi_table.value = selection
        finally:
            self._suspend_ui_events = False

        self._selected_roi_id = selection

        scope = (
            getattr(self.main_viewer.ui_component.image_selector, "value", "all FOVs")
            if self.ui_component.limit_to_fov_checkbox.value
            else "all FOVs"
        )
        self.set_status(f"{len(options) - 1} ROI(s) in {scope}")

    @staticmethod
    def _format_roi_label(record) -> str:
        marker = record.get("marker_set") or "—"
        tags = record.get("tags") or ""
        tag_display = f" [{tags}]" if tags else ""
        coords = (
            int(round(record.get("x_min", 0) or 0)),
            int(round(record.get("y_min", 0) or 0)),
            int(round(record.get("x_max", 0) or 0)),
            int(round(record.get("y_max", 0) or 0)),
        )
        return f"{record.get('fov', '—')} · {marker}{tag_display} · {coords[0]}:{coords[1]} → {coords[2]}:{coords[3]}"

    def _resolve_marker_set_choice(self) -> str:
        choice = self.ui_component.marker_dropdown.value
        if choice == self.CURRENT_MARKER_VALUE:
            active = getattr(self.main_viewer.ui_component.marker_set_dropdown, "value", None)
            if active:
                return active
            channels = getattr(self.main_viewer.ui_component.channel_selector, "value", ())
            if channels:
                return "current:" + ",".join(channels)
            return ""
        return choice or ""

    def _capture_current_view(self, _):
        viewport = self.main_viewer.capture_viewport_bounds()
        if viewport is None:
            self.set_status("Unable to read viewport bounds.", level="error")
            return

        record = {
            "fov": getattr(self.main_viewer.ui_component.image_selector, "value", ""),
            **viewport,
            "marker_set": self._resolve_marker_set_choice(),
            "tags": list(self.ui_component.tags.value),
            "comment": self.ui_component.comment.value.strip(),
        }

        result = self.main_viewer.roi_manager.add_roi(record)
        self._selected_roi_id = result.get("roi_id")
        self.refresh_roi_table()
        self.set_status("ROI captured from current view.", level="success")

    def _update_selected_roi(self, _):
        if not self._selected_roi_id:
            self.set_status("Select an ROI to update.", level="warning")
            return

        viewport = self.main_viewer.capture_viewport_bounds()
        if viewport is None:
            self.set_status("Unable to read viewport bounds.", level="error")
            return

        updates = {
            **viewport,
            "fov": getattr(self.main_viewer.ui_component.image_selector, "value", ""),
            "marker_set": self._resolve_marker_set_choice(),
            "tags": list(self.ui_component.tags.value),
            "comment": self.ui_component.comment.value.strip(),
        }

        updated = self.main_viewer.roi_manager.update_roi(self._selected_roi_id, updates)
        if updated:
            self.refresh_roi_table()
            self.set_status("ROI updated.", level="success")
        else:
            self.set_status("Failed to update ROI.", level="error")

    def _delete_selected_roi(self, _):
        if not self._selected_roi_id:
            self.set_status("Select an ROI to delete.", level="warning")
            return
        if self.main_viewer.roi_manager.delete_roi(self._selected_roi_id):
            self._selected_roi_id = None
            self.refresh_roi_table()
            self.set_status("ROI deleted.", level="success")
        else:
            self.set_status("Failed to delete ROI.", level="error")

    def _export_rois(self, _):
        path = self.ui_component.path.value.strip() or self.main_viewer.roi_manager.storage_path
        if not os.path.isabs(path):
            path = os.path.join(self.main_viewer.base_folder, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        target = self.main_viewer.roi_manager.export_to_csv(path)
        self.set_status(f"Exported ROI table to {target}.", level="success")

    def _import_rois(self, _):
        path = self.ui_component.path.value.strip() or self.main_viewer.roi_manager.storage_path
        if not os.path.isabs(path):
            path = os.path.join(self.main_viewer.base_folder, path)
        if not os.path.exists(path):
            self.set_status(f"File not found: {path}", level="error")
            return
        try:
            self.main_viewer.roi_manager.import_from_csv(path, merge=True)
            self.refresh_roi_table()
            self.set_status(f"Imported ROI table from {path}.", level="success")
        except Exception as exc:  # pragma: no cover - defensive
            self.set_status(f"Import failed: {exc}", level="error")

    def _center_on_selected_roi(self, _):
        if not self._selected_roi_id:
            self.set_status("Select an ROI to center on.", level="warning")
            return
        record = self.main_viewer.roi_manager.get_roi(self._selected_roi_id)
        if not record:
            self.set_status("Selected ROI no longer exists.", level="error")
            self.refresh_roi_table()
            return
        self.main_viewer.center_on_roi(record)
        self.set_status("Centered on ROI.", level="success")

    def _on_tags_value_change(self, change) -> None:
        if self._suspend_ui_events or change.get("name") != "value":
            return

        widget = self.ui_component.tags
        new_payload = change.get("new")
        old_payload = change.get("old")

        if getattr(self.main_viewer, "_debug", False):  # pragma: no cover - debug aid
            self._log(
                f"TagsInput change received (old={old_payload!r}, new={new_payload!r})",
                clear=False,
            )

        incoming_tags = self._normalise_tag_input(new_payload)
        current_value = list(self._normalise_tag_input(widget.value))

        if isinstance(new_payload, str) and incoming_tags:
            target_value = current_value or []
            for tag in incoming_tags:
                if tag not in target_value:
                    target_value.append(tag)
        elif incoming_tags:
            target_value = incoming_tags
        elif isinstance(new_payload, str):
            target_value = current_value
        else:
            target_value = []

        if not target_value and current_value:
            target_value = current_value

        self._merge_allowed_tags(tuple(target_value))
        self._apply_tags_value(tuple(target_value))
        self._sync_tag_entry_options()

    def _on_tag_entry_change(self, change) -> None:
        if self._suspend_ui_events or change.get("name") != "value":
            return

        new_payload = change.get("new")
        if getattr(self.main_viewer, "_debug", False):  # pragma: no cover - debug aid
            self._log(f"Tag entry change received: {new_payload!r}", clear=False)

        incoming = self._normalise_tag_input(new_payload)
        if not incoming:
            return

        tags_widget = self.ui_component.tags
        current = list(tags_widget.value or [])
        updated = False
        for tag in incoming:
            if tag not in current:
                current.append(tag)
                updated = True

        if updated:
            merged = tuple(current)
            self._merge_allowed_tags(merged)
            self._apply_tags_value(merged)
        self._sync_tag_entry_options()

        self._suspend_ui_events = True
        try:
            if isinstance(self.ui_component.tag_entry.value, str):
                self.ui_component.tag_entry.value = ""
            else:
                self.ui_component.tag_entry.value = None
        finally:
            self._suspend_ui_events = False

    def _apply_tags_value(self, tags: tuple[str, ...]) -> None:
        normalised = tuple(dict.fromkeys(tags))
        current = tuple(self.ui_component.tags.value or ())
        if normalised == current:
            return
        self._suspend_ui_events = True
        try:
            self.ui_component.tags.value = normalised
        finally:
            self._suspend_ui_events = False

    def _merge_allowed_tags(self, tags: tuple[str, ...]) -> None:
        if not tags:
            return
        existing = list(self.ui_component.tags.allowed_tags)
        merged = list(existing)
        updated = False
        for tag in tags:
            if tag not in merged:
                merged.append(tag)
                updated = True
        if updated:
            self.ui_component.tags.allowed_tags = merged
            self._sync_tag_entry_options()

    def _sync_tag_entry_options(self) -> None:
        entry = getattr(self.ui_component, "tag_entry", None)
        if entry is None:
            return
        current_options = list(entry.options or [])
        desired = list(current_options)
        for tag in self.ui_component.tags.allowed_tags:
            if tag not in desired:
                desired.append(tag)
        if desired != current_options:
            entry.options = desired

    @staticmethod
    def _normalise_tag_input(payload) -> list[str]:
        if payload is None:
            return []
        if isinstance(payload, str):
            iterable = [payload]
        elif isinstance(payload, (list, tuple, set)):
            iterable = payload
        else:
            try:
                iterable = list(payload)
            except TypeError:
                iterable = [payload]

        cleaned: list[str] = []
        for tag in iterable:
            text = str(tag).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned

    def _populate_fields(self, record) -> None:
        self._suspend_ui_events = True
        try:
            marker_value = record.get("marker_set") or ""
            options = list(self.ui_component.marker_dropdown.options)
            option_values = {value for _, value in options}
            if marker_value and marker_value not in option_values:
                options.append((marker_value, marker_value))
                self.ui_component.marker_dropdown.options = options
            if marker_value:
                self.ui_component.marker_dropdown.value = marker_value
            else:
                self.ui_component.marker_dropdown.value = ""

            tags = record.get("tags") or ""
            tag_values = tuple(tag.strip() for tag in str(tags).split(",") if tag.strip())
            self.ui_component.tags.value = tag_values
            comment_value = record.get("comment", "")
            if pd.isna(comment_value):
                comment_value = ""
            else:
                comment_value = str(comment_value)
            self.ui_component.comment.value = comment_value
        finally:
            self._suspend_ui_events = False

    def _clear_fields(self) -> None:
        self._suspend_ui_events = True
        try:
            self.ui_component.marker_dropdown.value = self.CURRENT_MARKER_VALUE
            self.ui_component.tags.value = ()
            self.ui_component.comment.value = ""
        finally:
            self._suspend_ui_events = False

    def set_status(self, message: str, level: str = "info") -> None:
        color = self.STATUS_COLORS.get(level, self.STATUS_COLORS["info"])
        self.ui_component.status.value = f"<span style='color:{color}'>{message}</span>"

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def on_fov_change(self) -> None:  # type: ignore[override]
        self.refresh_roi_table()

    def on_marker_sets_changed(self) -> None:
        self.refresh_marker_options()

    def after_all_plugins_loaded(self) -> None:  # type: ignore[override]
        super().after_all_plugins_loaded()
        self.refresh_marker_options()
        self.refresh_roi_table()