import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
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
    Output,
    SelectMultiple,
    Tab,
    TagsInput,
    Text,
    Textarea,
    VBox,
)
from matplotlib.colors import to_rgb

from ueler.rendering import ChannelRenderSettings, render_roi_to_array

from .plugin_base import PluginBase
from ..layout_utils import column_block_layout, flex_fill_layout


@dataclass
class _MarkerProfile:
    name: str
    selected_channels: Tuple[str, ...]
    channel_settings: Dict[str, ChannelRenderSettings]


class ROIManagerPlugin(PluginBase):
    """Plugin that encapsulates the ROI manager UI and interactions."""

    CURRENT_MARKER_VALUE = "__current__"

    STATUS_COLORS = {
        "info": "#424242",
        "success": "#2e7d32",
        "warning": "#f9a825",
        "error": "#c62828",
    }

    BROWSER_COLUMNS = 3
    BROWSER_MAX_TILES = 12

    def __init__(self, main_viewer, width: int = 6, height: int = 3) -> None:
        super().__init__(main_viewer, width, height)
        self.displayed_name = "ROI manager"
        self.SidePlots_id = "roi_manager_output"
        self.main_viewer = main_viewer
        self.ui_component = SimpleNamespace()
        self._selected_roi_id: Optional[str] = None
        self._suspend_ui_events = False
        self._suspend_browser_events = False
        self._browser_axis_to_roi: Dict[object, str] = {}
        self._browser_click_cid: Optional[int] = None
        self._browser_figure = None

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
        self._build_editor_widgets()
        self._build_browser_widgets()

    def _build_editor_widgets(self) -> None:
        full_width = column_block_layout
        button_layout = Layout(width="auto", flex="1 1 auto")
        style_auto = {"description_width": "auto"}

        self.ui_component.roi_table = Dropdown(
            options=[("—", None)],
            value=None,
            description="Saved ROI:",
            layout=flex_fill_layout(),
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
        self.ui_component.center_with_preset_button = Button(
            description="Center with preset",
            icon="bullseye",
            button_style="",
            layout=button_layout,
        )

        self.ui_component.marker_dropdown = Dropdown(
            options=[("Current set", self.CURRENT_MARKER_VALUE), ("None", "")],
            value=self.CURRENT_MARKER_VALUE,
            description="Marker set:",
            layout=full_width(),
            style=style_auto,
        )

        self.ui_component.tag_entry = Combobox(
            value="",
            options=[],
            placeholder="Type or select a tag",
            ensure_option=False,
            description="Add tag:",
            layout=full_width(),
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
            layout=full_width(),
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
            layout=full_width(),
            style=style_auto,
        )

        self.ui_component.annotation_summary = HTML("<em>Annotation palette: —</em>")
        self.ui_component.mask_summary = HTML("<em>Mask color set: —</em>")

        default_path = os.path.relpath(
            self.main_viewer.roi_manager.storage_path, self.main_viewer.base_folder
        )

        self.ui_component.path = Text(
            value=default_path,
            description="File path:",
            layout=flex_fill_layout(),
            style=style_auto,
        )

        self.ui_component.export_button = Button(
            description="Export", icon="download", layout=button_layout
        )
        self.ui_component.import_button = Button(
            description="Import", icon="upload", layout=button_layout
        )

        self.ui_component.status = HTML(value="")

        actions_primary = HBox(
            [
                self.ui_component.capture_button,
                self.ui_component.update_button,
                self.ui_component.center_button,
                self.ui_component.center_with_preset_button,
            ],
            layout=Layout(gap="6px", flex_flow="row wrap"),
        )

        actions_secondary = HBox(
            [
                self.ui_component.delete_button,
                self.ui_component.export_button,
                self.ui_component.import_button,
            ],
            layout=Layout(gap="6px", flex_flow="row wrap"),
        )

        metadata_box = VBox(
            [
                self.ui_component.marker_dropdown,
                self.ui_component.tag_entry,
                self.ui_component.tags,
                self.ui_component.comment,
                VBox(
                    [self.ui_component.annotation_summary, self.ui_component.mask_summary],
                    layout=column_block_layout(gap="2px"),
                ),
            ],
            layout=column_block_layout(gap="6px"),
        )

        file_box = VBox(
            [self.ui_component.path, actions_secondary],
            layout=column_block_layout(gap="6px"),
        )

        self.ui_component.editor_root = VBox(
            [
                HBox(
                    [self.ui_component.roi_table, self.ui_component.limit_to_fov_checkbox],
                    layout=Layout(
                        align_items="center",
                        gap="8px",
                        width="100%",
                        flex_flow="row nowrap",
                    ),
                ),
                actions_primary,
                metadata_box,
                file_box,
                self.ui_component.status,
            ],
            layout=column_block_layout(gap="10px"),
        )

    def _build_browser_widgets(self) -> None:
        self.ui_component.browser_tags_filter = TagsInput(
            value=(),
            allowed_tags=[],
            description="Tags:",
            allow_duplicates=False,
            allow_new=False,
            layout=Layout(width="100%"),
        )
        if hasattr(self.ui_component.browser_tags_filter, "restrict_to_allowed_tags"):
            try:
                self.ui_component.browser_tags_filter.restrict_to_allowed_tags = False
            except Exception:  # pragma: no cover
                pass

        self.ui_component.browser_fov_filter = SelectMultiple(
            options=[],
            value=(),
            description="FOVs:",
            layout=Layout(width="100%"),
        )

        self.ui_component.browser_limit_to_current = Checkbox(
            value=False,
            description="Only current FOV",
            indent=False,
            layout=Layout(width="auto"),
        )

        self.ui_component.browser_refresh_button = Button(
            description="Refresh",
            icon="refresh",
            button_style="",
            layout=Layout(width="auto"),
        )

        self.ui_component.browser_output = Output(layout=Layout(min_height="320px"))
        self.ui_component.browser_status = HTML("<em>No ROI captured yet.</em>")

        controls = VBox(
            [
                HBox(
                    [
                        self.ui_component.browser_tags_filter,
                        self.ui_component.browser_fov_filter,
                    ],
                    layout=Layout(gap="10px", flex_flow="row wrap"),
                ),
                HBox(
                    [
                        self.ui_component.browser_limit_to_current,
                        self.ui_component.browser_refresh_button,
                    ],
                    layout=Layout(gap="10px", align_items="center"),
                ),
            ],
            layout=column_block_layout(gap="6px"),
        )

        self.ui_component.browser_root = VBox(
            [controls, self.ui_component.browser_output, self.ui_component.browser_status],
            layout=column_block_layout(gap="8px"),
        )

    def _build_layout(self) -> None:
        header = HTML("<strong>ROI manager</strong>")
        self.ui_component.tabs = Tab(
            children=[self.ui_component.browser_root, self.ui_component.editor_root],
            layout=column_block_layout(gap="10px"),
        )
        self.ui_component.tabs.set_title(0, "ROI browser")
        self.ui_component.tabs.set_title(1, "ROI editor")

        content = VBox(
            [header, self.ui_component.tabs],
            layout=column_block_layout(gap="10px"),
        )

        self.panel = Accordion(
            children=[content],
            titles=("ROI manager",),
            selected_index=None,
            layout=column_block_layout(),
        )
        self.ui = self.ui_component.tabs

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
        self.ui_component.center_with_preset_button.on_click(self._center_with_preset)
        self.ui_component.tags.observe(self._on_tags_value_change, names="value")
        self.ui_component.tag_entry.observe(self._on_tag_entry_change, names="value")

        self.ui_component.browser_tags_filter.observe(
            self._on_browser_filter_change, names="value"
        )
        self.ui_component.browser_fov_filter.observe(
            self._on_browser_filter_change, names="value"
        )
        self.ui_component.browser_limit_to_current.observe(
            self._on_browser_filter_change, names="value"
        )
        self.ui_component.browser_refresh_button.on_click(lambda _btn: self._refresh_browser_gallery())

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

        filter_widget = getattr(self.ui_component, "browser_tags_filter", None)
        if filter_widget is not None:
            current_filter = tuple(tag for tag in getattr(filter_widget, "value", ()) if tag in allowed)
            self._suspend_browser_events = True
            try:
                if list(getattr(filter_widget, "allowed_tags", [])) != allowed:
                    filter_widget.allowed_tags = allowed
                if getattr(filter_widget, "value", ()) != current_filter:
                    filter_widget.value = current_filter
            finally:
                self._suspend_browser_events = False

    def refresh_roi_table(self, df: Optional[pd.DataFrame] = None) -> None:
        source_table = self.main_viewer.roi_manager.table
        self._update_available_tags(source_table)
        self._refresh_browser_filters()

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
        self._refresh_browser_gallery()

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

    def _refresh_browser_filters(self) -> None:
        widget = getattr(self.ui_component, "browser_fov_filter", None)
        if widget is None:
            return

        table = self.main_viewer.roi_manager.table
        if table is None or table.empty or "fov" not in table.columns:
            options: List[str] = []
        else:
            options = sorted({str(value) for value in table["fov"].dropna().astype(str)})

        current_value = tuple(value for value in getattr(widget, "value", ()) if value in options)
        self._suspend_browser_events = True
        try:
            widget.options = options
            if getattr(widget, "value", ()) != current_value:
                widget.value = current_value
        finally:
            self._suspend_browser_events = False

    def _filtered_browser_dataframe(self) -> pd.DataFrame:
        table = self.main_viewer.roi_manager.table
        if table is None or table.empty:
            return pd.DataFrame()

        df = table.copy()

        if self.ui_component.browser_limit_to_current.value:
            current_fov = getattr(self.main_viewer.ui_component.image_selector, "value", None)
            if current_fov:
                df = df[df["fov"] == current_fov]

        selected_fovs = tuple(getattr(self.ui_component.browser_fov_filter, "value", ()))
        if selected_fovs:
            df = df[df["fov"].astype(str).isin(selected_fovs)]

        selected_tags = set(getattr(self.ui_component.browser_tags_filter, "value", ()))
        if selected_tags:
            df = df[df["tags"].apply(lambda raw: selected_tags.issubset({tag.strip() for tag in str(raw).split(',') if tag.strip()}))]

        if "created_at" in df.columns:
            df = df.sort_values("created_at", ascending=False)

        return df.reset_index(drop=True)

    def _refresh_browser_gallery(self) -> None:
        output = getattr(self.ui_component, "browser_output", None)
        status = getattr(self.ui_component, "browser_status", None)
        if output is None or status is None:
            return

        if self._suspend_browser_events:
            return

        df = self._filtered_browser_dataframe()
        records = df.to_dict("records") if not df.empty else []
        total = len(records)
        limited_records = records[: self.BROWSER_MAX_TILES]

        self._disconnect_browser_events()
        self._browser_axis_to_roi.clear()

        with output:
            output.clear_output(wait=True)
            if not limited_records:
                status.value = "<em>No ROI matches current filters.</em>"
                return

            columns = min(self.BROWSER_COLUMNS, max(1, len(limited_records)))
            rows = math.ceil(len(limited_records) / columns)
            fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.0, rows * 3.0))
            axes_array = np.atleast_1d(np.array(axes)).ravel()

            rendered = 0
            for axis in axes_array:
                axis.axis("off")

            for axis, record in zip(axes_array, limited_records):
                marker_profile = self._build_marker_profile(record)
                if marker_profile is None or not marker_profile.selected_channels:
                    axis.text(0.5, 0.5, "No channels", ha="center", va="center", fontsize=9)
                    continue

                downsample = record.get("zoom", self.main_viewer.current_downsample_factor)
                try:
                    downsample_int = max(1, int(round(float(downsample))))
                except Exception:  # pragma: no cover - defensive
                    downsample_int = max(1, int(self.main_viewer.current_downsample_factor))

                tile = self._render_roi_tile(record, marker_profile, downsample_int)
                if tile is None or tile.size == 0:
                    axis.text(0.5, 0.5, "Preview unavailable", ha="center", va="center", fontsize=9)
                    continue

                axis.imshow(tile)
                axis.set_title(self._format_browser_title(record, marker_profile), fontsize=9)
                axis.axis("off")
                self._browser_axis_to_roi[axis] = record.get("roi_id")
                rendered += 1

            for axis in axes_array[len(limited_records) :]:
                axis.remove()

            fig.tight_layout()
            plt.show(fig)

        self._browser_figure = fig
        try:
            self._browser_click_cid = fig.canvas.mpl_connect("button_press_event", self._on_browser_click)
        except Exception:  # pragma: no cover - matplotlib backend quirks
            self._browser_click_cid = None

        summary = f"Displaying {rendered} of {total} ROI(s)."
        if total > len(limited_records):
            summary += " Showing most recent entries."
        status.value = summary

    def _disconnect_browser_events(self) -> None:
        if self._browser_figure is not None and self._browser_click_cid is not None:
            try:
                self._browser_figure.canvas.mpl_disconnect(self._browser_click_cid)
            except Exception:  # pragma: no cover - defensive cleanup
                pass
        self._browser_click_cid = None
        self._browser_figure = None

    def _format_browser_title(self, record: Mapping[str, object], profile: _MarkerProfile) -> str:
        fov = record.get("fov") or "—"
        marker = profile.name or (record.get("marker_set") or "—")
        return f"{fov} · {marker}"

    def _render_roi_tile(
        self,
        record: Mapping[str, object],
        marker_profile: _MarkerProfile,
        downsample_factor: int,
    ) -> Optional[np.ndarray]:
        fov_name = record.get("fov")
        if not fov_name:
            return None

        try:
            self.main_viewer.load_fov(fov_name, marker_profile.selected_channels)
        except Exception:  # pragma: no cover - cache loading failures
            return None

        channel_arrays = self.main_viewer.image_cache.get(fov_name)
        if channel_arrays is None:
            return None

        try:
            array = render_roi_to_array(
                fov_name,
                channel_arrays,
                marker_profile.selected_channels,
                marker_profile.channel_settings,
                roi_definition=record,
                downsample_factor=downsample_factor,
            )
        except Exception:  # pragma: no cover - rendering failures
            return None

        return np.clip(array, 0.0, 1.0)

    def _build_marker_profile(self, record: Mapping[str, object]) -> Optional[_MarkerProfile]:
        marker_ref = str(record.get("marker_set") or "").strip()

        if marker_ref and marker_ref in self.main_viewer.marker_sets:
            data = self.main_viewer.marker_sets.get(marker_ref, {})
            selected = tuple(data.get("selected_channels", ()))
            channel_settings = self._build_channel_settings_from_saved(data, selected)
            return _MarkerProfile(name=marker_ref, selected_channels=selected, channel_settings=channel_settings)

        if marker_ref.startswith("current:"):
            channels = tuple(
                ch.strip() for ch in marker_ref.split(":", 1)[1].split(",") if ch.strip()
            )
            channel_settings = self._snapshot_channel_controls(channels)
            return _MarkerProfile(name="current", selected_channels=channels, channel_settings=channel_settings)

        fallback_channels = self._current_selected_channels()
        channel_settings = self._snapshot_channel_controls(fallback_channels)
        return _MarkerProfile(name="current", selected_channels=fallback_channels, channel_settings=channel_settings)

    def _build_channel_settings_from_saved(
        self,
        payload: Mapping[str, object],
        channels: Sequence[str],
    ) -> Dict[str, ChannelRenderSettings]:
        result: Dict[str, ChannelRenderSettings] = {}
        raw_settings = payload.get("channel_settings", {}) if isinstance(payload, dict) else {}
        for channel in channels:
            entry = raw_settings.get(channel, {}) if isinstance(raw_settings, dict) else {}
            color_key = entry.get("color") if isinstance(entry, dict) else None
            color_value = self.main_viewer.predefined_colors.get(color_key, color_key) if color_key else color_key
            if color_value is None:
                color_value = "#FFFFFF"
            contrast_min = float(entry.get("contrast_min", 0.0)) if isinstance(entry, dict) else 0.0
            contrast_max = float(entry.get("contrast_max", 1.0)) if isinstance(entry, dict) else 1.0
            result[channel] = ChannelRenderSettings(
                color=to_rgb(color_value),
                contrast_min=contrast_min,
                contrast_max=contrast_max,
            )
        return result

    def _current_selected_channels(self) -> Tuple[str, ...]:
        selector = getattr(self.main_viewer.ui_component, "channel_selector", None)
        value = getattr(selector, "value", ())
        if isinstance(value, (list, tuple)):
            return tuple(str(v) for v in value)
        if value:
            return (str(value),)
        return ()

    def _snapshot_channel_controls(self, channels: Sequence[str]) -> Dict[str, ChannelRenderSettings]:
        ui = getattr(self.main_viewer, "ui_component", None)
        if ui is None:
            return {}
        color_controls = getattr(ui, "color_controls", {}) or {}
        contrast_min_controls = getattr(ui, "contrast_min_controls", {}) or {}
        contrast_max_controls = getattr(ui, "contrast_max_controls", {}) or {}
        result: Dict[str, ChannelRenderSettings] = {}
        for channel in channels:
            color_widget = color_controls.get(channel)
            color_key = getattr(color_widget, "value", "#FFFFFF")
            color_value = self.main_viewer.predefined_colors.get(color_key, color_key)
            min_widget = contrast_min_controls.get(channel)
            max_widget = contrast_max_controls.get(channel)
            try:
                contrast_min = float(getattr(min_widget, "value", 0.0))
            except Exception:  # pragma: no cover - widget value errors
                contrast_min = 0.0
            try:
                contrast_max = float(getattr(max_widget, "value", 1.0))
            except Exception:  # pragma: no cover - widget value errors
                contrast_max = 1.0
            result[channel] = ChannelRenderSettings(
                color=to_rgb(color_value),
                contrast_min=contrast_min,
                contrast_max=contrast_max,
            )
        return result

    def _on_browser_filter_change(self, change) -> None:
        if self._suspend_browser_events or change.get("name") != "value":
            return
        self._refresh_browser_gallery()

    def _on_browser_click(self, event) -> None:  # pragma: no cover - UI callback
        axis = getattr(event, "inaxes", None)
        if axis is None:
            return
        roi_id = self._browser_axis_to_roi.get(axis)
        if not roi_id:
            return
        self._activate_roi_from_browser(str(roi_id))

    def _activate_roi_from_browser(self, roi_id: str) -> None:
        record = self.main_viewer.roi_manager.get_roi(roi_id)
        if not record:
            self.set_status("Selected ROI no longer exists.", level="error")
            self.refresh_roi_table()
            return

        self._selected_roi_id = roi_id
        self.main_viewer.center_on_roi(record)
        success, missing = self._apply_roi_presets(record)

        self._suspend_ui_events = True
        try:
            self.ui_component.roi_table.value = roi_id
        finally:
            self._suspend_ui_events = False

        self._populate_fields(record)

        if success:
            self.set_status("Centered on ROI via browser.", level="success")
        else:
            missing_text = ", ".join(missing)
            self.set_status(
                f"Centered on ROI; missing preset(s): {missing_text}.",
                level="warning",
            )

    def _apply_roi_presets(self, record: Mapping[str, object]) -> Tuple[bool, List[str]]:
        missing: List[str] = []

        marker_ref = str(record.get("marker_set") or "").strip()
        if marker_ref:
            if not self._apply_marker_preset(marker_ref):
                missing.append("marker set")

        annotation_name = str(record.get("annotation_palette") or "").strip()
        if annotation_name:
            if not self._apply_annotation_palette(annotation_name):
                missing.append("annotation palette")

        mask_name = str(record.get("mask_color_set") or "").strip()
        if mask_name:
            if not self._apply_mask_color_set(mask_name):
                missing.append("mask color set")

        return not missing, missing

    def _apply_marker_preset(self, marker_ref: str) -> bool:
        marker_ref = marker_ref.strip()
        if not marker_ref:
            return True

        if marker_ref.startswith("current:"):
            channels = tuple(
                ch.strip() for ch in marker_ref.split(":", 1)[1].split(",") if ch.strip()
            )
            selector = getattr(self.main_viewer.ui_component, "channel_selector", None)
            if selector is None:
                return False
            self._suspend_ui_events = True
            try:
                selector.value = channels
            finally:
                self._suspend_ui_events = False
            try:
                self.main_viewer.update_controls(None)
                self.main_viewer.update_display(self.main_viewer.current_downsample_factor)
            except Exception:  # pragma: no cover - safety around UI refresh
                return False
            return True

        applier = getattr(self.main_viewer, "apply_marker_set_by_name", None)
        if callable(applier):
            try:
                return bool(applier(marker_ref))
            except Exception:  # pragma: no cover - downstream errors
                return False
        return False

    def _apply_annotation_palette(self, name: str) -> bool:
        name = name.strip()
        if not name:
            return True
        applier = getattr(self.main_viewer, "apply_annotation_palette_set", None)
        if callable(applier):
            try:
                return bool(applier(name))
            except Exception:  # pragma: no cover - downstream errors
                return False
        return False

    def _apply_mask_color_set(self, name: str) -> bool:
        name = name.strip()
        if not name:
            return True

        mask_plugin = getattr(self.main_viewer, "mask_painter_plugin", None)
        if mask_plugin is None:
            return False

        applier = getattr(mask_plugin, "apply_color_set_by_name", None)
        if callable(applier):
            try:
                return bool(applier(name))
            except Exception:  # pragma: no cover - downstream errors
                return False
        return False

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

    def _get_active_annotation_palette(self) -> str:
        getter = getattr(self.main_viewer, "get_active_annotation_palette_set", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:  # pragma: no cover - downstream errors
                return ""
            if value:
                return str(value)
        return ""

    def _get_active_mask_color_set(self) -> str:
        mask_plugin = getattr(self.main_viewer, "mask_painter_plugin", None)
        if mask_plugin is None:
            return ""
        getter = getattr(mask_plugin, "get_active_color_set_name", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:  # pragma: no cover
                return ""
            if value:
                return str(value)
        return ""

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
            "annotation_palette": self._get_active_annotation_palette(),
            "mask_color_set": self._get_active_mask_color_set(),
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
            "annotation_palette": self._get_active_annotation_palette(),
            "mask_color_set": self._get_active_mask_color_set(),
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

    def _center_with_preset(self, _):
        if not self._selected_roi_id:
            self.set_status("Select an ROI to center on.", level="warning")
            return

        record = self.main_viewer.roi_manager.get_roi(self._selected_roi_id)
        if not record:
            self.set_status("Selected ROI no longer exists.", level="error")
            self.refresh_roi_table()
            return

        self.main_viewer.center_on_roi(record)
        success, missing = self._apply_roi_presets(record)

        if success:
            self.set_status("Centered on ROI with preset.", level="success")
        else:
            missing_text = ", ".join(missing)
            self.set_status(
                f"Centered on ROI; missing preset(s): {missing_text}.",
                level="warning",
            )

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
            self._update_metadata_summaries(
                str(record.get("annotation_palette") or ""),
                str(record.get("mask_color_set") or ""),
            )
        finally:
            self._suspend_ui_events = False

    def _clear_fields(self) -> None:
        self._suspend_ui_events = True
        try:
            self.ui_component.marker_dropdown.value = self.CURRENT_MARKER_VALUE
            self.ui_component.tags.value = ()
            self.ui_component.comment.value = ""
            self._update_metadata_summaries("", "")
        finally:
            self._suspend_ui_events = False

    def _update_metadata_summaries(self, annotation_name: str, mask_name: str) -> None:
        annotation_text = annotation_name or "—"
        mask_text = mask_name or "—"
        self.ui_component.annotation_summary.value = f"<em>Annotation palette: {annotation_text}</em>"
        self.ui_component.mask_summary.value = f"<em>Mask color set: {mask_text}</em>"

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