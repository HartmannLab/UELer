import html
import json
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    ToggleButtons,
    VBox,
)
from matplotlib.colors import to_rgb
from IPython.display import HTML as IPythonHTML, display

from ueler.rendering import ChannelRenderSettings, render_roi_to_array

from .plugin_base import PluginBase
from ..layout_utils import column_block_layout, flex_fill_layout
from ..tag_expression import TagExpressionError, compile_tag_expression


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
    BROWSER_ROWS = 4
    BROWSER_PAGE_SIZE = BROWSER_COLUMNS * BROWSER_ROWS

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
        self._browser_current_page = 1
        self._browser_total_pages = 1
        self._browser_last_signature = None
        self._browser_expression_cache = None
        self._browser_expression_error = None
        self._browser_tag_buttons = {}
        self._browser_expression_selection: Tuple[int, int] = (0, 0)
        self._browser_expression_focused = False
        self._browser_expression_widget_bound = False

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

        self.ui_component.browser_tag_logic = ToggleButtons(
            options=(("All tags (AND)", "and"), ("Any tag (OR)", "or")),
            value="and",
            description="Tag logic",
            layout=Layout(width="auto"),
        )

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

        self.ui_component.browser_use_saved_preset = Checkbox(
            value=True,
            description="Apply saved preset on click",
            indent=False,
            layout=Layout(width="auto"),
        )

        self.ui_component.browser_expression_input = Text(
            value="",
            description="Expression:",
            placeholder="(good & figure1) & !excluded",
            layout=flex_fill_layout(),
            style={"description_width": "auto"},
        )

        operator_buttons: List[Button] = []
        for symbol in ("(", ")", "&", "|", "!"):
            operator_button = Button(
                description=symbol,
                tooltip=f"Insert '{symbol}'",
                layout=Layout(width="32px"),
            )
            operator_button.on_click(
                lambda _btn, token=symbol: self._insert_browser_expression_snippet(token)
            )
            operator_buttons.append(operator_button)
        self.ui_component.browser_expression_operator_box = HBox(
            operator_buttons,
            layout=Layout(gap="4px", flex_flow="row wrap"),
        )

        self.ui_component.browser_expression_tag_box = HBox(
            [],
            layout=Layout(gap="4px", flex_flow="row wrap"),
        )

        self.ui_component.browser_expression_feedback = HTML(
            "<em>Combine tags with () &amp; | !. Leave blank to use the simple tag filter.</em>"
        )

        # Allow the output widget to size to its content. Some frontends may
        # ignore 'height' or 'overflow' props; adjust if needed for your target.
        self.ui_component.browser_output = Output(
            layout=Layout(
                height="auto",
                width="98%",
                align_self="center",
                overflow_y="visible",
                border="1px solid var(--jp-border-color2, #ccc)",
            )
        )

        self.ui_component.browser_status = HTML("<em>No ROI captured yet.</em>")
        self.ui_component.browser_page_label = HTML("<em>Page 1 of 1</em>")
        self.ui_component.browser_prev_button = Button(
            description="Previous page",
            icon="angle-left",
            button_style="",
            layout=Layout(width="auto"),
        )
        self.ui_component.browser_next_button = Button(
            description="Next page",
            icon="angle-right",
            button_style="",
            layout=Layout(width="auto"),
        )
        self.ui_component.browser_pagination = HBox(
            [
                self.ui_component.browser_prev_button,
                self.ui_component.browser_page_label,
                self.ui_component.browser_next_button,
            ],
            layout=Layout(gap="8px", justify_content="center", align_items="center", flex_flow="row wrap"),
        )

        controls = VBox(
            [
                HBox(
                    [
                        self.ui_component.browser_tags_filter,
                        self.ui_component.browser_tag_logic,
                        self.ui_component.browser_fov_filter,
                    ],
                    layout=Layout(gap="10px", flex_flow="row wrap"),
                ),
                HBox(
                    [
                        self.ui_component.browser_limit_to_current,
                        self.ui_component.browser_use_saved_preset,
                        self.ui_component.browser_refresh_button,
                    ],
                    layout=Layout(gap="10px", align_items="center"),
                ),
                VBox(
                    [
                        self.ui_component.browser_expression_input,
                        self.ui_component.browser_expression_operator_box,
                        self.ui_component.browser_expression_tag_box,
                        self.ui_component.browser_expression_feedback,
                    ],
                    layout=column_block_layout(gap="4px"),
                ),
            ],
            layout=column_block_layout(gap="6px"),
        )

        self.ui_component.browser_root = VBox(
            [
                controls,
                self.ui_component.browser_output,
                self.ui_component.browser_pagination,
                self.ui_component.browser_status,
            ],
            layout=column_block_layout(gap="8px"),
        )
        self._ensure_expression_cursor_binding()

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
        self.ui_component.browser_tag_logic.observe(
            self._on_browser_filter_change, names="value"
        )
        self.ui_component.browser_fov_filter.observe(
            self._on_browser_filter_change, names="value"
        )
        self.ui_component.browser_limit_to_current.observe(
            self._on_browser_filter_change, names="value"
        )
        self.ui_component.browser_expression_input.observe(
            self._on_browser_expression_change, names="value"
        )
        self.ui_component.browser_refresh_button.on_click(self._on_browser_refresh_clicked)
        self.ui_component.browser_prev_button.on_click(self._on_browser_prev_clicked)
        self.ui_component.browser_next_button.on_click(self._on_browser_next_clicked)

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

        self._refresh_expression_tag_buttons(allowed)

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
        self._browser_current_page = 1
        self._browser_last_signature = None
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

        expression_widget = getattr(self.ui_component, "browser_expression_input", None)
        expression_text = str(getattr(expression_widget, "value", "") or "").strip()
        predicate = None
        if expression_text:
            predicate = self._compile_browser_expression(expression_text)
            if predicate is None:
                return pd.DataFrame()

        if predicate is not None:

            def _expression_match(raw_value: object) -> bool:
                row_tags = {tag.strip() for tag in str(raw_value).split(',') if tag.strip()}
                return predicate(row_tags)

            df = df[df["tags"].apply(_expression_match)]
        else:
            selected_tags = set(getattr(self.ui_component.browser_tags_filter, "value", ()))
            if selected_tags:
                tag_logic = getattr(self.ui_component.browser_tag_logic, "value", "and")

                def _tags_match(raw_value: object) -> bool:
                    row_tags = {tag.strip() for tag in str(raw_value).split(',') if tag.strip()}
                    if not row_tags:
                        return False
                    if tag_logic == "or":
                        return any(tag in row_tags for tag in selected_tags)
                    return selected_tags.issubset(row_tags)

                df = df[df["tags"].apply(_tags_match)]

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

        if total == 0:
            self._disconnect_browser_events()
            self._browser_axis_to_roi.clear()
            self._browser_last_signature = ("empty",)
            self._browser_total_pages = 0
            with output:
                output.clear_output(wait=True)
            self._update_browser_summary(0, 0)
            self._update_pagination_controls(0, 0, 0)
            return

        page_size = max(1, int(self.BROWSER_PAGE_SIZE))
        total_pages = max(1, math.ceil(total / page_size))
        self._browser_total_pages = total_pages
        current_page = max(1, min(self._browser_current_page, total_pages))
        self._browser_current_page = current_page

        start_index = (current_page - 1) * page_size
        end_index = min(start_index + page_size, total)
        limited_records = records[start_index:end_index]

        expression_text = str(getattr(self.ui_component.browser_expression_input, "value", "") or "").strip()

        signature = (
            tuple(record.get("roi_id") for record in limited_records),
            tuple(sorted(str(tag) for tag in getattr(self.ui_component.browser_tags_filter, "value", ()))),
            tuple(sorted(str(fov) for fov in getattr(self.ui_component.browser_fov_filter, "value", ()))),
            bool(getattr(self.ui_component.browser_limit_to_current, "value", False)),
            getattr(self.ui_component.browser_tag_logic, "value", "and"),
            expression_text,
            self._browser_expression_error,
            current_page,
            total_pages,
            total,
        )

        if self._browser_last_signature == signature:
            rendered_count = len(limited_records)
            self._update_browser_summary(rendered_count, total)
            self._update_pagination_controls(current_page, total_pages, total)
            return

        self._disconnect_browser_events()
        self._browser_axis_to_roi.clear()

        with output:
            output.clear_output(wait=True)
            if limited_records:
                columns = min(self.BROWSER_COLUMNS, max(1, len(limited_records)))
                rows = max(1, math.ceil(len(limited_records) / columns))
            else:
                columns = self.BROWSER_COLUMNS
                rows = max(1, self.BROWSER_ROWS)

            tile_inch = 2.4
            fig, axes = plt.subplots(rows, columns, figsize=(columns * tile_inch, rows * tile_inch))
            fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02, hspace=0.02)
            axes_array = np.atleast_1d(np.array(axes)).ravel()

            rendered = 0
            for axis in axes_array:
                axis.axis("off")

            for axis, record in zip(axes_array, limited_records):
                marker_profile = self._build_marker_profile(record)
                message_text = None
                if marker_profile is None or not marker_profile.selected_channels:
                    message_text = "No channels"

                downsample = record.get("zoom", self.main_viewer.current_downsample_factor)
                try:
                    downsample_int = max(1, int(round(float(downsample))))
                except Exception:  # pragma: no cover - defensive
                    downsample_int = max(1, int(self.main_viewer.current_downsample_factor))

                tile = None if message_text else self._render_roi_tile(record, marker_profile, downsample_int)
                if tile is None or tile.size == 0:
                    message_text = message_text or "Preview unavailable"

                if message_text:
                    axis.text(0.5, 0.5, message_text, ha="center", va="center", fontsize=9)
                else:
                    axis.imshow(tile)

                axis.axis("off")
                self._browser_axis_to_roi[axis] = record.get("roi_id")
                rendered += 1

            for axis in axes_array[len(limited_records) :]:
                axis.remove()

            plt.show(fig)

            try:
                canvas = getattr(fig, "canvas", None)
                if canvas is not None:
                    if hasattr(canvas, "toolbar_visible"):
                        canvas.toolbar_visible = False
                    if hasattr(canvas, "header_visible"):
                        canvas.header_visible = False
                    if hasattr(canvas, "footer_visible"):
                        canvas.footer_visible = False
            except Exception:  # pragma: no cover - backend differences
                pass

        self._browser_figure = fig
        try:
            self._browser_click_cid = fig.canvas.mpl_connect("button_press_event", self._on_browser_click)
        except Exception:  # pragma: no cover - matplotlib backend quirks
            self._browser_click_cid = None

        self._browser_last_signature = signature
        rendered_count = rendered
        self._update_browser_summary(rendered_count, total)
        self._update_pagination_controls(current_page, total_pages, total)

    def _disconnect_browser_events(self) -> None:
        if self._browser_figure is not None and self._browser_click_cid is not None:
            try:
                self._browser_figure.canvas.mpl_disconnect(self._browser_click_cid)
            except Exception:  # pragma: no cover - defensive cleanup
                pass
        self._browser_click_cid = None
        self._browser_figure = None

    def _update_browser_summary(self, rendered: int, total: int) -> None:
        status = getattr(self.ui_component, "browser_status", None)
        if status is None:
            return
        if total <= 0 or rendered <= 0:
            status.value = "<em>No ROI matches current filters.</em>"
            return

        current_page = max(1, int(getattr(self, "_browser_current_page", 1)))
        total_pages = max(1, int(getattr(self, "_browser_total_pages", 1)))
        status.value = (
            f"Displaying {rendered} ROI(s) — page {current_page} of {total_pages} (total {total})."
        )

    def _update_pagination_controls(self, current_page: int, total_pages: int, total: int) -> None:
        prev_button = getattr(self.ui_component, "browser_prev_button", None)
        next_button = getattr(self.ui_component, "browser_next_button", None)
        page_label = getattr(self.ui_component, "browser_page_label", None)

        if page_label is not None:
            if total_pages <= 0:
                page_label.value = "<em>No pages</em>"
            else:
                page_label.value = (
                    f"<em>Page {max(1, current_page)} of {max(1, total_pages)} ({total} total)</em>"
                )

        if prev_button is not None:
            prev_button.disabled = current_page <= 1 or total_pages <= 0

        if next_button is not None:
            next_button.disabled = current_page >= total_pages or total_pages <= 0

    def _insert_browser_expression_snippet(self, snippet: str) -> None:
        self._ensure_expression_cursor_binding()
        widget = getattr(self.ui_component, "browser_expression_input", None)
        if widget is None:
            return

        current = str(getattr(widget, "value", "") or "")
        start, end = self._browser_expression_selection
        text_length = len(current)
        if not isinstance(start, int) or start < 0 or start > text_length:
            start = text_length
        if not isinstance(end, int) or end < start or end > text_length:
            end = start

        before = current[:start]
        after = current[end:]
        insert, cursor_offset = self._format_expression_insertion(before, after, snippet)

        new_value = before + insert + after

        self._suspend_browser_events = True
        try:
            widget.value = new_value
        finally:
            self._suspend_browser_events = False

        new_cursor = start + cursor_offset
        self._browser_expression_selection = (new_cursor, new_cursor)
        self._send_browser_expression_cursor(new_cursor, new_cursor, focus=True)
        self._on_browser_expression_change({"name": "value", "new": widget.value})

    def _format_expression_insertion(self, before: str, after: str, snippet: str) -> Tuple[str, int]:
        before_char = before[-1] if before else ""
        after_char = after[0] if after else ""
        boundary_left = {"", " ", "	", "(", "&", "|", "!"}
        boundary_right = {"", " ", "	", ")", "&", "|"}

        needs_leading = bool(before) and before_char not in boundary_left

        if snippet == "!":
            insert = (" " if needs_leading else "") + "!"
            return insert, len(insert)

        if snippet == ")":
            insert = ")"
            if needs_leading and before_char not in {" ", "("}:
                insert = " " + insert
            return insert, len(insert)

        leading = " " if needs_leading else ""
        after_boundary = after_char in boundary_right

        if snippet == "(":
            needs_trailing = not after_boundary and after_char != ")"
        else:
            needs_trailing = not after_boundary

        trailing = " " if needs_trailing else ""
        insert = f"{leading}{snippet}{trailing}"
        cursor_offset = len(insert) if trailing else len(leading + snippet)
        return insert, cursor_offset

    def _send_browser_expression_cursor(self, start: int, end: int, focus: bool = False) -> None:
        widget = getattr(self.ui_component, "browser_expression_input", None)
        if widget is None:
            return
        payload = {
            "event": "set-selection",
            "start": max(0, int(start)),
            "end": max(0, int(end)),
        }
        if focus:
            payload["focus"] = True
        try:
            widget.send(payload)
        except Exception:  # pragma: no cover - widget may not support send
            pass

    def _refresh_expression_tag_buttons(self, tags: Sequence[str]) -> None:
        container = getattr(self.ui_component, "browser_expression_tag_box", None)
        if container is None:
            return

        if list(self._browser_tag_buttons.keys()) == list(tags):
            return

        self._browser_tag_buttons = {}
        buttons: List[Button] = []
        for tag in tags:
            button = Button(
                description=tag,
                tooltip=f"Insert '{tag}'",
                layout=Layout(width="auto"),
            )
            button.on_click(
                lambda _btn, token=tag: self._insert_browser_expression_snippet(token)
            )
            self._browser_tag_buttons[tag] = button
            buttons.append(button)

        container.children = tuple(buttons)

    def _on_browser_expression_change(self, change) -> None:
        if self._suspend_browser_events or change.get("name") != "value":
            return

        expression = str(change.get("new") or "")
        self._browser_expression_cache = None
        self._compile_browser_expression(expression)
        self._browser_current_page = 1
        self._browser_last_signature = None
        self._refresh_browser_gallery()

    def _compile_browser_expression(self, expression: str) -> Optional[Callable[[Iterable[str]], bool]]:
        text = expression.strip()
        if not text:
            self._browser_expression_cache = None
            self._set_browser_expression_error(None, expression="")
            return None

        if self._browser_expression_cache and self._browser_expression_cache[0] == text:
            self._set_browser_expression_error(None, expression=text)
            return self._browser_expression_cache[1]

        try:
            predicate = compile_tag_expression(text)
        except TagExpressionError as exc:
            self._browser_expression_cache = None
            self._set_browser_expression_error(str(exc), expression=text)
            return None

        self._browser_expression_cache = (text, predicate)
        self._set_browser_expression_error(None, expression=text)
        return predicate

    def _set_browser_expression_error(self, message: Optional[str], expression: str) -> None:
        self._browser_expression_error = message
        feedback = getattr(self.ui_component, "browser_expression_feedback", None)
        if feedback is None:
            return

        if message:
            escaped = html.escape(message)
            color = self.STATUS_COLORS.get("error", "#c62828")
            feedback.value = f"<span style='color:{color}'>Expression error: {escaped}</span>"
            return

        if expression.strip():
            feedback.value = "<em>Expression active.</em>"
        else:
            feedback.value = "<em>Combine tags with () &amp; | !. Leave blank to use the simple tag filter.</em>"

    def _expression_cursor_script(self, model_id: str) -> str:
        if not model_id:
            return ""

        encoded_id = json.dumps(model_id)
        return f"""
<script type="text/javascript">
(function() {{
    const widgetId = {encoded_id};
    const selectors = [`[data-widget-id="${{widgetId}}"]`, `[data-widgetid="${{widgetId}}"]`];
    let attempts = 0;
    let cachedModel = null;

    function resolveModel(callback) {{
        if (cachedModel) {{
            callback(cachedModel);
            return;
        }}

        const sendWithManager = manager => {{
            if (!manager || typeof manager.get_model !== "function") {{
                return;
            }}
            try {{
                manager.get_model(widgetId).then(model => {{
                    if (!model) {{
                        return;
                    }}
                    cachedModel = model;
                    callback(model);
                }}).catch(() => {{}});
            }} catch (err) {{
                console.debug('UELer ROI expression manager lookup failed', err);
            }}
        }};

        const candidates = [];
        if (window.__widget_manager__) {{
            candidates.push(window.__widget_manager__);
        }}
        if (window.jupyterWidgetManagers && window.jupyterWidgetManagers.length) {{
            candidates.push(window.jupyterWidgetManagers[0]);
        }}
        if (window.manager && typeof window.manager.get_model === "function") {{
            candidates.push(window.manager);
        }}

        let handled = false;
        for (const candidate of candidates) {{
            if (!candidate) {{
                continue;
            }}
            handled = true;
            sendWithManager(candidate);
        }}

        if (!handled && typeof window.require === "function") {{
            window.require(['@jupyter-widgets/base'], function(base) {{
                const managers = (base.ManagerBase && base.ManagerBase._managers) || [];
                if (managers.length) {{
                    sendWithManager(managers[0]);
                }}
            }});
        }}
    }}

    function sendSelection(input) {{
        resolveModel(model => {{
            if (!model) {{
                return;
            }}
            const payload = {{
                event: 'selection-change',
                start: input.selectionStart == null ? input.value.length : input.selectionStart,
                end: input.selectionEnd == null ? input.value.length : input.selectionEnd,
                focused: document.activeElement === input
            }};
            try {{
                model.send(payload);
            }} catch (err) {{
                console.debug('UELer ROI expression selection dispatch failed', err);
            }}
        }});
    }}

    function locateHost() {{
        for (const selector of selectors) {{
            const element = document.querySelector(selector);
            if (element) {{
                return element;
            }}
        }}
        return null;
    }}

    function attach() {{
        const host = locateHost();
        if (!host) {{
            if (attempts++ < 40) {{
                requestAnimationFrame(attach);
            }}
            return;
        }}

        const input = host.querySelector('input');
        if (!input) {{
            if (attempts++ < 40) {{
                requestAnimationFrame(attach);
            }}
            return;
        }}

        if (input.__ueler_expression_listener_attached) {{
            return;
        }}
        input.__ueler_expression_listener_attached = true;

        const handler = () => sendSelection(input);
        for (const eventName of ['keyup', 'mouseup', 'select', 'focus', 'blur', 'input']) {{
            input.addEventListener(eventName, handler);
        }}

        resolveModel(model => {{
            if (!model) {{
                return;
            }}
            if (model.__ueler_expression_listener_bound) {{
                return;
            }}
            model.__ueler_expression_listener_bound = true;
            model.on('msg:custom', msg => {{
                if (!msg || typeof msg !== 'object') {{
                    return;
                }}
                if (msg.event === 'set-selection') {{
                    const start = Number.isFinite(msg.start) ? msg.start : (input.selectionStart || 0);
                    const end = Number.isFinite(msg.end) ? msg.end : start;
                    try {{
                        if (msg.focus) {{
                            input.focus();
                        }}
                        if (typeof input.setSelectionRange === 'function') {{
                            input.setSelectionRange(start, end);
                        }}
                    }} catch (err) {{
                        console.debug('UELer ROI expression selection apply failed', err);
                    }}
                }}
            }});
        }});

        handler();
    }}

    attach();
}})();
</script>
"""

    def _on_browser_expression_msg(self, _, content, __):  # pragma: no cover - UI callback
        if not isinstance(content, dict):
            return
        if content.get("event") != "selection-change":
            return

        start = content.get("start")
        end = content.get("end")
        if not isinstance(start, (int, float)):
            start = 0
        if not isinstance(end, (int, float)):
            end = start

        start = max(0, int(start))
        end = max(start, int(end))
        self._browser_expression_selection = (start, end)
        self._browser_expression_focused = bool(content.get("focused"))

    def _ensure_expression_cursor_binding(self) -> None:
        if getattr(self, "_browser_expression_widget_bound", False):
            return

        widget = getattr(self.ui_component, "browser_expression_input", None)
        if widget is None:
            return

        model_id = getattr(widget, "model_id", None) or getattr(widget, "_model_id", None)
        if not model_id:
            return

        script = self._expression_cursor_script(model_id)
        if script:
            try:
                display(IPythonHTML(script))
            except Exception:  # pragma: no cover - display may fail in headless mode
                pass

        try:
            widget.on_msg(self._on_browser_expression_msg)
        except Exception:  # pragma: no cover - widget may not support custom messages
            return

        value = str(getattr(widget, "value", "") or "")
        length = len(value)
        self._browser_expression_selection = (length, length)
        self._browser_expression_widget_bound = True

        if hasattr(widget, "value"):
            self._send_browser_expression_cursor(length, length, focus=False)

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
        self._browser_current_page = 1
        self._browser_last_signature = None
        self._refresh_browser_gallery()

    def _on_browser_refresh_clicked(self, _button) -> None:
        self._browser_current_page = 1
        self._browser_last_signature = None
        self._refresh_browser_gallery()

    def _on_browser_prev_clicked(self, _button) -> None:
        if self._browser_current_page <= 1:
            return
        self._browser_current_page = max(1, self._browser_current_page - 1)
        self._browser_last_signature = None
        self._refresh_browser_gallery()

    def _on_browser_next_clicked(self, _button) -> None:
        total_pages = max(1, int(getattr(self, "_browser_total_pages", 1)))
        if self._browser_current_page >= total_pages:
            return
        self._browser_current_page = min(total_pages, self._browser_current_page + 1)
        self._browser_last_signature = None
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
        use_saved_toggle = getattr(self.ui_component, "browser_use_saved_preset", None)
        apply_presets = bool(getattr(use_saved_toggle, "value", True)) if use_saved_toggle is not None else True
        success = True
        missing: List[str] = []
        if apply_presets:
            success, missing = self._apply_roi_presets(record)

        self._suspend_ui_events = True
        try:
            self.ui_component.roi_table.value = roi_id
        finally:
            self._suspend_ui_events = False

        self._populate_fields(record)

        if not apply_presets:
            self.set_status("Centered on ROI without applying the saved preset.", level="info")
        elif success:
            self.set_status("Centered on ROI via browser.", level="success")
        else:
            missing_text = ", ".join(missing) if missing else "unknown"
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

        mask_visibility_payload = str(record.get("mask_visibility") or "").strip()
        if mask_visibility_payload:
            if not self._apply_mask_visibility(mask_visibility_payload):
                missing.append("mask visibility")

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

    def _get_mask_visibility_payload(self) -> str:
        getter = getattr(self.main_viewer, "get_mask_visibility_state", None)
        if not callable(getter):
            return ""
        try:
            state = getter()
        except Exception:  # pragma: no cover - defensive
            return ""
        if not state:
            return ""
        try:
            return json.dumps(state, sort_keys=True)
        except Exception:  # pragma: no cover - serialization errors
            return ""

    def _apply_mask_visibility(self, payload: str) -> bool:
        if not payload:
            return True
        applier = getattr(self.main_viewer, "apply_mask_visibility_state", None)
        if not callable(applier):
            return False
        try:
            state = json.loads(payload)
        except Exception:  # pragma: no cover - invalid payload
            return False
        if not isinstance(state, dict):
            return False
        try:
            return bool(applier(state))
        except Exception:  # pragma: no cover - downstream errors
            return False

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
            "mask_visibility": self._get_mask_visibility_payload(),
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
            "mask_visibility": self._get_mask_visibility_payload(),
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
                str(record.get("mask_visibility") or ""),
            )
        finally:
            self._suspend_ui_events = False

    def _clear_fields(self) -> None:
        self._suspend_ui_events = True
        try:
            self.ui_component.marker_dropdown.value = self.CURRENT_MARKER_VALUE
            self.ui_component.tags.value = ()
            self.ui_component.comment.value = ""
            self._update_metadata_summaries("", "", "")
        finally:
            self._suspend_ui_events = False

    def _update_metadata_summaries(self, annotation_name: str, mask_name: str, mask_visibility: str) -> None:
        annotation_text = annotation_name or "—"
        mask_text = mask_name or "—"
        if mask_visibility:
            try:
                state = json.loads(mask_visibility)
                if isinstance(state, dict) and state:
                    total_masks = len(state)
                    enabled = sum(1 for value in state.values() if bool(value))
                    mask_text += f" (visible: {enabled}/{total_masks})"
                else:
                    mask_text += " (state saved)"
            except Exception:  # pragma: no cover - defensive decoding
                mask_text += " (state saved)"
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