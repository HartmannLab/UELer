"""Interactive Batch Export plugin for the viewer UI."""

from __future__ import annotations

import asyncio
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from IPython import get_ipython
from IPython.display import display
from ipywidgets import (
    Button,
    Checkbox,
    Dropdown,
    FloatSlider,
    HBox,
    HTML,
    IntProgress,
    IntText,
    Layout,
    Output,
    SelectMultiple,
    Tab,
    Text,
    ToggleButtons,
    VBox,
)
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb

from ..rendering import ChannelRenderSettings, render_crop_to_array, render_fov_to_array, render_roi_to_array
from .plugin_base import PluginBase

PLACEHOLDER_MESSAGE = "Batch export UI is now available."


@dataclass(frozen=True)
class _MarkerProfile:
    name: str
    selected_channels: Sequence[str]
    channel_settings: Mapping[str, ChannelRenderSettings]


class BatchExportPlugin(PluginBase):
    """Batch export plugin providing UI, job management, and progress feedback."""

    MODE_FULL_FOV = "full_fov"
    MODE_SINGLE_CELLS = "single_cells"
    MODE_ROIS = "rois"

    _MODE_LABELS = {
        MODE_FULL_FOV: "Full FOV",
        MODE_SINGLE_CELLS: "Single Cells",
        MODE_ROIS: "ROIs",
    }

    _FILE_FORMATS = ("png", "jpg", "tif", "tiff", "pdf")
    _CELL_OPTION_LIMIT = 500

    def __init__(self, main_viewer, width: int, height: int) -> None:
        super().__init__(main_viewer, width, height)
        self.SidePlots_id = "batch_export_output"
        self.displayed_name = "Batch export"
        self.main_viewer = main_viewer
        self.ui_component = SimpleNamespace()

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._current_job: Optional[Any] = None
        self._current_future: Optional[Future] = None
        self._event_loop = self._capture_event_loop()
        self._io_loop = self._capture_io_loop()

        self._cell_records: Dict[Any, Dict[str, Any]] = {}
        self._cell_filter_snapshot = ""
        self._roi_records: Dict[str, Dict[str, Any]] = {}
        self._seen_results: set[str] = set()

        self._build_widgets()
        self._build_layout()
        self._connect_events()

        self.refresh_marker_options()
        self.refresh_fov_options()
        self.refresh_cell_options()
        self.refresh_roi_options()

        self.setup_widget_observers()
        self.initialized = True

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        layout_full = Layout(width="100%")
        style_auto = {"description_width": "auto"}

        self.ui_component.mode_selector = ToggleButtons(
            options=[(label, key) for key, label in self._MODE_LABELS.items()],
            value=self.MODE_FULL_FOV,
            description="Mode:",
            layout=Layout(width="100%"),
            style=style_auto,
            button_style="",
        )

        self.ui_component.marker_set_dropdown = Dropdown(
            options=[],
            description="Marker set:",
            layout=layout_full,
            style=style_auto,
        )

        default_output = os.path.join(self.main_viewer.base_folder, "exports")
        self.ui_component.output_path = Text(
            value=default_output,
            description="Output folder:",
            layout=layout_full,
            style=style_auto,
            placeholder="/path/to/exports",
        )
        self.ui_component.browse_button = Button(
            description="Browse",
            icon="folder-open",
            layout=Layout(width="110px"),
        )

        self.ui_component.file_format_dropdown = Dropdown(
            options=[(fmt.upper(), fmt) for fmt in self._FILE_FORMATS],
            value="png",
            description="Format:",
            layout=layout_full,
            style=style_auto,
        )

        self.ui_component.downsample_input = IntText(
            value=max(1, int(getattr(self.main_viewer, "current_downsample_factor", 1))),
            description="Downsample:",
            layout=Layout(width="180px"),
            style=style_auto,
        )

        self.ui_component.dpi_input = IntText(
            value=300,
            description="DPI:",
            layout=Layout(width="160px"),
            style=style_auto,
        )

        self.ui_component.include_scale_bar = Checkbox(
            value=False,
            description="Include scale bar",
            indent=False,
            layout=Layout(width="auto"),
        )
        self.ui_component.scale_bar_ratio = FloatSlider(
            value=10.0,
            min=1.0,
            max=10.0,
            step=0.5,
            description="Scale bar % width:",
            layout=Layout(width="100%"),
            style=style_auto,
        )

        self.ui_component.start_button = Button(
            description="Start",
            icon="play",
            button_style="success",
            layout=Layout(width="120px"),
        )
        self.ui_component.cancel_button = Button(
            description="Cancel",
            icon="stop",
            button_style="warning",
            layout=Layout(width="120px"),
            disabled=True,
        )

        self.ui_component.progress_bar = IntProgress(
            value=0,
            min=0,
            max=1,
            description="Progress",
            bar_style="info",
            layout=Layout(width="100%"),
        )
        self.ui_component.progress_summary = HTML(value="")
        self.ui_component.output_link = HTML(value="")
        self.ui_component.status_message = HTML(value="")
        self.ui_component.log_output = Output(layout=Layout(width="100%", border="1px solid #ddd"))

        # Mode specific containers -------------------------------------------------
        self._build_full_fov_widgets(layout_full, style_auto)
        self._build_single_cell_widgets(layout_full, style_auto)
        self._build_roi_widgets(layout_full, style_auto)

        self.ui_component.mode_tabs = Tab(
            children=[
                self.ui_component.full_fov_box,
                self.ui_component.single_cell_box,
                self.ui_component.roi_box,
            ],
            layout=Layout(width="100%"),
        )
        for idx, mode in enumerate((self.MODE_FULL_FOV, self.MODE_SINGLE_CELLS, self.MODE_ROIS)):
            self.ui_component.mode_tabs.set_title(idx, self._MODE_LABELS[mode])

    def _build_full_fov_widgets(self, layout_full: Layout, style_auto: Mapping[str, str]) -> None:
        selector = SelectMultiple(
            options=[],
            description="FOVs:",
            layout=layout_full,
            style=style_auto,
        )
        self.ui_component.full_fov_selector = selector

        self.ui_component.full_fov_use_all = Checkbox(
            value=True,
            description="Export all FOVs",
            indent=False,
            layout=Layout(width="auto"),
        )

        figure_width = IntText(
            value=0,
            description="Figure width (px):",
            layout=Layout(width="220px"),
            style=style_auto,
        )
        figure_height = IntText(
            value=0,
            description="Figure height (px):",
            layout=Layout(width="220px"),
            style=style_auto,
        )
        self.ui_component.figure_width = figure_width
        self.ui_component.figure_height = figure_height

        self.ui_component.full_fov_box = VBox(
            [
                self.ui_component.full_fov_use_all,
                selector,
                HBox([figure_width, figure_height], layout=Layout(gap="12px")),
            ],
            layout=Layout(gap="6px", width="100%"),
        )

    def _build_single_cell_widgets(self, layout_full: Layout, style_auto: Mapping[str, str]) -> None:
        self.ui_component.cell_filter = Text(
            value="",
            description="Filter (query):",
            placeholder="marker > 0",
            layout=layout_full,
            style=style_auto,
        )
        self.ui_component.cell_apply_filter = Button(
            description="Apply",
            icon="filter",
            layout=Layout(width="110px"),
        )
        self.ui_component.cell_selection = SelectMultiple(
            options=[],
            description="Cells:",
            layout=layout_full,
            style=style_auto,
        )
        self.ui_component.cell_crop_size = IntText(
            value=128,
            description="Crop size (px):",
            layout=Layout(width="220px"),
            style=style_auto,
        )
        self.ui_component.cell_preview_button = Button(
            description="Preview",
            icon="eye",
            layout=Layout(width="120px"),
        )
        self.ui_component.cell_preview_output = Output(layout=Layout(width="100%", border="1px solid #ddd", padding="6px"))

        filter_row = HBox(
            [self.ui_component.cell_filter, self.ui_component.cell_apply_filter],
            layout=Layout(gap="10px", align_items="center"),
        )

        self.ui_component.single_cell_box = VBox(
            [
                filter_row,
                self.ui_component.cell_selection,
                HBox(
                    [self.ui_component.cell_crop_size, self.ui_component.cell_preview_button],
                    layout=Layout(gap="12px"),
                ),
                self.ui_component.cell_preview_output,
            ],
            layout=Layout(gap="8px", width="100%"),
        )

    def _build_roi_widgets(self, layout_full: Layout, style_auto: Mapping[str, str]) -> None:
        self.ui_component.roi_limit_to_fov = Checkbox(
            value=False,
            description="Current FOV only",
            indent=False,
            layout=Layout(width="auto"),
        )
        self.ui_component.roi_selection = SelectMultiple(
            options=[],
            description="ROIs:",
            layout=layout_full,
            style=style_auto,
        )

        self.ui_component.roi_box = VBox(
            [self.ui_component.roi_limit_to_fov, self.ui_component.roi_selection],
            layout=Layout(gap="6px", width="100%"),
        )

    def _build_layout(self) -> None:
        header = HTML("<strong>Batch export</strong>")

        output_row = HBox(
            [self.ui_component.output_path, self.ui_component.browse_button],
            layout=Layout(gap="10px", align_items="center"),
        )

        controls = VBox(
            [
                header,
                self.ui_component.mode_selector,
                self.ui_component.marker_set_dropdown,
                output_row,
                self.ui_component.file_format_dropdown,
                HBox(
                    [self.ui_component.downsample_input, self.ui_component.dpi_input],
                    layout=Layout(gap="12px"),
                ),
                self.ui_component.include_scale_bar,
                self.ui_component.scale_bar_ratio,
                self.ui_component.mode_tabs,
                HBox(
                    [self.ui_component.start_button, self.ui_component.cancel_button],
                    layout=Layout(gap="10px"),
                ),
                self.ui_component.progress_bar,
                self.ui_component.progress_summary,
                self.ui_component.output_link,
                self.ui_component.status_message,
                self.ui_component.log_output,
            ],
            layout=Layout(gap="10px", width="100%"),
        )

        self.ui = controls

    def _connect_events(self) -> None:
        self.ui_component.mode_selector.observe(self._on_mode_change, names="value")
        self.ui_component.full_fov_use_all.observe(
            lambda change: self._toggle_fov_selector(change.get("new", False)), names="value"
        )
        self.ui_component.browse_button.on_click(lambda _: self._launch_file_chooser())
        self.ui_component.start_button.on_click(self._start_export)
        self.ui_component.cancel_button.on_click(self._cancel_export)
        self.ui_component.cell_apply_filter.on_click(lambda _: self.refresh_cell_options())
        self.ui_component.cell_preview_button.on_click(lambda _: self._preview_single_cell())
        self.ui_component.roi_limit_to_fov.observe(lambda _: self.refresh_roi_options(), names="value")

    # ------------------------------------------------------------------
    # UI refresh helpers
    # ------------------------------------------------------------------
    def refresh_marker_options(self) -> None:
        marker_sets = sorted(self.main_viewer.marker_sets.keys())
        if not marker_sets:
            self.ui_component.marker_set_dropdown.options = [("—", None)]
            self.ui_component.marker_set_dropdown.value = None
            return
        options = [(name, name) for name in marker_sets]
        current = self.ui_component.marker_set_dropdown.value
        if current not in marker_sets:
            current = marker_sets[0]
        self.ui_component.marker_set_dropdown.options = options
        self.ui_component.marker_set_dropdown.value = current

    def refresh_fov_options(self) -> None:
        fovs = sorted(getattr(self.main_viewer, "available_fovs", ()))
        options = [(fov, fov) for fov in fovs]
        self.ui_component.full_fov_selector.options = options
        if options:
            self.ui_component.full_fov_selector.value = tuple(value for _, value in options[: min(3, len(options))])
        self._toggle_fov_selector(self.ui_component.full_fov_use_all.value)

    def refresh_cell_options(self) -> None:
        df = getattr(self.main_viewer, "cell_table", None)
        self._cell_records.clear()
        filter_text = self.ui_component.cell_filter.value.strip()
        self._cell_filter_snapshot = filter_text

        if df is None or df.empty:
            self.ui_component.cell_selection.options = []
            self.ui_component.cell_preview_output.clear_output()
            self._notify("Cell table not loaded.", level="warning")
            return

        filtered = df
        if filter_text:
            try:
                filtered = df.query(filter_text, engine="python")
            except Exception as exc:  # pragma: no cover - guard for malformed queries
                self._notify(f"Filter error: {exc}", level="error")
                filtered = df.iloc[0:0]

        if filtered.empty:
            self.ui_component.cell_selection.options = []
            self._notify("No cells match the filter.", level="info")
            return

        fov_key = getattr(self.main_viewer, "fov_key", "fov")
        label_key = getattr(self.main_viewer, "label_key", "label")
        options = []
        for idx, row in filtered.head(self._CELL_OPTION_LIMIT).iterrows():
            record = row.to_dict()
            record["_index"] = idx
            self._cell_records[idx] = record
            fov_name = str(record.get(fov_key, "—"))
            label = record.get(label_key)
            if label is None or (isinstance(label, float) and np.isnan(label)):
                label = idx
            label = str(label)
            display = f"{fov_name} · {label}"
            options.append((display, idx))

        self.ui_component.cell_selection.options = options
        if options:
            self.ui_component.cell_selection.value = (options[0][1],)

    def refresh_roi_options(self) -> None:
        manager = getattr(self.main_viewer, "roi_manager", None)
        self._roi_records.clear()
        if manager is None:
            self.ui_component.roi_selection.options = []
            return

        df = manager.list_rois()
        if df.empty:
            self.ui_component.roi_selection.options = []
            return

        if self.ui_component.roi_limit_to_fov.value:
            current_fov = getattr(self.main_viewer.ui_component.image_selector, "value", None)
            if current_fov:
                df = df[df["fov"] == current_fov]

        options = []
        for _, row in df.iterrows():
            record = row.to_dict()
            roi_id = str(record.get("roi_id"))
            label = f"{record.get('fov', '—')} · {record.get('marker_set', '—')} · {roi_id[:8]}"
            options.append((label, roi_id))
            self._roi_records[roi_id] = record

        self.ui_component.roi_selection.options = options
        selected = tuple(value for _, value in options[: min(3, len(options))])
        self.ui_component.roi_selection.value = selected

    # ------------------------------------------------------------------
    # Trait / event handlers
    # ------------------------------------------------------------------
    def _on_mode_change(self, change) -> None:
        new_mode = change.get("new")
        if new_mode not in self._MODE_LABELS:
            return
        index = list(self._MODE_LABELS).index(new_mode)
        self.ui_component.mode_tabs.selected_index = index

    def _toggle_fov_selector(self, use_all: bool) -> None:
        self.ui_component.full_fov_selector.disabled = use_all

    # Plugin lifecycle hooks ------------------------------------------------------
    def on_fov_change(self) -> None:  # type: ignore[override]
        self.refresh_roi_options()

    def on_cell_table_change(self) -> None:  # type: ignore[override]
        self.refresh_cell_options()

    def on_mv_update_display(self) -> None:  # type: ignore[override]
        if self.ui_component.roi_limit_to_fov.value:
            self.refresh_roi_options()

    def after_all_plugins_loaded(self) -> None:  # type: ignore[override]
        super().after_all_plugins_loaded()
        self.refresh_marker_options()
        self.refresh_fov_options()
        self.refresh_cell_options()
        self.refresh_roi_options()

    # ------------------------------------------------------------------
    # Export orchestration
    # ------------------------------------------------------------------
    def _start_export(self, _button) -> None:
        if self._current_future and not self._current_future.done():
            self._notify("A job is already running.", level="warning")
            return

        try:
            marker_profile = self._resolve_marker_profile()
            mode = self.ui_component.mode_selector.value
            output_dir = self._normalise_output_dir(self.ui_component.output_path.value)
            file_format = self.ui_component.file_format_dropdown.value
            downsample = max(1, int(self.ui_component.downsample_input.value or 1))
            dpi = max(1, int(self.ui_component.dpi_input.value or 1))
            scale_bar = self.ui_component.include_scale_bar.value
            scale_ratio = float(self.ui_component.scale_bar_ratio.value)

            job = self._build_job(
                mode=mode,
                marker_profile=marker_profile,
                output_dir=output_dir,
                file_format=file_format,
                downsample=downsample,
                dpi=dpi,
                include_scale_bar=scale_bar,
                scale_ratio=scale_ratio,
            )
        except Exception as exc:
            self._notify(f"Unable to start export: {exc}", level="error")
            return

        if job is None:
            self._notify("Nothing to export for the selected mode.", level="warning")
            return

        self._current_job = job
        self._seen_results.clear()
        self._update_progress_ui(reset=True)
        self._notify("Job started.", level="info")
        self._set_running_state(True)

        future = self._executor.submit(self._run_job, job)
        self._current_future = future

    def _cancel_export(self, _button=None) -> None:
        job = self._current_job
        if job is None:
            return
        job.cancel()
        self._notify("Cancellation requested…", level="warning")

    def _run_job(self, job) -> None:
        try:
            job.start()
        except Exception as exc:  # pragma: no cover - defensive guard
            self._dispatch_to_main(lambda: self._notify(f"Job failed: {exc}", level="error"))
        finally:
            status = job.status()
            self._dispatch_to_main(lambda: self._finalise_job(status))

    def _finalise_job(self, status) -> None:
        self._set_running_state(False)
        success = status.failed == 0 and not status.cancelled
        summary = (
            f"Completed {status.completed}/{status.total} (success={status.succeeded}, fail={status.failed})."
        )
        if success:
            level = "success"
        elif status.failed == 0:
            level = "warning"
        else:
            level = "error"
        self._notify(summary, level=level)
        self._update_progress_ui(status=status)

        output_dir = getattr(self._current_job, "output_dir", None)
        if output_dir:
            path = Path(output_dir)
            try:
                link = path.as_uri()
            except ValueError:
                link = str(path)
            self.ui_component.output_link.value = f'<a href="{link}" target="_blank">Open output folder</a>'

        self._current_job = None
        self._current_future = None

    def _build_job(
        self,
        *,
        mode: str,
        marker_profile: _MarkerProfile,
        output_dir: str,
        file_format: str,
        downsample: int,
        dpi: int,
        include_scale_bar: bool,
        scale_ratio: float,
    ):
        overrides = {
            "downsample_factor": downsample,
            "dpi": dpi,
            "include_scale_bar": include_scale_bar,
            "scale_bar_ratio": scale_ratio,
            "filter": self._cell_filter_snapshot,
        }

        builder = {
            self.MODE_FULL_FOV: self._build_full_fov_items,
            self.MODE_SINGLE_CELLS: self._build_single_cell_items,
            self.MODE_ROIS: self._build_roi_items,
        }.get(mode)

        if builder is None:
            raise ValueError(f"Unknown export mode: {mode}")

        items = builder(
            marker_profile=marker_profile,
            output_dir=output_dir,
            file_format=file_format,
            downsample=downsample,
            dpi=dpi,
            include_scale_bar=include_scale_bar,
            scale_ratio=scale_ratio,
        )

        from ueler.export.job import Job  # Local import to avoid circular dependency at module import time

        if not items:
            return None

        job = Job(
            mode=mode,
            items=items,
            marker_set=marker_profile.name,
            output_dir=output_dir,
            file_format=file_format,
            overrides=overrides,
            progress_callback=self._progress_callback,
        )
        return job

    # ------------------------------------------------------------------
    # Job item builders per mode
    # ------------------------------------------------------------------
    def _build_full_fov_items(
        self,
        *,
        marker_profile: _MarkerProfile,
        output_dir: str,
        file_format: str,
        downsample: int,
        dpi: int,
        include_scale_bar: bool,
        scale_ratio: float,
    ):
        from ueler.export.job import JobItem

        if self.ui_component.full_fov_use_all.value:
            fovs = list(getattr(self.main_viewer, "available_fovs", ()))
        else:
            fovs = list(self.ui_component.full_fov_selector.value)

        items = []
        for fov in fovs:
            if not fov:
                continue

            output_path = os.path.join(output_dir, f"{self._safe_filename(fov)}.{file_format}")

            worker = partial(
                self._export_fov_worker,
                fov_name=fov,
                marker_profile=marker_profile,
                downsample=downsample,
                file_format=file_format,
                output_path=output_path,
                dpi=dpi,
                include_scale_bar=include_scale_bar,
                scale_ratio=scale_ratio,
            )

            metadata = {
                "fov": fov,
                "mode": self.MODE_FULL_FOV,
                "marker_set": marker_profile.name,
            }

            items.append(
                JobItem(
                    item_id=fov,
                    execute=worker,
                    output_path=output_path,
                    metadata=metadata,
                )
            )

        return items

    def _build_single_cell_items(
        self,
        *,
        marker_profile: _MarkerProfile,
        output_dir: str,
        file_format: str,
        downsample: int,
        dpi: int,
        include_scale_bar: bool,
        scale_ratio: float,
    ):
        from ueler.export.job import JobItem

        selected = tuple(self.ui_component.cell_selection.value)
        records = []
        if selected:
            records = [self._cell_records[idx] for idx in selected if idx in self._cell_records]
        else:
            records = list(self._cell_records.values())

        if not records:
            return []

        crop_size = max(16, int(self.ui_component.cell_crop_size.value or 128))
        fov_key = getattr(self.main_viewer, "fov_key", "fov")
        label_key = getattr(self.main_viewer, "label_key", "label")

        items = []
        for record in records:
            fov = record.get(fov_key)
            if not fov:
                continue
            label = record.get(label_key, record.get("_index"))
            if label is None or (isinstance(label, float) and np.isnan(label)):
                label = record.get("_index")
            label_str = self._safe_filename(str(label))
            output_path = os.path.join(
                output_dir,
                f"{self._safe_filename(str(fov))}_cell_{label_str}.{file_format}",
            )

            worker = partial(
                self._export_cell_worker,
                record=record,
                marker_profile=marker_profile,
                crop_size=crop_size,
                downsample=downsample,
                file_format=file_format,
                output_path=output_path,
                dpi=dpi,
                include_scale_bar=include_scale_bar,
                scale_ratio=scale_ratio,
            )

            metadata = {
                "fov": fov,
                "cell_label": label,
                "mode": self.MODE_SINGLE_CELLS,
                "marker_set": marker_profile.name,
            }

            item_id = f"{fov}::cell::{label_str}"
            items.append(
                JobItem(
                    item_id=item_id,
                    execute=worker,
                    output_path=output_path,
                    metadata=metadata,
                )
            )

        return items

    def _build_roi_items(
        self,
        *,
        marker_profile: _MarkerProfile,
        output_dir: str,
        file_format: str,
        downsample: int,
        dpi: int,
        include_scale_bar: bool,
        scale_ratio: float,
    ):
        from ueler.export.job import JobItem

        selected = tuple(self.ui_component.roi_selection.value)
        if not selected:
            return []

        items = []
        for roi_id in selected:
            record = self._roi_records.get(roi_id)
            if not record:
                continue
            fov = record.get("fov")
            if not fov:
                continue
            output_path = os.path.join(
                output_dir,
                f"{self._safe_filename(str(fov))}_roi_{self._safe_filename(roi_id[:12])}.{file_format}",
            )

            worker = partial(
                self._export_roi_worker,
                roi=record,
                marker_profile=marker_profile,
                downsample=downsample,
                file_format=file_format,
                output_path=output_path,
                dpi=dpi,
                include_scale_bar=include_scale_bar,
                scale_ratio=scale_ratio,
            )

            metadata = {
                "fov": fov,
                "roi_id": roi_id,
                "mode": self.MODE_ROIS,
                "marker_set": marker_profile.name,
            }

            items.append(
                JobItem(
                    item_id=f"roi::{roi_id}",
                    execute=worker,
                    output_path=output_path,
                    metadata=metadata,
                )
            )

        return items

    # ------------------------------------------------------------------
    # Worker implementations
    # ------------------------------------------------------------------
    def _export_fov_worker(
        self,
        *,
        fov_name: str,
        marker_profile: _MarkerProfile,
        downsample: int,
        file_format: str,
        output_path: str,
        dpi: int,
        include_scale_bar: bool,
        scale_ratio: float,
    ) -> Dict[str, Any]:
        self.main_viewer.load_fov(fov_name, marker_profile.selected_channels)
        channel_arrays = self.main_viewer.image_cache[fov_name]
        array = render_fov_to_array(
            fov_name,
            channel_arrays,
            marker_profile.selected_channels,
            marker_profile.channel_settings,
            downsample_factor=downsample,
        )
        image = self._finalise_array(array, include_scale_bar, scale_ratio)
        self._write_image(image, output_path, file_format, dpi)
        return {"output_path": output_path}

    def _export_cell_worker(
        self,
        *,
        record: Mapping[str, Any],
        marker_profile: _MarkerProfile,
        crop_size: int,
        downsample: int,
        file_format: str,
        output_path: str,
        dpi: int,
        include_scale_bar: bool,
        scale_ratio: float,
    ) -> Dict[str, Any]:
        fov_key = getattr(self.main_viewer, "fov_key", "fov")
        x_key = getattr(self.main_viewer, "x_key", "X")
        y_key = getattr(self.main_viewer, "y_key", "Y")
        fov_name = record.get(fov_key)
        center_xy = (float(record.get(x_key)), float(record.get(y_key)))

        self.main_viewer.load_fov(fov_name, marker_profile.selected_channels)
        channel_arrays = self.main_viewer.image_cache[fov_name]
        array = render_crop_to_array(
            fov_name,
            channel_arrays,
            marker_profile.selected_channels,
            marker_profile.channel_settings,
            center_xy=center_xy,
            size_px=crop_size,
            downsample_factor=downsample,
        )
        image = self._finalise_array(array, include_scale_bar, scale_ratio)
        self._write_image(image, output_path, file_format, dpi)
        return {"output_path": output_path}

    def _export_roi_worker(
        self,
        *,
        roi: Mapping[str, Any],
        marker_profile: _MarkerProfile,
        downsample: int,
        file_format: str,
        output_path: str,
        dpi: int,
        include_scale_bar: bool,
        scale_ratio: float,
    ) -> Dict[str, Any]:
        fov_name = roi.get("fov")
        self.main_viewer.load_fov(fov_name, marker_profile.selected_channels)
        channel_arrays = self.main_viewer.image_cache[fov_name]
        array = render_roi_to_array(
            fov_name,
            channel_arrays,
            marker_profile.selected_channels,
            marker_profile.channel_settings,
            roi_definition=roi,
            downsample_factor=downsample,
        )
        image = self._finalise_array(array, include_scale_bar, scale_ratio)
        self._write_image(image, output_path, file_format, dpi)
        return {"output_path": output_path}

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _resolve_marker_profile(self) -> _MarkerProfile:
        marker_name = self.ui_component.marker_set_dropdown.value
        if not marker_name:
            raise ValueError("Select a marker set before starting an export.")
        data = self.main_viewer.marker_sets.get(marker_name)
        if data is None:
            raise ValueError(f"Marker set '{marker_name}' not found.")

        selected = tuple(data.get("selected_channels", ()))
        raw_settings = data.get("channel_settings", {})
        settings = {}
        for channel in selected:
            entry = raw_settings.get(channel)
            if not entry:
                raise ValueError(f"Marker set '{marker_name}' missing settings for channel '{channel}'.")
            color_key = entry.get("color")
            color_value = self.main_viewer.predefined_colors.get(color_key, color_key)
            settings[channel] = ChannelRenderSettings(
                color=to_rgb(color_value),
                contrast_min=float(entry.get("contrast_min", 0.0)),
                contrast_max=float(entry.get("contrast_max", 1.0)),
            )
        return _MarkerProfile(name=marker_name, selected_channels=selected, channel_settings=settings)

    def _normalise_output_dir(self, value: str) -> str:
        path = value.strip() or os.path.join(self.main_viewer.base_folder, "exports")
        if not os.path.isabs(path):
            path = os.path.join(self.main_viewer.base_folder, path)
        os.makedirs(path, exist_ok=True)
        return path

    def _finalise_array(self, array: np.ndarray, include_scale_bar: bool, scale_ratio: float) -> np.ndarray:
        image = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
        if include_scale_bar:
            # Placeholder for Phase 4 scale bar integration; metadata already propagated.
            pass
        return (image * 255).astype(np.uint8)

    def _write_image(self, array: np.ndarray, output_path: str, file_format: str, dpi: int) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fmt = file_format.lower()
        if fmt == "pdf":
            height, width = array.shape[:2]
            fig_w = max(width / dpi, 1.0)
            fig_h = max(height / dpi, 1.0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.axis("off")
            ax.imshow(array)
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        else:
            saver = self._resolve_imsave()
            saver(output_path, array)

    @staticmethod
    def _resolve_imsave():
        try:
            from skimage.io import imsave as _imsave  # type: ignore
        except Exception:  # pragma: no cover - fallback for optional dependency
            try:
                from imageio import imwrite as _imsave  # type: ignore
            except Exception as exc:  # pragma: no cover - dependency missing
                raise RuntimeError(
                    "Neither skimage.io.imsave nor imageio.imwrite is available for writing images"
                ) from exc
        return _imsave

    def _progress_callback(self, status) -> None:
        self._dispatch_to_main(lambda: self._update_progress_ui(status=status))

    def _update_progress_ui(self, status=None, reset: bool = False) -> None:
        bar = self.ui_component.progress_bar
        if reset or status is None:
            bar.max = 1
            bar.value = 0
            bar.bar_style = "info"
            self.ui_component.progress_summary.value = ""
            self.ui_component.status_message.value = ""
            self.ui_component.log_output.clear_output()
            return

        bar.max = max(1, status.total)
        bar.value = status.completed
        if status.cancelled:
            bar.bar_style = "warning"
        elif status.failed:
            bar.bar_style = "danger"
        elif status.completed == status.total:
            bar.bar_style = "success"
        else:
            bar.bar_style = "info"

        self.ui_component.progress_summary.value = (
            f"Items: {status.completed}/{status.total} | Success: {status.succeeded} | Failures: {status.failed}"
        )

        for item_id, result in status.results.items():
            if item_id in self._seen_results:
                continue
            self._seen_results.add(item_id)
            message = f"✔ {item_id}: {result.output_path}" if result.ok else f"✖ {item_id}: {result.error}"
            self._log(message)

    def _preview_single_cell(self) -> None:
        selection = tuple(self.ui_component.cell_selection.value)
        if not selection:
            self._notify("Select a cell to preview.", level="warning")
            return
        record = self._cell_records.get(selection[0])
        if not record:
            self._notify("Selected cell not found.", level="error")
            return

        try:
            marker_profile = self._resolve_marker_profile()
            crop_size = max(16, int(self.ui_component.cell_crop_size.value or 128))
            downsample = max(1, int(self.ui_component.downsample_input.value or 1))
            include_scale_bar = self.ui_component.include_scale_bar.value
            scale_ratio = float(self.ui_component.scale_bar_ratio.value)

            fov_key = getattr(self.main_viewer, "fov_key", "fov")
            x_key = getattr(self.main_viewer, "x_key", "X")
            y_key = getattr(self.main_viewer, "y_key", "Y")
            fov_value = record.get(fov_key)
            if not fov_value:
                raise ValueError("Selected cell row does not include an FOV.")

            self.main_viewer.load_fov(fov_value, marker_profile.selected_channels)
            channel_arrays = self.main_viewer.image_cache[fov_value]
            array = render_crop_to_array(
                fov_value,
                channel_arrays,
                marker_profile.selected_channels,
                marker_profile.channel_settings,
                center_xy=(float(record.get(x_key)), float(record.get(y_key))),
                size_px=crop_size,
                downsample_factor=downsample,
            )
            image = self._finalise_array(array, include_scale_bar, scale_ratio)
        except Exception as exc:
            self._notify(f"Preview failed: {exc}", level="error")
            return

        dpi = int(self.ui_component.dpi_input.value or 300)
        height, width = image.shape[:2]
        fig_w = max(width / dpi, 1.5)
        fig_h = max(height / dpi, 1.5)

        self.ui_component.cell_preview_output.clear_output()
        with self.ui_component.cell_preview_output:
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.axis("off")
            ax.imshow(image)
            display(fig)
            plt.close(fig)

    def _notify(self, message: str, level: str = "info") -> None:
        color_map = {"info": "#424242", "success": "#2e7d32", "warning": "#f9a825", "error": "#c62828"}
        color = color_map.get(level, color_map["info"])
        self.ui_component.status_message.value = f"<span style='color:{color}'>{message}</span>"
        self._log(message)

    def _log(self, message: str) -> None:
        with self.ui_component.log_output:
            print(message)

    def _set_running_state(self, running: bool) -> None:
        self.ui_component.start_button.disabled = running
        self.ui_component.cancel_button.disabled = not running
        self.ui_component.mode_selector.disabled = running
        self.ui_component.marker_set_dropdown.disabled = running
        self.ui_component.output_path.disabled = running
        self.ui_component.file_format_dropdown.disabled = running

    @staticmethod
    def _safe_filename(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", value)

    def _launch_file_chooser(self) -> None:
        try:
            from ipyfilechooser import FileChooser  # type: ignore
        except Exception:
            self._notify("ipyfilechooser not installed; enter the path manually.", level="warning")
            return

        chooser = FileChooser(os.path.dirname(self.ui_component.output_path.value))
        chooser.show_only_dirs = True

        def _on_select(_):
            if chooser.selected_path:
                self.ui_component.output_path.value = chooser.selected_path
            container.children = tuple(child for child in container.children if child is not chooser)

        chooser.register_callback(_on_select)

        container = self.ui
        if chooser not in container.children:
            container.children = container.children + (chooser,)

    # ------------------------------------------------------------------
    # Thread/event helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _capture_event_loop():
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return None
        return loop if loop.is_running() else None

    @staticmethod
    def _capture_io_loop():
        shell = get_ipython()
        kernel = getattr(shell, "kernel", None) if shell else None
        return getattr(kernel, "io_loop", None)

    def _dispatch_to_main(self, callback) -> None:
        loop = self._event_loop
        if loop is not None:
            loop.call_soon_threadsafe(callback)
            return
        io_loop = self._io_loop
        if io_loop is not None:
            io_loop.add_callback(callback)
            return
        callback()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def __del__(self):  # pragma: no cover - interpreter shutdown timing varies
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


RunFlowsom = BatchExportPlugin  # Backwards compatibility


__all__ = ["BatchExportPlugin", "RunFlowsom", "PLACEHOLDER_MESSAGE"]
