"""Mixin layers for the heatmap plugin."""

from __future__ import annotations

import html
import logging
import os
import pickle
import inspect
import sys
from typing import Any, Iterable, List, Sequence, Set

_logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.cluster.hierarchy import cut_tree, linkage
try:
    from IPython.display import display
except Exception:  # pragma: no cover - optional in non-notebook contexts
    def display(*_args, **_kwargs):
        return None

from ueler.viewer.decorators import update_status_bar
from ipywidgets import HBox, HTML, Layout, Output, Tab, VBox


NO_CLUSTER_SELECTED_MSG = "No cluster selected."
UNASSIGNED_META_CLUSTER_ID = -1
UNASSIGNED_META_CLUSTER_NAME = "Unassigned"
UNASSIGNED_META_CLUSTER_COLOR = "#9e9e9e"


def _remove_patches(patches):
    if not patches:
        return []
    cleaned = []
    for patch in patches:
        if patch is None:
            continue
        try:
            patch.remove()
        except Exception:
            continue
    return cleaned


def _method_accepts_alpha(method) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):  # pragma: no cover - extension functions
        return True
    for param in signature.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            return True
    return "alpha" in signature.parameters


def _draw_orientation_span(adapter, axis, start, end, *, color, alpha, zorder):
    if axis is None:
        return None

    method_name = "axvspan" if adapter.is_wide() else "axhspan"
    method = getattr(axis, method_name, None)
    if not callable(method):
        return None

    kwargs = {"facecolor": color, "zorder": zorder}
    if _method_accepts_alpha(method):
        kwargs["alpha"] = alpha
    try:
        return method(start, end, **kwargs)
    except TypeError:
        kwargs.pop("alpha", None)
        return method(start, end, **kwargs)


def _update_cutoff_line(adapter, dend_axis, cut_value, current_line):
    if dend_axis is None:
        return current_line
    if current_line is not None and current_line.axes is not None:
        current_line.remove()
    if adapter.is_wide():
        return dend_axis.axhline(y=cut_value, color="red", linestyle="--")
    return dend_axis.axvline(x=cut_value, color="red", linestyle="--")


def _append_hist_axis(adapter, divider, ax_heatmap):
    if adapter.is_wide():
        return divider.append_axes("bottom", size="15%", pad=0.4, sharex=ax_heatmap)
    return divider.append_axes("right", size="15%", pad=0.4, sharey=ax_heatmap)


def _tick_centers(count):
    if count <= 0:
        return np.array([])
    return np.arange(count, dtype=float) + 0.5


def _apply_heatmap_tick_labels(adapter, ax_heatmap, cluster_leaves, marker_leaves, cluster_label):
    if adapter.is_wide():
        positions = _tick_centers(len(cluster_leaves))
        ax_heatmap.set_xticks(positions)
        ax_heatmap.set_xticklabels(cluster_leaves, rotation=45, ha="right", fontsize="small")
        if marker_leaves:
            ax_heatmap.set_yticks(_tick_centers(len(marker_leaves)))
            ax_heatmap.set_yticklabels(marker_leaves, fontsize="small")
        ax_heatmap.set_xlabel(cluster_label)
        return

    positions = np.arange(len(cluster_leaves))
    ax_heatmap.set_yticks(positions)
    ax_heatmap.set_yticklabels(list(reversed(cluster_leaves)), fontsize="small")
    ax_heatmap.set_ylabel("")
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha="right")


def _render_histogram(adapter, ax_hist, ax_heatmap, hist_series, cluster_leaves, marker_leaves, cluster_label):
    horizontal = adapter.is_wide()
    heatmap_ticks = ax_heatmap.get_xticks() if horizontal else ax_heatmap.get_yticks()
    if len(heatmap_ticks) != len(hist_series):
        heatmap_ticks = _tick_centers(len(hist_series)) if horizontal else np.arange(len(hist_series))

    if horizontal:
        display_values = hist_series.values
        ax_hist.bar(heatmap_ticks, display_values, width=0.7, color="purple", alpha=0.6)
        ax_hist.set_xlim(ax_heatmap.get_xlim())
        ax_hist.set_xticks(heatmap_ticks)
        ax_hist.set_xticklabels(cluster_leaves, rotation=45, ha="right", fontsize="small")
        ax_hist.set_yticks([])
        ax_hist.set_ylabel("Cell count")
    else:
        display_values = list(reversed(hist_series.values))
        ax_hist.barh(heatmap_ticks, display_values, height=0.7, color="purple", alpha=0.6)
        ax_hist.set_ylim(ax_heatmap.get_ylim())
        ax_hist.set_yticks([])
        ax_hist.set_xticks([])

    _apply_heatmap_tick_labels(adapter, ax_heatmap, cluster_leaves, marker_leaves, cluster_label)


class DataLayer:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cluster_assignment_cache = {}

    def _cache_cluster_assignments(self):
        data = getattr(self, "heatmap_data", None)
        cache = {}
        if data is not None and hasattr(data, "columns") and 'meta_cluster_revised' in data.columns:
            revised = data['meta_cluster_revised'].dropna()
            if not revised.empty:
                cache = revised.to_dict()
        self._cluster_assignment_cache = cache

    def _restore_cluster_assignments(self):
        cache = getattr(self, "_cluster_assignment_cache", None)
        if not cache:
            return
        data = getattr(self, "heatmap_data", None)
        if data is None or not hasattr(data, "index"):
            return
        if 'meta_cluster_revised' not in data.columns:
            if 'meta_cluster' in data.columns:
                data['meta_cluster_revised'] = data['meta_cluster']
            else:
                data['meta_cluster_revised'] = np.nan
        for label, value in cache.items():
            if label in data.index:
                data.at[label, 'meta_cluster_revised'] = value

    def _engage_cutoff_lock(self, reason):
        self._cutoff_lock_reason = reason
        self._lock_override_requested = False
        self.ui_component.lock_override_button.disabled = False
        if not self.ui_component.lock_cutoff_button.value:
            self.ui_component.lock_cutoff_button.value = True
        else:
            _logger.warning("%s. Use 'Unlock once' before editing the dendrogram.", reason)

    def _reset_selection_cache(self):
        self._last_scatter_selection = None
        self._last_highlighted_clusters = None

    @staticmethod
    def _normalize_meta_cluster_id(meta_cluster_id):
        if isinstance(meta_cluster_id, np.generic):
            return meta_cluster_id.item()
        return meta_cluster_id

    @staticmethod
    def _sort_meta_cluster_key(meta_cluster_id):
        value = DataLayer._normalize_meta_cluster_id(meta_cluster_id)
        if value == UNASSIGNED_META_CLUSTER_ID:
            return (0, 0)
        if isinstance(value, (int, np.integer)):
            return (1, int(value))
        return (2, str(value))

    @staticmethod
    def _as_hex_color(color_value):
        if color_value is None:
            return UNASSIGNED_META_CLUSTER_COLOR
        try:
            return mcolors.to_hex(color_value)
        except Exception:
            return str(color_value)

    def _meta_cluster_display_name(self, meta_cluster_id):
        names = getattr(self.data, 'meta_cluster_names', {}) or {}
        normalized = self._normalize_meta_cluster_id(meta_cluster_id)
        if normalized in names:
            return str(names[normalized])
        if normalized == UNASSIGNED_META_CLUSTER_ID:
            return UNASSIGNED_META_CLUSTER_NAME
        return f"Meta-cluster {normalized}"

    def _next_available_meta_cluster_id(self):
        names = getattr(self.data, 'meta_cluster_names', {}) or {}
        used = {
            int(key)
            for key in names
            if isinstance(key, (int, np.integer)) and int(key) >= 0
        }
        candidate = int(getattr(self.data, 'next_meta_cluster_id', 0) or 0)
        while candidate in used:
            candidate += 1
        self.data.next_meta_cluster_id = candidate + 1
        return candidate

    def _generate_meta_cluster_color(self, meta_cluster_id):
        if meta_cluster_id == UNASSIGNED_META_CLUSTER_ID:
            return UNASSIGNED_META_CLUSTER_COLOR
        palette_size = max(8, int(meta_cluster_id) + 2) if isinstance(meta_cluster_id, int) else 8
        palette = sns.color_palette('husl', palette_size)
        if not palette:
            palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            ]
        color = palette[int(meta_cluster_id) % len(palette)] if isinstance(meta_cluster_id, int) else palette[0]
        return self._as_hex_color(color)

    def _meta_cluster_dropdown_options(self):
        names = getattr(self.data, 'meta_cluster_names', {}) or {}
        colors = getattr(self.data, 'meta_cluster_colors', {}) or {}
        options = []
        for meta_cluster_id in sorted(names.keys(), key=self._sort_meta_cluster_key):
            color_hex = self._as_hex_color(colors.get(meta_cluster_id))
            display_name = self._meta_cluster_display_name(meta_cluster_id)
            label = f"[{color_hex}] {display_name} ({meta_cluster_id})"
            options.append((label, meta_cluster_id))
        return options

    def _refresh_meta_cluster_registry_preview(self):
        container = getattr(self.ui_component, 'meta_cluster_registry_box', None)
        if container is None:
            return

        names = getattr(self.data, 'meta_cluster_names', {}) or {}
        colors = getattr(self.data, 'meta_cluster_colors', {}) or {}
        rows = []
        for meta_cluster_id in sorted(names.keys(), key=self._sort_meta_cluster_key):
            color_hex = self._as_hex_color(colors.get(meta_cluster_id))
            display_name = html.escape(self._meta_cluster_display_name(meta_cluster_id))
            label_html = HTML(value=f"<span style='font-family:monospace;'>[{color_hex}]</span> {display_name} ({meta_cluster_id})")
            swatch_html = HTML(
                value=(
                    "<span style='display:inline-block;width:12px;height:12px;"
                    f"border:1px solid #555;background:{color_hex};margin-right:6px;'></span>"
                )
            )
            rows.append(HBox([swatch_html, label_html], layout=Layout(align_items='center')))
        container.children = tuple(rows)

    def _refresh_meta_cluster_controls(self):
        if not hasattr(self, 'ui_component'):
            return

        options = self._meta_cluster_dropdown_options()

        assign_dropdown = getattr(self.ui_component, 'cluster_id_dropdown', None)
        rename_dropdown = getattr(self.ui_component, 'rename_cluster_dropdown', None)

        for dropdown in (assign_dropdown, rename_dropdown):
            if dropdown is None:
                continue
            previous_value = getattr(dropdown, 'value', None)
            dropdown.options = options
            available_values = [value for _, value in options]
            if previous_value in available_values:
                dropdown.value = previous_value
            elif available_values:
                preferred = UNASSIGNED_META_CLUSTER_ID if UNASSIGNED_META_CLUSTER_ID in available_values else available_values[0]
                dropdown.value = preferred
            else:
                dropdown.value = None

        self._refresh_meta_cluster_registry_preview()
        if rename_dropdown is not None:
            self.on_rename_cluster_selection_change({'new': rename_dropdown.value})

    def _initialize_meta_cluster_registry(self):
        if not hasattr(self.data, 'meta_cluster_names') or self.data.meta_cluster_names is None:
            self.data.meta_cluster_names = {}
        if not hasattr(self.data, 'meta_cluster_colors') or self.data.meta_cluster_colors is None:
            self.data.meta_cluster_colors = {}

        self.data.meta_cluster_names[UNASSIGNED_META_CLUSTER_ID] = UNASSIGNED_META_CLUSTER_NAME
        self.data.meta_cluster_colors.setdefault(UNASSIGNED_META_CLUSTER_ID, UNASSIGNED_META_CLUSTER_COLOR)
        self.data.next_meta_cluster_id = int(getattr(self.data, 'next_meta_cluster_id', 0) or 0)
        self._refresh_meta_cluster_controls()

    def _ensure_meta_cluster_entry(self, meta_cluster_id):
        normalized = self._normalize_meta_cluster_id(meta_cluster_id)
        if normalized == UNASSIGNED_META_CLUSTER_ID:
            self.data.meta_cluster_names[normalized] = UNASSIGNED_META_CLUSTER_NAME
            self.data.meta_cluster_colors.setdefault(normalized, UNASSIGNED_META_CLUSTER_COLOR)
            return

        self.data.meta_cluster_names.setdefault(normalized, f"Meta-cluster {normalized}")
        if normalized not in self.data.meta_cluster_colors:
            self.data.meta_cluster_colors[normalized] = self._generate_meta_cluster_color(normalized)

    def _sync_meta_cluster_registry(self, meta_cluster_ids, palette_by_id=None):
        self._initialize_meta_cluster_registry()

        if palette_by_id is None:
            palette_by_id = {}

        incoming_ids = [] if meta_cluster_ids is None else list(meta_cluster_ids)
        seen_ids = [self._normalize_meta_cluster_id(value) for value in incoming_ids]

        if hasattr(self, 'heatmap_data') and self.heatmap_data is not None:
            color_column = 'meta_cluster_revised' if 'meta_cluster_revised' in self.heatmap_data.columns else 'meta_cluster'
            if color_column in self.heatmap_data.columns:
                revised_values = self.heatmap_data[color_column].dropna().tolist()
                seen_ids.extend(self._normalize_meta_cluster_id(value) for value in revised_values)

        for meta_cluster_id in seen_ids:
            self._ensure_meta_cluster_entry(meta_cluster_id)

            if meta_cluster_id in palette_by_id:
                self.data.meta_cluster_colors[meta_cluster_id] = self._as_hex_color(palette_by_id[meta_cluster_id])

        int_ids = [
            int(meta_cluster_id)
            for meta_cluster_id in self.data.meta_cluster_names
            if isinstance(meta_cluster_id, (int, np.integer)) and int(meta_cluster_id) >= 0
        ]
        next_id = (max(int_ids) + 1) if int_ids else 0
        self.data.next_meta_cluster_id = max(int(getattr(self.data, 'next_meta_cluster_id', 0) or 0), next_id)
        self._refresh_meta_cluster_controls()

    def _ensure_meta_cluster_revised_column(self):
        if 'meta_cluster_revised' not in self.heatmap_data.columns:
            self.heatmap_data['meta_cluster_revised'] = self.heatmap_data['meta_cluster']

    def _update_orientation_state(self):
        if not hasattr(self, "heatmap_data") or self.heatmap_data is None:
            return
        horizontal = self.adapter.is_wide()
        heatmap_view = self.heatmap_data.T if horizontal else self.heatmap_data
        cluster_axis = 1 if horizontal else 0
        marker_axis = 0 if horizontal else 1
        cluster_index = heatmap_view.index if cluster_axis == 0 else heatmap_view.columns
        marker_index = heatmap_view.columns if marker_axis == 1 else heatmap_view.index
        self.orientation_state.update({
            "horizontal": horizontal,
            "view": heatmap_view,
            "cluster_axis": cluster_axis,
            "marker_axis": marker_axis,
            "cluster_index": cluster_index,
            "marker_index": marker_index,
            "cluster_order_positions": [],
            "marker_order_positions": [],
            "cluster_leaves": list(cluster_index),
            "marker_leaves": list(marker_index)
        })

    def _cluster_index_labels(self):
        return self.orientation_state.get("cluster_index")

    def _marker_index_labels(self):
        return self.orientation_state.get("marker_index")

    def _cluster_order_positions(self):
        order = self.orientation_state.get("cluster_order_positions")
        if order:
            return order
        labels = self._cluster_index_labels()
        return list(range(len(labels))) if labels is not None else []

    def _marker_order_positions(self):
        order = self.orientation_state.get("marker_order_positions")
        if order:
            return order
        labels = self._marker_index_labels()
        return list(range(len(labels))) if labels is not None else []

    def _cluster_leaves(self):
        leaves = self.orientation_state.get("cluster_leaves")
        if leaves:
            return leaves
        labels = self._cluster_index_labels()
        return list(labels) if labels is not None else []

    def _marker_leaves(self):
        leaves = self.orientation_state.get("marker_leaves")
        if leaves:
            return leaves
        labels = self._marker_index_labels()
        return list(labels) if labels is not None else []

    def _cluster_label_from_selection_index(self, selection_index):
        order_positions = self._cluster_order_positions()
        if selection_index < 0 or selection_index >= len(order_positions):
            return None
        pos = order_positions[selection_index]
        cluster_index = self._cluster_index_labels()
        if cluster_index is None or pos >= len(cluster_index):
            return None
        return cluster_index[pos]

    def _current_cluster_label(self):
        if not hasattr(self, "heatmap_current_selection") or self.heatmap_current_selection is None:
            return None
        selection = self.heatmap_current_selection
        horizontal = self.adapter.is_wide()
        selection_index = selection[1] if horizontal else selection[0]
        return self._cluster_label_from_selection_index(selection_index)

    def _cluster_position_from_label(self, label):
        cluster_index = self._cluster_index_labels()
        if cluster_index is None:
            return None
        try:
            return cluster_index.get_loc(label)
        except KeyError:
            return None

    def _marker_label_from_selection_index(self, selection_index):
        order_positions = self._marker_order_positions()
        if selection_index < 0 or selection_index >= len(order_positions):
            return None
        pos = order_positions[selection_index]
        marker_index = self._marker_index_labels()
        if marker_index is None or pos >= len(marker_index):
            return None
        return marker_index[pos]

    def save_obj_data(self):
        base_folder = self.main_viewer.base_folder
        data_file = os.path.join(base_folder, "heatmap_data.pkl")
        with open(data_file, "wb") as handle:
            pickle.dump(self.data, handle)
        _logger.info("Data autosaved to %s", data_file)

    def prepare_heatmap_data(self):
        df = self.main_viewer.cell_table
        cluster_column = self.ui_component.high_level_cluster_dropdown.value
        subset_on = self.ui_component.subset_on_dropdown.value

        marker_columns = list(self.ui_component.channel_selector.value)
        channel = marker_columns + [cluster_column]

        _logger.debug("Preparing heatmap data for channels: %s", channel)
        _logger.debug("Using cluster: %s", [cluster_column])

        subset = list(self.ui_component.subset_selector.value)
        if subset:
            in_subset = df[subset_on].isin(subset)
            df = df[in_subset]
        cluster_values = df[cluster_column]
        try:
            cluster_count = int(cluster_values.nunique())
        except Exception:
            cluster_count = len(pd.unique(cluster_values))
        if cluster_count > 300:
            _logger.warning("The number of classes is too large to display. Please select a smaller number of classes.")
            return

        df_grouped = df.groupby(cluster_column)[marker_columns].median()

        zscore_across_markers = bool(
            getattr(self.ui_component, 'zscore_across_markers_checkbox', None)
            and self.ui_component.zscore_across_markers_checkbox.value
        )
        if zscore_across_markers:
            row_means = df_grouped.mean(axis=1)
            row_stds = df_grouped.std(axis=1).replace(0, np.nan)
            df_grouped = df_grouped.sub(row_means, axis=0).div(row_stds, axis=0).fillna(0)
        else:
            col_means = df_grouped.mean(axis=0)
            col_stds = df_grouped.std(axis=0).replace(0, np.nan)
            df_grouped = df_grouped.sub(col_means, axis=1).div(col_stds, axis=1).fillna(0)

        self.heatmap_data = df_grouped
        self._update_orientation_state()

    def generate_dendrogram(self):
        self._update_orientation_state()
        heatmap_view = self.orientation_state.get("view")
        if heatmap_view is None or heatmap_view.empty:
            return None

        linkage_method = self.ui_component.cluster_method_dropdown.value
        distance_metric = self.ui_component.distance_metric_dropdown.value

        data_matrix = heatmap_view.T if self.adapter.is_wide() else heatmap_view
        Z = linkage(data_matrix, method=linkage_method, metric=distance_metric)

        distances = Z[:, 2]
        dendrogram_min = distances.min()
        dendrogram_max = distances.max()
        self.data.dendrogram_cut = dendrogram_min + (dendrogram_max - dendrogram_min) * 0.5

        return Z

    def save_to_cell_table(self, *args):
        column_name = self.ui_component.column_name_text.value
        overwrite = self.ui_component.overwrite_checkbox.value

        if column_name in self.main_viewer.cell_table.columns:
            if not overwrite:
                _logger.warning("If you intend to overwrite the existing column, please check the 'Overwrite' checkbox.")
                return
            _logger.info("Overwriting the existing column.")
            self.main_viewer.cell_table.drop(column_name, axis=1, inplace=True)
            if f"{column_name}_revised" in self.main_viewer.cell_table.columns:
                self.main_viewer.cell_table.drop(f"{column_name}_revised", axis=1, inplace=True)
            self.ui_component.overwrite_checkbox.value = False

        high_level_cluster = self.ui_component.high_level_cluster_dropdown.value

        heatmap_data = pd.DataFrame(
            {'meta_cluster': self.heatmap_data['meta_cluster']},
            index=self.heatmap_data.index,
        )
        if 'meta_cluster_revised' in self.heatmap_data.columns:
            heatmap_data['meta_cluster_revised'] = self.heatmap_data['meta_cluster_revised']

        heatmap_data.index.name = high_level_cluster
        heatmap_data.reset_index(inplace=True)

        self.main_viewer.cell_table = pd.merge(
            self.main_viewer.cell_table,
            heatmap_data,
            on=high_level_cluster,
            how='left'
        )
        self.main_viewer.cell_table['meta_cluster'] = self.main_viewer.cell_table['meta_cluster'].fillna(-1).astype('int')
        self.main_viewer.cell_table.rename(columns={'meta_cluster': column_name}, inplace=True)

        if 'meta_cluster_revised' in self.heatmap_data.columns:
            self.main_viewer.cell_table['meta_cluster_revised'] = (
                self.main_viewer.cell_table['meta_cluster_revised'].fillna(-1).astype('int')
            )
            self.main_viewer.cell_table.rename(
                columns={'meta_cluster_revised': f"{column_name}_revised"}, inplace=True
            )

        label_source_column = (
            f"{column_name}_revised"
            if f"{column_name}_revised" in self.main_viewer.cell_table.columns
            else column_name
        )
        self.main_viewer.cell_table[column_name] = self.main_viewer.cell_table[label_source_column].map(
            self._meta_cluster_display_name
        )

        revised_label_column = f"{column_name}_revised"
        if revised_label_column in self.main_viewer.cell_table.columns:
            self.main_viewer.cell_table[revised_label_column] = self.main_viewer.cell_table[
                revised_label_column
            ].map(self._meta_cluster_display_name)

        cluster_columns = self.main_viewer.cell_table.select_dtypes(include=['int', 'int64', 'object']).columns.tolist()
        _logger.debug("Cluster-capable columns: %s", cluster_columns)

        self.main_viewer.inform_plugins('on_cell_table_change')

        _logger.info("Cluster labels saved to column '%s' in the cell table.", column_name)
        self.display_row_colors_as_patches()

    def on_cell_table_change(self):
        cluster_columns = self.main_viewer.cell_table.select_dtypes(include=['int', 'int64', 'object']).columns.tolist()
        old_cluster = self.ui_component.high_level_cluster_dropdown.value
        self.ui_component.high_level_cluster_dropdown.options = cluster_columns
        self.ui_component.high_level_cluster_dropdown.value = old_cluster

        old_subset = self.ui_component.subset_on_dropdown.value
        self.ui_component.subset_on_dropdown.options = cluster_columns
        self.ui_component.subset_on_dropdown.value = old_subset

    # ------------------------------------------------------------------
    # Checkpoint export / import
    # ------------------------------------------------------------------

    def export_heatmap_state(self, *, include_raw_medians: bool = True) -> "Any":
        """Capture current heatmap state as an AnnData object.

        The returned object is ready to be passed to ``CheckpointStore.write_checkpoint``.
        ``uns["checkpoint"]`` is intentionally left empty here — the store fills it in.
        """
        import anndata

        data = getattr(self, "heatmap_data", None)
        if data is None or not hasattr(data, "columns"):
            raise RuntimeError("No heatmap data available to export. Run the heatmap first.")

        _meta_cols = {"meta_cluster", "meta_cluster_revised"}
        marker_cols = [c for c in data.columns if c not in _meta_cols]
        if not marker_cols:
            raise RuntimeError("No marker columns found in heatmap data.")

        X = data[marker_cols].values.astype("float32")
        obs = {}
        if "meta_cluster" in data.columns:
            obs["meta_cluster"] = [
                int(v) if not (isinstance(v, float) and v != v) else -1
                for v in data["meta_cluster"]
            ]
        if "meta_cluster_revised" in data.columns:
            obs["meta_cluster_revised"] = [
                int(v) if not (isinstance(v, float) and v != v) else -1
                for v in data["meta_cluster_revised"]
            ]

        adata = anndata.AnnData(
            X=X,
            obs=pd.DataFrame(obs, index=[str(idx) for idx in data.index]),
            var=pd.DataFrame(index=marker_cols),
        )

        # Palette — convert int keys to str for JSON-safe serialisation
        mc_colors = getattr(self.data, "meta_cluster_colors", {}) or {}
        mc_names = getattr(self.data, "meta_cluster_names", {}) or {}
        adata.uns["palette"] = {
            "colors": {str(k): v for k, v in mc_colors.items()},
            "names": {str(k): v for k, v in mc_names.items()},
            "next_id": int(getattr(self.data, "next_meta_cluster_id", 0) or 0),
        }

        # UI widget values
        ui = self.ui_component
        adata.uns["ui"] = {
            "selected_channels": list(getattr(ui.channel_selector, "value", [])),
            "cluster_method": getattr(ui.cluster_method_dropdown, "value", "ward"),
            "distance_metric": getattr(ui.distance_metric_dropdown, "value", "euclidean"),
            "zscore_across_markers": bool(getattr(ui.zscore_across_markers_checkbox, "value", False)),
            "horizontal_layout": bool(getattr(ui.horizontal_layout_checkbox, "value", False)),
            "high_level_cluster_column": getattr(ui.high_level_cluster_dropdown, "value", ""),
            "subset_on": getattr(ui.subset_on_dropdown, "value", ""),
            "subset_values": list(getattr(ui.subset_selector, "value", []) or []),
        }

        # Dendrogram
        raw_dend = getattr(self, "dendrogram", None)
        if raw_dend is not None:
            adata.uns["row_linkage"] = raw_dend.tolist()

        # Dendrogram cutoff
        cut = getattr(self.data, "dendrogram_cut", None)
        if cut is not None:
            adata.uns["dendrogram_cut"] = float(cut)

        return adata

    def import_heatmap_state(self, adata: "Any") -> None:
        """Restore heatmap state from a previously exported AnnData checkpoint.

        Reconstructs ``heatmap_data``, ``dendrogram``, meta-cluster registry, and
        UI widget values, then re-renders the heatmap from the saved state.
        """
        # 1. Reconstruct heatmap_data DataFrame
        marker_cols = list(adata.var_names)
        obs_index = list(adata.obs_names)
        df = pd.DataFrame(adata.X, index=obs_index, columns=marker_cols)
        for col in ("meta_cluster", "meta_cluster_revised"):
            if col in adata.obs.columns:
                df[col] = adata.obs[col].values
        self.heatmap_data = df

        # 2. Restore dendrogram
        raw_link = adata.uns.get("row_linkage")
        if raw_link is not None:
            self.dendrogram = np.array(raw_link)
        else:
            self.dendrogram = None

        # 3. Restore cutoff
        cut = adata.uns.get("dendrogram_cut")
        if cut is not None:
            self.data.dendrogram_cut = float(cut)

        # 4. Cache revised assignments so _restore_cluster_assignments can reapply
        if "meta_cluster_revised" in df.columns:
            self._cluster_assignment_cache = {
                idx: int(v) for idx, v in df["meta_cluster_revised"].dropna().items()
            }
        else:
            self._cluster_assignment_cache = {}

        # 5. Restore UI widget values (guarded — widgets may be stubs in test env)
        ui_state = adata.uns.get("ui", {})
        _widget_map = [
            ("cluster_method_dropdown",        "cluster_method"),
            ("distance_metric_dropdown",       "distance_metric"),
            ("zscore_across_markers_checkbox", "zscore_across_markers"),
            ("horizontal_layout_checkbox",     "horizontal_layout"),
        ]
        for attr, key in _widget_map:
            widget = getattr(self.ui_component, attr, None)
            val = ui_state.get(key)
            if widget is not None and val is not None:
                try:
                    widget.value = val
                except Exception:
                    pass
        # channel_selector — restore marker list
        ch_widget = getattr(self.ui_component, "channel_selector", None)
        saved_channels = ui_state.get("selected_channels")
        if ch_widget is not None and saved_channels is not None:
            try:
                ch_widget.value = tuple(saved_channels)
            except Exception:
                pass

        # 6. Update orientation state from the loaded data
        self._update_orientation_state()

        # 7. Re-render the heatmap from the saved state
        if self.dendrogram is not None and getattr(self.data, "dendrogram_cut", None) is not None:
            self._refresh_plot()

        # 8. Re-apply saved palette on top of whatever generate_heatmap set
        pal = adata.uns.get("palette", {})
        saved_colors = {int(k): v for k, v in pal.get("colors", {}).items()}
        saved_names = {int(k): v for k, v in pal.get("names", {}).items()}
        next_id = int(pal.get("next_id", 0))
        if saved_colors:
            self.data.meta_cluster_colors.update(saved_colors)
        if saved_names:
            self.data.meta_cluster_names.update(saved_names)
        if next_id:
            self.data.next_meta_cluster_id = max(
                int(getattr(self.data, "next_meta_cluster_id", 0) or 0), next_id
            )
        self._refresh_meta_cluster_controls()

        # Re-render color patches with the correct saved palette
        try:
            self.display_row_colors_as_patches()
        except Exception:
            pass

    def _map_indices_to_cluster_positions(self, selection_indices: Sequence[int] | Set[int]):
        cluster_index = self._cluster_index_labels()
        if cluster_index is None:
            return None

        order_positions = list(self._cluster_order_positions())
        cell_table = getattr(self.main_viewer, 'cell_table', None)
        cluster_column = getattr(self.ui_component.high_level_cluster_dropdown, 'value', None)

        if cell_table is None or cluster_column not in getattr(cell_table, 'columns', []):
            return []

        existing_indices = [idx for idx in selection_indices if idx in cell_table.index]
        if not existing_indices:
            return []

        try:
            cluster_values = cell_table.loc[existing_indices, cluster_column]
        except KeyError:
            return []

        if isinstance(cluster_values, pd.Series):
            values_iter = cluster_values.dropna().tolist()
        elif isinstance(cluster_values, (list, tuple, np.ndarray)):
            values_iter = [value for value in cluster_values if not pd.isna(value)]
        else:
            values_iter = [] if pd.isna(cluster_values) else [cluster_values]

        if not values_iter:
            return []

        positions = []
        seen = set()
        for label in values_iter:
            try:
                base_position = cluster_index.get_loc(label)
            except KeyError:
                continue

            if isinstance(base_position, slice):
                base_range = range(base_position.start, base_position.stop)
            elif isinstance(base_position, (list, tuple, np.ndarray)):
                base_range = base_position
            else:
                base_range = [int(base_position)]

            for base in base_range:
                if isinstance(base, np.generic):
                    base = int(base)
                if base < 0 or base >= len(cluster_index):
                    continue
                if order_positions:
                    try:
                        display_position = order_positions.index(base)
                    except ValueError:
                        continue
                else:
                    display_position = base
                if display_position not in seen:
                    seen.add(display_position)
                    positions.append(int(display_position))

        positions.sort()
        return positions


class InteractionLayer:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _cluster_and_marker_counts(self):
        cluster_count = len(self._cluster_order_positions())
        marker_count = len(self.orientation_state.get("marker_leaves") or [])
        return cluster_count, marker_count

    def _is_valid_heatmap_cell(self, row_index, col_index):
        cluster_count, marker_count = self._cluster_and_marker_counts()
        if self.adapter.is_wide():
            if col_index < 0 or col_index >= cluster_count:
                return False
            if marker_count and (row_index < 0 or row_index >= marker_count):
                return False
            return True

        if row_index < 0 or row_index >= cluster_count:
            return False
        if marker_count and (col_index < 0 or col_index >= marker_count):
            return False
        return True

    def _selection_index_for_cell(self, row_index, col_index):
        return col_index if self.adapter.is_wide() else row_index

    @staticmethod
    def _heatmap_coords_from_event(event):
        if event.xdata is None or event.ydata is None:
            return None
        return int(np.floor(event.ydata)), int(np.floor(event.xdata))

    def _dendrogram_coord_from_event(self, event):
        return event.ydata if self.adapter.is_wide() else event.xdata

    def _color_axis_coord_from_event(self, event):
        return event.xdata if self.adapter.is_wide() else event.ydata

    @staticmethod
    def _cluster_index_from_coord(coord):
        if coord is None:
            return None
        return int(np.ceil(coord) - 1)

    def on_mode_toggle(self, mode):
        if mode not in {"wide", "vertical"}:
            return
        self.adapter.mode = mode

    def on_orientation_toggle(self, change):
        if not self.initialized:
            return
        mode = "wide" if change.get('new') else "vertical"
        self.on_mode_toggle(mode)
        self._reset_selection_cache()
        self._sync_panel_location()
        # Suppress the cached-pane refresh's render (request_cached_wide_panel_refresh
        # early-returns while this flag is set) so refresh_bottom_panel only re-homes the
        # footer pane; the single explicit plot_heatmap() below does the one render.
        self._plot_refresh_inflight = True
        try:
            if hasattr(self.main_viewer, 'refresh_bottom_panel'):
                self.main_viewer.refresh_bottom_panel()
        finally:
            self._plot_refresh_inflight = False
        self.plot_heatmap()

    def after_all_plugins_loaded(self):
        super().after_all_plugins_loaded()
        self._sync_panel_location()
        if hasattr(self.main_viewer, 'refresh_bottom_panel'):
            self.main_viewer.refresh_bottom_panel()
        self.main_viewer.SidePlots.chart_output.selected_indices.add_observer(
            self.on_selected_indices_change
        )

    def _make_click_handler(self, g):
        def on_click(event):
            ax_heatmap = g.ax_heatmap
            dend_axis, color_axis = self._get_cluster_axes()

            if event.inaxes == ax_heatmap:
                coords = self._heatmap_coords_from_event(event)
                if coords is None:
                    return
                row_ind, col_ind = coords
                if not self._is_valid_heatmap_cell(row_ind, col_ind):
                    return

                selection_index = self._selection_index_for_cell(row_ind, col_ind)
                cluster_label = self._cluster_label_from_selection_index(selection_index)
                if cluster_label is None:
                    return

                self.highlight_a_heatmap_grid(row_ind, col_ind)
                self.heatmap_current_selection = (row_ind, col_ind)
                _logger.debug("Clicked cluster: %s", cluster_label)
                cell_count = self.main_viewer.cell_table[
                    self.main_viewer.cell_table[self.ui_component.high_level_cluster_dropdown.value] == cluster_label
                ].shape[0]
                _logger.debug("cell number: %s", cell_count)
                self.update_linked()

            elif dend_axis is not None and event.inaxes == dend_axis:
                if self.ui_component.lock_cutoff_button.value:
                    _logger.warning("Cutoff is locked. Please unlock to apply new cutoff.")
                    return
                value = self._dendrogram_coord_from_event(event)
                if value is not None:
                    self.data.dendrogram_cut = value
                    _logger.debug("New dendrogram cutoff: %s", value)
                    self._draw_cutoff_line(dend_axis)
                    self.apply_new_cutoff()
                    self._engage_cutoff_lock("Cutoff locked after dendrogram update")

            elif color_axis is not None and event.inaxes == color_axis:
                coord = self._color_axis_coord_from_event(event)
                selected_idx = self._cluster_index_from_coord(coord)
                cluster_count, _ = self._cluster_and_marker_counts()
                if selected_idx is None or selected_idx < 0 or selected_idx >= cluster_count:
                    _logger.debug("Clicked outside the color bar.")
                    return

                if event.button == MouseButton.RIGHT:
                    self.current_selection = []
                    self.data.current_clusters["index"].value = []
                    _logger.debug("Cleared selection.")
                else:
                    if event.key == 'shift' and self.data.current_clusters["index"].value:
                        start = min(self.data.current_clusters["index"].value[-1], selected_idx)
                        end = max(self.data.current_clusters["index"].value[-1], selected_idx) + 1
                        self.current_selection = list(range(start, end))
                    elif event.key == 'control':
                        if not hasattr(self, 'ctrl_selection'):
                            self.ctrl_selection = set()
                        self.ctrl_selection.update(self.data.current_clusters["index"].value)
                        self.ctrl_selection.add(selected_idx)
                        self.current_selection = sorted(self.ctrl_selection)
                        self.ctrl_selection = set()
                    else:
                        self.current_selection = [selected_idx]

                    self.data.current_clusters["index"].value = self.current_selection
                    cluster_labels = [self._cluster_label_from_selection_index(idx) for idx in self.current_selection]
                    cluster_labels = [label for label in cluster_labels if label is not None]
                    if cluster_labels:
                        meta_cluster_id = self.heatmap_data.loc[cluster_labels, 'meta_cluster']
                        values = meta_cluster_id.values if hasattr(meta_cluster_id, 'values') else meta_cluster_id
                        _logger.debug("Meta cluster IDs: %s", list(values))
                    self.highlight_row_colors(self.current_selection)

        return on_click

    def highlight_row_colors(self, indices):
        _, color_axis = self._get_cluster_axes()
        if color_axis is None:
            return

        self.highlight_patches = _remove_patches(getattr(self, 'highlight_patches', []))

        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, range):
            indices = list(indices)

        for position in indices:
            start = position
            end = position + 1
            patch = _draw_orientation_span(
                self.adapter,
                color_axis,
                start,
                end,
                color='white',
                alpha=0.3,
                zorder=10,
            )
            if patch is not None:
                self.highlight_patches.append(patch)

        color_axis.figure.canvas.draw_idle()

    def highlight_a_heatmap_grid(self, row_index, col_index):
        self.heatmap_highlight_patches = _remove_patches(getattr(self, 'heatmap_highlight_patches', []))

        ax_heatmap = self.data.g.ax_heatmap
        if not self._is_valid_heatmap_cell(row_index, col_index):
            return

        patch = Rectangle((col_index, row_index), 1, 1, facecolor='yellow', alpha=0.3, edgecolor='none')
        ax_heatmap.add_patch(patch)
        self.heatmap_highlight_patches.append(patch)
        self.data.g.fig.canvas.draw()

    def update_linked(self):
        if self.ui_component.main_viewer_checkbox.value:
            self.highlight_cells()

        if self.ui_component.cell_gallery_checkbox.value:
            self.display_cells()

        # Scatter plot and histogram are now separate plugins (see #112), so the
        # heatmap links to each independently. The scatter link colours + selects
        # the active cluster's points; the histogram link overlays the cluster's
        # distribution. Both are guarded by their own checkbox (#114).
        if self.ui_component.chart_checkbox.value:
            self.color_points_by_meta_cluster()
            self.highlight_scatter_plot()

        if self.ui_component.histogram_checkbox.value:
            self.update_histogram_distribution()

    def color_points_by_meta_cluster(self):
        heatmap_data = self.heatmap_data.copy()

        high_level_cluster = self.ui_component.high_level_cluster_dropdown.value

        heatmap_data.index.name = high_level_cluster
        heatmap_data.reset_index(inplace=True)
        if 'meta_cluster_revised' not in heatmap_data.columns:
            heatmap_data = heatmap_data[[high_level_cluster, 'meta_cluster']]
        else:
            heatmap_data = heatmap_data[[high_level_cluster, 'meta_cluster', 'meta_cluster_revised']]

        used_cluster = 'meta_cluster_revised' if 'meta_cluster_revised' in heatmap_data.columns else 'meta_cluster'

        cell_table = self.main_viewer.cell_table.copy()

        merged_table = pd.merge(
            cell_table,
            heatmap_data,
            on=high_level_cluster,
            how='left'
        )

        self.data.celltable = merged_table

        chart_display = getattr(self.main_viewer.SidePlots, "chart_output", None)
        if chart_display is None:
            _logger.warning("Chart plugin not available for coloring.")
            return
        meta_colors = getattr(self.data, "meta_cluster_colors", {}) or {}
        cluster_colors = getattr(self.data, "cluster_colors", {}) or {}
        if meta_colors:
            color_map_source = meta_colors
            color_series = merged_table[used_cluster]
        elif cluster_colors:
            color_map_source = cluster_colors
            color_series = merged_table[high_level_cluster]
        else:
            color_map_source = {}
            color_series = merged_table[used_cluster]
        if not color_map_source:
            _logger.warning("Cluster color palette not available.")
            return
        normalized_map = {}
        for key, value in color_map_source.items():
            if isinstance(value, np.ndarray):
                normalized_map[key] = tuple(float(part) for part in value.tolist())
            elif isinstance(value, (list, tuple)):
                normalized_map[key] = tuple(float(part) for part in value)
            else:
                normalized_map[key] = value
        chart_display.apply_color_mapping(color_series, normalized_map, default_color="grey")

    def highlight_scatter_plot(self):
        cluster_label = self._current_cluster_label()
        if cluster_label is None:
            _logger.warning(NO_CLUSTER_SELECTED_MSG)
            return

        high_level_cluster = self.ui_component.high_level_cluster_dropdown.value
        cell_table = self.main_viewer.cell_table

        if self.main_viewer.SidePlots.chart_output.ui_component.impose_fov_checkbox.value:
            row_indices = cell_table.loc[
                (cell_table[self.main_viewer.fov_key] == self.main_viewer.ui_component.image_selector.value) &
                (cell_table[high_level_cluster] == cluster_label)
            ].reindex().index.tolist()
        else:
            row_indices = cell_table.loc[
                cell_table[high_level_cluster] == cluster_label
            ].index.tolist()

        _logger.debug("Selected indices: %s", row_indices)
        self.main_viewer.SidePlots.chart_output.color_points(row_indices)

    def update_histogram_distribution(self):
        """Overlay the active cluster's cells as a distribution in the histogram plugin.

        Implements the previously-missing histogram response for the heatmap link
        (#114). The selected cluster's row indices are pushed to the standalone
        histogram plugin, which draws them as the "Selected" overlay on every
        plotted channel. The histogram's own subset/FOV settings decide which of
        those cells are actually shown, so we forward the full cluster here.
        """
        cluster_label = self._current_cluster_label()
        if cluster_label is None:
            _logger.warning(NO_CLUSTER_SELECTED_MSG)
            return

        histogram = getattr(self.main_viewer.SidePlots, "histogram_output", None)
        if histogram is None:
            _logger.warning("Histogram plugin not available.")
            return

        high_level_cluster = self.ui_component.high_level_cluster_dropdown.value
        cell_table = self.main_viewer.cell_table
        row_indices = cell_table.loc[
            cell_table[high_level_cluster] == cluster_label
        ].index.tolist()

        _logger.debug("Histogram selection indices: %s", row_indices)
        histogram.show_external_selection(row_indices)

    def display_cells(self):
        cluster_label = self._current_cluster_label()
        if cluster_label is None:
            _logger.warning(NO_CLUSTER_SELECTED_MSG)
            return

        high_level_cluster = self.ui_component.high_level_cluster_dropdown.value
        cell_table = self.main_viewer.cell_table

        if self.ui_component.current_fov_checkbox.value:
            row_indices = cell_table.loc[
                (cell_table[self.main_viewer.fov_key] == self.main_viewer.ui_component.image_selector.value) &
                (cell_table[high_level_cluster] == cluster_label)
            ].index.tolist()
        else:
            row_indices = cell_table.loc[
                cell_table[high_level_cluster] == cluster_label
            ].index.tolist()

        self.main_viewer.SidePlots.cell_gallery_output.set_selected_cells(row_indices)

    def highlight_cells(self):
        cluster_label = self._current_cluster_label()
        if cluster_label is None:
            _logger.warning(NO_CLUSTER_SELECTED_MSG)
            return

        high_level_cluster = self.ui_component.high_level_cluster_dropdown.value
        cell_table = self.main_viewer.cell_table

        sub_table = cell_table.loc[cell_table[self.main_viewer.fov_key] == self.main_viewer.ui_component.image_selector.value, :]
        label_key = self.main_viewer.label_key
        mask_label = sub_table.loc[sub_table[high_level_cluster] == cluster_label, label_key].tolist()

        self.main_viewer.image_display.set_mask_ids(mask_name=self.main_viewer.mask_key, mask_ids=mask_label)
        _logger.debug("%s in the main viewer.", mask_label)
        _logger.info("Highlighted cells from cluster %s of %s in the main viewer.", cluster_label, high_level_cluster)

    def trace_cluster(self, *args):
        selections = self.main_viewer.image_display.selected_masks_label
        if not selections:
            _logger.warning("Please select a cell in the main viewer.")
            return
        selection = next(iter(selections))
        cell_id = getattr(selection, "mask_id", None)
        if cell_id is None:
            _logger.warning("Could not determine selected cell identifier.")
            return
        fov = getattr(selection, "fov", None) or self.main_viewer.ui_component.image_selector.value
        cluster_column = self.ui_component.high_level_cluster_dropdown.value
        label_key = self.main_viewer.label_key
        fov_key = self.main_viewer.fov_key
        cluster = self.main_viewer.cell_table.loc[
            (self.main_viewer.cell_table[label_key] == cell_id) & (self.main_viewer.cell_table[fov_key] == fov),
            cluster_column
        ].values[0]
        _logger.debug("Cluster of the selected cell: %s", cluster)

        if cluster_column != self.heatmap_data.index.name:
            _logger.warning("Cluster column not found in the heatmap data.")
            return

        cluster_index = self._cluster_index_labels()
        order_positions = list(self._cluster_order_positions())
        if cluster_index is None or not order_positions:
            _logger.warning("Heatmap ordering not available.")
            return

        try:
            base_position = cluster_index.get_loc(cluster)
        except KeyError:
            _logger.warning("Cluster not found in the heatmap data.")
            return

        try:
            selection_index = order_positions.index(base_position)
        except ValueError:
            _logger.warning("Cluster position not found in current ordering.")
            return

        if self.adapter.is_wide():
            row_ind = 0
            col_ind = selection_index
        else:
            row_ind = selection_index
            col_ind = 0

        self.heatmap_current_selection = (row_ind, col_ind)
        self.highlight_a_heatmap_grid(row_ind, col_ind)
        _logger.debug("Found cluster at heatmap index: %s", selection_index)

        self.highlight_cells()

    def trace_metacluster(self, *args):
        """Placeholder for future metacluster tracing workflow.

        The legacy heatmap display exposed this hook but never shipped the
        implementation. We keep the stub so callers depending on the method
        do not break while we finish the refactor series.
        """
        pass

    def _apply_cluster_highlights(self, positions):
        normalized = [int(pos) for pos in positions]
        normalized_tuple = tuple(normalized)
        if getattr(self, '_last_highlighted_clusters', None) == normalized_tuple:
            return

        self.highlight_row_colors(normalized)

        current_selection = list(self.data.current_clusters["index"].value or [])
        if current_selection != normalized:
            self.data.current_clusters["index"].value = normalized

        self._last_highlighted_clusters = normalized_tuple

    def on_selected_indices_change(self, selected_indices):
        _logger.debug("Selected indices have changed.")

        if selected_indices is None:
            selected_items = []
        elif isinstance(selected_indices, (set, frozenset)):
            selected_items = list(selected_indices)
        elif isinstance(selected_indices, (list, tuple)):
            selected_items = list(selected_indices)
        else:
            try:
                selected_items = list(selected_indices)
            except TypeError:
                selected_items = [selected_indices]

        selection_set = set(selected_items)
        normalized_selection = tuple(sorted(selection_set))

        if getattr(self, '_last_scatter_selection', None) == normalized_selection:
            return

        if not hasattr(self, 'heatmap_data') or self.heatmap_data is None:
            self._last_scatter_selection = normalized_selection
            if selection_set:
                try:
                    self.plot_heatmap()
                except Exception:
                    _logger.error("Error while plotting heatmap", exc_info=True)
            return

        if not selection_set:
            self._apply_cluster_highlights([])
            self._last_scatter_selection = normalized_selection
            return

        positions = self._map_indices_to_cluster_positions(selection_set)
        if positions is None:
            self._last_scatter_selection = normalized_selection
            try:
                self.plot_heatmap()
            except Exception:
                _logger.error("Error while plotting heatmap", exc_info=True)
            return

        self._apply_cluster_highlights(positions)
        self._last_scatter_selection = normalized_selection

    def on_subset_on_dropdown_change(self, change):
        selected_clusters = change['new']
        if selected_clusters:
            filtered_fovs = self.main_viewer.cell_table[selected_clusters].unique()
            self.ui_component.subset_selector.options = filtered_fovs
        else:
            self.ui_component.subset_selector.options = []

    def update_ui_components(self, indices):
        has_options = bool(getattr(self.ui_component.cluster_id_dropdown, 'options', []))
        is_enabled = bool(indices) and has_options
        self.ui_component.cluster_id_dropdown.disabled = not is_enabled
        self.ui_component.cluster_id_apply_button.disabled = not is_enabled

    def on_rename_cluster_selection_change(self, change):
        selected_id = change.get('new') if isinstance(change, dict) else None
        if selected_id is None:
            self.ui_component.rename_cluster_name.value = ""
            return
        self.ui_component.rename_cluster_name.value = self._meta_cluster_display_name(selected_id)

    def apply_new_cluster_id(self, *args):
        new_cluster_id = self.ui_component.cluster_id_dropdown.value
        selected_indices = list(self.data.current_clusters["index"].value or [])
        if new_cluster_id is None:
            _logger.warning("Please select a meta-cluster.")
            return
        if not selected_indices:
            _logger.warning("No clusters selected to update.")
            return

        sorted_positions = self._cluster_order_positions()
        if not sorted_positions:
            _logger.warning("No cluster ordering available.")
            return

        self._ensure_meta_cluster_revised_column()
        self._ensure_meta_cluster_entry(new_cluster_id)
        self._refresh_meta_cluster_controls()

        cluster_index = self._cluster_index_labels()
        if cluster_index is None:
            _logger.warning("Cluster index not available.")
            return

        selection_positions = np.array(sorted_positions)[np.array(selected_indices)]
        cluster_labels = cluster_index.take(selection_positions)

        self.heatmap_data.loc[cluster_labels, 'meta_cluster_revised'] = new_cluster_id

        self.display_row_colors_as_patches()

        if not self.adapter.is_wide():
            self.update_text_labels()

        self._engage_cutoff_lock("Cutoff locked after meta-cluster reassignment")
        _logger.info("Assigned selected clusters to %s (%s).", self._meta_cluster_display_name(new_cluster_id), new_cluster_id)

    def rename_meta_cluster(self, *args):
        selected_id = self.ui_component.rename_cluster_dropdown.value
        new_name = (self.ui_component.rename_cluster_name.value or "").strip()
        if selected_id is None:
            _logger.warning("Please select a meta-cluster to rename.")
            return
        if not new_name:
            _logger.warning("Please enter a non-empty name.")
            return

        self.data.meta_cluster_names[selected_id] = new_name
        self._refresh_meta_cluster_controls()
        self.ui_component.rename_cluster_dropdown.value = selected_id
        self.ui_component.cluster_id_dropdown.value = selected_id
        _logger.info("Renamed meta-cluster %s to '%s'.", selected_id, new_name)

    def add_meta_cluster(self, *args):
        new_id = self._next_available_meta_cluster_id()
        requested_name = (self.ui_component.new_cluster_name.value or "").strip()
        new_name = requested_name if requested_name else f"Meta-cluster {new_id}"

        self.data.meta_cluster_names[new_id] = new_name
        self.data.meta_cluster_colors[new_id] = self._generate_meta_cluster_color(new_id)
        self.ui_component.new_cluster_name.value = ""

        self._refresh_meta_cluster_controls()
        self.ui_component.rename_cluster_dropdown.value = new_id
        self.ui_component.cluster_id_dropdown.value = new_id
        _logger.info("Added meta-cluster %s (%s).", new_id, new_name)

    def remove_meta_cluster(self, *args):
        selected_id = self.ui_component.rename_cluster_dropdown.value
        if selected_id is None:
            _logger.warning("Please select a meta-cluster to remove.")
            return
        if selected_id == UNASSIGNED_META_CLUSTER_ID:
            _logger.warning("The unassigned meta-cluster cannot be removed.")
            return

        if hasattr(self, 'heatmap_data') and self.heatmap_data is not None:
            self._ensure_meta_cluster_revised_column()
            column = 'meta_cluster_revised'
            data_store = getattr(self.heatmap_data, '_data', None)
            if isinstance(data_store, dict) and column in data_store:
                data_store[column] = [
                    UNASSIGNED_META_CLUSTER_ID if value == selected_id else value
                    for value in data_store[column]
                ]
            else:
                self.heatmap_data.loc[
                    self.heatmap_data[column] == selected_id,
                    column,
                ] = UNASSIGNED_META_CLUSTER_ID

        self.data.meta_cluster_names.pop(selected_id, None)
        self.data.meta_cluster_colors.pop(selected_id, None)
        self._ensure_meta_cluster_entry(UNASSIGNED_META_CLUSTER_ID)
        self._refresh_meta_cluster_controls()
        self.ui_component.rename_cluster_dropdown.value = UNASSIGNED_META_CLUSTER_ID
        self.ui_component.cluster_id_dropdown.value = UNASSIGNED_META_CLUSTER_ID

        self.display_row_colors_as_patches()
        if not self.adapter.is_wide():
            self.update_text_labels()

        self._engage_cutoff_lock("Cutoff locked after meta-cluster removal")
        _logger.info(
            "Removed meta-cluster %s. Existing assignments were moved to %s (%s).",
            selected_id, UNASSIGNED_META_CLUSTER_NAME, UNASSIGNED_META_CLUSTER_ID,
        )


class DisplayLayer:
    def __init__(self, *args, **kwargs):
        self._restoring_plot_section = False
        self._plot_refresh_inflight = False
        super().__init__(*args, **kwargs)

    def _get_cluster_axes(self):
        g = getattr(self.data, 'g', None)
        if g is None:
            return None, None
        if self.adapter.is_wide():
            dend_axis = getattr(g, 'ax_col_dendrogram', None)
            color_axis = getattr(g, 'ax_col_colors', None)
        else:
            dend_axis = getattr(g, 'ax_row_dendrogram', None)
            color_axis = getattr(g, 'ax_row_colors', None)
        return dend_axis, color_axis

    def _draw_cutoff_line(self, dend_axis):
        self.current_vline = _update_cutoff_line(
            self.adapter,
            dend_axis,
            self.data.dendrogram_cut,
            getattr(self, 'current_vline', None),
        )

    def initiate_ui(self):
        setup = VBox([
            HBox([
                VBox([
                    self.ui_component.channel_selector_text,
                    self.ui_component.channel_selector,
                    self.ui_component.high_level_cluster_dropdown
                    ], layout=Layout(width='50%', overflow='hidden')),
                VBox([
                    self.ui_component.subset_on_dropdown,
                    self.ui_component.subset_selector
                    ], layout=Layout(width='50%', overflow='hidden')),
                ]),
            HBox([
                self.ui_component.cluster_method_dropdown,
                self.ui_component.distance_metric_dropdown,
                self.ui_component.horizontal_layout_checkbox,
                self.ui_component.zscore_across_markers_checkbox,
            ], layout=Layout(gap='8px')),
            HBox([self.ui_component.plot_button])
        ])

        edit = VBox([
            HBox([
                self.ui_component.lock_cutoff_button,
                self.ui_component.lock_override_button
            ], layout=Layout(gap='8px')),
            HBox([self.ui_component.cluster_id_dropdown,
                self.ui_component.cluster_id_apply_button])
        ])

        rename = VBox([
            HBox([self.ui_component.rename_cluster_dropdown]),
            HBox([self.ui_component.rename_cluster_name, self.ui_component.rename_cluster_apply_button]),
            HBox([self.ui_component.new_cluster_name, self.ui_component.add_cluster_button]),
            HBox([self.ui_component.remove_cluster_button]),
            self.ui_component.meta_cluster_registry_box,
        ], layout=Layout(gap='6px'))

        link = VBox([
            HBox([self.ui_component.main_viewer_checkbox]),
            HBox([self.ui_component.chart_checkbox, self.ui_component.histogram_checkbox]),
            HBox([self.ui_component.cell_gallery_checkbox, self.ui_component.current_fov_checkbox]),
        ])

        trace = VBox([
            HBox([self.ui_component.trace_cluster_button]),
            HBox([self.ui_component.trace_metacluster_button])
        ])

        save = VBox([
            HBox([self.ui_component.column_name_text]),
            HBox([self.ui_component.save_to_cell_table_button]),
            HBox([self.ui_component.overwrite_checkbox])
        ])

        self.controls_tab = Tab(
            children=[setup, edit, rename, trace, link, save],
            titles=('Setup', 'Assign', 'Rename', 'Trace', 'Linked plugins', 'Save')
        )

        self.controls_section = VBox(
            [self.controls_tab],
            layout=Layout(width='100%', max_width='99%', min_width='0', box_sizing='border-box', gap='8px'),
        )
        self.plot_section = VBox(
            [self.plot_output],
            layout=Layout(width='100%', max_width='99%', min_width='0', box_sizing='border-box', flex='1 1 auto'),
        )

        self.ui = VBox(
            [self.controls_section, self.plot_section],
            layout=Layout(width='100%', max_width='99%', min_width='0', box_sizing='border-box', max_height='800px', gap='12px')
        )

        self._wide_notice = HTML(
            value="<b>Horizontal layout enabled.</b> Controls and plots live in the footer tabs.",
            layout=Layout(width='100%', max_width='99%', min_width='0', box_sizing='border-box', padding='8px')
        )
        self._section_location = 'vertical'
        self._ensure_plot_canvas_attached()

    def _place_sections_vertical(self):
        already_vertical = getattr(self, '_section_location', 'vertical') == 'vertical'
        if not already_vertical:
            self.ui.children = [self.controls_section, self.plot_section]
            self.ui.layout.display = ''
            self._section_location = 'vertical'
        self._ensure_plot_canvas_attached()

    def _place_sections_horizontal(self):
        if getattr(self, '_section_location', 'vertical') == 'horizontal':
            return
        self.ui.children = [self._wide_notice]
        self.ui.layout.display = ''
        self._section_location = 'horizontal'
        self._ensure_plot_canvas_attached()

    def _sync_panel_location(self):
        if self.adapter.is_wide():
            self._place_sections_horizontal()
        else:
            self._place_sections_vertical()

    def wide_panel_layout(self):
        if self.adapter.is_wide():
            self._place_sections_horizontal()
            self._ensure_plot_canvas_attached()
            return {
                "title": self.displayed_name,
                "control": self.controls_section,
                "content": self.plot_section
            }
        self._place_sections_vertical()
        return None

    def request_cached_wide_panel_refresh(self):
        if not getattr(self, 'initialized', False):
            _logger.debug('[heatmap] skip cached refresh: plugin not initialised')
            return
        if not self.adapter.is_wide():
            _logger.debug('[heatmap] skip cached refresh: not in wide layout')
            return
        if getattr(self, '_plot_refresh_inflight', False):
            _logger.debug('[heatmap] skip cached refresh: refresh already running')
            return
        _logger.debug('[heatmap] refreshing cached wide pane')
        self._plot_refresh_inflight = True
        try:
            self._ensure_plot_canvas_attached()
            self.plot_heatmap()
            _logger.debug('[heatmap] wide pane refresh complete')
        finally:
            self._plot_refresh_inflight = False

    def plot_heatmap(self, *args):
        self._cache_cluster_assignments()
        self._reset_selection_cache()
        _logger.debug("[heatmap] plot_heatmap: starting data preparation")
        self.prepare_heatmap_data()
        _logger.debug("[heatmap] plot_heatmap: generating dendrogram")
        self.dendrogram = self.generate_dendrogram()
        _logger.debug("[heatmap] plot_heatmap: dendrogram=%s", "ok" if self.dendrogram is not None else "None")
        self._refresh_plot()
        mode = 'wide' if self.adapter.is_wide() else 'vertical'
        _logger.debug('[heatmap] plot refreshed in %s mode', mode)

    def load_heatmap(self, heatmap_df, cutoff, *args):
        self._reset_selection_cache()
        df = heatmap_df.copy()
        df = df.drop(df.columns[-2:], axis=1)
        df = df.drop(df.columns[0], axis=1)
        self.heatmap_data = df
        self._update_orientation_state()

        self.dendrogram = self.generate_dendrogram()
        heatmap_df.set_index(heatmap_df.columns[0], inplace=True)
        self.heatmap_data = heatmap_df
        self._update_orientation_state()

        self._refresh_plot()

        self.data.dendrogram_cut = cutoff

        _logger.info("New dendrogram cutoff: %s", cutoff)
        if getattr(self, 'current_vline', None):
            self.current_vline.remove()
        self.current_vline = self.data.g.ax_row_dendrogram.axvline(
            x=self.data.dendrogram_cut,
            color='red',
            linestyle='--'
        )
        self.apply_new_cutoff()

        self.display_row_colors_as_patches()
        self.update_text_labels()

    def _ensure_plot_canvas_attached(self):
        """Make the current ``plot_output`` the sole child of ``plot_section``.

        Reassigning ``.children`` is what forces the frontend to (re)instantiate the
        Output view — the same mechanism the Chart plugin's histogram relies on.
        """
        if getattr(self, '_restoring_plot_section', False):
            return
        plot_section = getattr(self, 'plot_section', None)
        plot_output = getattr(self, 'plot_output', None)
        if plot_section is None or plot_output is None:
            return
        self._restoring_plot_section = True
        try:
            plot_section.children = (plot_output,)
        finally:
            self._restoring_plot_section = False

    def _refresh_plot(self, restore_size=None):
        """Render the heatmap into a fresh Output and swap it into ``plot_section``.

        Mirrors the Chart plugin's reliable interactive histogram idiom
        (``chart.py:_render_histogram``): the figure is **built outside** any Output
        display context (here, with ``plt.ioff()`` so the ipympl backend does not
        auto-emit the clustermap canvas), then the interactive canvas is **emitted exactly
        once** via ``display(fig.canvas)`` inside a brand-new ``Output``, which is then
        swapped into ``plot_section.children`` to force the frontend to repaint.

        Building the clustermap *inside* the Output (the previous behavior) made ipympl
        emit the canvas on creation and again on ``plt.show()`` — a duplicate/blank canvas.
        That was the heatmap-specific interaction with ipympl behind issue #108.

        ``restore_size`` (a ``(width, height)`` in inches from ``_capture_heatmap_scale``)
        is applied as the clustermap ``figsize`` so a tree-cut update keeps the size the
        user set with the ipympl resize triangle (issue #109). It is ``None`` for a fresh
        Plot / load, which use the adapter's default figure size.
        """
        self.restore_vertical_canvas()
        new_out = Output(layout=Layout(width='100%'))

        # Build the figure OUTSIDE the Output context with interactive auto-display off.
        was_interactive = plt.isinteractive()
        plt.ioff()
        try:
            self.generate_heatmap(figsize_override=restore_size)
        finally:
            if was_interactive:
                plt.ion()

        g = getattr(self.data, 'g', None)
        fig = getattr(g, 'fig', None) if g is not None else None

        self.plot_output = new_out
        if fig is not None:
            canvas = getattr(fig, 'canvas', None)
            with new_out:
                display_target = canvas if canvas is not None else fig
                display(display_target)
        self._swap_plot_output_in_section(new_out)
        self._present_footer_canvas_if_wide(fig)

    def _capture_heatmap_scale(self):
        """Return the current figure size ``(width, height)`` in inches, or ``None``.

        Used to remember the "scale" the user set by dragging the ipympl resize handle (the
        triangle at the bottom-right corner) before a tree-cut rebuild (issue #109).
        ``Canvas.handle_resize`` writes the manual resize back via ``fig.set_size_inches``,
        so ``fig.get_size_inches()`` reflects it.
        """
        g = getattr(self.data, 'g', None)
        fig = getattr(g, 'fig', None) if g is not None else None
        if fig is None:
            return None
        try:
            w, h = fig.get_size_inches()
            return (float(w), float(h))
        except Exception:
            return None

    def _present_footer_canvas_if_wide(self, fig):
        """Force the (reparented) ipympl canvas to repaint once the footer is visible.

        In wide mode ``plot_section`` lives inside the footer tab, which is unhidden in the
        same synchronous handler as this render — so the canvas is drawn before the
        frontend has laid the footer out and stays blank until a later resize. A synchronous
        ``draw()`` is the immediate backstop (same pattern as the main image canvas in
        ``main_viewer``); a single-shot timer then issues ``draw_idle()`` after the frontend
        has processed the layout (same deferred-timer primitive as ``image_display``).
        """
        if fig is None or not self.adapter.is_wide():
            return
        canvas = getattr(fig, 'canvas', None)
        if canvas is None:
            return
        if hasattr(canvas, 'draw'):
            try:
                canvas.draw()
            except Exception:
                pass
        new_timer = getattr(canvas, 'new_timer', None)
        if not callable(new_timer):
            return
        try:
            timer = new_timer(interval=150)
            timer.single_shot = True
            timer.add_callback(self._deferred_footer_draw, canvas)
            timer.start()
        except Exception:
            pass

    def _deferred_footer_draw(self, canvas):
        draw_idle = getattr(canvas, 'draw_idle', None)
        if callable(draw_idle):
            try:
                draw_idle()
            except Exception:
                pass

    def _swap_plot_output_in_section(self, new_out):
        plot_section = getattr(self, 'plot_section', None)
        if plot_section is None:
            return
        self._restoring_plot_section = True
        try:
            plot_section.children = (new_out,)
        finally:
            self._restoring_plot_section = False

    def restore_vertical_canvas(self):
        if self.adapter.is_wide():
            return
        self._ensure_plot_canvas_attached()

    def restore_footer_canvas(self):
        if not self.adapter.is_wide():
            return
        self._ensure_plot_canvas_attached()

    def _heatmap_colormap_settings(self):
        zscore_toggle = bool(
            getattr(self.ui_component, 'zscore_across_markers_checkbox', None)
            and self.ui_component.zscore_across_markers_checkbox.value
        )
        if zscore_toggle:
            return {"cmap": "bwr", "center": 0}
        return {"cmap": "Reds", "center": None}

    @update_status_bar
    def generate_heatmap(self, figsize_override=None):
        _logger.debug("[heatmap] generate_heatmap: entry")
        markers = list(self.ui_component.channel_selector.value)
        if not markers:
            _logger.debug("[heatmap] generate_heatmap: early return — no markers selected")
            return

        self._update_orientation_state()
        heatmap_view = self.orientation_state.get("view")
        if heatmap_view is None or heatmap_view.empty:
            _logger.debug("[heatmap] generate_heatmap: early return — heatmap view is None or empty")
            return

        cutoff = self.data.dendrogram_cut
        _logger.debug("[heatmap] generate_heatmap: dendrogram=%s, cutoff=%s", self.dendrogram is not None, cutoff)
        if self.dendrogram is None:
            _logger.debug("[heatmap] generate_heatmap: early return — dendrogram not available")
            return
        meta_cluster_labels = cut_tree(self.dendrogram, height=cutoff).flatten()

        cluster = [self.ui_component.high_level_cluster_dropdown.value]
        subset_on = self.ui_component.subset_on_dropdown.value

        base_index = self.orientation_state.get("cluster_index")
        if base_index is None:
            _logger.debug("[heatmap] generate_heatmap: early return — cluster_index is None")
            return
        self.heatmap_data = self.heatmap_data.reindex(base_index)
        self.heatmap_data['meta_cluster'] = meta_cluster_labels
        self._restore_cluster_assignments()

        num_clusters = len(np.unique(meta_cluster_labels))
        cluster_palette = sns.color_palette('husl', num_clusters)
        cluster_colors = dict(zip(np.unique(meta_cluster_labels), cluster_palette))

        cluster_labels = pd.Series(meta_cluster_labels, index=base_index)
        cluster_colors_series = cluster_labels.map(cluster_colors)

        self.data.cluster_colors = cluster_colors_series.to_dict()
        self._sync_meta_cluster_registry(np.unique(meta_cluster_labels), dict(cluster_colors))

        sns.set_context('notebook')

        marker_axis_labels = list(self.adapter.map_markers_to_axis(heatmap_view))

        requested_markers = [m for m in markers if m in marker_axis_labels]
        missing_markers = [m for m in markers if m not in marker_axis_labels]
        if missing_markers:
            _logger.debug("[heatmap] generate_heatmap: markers not in data (skipped): %s", missing_markers)

        if not requested_markers:
            _logger.debug("[heatmap] generate_heatmap: early return — no requested markers available in heatmap data")
            return

        plot_data = self.adapter.slice_for_markers(heatmap_view, requested_markers)

        self.orientation_state.update({
            "view": plot_data,
            "cluster_index": pd.Index(base_index),
            "marker_index": pd.Index(requested_markers)
        })

        clustermap_kwargs = self.adapter.build_clustermap_kwargs(
            plot_data,
            self.dendrogram,
            meta_cluster_labels,
            self.width,
            self.height,
            cluster_colors_series,
            **self._heatmap_colormap_settings(),
        )

        # Preserve a user-set figure size (ipympl resize triangle) across a tree-cut
        # rebuild by building the clustermap at that size, so tight_layout is correct too
        # (issue #109). Only the cutoff path passes this; a fresh Plot uses the default.
        if figsize_override is not None:
            clustermap_kwargs['figsize'] = tuple(figsize_override)

        # Close the previous figure before building a new one so repeated Plot clicks and
        # cutoff drags don't leak figures (each sns.clustermap opens a new one).
        previous_g = getattr(self.data, 'g', None)
        previous_fig = getattr(previous_g, 'fig', None) if previous_g is not None else None
        if previous_fig is not None:
            try:
                plt.close(previous_fig)
            except Exception:
                pass

        _logger.debug("[heatmap] generate_heatmap: calling sns.clustermap (shape=%s)", plot_data.shape)
        try:
            g = sns.clustermap(**clustermap_kwargs)
        except Exception:
            _logger.debug("[heatmap] generate_heatmap: sns.clustermap FAILED", exc_info=True)
            return

        self.data.g = g
        _logger.debug("[heatmap] generate_heatmap: clustermap ok, setting up layout")

        if hasattr(g, 'cax') and g.cax is not None:
            g.cax.set_visible(False)
            g.cax.remove()

        try:
            self._setup_layout(g, cluster, subset_on)
        except Exception:
            _logger.debug("[heatmap] generate_heatmap: _setup_layout FAILED", exc_info=True)
            return

        _logger.debug("[heatmap] generate_heatmap: render complete")

    def _setup_layout(self, g, cluster, subset_on):
        cluster_index = self.orientation_state.get("cluster_index")
        marker_index = self.orientation_state.get("marker_index")
        if cluster_index is None:
            return

        horizontal = self.adapter.is_wide()

        if horizontal and hasattr(g, 'dendrogram_col') and g.dendrogram_col is not None:
            cluster_order_positions = g.dendrogram_col.reordered_ind
        elif not horizontal and hasattr(g, 'dendrogram_row') and g.dendrogram_row is not None:
            cluster_order_positions = g.dendrogram_row.reordered_ind
        else:
            cluster_order_positions = list(range(len(cluster_index)))

        if marker_index is None:
            marker_order_positions = []
        elif not horizontal and hasattr(g, 'dendrogram_col') and g.dendrogram_col is not None:
            marker_order_positions = g.dendrogram_col.reordered_ind
        else:
            marker_order_positions = list(range(len(marker_index)))

        cluster_leaves = [cluster_index[i] for i in cluster_order_positions]
        marker_leaves = [marker_index[i] for i in marker_order_positions] if marker_index is not None else []

        self.orientation_state.update({
            "horizontal": horizontal,
            "cluster_order_positions": cluster_order_positions,
            "marker_order_positions": marker_order_positions,
            "cluster_leaves": cluster_leaves,
            "marker_leaves": marker_leaves,
        })

        if horizontal:
            if hasattr(g, 'ax_row_dendrogram') and g.ax_row_dendrogram is not None:
                g.ax_row_dendrogram.set_visible(False)
                g.ax_row_dendrogram.remove()
            if hasattr(g, 'ax_row_colors') and g.ax_row_colors is not None:
                g.ax_row_colors.set_visible(False)
                g.ax_row_colors.remove()

        dend_axis, _ = self._get_cluster_axes()
        self._draw_cutoff_line(dend_axis)

        ax_heatmap = g.ax_heatmap
        divider = make_axes_locatable(ax_heatmap)
        ax_hist = _append_hist_axis(self.adapter, divider, ax_heatmap)

        df = self.main_viewer.cell_table
        subset = list(self.ui_component.subset_selector.value)
        if subset:
            in_subset = df[subset_on].isin(subset)
            df = df[in_subset]
        hist_series = df[cluster[0]].value_counts(sort=False).reindex(cluster_leaves).fillna(0)

        _render_histogram(
            self.adapter,
            ax_hist,
            ax_heatmap,
            hist_series,
            cluster_leaves,
            marker_leaves,
            cluster[0],
        )

        plt.tight_layout()
        # NOTE: do NOT emit the figure here. Building/laying out the clustermap happens
        # while ``_refresh_plot`` has ``plt.ioff()`` active and runs outside any Output
        # display context, so ipympl does not auto-emit the canvas. ``_refresh_plot`` then
        # emits the interactive canvas exactly once via ``display(fig.canvas)``. Emitting
        # here (the old ``plt.show()``) produced a duplicate/blank ipympl canvas — the
        # heatmap-specific difference from the Chart histogram (which builds outside its
        # Output and emits once). See issue #108.

        self.display_row_colors_as_patches()

        g.fig.canvas.header_visible = False
        g.fig.canvas.mpl_connect('button_press_event', self._make_click_handler(g))

    def _cluster_color_axis(self):
        _, color_axis = self._get_cluster_axes()
        return color_axis

    def _clear_cluster_color_patches(self):
        self.highlight_patches = _remove_patches(getattr(self, 'highlight_patches', []))

        for attr in ('row_colors_patches', '_cluster_color_patches'):
            patches = getattr(self, attr, [])
            setattr(self, attr, _remove_patches(patches))

        if not hasattr(self, '_cluster_color_patches'):
            self._cluster_color_patches = []

    def _build_cluster_color_resolver(self, color_axis):
        cmap = None
        norm = None

        if color_axis is not None:
            collections = getattr(color_axis, 'collections', []) or []
            for artist in collections:
                if hasattr(artist, 'get_cmap'):
                    cmap = artist.get_cmap()
                    norm = getattr(artist, 'norm', None)
                    break

            if cmap is None and hasattr(color_axis, 'get_children'):
                for artist in color_axis.get_children():
                    if hasattr(artist, 'get_cmap'):
                        cmap = artist.get_cmap()
                        norm = getattr(artist, 'norm', None)
                        break

        meta_palette = getattr(self.data, 'meta_cluster_colors', {}) or {}
        cluster_palette = getattr(self.data, 'cluster_colors', {}) or {}

        def resolve(meta_value, cluster_label):
            if pd.isna(meta_value):
                return None

            # Always honor explicit registry colors first. This keeps colors stable
            # for user-added meta-clusters that may not exist in the cutoff palette.
            if meta_value in meta_palette:
                return meta_palette[meta_value]

            if cluster_label in cluster_palette:
                return cluster_palette[cluster_label]

            if cmap is not None:
                mapped_value = meta_value
                if norm is not None and hasattr(norm, '__call__'):
                    try:
                        mapped_value = norm(mapped_value)
                    except Exception:
                        pass
                try:
                    return cmap(mapped_value)
                except Exception:
                    pass

            return None

        return resolve

    def _render_cluster_color_patches(self):
        color_axis = self._cluster_color_axis()
        g = getattr(self.data, 'g', None)
        if color_axis is None or g is None:
            return

        if not hasattr(self, 'heatmap_data') or self.heatmap_data is None:
            return

        sorted_positions = list(self._cluster_order_positions())
        cluster_index = self._cluster_index_labels()
        if not sorted_positions or cluster_index is None:
            return

        color_column = 'meta_cluster_revised' if 'meta_cluster_revised' in self.heatmap_data.columns else 'meta_cluster'
        if color_column not in self.heatmap_data.columns:
            return

        resolver = self._build_cluster_color_resolver(color_axis)

        self._clear_cluster_color_patches()

        for position, cluster_pos in enumerate(sorted_positions):
            if cluster_pos >= len(cluster_index):
                continue
            label = cluster_index[cluster_pos]
            if label not in self.heatmap_data.index:
                continue
            meta_value = self.heatmap_data.loc[label, color_column]
            color = resolver(meta_value, label)
            if color is None:
                continue

            start = position
            end = position + 1
            patch = _draw_orientation_span(
                self.adapter,
                color_axis,
                start,
                end,
                color=color,
                alpha=1.0,
                zorder=10,
            )
            if patch is not None:
                self._cluster_color_patches.append(patch)

        color_axis.figure.canvas.draw_idle()

    def display_row_colors_as_patches(self):
        self._render_cluster_color_patches()

    def update_text_labels(self):
        if self.adapter.is_wide():
            return
        if not hasattr(self.data, 'g') or not hasattr(self.data.g, 'dendrogram_row') or self.data.g.dendrogram_row is None:
            return

        sorted_row_indices = self.data.g.dendrogram_row.reordered_ind
        color_column = 'meta_cluster_revised' if 'meta_cluster_revised' in self.heatmap_data.columns else 'meta_cluster'
        for text in self.data.g.ax_heatmap.texts:
            row_index = int(text.get_position()[1] * len(sorted_row_indices))
            text.set_text(str(self.heatmap_data[color_column].iloc[sorted_row_indices[::-1][row_index]]))
        self.data.g.fig.canvas.draw_idle()

    def apply_new_cutoff(self, *args):
        # Remember the user-set figure size (ipympl resize triangle) across the tree-cut
        # rebuild (issue #109). Capture from the OLD figure before _refresh_plot rebuilds it
        # (generate_heatmap reassigns self.data.g and closes the old figure).
        saved_size = self._capture_heatmap_scale()
        self._refresh_plot(restore_size=saved_size)
