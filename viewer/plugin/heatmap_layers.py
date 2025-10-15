"""Mixin layers for the heatmap plugin."""

from __future__ import annotations

import os
import pickle
from typing import Iterable, List, Sequence, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.cluster.hierarchy import cut_tree, linkage

from viewer.decorators import update_status_bar
from ipywidgets import HBox, HTML, Layout, Tab, VBox


NO_CLUSTER_SELECTED_MSG = "No cluster selected."


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


def _draw_orientation_span(adapter, axis, start, end, *, color, alpha, zorder):
    if axis is None:
        return None
    if adapter.is_wide():
        return axis.axvspan(start, end, facecolor=color, alpha=alpha, zorder=zorder)
    return axis.axhspan(start, end, facecolor=color, alpha=alpha, zorder=zorder)


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


def _apply_heatmap_tick_labels(adapter, ax_heatmap, cluster_leaves, marker_leaves, cluster_label):
    if adapter.is_wide():
        positions = np.arange(len(cluster_leaves))
        ax_heatmap.set_xticks(positions)
        ax_heatmap.set_xticklabels(cluster_leaves, rotation=45, ha="right", fontsize="small")
        if marker_leaves:
            ax_heatmap.set_yticks(np.arange(len(marker_leaves)))
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
        heatmap_ticks = np.arange(len(hist_series))

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

    def _reset_selection_cache(self):
        self._last_scatter_selection = None
        self._last_highlighted_clusters = None

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
        print(f"Data autosaved to {data_file}")

    def prepare_heatmap_data(self):
        df = self.main_viewer.cell_table
        cluster = [self.ui_component.high_level_cluster_dropdown.value]
        subset_on = self.ui_component.subset_on_dropdown.value

        channel = list(self.ui_component.channel_selector.value) + cluster

        print(f"Preparing heatmap data for channels: {channel}")
        print(f"Using cluster: {cluster}")

        subset = list(self.ui_component.subset_selector.value)
        if subset:
            in_subset = df[subset_on].isin(subset)
            df = df[in_subset]
        if df[cluster].nunique().values[0] > 300:
            print("The number of classes is too large to display. Please select a smaller number of classes.")
            return

        df_grouped = df[channel].groupby(cluster).median()
        df_grouped = (df_grouped - df_grouped.mean()) / df_grouped.std()

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
                print("If you intend to overwrite the existing column, please check the 'Overwrite' checkbox.")
                return
            print("Overwriting the existing column.")
            self.main_viewer.cell_table.drop(column_name, axis=1, inplace=True)
            if f"{column_name}_revised" in self.main_viewer.cell_table.columns:
                self.main_viewer.cell_table.drop(f"{column_name}_revised", axis=1, inplace=True)
            self.ui_component.overwrite_checkbox.value = False

        high_level_cluster = self.ui_component.high_level_cluster_dropdown.value

        if 'meta_cluster_revised' not in self.heatmap_data.columns:
            heatmap_data = self.heatmap_data[['meta_cluster']].copy()
        else:
            heatmap_data = self.heatmap_data[['meta_cluster', 'meta_cluster_revised']].copy()

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

        cluster_columns = self.main_viewer.cell_table.select_dtypes(include=['int', 'int64', 'object']).columns.tolist()
        print(cluster_columns)

        self.main_viewer.inform_plugins('on_cell_table_change')

        print(f"Cluster labels saved to column '{column_name}' in the cell table.")
        self.display_row_colors_as_patches()

    def on_cell_table_change(self):
        cluster_columns = self.main_viewer.cell_table.select_dtypes(include=['int', 'int64', 'object']).columns.tolist()
        old_cluster = self.ui_component.high_level_cluster_dropdown.value
        self.ui_component.high_level_cluster_dropdown.options = cluster_columns
        self.ui_component.high_level_cluster_dropdown.value = old_cluster

        old_subset = self.ui_component.subset_on_dropdown.value
        self.ui_component.subset_on_dropdown.options = cluster_columns
        self.ui_component.subset_on_dropdown.value = old_subset

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
        if hasattr(self.main_viewer, 'refresh_bottom_panel'):
            self.main_viewer.refresh_bottom_panel()
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
                print(f"Clicked cluster: {cluster_label}")
                cell_count = self.main_viewer.cell_table[
                    self.main_viewer.cell_table[self.ui_component.high_level_cluster_dropdown.value] == cluster_label
                ].shape[0]
                print(f"cell number: {cell_count}")
                self.update_linked()

            elif dend_axis is not None and event.inaxes == dend_axis:
                if self.ui_component.lock_cutoff_button.value:
                    print("Cutoff is locked. Please unlock to apply new cutoff.")
                    return
                value = self._dendrogram_coord_from_event(event)
                if value is not None:
                    self.data.dendrogram_cut = value
                    print(f"New dendrogram cutoff: {value}")
                    self._draw_cutoff_line(dend_axis)
                    self.apply_new_cutoff()

            elif color_axis is not None and event.inaxes == color_axis:
                coord = self._color_axis_coord_from_event(event)
                selected_idx = self._cluster_index_from_coord(coord)
                cluster_count, _ = self._cluster_and_marker_counts()
                if selected_idx is None or selected_idx < 0 or selected_idx >= cluster_count:
                    print("Clicked outside the color bar.")
                    return

                if event.button == MouseButton.RIGHT:
                    self.current_selection = []
                    self.data.current_clusters["index"].value = []
                    print("Cleared selection.")
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
                        print(f"Meta cluster IDs: {list(values)}")
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

        if self.ui_component.chart_checkbox.value:
            if self.main_viewer.SidePlots.chart_output.ui_component.y_axis_selector.value == "None":
                print("The response of a histogram is not implemented yet.")
            else:
                self.color_points_by_meta_cluster()
                self.highlight_scatter_plot()

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
            print("Chart plugin not available for coloring.")
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
            print("Cluster color palette not available.")
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
            print(NO_CLUSTER_SELECTED_MSG)
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

        print(f"Selected indices: {row_indices}")
        self.main_viewer.SidePlots.chart_output.color_points(row_indices)

    def display_cells(self):
        cluster_label = self._current_cluster_label()
        if cluster_label is None:
            print(NO_CLUSTER_SELECTED_MSG)
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
            print(NO_CLUSTER_SELECTED_MSG)
            return

        high_level_cluster = self.ui_component.high_level_cluster_dropdown.value
        cell_table = self.main_viewer.cell_table

        sub_table = cell_table.loc[cell_table[self.main_viewer.fov_key] == self.main_viewer.ui_component.image_selector.value, :]
        label_key = self.main_viewer.label_key
        mask_label = sub_table.loc[sub_table[high_level_cluster] == cluster_label, label_key].tolist()

        self.main_viewer.image_display.set_mask_ids(mask_name=self.main_viewer.mask_key, mask_ids=mask_label)
        print(f"{mask_label} in the main viewer.")
        print(f"Highlighted cells from cluster {cluster_label} of {high_level_cluster} in the main viewer.")

    def trace_cluster(self, *args):
        cell_id = self.main_viewer.image_display.selected_masks_label
        if cell_id is None:
            print("Please select a cell in the main viewer.")
            return
        cell_id = cell_id.pop()
        cell_id = cell_id[1]
        fov = self.main_viewer.ui_component.image_selector.value
        cluster_column = self.ui_component.high_level_cluster_dropdown.value
        label_key = self.main_viewer.label_key
        fov_key = self.main_viewer.fov_key
        cluster = self.main_viewer.cell_table.loc[
            (self.main_viewer.cell_table[label_key] == cell_id) & (self.main_viewer.cell_table[fov_key] == fov),
            cluster_column
        ].values[0]
        print(f"Cluster of the selected cell: {cluster}")

        if cluster_column != self.heatmap_data.index.name:
            print("Cluster column not found in the heatmap data.")
            return

        cluster_index = self._cluster_index_labels()
        order_positions = list(self._cluster_order_positions())
        if cluster_index is None or not order_positions:
            print("Heatmap ordering not available.")
            return

        try:
            base_position = cluster_index.get_loc(cluster)
        except KeyError:
            print("Cluster not found in the heatmap data.")
            return

        try:
            selection_index = order_positions.index(base_position)
        except ValueError:
            print("Cluster position not found in current ordering.")
            return

        if self.adapter.is_wide():
            row_ind = 0
            col_ind = selection_index
        else:
            row_ind = selection_index
            col_ind = 0

        self.heatmap_current_selection = (row_ind, col_ind)
        self.highlight_a_heatmap_grid(row_ind, col_ind)
        print(f"Found cluster at heatmap index: {selection_index}")

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
        print("Selected indices have changed.")

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
                except Exception as exc:
                    print(f"Error while plotting heatmap: {exc}")
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
            except Exception as exc:
                print(f"Error while plotting heatmap: {exc}")
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
        is_enabled = bool(indices)
        self.ui_component.cluster_id_text.disabled = not is_enabled
        self.ui_component.cluster_id_apply_button.disabled = not is_enabled

    def apply_new_cluster_id(self, *args):
        new_cluster_id = self.ui_component.cluster_id_text.value
        selected_indices = list(self.data.current_clusters["index"].value or [])
        if new_cluster_id is None:
            print("Please enter a valid cluster ID.")
            return
        if not selected_indices:
            print("No clusters selected to update.")
            return

        sorted_positions = self._cluster_order_positions()
        if not sorted_positions:
            print("No cluster ordering available.")
            return

        if 'meta_cluster_revised' not in self.heatmap_data.columns:
            self.heatmap_data['meta_cluster_revised'] = self.heatmap_data['meta_cluster']

        cluster_index = self._cluster_index_labels()
        if cluster_index is None:
            print("Cluster index not available.")
            return

        selection_positions = np.array(sorted_positions)[np.array(selected_indices)]
        cluster_labels = cluster_index.take(selection_positions)

        self.heatmap_data.loc[cluster_labels, 'meta_cluster_revised'] = new_cluster_id

        self.display_row_colors_as_patches()

        if not self.adapter.is_wide():
            self.update_text_labels()

        print(f"New cluster ID {new_cluster_id} applied to selected rows.")


class DisplayLayer:
    def __init__(self, *args, **kwargs):
        self._restoring_plot_section = False
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
                self.ui_component.horizontal_layout_checkbox
            ], layout=Layout(gap='8px')),
            HBox([self.ui_component.plot_button])
        ])

        edit = VBox([
            HBox([self.ui_component.lock_cutoff_button]),
            HBox([self.ui_component.cluster_id_text,
                self.ui_component.cluster_id_apply_button])
        ])

        link = VBox([
            HBox([self.ui_component.main_viewer_checkbox]),
            HBox([self.ui_component.chart_checkbox]),
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
            children=[setup, edit, trace, link, save],
            titles=('Setup', 'Assign', 'Trace', 'Linked plugins', 'Save')
        )

        self.controls_section = VBox([self.controls_tab], layout=Layout(width='100%', gap='8px'))
        self.plot_section = VBox([self.plot_output], layout=Layout(width='100%', flex='1 1 auto'))

        self.ui = VBox(
            [self.controls_section, self.plot_section],
            layout=Layout(width='100%', max_height='800px', gap='12px')
        )

        self._wide_notice = HTML(
            value="<b>Horizontal layout enabled.</b> Controls and plots live in the footer tabs.",
            layout=Layout(width='100%', padding='8px')
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

    def plot_heatmap(self, *args):
        self._reset_selection_cache()
        self.prepare_heatmap_data()
        self.dendrogram = self.generate_dendrogram()
        self.restore_vertical_canvas()

        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            self.generate_heatmap()

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
        self.restore_vertical_canvas()

        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            self.generate_heatmap()

        self.data.dendrogram_cut = cutoff

        print(f"New dendrogram cutoff: {cutoff}")
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
        if getattr(self, '_restoring_plot_section', False):
            return
        plot_section = getattr(self, 'plot_section', None)
        plot_output = getattr(self, 'plot_output', None)
        if plot_section is None or plot_output is None:
            return
        children = tuple(plot_section.children)
        if any(child is plot_output for child in children):
            return
        self._restoring_plot_section = True
        try:
            if children:
                plot_section.children = children + (plot_output,)
            else:
                plot_section.children = (plot_output,)
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
        g = getattr(self.data, 'g', None)
        fig = getattr(g, 'fig', None) if g is not None else None
        canvas = getattr(fig, 'canvas', None) if fig is not None else None
        if canvas is not None and hasattr(canvas, 'draw_idle'):
            try:
                canvas.draw_idle()
            except Exception:
                pass

    @update_status_bar
    def generate_heatmap(self):
        markers = list(self.ui_component.channel_selector.value)
        if not markers:
            print("No markers selected for display.")
            return

        self._update_orientation_state()
        heatmap_view = self.orientation_state.get("view")
        if heatmap_view is None or heatmap_view.empty:
            print("Heatmap view is empty.")
            return

        cutoff = self.data.dendrogram_cut
        if self.dendrogram is None:
            print("Dendrogram not available for plotting.")
            return
        meta_cluster_labels = cut_tree(self.dendrogram, height=cutoff).flatten()

        cluster = [self.ui_component.high_level_cluster_dropdown.value]
        subset_on = self.ui_component.subset_on_dropdown.value

        base_index = self.orientation_state.get("cluster_index")
        if base_index is None:
            print("Cluster index not available.")
            return
        self.heatmap_data = self.heatmap_data.reindex(base_index)
        self.heatmap_data['meta_cluster'] = meta_cluster_labels

        num_clusters = len(np.unique(meta_cluster_labels))
        cluster_palette = sns.color_palette('husl', num_clusters)
        cluster_colors = dict(zip(np.unique(meta_cluster_labels), cluster_palette))

        cluster_labels = pd.Series(meta_cluster_labels, index=base_index)
        cluster_colors_series = cluster_labels.map(cluster_colors)

        self.data.cluster_colors = cluster_colors_series.to_dict()
        self.data.meta_cluster_colors = dict(cluster_colors)

        sns.set_context('notebook')

        marker_axis_labels = list(self.adapter.map_markers_to_axis(heatmap_view))

        requested_markers = [m for m in markers if m in marker_axis_labels]
        missing_markers = [m for m in markers if m not in marker_axis_labels]
        if missing_markers:
            print(f"Warning: markers not found in heatmap data and will be skipped: {missing_markers}")

        if not requested_markers:
            print("No requested markers are available in the heatmap data.")
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
        )

        g = sns.clustermap(**clustermap_kwargs)

        self.data.g = g

        if hasattr(g, 'cax') and g.cax is not None:
            g.cax.set_visible(False)
            g.cax.remove()

        self._setup_layout(g, cluster, subset_on)

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
        plt.show()

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

            if meta_value in meta_palette:
                return meta_palette[meta_value]

            if cluster_label in cluster_palette:
                return cluster_palette[cluster_label]

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
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            self.generate_heatmap()
