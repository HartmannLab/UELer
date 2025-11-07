"""Orientation helper for the heatmap plugin."""


class HeatmapModeAdapter:
    """Translate logical heatmap semantics into orientation-specific behavior."""

    def __init__(self, mode: str = "vertical") -> None:
        self.mode = mode

    def is_wide(self) -> bool:
        return self.mode == "wide"

    def map_clusters_to_axis(self, dataframe):
        """Return the axis containing cluster labels for the active orientation."""
        if dataframe is None:
            return None
        return dataframe.index if not self.is_wide() else dataframe.columns

    def map_markers_to_axis(self, dataframe):
        """Return the axis containing marker labels for the active orientation."""
        if dataframe is None:
            return None
        return dataframe.columns if not self.is_wide() else dataframe.index

    def slice_for_markers(self, dataframe, markers):
        """Return a view filtered to the requested markers for the active orientation."""
        if dataframe is None:
            return None
        if self.is_wide():
            return dataframe.loc[markers, :]
        return dataframe.loc[:, markers]

    def transform_coordinates(self, row, col):
        """Adapt heatmap coordinates between logical and rendered orientation."""
        return (col, row) if self.is_wide() else (row, col)

    def dock_controls(self, ui):
        """Position controls according to the active orientation."""
        if ui is None:
            return
        if self.is_wide() and hasattr(ui, "place_footer"):
            ui.place_footer()
        elif hasattr(ui, "place_side"):
            ui.place_side()

    def build_clustermap_kwargs(
        self,
        plot_data,
        dendrogram,
        meta_cluster_labels,
        width,
        height,
        cluster_colors_series,
    ):
        """Return keyword arguments for seaborn.clustermap respecting orientation."""
        if self.is_wide():
            return {
                "data": plot_data,
                "row_cluster": False,
                "col_cluster": True,
                "col_linkage": dendrogram,
                "dendrogram_ratio": (0, 0.2),
                "cmap": "Purples",
                "figsize": (len(meta_cluster_labels) * 0.3, height * 0.9),
                "col_colors": cluster_colors_series,
            }

        return {
            "data": plot_data,
            "row_cluster": True,
            "col_cluster": False,
            "row_linkage": dendrogram,
            "dendrogram_ratio": (0.2, 0),
            "cmap": "Purples",
            "figsize": (width * 0.9, len(meta_cluster_labels) * 0.3),
            "row_colors": cluster_colors_series,
        }
