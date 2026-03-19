import unittest

import tests.bootstrap  # noqa: F401

from ueler.viewer.plugin.heatmap_adapter import HeatmapModeAdapter


class HeatmapModeAdapterTests(unittest.TestCase):
    def test_wide_layout_figsize_is_clamped_to_plugin_width(self):
        adapter = HeatmapModeAdapter(mode="wide")

        kwargs = adapter.build_clustermap_kwargs(
            plot_data=[[1.0]],
            dendrogram=object(),
            meta_cluster_labels=list(range(100)),
            width=6,
            height=8,
            cluster_colors_series=["#000000"],
        )

        self.assertEqual(kwargs["figsize"][0], 5.4)
        self.assertEqual(kwargs["figsize"][1], 7.2)

    def test_wide_layout_uses_data_driven_width_when_under_limit(self):
        adapter = HeatmapModeAdapter(mode="wide")

        kwargs = adapter.build_clustermap_kwargs(
            plot_data=[[1.0]],
            dendrogram=object(),
            meta_cluster_labels=list(range(10)),
            width=6,
            height=8,
            cluster_colors_series=["#000000"],
        )

        self.assertEqual(kwargs["figsize"][0], 3.0)

    def test_vertical_layout_keeps_existing_figsize_logic(self):
        adapter = HeatmapModeAdapter(mode="vertical")

        kwargs = adapter.build_clustermap_kwargs(
            plot_data=[[1.0]],
            dendrogram=object(),
            meta_cluster_labels=list(range(10)),
            width=6,
            height=8,
            cluster_colors_series=["#000000"],
        )

        self.assertEqual(kwargs["figsize"][0], 5.4)
        self.assertEqual(kwargs["figsize"][1], 3.0)


if __name__ == "__main__":
    unittest.main()
