"""Regression tests for issue #109 — heatmap should remember its scale after a tree cut.

The "scale" is the **figure size** the user sets by dragging the ipympl resize handle (the
triangle at the bottom-right corner of the canvas), not the toolbar zoom. Changing the
dendrogram cutoff rebuilds the heatmap as a brand-new ``sns.clustermap`` figure, which
otherwise reverts to the adapter's default ``figsize``.

The fix: ``apply_new_cutoff`` captures ``fig.get_size_inches()`` before the rebuild and
threads it through ``_refresh_plot(restore_size=...)`` → ``generate_heatmap(figsize_override
=...)``, which overrides ``clustermap_kwargs['figsize']`` so the grid (and ``tight_layout``)
is rebuilt at the preserved size. A fresh Plot / load uses the adapter default.

Imports the real ``ueler`` package directly, so it runs in the project conda environment.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ueler.viewer.plugin.heatmap_layers as heatmap_layers
from ueler.viewer.plugin.heatmap import HeatmapDisplay


class Issue109HeatmapScaleTests(unittest.TestCase):
    def setUp(self):
        self._was_interactive = plt.isinteractive()
        plt.ion()
        self._orig_display = heatmap_layers.display
        heatmap_layers.display = lambda _target: None

    def tearDown(self):
        heatmap_layers.display = self._orig_display
        if self._was_interactive:
            plt.ion()
        else:
            plt.ioff()

    def _make_heatmap(self, *, old_size):
        from ipywidgets import Output, VBox

        heatmap = HeatmapDisplay.__new__(HeatmapDisplay)
        heatmap.adapter = SimpleNamespace(is_wide=lambda: False)
        heatmap._restoring_plot_section = False
        out = Output()
        heatmap.plot_output = out
        heatmap.plot_section = VBox([out])
        # The "old" figure carries whatever size the user dragged it to.
        old_fig = plt.figure(figsize=old_size)
        self._old_fig = old_fig
        heatmap.data = SimpleNamespace(g=SimpleNamespace(fig=old_fig))

        self.figsize_overrides = []

        def _fake_generate(figsize_override=None):
            self.figsize_overrides.append(figsize_override)
            heatmap.data = SimpleNamespace(g=SimpleNamespace(fig=SimpleNamespace(canvas=None)))

        heatmap.generate_heatmap = _fake_generate
        return heatmap

    # --- capture ---------------------------------------------------------------

    def test_capture_scale_reads_figure_size(self):
        heatmap = self._make_heatmap(old_size=(9.0, 4.0))
        self.assertEqual(heatmap._capture_heatmap_scale(), (9.0, 4.0))

    def test_capture_scale_none_without_figure(self):
        heatmap = self._make_heatmap(old_size=(6.0, 6.0))
        heatmap.data = SimpleNamespace(g=None)
        self.assertIsNone(heatmap._capture_heatmap_scale())

    # --- threading through the cutoff path -------------------------------------

    def test_apply_new_cutoff_passes_captured_size(self):
        heatmap = self._make_heatmap(old_size=(7.5, 5.5))
        heatmap.apply_new_cutoff()
        self.assertEqual(self.figsize_overrides, [(7.5, 5.5)])

    def test_fresh_refresh_uses_no_override(self):
        heatmap = self._make_heatmap(old_size=(7.5, 5.5))
        heatmap._refresh_plot()  # fresh Plot path (no restore_size)
        self.assertEqual(self.figsize_overrides, [None])

    # --- the override actually reaches sns.clustermap --------------------------

    def test_generate_heatmap_overrides_clustermap_figsize(self):
        heatmap = HeatmapDisplay.__new__(HeatmapDisplay)
        heatmap.width = 8
        heatmap.height = 6
        heatmap.dendrogram = object()
        heatmap.heatmap_data = pd.DataFrame(index=["A", "B"])
        heatmap.orientation_state = {}
        heatmap.data = SimpleNamespace(g=None, dendrogram_cut=1.0, cluster_colors=None)
        view = pd.DataFrame({"m1": [0.1, 0.2]}, index=["A", "B"])
        heatmap.ui_component = SimpleNamespace(
            channel_selector=SimpleNamespace(value=("m1",)),
            high_level_cluster_dropdown=SimpleNamespace(value="cluster"),
            subset_on_dropdown=SimpleNamespace(value="fov"),
        )
        captured = {}

        def _fake_clustermap(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(cax=None, fig=SimpleNamespace(canvas=None))

        heatmap.adapter = SimpleNamespace(
            map_markers_to_axis=lambda v: ["m1"],
            slice_for_markers=lambda v, m: view,
            build_clustermap_kwargs=lambda *a, **k: {"figsize": (3.0, 3.0), "data": view},
        )
        heatmap._update_orientation_state = lambda: heatmap.orientation_state.update(
            {"view": view, "cluster_index": pd.Index(["A", "B"])}
        )
        heatmap._restore_cluster_assignments = lambda: None
        heatmap._sync_meta_cluster_registry = lambda *a, **k: None
        heatmap._heatmap_colormap_settings = lambda: {}
        heatmap._setup_layout = lambda *a, **k: None

        with patch.object(heatmap_layers, "cut_tree", return_value=np.array([[0], [1]])), \
             patch.object(heatmap_layers.sns, "clustermap", side_effect=_fake_clustermap):
            heatmap.generate_heatmap(figsize_override=(11.0, 7.0))

        self.assertEqual(captured.get("figsize"), (11.0, 7.0))

    def test_adapter_always_provides_figsize(self):
        # The override replaces this key, so it must exist in both orientations.
        from ueler.viewer.plugin.heatmap_adapter import HeatmapModeAdapter
        from scipy.cluster.hierarchy import linkage

        data = pd.DataFrame(np.random.rand(5, 3), index=[f"c{i}" for i in range(5)],
                            columns=["m1", "m2", "m3"])
        dend = linkage(data.values, method="average")
        labels = np.array([0, 0, 1, 1, 2])
        colors = pd.Series(["red", "red", "blue", "blue", "green"], index=data.index)
        for mode in ("vertical", "wide"):
            kw = HeatmapModeAdapter(mode=mode).build_clustermap_kwargs(
                data, dend, labels, 8, 6, colors, cmap="Reds", center=None)
            self.assertIn("figsize", kw, f"figsize missing in {mode} mode")


if __name__ == "__main__":
    unittest.main()
