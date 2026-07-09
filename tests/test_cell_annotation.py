"""Unit tests for CellAnnotationPlugin — save/load flow and tree rendering."""

from __future__ import annotations

import sys
import tempfile
import unittest
from types import SimpleNamespace

import sys as _sys
# test_annotation_palettes installs a dask stub with __spec__=None which
# makes anndata's find_spec("dask") raise ValueError.  Remove any such
# stub before importing anndata so it can load real dask from disk.
if _sys.modules.get("dask") is not None and getattr(_sys.modules["dask"], "__spec__", None) is None:
    for _k in [k for k in _sys.modules if k == "dask" or k.startswith("dask.")]:
        _sys.modules.pop(_k, None)

import anndata  # noqa: F401 — must be imported before tests.bootstrap runs initialize()

import tests.bootstrap  # noqa: F401

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Minimal stubs used by CellAnnotationPlugin
# ---------------------------------------------------------------------------

class _FakeHeatmapPlugin:
    """Minimal heatmap plugin stub implementing HeatmapStateProvider."""

    def __init__(self, n_clusters=5, n_markers=3):
        self._n_clusters = n_clusters
        self._n_markers = n_markers
        self._import_calls: list[anndata.AnnData] = []

    def export_heatmap_state(self, *, include_raw_medians: bool = True):
        X = np.ones((self._n_clusters, self._n_markers), dtype="float32")
        obs = pd.DataFrame(
            {"meta_cluster": list(range(self._n_clusters))},
            index=[str(i) for i in range(self._n_clusters)],
        )
        var = pd.DataFrame(index=["CD4", "CD8", "CD3"][: self._n_markers])
        adata = anndata.AnnData(X=X, obs=obs, var=var)
        adata.uns["palette"] = {"colors": {}, "names": {}, "next_id": 0}
        adata.uns["ui"] = {}
        return adata

    def import_heatmap_state(self, adata: anndata.AnnData) -> None:
        self._import_calls.append(adata)


class _FakeFlowsomPlugin:
    """Minimal FlowSOM plugin stub."""

    def __init__(self):
        self._import_calls: list[dict] = []

    def export_flowsom_params(self) -> dict:
        return {"xdim": 10, "ydim": 10, "rlen": 10, "seed": 42}

    def import_flowsom_params(self, params: dict) -> None:
        self._import_calls.append(params)


def _make_plugin(root: str):
    """Instantiate CellAnnotationPlugin with stubs wired in."""
    from ueler.viewer.plugin.cell_annotation import CellAnnotationPlugin

    heatmap = _FakeHeatmapPlugin()
    flowsom = _FakeFlowsomPlugin()

    # Build a minimal main_viewer-like namespace
    viewer = SimpleNamespace(
        base_folder=root,
        SidePlots=SimpleNamespace(
            heatmap_output=heatmap,
            run_flowsom_output=flowsom,
        ),
    )

    plugin = CellAnnotationPlugin.__new__(CellAnnotationPlugin)
    # Bypass __init__; manually set the attributes that matter for tests
    plugin.main_viewer = viewer
    plugin.SidePlots_id = "cell_annotation_output"
    plugin.displayed_name = "Cell Annotation"
    plugin._store = None
    plugin._heatmap_plugin = None
    plugin._flowsom_plugin = None
    plugin.status_label = SimpleNamespace(value="")

    # Wire the store and plugin refs (simulating after_all_plugins_loaded)
    from ueler.viewer.checkpoint_store import CheckpointStore
    plugin._store = CheckpointStore(root)
    plugin._heatmap_plugin = heatmap
    plugin._flowsom_plugin = flowsom

    return plugin


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCellAnnotationSaveLoad(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_save_checkpoint_returns_id(self):
        plugin = _make_plugin(self.root)
        ckpt_id = plugin.save_checkpoint(step_id="1", description="test", op="initial")
        self.assertIsNotNone(ckpt_id)
        self.assertIsInstance(ckpt_id, str)

    def test_save_writes_h5ad_file(self):
        plugin = _make_plugin(self.root)
        plugin.save_checkpoint(step_id="1", description="test")
        entries = plugin._store.list_checkpoints()
        self.assertEqual(len(entries), 1)
        h5ad_path = plugin._store._dataset_dir / entries[0]["path"]
        self.assertTrue(h5ad_path.exists())

    def test_load_checkpoint_calls_import(self):
        plugin = _make_plugin(self.root)
        ckpt_id = plugin.save_checkpoint(step_id="1", description="test")
        ok = plugin.load_checkpoint(ckpt_id)
        self.assertTrue(ok)
        self.assertEqual(len(plugin._heatmap_plugin._import_calls), 1)

    def test_load_checkpoint_restores_correct_data(self):
        plugin = _make_plugin(self.root)
        ckpt_id = plugin.save_checkpoint(step_id="2", description="round-trip")
        plugin.load_checkpoint(ckpt_id)
        loaded_adata = plugin._heatmap_plugin._import_calls[-1]
        self.assertEqual(loaded_adata.n_obs, 5)
        self.assertEqual(loaded_adata.n_vars, 3)

    def test_load_restores_flowsom_params(self):
        plugin = _make_plugin(self.root)
        ckpt_id = plugin.save_checkpoint(step_id="1", description="fs test")
        plugin.load_checkpoint(ckpt_id)
        self.assertEqual(len(plugin._flowsom_plugin._import_calls), 1)
        self.assertEqual(plugin._flowsom_plugin._import_calls[0]["xdim"], 10)

    def test_parent_id_recorded_in_manifest(self):
        plugin = _make_plugin(self.root)
        parent_id = plugin.save_checkpoint(step_id="1", description="root")
        plugin.save_checkpoint(step_id="2", description="child", parent_id=parent_id)
        entries = plugin._store.list_checkpoints()
        child = next(e for e in entries if e["step_id"] == "2")
        self.assertEqual(child["parent_id"], parent_id)

    def test_load_missing_checkpoint_returns_false(self):
        plugin = _make_plugin(self.root)
        ok = plugin.load_checkpoint("nonexistent-id")
        self.assertFalse(ok)

    def test_save_without_heatmap_returns_none(self):
        plugin = _make_plugin(self.root)
        plugin._heatmap_plugin = None
        result = plugin.save_checkpoint(step_id="1", description="no heatmap")
        self.assertIsNone(result)

    def test_save_without_store_returns_none(self):
        plugin = _make_plugin(self.root)
        plugin._store = None
        result = plugin.save_checkpoint(step_id="1", description="no store")
        self.assertIsNone(result)

    def test_tree_nodes_populated_after_save(self):
        from ueler.viewer.plugin.cell_annotation import CheckpointTreeWidget
        plugin = _make_plugin(self.root)
        # Attach a tree widget to the plugin
        plugin.tree_widget = CheckpointTreeWidget()
        plugin.save_checkpoint(step_id="1", description="tree test")
        plugin._refresh_tree()
        self.assertEqual(len(plugin.tree_widget.nodes), 1)
        self.assertEqual(plugin.tree_widget.nodes[0]["step_id"], "1")

    def test_delete_removes_from_tree(self):
        from ueler.viewer.plugin.cell_annotation import CheckpointTreeWidget
        plugin = _make_plugin(self.root)
        plugin.tree_widget = CheckpointTreeWidget()
        ckpt_id = plugin.save_checkpoint(step_id="1", description="to delete")
        plugin._refresh_tree()
        self.assertEqual(len(plugin.tree_widget.nodes), 1)
        plugin._store.delete_checkpoint(ckpt_id)
        plugin._refresh_tree()
        self.assertEqual(len(plugin.tree_widget.nodes), 0)


class TestAfterAllPluginsLoaded(unittest.TestCase):
    """Regression: after_all_plugins_loaded must not raise even without a state file."""

    def test_no_crash_when_no_state_file(self):
        """CellAnnotationPlugin.after_all_plugins_loaded() must not crash.

        Historically, calling super().after_all_plugins_loaded() triggered
        PluginBase.load_widget_states() which did vars(self.ui_component) —
        CellAnnotationPlugin has no ui_component, causing AttributeError.
        """
        plugin = _make_plugin(self.root if hasattr(self, "root") else tempfile.mkdtemp())
        try:
            plugin.after_all_plugins_loaded()
        except AttributeError as exc:
            self.fail(f"after_all_plugins_loaded raised AttributeError: {exc}")

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()


class TestCheckpointTreeWidget(unittest.TestCase):
    """Verify the CheckpointTreeWidget traitlets are settable in the fallback env."""

    def test_nodes_traitlet(self):
        from ueler.viewer.plugin.cell_annotation import CheckpointTreeWidget
        w = CheckpointTreeWidget()
        w.nodes = [{"id": "abc", "step_id": "1"}]
        self.assertEqual(len(w.nodes), 1)

    def test_selected_id_traitlet(self):
        from ueler.viewer.plugin.cell_annotation import CheckpointTreeWidget
        w = CheckpointTreeWidget()
        w.selected_id = "abc-123"
        self.assertEqual(w.selected_id, "abc-123")

    def test_action_requested_traitlet(self):
        from ueler.viewer.plugin.cell_annotation import CheckpointTreeWidget
        w = CheckpointTreeWidget()
        w.action_requested = "load:abc-123"
        self.assertEqual(w.action_requested, "load:abc-123")


if __name__ == "__main__":
    unittest.main()
