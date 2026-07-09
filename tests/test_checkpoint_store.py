"""Unit tests for CheckpointStore — atomic read/write, manifest management."""

from __future__ import annotations

import json
import os
import sys
import unittest

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


class TestCheckpointStore(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_adata(self, n_clusters=5, n_markers=3):
        X = np.random.rand(n_clusters, n_markers).astype("float32")
        obs = pd.DataFrame({"meta_cluster": list(range(n_clusters))}, index=[str(i) for i in range(n_clusters)])
        var = pd.DataFrame(index=[f"marker_{i}" for i in range(n_markers)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)
        adata.uns["palette"] = {
            "colors": {"1": "#ff0000"},
            "names": {"1": "CD4 T"},
            "next_id": 2,
        }
        return adata

    def _make_store(self):
        from ueler.viewer.checkpoint_store import CheckpointStore
        return CheckpointStore(self.root)

    # ------------------------------------------------------------------
    # Basic round-trip
    # ------------------------------------------------------------------

    def test_write_and_read_roundtrip(self):
        store = self._make_store()
        adata = self._make_adata()
        ckpt_id = store.write_checkpoint(
            adata, parent_id=None, op="initial", step_id="1", description="first"
        )
        self.assertIsInstance(ckpt_id, str)
        loaded = store.read_checkpoint(ckpt_id)
        np.testing.assert_array_almost_equal(loaded.X, adata.X)
        self.assertEqual(list(loaded.var_names), list(adata.var_names))
        self.assertEqual(list(loaded.obs_names), list(adata.obs_names))

    def test_uns_preserved_in_roundtrip(self):
        store = self._make_store()
        adata = self._make_adata()
        ckpt_id = store.write_checkpoint(
            adata, parent_id=None, op="initial", step_id="1", description="uns test"
        )
        loaded = store.read_checkpoint(ckpt_id)
        self.assertIn("palette", loaded.uns)
        self.assertEqual(loaded.uns["palette"]["colors"]["1"], "#ff0000")

    def test_checkpoint_uns_stamped(self):
        store = self._make_store()
        adata = self._make_adata()
        ckpt_id = store.write_checkpoint(
            adata, parent_id=None, op="recluster", step_id="2", description="step2"
        )
        loaded = store.read_checkpoint(ckpt_id)
        cp = loaded.uns["checkpoint"]
        self.assertEqual(cp["id"], ckpt_id)
        self.assertEqual(cp["op"], "recluster")
        self.assertEqual(cp["step_id"], "2")
        self.assertEqual(cp["description"], "step2")
        # None parent_id is serialised as "" in HDF5
        self.assertFalse(cp["parent_id"])

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def test_manifest_created_on_write(self):
        store = self._make_store()
        adata = self._make_adata()
        store.write_checkpoint(adata, parent_id=None, op="initial", step_id="1", description="d")
        manifest = store.load_manifest()
        self.assertEqual(len(manifest["checkpoints"]), 1)

    def test_manifest_accumulates_entries(self):
        store = self._make_store()
        store.write_checkpoint(self._make_adata(), parent_id=None, op="initial", step_id="1", description="a")
        store.write_checkpoint(self._make_adata(), parent_id=None, op="recluster", step_id="2", description="b")
        self.assertEqual(len(store.list_checkpoints()), 2)

    def test_parent_child_relationship_in_manifest(self):
        store = self._make_store()
        parent_id = store.write_checkpoint(
            self._make_adata(), parent_id=None, op="initial", step_id="1", description="root"
        )
        child_id = store.write_checkpoint(
            self._make_adata(), parent_id=parent_id, op="subset", step_id="2", description="child"
        )
        entries = store.list_checkpoints()
        child_entry = next(e for e in entries if e["id"] == child_id)
        self.assertEqual(child_entry["parent_id"], parent_id)

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_read_missing_raises(self):
        store = self._make_store()
        with self.assertRaises(FileNotFoundError):
            store.read_checkpoint("nonexistent-id")

    def test_read_after_delete_raises(self):
        store = self._make_store()
        ckpt_id = store.write_checkpoint(
            self._make_adata(), parent_id=None, op="initial", step_id="1", description="d"
        )
        store.delete_checkpoint(ckpt_id)
        with self.assertRaises(FileNotFoundError):
            store.read_checkpoint(ckpt_id)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def test_delete_removes_manifest_entry(self):
        store = self._make_store()
        ckpt_id = store.write_checkpoint(
            self._make_adata(), parent_id=None, op="initial", step_id="1", description="d"
        )
        store.delete_checkpoint(ckpt_id)
        self.assertEqual(store.list_checkpoints(), [])

    def test_delete_removes_h5ad_file(self):
        store = self._make_store()
        ckpt_id = store.write_checkpoint(
            self._make_adata(), parent_id=None, op="initial", step_id="1", description="d"
        )
        # Find the file path
        manifest = store.load_manifest()
        entry = manifest["checkpoints"][0]
        h5ad_path = store._dataset_dir / entry["path"]
        self.assertTrue(h5ad_path.exists())
        store.delete_checkpoint(ckpt_id)
        self.assertFalse(h5ad_path.exists())

    # ------------------------------------------------------------------
    # Atomic write
    # ------------------------------------------------------------------

    def test_no_partial_file_after_write(self):
        store = self._make_store()
        store.write_checkpoint(
            self._make_adata(), parent_id=None, op="initial", step_id="1", description="d"
        )
        leftover = list(store._checkpoints_dir.glob("*.partial"))
        self.assertEqual(leftover, [])

    # ------------------------------------------------------------------
    # Empty manifest
    # ------------------------------------------------------------------

    def test_load_manifest_returns_empty_if_missing(self):
        store = self._make_store()
        manifest = store.load_manifest()
        self.assertEqual(manifest["checkpoints"], [])

    def test_load_manifest_returns_empty_if_corrupt(self):
        store = self._make_store()
        store._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        store._manifest_path.write_text("NOT JSON {{{{", encoding="utf-8")
        manifest = store.load_manifest()
        self.assertEqual(manifest["checkpoints"], [])

    # ------------------------------------------------------------------
    # list_checkpoints
    # ------------------------------------------------------------------

    def test_list_checkpoints_includes_metadata(self):
        store = self._make_store()
        store.write_checkpoint(
            self._make_adata(n_clusters=7, n_markers=4),
            parent_id=None, op="initial", step_id="s1", description="my desc"
        )
        entries = store.list_checkpoints()
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["n_clusters"], 7)
        self.assertEqual(e["n_markers"], 4)
        self.assertEqual(e["step_id"], "s1")
        self.assertEqual(e["description"], "my desc")
        self.assertEqual(e["op"], "initial")


if __name__ == "__main__":
    unittest.main()
