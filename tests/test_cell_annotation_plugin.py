"""Unit tests for Cell Annotation scaffolding."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ueler.viewer.plugin.cell_annotation.manifest import Manifest
from ueler.viewer.plugin.cell_annotation.plugin import CellAnnotationPlugin, _flag_enabled
from ueler.viewer.plugin.cell_annotation.selection_spec import MaterializedSelectionSpec
from ueler.viewer.plugin.cell_annotation.store import (
    STORE_SUBDIRS,
    DatasetStore,
    _dataset_id,
    atomic_replace,
    atomic_write_json,
)


class TestDatasetStore(unittest.TestCase):
    def test_dataset_id_is_stable_and_short(self):
        with tempfile.TemporaryDirectory() as dataset_root:
            value = _dataset_id(dataset_root)
            self.assertEqual(value, _dataset_id(dataset_root))
            self.assertEqual(len(value), 16)
            self.assertTrue(all(char in "0123456789abcdef" for char in value))

    def test_ensure_dirs_creates_expected_tree(self):
        with tempfile.TemporaryDirectory() as dataset_root:
            store = DatasetStore(dataset_root)
            store.ensure_dirs()
            self.assertTrue(store.store_path.is_dir())
            for subdir in STORE_SUBDIRS:
                self.assertTrue((store.store_path / subdir).is_dir())

    def test_subdir_returns_child_path(self):
        with tempfile.TemporaryDirectory() as dataset_root:
            store = DatasetStore(dataset_root)
            self.assertEqual(store.subdir("checkpoints"), store.store_path / "checkpoints")


class TestAtomicHelpers(unittest.TestCase):
    def test_atomic_write_json_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "manifest.json"
            atomic_write_json(target, {"version": 1})
            atomic_write_json(target, {"version": 2})
            self.assertEqual(json.loads(target.read_text()), {"version": 2})

    def test_atomic_write_json_removes_partial_file_on_failure(self):
        class Unserializable:
            pass

        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "manifest.json"
            atomic_write_json(target, {"version": 1})
            with self.assertRaises(TypeError):
                atomic_write_json(target, {"bad": Unserializable()})
            self.assertEqual(json.loads(target.read_text()), {"version": 1})
            self.assertEqual(list(Path(root).glob(".*.tmp*")), [])

    def test_atomic_replace_moves_file_into_place(self):
        with tempfile.TemporaryDirectory() as root:
            src = Path(root) / "src.json"
            dst = Path(root) / "deep" / "dst.json"
            src.write_text('{"ok": true}')
            atomic_replace(src, dst)
            self.assertFalse(src.exists())
            self.assertEqual(json.loads(dst.read_text()), {"ok": True})


class TestManifest(unittest.TestCase):
    def test_manifest_load_and_save_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = Manifest(root)
            manifest.data["checkpoints"] = []
            manifest.save_atomic()
            self.assertEqual(Manifest(root).load(), {"checkpoints": []})

    def test_manifest_rebuild_stub_resets_to_empty_dict(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = Manifest(root)
            manifest.data["checkpoints"] = ["stale"]
            self.assertEqual(manifest.rebuild_from_disk(), {})
            self.assertEqual(manifest.data, {})


class TestSelectionSpec(unittest.TestCase):
    def test_subset_and_union(self):
        parent = MaterializedSelectionSpec.from_cells("dataset_a", [("fov1", 1), ("fov1", 2)])
        child = MaterializedSelectionSpec.from_cells("dataset_a", [("fov1", 2)])
        sibling = MaterializedSelectionSpec.from_cells("dataset_a", [("fov2", 9)])

        self.assertTrue(child.subset_of(parent))
        self.assertEqual(child.union(sibling).cardinality(), 2)

    def test_union_requires_matching_dataset(self):
        left = MaterializedSelectionSpec.from_cells("dataset_a", [("fov1", 1)])
        right = MaterializedSelectionSpec.from_cells("dataset_b", [("fov1", 1)])

        with self.assertRaises(ValueError):
            left.union(right)


class TestFeatureFlagAndPluginLifecycle(unittest.TestCase):
    def test_flag_defaults_to_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_flag_enabled())

    def test_truthy_feature_flag_values_enable_plugin(self):
        for value in ("1", "true", "TRUE", "yes"):
            with self.subTest(value=value), patch.dict(os.environ, {"ENABLE_CELL_ANNOTATION": value}, clear=True):
                self.assertTrue(_flag_enabled())

    def test_plugin_lifecycle_initializes_store_manifest_and_providers(self):
        viewer = MagicMock()
        plugin = CellAnnotationPlugin(viewer)
        self.assertIsNone(plugin.store)
        self.assertIsNone(plugin.manifest)

        heatmap = MagicMock()
        flowsom = MagicMock()
        plugin.register_heatmap(heatmap)
        plugin.register_flowsom(flowsom)
        self.assertIs(plugin.heatmap_provider, heatmap)
        self.assertIs(plugin.flowsom_provider, flowsom)

        with tempfile.TemporaryDirectory() as dataset_root:
            plugin.on_dataset_opened(dataset_root)
            self.assertIsNotNone(plugin.store)
            self.assertIsNotNone(plugin.manifest)
            for subdir in STORE_SUBDIRS:
                self.assertTrue((plugin.store.store_path / subdir).is_dir())
            plugin.on_dataset_closed()

        self.assertIsNone(plugin.store)
        self.assertIsNone(plugin.manifest)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
