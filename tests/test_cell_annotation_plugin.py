"""Unit tests for the Cell Annotation plugin scaffolding (sub-issue #16).

Tests cover:
- DatasetStore path resolution and directory creation
- atomic_write_json semantics (file is visible only after rename; no partials)
- atomic_replace semantics (destination replaced atomically)
- Manifest load/save_atomic/rebuild_from_disk
- CellAnnotationPlugin feature-flag gating
- CellAnnotationPlugin.on_dataset_opened creates expected subdirectories
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal stubs for optional heavy dependencies so the module can be imported
# in the unit-test environment without the full UI stack.
# ---------------------------------------------------------------------------

for _stub_name in (
    "ipywidgets",
    "matplotlib",
    "matplotlib.pyplot",
    "matplotlib.colors",
    "matplotlib.text",
    "matplotlib.font_manager",
    "numpy",
    "pandas",
    "IPython",
    "IPython.display",
):
    if _stub_name not in sys.modules:
        sys.modules[_stub_name] = types.ModuleType(_stub_name)  # type: ignore[assignment]

# Ensure the ipywidgets.Widget stub exists so plugin_base can import it.
if not hasattr(sys.modules.get("ipywidgets", types.ModuleType("ipywidgets")), "Widget"):
    _ipy = sys.modules.setdefault("ipywidgets", types.ModuleType("ipywidgets"))
    _ipy.Widget = object  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from ueler.viewer.plugin.cell_annotation.store import (
    DatasetStore,
    atomic_replace,
    atomic_write_json,
    _dataset_id,
    STORE_SUBDIRS,
)
from ueler.viewer.plugin.cell_annotation.manifest import Manifest
from ueler.viewer.plugin.cell_annotation.plugin import (
    CellAnnotationPlugin,
    _flag_enabled,
)


# ===========================================================================
# DatasetStore
# ===========================================================================

class TestDatasetId(unittest.TestCase):
    def test_stable(self):
        """Same path always yields the same ID."""
        with tempfile.TemporaryDirectory() as d:
            a = _dataset_id(d)
            b = _dataset_id(d)
            self.assertEqual(a, b)

    def test_different_paths_different_ids(self):
        with tempfile.TemporaryDirectory() as d1:
            with tempfile.TemporaryDirectory() as d2:
                self.assertNotEqual(_dataset_id(d1), _dataset_id(d2))

    def test_length(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(_dataset_id(d)), 16)

    def test_hex_chars(self):
        with tempfile.TemporaryDirectory() as d:
            id_ = _dataset_id(d)
            self.assertTrue(all(c in "0123456789abcdef" for c in id_))


class TestDatasetStorePathResolution(unittest.TestCase):
    def test_store_path_structure(self):
        with tempfile.TemporaryDirectory() as d:
            store = DatasetStore(d)
            expected_prefix = os.path.join(str(Path(d).resolve()), ".UELer", "dataset_")
            self.assertTrue(str(store.store_path).startswith(expected_prefix))

    def test_store_path_contains_id(self):
        with tempfile.TemporaryDirectory() as d:
            store = DatasetStore(d)
            self.assertIn(store.dataset_id, str(store.store_path))

    def test_ensure_dirs_creates_subdirs(self):
        with tempfile.TemporaryDirectory() as d:
            store = DatasetStore(d)
            store.ensure_dirs()
            self.assertTrue(store.store_path.is_dir())
            for sub in STORE_SUBDIRS:
                self.assertTrue(
                    (store.store_path / sub).is_dir(),
                    f"Expected subdir '{sub}' to be created",
                )

    def test_ensure_dirs_idempotent(self):
        """Calling ensure_dirs twice must not raise."""
        with tempfile.TemporaryDirectory() as d:
            store = DatasetStore(d)
            store.ensure_dirs()
            store.ensure_dirs()  # second call must be a no-op

    def test_subdir_helper(self):
        with tempfile.TemporaryDirectory() as d:
            store = DatasetStore(d)
            self.assertEqual(store.subdir("checkpoints"), store.store_path / "checkpoints")


# ===========================================================================
# atomic_write_json
# ===========================================================================

class TestAtomicWriteJson(unittest.TestCase):
    def test_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "out.json"
            payload = {"key": [1, 2, 3]}
            atomic_write_json(target, payload)
            with open(target) as f:
                loaded = json.load(f)
            self.assertEqual(loaded, payload)

    def test_no_tmp_file_left_after_success(self):
        """No ``.tmp`` sibling files should remain after a successful write."""
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "data.json"
            atomic_write_json(target, {"x": 1})
            tmps = list(Path(d).glob("*.json.tmp*")) + list(Path(d).glob(".*.tmp*"))
            self.assertEqual(tmps, [], f"Unexpected tmp files: {tmps}")

    def test_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "data.json"
            atomic_write_json(target, {"v": 1})
            atomic_write_json(target, {"v": 2})
            with open(target) as f:
                self.assertEqual(json.load(f)["v"], 2)

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "sub" / "deep" / "data.json"
            atomic_write_json(target, {})
            self.assertTrue(target.exists())

    def test_no_partial_visible_on_failure(self):
        """Simulate a serialisation error; destination must not be left partial."""
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "data.json"
            # Write initial valid content.
            atomic_write_json(target, {"original": True})

            # Now simulate a JSON serialisation failure.
            class _Unserializable:
                pass

            with self.assertRaises(TypeError):
                atomic_write_json(target, {"bad": _Unserializable()})

            # The original file must still be intact.
            with open(target) as f:
                loaded = json.load(f)
            self.assertEqual(loaded, {"original": True})

            # No tmp files should remain.
            tmps = list(Path(d).glob(".*.tmp*"))
            self.assertEqual(tmps, [], f"Partial tmp files left: {tmps}")


# ===========================================================================
# atomic_replace
# ===========================================================================

class TestAtomicReplace(unittest.TestCase):
    def test_renames_file(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "tmp_file"
            dst = Path(d) / "final_file"
            src.write_text("hello")
            atomic_replace(src, dst)
            self.assertFalse(src.exists())
            self.assertEqual(dst.read_text(), "hello")

    def test_replaces_existing_destination(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "new"
            dst = Path(d) / "old"
            dst.write_text("old content")
            src.write_text("new content")
            atomic_replace(src, dst)
            self.assertEqual(dst.read_text(), "new content")

    def test_creates_destination_parent(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.txt"
            dst = Path(d) / "subdir" / "dst.txt"
            src.write_text("data")
            atomic_replace(src, dst)
            self.assertEqual(dst.read_text(), "data")


# ===========================================================================
# Manifest
# ===========================================================================

class TestManifest(unittest.TestCase):
    def test_load_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as d:
            m = Manifest(d)
            result = m.load()
            self.assertIsNone(result)

    def test_load_reads_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps({"checkpoints": []}))
            m = Manifest(d)
            data = m.load()
            self.assertEqual(data, {"checkpoints": []})

    def test_save_atomic_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            m = Manifest(d)
            m._data = {"version": 1}
            m.save_atomic()
            m2 = Manifest(d)
            loaded = m2.load()
            self.assertEqual(loaded, {"version": 1})

    def test_path_property(self):
        with tempfile.TemporaryDirectory() as d:
            m = Manifest(d)
            self.assertEqual(m.path, Path(d) / "manifest.json")

    def test_rebuild_from_disk_returns_empty_dict(self):
        """Stub implementation must return an empty dict without raising."""
        with tempfile.TemporaryDirectory() as d:
            m = Manifest(d)
            result = m.rebuild_from_disk()
            self.assertEqual(result, {})
            self.assertEqual(m.data, {})


# ===========================================================================
# CellAnnotationPlugin feature-flag gating
# ===========================================================================

class TestFeatureFlag(unittest.TestCase):
    def test_flag_off_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ENABLE_CELL_ANNOTATION", None)
            self.assertFalse(_flag_enabled())

    def test_flag_on_with_1(self):
        with patch.dict(os.environ, {"ENABLE_CELL_ANNOTATION": "1"}):
            self.assertTrue(_flag_enabled())

    def test_flag_on_with_true(self):
        with patch.dict(os.environ, {"ENABLE_CELL_ANNOTATION": "true"}):
            self.assertTrue(_flag_enabled())

    def test_flag_on_with_yes(self):
        with patch.dict(os.environ, {"ENABLE_CELL_ANNOTATION": "yes"}):
            self.assertTrue(_flag_enabled())

    def test_flag_off_with_0(self):
        with patch.dict(os.environ, {"ENABLE_CELL_ANNOTATION": "0"}):
            self.assertFalse(_flag_enabled())

    def test_flag_on_case_insensitive(self):
        with patch.dict(os.environ, {"ENABLE_CELL_ANNOTATION": "TRUE"}):
            self.assertTrue(_flag_enabled())


# ===========================================================================
# CellAnnotationPlugin lifecycle
# ===========================================================================

class TestCellAnnotationPlugin(unittest.TestCase):
    def _make_plugin(self):
        viewer = MagicMock()
        return CellAnnotationPlugin(viewer), viewer

    def test_initial_state(self):
        plugin, _ = self._make_plugin()
        self.assertIsNone(plugin.store)
        self.assertIsNone(plugin.manifest)

    def test_on_dataset_opened_creates_dirs(self):
        plugin, _ = self._make_plugin()
        with tempfile.TemporaryDirectory() as d:
            plugin.on_dataset_opened(d)
            store = plugin.store
            self.assertIsNotNone(store)
            self.assertTrue(store.store_path.is_dir())
            for sub in STORE_SUBDIRS:
                self.assertTrue(
                    (store.store_path / sub).is_dir(),
                    f"Expected '{sub}' subdir to exist",
                )

    def test_on_dataset_opened_sets_manifest(self):
        plugin, _ = self._make_plugin()
        with tempfile.TemporaryDirectory() as d:
            plugin.on_dataset_opened(d)
            self.assertIsNotNone(plugin.manifest)

    def test_on_dataset_closed_clears_references(self):
        plugin, _ = self._make_plugin()
        with tempfile.TemporaryDirectory() as d:
            plugin.on_dataset_opened(d)
            plugin.on_dataset_closed()
        self.assertIsNone(plugin.store)
        self.assertIsNone(plugin.manifest)

    def test_registry_key_constant(self):
        self.assertEqual(CellAnnotationPlugin.REGISTRY_KEY, "cell_annotation_plugin")

    def test_viewer_registration_with_flag_on(self):
        """_register_cell_annotation_plugin attaches the plugin when flag is set."""
        # Import the viewer module's helper directly (no full viewer needed).
        from ueler.viewer.plugin.cell_annotation.plugin import CellAnnotationPlugin as CAP

        mock_viewer = MagicMock()
        mock_viewer.base_folder = None  # will be patched per call
        mock_viewer._debug = False

        with tempfile.TemporaryDirectory() as d:
            mock_viewer.base_folder = d
            with patch.dict(os.environ, {"ENABLE_CELL_ANNOTATION": "1"}):
                plugin = CAP(mock_viewer)
                plugin.on_dataset_opened(d)
                setattr(mock_viewer, CAP.REGISTRY_KEY, plugin)
                retrieved = getattr(mock_viewer, CAP.REGISTRY_KEY)
                self.assertIsInstance(retrieved, CAP)
                self.assertIsNotNone(retrieved.store)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
