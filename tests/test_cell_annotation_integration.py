"""Focused integration tests for Cell Annotation wiring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ueler.viewer.plugin.cell_annotation import CellAnnotationPlugin


class TestCellAnnotationCompatibility(unittest.TestCase):
    def test_legacy_viewer_namespace_resolves_to_packaged_plugin(self):
        from viewer.plugin.cell_annotation import CellAnnotationPlugin as LegacyPlugin
        from viewer.plugin.cell_annotation import DatasetStore as LegacyStore

        self.assertIs(LegacyPlugin, CellAnnotationPlugin)
        self.assertEqual(LegacyStore.__name__, "DatasetStore")

    def test_plugin_opened_dataset_creates_manifest_location(self):
        plugin = CellAnnotationPlugin(object())
        with tempfile.TemporaryDirectory() as dataset_root:
            plugin.on_dataset_opened(dataset_root)

            self.assertTrue((plugin.store.store_path / "checkpoints").is_dir())
            self.assertTrue((plugin.store.store_path / "thumbnails").is_dir())
            self.assertTrue((plugin.store.store_path / "selections").is_dir())
            self.assertEqual(plugin.manifest.path, Path(plugin.store.store_path) / "manifest.json")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
