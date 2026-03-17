"""Unit tests for Cell Annotation scaffolding."""

from __future__ import annotations

import json
import importlib.util
import inspect
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from viewer.interfaces import FlowsomParamsProvider, HeatmapStateProvider, SelectionSpec
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

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module_from_file(module_name: str, file_path: Path, stubs: dict[str, types.ModuleType]):
    original_modules = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _widget_module() -> types.ModuleType:
    module = types.ModuleType("ipywidgets")

    class _Widget:
        def __init__(self, *args, **kwargs):
            self.value = kwargs.get("value")
            self.options = kwargs.get("options", [])
            self.allowed_tags = kwargs.get("allowed_tags", [])
            self.children = tuple(kwargs.get("children", ()))

        def observe(self, *_args, **_kwargs):
            return None

        def on_click(self, *_args, **_kwargs):
            return None

    for name in (
        "SelectMultiple",
        "FloatSlider",
        "Dropdown",
        "VBox",
        "Output",
        "TagsInput",
        "Checkbox",
        "IntText",
        "Text",
        "Button",
        "HBox",
        "Layout",
        "IntSlider",
        "Tab",
        "RadioButtons",
        "HTML",
    ):
        setattr(module, name, _Widget)
    module.Widget = _Widget
    return module


class TestDatasetStore(unittest.TestCase):
    def test_dataset_id_is_stable_and_short(self):
        with tempfile.TemporaryDirectory() as dataset_root:
            value = _dataset_id(dataset_root)
            self.assertEqual(value, _dataset_id(dataset_root))
            self.assertEqual(len(value), 32)
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
            self.assertEqual(list(Path(root).glob(".*.tmp*")), [])

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

    def test_atomic_replace_fsyncs_source_and_target_directories(self):
        with tempfile.TemporaryDirectory() as root:
            src = Path(root) / "src.json"
            dst = Path(root) / "deep" / "dst.json"
            src.write_text('{"ok": true}')

            with patch("ueler.viewer.plugin.cell_annotation.store._fsync_dir") as mock_fsync_dir:
                atomic_replace(src, dst)

            mock_fsync_dir.assert_any_call(Path(root))
            mock_fsync_dir.assert_any_call(dst.parent)


class TestManifest(unittest.TestCase):
    def test_manifest_load_and_save_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = Manifest(root)
            manifest.data["checkpoints"] = []
            manifest.save_atomic()
            self.assertEqual(Manifest(root).load(), {"checkpoints": []})

    def test_manifest_rebuild_discovers_checkpoint_artifacts_and_persists_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = Manifest(root)
            checkpoints_dir = Path(root) / "checkpoints"
            thumbnails_dir = Path(root) / "thumbnails"
            selections_dir = Path(root) / "selections"
            checkpoints_dir.mkdir()
            thumbnails_dir.mkdir()
            selections_dir.mkdir()

            (checkpoints_dir / "abc123.json").write_text(
                json.dumps({"id": "abc123", "parents": ["root"], "op": "save"}),
                encoding="utf-8",
            )
            (checkpoints_dir / "ignored.partial.json").write_text("{}", encoding="utf-8")
            (thumbnails_dir / "abc123.png").write_text("png", encoding="utf-8")
            (selections_dir / "abc123.parquet").write_text("parquet", encoding="utf-8")

            rebuilt = manifest.rebuild_from_disk()

            self.assertEqual(
                rebuilt,
                {
                    "checkpoints": [
                        {
                            "id": "abc123",
                            "parents": ["root"],
                            "op": "save",
                            "artifacts": {
                                "checkpoint": "checkpoints/abc123.json",
                                "thumbnail": "thumbnails/abc123.png",
                                "selection": "selections/abc123.parquet",
                            },
                        }
                    ]
                },
            )
            self.assertEqual(json.loads(manifest.path.read_text(encoding="utf-8")), rebuilt)


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


class TestCrossPluginInterfaces(unittest.TestCase):
    def test_compatibility_interfaces_are_importable(self):
        self.assertTrue(hasattr(SelectionSpec, "cardinality"))
        self.assertTrue(hasattr(HeatmapStateProvider, "export_heatmap_state"))
        self.assertTrue(hasattr(FlowsomParamsProvider, "run_flowsom"))

    def test_flowsom_protocol_exposes_expected_signature(self):
        signature = inspect.signature(FlowsomParamsProvider.run_flowsom)
        self.assertEqual(
            list(signature.parameters),
            [
                "self",
                "selection",
                "params",
                "training_markers",
                "extra_markers",
                "imputation",
                "projection",
            ],
        )


class TestFeatureFlagAndPluginLifecycle(unittest.TestCase):
    def test_flag_defaults_to_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_flag_enabled())

    def test_truthy_feature_flag_values_enable_plugin(self):
        for value in ("1", "true", "TRUE", "yes", "on"):
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

    def test_plugin_logs_store_path_on_dataset_opened(self):
        plugin = CellAnnotationPlugin(MagicMock())

        with tempfile.TemporaryDirectory() as dataset_root:
            with self.assertLogs("ueler.viewer.plugin.cell_annotation.plugin", level="INFO") as logs:
                plugin.on_dataset_opened(dataset_root)

        self.assertTrue(
            any("dataset store ready" in message and ".UELer" in message for message in logs.output),
            logs.output,
        )

    def test_plugin_rebuilds_manifest_when_missing(self):
        plugin = CellAnnotationPlugin(MagicMock())

        with tempfile.TemporaryDirectory() as dataset_root:
            store = DatasetStore(dataset_root)
            store.ensure_dirs()
            (store.subdir("checkpoints") / "checkpoint-a.json").write_text(
                json.dumps({"op": "save"}),
                encoding="utf-8",
            )

            plugin.on_dataset_opened(dataset_root)

            self.assertEqual(
                plugin.manifest.data,
                {
                    "checkpoints": [
                        {
                            "id": "checkpoint-a",
                            "op": "save",
                            "artifacts": {"checkpoint": "checkpoints/checkpoint-a.json"},
                        }
                    ]
                },
            )
            self.assertTrue(plugin.manifest.path.exists())

    def test_register_loaded_providers_logs_each_provider_once(self):
        plugin = CellAnnotationPlugin(MagicMock())
        heatmap = MagicMock(name="heatmap")
        flowsom = MagicMock(name="flowsom")
        side_plots = SimpleNamespace(
            heatmap_output=SimpleNamespace(_register_cell_annotation_provider=lambda: plugin.register_heatmap(heatmap)),
            flowsom_output=SimpleNamespace(_register_cell_annotation_provider=lambda: plugin.register_flowsom(flowsom)),
        )

        with self.assertLogs("ueler.viewer.plugin.cell_annotation.plugin", level="INFO") as logs:
            plugin.register_loaded_providers(side_plots)
            plugin.register_loaded_providers(side_plots)

        self.assertIs(plugin.heatmap_provider, heatmap)
        self.assertIs(plugin.flowsom_provider, flowsom)
        self.assertEqual(sum("registered Heatmap provider" in message for message in logs.output), 1)
        self.assertEqual(sum("registered FlowSOM provider" in message for message in logs.output), 1)

    def test_register_loaded_providers_allows_missing_flowsom(self):
        plugin = CellAnnotationPlugin(MagicMock())
        heatmap = MagicMock(name="heatmap")
        side_plots = SimpleNamespace(
            heatmap_output=SimpleNamespace(_register_cell_annotation_provider=lambda: plugin.register_heatmap(heatmap)),
            other_output=object(),
        )

        plugin.register_loaded_providers(side_plots)

        self.assertIs(plugin.heatmap_provider, heatmap)
        self.assertIsNone(plugin.flowsom_provider)


class TestProviderStubMethods(unittest.TestCase):
    def test_heatmap_provider_registration_calls_cell_annotation_hook(self):
        heatmap = types.SimpleNamespace()
        register_heatmap = MagicMock()
        heatmap.main_viewer = types.SimpleNamespace(
            cell_annotation_plugin=types.SimpleNamespace(register_heatmap=register_heatmap)
        )

        heatmap_stubs = {
            "ipywidgets": _widget_module(),
            "pandas": types.ModuleType("pandas"),
            "scipy.cluster.hierarchy": types.SimpleNamespace(dendrogram=lambda *_a, **_k: None),
            "ueler.viewer.observable": types.SimpleNamespace(Observable=object),
            "ueler.viewer.plugin.plugin_base": types.SimpleNamespace(
                PluginBase=type("PluginBase", (), {"__init__": lambda self, *_args, **_kwargs: None})
            ),
            "ueler.viewer.plugin.heatmap_adapter": types.SimpleNamespace(
                HeatmapModeAdapter=type("HeatmapModeAdapter", (), {"__init__": lambda self, *_args, **_kwargs: None})
            ),
            "ueler.viewer.plugin.heatmap_layers": types.SimpleNamespace(
                DataLayer=type("DataLayer", (), {}),
                InteractionLayer=type("InteractionLayer", (), {}),
                DisplayLayer=type("DisplayLayer", (), {}),
            ),
        }
        module = _load_module_from_file(
            "test_heatmap_module_registration",
            REPO_ROOT / "ueler/viewer/plugin/heatmap.py",
            heatmap_stubs,
        )

        module.HeatmapDisplay._register_cell_annotation_provider(heatmap)

        register_heatmap.assert_called_once_with(heatmap)

    def test_heatmap_import_stub_records_last_path(self):
        heatmap_stubs = {
            "ipywidgets": _widget_module(),
            "pandas": types.ModuleType("pandas"),
            "scipy.cluster.hierarchy": types.SimpleNamespace(dendrogram=lambda *_a, **_k: None),
            "ueler.viewer.observable": types.SimpleNamespace(Observable=object),
            "ueler.viewer.plugin.plugin_base": types.SimpleNamespace(
                PluginBase=type("PluginBase", (), {"__init__": lambda self, *_args, **_kwargs: None})
            ),
            "ueler.viewer.plugin.heatmap_adapter": types.SimpleNamespace(
                HeatmapModeAdapter=type("HeatmapModeAdapter", (), {"__init__": lambda self, *_args, **_kwargs: None})
            ),
            "ueler.viewer.plugin.heatmap_layers": types.SimpleNamespace(
                DataLayer=type("DataLayer", (), {}),
                InteractionLayer=type("InteractionLayer", (), {}),
                DisplayLayer=type("DisplayLayer", (), {}),
            ),
        }
        module = _load_module_from_file(
            "test_heatmap_module",
            REPO_ROOT / "ueler/viewer/plugin/heatmap.py",
            heatmap_stubs,
        )
        heatmap = module.HeatmapDisplay.__new__(module.HeatmapDisplay)

        module.HeatmapDisplay.import_heatmap_state(heatmap, "/tmp/checkpoint.h5ad")

        self.assertEqual(heatmap._last_imported_heatmap_state_path, "/tmp/checkpoint.h5ad")

    def test_flowsom_selection_context_stub_is_stored(self):
        numpy_stub = types.ModuleType("numpy")
        numpy_stub.inf = float("inf")
        flowsom_stubs = {
            "numpy": numpy_stub,
            "pandas": types.ModuleType("pandas"),
            "seaborn": types.ModuleType("seaborn"),
            "ipywidgets": _widget_module(),
            "matplotlib.font_manager": types.ModuleType("matplotlib.font_manager"),
            "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot"),
            "matplotlib.backend_bases": types.SimpleNamespace(MouseButton=object),
            "matplotlib.text": types.SimpleNamespace(Annotation=object),
            "IPython.display": types.SimpleNamespace(display=lambda *_a, **_k: None),
            "mpl_toolkits.axes_grid1": types.SimpleNamespace(make_axes_locatable=lambda *_a, **_k: None),
            "mpl_toolkits.axes_grid1.anchored_artists": types.SimpleNamespace(AnchoredSizeBar=object),
            "scipy.cluster.hierarchy": types.SimpleNamespace(
                cut_tree=lambda *_a, **_k: None,
                dendrogram=lambda *_a, **_k: None,
                linkage=lambda *_a, **_k: None,
            ),
            "ueler.image_utils": types.SimpleNamespace(
                color_one_image=lambda *_a, **_k: None,
                estimate_color_range=lambda *_a, **_k: None,
                process_single_crop=lambda *_a, **_k: None,
            ),
            "ueler.viewer.decorators": types.SimpleNamespace(update_status_bar=lambda func: func),
            "ueler.viewer.observable": types.SimpleNamespace(Observable=object),
            "ueler.viewer.plugin.plugin_base": types.SimpleNamespace(
                PluginBase=type("PluginBase", (), {"__init__": lambda self, *_args, **_kwargs: None})
            ),
        }
        module = _load_module_from_file(
            "test_flowsom_module",
            REPO_ROOT / "ueler/viewer/plugin/run_flowsom.py",
            flowsom_stubs,
        )
        flowsom = module.RunFlowsom.__new__(module.RunFlowsom)
        selection = MaterializedSelectionSpec.from_cells("dataset_a", [("fov1", 1)])

        module.RunFlowsom.set_selection_context(flowsom, selection)

        self.assertIs(flowsom._selection_context, selection)

    def test_flowsom_provider_registration_calls_cell_annotation_hook(self):
        flowsom = types.SimpleNamespace()
        register_flowsom = MagicMock()
        flowsom.main_viewer = types.SimpleNamespace(
            cell_annotation_plugin=types.SimpleNamespace(register_flowsom=register_flowsom)
        )

        numpy_stub = types.ModuleType("numpy")
        numpy_stub.inf = float("inf")
        flowsom_stubs = {
            "numpy": numpy_stub,
            "pandas": types.ModuleType("pandas"),
            "seaborn": types.ModuleType("seaborn"),
            "ipywidgets": _widget_module(),
            "matplotlib.font_manager": types.ModuleType("matplotlib.font_manager"),
            "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot"),
            "matplotlib.backend_bases": types.SimpleNamespace(MouseButton=object),
            "matplotlib.text": types.SimpleNamespace(Annotation=object),
            "IPython.display": types.SimpleNamespace(display=lambda *_a, **_k: None),
            "mpl_toolkits.axes_grid1": types.SimpleNamespace(make_axes_locatable=lambda *_a, **_k: None),
            "mpl_toolkits.axes_grid1.anchored_artists": types.SimpleNamespace(AnchoredSizeBar=object),
            "scipy.cluster.hierarchy": types.SimpleNamespace(
                cut_tree=lambda *_a, **_k: None,
                dendrogram=lambda *_a, **_k: None,
                linkage=lambda *_a, **_k: None,
            ),
            "ueler.image_utils": types.SimpleNamespace(
                color_one_image=lambda *_a, **_k: None,
                estimate_color_range=lambda *_a, **_k: None,
                process_single_crop=lambda *_a, **_k: None,
            ),
            "ueler.viewer.decorators": types.SimpleNamespace(update_status_bar=lambda func: func),
            "ueler.viewer.observable": types.SimpleNamespace(Observable=object),
            "ueler.viewer.plugin.plugin_base": types.SimpleNamespace(
                PluginBase=type("PluginBase", (), {"__init__": lambda self, *_args, **_kwargs: None})
            ),
        }
        module = _load_module_from_file(
            "test_flowsom_module_registration",
            REPO_ROOT / "ueler/viewer/plugin/run_flowsom.py",
            flowsom_stubs,
        )

        module.RunFlowsom._register_cell_annotation_provider(flowsom)

        register_flowsom.assert_called_once_with(flowsom)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
