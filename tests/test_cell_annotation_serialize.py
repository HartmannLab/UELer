"""Tests for Cell Annotation AnnData serialization helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ueler.viewer.plugin.cell_annotation.serialize import (
    SCHEMA_HASH,
    serialize_heatmap_state,
    validate_artifact,
    write_h5ad_atomic,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "cell_annotation_roundtrip_fixture.h5ad"


def _example_payloads() -> tuple[dict, dict, dict]:
    display = {
        "obs_names": ["cluster_a", "cluster_b"],
        "var_names": ["CD3", "CD4", "CD8"],
        "X": [[-1.25, 0.5, 1.0], [0.25, -0.5, 0.75]],
        "median_matrix": [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
        "obsm": {"X_umap": [[0.0, 0.1], [1.0, 1.1]]},
        "ui": {
            "orientation": {"horizontal": False},
            "row_sort": {"by": "dendrogram"},
            "col_sort": {"by": "selected_channels"},
            "selected_channels_ordered": ["CD8", "CD4", "CD3"],
            "row_order": ["cluster_b", "cluster_a"],
            "col_order": ["CD8", "CD4", "CD3"],
        },
        "palette": {
            "meta_cluster_colors_present": {"cluster_a": "#112233", "cluster_b": "#445566"},
            "meta_cluster_colors_all": {"cluster_a": "#112233", "cluster_b": "#445566", "cluster_c": "#778899"},
        },
        "zscore_params": {
            "method": "per-marker-zscore",
            "per_marker": {
                "CD3": {"mean": 11.5, "std": 1.5},
                "CD4": {"mean": 12.5, "std": 1.5},
                "CD8": {"mean": 13.5, "std": 1.5},
            },
            "clipped": True,
        },
        "filters": {
            "expr": "cluster in ['T cell']",
            "structured": {"subset_on": "meta_cluster", "values": ["T cell"]},
            "source": "heatmap",
        },
        "row_linkage": [
            [0.0, 1.0, 0.42, 2.0],
        ],
        "row_linkage_basis": {"marker_ids": ["CD3", "CD8"], "distance": "euclidean"},
        "marker_sets": {
            "training": ["CD3", "CD8"],
            "display_extra": ["CD4"],
            "available": ["CD3", "CD4", "CD8"],
            "linkage": ["CD3", "CD8"],
            "expanded_training": ["CD3", "CD8", "CD4"],
            "panel": ["CD3", "CD4", "CD8"],
        },
    }
    flowsom = {
        "training_markers": ["CD3", "CD8"],
        "imputation": {"enabled": False},
        "projection": {"method": "none"},
        "availability": {"flowsom_plugin": True},
        "seed": 7,
        "grid": {"xdim": 2, "ydim": 2, "rlen": 10},
        "params": {"seed": 7, "xdim": 2, "ydim": 2, "rlen": 10},
        "deps": ["numpy", "anndata"],
        "hashes": {"input": "abc123"},
    }
    meta = {
        "artifact_version": "1.0.0",
        "checkpoint": {
            "id": "018f05c9-1d4e-7f0a-b341-c85e6020d0b6",
            "parents": ["root"],
            "op": "save",
            "step_id": "heatmap.export",
            "description": "Example checkpoint",
            "created_at": "2026-03-17T23:54:11Z",
            "producer": {"name": "UELer", "version": "test"},
            "id_namespace": "ueler.test",
        },
    }
    return display, flowsom, meta


class TestCellAnnotationSerialize(unittest.TestCase):
    def test_round_trip_preserves_axes_orders_and_required_schema(self):
        display, flowsom, meta = _example_payloads()
        adata = serialize_heatmap_state(display, flowsom=flowsom, meta=meta)

        self.assertEqual(adata.uns["artifact"]["schema_hash"], SCHEMA_HASH)
        self.assertEqual(adata.uns["flowsom"]["training_markers"], adata.uns["marker_sets"]["training"])
        self.assertEqual(adata.uns["row_linkage_basis"]["marker_ids"], adata.uns["marker_sets"]["linkage"])

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "checkpoint.h5ad"
            write_h5ad_atomic(adata, path)

            restored = validate_artifact(path)

        self.assertEqual(restored.obs_names.tolist(), ["cluster_a", "cluster_b"])
        self.assertEqual(restored.var_names.tolist(), ["CD3", "CD4", "CD8"])
        self.assertEqual(restored.uns["ui"]["row_order"], ["cluster_b", "cluster_a"])
        self.assertEqual(restored.uns["ui"]["col_order"], ["CD8", "CD4", "CD3"])
        self.assertIn("h5ad_sha256", restored.uns["artifact"]["checksums"])

    def test_validator_rejects_bad_hex_colors(self):
        display, flowsom, meta = _example_payloads()
        display["palette"]["meta_cluster_colors_present"]["cluster_a"] = "blue"

        with self.assertRaisesRegex(ValueError, "invalid hex color"):
            serialize_heatmap_state(display, flowsom=flowsom, meta=meta)

    def test_validator_rejects_non_permutation_orders(self):
        display, flowsom, meta = _example_payloads()
        adata = serialize_heatmap_state(display, flowsom=flowsom, meta=meta)
        adata.uns["ui"]["row_order"] = ["cluster_a", "cluster_a"]

        with self.assertRaisesRegex(ValueError, "row_order must be a permutation"):
            validate_artifact(adata)

    def test_validator_requires_zscore_params(self):
        display, flowsom, meta = _example_payloads()
        adata = serialize_heatmap_state(display, flowsom=flowsom, meta=meta)
        del adata.uns["zscore_params"]

        with self.assertRaisesRegex(ValueError, "missing required block 'zscore_params'"):
            validate_artifact(adata)

    def test_validator_detects_checksum_mismatch(self):
        display, flowsom, meta = _example_payloads()
        adata = serialize_heatmap_state(display, flowsom=flowsom, meta=meta)
        adata.uns["artifact"]["checksums"]["x_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "checksum mismatch for X"):
            validate_artifact(adata)

    def test_atomic_writer_cleans_temp_file_on_failure(self):
        display, flowsom, meta = _example_payloads()
        adata = serialize_heatmap_state(display, flowsom=flowsom, meta=meta)

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "checkpoint.h5ad"
            with patch("ueler.viewer.plugin.cell_annotation.serialize.atomic_replace", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    write_h5ad_atomic(adata, path)

            self.assertFalse(path.exists())
            self.assertEqual(list(Path(root).glob(".*.tmp*.h5ad")), [])

    def test_checked_in_fixture_validates(self):
        fixture = validate_artifact(FIXTURE_PATH)

        self.assertEqual(fixture.obs_names.tolist(), ["cluster_a", "cluster_b"])
        self.assertEqual(fixture.var_names.tolist(), ["CD3", "CD4", "CD8"])
        self.assertEqual(fixture.uns["ui"]["row_order"], ["cluster_b", "cluster_a"])
        self.assertEqual(fixture.uns["ui"]["col_order"], ["CD8", "CD4", "CD3"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
