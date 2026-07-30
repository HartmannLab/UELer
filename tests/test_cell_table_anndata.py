"""Unit tests for AnnData cell-table support (issue #123)."""

from __future__ import annotations

import os
import tempfile
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

from ueler.cell_table import (
    categorical_columns,
    dataframe_to_anndata,
    flatten_anndata,
    is_anndata,
    sync_cell_table_to_obs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _obs(n_rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov2"][:n_rows],
            "label": [1, 2, 3][:n_rows],
            "cell_type": ["TypeA", "TypeB", "TypeA"][:n_rows],
            "area": [10.0, 20.0, 30.0][:n_rows],
        },
        index=["c1", "c2", "c3"][:n_rows],
    )


def _adata(*, var_names=("CD4", "CD8", "CD3"), with_obsm=False, clashing_var=False):
    obs = _obs()
    names = list(var_names)
    if clashing_var:
        names[-1] = "area"  # deliberately collides with an obs column
    X = np.arange(len(obs) * len(names), dtype="float32").reshape(len(obs), len(names))
    adata = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(names)))
    adata.layers["counts"] = X * 10.0
    if with_obsm:
        adata.obsm["X_umap"] = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        adata.obsm["X_pca"] = np.zeros((len(obs), 10), dtype="float64")
    return adata


def _roundtripped(adata):
    """Write and re-read the object — this is what turns strings into categories."""
    path = os.path.join(tempfile.mkdtemp(), "cells.h5ad")
    adata.write_h5ad(path)
    return anndata.read_h5ad(path), path


def _viewer():
    """A viewer with the real methods but without the heavy image-folder __init__."""
    from ueler.viewer.main_viewer import ImageMaskViewer

    class _Viewer(ImageMaskViewer):
        def __init__(self):  # noqa: D107 - deliberately skips the base __init__
            self.cell_table = None
            self.cell_table_adata = None
            self.cell_table_columns = None
            self.fov_key = "fov"
            self.label_key = "label"
            self.x_key = "X"
            self.y_key = "Y"
            self.mask_key = "cell"
            self._debug = False

    return _Viewer()


# ---------------------------------------------------------------------------
# flatten_anndata
# ---------------------------------------------------------------------------
class TestFlattenAnnData(unittest.TestCase):
    def test_is_anndata_discriminates(self):
        self.assertTrue(is_anndata(_adata()))
        self.assertFalse(is_anndata(_obs()))
        self.assertFalse(is_anndata(None))
        self.assertFalse(is_anndata(pd.Series([1, 2])))

    def test_columns_are_obs_then_markers_then_obsm_then_index(self):
        frame, provenance = flatten_anndata(_adata(with_obsm=True))

        self.assertEqual(
            frame.columns.tolist(),
            [
                "fov",
                "label",
                "cell_type",
                "area",
                "CD4",
                "CD8",
                "CD3",
                "X_umap1",
                "X_umap2",
                "obs_names",
            ],
        )
        self.assertEqual(provenance["obs"], ["fov", "label", "cell_type", "area"])
        self.assertEqual(provenance["var"], ["CD4", "CD8", "CD3"])
        self.assertEqual(provenance["obsm"], ["X_umap1", "X_umap2"])
        self.assertEqual(provenance["index"], "obs_names")

    def test_frame_has_range_index_and_keeps_obs_names(self):
        # The plugins mix ``cell_table.index`` with ``cell_table.iloc[i]``; those
        # only agree on a RangeIndex, which is what the CSV path produces.
        frame, provenance = flatten_anndata(_adata())

        self.assertIsInstance(frame.index, pd.RangeIndex)
        self.assertEqual(frame.index.tolist(), [0, 1, 2])
        self.assertEqual(frame[provenance["index"]].tolist(), ["c1", "c2", "c3"])

    def test_marker_dtype_is_preserved(self):
        frame, _ = flatten_anndata(_adata())

        self.assertEqual(frame["CD4"].dtype, np.dtype("float32"))
        self.assertEqual(frame["CD4"].tolist(), [0.0, 3.0, 6.0])

    def test_wide_obsm_is_skipped_unless_requested(self):
        _, provenance = flatten_anndata(_adata(with_obsm=True))
        self.assertNotIn("X_pca1", provenance["obsm"])

        frame, provenance = flatten_anndata(_adata(with_obsm=True), obsm_keys=["X_pca"])
        self.assertEqual(len(provenance["obsm"]), 10)
        self.assertIn("X_pca10", frame.columns)

    def test_layer_selects_a_different_matrix(self):
        frame, provenance = flatten_anndata(_adata(), layer="counts")

        self.assertEqual(frame["CD4"].tolist(), [0.0, 30.0, 60.0])
        self.assertEqual(provenance["layer"], "counts")

    def test_marker_name_clashing_with_obs_is_suffixed(self):
        frame, provenance = flatten_anndata(_adata(clashing_var=True))

        # obs wins the name; the marker is suffixed and the rename recorded.
        self.assertIn("area", provenance["obs"])
        self.assertIn("area_var", frame.columns)
        self.assertEqual(provenance["renamed"], {"area": "area_var"})
        self.assertEqual(frame["area"].tolist(), [10.0, 20.0, 30.0])

    def test_sparse_matrix_is_densified(self):
        from scipy import sparse

        adata = _adata()
        adata.X = sparse.csr_matrix(adata.X)

        frame, _ = flatten_anndata(adata)
        self.assertEqual(frame["CD4"].tolist(), [0.0, 3.0, 6.0])

    def test_obs_only_anndata_has_no_marker_columns(self):
        adata = anndata.AnnData(obs=_obs().reset_index(drop=True))

        frame, provenance = flatten_anndata(adata)
        self.assertEqual(provenance["var"], [])
        self.assertIn("cell_type", frame.columns)

    def test_duplicate_obs_names_do_not_scramble_the_frame(self):
        adata = _adata()
        adata.obs_names = ["dup", "dup", "dup"]

        frame, _ = flatten_anndata(adata)
        self.assertEqual(frame["label"].tolist(), [1, 2, 3])
        self.assertEqual(frame["CD4"].tolist(), [0.0, 3.0, 6.0])


# ---------------------------------------------------------------------------
# Categorical handling
# ---------------------------------------------------------------------------
class TestCategoricalColumns(unittest.TestCase):
    def test_h5ad_roundtrip_produces_categorical_obs(self):
        # Guards the premise of the two tests below.
        adata, _ = _roundtripped(_adata())
        self.assertIsInstance(adata.obs["cell_type"].dtype, pd.CategoricalDtype)

    def test_flatten_widens_categorical_so_class_columns_stay_selectable(self):
        adata, _ = _roundtripped(_adata())
        frame, provenance = flatten_anndata(adata)

        self.assertEqual(frame["cell_type"].dtype, np.dtype("object"))
        self.assertEqual(sorted(provenance["categorical_obs"]), ["cell_type", "fov"])
        # Without the widening these would be missing from every cluster /
        # identifier / tooltip-label dropdown.
        for column in ("fov", "cell_type", "label"):
            self.assertIn(column, categorical_columns(frame))

    def test_widened_frame_supports_assigning_an_unseen_class(self):
        # What the Cell Table Editor does; a Categorical would raise TypeError.
        adata, _ = _roundtripped(_adata())
        frame, _ = flatten_anndata(adata)

        frame.loc[0, "cell_type"] = "BrandNewClass"
        self.assertEqual(frame.loc[0, "cell_type"], "BrandNewClass")

    def test_numeric_categories_become_a_numeric_column(self):
        adata = _adata()
        adata.obs["cluster"] = pd.Categorical([1, 2, 1])

        frame, _ = flatten_anndata(adata)
        self.assertTrue(pd.api.types.is_numeric_dtype(frame["cluster"]))

    def test_categorical_columns_matches_the_legacy_rule_for_a_dataframe(self):
        # Guards the refactor of the eight ``select_dtypes`` call sites: a
        # CSV-style table must yield exactly what it did before.
        frame = pd.DataFrame(
            {
                "fov": ["fov1", "fov2"],
                "label": [1, 2],
                "intensity": [1.0, 2.0],
                "flag": [True, False],
            }
        )
        legacy = frame.select_dtypes(include=["int", "int64", "object"]).columns.tolist()

        self.assertEqual(categorical_columns(frame), legacy)
        self.assertEqual(
            categorical_columns(frame, include_bool=True),
            frame.select_dtypes(include=["int", "int64", "object", "bool"]).columns.tolist(),
        )

    def test_categorical_columns_handles_a_category_dtype_dataframe(self):
        # A user may pass ``adata.obs`` straight in as the cell table.
        frame = pd.DataFrame({"cell_type": pd.Categorical(["A", "B"]), "label": [1, 2]})

        self.assertEqual(categorical_columns(frame), ["cell_type", "label"])

    def test_categorical_columns_tolerates_none(self):
        self.assertEqual(categorical_columns(None), [])


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------
class TestSyncBackToObs(unittest.TestCase):
    def test_new_column_lands_in_obs(self):
        adata = _adata()
        frame, provenance = flatten_anndata(adata)
        frame["FlowSOM_cluster"] = [7, 8, 9]

        written = sync_cell_table_to_obs(adata, frame, provenance)

        self.assertIn("FlowSOM_cluster", written)
        self.assertEqual(adata.obs["FlowSOM_cluster"].tolist(), [7, 8, 9])

    def test_edited_obs_column_is_updated(self):
        adata = _adata()
        frame, provenance = flatten_anndata(adata)
        frame.loc[0, "cell_type"] = "Edited"

        sync_cell_table_to_obs(adata, frame, provenance)

        self.assertEqual(adata.obs["cell_type"].tolist(), ["Edited", "TypeB", "TypeA"])

    def test_marker_and_index_columns_are_never_written_to_obs(self):
        adata = _adata(with_obsm=True)
        frame, provenance = flatten_anndata(adata)

        sync_cell_table_to_obs(adata, frame, provenance)

        for column in ("CD4", "CD8", "CD3", "X_umap1", "obs_names"):
            self.assertNotIn(column, adata.obs.columns)

    def test_unchanged_columns_are_not_rewritten(self):
        adata, _ = _roundtripped(_adata())
        frame, provenance = flatten_anndata(adata)

        self.assertEqual(sync_cell_table_to_obs(adata, frame, provenance), [])
        # The user's dtypes are left alone when nothing actually changed.
        self.assertIsInstance(adata.obs["cell_type"].dtype, pd.CategoricalDtype)

    def test_dropping_a_synced_column_removes_it_from_obs(self):
        adata = _adata()
        frame, provenance = flatten_anndata(adata)
        frame["FlowSOM_cluster"] = [7, 8, 9]
        sync_cell_table_to_obs(adata, frame, provenance)

        frame.drop(columns=["FlowSOM_cluster"], inplace=True)
        sync_cell_table_to_obs(adata, frame, provenance)

        self.assertNotIn("FlowSOM_cluster", adata.obs.columns)
        self.assertIn("cell_type", adata.obs.columns)

    def test_row_count_mismatch_is_a_no_op(self):
        adata = _adata()
        frame, provenance = flatten_anndata(adata)
        doubled = pd.concat([frame, frame], ignore_index=True)
        doubled["ghost"] = 1

        self.assertEqual(sync_cell_table_to_obs(adata, doubled, provenance), [])
        self.assertNotIn("ghost", adata.obs.columns)

    def test_missing_arguments_are_tolerated(self):
        self.assertEqual(sync_cell_table_to_obs(None, _obs(), {}), [])
        self.assertEqual(sync_cell_table_to_obs(_adata(), None, {}), [])
        self.assertEqual(sync_cell_table_to_obs(_adata(), _obs(), None), [])


# ---------------------------------------------------------------------------
# DataFrame -> AnnData
# ---------------------------------------------------------------------------
class TestDataFrameToAnnData(unittest.TestCase):
    def test_roundtrip_restores_the_original_split(self):
        source = _adata(with_obsm=True, clashing_var=True)
        source.uns["note"] = "keep me"
        frame, provenance = flatten_anndata(source)

        out = dataframe_to_anndata(frame, provenance, source=source)

        self.assertEqual(list(out.var_names), ["CD4", "CD8", "area"])  # rename undone
        self.assertEqual(list(out.obs_names), ["c1", "c2", "c3"])
        self.assertEqual(out.obs.columns.tolist(), ["fov", "label", "cell_type", "area"])
        self.assertEqual(out.uns["note"], "keep me")
        self.assertIn("X_umap", out.obsm)

    def test_plugin_added_column_survives_the_roundtrip(self):
        source = _adata()
        frame, provenance = flatten_anndata(source)
        frame["FlowSOM_cluster"] = [7, 8, 9]

        out = dataframe_to_anndata(frame, provenance, source=source)
        self.assertEqual(out.obs["FlowSOM_cluster"].tolist(), [7, 8, 9])

    def test_dataframe_sourced_table_uses_numeric_non_key_columns_as_x(self):
        frame = pd.DataFrame(
            {
                "fov": ["fov1", "fov2"],
                "label": [1, 2],
                "CD4": [0.5, 1.5],
                "cell_type": ["A", "B"],
            }
        )

        out = dataframe_to_anndata(
            frame, None, system_keys=["fov", "label", "X", "Y", "cell"]
        )

        self.assertEqual(list(out.var_names), ["CD4"])
        self.assertEqual(out.obs.columns.tolist(), ["fov", "label", "cell_type"])


# ---------------------------------------------------------------------------
# Viewer integration
# ---------------------------------------------------------------------------
class TestViewerIntegration(unittest.TestCase):
    def test_set_cell_table_with_anndata_keeps_the_object(self):
        viewer = _viewer()
        adata = _adata()

        viewer.set_cell_table(adata)

        self.assertIs(viewer.cell_table_adata, adata)
        self.assertIsInstance(viewer.cell_table, pd.DataFrame)
        self.assertEqual(viewer.cell_table_columns["var"], ["CD4", "CD8", "CD3"])

    def test_set_cell_table_with_dataframe_is_unchanged(self):
        viewer = _viewer()
        frame = _obs()

        viewer.set_cell_table(frame)

        self.assertIs(viewer.cell_table, frame)
        self.assertIsNone(viewer.cell_table_adata)
        self.assertIsNone(viewer.cell_table_columns)

    def test_anndata_state_is_reset_when_a_dataframe_replaces_it(self):
        viewer = _viewer()
        viewer.set_cell_table(_adata())

        viewer.set_cell_table(_obs())

        self.assertIsNone(viewer.cell_table_adata)
        self.assertIsNone(viewer.cell_table_columns)

    def test_layer_argument_is_rejected_for_a_dataframe(self):
        viewer = _viewer()

        with self.assertRaises(ValueError):
            viewer.set_cell_table(_obs(), layer="counts")

    def test_h5ad_path_is_loaded_as_anndata(self):
        viewer = _viewer()
        _, path = _roundtripped(_adata())

        viewer.load_cell_table_from_path(path, layer="counts")

        self.assertIsNotNone(viewer.cell_table_adata)
        self.assertEqual(viewer.cell_table["CD4"].tolist(), [0.0, 30.0, 60.0])

    def test_csv_path_still_coerces_integral_floats(self):
        viewer = _viewer()
        path = os.path.join(tempfile.mkdtemp(), "cells.csv")
        pd.DataFrame({"fov": ["fov1"], "label": [1.0], "CD4": [0.5]}).to_csv(
            path, index=False
        )

        viewer.load_cell_table_from_path(path)

        self.assertEqual(str(viewer.cell_table["label"].dtype), "Int64")
        self.assertIsNone(viewer.cell_table_adata)

    def test_on_cell_table_change_broadcast_syncs_to_obs(self):
        viewer = _viewer()
        adata = _adata()
        viewer.set_cell_table(adata)
        viewer.cell_table["FlowSOM_cluster"] = [7, 8, 9]

        # The broadcast every write path already fires.
        viewer.inform_plugins("on_cell_table_change")

        self.assertEqual(adata.obs["FlowSOM_cluster"].tolist(), [7, 8, 9])

    def test_other_broadcasts_do_not_sync(self):
        viewer = _viewer()
        adata = _adata()
        viewer.set_cell_table(adata)
        viewer.cell_table["FlowSOM_cluster"] = [7, 8, 9]

        viewer.inform_plugins("refresh_roi_table")

        self.assertNotIn("FlowSOM_cluster", adata.obs.columns)

    def test_get_cell_table_adata_returns_the_synced_original(self):
        viewer = _viewer()
        adata = _adata()
        viewer.set_cell_table(adata)
        viewer.cell_table["manual_label"] = ["a", "b", "c"]

        result = viewer.get_cell_table_adata()

        self.assertIs(result, adata)
        self.assertEqual(result.obs["manual_label"].tolist(), ["a", "b", "c"])

    def test_get_cell_table_adata_builds_one_for_a_dataframe_table(self):
        viewer = _viewer()
        viewer.set_cell_table(
            pd.DataFrame({"fov": ["fov1"], "label": [1], "CD4": [0.5]})
        )

        result = viewer.get_cell_table_adata()

        self.assertEqual(list(result.var_names), ["CD4"])
        self.assertEqual(result.obs.columns.tolist(), ["fov", "label"])

    def test_get_cell_table_adata_without_a_table(self):
        self.assertIsNone(_viewer().get_cell_table_adata())


# ---------------------------------------------------------------------------
# Marker-first ordering
# ---------------------------------------------------------------------------
class TestMarkerFirstOrdering(unittest.TestCase):
    def test_markers_precede_obs_columns_for_anndata(self):
        from ueler.viewer.plugin import _chart_common

        viewer = _viewer()
        viewer.set_cell_table(_adata())

        columns = _chart_common.numeric_columns(viewer)

        self.assertEqual(columns[:3], ["CD4", "CD8", "CD3"])
        self.assertEqual(set(columns[3:]), {"label", "area"})

    def test_order_is_untouched_for_a_dataframe(self):
        from ueler.viewer.plugin import _chart_common

        viewer = _viewer()
        viewer.set_cell_table(
            pd.DataFrame({"label": [1], "intensity": [1.0], "area": [2.0]})
        )

        self.assertEqual(
            _chart_common.numeric_columns(viewer), ["label", "intensity", "area"]
        )

    def test_marker_first_is_membership_preserving(self):
        from ueler.viewer.plugin import _chart_common

        viewer = _viewer()
        viewer.set_cell_table(_adata())
        columns = list(viewer.cell_table.columns)

        self.assertEqual(
            sorted(_chart_common.marker_first(viewer, columns)), sorted(columns)
        )


if __name__ == "__main__":  # pragma: no cover - unittest entrypoint
    unittest.main()
