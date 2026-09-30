"""Unit tests for the BIA cell-table support (issue #140).

Network access is fully mocked; these tests cover the descriptor's ``cell_table``
key, :meth:`BIADataSource.fetch_cell_table` (whole-file and FOV-filtered
streaming paths) and the ``run_viewer_bia`` / ``load_bia_cell_table`` wiring.
"""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import ueler.bia_loader as bia_loader
from ueler.bia_loader import BIADataSource, BIAStudyIndex, _layout_from_descriptor
from ueler.runner import load_bia_cell_table, run_viewer_bia
from ueler.viewer.main_viewer import ImageMaskViewer


CELL_TABLE_PATH = "Files/study/cell_table/cells.csv"

CSV_BODY = "\n".join(
    [
        "cell_id,fov,area,label",
        "1,fov1,10,tumour",
        "2,fov1,11,stroma",
        "3,fov2,12,tumour",
        "4,fov3,13,stroma",
        '5,fov2,14,"stroma, other"',
    ]
)


# --- fakes -----------------------------------------------------------------
class FakeIndex:
    """In-memory stand-in for :class:`BIAStudyIndex` (no network)."""

    def __init__(self, tree, base_url="https://example.org/study"):
        self.base_url = base_url
        self.tree = tree

    def url_for(self, rel_path):
        return f"{self.base_url}/{rel_path.strip('/')}"

    def list_dir(self, rel_path=""):
        return self.tree.get(rel_path, [])

    def list_subdirs(self, rel_path=""):
        return sorted(n for n, is_dir in self.list_dir(rel_path) if is_dir)

    def list_files(self, rel_path=""):
        return sorted(n for n, is_dir in self.list_dir(rel_path) if not is_dir)


class FakeResponse:
    """Minimal streaming ``requests`` response over a fixed CSV body."""

    def __init__(self, body):
        self.body = body
        self.encoding = "utf-8"

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=False):
        for line in self.body.splitlines():
            yield line

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _tree():
    return {
        "Files/img": [("fov1", True), ("fov2", True), ("fov3", True)],
        "Files/img/fov1": [("CD3.tiff", False)],
        "Files/img/fov2": [("CD3.tiff", False)],
        "Files/img/fov3": [("CD3.tiff", False)],
    }


def _descriptor(cell_table=CELL_TABLE_PATH):
    descriptor = {"mode": "folder", "base": "Files/img"}
    if cell_table is not None:
        descriptor["cell_table"] = cell_table
    return descriptor


def _make_source(cache_dir, cell_table=CELL_TABLE_PATH):
    with patch.object(BIAStudyIndex, "from_source", return_value=FakeIndex(_tree())):
        source = BIADataSource(
            "https://example.org/study",
            cache_dir=cache_dir,
            descriptor=_descriptor(cell_table),
        )
    source.index = FakeIndex(_tree())
    return source


def _rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


# --- descriptor ------------------------------------------------------------
class CellTableDescriptorTest(unittest.TestCase):
    def test_string_form_defaults_to_the_fov_column(self):
        layout = _layout_from_descriptor(_descriptor())
        self.assertEqual(layout.cell_table, {"path": CELL_TABLE_PATH, "fov_column": "fov"})

    def test_mapping_form_names_the_fov_column(self):
        layout = _layout_from_descriptor(
            _descriptor({"path": "/" + CELL_TABLE_PATH, "fov_column": "FOV_id"})
        )
        self.assertEqual(layout.cell_table, {"path": CELL_TABLE_PATH, "fov_column": "FOV_id"})

    def test_absent_key_leaves_no_cell_table(self):
        self.assertIsNone(_layout_from_descriptor(_descriptor(None)).cell_table)

    def test_mapping_without_path_is_ignored(self):
        self.assertIsNone(_layout_from_descriptor(_descriptor({"fov_column": "fov"})).cell_table)

    def test_ome_mode_carries_the_cell_table_too(self):
        layout = _layout_from_descriptor(
            {"mode": "ome-tiff", "fov_glob": "Files/*.ome.tiff", "cell_table": CELL_TABLE_PATH}
        )
        self.assertEqual(layout.mode, "ome-tiff")
        self.assertEqual(layout.cell_table["path"], CELL_TABLE_PATH)


# --- data source -----------------------------------------------------------
class FetchCellTableTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.cache = Path(self._tmp.name) / "cache"

    def tearDown(self):
        self._tmp.cleanup()

    def test_has_cell_table_reflects_the_descriptor(self):
        self.assertTrue(_make_source(str(self.cache)).has_cell_table)
        self.assertFalse(_make_source(str(self.cache), cell_table=None).has_cell_table)

    def test_cell_table_url_is_resolved_through_the_index(self):
        source = _make_source(str(self.cache))
        self.assertEqual(
            source.cell_table_url, f"https://example.org/study/{CELL_TABLE_PATH}"
        )
        self.assertIsNone(_make_source(str(self.cache), cell_table=None).cell_table_url)

    def test_fetch_without_a_cell_table_returns_none(self):
        source = _make_source(str(self.cache), cell_table=None)
        self.assertIsNone(source.fetch_cell_table())

    def test_fetch_whole_table_downloads_once_into_the_tables_cache(self):
        source = _make_source(str(self.cache))
        with patch.object(bia_loader, "_download", side_effect=lambda url, dest: dest) as download:
            path = source.fetch_cell_table()
        download.assert_called_once()
        url, dest = download.call_args[0]
        self.assertEqual(url, f"https://example.org/study/{CELL_TABLE_PATH}")
        self.assertEqual(Path(dest), self.cache / "tables" / "cells.csv")
        self.assertEqual(Path(path), self.cache / "tables" / "cells.csv")

    def test_fetch_with_fovs_keeps_only_those_rows(self):
        source = _make_source(str(self.cache))
        fake_requests = SimpleNamespace(get=MagicMock(return_value=FakeResponse(CSV_BODY)))
        with patch.object(bia_loader, "_ensure_requests", return_value=fake_requests):
            path = source.fetch_cell_table(["fov1", "fov3"])

        rows = _rows(path)
        self.assertEqual(rows[0], ["cell_id", "fov", "area", "label"])
        self.assertEqual([row[1] for row in rows[1:]], ["fov1", "fov1", "fov3"])
        self.assertTrue(Path(path).parent == self.cache / "tables")
        self.assertNotEqual(Path(path).name, "cells.csv")

    def test_filtered_rows_preserve_quoted_commas(self):
        source = _make_source(str(self.cache))
        fake_requests = SimpleNamespace(get=MagicMock(return_value=FakeResponse(CSV_BODY)))
        with patch.object(bia_loader, "_ensure_requests", return_value=fake_requests):
            path = source.fetch_cell_table(["fov2"])

        self.assertEqual([row[3] for row in _rows(path)[1:]], ["tumour", "stroma, other"])

    def test_a_cached_subset_is_reused_unless_forced(self):
        source = _make_source(str(self.cache))
        fake_requests = SimpleNamespace(get=MagicMock(return_value=FakeResponse(CSV_BODY)))
        with patch.object(bia_loader, "_ensure_requests", return_value=fake_requests):
            first = source.fetch_cell_table(["fov1"])
            second = source.fetch_cell_table(["fov1"])
            self.assertEqual(first, second)
            self.assertEqual(fake_requests.get.call_count, 1)

            source.fetch_cell_table(["fov1"], force=True)
            self.assertEqual(fake_requests.get.call_count, 2)

    def test_different_fov_sets_are_cached_separately(self):
        source = _make_source(str(self.cache))
        fake_requests = SimpleNamespace(get=MagicMock(return_value=FakeResponse(CSV_BODY)))
        with patch.object(bia_loader, "_ensure_requests", return_value=fake_requests):
            first = source.fetch_cell_table(["fov1"])
            second = source.fetch_cell_table(["fov2"])
        self.assertNotEqual(first, second)
        self.assertEqual([row[1] for row in _rows(second)[1:]], ["fov2", "fov2"])

    def test_missing_fov_column_raises_an_actionable_error(self):
        source = _make_source(
            str(self.cache), cell_table={"path": CELL_TABLE_PATH, "fov_column": "FOV_id"}
        )
        fake_requests = SimpleNamespace(get=MagicMock(return_value=FakeResponse(CSV_BODY)))
        with patch.object(bia_loader, "_ensure_requests", return_value=fake_requests):
            with self.assertRaises(ValueError) as ctx:
                source.fetch_cell_table(["fov1"])

        message = str(ctx.exception)
        self.assertIn("FOV_id", message)
        self.assertIn("fov_column", message)
        self.assertFalse(list((self.cache / "tables").glob("*.csv")))

    def test_non_csv_table_ignores_the_fov_filter(self):
        source = _make_source(str(self.cache), cell_table="Files/study/cells.h5ad")
        with patch.object(bia_loader, "_download", side_effect=lambda url, dest: dest) as download:
            path = source.fetch_cell_table(["fov1"])
        download.assert_called_once()
        self.assertEqual(Path(path), self.cache / "tables" / "cells.h5ad")


# --- viewer / runner wiring ------------------------------------------------
class ViewerDataSourcePropertyTest(unittest.TestCase):
    def test_property_exposes_the_private_attribute(self):
        sentinel = object()
        holder = SimpleNamespace(_data_source=sentinel)
        self.assertIs(ImageMaskViewer.data_source.fget(holder), sentinel)
        self.assertIsNone(ImageMaskViewer.data_source.fget(SimpleNamespace(_data_source=None)))


class RunViewerBiaCellTableTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.workspace = Path(self._tmp.name) / "ws"
        self.table = Path(self._tmp.name) / "cells.csv"
        self.table.write_text(CSV_BODY, encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _data_source(self, has_cell_table=True):
        return SimpleNamespace(
            has_cell_table=has_cell_table,
            fetch_cell_table=MagicMock(return_value=str(self.table)),
        )

    def _run(self, viewer, data_source, **kwargs):
        with patch("ueler.runner._load_display_helpers") as load_display:
            load_display.return_value = (MagicMock(), MagicMock())
            return run_viewer_bia(
                "S-BIAD2557",
                descriptor=_descriptor(),
                local_dir=self.workspace,
                viewer_factory=lambda base, *, data_source=None, **k: viewer,
                data_source_factory=lambda *a, **k: data_source,
                **kwargs,
            )

    def test_cell_table_is_not_fetched_by_default(self):
        viewer = MagicMock()
        data_source = self._data_source()
        self._run(viewer, data_source)
        data_source.fetch_cell_table.assert_not_called()
        viewer.load_cell_table_from_path.assert_not_called()

    def test_cell_table_true_attaches_before_the_plugin_tail(self):
        viewer = MagicMock()
        calls = []
        viewer.load_cell_table_from_path.side_effect = lambda *a, **k: calls.append("table")
        viewer.after_all_plugins_loaded.side_effect = lambda *a, **k: calls.append("plugins")
        data_source = self._data_source()

        self._run(viewer, data_source, cell_table=True, cell_table_fovs=["fov1", "fov2"])

        data_source.fetch_cell_table.assert_called_once_with(["fov1", "fov2"], force=False)
        viewer.load_cell_table_from_path.assert_called_once_with(str(self.table))
        self.assertEqual(calls, ["table", "plugins"])

    def test_cell_table_true_without_a_declared_table_raises(self):
        viewer = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            self._run(viewer, self._data_source(has_cell_table=False), cell_table=True)
        self.assertIn("cell_table", str(ctx.exception))
        viewer.load_cell_table_from_path.assert_not_called()


class LoadBiaCellTableTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.table = Path(self._tmp.name) / "cells.csv"
        self.table.write_text(CSV_BODY, encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def _viewer(self, data_source):
        viewer = MagicMock()
        viewer.data_source = data_source
        viewer._data_source = data_source
        viewer._map_mode_active = False
        return viewer

    def test_fetches_the_subset_and_delegates_to_load_cell_table(self):
        data_source = SimpleNamespace(
            has_cell_table=True, fetch_cell_table=MagicMock(return_value=str(self.table))
        )
        viewer = self._viewer(data_source)

        with patch("ueler.runner._load_display_helpers") as load_display:
            load_display.return_value = (MagicMock(), MagicMock())
            returned = load_bia_cell_table(viewer, fovs=["fov1"])

        data_source.fetch_cell_table.assert_called_once_with(["fov1"], force=False)
        viewer.load_cell_table_from_path.assert_called_once_with(str(self.table))
        viewer.after_all_plugins_loaded.assert_called_once()
        self.assertIs(returned, viewer)

    def test_force_is_forwarded(self):
        data_source = SimpleNamespace(
            has_cell_table=True, fetch_cell_table=MagicMock(return_value=str(self.table))
        )
        viewer = self._viewer(data_source)
        with patch("ueler.runner._load_display_helpers") as load_display:
            load_display.return_value = (MagicMock(), MagicMock())
            load_bia_cell_table(viewer, force=True, auto_display=False, after_plugins=False)
        data_source.fetch_cell_table.assert_called_once_with(None, force=True)

    def test_local_viewer_is_rejected_with_an_actionable_error(self):
        viewer = self._viewer(None)
        with self.assertRaises(ValueError) as ctx:
            load_bia_cell_table(viewer)
        self.assertIn("run_viewer_bia", str(ctx.exception))

    def test_study_without_a_cell_table_is_rejected(self):
        viewer = self._viewer(SimpleNamespace(has_cell_table=False))
        with self.assertRaises(ValueError) as ctx:
            load_bia_cell_table(viewer)
        self.assertIn("cell_table", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
