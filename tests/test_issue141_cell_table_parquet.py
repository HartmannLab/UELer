"""Tests for the column-lazy Parquet cell table (issue #141, follow-up to #140).

Three things are worth testing here and two of them are unusual:

1. The ordinary unit behaviour — schema without data, ``ensure_columns``
   idempotence, ``InMemorySource`` equivalence with the eager path.
2. That the converted call sites enumerate the **schema**.  A dropdown built from
   the frame looks fine in every eager test and silently offers four columns out
   of fifty against a lazy table.
3. **Byte budgets.**  ``test_budget_*`` serve a real Parquet file over a local
   HTTP server that supports range requests and count the bytes that cross it.
   This is the only kind of test that catches the actual regression risk: one
   stray whole-table read makes the feature pointless while every other test
   stays green.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Sequence
from unittest import mock

import numpy as np
import pandas as pd

from ueler.cell_table import (
    categorical_columns,
    ensure_table_columns,
    table_columns,
    table_has_column,
    table_schema,
)
from ueler.cell_table_source import (
    InMemorySource,
    ParquetSource,
    guarded_frame,
    is_parquet_path,
    open_parquet_source,
)


N_ROWS = 600
MARKERS = [f"CD{index}" for index in (3, 4, 8, 20, 45, 68)]


def make_frame(n_rows: int = N_ROWS, markers: Sequence[str] = MARKERS) -> pd.DataFrame:
    """A cell table shaped like a real one: spine, markers, and a class column."""
    rng = np.random.default_rng(1410)
    n_fovs = 12
    frame = pd.DataFrame(
        {
            "fov": [f"fov{index % n_fovs}" for index in range(n_rows)],
            "cell_id": np.arange(1, n_rows + 1, dtype="int64"),
            "X": rng.random(n_rows) * 1024,
            "Y": rng.random(n_rows) * 1024,
        }
    )
    for marker in markers:
        frame[marker] = rng.random(n_rows).astype("float32")
    frame["lineage"] = [("tumour", "immune", "stroma")[index % 3] for index in range(n_rows)]
    return frame.sort_values("fov", kind="stable").reset_index(drop=True)


def write_parquet(frame: pd.DataFrame, path: str, *, row_groups: int = 4) -> str:
    from tools.cell_table_to_parquet import convert

    csv_path = path + ".csv"
    frame.to_csv(csv_path, index=False)
    convert(csv_path, path, row_groups=row_groups, quiet=True)
    return path


class _RangeHandler(BaseHTTPRequestHandler):
    """A static file server that honours ``Range``, and counts what it serves.

    ``SimpleHTTPRequestHandler`` answers every request with the whole file, which
    would make a byte budget meaningless — pyarrow's range requests have to be
    answered as ranges or the measurement is of nothing.
    """

    directory = ""
    counters = None
    lock = None

    def log_message(self, *args):  # pragma: no cover - silence the test run
        pass

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's spelling
        self._serve(body=True)

    def do_HEAD(self):  # noqa: N802
        self._serve(body=False)

    def _serve(self, *, body: bool) -> None:
        path = os.path.join(self.directory, self.path.lstrip("/"))
        if not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        header = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if header and header.startswith("bytes="):
            first, _, last = header[len("bytes="):].partition("-")
            if first:
                start = int(first)
                end = int(last) if last else size - 1
                partial = True
        end = min(end, size - 1)
        length = max(0, end - start + 1)

        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        if self.counters is not None:
            with self.lock:
                self.counters["requests"] += 1
        if not body:
            return

        # Count what actually crosses the wire, not the declared Content-Length.
        # fsspec probes a file's size with an unranged GET and aborts as soon as it
        # has the headers, so charging the budget for the whole body would make
        # every measurement here meaningless -- it read a few hundred bytes.
        sent = 0
        with open(path, "rb") as handle:
            handle.seek(start)
            remaining = length
            try:
                while remaining > 0:
                    chunk = handle.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    sent += len(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):  # pragma: no cover
                # Normal: the client had what it asked for and hung up.
                pass
        if self.counters is not None:
            with self.lock:
                self.counters["bytes"] += sent


class _RangeServer:
    """Context manager around :class:`_RangeHandler` on an ephemeral port."""

    def __init__(self, directory: str):
        self.directory = directory
        self.counters = {"requests": 0, "bytes": 0}
        self.lock = threading.Lock()

    def __enter__(self):
        handler = type(
            "_BoundHandler",
            (_RangeHandler,),
            {"directory": self.directory, "counters": self.counters, "lock": self.lock},
        )
        # Threaded on purpose: the concurrent prefetch issues its ranges in
        # parallel, and a single-threaded server would queue them and measure the
        # harness rather than the code.
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        return False

    def reset(self):
        with self.lock:
            self.counters["requests"] = 0
            self.counters["bytes"] = 0


@contextlib.contextmanager
def _spy_on_warm_cache():
    """Record what ``_warm_cache`` returned, without changing what it does.

    ``unittest.mock`` has no return-value spy (that is ``pytest-mock``'s
    ``spy_return``), and the distinction the tests need — the prefetch ran, versus
    it stood aside — lives entirely in the return value.
    """
    import ueler.cell_table_source as module

    original = module._warm_cache
    record = {"returns": []}

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        record["returns"].append(result)
        return result

    module._warm_cache = spy
    try:
        yield record
    finally:
        module._warm_cache = original


class _FakeViewer:
    """The viewer surface the helpers and the converted call sites rely on.

    Mirrors ``ImageMaskViewer``'s real contract: ``cell_table`` is the materialised
    frame, ``cell_table_schema`` is the truth about which columns exist, and
    ``ensure_columns`` moves a column from the second to the first.
    """

    def __init__(self, source, spine):
        self._source = source
        self.cell_table = source.read_columns(spine)
        self.cell_table_adata = None
        self.cell_table_columns = None
        self.fov_key = "fov"
        self.label_key = "cell_id"
        self.x_key = "X"
        self.y_key = "Y"

    @property
    def cell_table_schema(self):
        schema = dict(self._source.schema)
        schema.update({str(n): d for n, d in self.cell_table.dtypes.items()})
        return schema

    def ensure_columns(self, names):
        missing = [
            name
            for name in names
            if name and name not in self.cell_table.columns and name in self._source.schema
        ]
        if missing:
            data = self._source.read_columns(missing)
            for name in missing:
                values = data[name]
                values.index = self.cell_table.index
                self.cell_table[name] = values
        return self.cell_table


class ParquetSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ueler-parquet-")
        cls.frame = make_frame()
        cls.path = write_parquet(cls.frame, os.path.join(cls.tmp, "cells.parquet"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_is_parquet_path(self):
        self.assertTrue(is_parquet_path("a/b/cells.parquet"))
        self.assertTrue(is_parquet_path("CELLS.PQ"))
        self.assertFalse(is_parquet_path("cells.csv"))
        self.assertFalse(is_parquet_path("cells.h5ad"))

    def test_schema_matches_the_frame_without_reading_it(self):
        source = ParquetSource(self.path)
        try:
            self.assertEqual(list(source.schema), list(self.frame.columns))
            self.assertEqual(source.n_rows, len(self.frame))
            self.assertTrue(source.is_lazy)
        finally:
            source.close()

    def test_read_columns_returns_every_row_in_file_order(self):
        source = ParquetSource(self.path)
        try:
            got = source.read_columns(["CD45", "fov"])
            self.assertEqual(list(got.columns), ["CD45", "fov"])
            self.assertEqual(len(got), len(self.frame))
            np.testing.assert_allclose(got["CD45"].to_numpy(), self.frame["CD45"].to_numpy())
            self.assertEqual(list(got.index), list(range(len(self.frame))))
        finally:
            source.close()

    def test_read_columns_rejects_an_unknown_column(self):
        source = ParquetSource(self.path)
        try:
            with self.assertRaises(KeyError) as ctx:
                source.read_columns(["not_a_column"])
            self.assertIn("not_a_column", str(ctx.exception))
        finally:
            source.close()

    def test_spine_columns_keeps_schema_order_and_skips_absent_keys(self):
        source = ParquetSource(self.path)
        try:
            spine = source.spine_columns(["Y", "fov", "cell_id", "mask_name"])
            self.assertEqual(spine, ["fov", "cell_id", "Y"])
        finally:
            source.close()

    def test_close_is_idempotent(self):
        source = ParquetSource(self.path)
        source.close()
        source.close()

    def test_in_memory_source_is_not_lazy_and_mirrors_the_frame(self):
        source = InMemorySource(self.frame)
        self.assertFalse(source.is_lazy)
        self.assertEqual(list(source.schema), list(self.frame.columns))
        self.assertEqual(source.n_rows, len(self.frame))
        pd.testing.assert_frame_equal(source.read_columns(["CD45"]), self.frame[["CD45"]])

    def test_converter_produces_the_requested_row_groups_and_float32(self):
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(self.path)
        self.assertEqual(parquet_file.metadata.num_row_groups, 4)
        dtypes = dict(zip(parquet_file.schema_arrow.names, parquet_file.schema_arrow.types))
        self.assertEqual(str(dtypes["CD45"]), "float")  # float32
        # The low-cardinality string columns are dictionary encoded.
        self.assertIn("dictionary", str(dtypes["lineage"]))

    def test_converter_sorts_by_the_fov_column(self):
        source = ParquetSource(self.path)
        try:
            fovs = source.read_columns(["fov"])["fov"].astype(str).tolist()
            self.assertEqual(fovs, sorted(fovs))
        finally:
            source.close()


class EnsureColumnsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ueler-ensure-")
        cls.frame = make_frame()
        cls.path = write_parquet(cls.frame, os.path.join(cls.tmp, "cells.parquet"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _viewer(self):
        return _FakeViewer(ParquetSource(self.path), ["fov", "cell_id", "X", "Y"])

    def test_schema_covers_everything_while_the_frame_holds_the_spine(self):
        viewer = self._viewer()
        self.assertEqual(list(viewer.cell_table.columns), ["fov", "cell_id", "X", "Y"])
        self.assertEqual(table_columns(viewer), list(self.frame.columns))
        for marker in MARKERS:
            self.assertTrue(table_has_column(viewer, marker))
            self.assertNotIn(marker, viewer.cell_table.columns)

    def test_ensure_materialises_and_is_idempotent(self):
        viewer = self._viewer()
        ensure_table_columns(viewer, ["CD45"])
        self.assertIn("CD45", viewer.cell_table.columns)
        np.testing.assert_allclose(
            viewer.cell_table["CD45"].to_numpy(), self.frame["CD45"].to_numpy()
        )
        before = viewer.cell_table["CD45"].to_numpy().copy()
        ensure_table_columns(viewer, ["CD45"])
        np.testing.assert_array_equal(viewer.cell_table["CD45"].to_numpy(), before)

    def test_ensure_preserves_the_index_and_the_row_count(self):
        viewer = self._viewer()
        index_before = list(viewer.cell_table.index)
        ensure_table_columns(viewer, ["CD3", "CD8"])
        self.assertEqual(list(viewer.cell_table.index), index_before)
        self.assertEqual(len(viewer.cell_table), len(self.frame))

    def test_ensure_does_not_disturb_a_plugin_added_column(self):
        """FlowSOM clusters live only in the frame; a later join must not lose them."""
        viewer = self._viewer()
        viewer.cell_table["flowsom_cluster"] = np.arange(len(self.frame)) % 7
        ensure_table_columns(viewer, ["CD45", "lineage"])
        self.assertIn("flowsom_cluster", viewer.cell_table.columns)
        np.testing.assert_array_equal(
            viewer.cell_table["flowsom_cluster"].to_numpy(),
            np.arange(len(self.frame)) % 7,
        )
        # ...and the plugin column is in the schema too, after the file's own.
        schema = table_columns(viewer)
        self.assertEqual(schema[-1], "flowsom_cluster")

    def test_ensure_ignores_names_that_are_not_columns(self):
        viewer = self._viewer()
        ensure_table_columns(viewer, ["nope", None, ""])
        self.assertEqual(list(viewer.cell_table.columns), ["fov", "cell_id", "X", "Y"])

    def test_materialised_column_keeps_its_categorical_dtype(self):
        viewer = self._viewer()
        ensure_table_columns(viewer, ["lineage"])
        self.assertIsInstance(viewer.cell_table["lineage"].dtype, pd.CategoricalDtype)
        self.assertEqual(
            sorted(map(str, viewer.cell_table["lineage"].unique())),
            ["immune", "stroma", "tumour"],
        )

    def test_helpers_fall_back_to_the_frame_for_an_eager_viewer(self):
        """The plugin test doubles have no schema; a converted site must still work."""

        class Plain:
            cell_table = make_frame(10)

        viewer = Plain()
        self.assertEqual(table_columns(viewer), list(viewer.cell_table.columns))
        self.assertTrue(table_has_column(viewer, "CD45"))
        self.assertFalse(table_has_column(viewer, "absent"))
        self.assertIs(ensure_table_columns(viewer, ["CD45"]), viewer.cell_table)

    def test_categorical_columns_agree_between_frame_and_schema(self):
        viewer = self._viewer()
        from_schema = categorical_columns(table_schema(viewer))
        from_frame = categorical_columns(self.frame)
        self.assertEqual(sorted(from_schema), sorted(from_frame))


class GuardedFrameTests(unittest.TestCase):
    def test_membership_of_an_unmaterialised_column_warns_and_stays_false(self):
        frame = make_frame(5)[["fov", "cell_id"]]
        guarded = guarded_frame(frame, {"fov": "object", "cell_id": "int64", "CD45": "float32"})
        with self.assertLogs("ueler.cell_table_source", level="WARNING") as logs:
            self.assertFalse("CD45" in guarded)
        self.assertIn("CD45", "\n".join(logs.output))

    def test_a_materialised_column_does_not_warn(self):
        frame = make_frame(5)[["fov", "cell_id"]]
        guarded = guarded_frame(frame, {"fov": "object", "cell_id": "int64"})
        self.assertTrue("fov" in guarded)

    def test_the_guarded_frame_is_still_a_dataframe(self):
        frame = make_frame(5)
        guarded = guarded_frame(frame, {})
        self.assertIsInstance(guarded, pd.DataFrame)
        pd.testing.assert_frame_equal(pd.DataFrame(guarded), frame)


class ByteBudgetTests(unittest.TestCase):
    """What a lazy table actually costs, measured over a range-serving HTTP server.

    The numbers below are budgets, not equalities: they are loose enough to
    survive a pyarrow or fsspec upgrade and tight enough that materialising the
    whole table would blow every one of them.
    """

    #: 40 markers, so the spine is 4 columns out of 44 — the shape of a real
    #: study table.  With the six-marker fixture the spine would be a third of the
    #: file and every budget below would measure the fixture, not the design.
    WIDE_MARKERS = [f"M{index:02d}" for index in range(40)]

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ueler-budget-")
        cls.frame = make_frame(20_000, markers=cls.WIDE_MARKERS)
        cls.path = write_parquet(
            cls.frame, os.path.join(cls.tmp, "cells.parquet"), row_groups=4
        )
        cls.size = os.path.getsize(cls.path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_budget_opening_the_schema_does_not_fetch_the_table(self):
        with _RangeServer(self.tmp) as server:
            source = open_parquet_source(f"{server.url}/cells.parquet")
            try:
                self.assertEqual(list(source.schema), list(self.frame.columns))
                self.assertEqual(source.n_rows, len(self.frame))
            finally:
                source.close()
            # Measured: 1.5% of the file, 3 requests.
            self.assertLess(
                server.counters["bytes"],
                self.size * 0.05,
                "reading the schema must not fetch any appreciable part of the file",
            )

    def test_budget_one_column_costs_about_one_column(self):
        with _RangeServer(self.tmp) as server:
            source = open_parquet_source(f"{server.url}/cells.parquet")
            try:
                server.reset()
                got = source.read_columns(["M07"])
            finally:
                source.close()
            self.assertEqual(len(got), len(self.frame))
            np.testing.assert_allclose(
                got["M07"].to_numpy(), self.frame["M07"].to_numpy()
            )
            # 45 columns in the file; measured at 5.4% for one of them.  The
            # budget allows generous headroom and still fails loudly if a stray
            # call materialises the table.
            self.assertLess(
                server.counters["bytes"],
                self.size * 0.15,
                f"reading one column fetched {server.counters['bytes']} of {self.size} bytes",
            )

    def test_budget_a_second_read_of_the_same_column_is_free_to_the_viewer(self):
        """``ensure_columns`` caches, so the second ask must cost nothing at all."""
        with _RangeServer(self.tmp) as server:
            source = open_parquet_source(f"{server.url}/cells.parquet")
            try:
                viewer = _FakeViewer(source, ["fov", "cell_id"])
                ensure_table_columns(viewer, ["M07"])
                server.reset()
                ensure_table_columns(viewer, ["M07"])
            finally:
                source.close()
            self.assertEqual(server.counters["requests"], 0)
            self.assertEqual(server.counters["bytes"], 0)

    def test_budget_opening_a_viewer_costs_the_spine_not_the_table(self):
        with _RangeServer(self.tmp) as server:
            source = open_parquet_source(f"{server.url}/cells.parquet")
            try:
                viewer = _FakeViewer(source, ["fov", "cell_id", "X", "Y"])
            finally:
                source.close()
            self.assertEqual(len(viewer.cell_table), len(self.frame))
            self.assertEqual(
                list(viewer.cell_table.columns), ["fov", "cell_id", "X", "Y"]
            )
            # Measured: 14% — the spine is 4 columns of 45, and the coordinates do
            # not compress.
            self.assertLess(
                server.counters["bytes"],
                self.size * 0.30,
                "opening the viewer must cost the spine, not the table",
            )

    def test_budget_materialising_everything_does_fetch_everything(self):
        """The counterweight: without it the budgets above could pass vacuously.

        If the harness were mismeasuring — counting nothing, or the server were
        answering ranges with empty bodies — every budget would pass and the
        feature could be entirely broken.  Reading all 45 columns has to cost the
        whole file.
        """
        with _RangeServer(self.tmp) as server:
            source = open_parquet_source(f"{server.url}/cells.parquet")
            try:
                names = list(source.schema)
                server.reset()
                got = source.read_columns(names)
            finally:
                source.close()
            self.assertEqual(len(got.columns), len(self.frame.columns))
            self.assertGreater(server.counters["bytes"], self.size * 0.9)

    def test_the_prefetch_is_used_and_fetches_exact_ranges(self):
        """One concurrent batch, and not a byte more than the column chunks."""
        with _RangeServer(self.tmp) as server:
            source = open_parquet_source(f"{server.url}/cells.parquet")
            try:
                with _spy_on_warm_cache() as spy:
                    server.reset()
                    source.read_columns(["M07"])
            finally:
                source.close()
            self.assertEqual(len(spy["returns"]), 1)
            warmed = spy["returns"][0]
            self.assertIsNotNone(warmed, "the prefetch fast path was skipped")
            exact = sum(end - start for start, end in warmed)
            self.assertEqual(server.counters["bytes"], exact)

    def test_the_prefetch_stands_aside_for_a_whole_table_read(self):
        """Fetching most of the file as hundreds of ranges is worse than reading it."""
        with _RangeServer(self.tmp) as server:
            source = open_parquet_source(f"{server.url}/cells.parquet")
            try:
                with _spy_on_warm_cache() as spy:
                    source.read_columns(list(source.schema))
            finally:
                source.close()
            self.assertEqual(spy["returns"], [None])

    def test_prefetch_failure_falls_back_to_a_plain_read(self):
        """The concurrent prefetch is an optimisation; a broken one must not break a plot."""
        with _RangeServer(self.tmp) as server:
            source = open_parquet_source(f"{server.url}/cells.parquet")
            try:
                with mock.patch(
                    "ueler.cell_table_source._warm_cache", side_effect=RuntimeError("boom")
                ):
                    got = source.read_columns(["M07"])
            finally:
                source.close()
            np.testing.assert_allclose(
                got["M07"].to_numpy(), self.frame["M07"].to_numpy()
            )


class ViewerIntegrationTests(unittest.TestCase):
    """The real ``ImageMaskViewer`` methods, against a real Parquet file."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ueler-viewer-parquet-")
        cls.frame = make_frame()
        cls.path = write_parquet(cls.frame, os.path.join(cls.tmp, "cells.parquet"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _viewer(self, *, debug: bool = False):
        """A viewer with the real methods but without the heavy image-folder init."""
        from ueler.viewer.main_viewer import ImageMaskViewer

        class _Viewer(ImageMaskViewer):
            def __init__(self):  # noqa: D107 - deliberately skips the base __init__
                self.cell_table = None
                self.cell_table_adata = None
                self.cell_table_columns = None
                self._cell_table_source = None
                self.fov_key = "fov"
                self.label_key = "cell_id"
                self.x_key = "X"
                self.y_key = "Y"
                self.mask_key = "cell"
                self._debug = debug

        return _Viewer()

    def test_load_from_path_routes_parquet_to_the_lazy_source(self):
        viewer = self._viewer()
        viewer.load_cell_table_from_path(self.path)
        self.assertTrue(viewer.cell_table_source.is_lazy)
        self.assertEqual(len(viewer.cell_table), len(self.frame))
        self.assertEqual(
            sorted(viewer.cell_table.columns), sorted(["fov", "cell_id", "X", "Y"])
        )
        self.assertEqual(list(viewer.cell_table_schema), list(self.frame.columns))
        self.assertEqual(sorted(viewer.unmaterialised_columns), sorted(MARKERS + ["lineage"]))

    def test_ensure_columns_on_the_viewer(self):
        viewer = self._viewer()
        viewer.load_cell_table_from_path(self.path)
        viewer.ensure_columns(["CD8"])
        self.assertIn("CD8", viewer.cell_table.columns)
        np.testing.assert_allclose(
            viewer.cell_table["CD8"].to_numpy(), self.frame["CD8"].to_numpy()
        )
        self.assertNotIn("CD8", viewer.unmaterialised_columns)

    def test_ensure_all_columns_reproduces_the_whole_table(self):
        viewer = self._viewer()
        viewer.load_cell_table_from_path(self.path)
        viewer.ensure_all_columns()
        self.assertEqual(viewer.unmaterialised_columns, [])
        got = viewer.cell_table[list(self.frame.columns)].reset_index(drop=True)
        expected = self.frame
        # The converter casts floats to float32 and strings to categories, so
        # compare values at float32 precision rather than dtypes.
        for column in expected.columns:
            if pd.api.types.is_numeric_dtype(expected[column]):
                np.testing.assert_allclose(
                    got[column].to_numpy(dtype="float64"),
                    expected[column].to_numpy(dtype="float64"),
                    rtol=1e-6,
                )
            else:
                np.testing.assert_array_equal(
                    got[column].astype(str).to_numpy(),
                    expected[column].astype(str).to_numpy(),
                )

    def test_get_cell_table_adata_materialises_first(self):
        """``X`` is "the numeric non-key columns" — a lazy table must be complete."""
        viewer = self._viewer()
        viewer.load_cell_table_from_path(self.path)
        adata = viewer.get_cell_table_adata()
        self.assertEqual(adata.n_obs, len(self.frame))
        self.assertEqual(sorted(adata.var_names), sorted(MARKERS))

    def test_a_csv_table_attaches_no_source_and_behaves_as_before(self):
        csv_path = os.path.join(self.tmp, "cells.csv")
        self.frame.to_csv(csv_path, index=False)
        viewer = self._viewer()
        viewer.load_cell_table_from_path(csv_path)
        self.assertIsNone(viewer.cell_table_source)
        self.assertEqual(list(viewer.cell_table.columns), list(self.frame.columns))
        self.assertEqual(viewer.unmaterialised_columns, [])
        self.assertEqual(list(viewer.cell_table_schema), list(self.frame.columns))
        # ``ensure_columns`` is a no-op rather than an error on an eager table.
        self.assertIs(viewer.ensure_columns(["CD45"]), viewer.cell_table)

    def test_replacing_the_table_releases_the_previous_source(self):
        viewer = self._viewer()
        viewer.load_cell_table_from_path(self.path)
        source = viewer.cell_table_source
        with mock.patch.object(source, "close", wraps=source.close) as closed:
            viewer.set_cell_table(self.frame)
        closed.assert_called_once()
        self.assertIsNone(viewer.cell_table_source)

    def test_update_keys_materialises_a_retargeted_key(self):
        viewer = self._viewer()
        viewer.load_cell_table_from_path(self.path)
        viewer.ui_component = SimpleNamespace(
            x_key=SimpleNamespace(value="X"),
            y_key=SimpleNamespace(value="Y"),
            label_key=SimpleNamespace(value="lineage"),
            fov_key=SimpleNamespace(value="fov"),
        )
        viewer.update_keys(None)
        self.assertIn("lineage", viewer.cell_table.columns)

    def test_debug_mode_wraps_the_frame_in_the_guard(self):
        viewer = self._viewer(debug=True)
        viewer.load_cell_table_from_path(self.path)
        with self.assertLogs("ueler.cell_table_source", level="WARNING") as logs:
            self.assertFalse("CD45" in viewer.cell_table)
        self.assertIn("CD45", "\n".join(logs.output))

    def test_parquet_rejects_anndata_only_arguments(self):
        viewer = self._viewer()
        with self.assertRaises(ValueError):
            viewer.load_cell_table_from_path(self.path, layer="counts")


class ConvertedCallSiteTests(unittest.TestCase):
    """The enumeration sites must offer every column, not the materialised ones.

    Each of these fails against the pre-#141 code, which is the point: they are
    the regression guard for the seventeen sites the refactor touched.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ueler-sites-")
        cls.frame = make_frame()
        cls.path = write_parquet(cls.frame, os.path.join(cls.tmp, "cells.parquet"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _viewer(self):
        return _FakeViewer(ParquetSource(self.path), ["fov", "cell_id", "X", "Y"])

    def test_numeric_columns_offers_every_marker(self):
        from ueler.viewer.plugin import _chart_common

        viewer = self._viewer()
        offered = _chart_common.numeric_columns(viewer)
        for marker in MARKERS:
            self.assertIn(marker, offered)
        self.assertNotIn("lineage", offered)

    def test_subset_options_materialise_the_column_they_read(self):
        from ueler.viewer.plugin import _chart_common

        viewer = self._viewer()
        options = _chart_common.subset_options_for(viewer, "lineage")
        self.assertEqual(sorted(map(str, options)), ["immune", "stroma", "tumour"])
        self.assertIn("lineage", viewer.cell_table.columns)

    def test_subset_options_for_an_unknown_column_stay_empty(self):
        from ueler.viewer.plugin import _chart_common

        self.assertEqual(_chart_common.subset_options_for(self._viewer(), "nope"), [])

    def test_prepare_dataframe_materialises_what_it_filters_on(self):
        from ueler.viewer.plugin import _chart_common

        viewer = self._viewer()
        viewer.ui_component = SimpleNamespace(
            image_selector=SimpleNamespace(value="fov1")
        )
        got = _chart_common.prepare_dataframe(
            viewer,
            subset_on="lineage",
            subset_values=["tumour"],
            impose_fov=False,
            columns=["CD45"],
        )
        self.assertIn("CD45", got.columns)
        self.assertIn("lineage", viewer.cell_table.columns)
        self.assertTrue((got["lineage"].astype(str) == "tumour").all())

    def test_flowsom_offers_every_column_as_a_feature(self):
        from ueler.viewer.plugin.run_flowsom import _feature_columns

        self.assertEqual(sorted(_feature_columns(self._viewer())), sorted(self.frame.columns))


class BiaParquetTests(unittest.TestCase):
    """The BIA descriptor and runner route a Parquet table to the lazy source."""

    def _source(self, cell_table):
        from ueler.bia_loader import BIADataSource, BIAStudyIndex

        class _Index:
            base_url = "https://example.org/study"

            def url_for(self, rel_path):
                return f"{self.base_url}/{rel_path.strip('/')}"

            def list_dir(self, rel_path=""):
                return []

            def list_subdirs(self, rel_path=""):
                return []

            def list_files(self, rel_path=""):
                return []

        descriptor = {"mode": "folder", "base": "Files/img", "cell_table": cell_table}
        with tempfile.TemporaryDirectory() as cache:
            with mock.patch.object(BIAStudyIndex, "from_source", return_value=_Index()):
                source = BIADataSource(
                    "https://example.org/study", cache_dir=cache, descriptor=descriptor
                )
        source.index = _Index()
        return source

    def test_cell_table_is_parquet_reflects_the_suffix(self):
        self.assertTrue(self._source("Files/t/cells.parquet").cell_table_is_parquet)
        self.assertTrue(self._source("Files/t/cells.PQ").cell_table_is_parquet)
        self.assertFalse(self._source("Files/t/cells.csv").cell_table_is_parquet)

    def test_open_cell_table_source_is_none_for_a_csv(self):
        self.assertIsNone(self._source("Files/t/cells.csv").open_cell_table_source())

    def test_open_cell_table_source_opens_the_resolved_url(self):
        source = self._source("Files/t/cells.parquet")
        with mock.patch("ueler.cell_table_source.open_parquet_source") as opener:
            opener.return_value = mock.sentinel.parquet_source
            got = source.open_cell_table_source()
        opener.assert_called_once_with("https://example.org/study/Files/t/cells.parquet")
        self.assertIs(got, mock.sentinel.parquet_source)

    def test_load_bia_cell_table_streams_a_parquet_table(self):
        from ueler.runner import load_bia_cell_table

        opened = mock.Mock(name="ParquetSource")
        data_source = mock.Mock(
            has_cell_table=True,
            cell_table_is_parquet=True,
            open_cell_table_source=mock.Mock(return_value=opened),
        )
        viewer = mock.MagicMock()
        viewer.data_source = data_source
        viewer._data_source = data_source
        viewer._map_mode_active = False

        with mock.patch("ueler.runner._load_display_helpers") as helpers:
            helpers.return_value = (mock.MagicMock(), mock.MagicMock())
            returned = load_bia_cell_table(viewer)

        viewer.set_parquet_cell_table.assert_called_once_with(opened)
        # Nothing is downloaded and nothing is loaded from a path.
        data_source.fetch_cell_table.assert_not_called()
        viewer.load_cell_table_from_path.assert_not_called()
        viewer.after_all_plugins_loaded.assert_called_once()
        self.assertIs(returned, viewer)

    def test_a_fovs_filter_is_ignored_for_parquet_rather_than_failing(self):
        """Every row is there anyway, so asking for a subset is merely unnecessary."""
        from ueler.runner import load_bia_cell_table

        data_source = mock.Mock(
            has_cell_table=True,
            cell_table_is_parquet=True,
            open_cell_table_source=mock.Mock(return_value=mock.sentinel.src),
        )
        viewer = mock.MagicMock()
        viewer.data_source = data_source
        viewer._data_source = data_source
        viewer._map_mode_active = False

        with mock.patch("ueler.runner._load_display_helpers") as helpers:
            helpers.return_value = (mock.MagicMock(), mock.MagicMock())
            load_bia_cell_table(viewer, fovs=["fov1", "fov2"])

        data_source.open_cell_table_source.assert_called_once_with()
        data_source.fetch_cell_table.assert_not_called()

    def test_a_csv_table_still_takes_the_download_path(self):
        from ueler.runner import load_bia_cell_table

        with tempfile.TemporaryDirectory() as tmp:
            table = os.path.join(tmp, "cells.csv")
            make_frame(5).to_csv(table, index=False)
            data_source = mock.Mock(
                has_cell_table=True,
                cell_table_is_parquet=False,
                fetch_cell_table=mock.Mock(return_value=table),
            )
            viewer = mock.MagicMock()
            viewer.data_source = data_source
            viewer._data_source = data_source
            viewer._map_mode_active = False

            with mock.patch("ueler.runner._load_display_helpers") as helpers:
                helpers.return_value = (mock.MagicMock(), mock.MagicMock())
                load_bia_cell_table(viewer, fovs=["fov1"])

        data_source.fetch_cell_table.assert_called_once_with(["fov1"], force=False)
        viewer.load_cell_table_from_path.assert_called_once_with(table)
        viewer.set_parquet_cell_table.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
