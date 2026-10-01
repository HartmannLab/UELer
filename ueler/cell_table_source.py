"""Column-lazy backing stores for the viewer's cell table.

A study-scale cell table is wide, not just long: ``S-BIAD2557``'s table is 439,339
rows by 50 columns, 361 MB as CSV and 402 MB once pandas has parsed it.  The
viewer needs the *column names and dtypes* immediately — every marker, cluster and
annotation dropdown in the GUI is built from them — but it only ever plots one or
two columns at a time.  Paying for all fifty to populate a dropdown is what makes
a large table unusable in a memory-constrained session such as Binder.

This module introduces the one new idea needed to fix that: **a column may not be
here yet**.

Design: column-lazy, row-complete
---------------------------------

``ImageMaskViewer.cell_table`` stays a genuine ``pandas.DataFrame`` for its whole
life.  It starts as the *spine* (the FOV, label and coordinate keys) and **gains
whole columns**; rows are never dropped and the index never changes.  Three
consequences, and they are the reason for this shape rather than a lazier one:

* Every existing row filter keeps working untouched — the dozen
  ``cell_table[fov_key] == current_fov`` sites in the plugins, plus every
  ``.loc``, boolean mask and ``.to_numpy()`` call.
* Global statistics stay **exact**.  ``MaskPainter._compute_auto_range`` computes
  a global percentile range over a whole column; the heatmap, FlowSOM and the
  histogram gates are equally whole-table.  A row-lazy table would silently
  redefine all of them as "over the FOVs you happen to have opened", and the mask
  colours would shift as the user browsed.
* No new execution model — no deferred graph, no ``.compute()``.

Sources
-------

* :class:`InMemorySource` wraps an already-materialised frame, so today's
  CSV/AnnData/DataFrame paths become the degenerate case of the new one and need
  no branch anywhere else.
* :class:`ParquetSource` holds one open ``pq.ParquetFile`` for the session and
  reads column chunks on demand, from a local path or from an fsspec handle over
  HTTP.  Given a filesystem it can also fetch the chunks of several columns
  concurrently with ``cat_ranges`` (see :func:`_warm_cache`), which on a WAN is
  the difference between one round trip and one per row group.

Measured against the real table, converted to float32 with dictionary-encoded
strings and 9 row groups: the schema costs 0.13 MB in 2 requests, the spine
6.06 MB in 18, and one full marker column 2.41 MB in 9 — against 91 MB for the
whole file.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "CellTableSource",
    "InMemorySource",
    "ParquetSource",
    "PARQUET_SUFFIXES",
    "guarded_frame",
    "is_parquet_path",
    "open_parquet_source",
]

#: Suffixes routed to :class:`ParquetSource` by the loaders.
PARQUET_SUFFIXES = (".parquet", ".pq")

#: Columns read eagerly when a lazy table is attached, if they exist: without the
#: FOV/label/coordinate keys the viewer cannot locate a single cell.
DEFAULT_SPINE = ("fov", "cell_id", "label", "centroid-0", "centroid-1", "X", "Y")


def is_parquet_path(path: Any) -> bool:
    """Whether *path* names a Parquet file, by suffix."""
    return str(path).lower().endswith(PARQUET_SUFFIXES)


class CellTableSource:
    """The store a viewer's cell table is read from.

    Subclasses provide :attr:`schema` (an ordered ``{name: dtype}`` mapping that is
    available *without reading any data*), :attr:`n_rows`, and
    :meth:`read_columns`.  ``is_lazy`` tells the viewer whether asking for a column
    can cost anything; it is ``False`` for :class:`InMemorySource`, which is what
    keeps the eager paths free of new behaviour.
    """

    #: Whether reading a column may do I/O.
    is_lazy = False

    @property
    def schema(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def n_rows(self) -> int:
        raise NotImplementedError

    def read_columns(self, names: Sequence[str]) -> pd.DataFrame:
        """Return a frame of *names*, all rows, in file order."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any handle held open for the session.  Idempotent."""

    # -- convenience --------------------------------------------------------
    def spine_columns(self, preferred: Optional[Iterable[str]] = None) -> List[str]:
        """Return the columns to materialise eagerly, in schema order.

        *preferred* names the viewer's key columns; whichever of them the schema
        actually has are returned, falling back to :data:`DEFAULT_SPINE`.
        """
        wanted = [str(name) for name in (preferred or DEFAULT_SPINE) if name]
        schema = self.schema
        seen = set()
        ordered = []
        for name in schema:
            if name in wanted and name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered


class InMemorySource(CellTableSource):
    """A source over a frame that is already fully materialised.

    Reading a column is a lookup, so ``is_lazy`` is ``False`` and the viewer skips
    the laziness machinery entirely.  This exists so that every call site can talk
    to a source unconditionally instead of branching on "is this table lazy".
    """

    is_lazy = False

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    @property
    def schema(self) -> Dict[str, Any]:
        return {str(name): dtype for name, dtype in self._frame.dtypes.items()}

    @property
    def n_rows(self) -> int:
        return len(self._frame)

    def read_columns(self, names: Sequence[str]) -> pd.DataFrame:
        return self._frame[[str(name) for name in names]]


def _pandas_dtypes(arrow_schema) -> Dict[str, Any]:
    """Map an Arrow schema to the pandas dtypes ``to_pandas`` would produce.

    Converting the *empty* table is exact and costs no I/O, which beats
    maintaining a hand-written Arrow→pandas type table that would drift from
    whatever pyarrow does with dictionaries, nullable integers and timestamps.
    """
    empty = arrow_schema.empty_table().to_pandas()
    return {str(name): dtype for name, dtype in empty.dtypes.items()}


def _column_ranges(metadata, column_indices: Sequence[int]) -> List[tuple]:
    """Byte ranges of the given columns' chunks across every row group.

    A column chunk starts at its dictionary page when it has one (dictionary
    encoding is what makes the string columns cheap) and at its first data page
    otherwise; ``total_compressed_size`` covers both.
    """
    ranges = []
    for group in range(metadata.num_row_groups):
        row_group = metadata.row_group(group)
        for index in column_indices:
            chunk = row_group.column(index)
            start = chunk.data_page_offset
            dictionary = chunk.dictionary_page_offset
            if dictionary is not None and dictionary > 0:
                start = min(start, dictionary)
            ranges.append((int(start), int(start + chunk.total_compressed_size)))
    return ranges


#: Above this share of the file, a prefetch is not worth it: fetching most of the
#: bytes as hundreds of separate ranges is slower and no cheaper than letting
#: fsspec read the file through sequentially, which is what ``ensure_all_columns``
#: ends up doing.
PREFETCH_COVERAGE_LIMIT = 0.5


def _warm_cache(fs, path: str, handle, metadata, column_indices: Sequence[int], size: int):
    """Seed *handle*'s cache with the needed column chunks, fetched concurrently.

    ``fs.cat_ranges`` issues the range requests in parallel and returns exactly
    the bytes asked for; seeding them into ``KnownPartsOfAFile`` lets pyarrow read
    the columns without touching the network again.  On a WAN that turns one round
    trip per row group into one batch — the difference between 12 s and 0.3 s for
    a column out of a finely partitioned file.

    The cache is seeded on the **session handle** rather than a fresh one, which
    matters more than it looks: opening a second reader would re-read the footer,
    and on a 50-column table the footer is larger than a column (90 KB against
    70 KB here), so the "optimisation" cost more than it saved.

    Returns the list of ranges it fetched on success and ``None`` when the fast
    path does not apply, in which case the caller does the ordinary sequential
    read.
    """
    try:
        from fsspec.caching import KnownPartsOfAFile
    except ImportError:  # pragma: no cover - fsspec is a hard dependency
        return None

    ranges = _column_ranges(metadata, column_indices)
    if not ranges:
        return None
    covered = sum(end - start for start, end in ranges)
    if size and covered > size * PREFETCH_COVERAGE_LIMIT:
        return None

    starts = [start for start, _ in ranges]
    ends = [end for _, end in ranges]
    blobs = fs.cat_ranges([path] * len(ranges), starts, ends)
    data = {
        (start, end): blob
        for (start, end), blob in zip(ranges, blobs)
        if isinstance(blob, (bytes, bytearray))
    }
    if len(data) != len(ranges):
        return None

    handle.cache = KnownPartsOfAFile(
        blocksize=handle.blocksize,
        fetcher=handle._fetch_range,
        size=size,
        data=data,
        strict=False,
    )
    # ``KnownPartsOfAFile`` consolidates contiguous blocks by *popping* them out
    # of the dict it is given, so ``data`` is empty by now — return the ranges.
    return ranges


class ParquetSource(CellTableSource):
    """A column-lazy source over one Parquet file.

    The file handle is held for the session, so the footer is paid once and every
    later read is a column chunk.  *source* is a local path, an open file object,
    or an fsspec URL; passing *fs* as well enables the concurrent prefetch.
    """

    is_lazy = True

    def __init__(self, source, *, fs=None, path: Optional[str] = None, prefetch: bool = True):
        import pyarrow.parquet as pq

        self._fs = fs
        self._path = path if path is not None else (source if isinstance(source, str) else None)
        self._prefetch = bool(prefetch and fs is not None and self._path)
        self._owns_handle = False

        if fs is not None and self._path and not hasattr(source, "read"):
            # ``cache_type="none"`` keeps fsspec from reading ahead: the whole point
            # is to fetch the requested byte ranges and nothing else.
            handle = fs.open(self._path, "rb", cache_type="none")
            self._owns_handle = True
            self._handle = handle
        else:
            self._handle = source

        self._file = pq.ParquetFile(self._handle)
        self._metadata = self._file.metadata
        self._schema = _pandas_dtypes(self._file.schema_arrow)
        self._names = list(self._schema)
        self._size = None
        if self._prefetch:
            try:
                self._size = int(fs.size(self._path))
            except Exception:  # pragma: no cover - filesystem without size()
                self._prefetch = False

        logger.debug(
            "[cell table] Parquet source opened: %d rows, %d columns, %d row groups.",
            self._metadata.num_rows,
            len(self._names),
            self._metadata.num_row_groups,
        )

    # -- CellTableSource ----------------------------------------------------
    @property
    def schema(self) -> Dict[str, Any]:
        return dict(self._schema)

    @property
    def n_rows(self) -> int:
        return int(self._metadata.num_rows)

    @property
    def n_row_groups(self) -> int:
        return int(self._metadata.num_row_groups)

    def read_columns(self, names: Sequence[str]) -> pd.DataFrame:
        wanted = [str(name) for name in names]
        unknown = [name for name in wanted if name not in self._schema]
        if unknown:
            raise KeyError(
                f"Column(s) {unknown} are not in the Parquet cell table; "
                f"available: {self._names}"
            )
        if not wanted:
            return pd.DataFrame(index=pd.RangeIndex(self.n_rows))

        table = self._read_arrow(wanted)
        frame = table.to_pandas()
        frame.index = pd.RangeIndex(len(frame))
        return frame

    def close(self) -> None:
        file_obj, self._file = getattr(self, "_file", None), None
        if file_obj is not None:
            try:
                file_obj.close()
            except Exception:  # pragma: no cover - best effort
                logger.debug("[cell table] closing the Parquet reader failed.", exc_info=True)
        if self._owns_handle:
            try:
                self._handle.close()
            except Exception:  # pragma: no cover - best effort
                logger.debug("[cell table] closing the Parquet handle failed.", exc_info=True)
            self._owns_handle = False

    # -- internals ----------------------------------------------------------
    def _read_arrow(self, wanted: Sequence[str]):
        if self._prefetch:
            table = self._read_prefetched(wanted)
            if table is not None:
                return table
        return self._file.read(columns=list(wanted), use_threads=True)

    def _read_prefetched(self, wanted: Sequence[str]):
        """Read *wanted* through the session handle's cache, warmed in one batch."""
        original = getattr(self._handle, "cache", None)
        try:
            indices = [self._names.index(name) for name in wanted]
            warmed = _warm_cache(
                self._fs, self._path, self._handle, self._metadata, indices, self._size
            )
            if warmed is None:
                return None
            try:
                return self._file.read(columns=list(wanted), use_threads=True)
            finally:
                if original is not None:
                    self._handle.cache = original
        except Exception:
            if original is not None:
                self._handle.cache = original
            # The prefetch is a pure optimisation: any surprise (a server that
            # drops multi-range requests, an fsspec version without cat_ranges)
            # falls back to the plain sequential read rather than failing a plot.
            logger.debug(
                "[cell table] concurrent prefetch failed; falling back to a plain read.",
                exc_info=True,
            )
            return None


def open_parquet_source(path_or_url, *, storage_options: Optional[Mapping] = None) -> ParquetSource:
    """Open *path_or_url* as a :class:`ParquetSource`, local or remote.

    A local path is opened directly; anything with a URL scheme goes through
    fsspec, which also gives the source the filesystem it needs for the
    concurrent prefetch.
    """
    target = str(path_or_url)
    if "://" not in target or target.startswith("file://"):
        return ParquetSource(target.replace("file://", "", 1) if target.startswith("file://") else target)

    import fsspec

    fs, _, paths = fsspec.get_fs_token_paths(target, storage_options=dict(storage_options or {}))
    return ParquetSource(target, fs=fs, path=paths[0])


class _GuardedFrame(pd.DataFrame):
    """A cell table that complains when a guard asks about an unloaded column.

    The realistic failure mode of a column-lazy table is a ``col in
    cell_table.columns`` test that silently answers "no" for a column which exists
    in the file but has not been materialised — the plugin then quietly draws
    nothing.  In ``debug`` mode the frame is wrapped in this subclass, whose
    ``__contains__`` logs a warning naming the column before returning the same
    answer as before.  Behaviour is unchanged; only the silence is.

    ``__contains__`` rather than ``__getitem__`` is deliberate: a missing-column
    ``__getitem__`` already raises loudly, and pandas calls ``__getitem__`` from
    copy, slice and merge paths where a blocking read has no business happening.
    """

    _metadata = ["_lazy_schema"]

    @property
    def _constructor(self):
        return _GuardedFrame

    def __contains__(self, key) -> bool:
        present = super().__contains__(key)
        if not present:
            schema = getattr(self, "_lazy_schema", None) or {}
            if key in schema:
                logger.warning(
                    "[cell table] column %r exists in the cell table but has not been "
                    "materialised; this membership test answered False. The caller "
                    "should go through ensure_columns().",
                    key,
                )
        return present


def guarded_frame(frame: pd.DataFrame, schema: Mapping[str, Any]) -> pd.DataFrame:
    """Wrap *frame* so unmaterialised-column membership tests are logged."""
    guarded = _GuardedFrame(frame)
    guarded._lazy_schema = dict(schema)
    return guarded
