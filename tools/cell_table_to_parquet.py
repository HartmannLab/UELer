#!/usr/bin/env python
"""Convert a cell table (CSV or ``.h5ad``) to the Parquet layout UELer reads lazily.

Why a converter rather than "just write Parquet": the layout decides what the
viewer's access pattern costs, and the defaults here are the ones measured on
``S-BIAD2557``'s 439,339 x 50 table.

* **float32** halves the file and costs nothing a marker intensity cares about:
  361 MB CSV -> 159 MB Parquet at float64 -> 91 MB at float32 with dictionary-
  encoded strings.  ``--no-float32`` opts out.
* **Coarse row groups (16 by default).**  The tempting layout is one row group per
  FOV, so a single FOV can be read without touching the rest.  Measurement says
  that is the wrong trade: the viewer reads *whole* columns (global percentile
  ranges, heatmaps, FlowSOM all need every row), and a 455-row-group file costs
  455 range requests for one 2.2 MB column -- 12 s on a WAN, against 9 requests
  and 0.3 s for the same column out of a 9-row-group file.  Row groups are sized
  for sequential column scans, not for row pruning.
* **Sorted by FOV**, so each row group still covers a contiguous block of FOVs.
  That is what keeps a future per-FOV optimisation possible without making the
  common path slow today.  ``--no-sort`` preserves the input order.
* **zstd**, which beats snappy by ~20% here at a decompression cost that is
  invisible next to the network.

Usage:

    python tools/cell_table_to_parquet.py in.csv out.parquet
    python tools/cell_table_to_parquet.py in.h5ad out.parquet --row-groups 32
    python tools/cell_table_to_parquet.py in.csv out.parquet --fov-column sample_id
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Optional

import pandas as pd

#: A string column with at most this fraction of distinct values is dictionary
#: encoded.  Marker names, FOV ids, tissue zones and lineage labels all sit far
#: below it; a free-text note column would not, and is better left alone.
DICTIONARY_CARDINALITY_RATIO = 0.5

#: Candidate FOV columns, in the order they are tried when ``--fov-column`` is not
#: given.  These are the names the viewer itself looks for.
FOV_COLUMN_CANDIDATES = ("fov", "FOV", "sample_id", "image", "point")


def _human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024 or unit == "GB":
            return f"{n_bytes:,.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:,.1f} GB"


def read_table(path: str) -> pd.DataFrame:
    """Read *path* into a DataFrame, flattening an ``.h5ad`` the way the viewer does."""
    if path.lower().endswith(".h5ad"):
        import anndata

        from ueler.cell_table import flatten_anndata

        frame, _provenance = flatten_anndata(anndata.read_h5ad(path))
        return frame
    return pd.read_csv(path)


def pick_fov_column(frame: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    """Return the column to sort by, or ``None`` when there is nothing obvious."""
    if requested:
        if requested not in frame.columns:
            raise SystemExit(
                f"--fov-column {requested!r} is not in the table; columns are: "
                f"{list(frame.columns)}"
            )
        return requested
    for candidate in FOV_COLUMN_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    return None


def downcast_floats(frame: pd.DataFrame) -> List[str]:
    """Cast every float64 column to float32 in place; return the columns changed."""
    changed = []
    for name in frame.columns:
        if str(frame[name].dtype) == "float64":
            frame[name] = frame[name].astype("float32")
            changed.append(str(name))
    return changed


def dictionary_columns(frame: pd.DataFrame) -> List[str]:
    """Return the string columns worth dictionary encoding."""
    encoded = []
    n_rows = max(len(frame), 1)
    for name in frame.columns:
        dtype = frame[name].dtype
        if str(dtype) == "category":
            encoded.append(str(name))
            continue
        if not (pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)):
            continue
        if frame[name].nunique(dropna=True) / n_rows <= DICTIONARY_CARDINALITY_RATIO:
            encoded.append(str(name))
    return encoded


def convert(
    source: str,
    destination: str,
    *,
    fov_column: Optional[str] = None,
    row_groups: int = 16,
    float32: bool = True,
    sort: bool = True,
    compression: str = "zstd",
    quiet: bool = False,
) -> str:
    """Convert *source* to *destination* and return the destination path."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    def say(message: str) -> None:
        if not quiet:
            print(message)

    frame = read_table(source)
    say(f"read {source}: {len(frame):,} rows x {len(frame.columns)} columns")

    key = pick_fov_column(frame, fov_column)
    if sort and key is not None:
        frame = frame.sort_values(key, kind="stable").reset_index(drop=True)
        say(f"sorted by {key!r} ({frame[key].nunique():,} distinct values)")
    else:
        frame = frame.reset_index(drop=True)
        if sort:
            say("no FOV-like column found; keeping the input row order")

    if float32:
        changed = downcast_floats(frame)
        say(f"cast {len(changed)} float64 column(s) to float32")

    encoded = dictionary_columns(frame)
    for name in encoded:
        if str(frame[name].dtype) != "category":
            frame[name] = frame[name].astype("category")
    if encoded:
        say(f"dictionary encoding {len(encoded)} string column(s): {', '.join(encoded)}")

    table = pa.Table.from_pandas(frame, preserve_index=False)
    row_groups = max(1, int(row_groups))
    row_group_size = max(1, math.ceil(len(frame) / row_groups))

    parent = os.path.dirname(os.path.abspath(destination))
    if parent:
        os.makedirs(parent, exist_ok=True)
    pq.write_table(
        table,
        destination,
        compression=compression,
        row_group_size=row_group_size,
        use_dictionary=True,
        write_statistics=True,
    )

    written = pq.ParquetFile(destination)
    size = os.path.getsize(destination)
    say(
        f"wrote {destination}: {_human(size)}, {written.metadata.num_row_groups} row group(s) "
        f"of ~{row_group_size:,} rows, compression={compression}"
    )
    if not quiet:
        _report_columns(written, size)
    return destination


def _report_columns(parquet_file, total_size: int) -> None:
    """Print the compressed footprint of each column — the cost of materialising it."""
    metadata = parquet_file.metadata
    per_column = {}
    for group in range(metadata.num_row_groups):
        row_group = metadata.row_group(group)
        for index in range(row_group.num_columns):
            chunk = row_group.column(index)
            name = chunk.path_in_schema
            per_column[name] = per_column.get(name, 0) + chunk.total_compressed_size
    print(f"\nper-column footprint ({_human(total_size)} total):")
    for name, size in sorted(per_column.items(), key=lambda item: -item[1]):
        print(f"  {size / total_size:6.1%}  {_human(size):>12}  {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="input cell table (.csv or .h5ad)")
    parser.add_argument("destination", help="output .parquet path")
    parser.add_argument(
        "--fov-column",
        default=None,
        help=f"column to sort row groups on (default: the first of {', '.join(FOV_COLUMN_CANDIDATES)} present)",
    )
    parser.add_argument(
        "--row-groups",
        type=int,
        default=16,
        help="number of row groups (default: 16; coarse is faster for whole-column reads)",
    )
    parser.add_argument("--no-float32", dest="float32", action="store_false", help="keep float64 columns")
    parser.add_argument("--no-sort", dest="sort", action="store_false", help="keep the input row order")
    parser.add_argument("--compression", default="zstd", help="parquet codec (default: zstd)")
    parser.add_argument("--quiet", action="store_true", help="print nothing but errors")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    convert(
        args.source,
        args.destination,
        fov_column=args.fov_column,
        row_groups=args.row_groups,
        float32=args.float32,
        sort=args.sort,
        compression=args.compression,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
