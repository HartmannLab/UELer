"""Saved cell sets: named populations of cells, their tags and how they were made.

A cell set is a selection kept for later. It is stored by **cell identity**
(FOV, mask layer, mask label), not by cell-table row index, so it survives
reloading, filtering or re-sorting the cell table; rows are resolved again from
``(fov_key, label_key)`` when the set is used. The mask layer is kept because a
dataset can have several layers (cell and nucleus masks) that share labels, and
a set made on one layer should be outlined on that layer.

Files live in the per-dataset folder the checkpoints use::

    <storage>/.UELer/dataset_<sha1>/cell_sets/
        index.json          every set's metadata: name, tags, note, recipe, parents, trail
        <set id>.parquet    the set's cells: fov, mask, label

The metadata sits in one index so the Cell sets plugin can list the sets without
reading any cells; the cells go to Parquet because a set can hold hundreds of
thousands of them. Every write goes to a temporary file first and is then
renamed over the old one, so an interrupted write never leaves a half file.

**Tags** are ``key:value`` pairs (``celltype:CD8 T``). The key names a
cell-table column and the value is what the set's cells get in it, so one rule
covers every write: :func:`write_column` puts *value* into column *key* for the
given rows.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from ueler.viewer.checkpoint_store import dataset_storage_dir

__all__ = [
    "CellSetStore",
    "cells_from_indices",
    "format_tags",
    "indices_from_cells",
    "parse_tags",
    "system_columns",
    "write_column",
]

FORMAT_VERSION = 1
CELL_COLUMNS = ("fov", "mask", "label")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------
def parse_tags(text: str) -> Tuple[List[str], List[str]]:
    """Split comma-separated ``key:value`` tags; return ``(tags, problems)``.

    Whitespace around the key and value is dropped (``celltype : CD8 T`` becomes
    ``celltype:CD8 T``). A tag without a ``:``, with an empty side, or a second
    value for a key already given is reported as a problem and left out.
    """
    tags: List[str] = []
    problems: List[str] = []
    seen: Dict[str, str] = {}
    for raw in (text or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        key, colon, value = raw.partition(":")
        key, value = key.strip(), value.strip()
        if not colon or not key or not value:
            problems.append(f"'{raw}' is not key:value")
            continue
        if key in seen:
            if seen[key] != value:
                problems.append(f"'{key}' is given two values ({seen[key]}, {value})")
            continue
        seen[key] = value
        tags.append(f"{key}:{value}")
    return tags, problems


def format_tags(tags: Iterable[str]) -> str:
    return ", ".join(tags)


def split_tag(tag: str) -> Tuple[str, str]:
    key, _, value = tag.partition(":")
    return key.strip(), value.strip()


# ---------------------------------------------------------------------------
# Cell identity <-> cell-table rows
# ---------------------------------------------------------------------------
def system_columns(viewer) -> Set[str]:
    """The columns that link the cell table to the images; never written to."""
    return {
        str(getattr(viewer, name, default))
        for name, default in (
            ("fov_key", "fov"),
            ("x_key", "X"),
            ("y_key", "Y"),
            ("label_key", "label"),
            ("mask_key", "mask"),
        )
    }


def _label_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def cells_from_indices(
    viewer,
    indices: Iterable[Hashable],
    image_masks: Optional[Mapping[Tuple[str, int], str]] = None,
) -> pd.DataFrame:
    """The ``fov, mask, label`` of the cell-table rows *indices*.

    *image_masks* maps ``(fov, label)`` to the mask layer a cell was selected on
    in the image; cells not in it get the viewer's ``mask_key`` (the layer the
    cell table describes).
    """
    table = viewer.cell_table
    rows = table.index.intersection(list(indices))
    if not len(rows):
        return pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in zip(CELL_COLUMNS, ("object", "object", "Int64"))})
    part = table.loc[rows, [viewer.fov_key, viewer.label_key]]
    cells = pd.DataFrame(
        {
            "fov": part[viewer.fov_key].astype(str).to_numpy(),
            "label": _label_series(part[viewer.label_key]).to_numpy(),
        }
    )
    default = str(getattr(viewer, "mask_key", "") or "")
    if image_masks:
        cells["mask"] = [
            image_masks.get((fov, int(label)), default) if pd.notna(label) else default
            for fov, label in zip(cells["fov"], cells["label"])
        ]
    else:
        cells["mask"] = default
    cells = cells.dropna(subset=["label"]).drop_duplicates(subset=["fov", "mask", "label"])
    return cells[list(CELL_COLUMNS)].reset_index(drop=True)


def indices_from_cells(viewer, cells: pd.DataFrame) -> Tuple[Set[Hashable], int]:
    """Resolve *cells* to cell-table row indices; return ``(indices, missing)``.

    *missing* counts the set's cells no row matches any more (a re-segmented
    FOV, a filtered table). Matching is on ``(fov, label)``, the pair the cell
    table keys on.
    """
    if cells is None or not len(cells):
        return set(), 0
    table = viewer.cell_table
    keys = pd.DataFrame(
        {
            "fov": table[viewer.fov_key].astype(str).to_numpy(),
            "label": _label_series(table[viewer.label_key]).to_numpy(),
            "_row": table.index,
        }
    )
    wanted = cells[["fov", "label"]].drop_duplicates().copy()
    wanted["fov"] = wanted["fov"].astype(str)
    wanted["label"] = _label_series(wanted["label"])
    merged = wanted.merge(keys, on=["fov", "label"], how="left", indicator=True)
    found = merged["_merge"] == "both"
    matched = len(merged.loc[found, ["fov", "label"]].drop_duplicates())
    indices = {int(i) if isinstance(i, np.integer) else i for i in merged.loc[found, "_row"]}
    return indices, int(len(wanted) - matched)


def write_column(viewer, indices: Iterable[Hashable], column: str, value: str) -> int:
    """Write *value* into *column* for the rows *indices*; return how many rows.

    A new column starts as empty strings, as the old Cell Table Editor did. An
    existing column that is not text is turned into one first, because putting
    a string into a numeric column would otherwise fail or silently change its
    type in place. The columns that link the table to the images are refused.
    """
    column = (column or "").strip()
    if not column:
        raise ValueError("No column name.")
    if column in system_columns(viewer):
        raise ValueError(f"'{column}' links the cell table to the images and cannot be written to.")
    table = viewer.cell_table
    rows = table.index.intersection(list(indices))
    if not len(rows):
        return 0
    if column not in table.columns:
        table[column] = ""
    elif table[column].dtype != object:
        table[column] = table[column].astype(object)
    table.loc[rows, column] = value
    return int(len(rows))


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
class CellSetStore:
    """Read and write cell sets under ``<storage>/.UELer/dataset_*/cell_sets/``."""

    def __init__(self, dataset_root: "str | Path", storage_root: "str | Path | None" = None) -> None:
        self._dir = dataset_storage_dir(dataset_root, storage_root) / "cell_sets"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"

    @property
    def folder(self) -> Path:
        return self._dir

    # -- index -----------------------------------------------------------
    def _load_index(self) -> Dict[str, Any]:
        if not self._index_path.exists():
            return {"format": FORMAT_VERSION, "sets": []}
        try:
            with self._index_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return {"format": FORMAT_VERSION, "sets": []}
        if not isinstance(data, dict) or not isinstance(data.get("sets"), list):
            return {"format": FORMAT_VERSION, "sets": []}
        return data

    def _save_index(self, index: Dict[str, Any]) -> None:
        index["format"] = FORMAT_VERSION
        partial = self._index_path.with_suffix(".json.partial")
        with partial.open("w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=1, ensure_ascii=False, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(partial, self._index_path)

    def list_sets(self) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self._load_index()["sets"]]

    def get(self, set_id: str) -> Optional[Dict[str, Any]]:
        for entry in self._load_index()["sets"]:
            if entry.get("id") == set_id:
                return dict(entry)
        return None

    def _check_name(self, index: Dict[str, Any], name: str, own_id: str = "") -> str:
        name = (name or "").strip()
        if not name:
            raise ValueError("A cell set needs a name.")
        for entry in index["sets"]:
            if entry.get("id") != own_id and str(entry.get("name", "")).casefold() == name.casefold():
                raise ValueError(f"A cell set named '{name}' already exists.")
        return name

    # -- writing ---------------------------------------------------------
    def save_set(
        self,
        name: str,
        cells: pd.DataFrame,
        *,
        tags: Sequence[str] = (),
        note: str = "",
        recipe: Optional[Mapping[str, Any]] = None,
        source: str = "",
        parents: Sequence[str] = (),
        trail: Sequence[Mapping[str, Any]] = (),
    ) -> Dict[str, Any]:
        """Store *cells* as a new set and return its metadata."""
        index = self._load_index()
        name = self._check_name(index, name)
        if cells is None or not len(cells):
            raise ValueError("A cell set needs at least one cell.")
        set_id = uuid.uuid4().hex[:12]
        cells = cells[list(CELL_COLUMNS)].reset_index(drop=True).copy()
        cells["fov"] = cells["fov"].astype(str)
        cells["mask"] = cells["mask"].astype(str)
        cells["label"] = _label_series(cells["label"]).astype("int64")
        filename = f"{set_id}.parquet"
        partial = self._dir / f"{filename}.partial"
        cells.to_parquet(partial, index=False)
        os.replace(partial, self._dir / filename)
        now = _now()
        entry = {
            "id": set_id,
            "name": name,
            "tags": list(tags),
            "note": note or "",
            "created": now,
            "updated": now,
            "count": int(len(cells)),
            "fovs": int(cells["fov"].nunique()),
            "masks": sorted(cells["mask"].unique().tolist()),
            "source": source,
            "recipe": dict(recipe or {}),
            "parents": list(parents),
            "trail": [dict(step) for step in trail],
            "file": filename,
        }
        index["sets"].append(entry)
        self._save_index(index)
        return dict(entry)

    def read_cells(self, set_id: str) -> pd.DataFrame:
        entry = self.get(set_id)
        if entry is None:
            raise KeyError(set_id)
        return pd.read_parquet(self._dir / entry["file"])

    def update_set(
        self,
        set_id: str,
        *,
        name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Change a set's name, tags or note; its cells never change."""
        index = self._load_index()
        for entry in index["sets"]:
            if entry.get("id") != set_id:
                continue
            if name is not None:
                entry["name"] = self._check_name(index, name, set_id)
            if tags is not None:
                entry["tags"] = list(tags)
            if note is not None:
                entry["note"] = note
            entry["updated"] = _now()
            self._save_index(index)
            return dict(entry)
        raise KeyError(set_id)

    def delete_set(self, set_id: str) -> None:
        index = self._load_index()
        kept = [entry for entry in index["sets"] if entry.get("id") != set_id]
        if len(kept) == len(index["sets"]):
            raise KeyError(set_id)
        removed = next(entry for entry in index["sets"] if entry.get("id") == set_id)
        index["sets"] = kept
        self._save_index(index)
        try:
            (self._dir / removed["file"]).unlink()
        except OSError:
            pass
