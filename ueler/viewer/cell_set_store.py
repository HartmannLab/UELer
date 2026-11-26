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

**Sets are editable.** Cells can be added and removed at any time, so a set
also works as a *cart* that collects cells over several sessions; every change
is logged in the set's ``edits``. **Snapshot** makes a frozen copy that refuses
edits. A set can hold children (``container`` names the set a child belongs
to), and a child is always a subset of its parent: adding to a child adds to the
parent, removing from a parent removes from its children. The one child built
so far is a set's **★ favourites** (``role: "favourites"``).

**Tags** are free-form labels for organising the list (``pilot``, ``todo``,
``celltype:CD8 T``). They are not read as column names; :func:`write_column`
is the explicit way to put a value into the cell table.
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

FORMAT_VERSION = 2
FAVOURITES = "favourites"
FAVOURITES_NAME = "★ favourites"
CELL_COLUMNS = ("fov", "mask", "label")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------
def parse_tags(text: str) -> Tuple[List[str], List[str]]:
    """Split comma-separated tags; return ``(tags, problems)``.

    Tags are free-form: whitespace around each is dropped, empty ones and exact
    repeats are left out. *problems* is always empty; it is kept so callers can
    report one if a rule is added later.
    """
    tags: List[str] = []
    for raw in (text or "").split(","):
        tag = " ".join(raw.split())
        if tag and tag not in tags:
            tags.append(tag)
    return tags, []


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
_ENTRY_DEFAULTS: Dict[str, Any] = {
    "tags": [],
    "note": "",
    "container": None,
    "role": None,
    "frozen": False,
    "edits": [],
    "parents": [],
    "trail": [],
    "recipe": {},
    "source": "",
    "sample": None,
}


def _empty_cells() -> pd.DataFrame:
    return pd.DataFrame({"fov": pd.Series(dtype=object), "mask": pd.Series(dtype=object), "label": pd.Series(dtype="int64")})


def _clean_cells(cells: Optional[pd.DataFrame]) -> pd.DataFrame:
    """*cells* with the three identity columns typed as stored, without repeats."""
    if cells is None or not len(cells):
        return _empty_cells()
    cells = cells[list(CELL_COLUMNS)].copy()
    cells["fov"] = cells["fov"].astype(str)
    cells["mask"] = cells["mask"].astype(str)
    cells["label"] = _label_series(cells["label"])
    cells = cells.dropna(subset=["label"])
    cells["label"] = cells["label"].astype("int64")
    return cells.drop_duplicates(subset=list(CELL_COLUMNS)).reset_index(drop=True)


def _in(cells: pd.DataFrame, other: pd.DataFrame) -> pd.Series:
    """For each row of *cells*, whether *other* holds the same cell."""
    if not len(cells) or not len(other):
        return pd.Series(False, index=cells.index)
    keys = pd.MultiIndex.from_frame(other[list(CELL_COLUMNS)])
    return pd.Series(pd.MultiIndex.from_frame(cells[list(CELL_COLUMNS)]).isin(keys), index=cells.index)


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
        """The index, with the fields added after format 1 filled in."""
        if not self._index_path.exists():
            return {"format": FORMAT_VERSION, "sets": []}
        try:
            with self._index_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return {"format": FORMAT_VERSION, "sets": []}
        if not isinstance(data, dict) or not isinstance(data.get("sets"), list):
            return {"format": FORMAT_VERSION, "sets": []}
        for entry in data["sets"]:
            for key, default in _ENTRY_DEFAULTS.items():
                if key not in entry:
                    entry[key] = list(default) if isinstance(default, list) else (dict(default) if isinstance(default, dict) else default)
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
        """Every set, children included (``container`` tells them apart)."""
        return [dict(entry) for entry in self._load_index()["sets"]]

    def get(self, set_id: str) -> Optional[Dict[str, Any]]:
        for entry in self._load_index()["sets"]:
            if entry.get("id") == set_id:
                return dict(entry)
        return None

    def children(self, set_id: str) -> List[Dict[str, Any]]:
        return [dict(entry) for entry in self._load_index()["sets"] if entry.get("container") == set_id]

    def favourites(self, set_id: str) -> Optional[Dict[str, Any]]:
        """The set's ★ favourites child, or ``None`` before the first star."""
        return next((c for c in self.children(set_id) if c.get("role") == FAVOURITES), None)

    @staticmethod
    def _entry(index: Dict[str, Any], set_id: str) -> Dict[str, Any]:
        for entry in index["sets"]:
            if entry.get("id") == set_id:
                return entry
        raise KeyError(set_id)

    def _check_name(self, index: Dict[str, Any], name: str, own_id: str = "", container: Optional[str] = None) -> str:
        """*name*, stripped; refused if empty or used by a set with the same container."""
        name = (name or "").strip()
        if not name:
            raise ValueError("A cell set needs a name.")
        for entry in index["sets"]:
            if (
                entry.get("id") != own_id
                and entry.get("container") == container
                and str(entry.get("name", "")).casefold() == name.casefold()
            ):
                raise ValueError(f"A cell set named '{name}' already exists.")
        return name

    # -- cells on disk ---------------------------------------------------
    def read_cells(self, set_id: str) -> pd.DataFrame:
        entry = self.get(set_id)
        if entry is None:
            raise KeyError(set_id)
        return self._read_file(entry["file"])

    def _read_file(self, filename: str) -> pd.DataFrame:
        path = self._dir / filename
        if not path.exists():
            return _empty_cells()
        return pd.read_parquet(path)

    def _write_cells(self, entry: Dict[str, Any], cells: pd.DataFrame) -> None:
        """Replace the entry's cells on disk and its counts in the index entry."""
        partial = self._dir / f"{entry['file']}.partial"
        cells.to_parquet(partial, index=False)
        os.replace(partial, self._dir / entry["file"])
        entry["count"] = int(len(cells))
        entry["fovs"] = int(cells["fov"].nunique()) if len(cells) else 0
        entry["masks"] = sorted(cells["mask"].unique().tolist()) if len(cells) else []
        entry["updated"] = _now()

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
        container: Optional[str] = None,
        role: Optional[str] = None,
        frozen: bool = False,
        allow_empty: bool = False,
        sample: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store *cells* as a new set and return its metadata.

        *allow_empty* lets a cart start with no cells. A set with a *container*
        is its child; its cells are added to the container as well. *sample*
        records how a set saved from the Gallery was drawn (source set, size,
        draw number, seed).
        """
        index = self._load_index()
        if container is not None:
            self._entry(index, container)  # KeyError for an unknown container
        name = self._check_name(index, name, container=container)
        cells = _clean_cells(cells)
        if not len(cells) and not allow_empty:
            raise ValueError("A cell set needs at least one cell.")
        set_id = uuid.uuid4().hex[:12]
        now = _now()
        entry = {
            "id": set_id,
            "name": name,
            "tags": list(tags),
            "note": note or "",
            "created": now,
            "updated": now,
            "source": source,
            "recipe": dict(recipe or {}),
            "parents": list(parents),
            "trail": [dict(step) for step in trail],
            "container": container,
            "role": role,
            "frozen": bool(frozen),
            "edits": [],
            "sample": dict(sample) if sample else None,
            "file": f"{set_id}.parquet",
        }
        self._write_cells(entry, cells)
        entry["updated"] = now
        index["sets"].append(entry)
        if container is not None and len(cells):
            self._add_upward(index, container, cells, recipe)
        self._save_index(index)
        return dict(entry)

    def update_set(
        self,
        set_id: str,
        *,
        name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Change a set's name, tags or note; :meth:`add_cells` and :meth:`remove_cells` change its cells."""
        index = self._load_index()
        entry = self._entry(index, set_id)
        if name is not None:
            entry["name"] = self._check_name(index, name, set_id, entry.get("container"))
        if tags is not None:
            entry["tags"] = list(tags)
        if note is not None:
            entry["note"] = note
        entry["updated"] = _now()
        self._save_index(index)
        return dict(entry)

    # -- editing the cells -----------------------------------------------
    @staticmethod
    def _log_edit(entry: Dict[str, Any], op: str, count: int, recipe: Optional[Mapping[str, Any]]) -> None:
        entry.setdefault("edits", []).append({"time": _now(), "op": op, "count": int(count), "recipe": dict(recipe or {})})

    @staticmethod
    def _refuse_frozen(entry: Dict[str, Any]) -> None:
        if entry.get("frozen"):
            raise ValueError(f"'{entry.get('name')}' is a snapshot; its cells cannot change.")

    def _add_upward(self, index, set_id, cells, recipe) -> int:
        """Add *cells* to the set and to every set it is part of; return how many were new to it."""
        entry = self._entry(index, set_id)
        current = self._read_file(entry["file"])
        new = cells[~_in(cells, current)]
        if len(new):
            self._refuse_frozen(entry)  # only when something would change
            self._write_cells(entry, pd.concat([current, new], ignore_index=True))
            self._log_edit(entry, "add", len(new), recipe)
        container = entry.get("container")
        if container is not None:
            self._add_upward(index, container, cells, recipe)
        return int(len(new))

    def _remove_downward(self, index, set_id, cells, recipe) -> int:
        """Remove *cells* from the set and every set inside it; return how many it lost."""
        entry = self._entry(index, set_id)
        current = self._read_file(entry["file"])
        gone = _in(current, cells)
        if gone.any():
            self._refuse_frozen(entry)
            self._write_cells(entry, current[~gone].reset_index(drop=True))
            self._log_edit(entry, "remove", int(gone.sum()), recipe)
        for child in [e for e in index["sets"] if e.get("container") == set_id]:
            if not child.get("frozen"):
                self._remove_downward(index, child["id"], cells, recipe)
        return int(gone.sum())

    def add_cells(self, set_id: str, cells: pd.DataFrame, recipe: Optional[Mapping[str, Any]] = None) -> int:
        """Add *cells* to the set (and to its container); return how many were new.

        Refused (``ValueError``) on a snapshot. *recipe* says how the added cells
        were selected and goes into the set's ``edits``.
        """
        index = self._load_index()
        self._refuse_frozen(self._entry(index, set_id))
        added = self._add_upward(index, set_id, _clean_cells(cells), recipe)
        self._save_index(index)
        return added

    def remove_cells(self, set_id: str, cells: pd.DataFrame, recipe: Optional[Mapping[str, Any]] = None) -> int:
        """Remove *cells* from the set and from its children; return how many it lost."""
        index = self._load_index()
        self._refuse_frozen(self._entry(index, set_id))
        removed = self._remove_downward(index, set_id, _clean_cells(cells), recipe)
        self._save_index(index)
        return removed

    def contains(self, set_id: str, cells: pd.DataFrame) -> pd.Series:
        """For each row of *cells*, whether the set holds it."""
        cells = _clean_cells(cells)
        return _in(cells, self.read_cells(set_id))

    # -- stars -----------------------------------------------------------
    def star(self, set_id: str, cells: pd.DataFrame, recipe: Optional[Mapping[str, Any]] = None) -> int:
        """Star *cells* in the set; a cell not in the set yet is added to it first.

        The stars are the set's ★ favourites child, made on the first star.
        Returns how many cells were newly starred.
        """
        parent = self.get(set_id)
        if parent is None:
            raise KeyError(set_id)
        self._refuse_frozen(parent)
        child = self.favourites(set_id)
        if child is None:
            child = self.save_set(FAVOURITES_NAME, _empty_cells(), container=set_id, role=FAVOURITES, allow_empty=True,
                                  source="star", recipe={"favourites_of": set_id})
        return self.add_cells(child["id"], cells, recipe)

    def unstar(self, set_id: str, cells: pd.DataFrame, recipe: Optional[Mapping[str, Any]] = None) -> int:
        """Take the star off *cells*; they stay in the set. Returns how many lost it."""
        child = self.favourites(set_id)
        if child is None:
            return 0
        return self.remove_cells(child["id"], cells, recipe)

    # -- snapshots -------------------------------------------------------
    def snapshot(self, set_id: str, name: str) -> Dict[str, Any]:
        """A frozen copy of the set, with its stars; the copy names the set in ``parents``."""
        source = self.get(set_id)
        if source is None:
            raise KeyError(set_id)
        copy = self.save_set(
            name,
            self.read_cells(set_id),
            tags=source.get("tags", []),
            note=source.get("note", ""),
            recipe={"snapshot_of": set_id, "name": source.get("name", "")},
            source="snapshot",
            parents=[set_id],
            trail=source.get("trail", []),
            frozen=True,
            allow_empty=True,
        )
        stars = self.favourites(set_id)
        if stars is not None and stars.get("count"):
            self.save_set(FAVOURITES_NAME, self.read_cells(stars["id"]), container=copy["id"], role=FAVOURITES,
                          frozen=True, source="snapshot", recipe={"favourites_of": copy["id"]})
        return self.get(copy["id"]) or copy

    # -- deleting --------------------------------------------------------
    def delete_set(self, set_id: str) -> List[str]:
        """Delete the set and every set inside it; return the deleted ids."""
        index = self._load_index()
        self._entry(index, set_id)
        doomed = [set_id]
        for current in doomed:
            doomed.extend(e["id"] for e in index["sets"] if e.get("container") == current)
        removed = [e for e in index["sets"] if e.get("id") in doomed]
        index["sets"] = [e for e in index["sets"] if e.get("id") not in doomed]
        self._save_index(index)
        for entry in removed:
            try:
                (self._dir / entry["file"]).unlink()
            except OSError:
                pass
        return doomed
