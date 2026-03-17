"""Selection specification helpers for Cell Annotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Tuple

CellRef = Tuple[str, int]


@dataclass(frozen=True)
class MaterializedSelectionSpec:
    """Concrete selection handle backed by explicit ``(fov_id, cell_id)`` pairs."""

    dataset_id: str
    cells: FrozenSet[CellRef]

    @classmethod
    def from_cells(cls, dataset_id: str, cells: Iterable[CellRef]) -> "MaterializedSelectionSpec":
        return cls(dataset_id=dataset_id, cells=frozenset(cells))

    def cardinality(self) -> int:
        return len(self.cells)

    def subset_of(self, other: "MaterializedSelectionSpec") -> bool:
        self._validate_same_dataset(other)
        return self.cells.issubset(other.cells)

    def union(self, other: "MaterializedSelectionSpec") -> "MaterializedSelectionSpec":
        self._validate_same_dataset(other)
        return MaterializedSelectionSpec(self.dataset_id, self.cells | other.cells)

    def to_payload(self) -> dict[str, object]:
        """Serialize this materialized selection for manifest/checkpoint metadata."""
        ordered = sorted(self.cells)
        return {
            "type": "materialized",
            "dataset_id": self.dataset_id,
            "n_cells": len(ordered),
            "cells": [[fov_id, cell_id] for fov_id, cell_id in ordered],
        }

    def _validate_same_dataset(self, other: "MaterializedSelectionSpec") -> None:
        if self.dataset_id != other.dataset_id:
            raise ValueError("Selection specifications must belong to the same dataset")
