"""Cross-plugin protocols for the Cell Annotation workflow."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class SelectionSpec(Protocol):
    """Opaque handle describing a saved cell selection."""

    dataset_id: str

    def cardinality(self) -> int:
        """Return the number of selected cells represented by this spec."""

    def subset_of(self, other: "SelectionSpec") -> bool:
        """Return ``True`` when the current selection is contained in *other*."""

    def union(self, other: "SelectionSpec") -> "SelectionSpec":
        """Return a selection representing the union with *other*."""


@runtime_checkable
class HeatmapStateProvider(Protocol):
    """Protocol implemented by the Heatmap plugin for checkpoint export/import."""

    def export_heatmap_state(
        self,
        *,
        include_embeddings: bool,
        include_raw_medians: bool,
        extra_obs_cols: list[str] | None,
    ) -> dict[str, Any]:
        """Export the current Heatmap view state."""

    def import_heatmap_state(self, adata_path: str) -> None:
        """Restore Heatmap state from a checkpoint artifact."""


@runtime_checkable
class FlowsomParamsProvider(Protocol):
    """Protocol implemented by the FlowSOM plugin for orchestration hooks."""

    def export_flowsom_params(self) -> dict[str, Any]:
        """Return the current FlowSOM configuration."""

    def import_flowsom_params(self, params: Mapping[str, Any]) -> None:
        """Restore FlowSOM configuration from a checkpoint snapshot."""

    def set_selection_context(self, selection: SelectionSpec) -> None:
        """Constrain the FlowSOM plugin to a Cell Annotation selection."""

