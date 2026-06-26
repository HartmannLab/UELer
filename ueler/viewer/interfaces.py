"""Protocol interfaces for cross-plugin communication in checkpoint workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any


class HeatmapStateProvider(Protocol):
    """Implemented by HeatmapDisplay to expose save/load hooks."""

    def export_heatmap_state(self, *, include_raw_medians: bool = True) -> "Any":
        """Return an AnnData object capturing the full current heatmap state."""
        ...

    def import_heatmap_state(self, adata: "Any") -> None:
        """Restore heatmap state from a previously exported AnnData object."""
        ...


class FlowsomParamsProvider(Protocol):
    """Implemented by RunFlowsom to expose save/load hooks."""

    def export_flowsom_params(self) -> dict:
        """Return a dict of the current FlowSOM UI parameter values."""
        ...

    def import_flowsom_params(self, params: dict) -> None:
        """Restore FlowSOM UI widgets from a previously exported params dict."""
        ...
