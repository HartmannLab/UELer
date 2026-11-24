"""The viewer's current cell selection, whichever view made it.

A selection can start in the image (a click or the lasso), in the scatter plot,
in a histogram gate or in a heatmap cluster, and each of those forwards it on
its own: a plot to the Gallery or the image when its link boxes are ticked, the
image to the plots that follow it. Nothing recorded which selection was the
latest, so no single place could say what is selected or offer to send it on.

:class:`SelectionHub` is that record. Every source publishes to it after a
selection the user made; the most recent one wins, as the image highlight
already does (#119). It also notes which views have received the selection
(``delivered``), so the selection bar can show what is already done.

The hub does not replace the views' own copies (``selected_masks_label``, each
plot's ``selected_indices``); the link boxes keep forwarding as before. It sits
beside them, which keeps this change small enough to try out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, FrozenSet, Hashable, Iterable, List

#: Shown in the selection bar for each source key.
SOURCE_LABELS = {
    "image click": "image click",
    "image lasso": "image lasso",
    "scatter": "scatter plot",
    "histogram": "histogram gate",
    "heatmap": "heatmap cluster",
}

#: Views a selection can be delivered to.
TARGETS = ("image", "gallery", "plots")


@dataclass(frozen=True)
class CellSelection:
    """Cell-table row indices, the source that selected them, and where they went."""

    indices: FrozenSet[Hashable] = frozenset()
    source: str = ""
    delivered: FrozenSet[str] = field(default_factory=frozenset)

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS.get(self.source, self.source)


class SelectionHub:
    """Holds the latest :class:`CellSelection` and notifies listeners of changes."""

    def __init__(self) -> None:
        self.current = CellSelection()
        self._listeners: List[Callable[[CellSelection], None]] = []

    def subscribe(self, listener: Callable[[CellSelection], None]) -> None:
        self._listeners.append(listener)

    def publish(self, indices: Iterable[Hashable], source: str, delivered: Iterable[str] = ()) -> None:
        """Make *indices* the current selection; an empty set clears it."""
        indices = frozenset(indices)
        if not indices:
            self.current = CellSelection()
        else:
            self.current = CellSelection(indices, source, frozenset(delivered))
        self._notify()

    def clear(self, source: str = "") -> None:
        """Clear the selection, or only if it came from *source* when one is given."""
        if source and self.current.source != source:
            return
        if self.current.indices:
            self.current = CellSelection()
            self._notify()

    def mark_delivered(self, target: str) -> None:
        """Record that *target* now shows the current selection."""
        if not self.current.indices or target in self.current.delivered:
            return
        self.current = CellSelection(
            self.current.indices, self.current.source, self.current.delivered | {target}
        )
        self._notify()

    def _notify(self) -> None:
        for listener in list(self._listeners):
            listener(self.current)
