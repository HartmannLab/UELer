"""The viewer's current cell selection, whichever view made it, and its history.

A selection can start in the image (a click or the lasso), in the scatter plot,
in a histogram gate, in a heatmap cluster or in a saved cell set, and each of
those forwards it on its own: a plot to the Gallery or the image when its link
boxes are ticked, the image to the plots that follow it. Nothing recorded which
selection was the latest, so no single place could say what is selected or offer
to send it on.

:class:`SelectionHub` is that record. Every source publishes to it after a
selection the user made; the most recent one wins, as the image highlight
already does (#119). It also notes which views have received the selection
(``delivered``), so the selection bar can show what is already done and take it
back out of a view (``mark_removed``).

The hub also keeps the session's **history** in two parts with different
limits:

* the *log*: one :class:`HistoryStep` per selection, with its time, source,
  cell count and *recipe* (a JSON-ready dict saying how it was made). The log is
  never truncated within a session, because a saved cell set stores all of it as
  the record of how the set came about;
* the *cells* of each step, which ◀ ▶ need to restore an earlier selection. A
  step can hold hundreds of thousands of cells, so only the last
  :data:`KEEP_CELLS` selection steps keep them; ◀ stops at the first step that
  no longer has its cells.

Moving back or forward and other actions that are not selections (a column
write, a set saved) are logged too, as steps of another *kind*, so the log reads
as what happened, in order.

The hub does not replace the views' own copies (``selected_masks_label``, each
plot's ``selected_indices``); the link boxes keep forwarding as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, FrozenSet, Hashable, Iterable, List, Mapping, Optional

#: Shown in the selection bar for each source key.
SOURCE_LABELS = {
    "image click": "image click",
    "image lasso": "image lasso",
    "scatter": "scatter plot",
    "histogram": "histogram gate",
    "heatmap": "heatmap cluster",
    "set": "cell set",
    "history": "history",
    "clear": "unselected",
}

#: Views a selection can be delivered to.
TARGETS = ("image", "gallery", "plots")

#: The view each source belongs to. That view holds the selection by making it,
#: so it cannot be unlinked from it; the heatmap is not one of the targets.
SOURCE_TARGETS = {
    "image click": "image",
    "image lasso": "image",
    "scatter": "plots",
    "histogram": "plots",
}

#: How many selection steps keep their cells for ◀ ▶.
KEEP_CELLS = 50

#: Step kinds. Only ``selection`` steps are places ◀ ▶ can go to.
SELECTION, NAVIGATION, EVENT = "selection", "navigation", "event"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class CellSelection:
    """Cell-table row indices, the source that selected them, and where they went.

    *recipe* says how the selection was made (see the module doc); it is not
    part of equality, since two selections of the same cells from the same
    source are the same selection.
    """

    indices: FrozenSet[Hashable] = frozenset()
    source: str = ""
    delivered: FrozenSet[str] = field(default_factory=frozenset)
    recipe: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def source_label(self) -> str:
        if self.source == "set" and self.recipe.get("name"):
            return f"set: {self.recipe['name']}"
        if self.source == "history" and self.recipe.get("step"):
            restored = SOURCE_LABELS.get(self.recipe.get("source", ""), self.recipe.get("source", ""))
            return f"step {self.recipe['step']} ({restored})" if restored else f"step {self.recipe['step']}"
        return SOURCE_LABELS.get(self.source, self.source)

    @property
    def source_target(self) -> str:
        """The target view the selection was made in, or ``""``."""
        return SOURCE_TARGETS.get(self.source, "")

    def shows_in(self, target: str) -> bool:
        """Whether *target* shows the selection, by delivery or as its source."""
        return target in self.delivered or target == self.source_target

    def with_delivered(self, delivered: Iterable[str]) -> "CellSelection":
        return CellSelection(self.indices, self.source, frozenset(delivered), self.recipe)


@dataclass
class HistoryStep:
    """One entry of the session log. ``indices`` is ``None`` once the cells are dropped."""

    step: int
    time: str
    kind: str
    source: str
    count: int
    recipe: Dict[str, Any]
    indices: Optional[FrozenSet[Hashable]] = None

    @property
    def label(self) -> str:
        if self.kind == NAVIGATION:
            return f"back to step {self.recipe.get('to_step')}" if self.recipe.get("direction") == "back" else f"forward to step {self.recipe.get('to_step')}"
        if self.kind == EVENT:
            return str(self.recipe.get("action", self.source))
        return CellSelection(frozenset(), self.source, recipe=self.recipe).source_label or "unselected"

    def to_dict(self) -> Dict[str, Any]:
        """The step without its cells, as stored in a cell set's trail."""
        return {
            "step": self.step,
            "time": self.time,
            "kind": self.kind,
            "source": self.source,
            "label": self.label,
            "count": self.count,
            "recipe": dict(self.recipe),
        }


class SelectionHub:
    """Holds the latest :class:`CellSelection`, the session log, and notifies listeners."""

    def __init__(self) -> None:
        self.current = CellSelection()
        self.history: List[HistoryStep] = []
        #: Index in ``history`` of the selection step shown now, or ``None``.
        self._cursor: Optional[int] = None
        self._listeners: List[Callable[[CellSelection], None]] = []

    def subscribe(self, listener: Callable[[CellSelection], None]) -> None:
        self._listeners.append(listener)

    # -- selections ------------------------------------------------------
    def publish(
        self,
        indices: Iterable[Hashable],
        source: str,
        delivered: Iterable[str] = (),
        recipe: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Make *indices* the current selection; an empty set clears it."""
        indices = frozenset(indices)
        recipe = dict(recipe or {})
        if not indices:
            self._set_empty(source, recipe)
            return
        self.current = CellSelection(indices, source, frozenset(delivered), recipe)
        self._record(indices, source, recipe)
        self._notify()

    def clear(self, source: str = "", recipe: Optional[Mapping[str, Any]] = None) -> None:
        """Clear the selection, or only if it came from *source* when one is given."""
        if source and self.current.source != source:
            return
        self._set_empty(source or "clear", dict(recipe or {}))

    def _set_empty(self, source: str, recipe: Dict[str, Any]) -> None:
        if not self.current.indices:
            return
        recipe.setdefault("cleared", source)
        self.current = CellSelection()
        self._record(frozenset(), "clear", recipe)
        self._notify()

    def mark_delivered(self, target: str) -> None:
        """Record that *target* now shows the current selection."""
        if not self.current.indices or target in self.current.delivered:
            return
        self.current = self.current.with_delivered(self.current.delivered | {target})
        self._notify()

    def mark_removed(self, target: str) -> None:
        """Record that *target* no longer shows the current selection."""
        if target not in self.current.delivered:
            return
        self.current = self.current.with_delivered(self.current.delivered - {target})
        self._notify()

    # -- history ---------------------------------------------------------
    def _append(self, kind: str, source: str, count: int, recipe: Dict[str, Any], indices=None) -> HistoryStep:
        step = HistoryStep(len(self.history) + 1, _now(), kind, source, count, recipe, indices)
        self.history.append(step)
        return step

    def _record(self, indices: FrozenSet[Hashable], source: str, recipe: Dict[str, Any]) -> None:
        self._append(SELECTION, source, len(indices), recipe, indices)
        self._cursor = len(self.history) - 1
        self._drop_old_cells()

    def _drop_old_cells(self) -> None:
        kept = 0
        for step in reversed(self.history):
            if step.kind != SELECTION or step.indices is None:
                continue
            kept += 1
            if kept > KEEP_CELLS:
                step.indices = None

    def note(self, action: str, recipe: Optional[Mapping[str, Any]] = None, *, count: int = 0) -> None:
        """Log something done with the selection that is not a selection (a write, a save)."""
        body = {"action": action, **dict(recipe or {})}
        self._append(EVENT, "event", count, body)
        self._notify()

    def _target(self, direction: int) -> Optional[int]:
        """The selection step ◀ (``-1``) or ▶ (``+1``) would go to, if it still has its cells."""
        if self._cursor is None:
            return None
        position = self._cursor + direction
        while 0 <= position < len(self.history):
            step = self.history[position]
            if step.kind == SELECTION:
                return position if step.indices is not None else None
            position += direction
        return None

    def can_go_back(self) -> bool:
        return self._target(-1) is not None

    def can_go_forward(self) -> bool:
        return self._target(+1) is not None

    def go(self, direction: int, delivered: Iterable[str] = ()) -> Optional[HistoryStep]:
        """Restore the selection one step back (``-1``) or forward (``+1``).

        The restored selection comes back with source ``"history"`` (its original
        source is in the recipe), so every bar button can act on it. The move is
        logged as a navigation step, which ◀ ▶ themselves skip.
        """
        position = self._target(direction)
        if position is None:
            return None
        step = self.history[position]
        self._cursor = position
        self._append(
            NAVIGATION,
            "history",
            step.count,
            {"direction": "back" if direction < 0 else "forward", "to_step": step.step},
        )
        if step.indices:
            recipe = {"step": step.step, "source": step.source, "recipe": dict(step.recipe)}
            self.current = CellSelection(step.indices, "history", frozenset(delivered), recipe)
        else:
            self.current = CellSelection()
        self._notify()
        return step

    def trail(self) -> List[Dict[str, Any]]:
        """The whole session log without cells: what a saved set keeps."""
        return [step.to_dict() for step in self.history]

    def _notify(self) -> None:
        for listener in list(self._listeners):
            listener(self.current)
