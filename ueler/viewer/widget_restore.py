"""Restoring saved widget values without letting one bad entry stop the viewer.

``widget_states.json`` is written from a live session and read back into a
different one, and the two need not agree.  A dataset can lose the column a key
pointed at, a marker set can be deleted, a slider's bounds can move, the file
can be hand-edited, and a session killed mid-write leaves it truncated.  Any of
those used to surface as a ``TraitError`` out of ``load_widget_states`` -- which
runs inside ``ImageMaskViewer.__init__``, so the result was not a wrong setting
but **no viewer at all**, with a traceback naming a widget rather than the file
that caused it.

That trade is always wrong.  A saved setting is a convenience; the viewer
opening is not.  So a value the widget refuses is replaced by the nearest thing
it will accept -- the same value in the type the widget holds, clamped into its
range, the first option it offers, or the default it would have had with no
saved state -- and the substitution is logged as a warning naming the widget,
the rejected value and what was used instead.  Silence would be worse than the
crash it replaces: a setting that quietly reverted is a wrong figure waiting to
happen.

Not every widget is strict.  ``IntText`` coerces ``"7"`` to ``7`` and
``IntSlider`` clamps out-of-range values on its own, and the data-mapping keys
are ``Combobox`` (#142), which accepts a column name this dataset does not have
-- deliberately, because a key that is visibly wrong can be fixed and one that
has been silently retargeted cannot.  The ladder here is for the rest:
``Dropdown``, ``Checkbox``, ``ColorPicker`` and anything else whose trait
validates.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

# Private, same package, and deliberately not duplicated: the rule for reading
# a widget's options -- including ``(label, value)`` pairs -- has to stay in step
# with the one the data-mapping dropdowns are built on.
from .data_mapping import _option_values

logger = logging.getLogger(__name__)

__all__ = ["restore_widget_value", "restore_widget_values"]


def _trait_default(widget) -> Any:
    """What ``widget.value`` would be with no saved state, or ``None``."""
    try:
        return widget.trait_defaults("value")["value"]
    except Exception:
        return None


def _candidates(widget, value) -> Iterator[Any]:
    """Progressively weaker substitutes for a value the widget refused.

    Ordered by how much of the user's intent each one keeps: a cast preserves
    it exactly, a clamp preserves the direction, an option preserves only that
    the widget stays usable, and the default preserves nothing but is always
    valid.
    """
    current = getattr(widget, "value", None)

    # The saved value in the type the widget currently holds -- a JSON round
    # trip is the usual reason a bool arrives as "true" or an int as "7".
    if isinstance(current, bool):
        # ``bool("false")`` is ``True``, so the generic cast is not merely
        # useless here but actively wrong.  Only the spellings that have an
        # unambiguous reading are accepted; anything else falls through to the
        # default, which is the honest answer for a checkbox.
        text = str(value).strip().lower()
        if text in ("true", "1", "yes", "on"):
            yield True
        elif text in ("false", "0", "no", "off"):
            yield False
    elif current is not None and not isinstance(value, type(current)):
        try:
            yield type(current)(value)
        except Exception:
            pass

    # Inside the widget's own bounds.  Most range widgets clamp for themselves;
    # this covers the ones that validate instead.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        low = getattr(widget, "min", None)
        high = getattr(widget, "max", None)
        if isinstance(low, (int, float)) and value < low:
            yield low
        if isinstance(high, (int, float)) and value > high:
            yield high

    # The first thing a constrained widget offers -- "the next possible value".
    options = _option_values(widget)
    if options:
        yield options[0]

    default = _trait_default(widget)
    if default is not None:
        yield default


def restore_widget_value(widget, value, *, label: str, log=None) -> bool:
    """Assign *value* to *widget*, substituting if it is refused.

    Returns ``True`` when the saved value itself was applied.  Never raises:
    every exit, including the one where nothing is accepted, leaves the caller
    free to carry on with the next setting.
    """
    log = log or logger
    if widget is None:
        return False
    try:
        widget.value = value
        return True
    except Exception as exc:
        rejected = exc

    for candidate in _candidates(widget, value):
        try:
            widget.value = candidate
        except Exception:
            continue
        log.warning(
            "[settings] %s could not be restored to %r (%s); using %r instead. "
            "If that setting matters for this dataset, set it again in the viewer.",
            label, value, type(rejected).__name__, candidate,
        )
        return False

    log.warning(
        "[settings] %s could not be restored to %r (%s) and no fallback was "
        "accepted; it keeps its current value %r.",
        label, value, type(rejected).__name__, getattr(widget, "value", None),
    )
    return False


def restore_widget_values(mapping, values, *, label: str, log=None) -> int:
    """Restore a ``{key: widget}`` mapping from a ``{key: value}`` dict.

    Returns how many were restored exactly.  A *values* that is not a mapping
    is ignored with a warning rather than raising -- a hand-edited file can put
    anything there.
    """
    log = log or logger
    if not isinstance(values, dict):
        log.warning("[settings] %s expected a mapping of saved values, got %s; ignored.",
                    label, type(values).__name__)
        return 0
    restored = 0
    for key, value in values.items():
        widget = mapping.get(key)
        if widget is None or not hasattr(widget, "value"):
            continue
        if restore_widget_value(widget, value, label=f"{label}[{key!r}]", log=log):
            restored += 1
    return restored
