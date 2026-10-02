"""Populating the Data-mapping dropdowns from the loaded data (#142).

The five key widgets -- ``X key:``, ``Y key:``, ``Label key:``, ``Mask key:``
and ``Fov key:`` -- used to be free text pre-filled with one lab's column names.
Typing a key from memory is the main way a dataset fails to show anything, and a
typo is indistinguishable from a wrong dataset: both render an empty overlay
with nothing in the log.  They are ``Dropdown`` widgets now, offering the
columns the cell table actually has and the mask suffixes actually on disk.

Everything in this module exists to make *replacing a dropdown's options* safe.
``ipywidgets.Dropdown`` rejects a value that is not among its options, so the
naive assignment is a ``TraitError`` (or, worse, a silent retarget of a key the
user chose) in three ordinary situations: restoring ``widget_states.json``,
loading a second cell table over a first, and an images-only session that has no
columns to offer at all.  :func:`apply_options` is the single rule that covers
all three, and :func:`ensure_option` is the same invariant for the restore path,
which assigns a value without knowing where it came from.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "KeyField",
    "KEY_FIELDS",
    "CELL_TABLE_FIELDS",
    "MASK_FIELD",
    "column_options",
    "apply_options",
    "ensure_option",
]


class KeyField(NamedTuple):
    """One Data-mapping dropdown, and how to fill it.

    *preferred* is consulted only when the widget's current value names
    something the data does not have -- see :func:`apply_options`.  The order is
    the order of preference, and the first entry is the widget's own shipped
    default, so a dataset that follows the original convention is unaffected.
    """

    #: Attribute on ``viewer.ui_component``, and on the viewer itself.
    attribute: str
    #: The widget's ``description``; ``on_key_change`` dispatches on this.
    description: str
    #: Column-name aliases to fall back to, best first.
    preferred: tuple
    #: Restrict the options to numeric columns where any are numeric.
    numeric_only: bool = False


#: The four keys that name columns of the cell table.
CELL_TABLE_FIELDS = (
    KeyField(
        "x_key",
        "X key:",
        ("centroid-1", "x", "X", "centroid_x", "x_centroid", "cell_x", "centroid-x"),
        numeric_only=True,
    ),
    KeyField(
        "y_key",
        "Y key:",
        ("centroid-0", "y", "Y", "centroid_y", "y_centroid", "cell_y", "centroid-y"),
        numeric_only=True,
    ),
    KeyField(
        "label_key",
        "Label key:",
        ("label", "cell_label", "cellLabel", "cell_id", "CellID", "mask_label"),
    ),
    KeyField(
        "fov_key",
        "Fov key:",
        ("fov", "fov_name", "FOV", "sample", "sample_id", "image", "point"),
    ),
)

#: The one key that names a mask layer rather than a column.
MASK_FIELD = KeyField("mask_key", "Mask key:", ("whole_cell", "cell", "nuclear", "nucleus"))

KEY_FIELDS = CELL_TABLE_FIELDS + (MASK_FIELD,)


def _is_numeric(dtype: Any) -> bool:
    """``True`` for a dtype a coordinate could plausibly be stored in.

    Imported lazily and guarded: this module is imported by ``ui_components``,
    which several tests exercise against a stripped widget stack, and a dropdown
    that offers every column is a far better failure than an import error.
    """
    try:
        from pandas.api.types import is_numeric_dtype
    except Exception:  # pragma: no cover - pandas is a hard dependency
        return True
    try:
        return bool(is_numeric_dtype(dtype))
    except Exception:
        return False


def column_options(schema: Mapping[str, Any], *, numeric_only: bool = False) -> list:
    """Column names from *schema*, in schema order.

    *numeric_only* drops the non-numeric columns, but only when that leaves
    something to offer: a schema whose dtypes are all ``object`` (a CSV read
    with mixed values, say) would otherwise hand ``X key:`` an empty list and
    lose the user the only control that could fix it.
    """
    names = [str(name) for name in (schema or {})]
    if not numeric_only:
        return names
    numeric = [str(name) for name, dtype in (schema or {}).items() if _is_numeric(dtype)]
    return numeric or names


def _option_values(widget) -> list:
    """The plain values of *widget*'s options, tolerating ``(label, value)`` pairs."""
    values = []
    for option in getattr(widget, "options", ()) or ():
        if isinstance(option, (tuple, list)) and len(option) == 2:
            values.append(option[1])
        else:
            values.append(option)
    return values


def ensure_option(widget, value) -> bool:
    """Make *value* assignable to *widget*; ``True`` if the options grew.

    A dropdown restored from ``widget_states.json`` may hold a key that the
    current dataset no longer has a column for.  Dropping it silently would
    retarget the viewer to a different column without saying so, and letting the
    ``TraitError`` through would abort the whole widget-state restore over one
    stale string, so the stale value is re-admitted as an option of its own.
    """
    if widget is None or value is None:
        return False
    options = _option_values(widget)
    if not options or value in options:
        return False
    try:
        widget.options = list(options) + [value]
    except Exception:
        logger.debug("[data mapping] could not extend options of %r.", widget, exc_info=True)
        return False
    return True


def apply_options(widget, options: Sequence, *, preferred: Iterable = ()) -> bool:
    """Replace *widget*'s options without changing a valid answer.

    Returns ``True`` when the widget's value ended up different from the one it
    started with.  The rule, in order:

    * an empty *options* leaves the widget completely alone -- an images-only
      session has no columns to offer and must keep working;
    * a current value the data has stays selected;
    * otherwise the first *preferred* alias the data does have is selected.
      This is the pre-fill: a table using ``x``/``y`` rather than
      ``centroid-1``/``centroid-0`` is mapped correctly on arrival.  It can only
      fire when the current value names something that does not exist, which is
      to say when the viewer is already showing nothing;
    * otherwise the current value is kept, as an option of its own, so an
      unrecognised key is never silently swapped for a plausible-looking one.
    """
    if widget is None:
        return False
    names = []
    for option in options or ():
        text = str(option)
        if text and text not in names:
            names.append(text)
    if not names:
        return False

    current = getattr(widget, "value", None)
    current = str(current) if current not in (None, "") else ""

    if current and current not in names:
        chosen = next((str(alias) for alias in preferred if str(alias) in names), "")
        if not chosen:
            # Keep the unrecognised key visible and selectable rather than
            # replacing it with a guess the user never made.
            names = names + [current]
            chosen = current
    else:
        chosen = current or names[0]

    try:
        widget.options = names
        widget.value = chosen
    except Exception:
        logger.debug("[data mapping] could not apply options to %r.", widget, exc_info=True)
        return False
    return chosen != current
