"""The guided first-run setup dialog (#142).

Two settings decide whether a dataset shows anything useful, and neither
announces itself.  The **Data mapping** keys name the cell table's coordinate,
label and FOV columns, and a wrong one renders an empty overlay with nothing in
the log.  ``Pixel Size (nm):`` defaults to 390 -- the MIBI detector pitch --
and is applied to every dataset silently, mis-scaling the scale bar in the
viewer and in every exported image.  Both live inside an **Advanced Settings**
accordion that a first-time user has no reason to open.

``SetupDialog`` asks instead, once, when the data arrives.

It is a sibling of :mod:`ueler.viewer.confirm_dialog` rather than an extension
of it.  ``ConfirmDialog.ask`` is strictly yes/no, carries no body widget, and
deliberately refuses to open over an open dialog so that two clicks on a delete
button cannot stack two confirmations; a multi-step form needs all three of
those to be different.  What the two share is the *mount convention*, and that
is what is copied here:

* **Not a** ``Widget`` **subclass.** ``save_widget_states`` walks
  ``vars(viewer.ui_component)`` and persists the ``.value`` of every ``Widget``
  it finds; the stylesheet below is an ``HTML`` widget whose value is the CSS
  itself, which has no business in ``widget_states.json``.
* **Positioning arrives as a CSS class.** ``ipywidgets.Layout`` has no
  ``position`` trait, so ``position: fixed`` cannot be expressed per widget.

The fields on the card are **the accordion's own widget objects**, not copies.
ipywidgets renders a second view of the same model, so the two stay in sync with
no mirroring code: what the user picks here *is* the live setting, and stays
editable in the accordion afterwards.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, List, NamedTuple, Optional, Sequence

from ipywidgets import HTML, Box, Button, HBox, Layout, VBox

# The same shim ``ConfirmDialog`` uses: ``ui_components`` installs a minimal
# stand-in for ``ipywidgets.Widget`` when the real front end is unavailable, and
# that stand-in has no ``add_class``.
from .confirm_dialog import _add_class

logger = logging.getLogger(__name__)

__all__ = [
    "SetupStep",
    "SetupDialog",
    "build_setup_steps",
    "setup_state_path",
    "load_seen_steps",
    "record_seen_steps",
    "STATE_FILENAME",
]


#: One above the confirmation dialog: setup can ask a question of its own.
_Z_INDEX = 10001

#: Where the per-dataset record of answered steps lives, inside ``.UELer``.
STATE_FILENAME = "setup_dialog.json"


_STYLE = """
<style>
.ueler-setup-scrim {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: __Z_INDEX__;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.45);
}
.ueler-setup-card {
    background: var(--jp-layout-color1, #ffffff);
    color: var(--jp-ui-font-color1, #1a1a1a);
    border: 1px solid var(--jp-border-color2, #cccccc);
    border-radius: 6px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
    padding: 16px 18px;
    min-width: 360px;
    max-width: 520px;
    box-sizing: border-box;
}
.ueler-setup-title {
    font-size: 1.05em;
    font-weight: 600;
    margin-bottom: 6px;
}
.ueler-setup-intro {
    line-height: 1.4;
    margin-bottom: 4px;
}
.ueler-setup-progress {
    opacity: 0.7;
    font-size: 0.9em;
}
</style>
""".replace("__Z_INDEX__", str(_Z_INDEX))


class SetupStep(NamedTuple):
    """One card of the wizard.

    *widgets* are the live widgets from ``viewer.ui_component``; the dialog
    never copies a value in or out.
    """

    key: str
    title: str
    intro: str
    widgets: tuple


# ----------------------------------------------------------------------
# Which steps apply
# ----------------------------------------------------------------------


def _widgets_for(ui_component, names: Sequence[str]) -> tuple:
    """The named widgets that exist, in order; missing ones are skipped."""
    found = []
    for name in names:
        widget = getattr(ui_component, name, None)
        if widget is not None:
            found.append(widget)
    return tuple(found)


def build_setup_steps(viewer, *, seen: Sequence[str] = ()) -> List[SetupStep]:
    """The steps this dataset needs, minus the ones already answered.

    Driven by what was actually loaded, so an images-only dataset sees one card
    and nothing about cell tables.  Steps already in *seen* are dropped, which
    is what lets ``load_cell_table`` later in the same notebook ask only about
    the cell table instead of repeating the whole wizard.
    """
    ui_component = getattr(viewer, "ui_component", None)
    if ui_component is None:
        return []
    answered = set(seen or ())
    steps: List[SetupStep] = []

    if "images" not in answered:
        widgets = _widgets_for(
            ui_component,
            ("cache_size_input", "pixel_size_inttext", "enable_downsample_checkbox"),
        )
        if widgets:
            steps.append(
                SetupStep(
                    "images",
                    "Main viewer settings",
                    "<b>Pixel Size (nm)</b> drives the scale bar here and in every exported "
                    "image. It defaults to 390 nm (the MIBI detector pitch), which is wrong "
                    "for most other instruments &mdash; set it now rather than discovering it "
                    "in a figure. <b>Cache Size</b> is how many FOVs stay in memory, and "
                    "<b>Downsample</b> caps the drawn view for speed.",
                    widgets,
                )
            )

    if getattr(viewer, "masks_available", False) and "masks" not in answered:
        widgets = _widgets_for(ui_component, ("mask_key",))
        if widgets:
            steps.append(
                SetupStep(
                    "masks",
                    "Mask mapping",
                    "<b>Mask key</b> is the mask layer the cell table and the plugins are "
                    "linked to. The list is the mask suffixes found in your masks folder.",
                    widgets,
                )
            )

    if getattr(viewer, "cell_table", None) is not None and "cell_table" not in answered:
        widgets = _widgets_for(
            ui_component, ("x_key", "y_key", "label_key", "fov_key")
        )
        if widgets:
            steps.append(
                SetupStep(
                    "cell_table",
                    "Cell table mapping",
                    "Which columns of your cell table hold the cell coordinates, the mask "
                    "label and the FOV name. Every list is built from the columns your table "
                    "actually has; <b>X</b> and <b>Y</b> offer the numeric ones. Getting one "
                    "of these wrong is what makes the viewer find no cells at all.",
                    widgets,
                )
            )

    return steps


# ----------------------------------------------------------------------
# Remembering what was answered
# ----------------------------------------------------------------------


def setup_state_path(viewer) -> Optional[str]:
    """``.UELer/setup_dialog.json`` for this dataset, or ``None``."""
    folder = getattr(viewer, "settings_folder", None)
    if not folder:
        return None
    return os.path.join(str(folder), STATE_FILENAME)


def load_seen_steps(path: Optional[str]) -> set:
    """Step keys already shown for this dataset.  Never raises."""
    if not path or not os.path.exists(path):
        return set()
    try:
        with open(path, "r") as handle:
            state = json.load(handle)
    except Exception:
        logger.debug("[setup] could not read %s.", path, exc_info=True)
        return set()
    return {str(key) for key in (state.get("seen_steps") or ())}


def record_seen_steps(path: Optional[str], keys: Sequence[str]) -> None:
    """Add *keys* to the record.  Never raises -- a read-only settings folder
    costs the user a repeated dialog, not a failed session."""
    if not path or not keys:
        return
    merged = sorted(load_seen_steps(path) | {str(key) for key in keys})
    try:
        with open(path, "w") as handle:
            json.dump({"seen_steps": merged}, handle, indent=2)
    except Exception:
        logger.debug("[setup] could not write %s.", path, exc_info=True)


# ----------------------------------------------------------------------
# The dialog
# ----------------------------------------------------------------------


class SetupDialog:
    """A multi-step modal form.  Hidden until :meth:`open` is called.

    Mount :attr:`view` once, anywhere in the widget tree -- the card is
    positioned against the browser viewport, not against its parent.
    """

    def __init__(self) -> None:
        self._steps: List[SetupStep] = []
        self._index = 0
        self._on_close: Optional[Callable[[Sequence[str]], None]] = None

        self.title_label = HTML(value="")
        _add_class(self.title_label, "ueler-setup-title")
        self.intro_label = HTML(value="")
        _add_class(self.intro_label, "ueler-setup-intro")
        self.progress_label = HTML(value="")
        _add_class(self.progress_label, "ueler-setup-progress")

        self.body = VBox(children=(), layout=Layout(width="auto", gap="4px", margin="10px 0 0 0"))

        self.skip_button = Button(description="Skip setup", button_style="")
        self.skip_button.on_click(self._on_skip_clicked)
        self.back_button = Button(description="Back", button_style="")
        self.back_button.on_click(self._on_back_clicked)
        self.next_button = Button(description="Next", button_style="primary")
        self.next_button.on_click(self._on_next_clicked)

        footer = HBox(
            children=(
                self.progress_label,
                Box(layout=Layout(flex="1 1 auto")),
                self.skip_button,
                self.back_button,
                self.next_button,
            ),
            layout=Layout(align_items="center", gap="8px", margin="14px 0 0 0"),
        )

        card = VBox(
            children=(self.title_label, self.intro_label, self.body, footer),
            layout=Layout(width="auto"),
        )
        _add_class(card, "ueler-setup-card")

        self._scrim = Box(children=(card,), layout=Layout(width="auto"))
        _add_class(self._scrim, "ueler-setup-scrim")

        self.view = VBox(
            children=(HTML(value=_STYLE), self._scrim),
            layout=Layout(display="none", width="auto"),
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return bool(self._steps)

    @property
    def current_step(self) -> Optional[SetupStep]:
        if not self._steps or not 0 <= self._index < len(self._steps):
            return None
        return self._steps[self._index]

    # ------------------------------------------------------------------
    # Opening and navigating
    # ------------------------------------------------------------------

    def open(
        self,
        steps: Sequence[SetupStep],
        *,
        on_close: Optional[Callable[[Sequence[str]], None]] = None,
    ) -> bool:
        """Show *steps* from the first.  ``False`` if there is nothing to ask.

        Re-opening over an open wizard is refused rather than stacked: the two
        entry points (``run_viewer`` and ``load_cell_table``) can both reach
        here in one cell, and the second must not replace a card the user is
        part-way through.
        """
        if self.is_open or not steps:
            return False
        self._steps = list(steps)
        self._index = 0
        self._on_close = on_close
        self._render()
        self.view.layout.display = ""
        return True

    def advance(self) -> None:
        """**Next**, or **Done** on the last step."""
        if not self.is_open:
            return
        if self._index >= len(self._steps) - 1:
            self.close()
            return
        self._index += 1
        self._render()

    def back(self) -> None:
        if not self.is_open or self._index == 0:
            return
        self._index -= 1
        self._render()

    def skip(self) -> None:
        """Dismiss the whole wizard.

        Skipping counts as answered for *every* remaining step, not just the one
        on screen: re-asking a question the user explicitly dismissed, on every
        load, is worse than not asking it.  The fields stay in **Advanced
        Settings**, which is where the intro text points.
        """
        self.close()

    def close(self) -> None:
        """Hide the dialog and report every step it was opened with."""
        keys = tuple(step.key for step in self._steps)
        callback = self._on_close
        # Clear first: the callback writes to disk and may raise, and a dialog
        # that stays open over a failed write blocks the whole UI.
        self._steps = []
        self._index = 0
        self._on_close = None
        self.body.children = ()
        self.view.layout.display = "none"
        if callback is None or not keys:
            return
        try:
            callback(keys)
        except Exception:  # pragma: no cover - defensive; surfaced in the log
            logger.exception("Recording the completed setup steps failed.")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        step = self.current_step
        if step is None:
            return
        self.title_label.value = step.title
        self.intro_label.value = step.intro
        self.body.children = tuple(step.widgets)
        total = len(self._steps)
        self.progress_label.value = (
            f"Step {self._index + 1} of {total}" if total > 1 else ""
        )
        self.back_button.disabled = self._index == 0
        is_last = self._index >= total - 1
        self.next_button.description = "Done" if is_last else "Next"

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_next_clicked(self, _button) -> None:
        self.advance()

    def _on_back_clicked(self, _button) -> None:
        self.back()

    def _on_skip_clicked(self, _button) -> None:
        self.skip()
