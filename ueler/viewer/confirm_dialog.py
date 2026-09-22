"""A modal confirmation dialog for destructive actions (#139).

The viewer has a handful of buttons that throw away work the user cannot
reconstruct — a marker set, a saved mask colour set, an ROI. Guarding them with
a separate "Confirm" checkbox, as the marker-set delete used to, detaches the
question from the action: the button appears to do nothing, the answer lives in
a different row, and the prompt never names what is about to be destroyed.

``ConfirmDialog`` asks instead. It renders a scrim over the whole browser
viewport with a card on top, and calls back only when the user picks the
confirming button.

Two things about the implementation are worth knowing before changing it:

* **It is not a** ``Widget`` **subclass, on purpose.** ``save_widget_states``
  walks ``vars(viewer.ui_component)`` and persists the ``.value`` of every
  ``Widget`` it finds. The stylesheet below lives in an ``HTML`` widget whose
  ``.value`` is the CSS itself, which has no business being written to
  ``widget_states.json``. Holding the widgets inside a plain object keeps them
  out of that walk.
* **The positioning arrives as a CSS class, not as a layout.**
  ``ipywidgets.Layout`` has no ``position`` trait, so ``position: fixed`` cannot
  be expressed per widget. A single ``<style>`` block is mounted with the dialog
  and the boxes carry classes.
"""

from __future__ import annotations

import logging
from html import escape
from typing import Callable, Optional

from ipywidgets import HTML, Box, Button, HBox, Layout, VBox

logger = logging.getLogger(__name__)

__all__ = ["ConfirmDialog"]


#: Above the notebook's own chrome, below nothing in particular — the viewer is
#: the only thing on the page that positions anything.
_Z_INDEX = 10000

_STYLE = """
<style>
.ueler-confirm-scrim {
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
.ueler-confirm-card {
    background: var(--jp-layout-color1, #ffffff);
    color: var(--jp-ui-font-color1, #1a1a1a);
    border: 1px solid var(--jp-border-color2, #cccccc);
    border-radius: 6px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
    padding: 16px 18px;
    min-width: 280px;
    max-width: 440px;
    box-sizing: border-box;
}
.ueler-confirm-title {
    font-size: 1.05em;
    font-weight: 600;
    margin-bottom: 6px;
}
.ueler-confirm-message {
    line-height: 1.4;
}
</style>
""".replace("__Z_INDEX__", str(_Z_INDEX))


def _add_class(widget, name: str) -> None:
    """Attach a CSS class, tolerating widget shims that do not support it.

    ``ui_components`` installs a minimal stand-in for ``ipywidgets.Widget`` when
    the real front end is unavailable, and that stand-in has no ``add_class``.
    A dialog without its stylesheet is still functional in Python, which is what
    those environments exercise.
    """
    adder = getattr(widget, "add_class", None)
    if callable(adder):
        adder(name)


class ConfirmDialog:
    """A yes/no modal. Hidden until :meth:`ask` is called.

    Mount :attr:`view` once, anywhere in the widget tree — the card is
    positioned against the browser viewport, not against its parent, so where it
    is mounted only decides when it exists, not where it appears.
    """

    def __init__(self) -> None:
        self._on_confirm: Optional[Callable[[], None]] = None

        # Public so callers and tests can read back what is on the card.
        self.title_label = HTML(value="")
        _add_class(self.title_label, "ueler-confirm-title")
        self.message_label = HTML(value="")
        _add_class(self.message_label, "ueler-confirm-message")

        self.cancel_button = Button(description="Cancel", button_style="")
        self.cancel_button.on_click(self._on_cancel_clicked)
        self.confirm_button = Button(description="Confirm", button_style="danger")
        self.confirm_button.on_click(self._on_confirm_clicked)

        buttons = HBox(
            children=(self.cancel_button, self.confirm_button),
            layout=Layout(justify_content="flex-end", gap="8px", margin="14px 0 0 0"),
        )
        card = VBox(
            children=(self.title_label, self.message_label, buttons),
            layout=Layout(width="auto"),
        )
        _add_class(card, "ueler-confirm-card")

        self._scrim = Box(children=(card,), layout=Layout(width="auto"))
        _add_class(self._scrim, "ueler-confirm-scrim")

        # ``display: none`` takes the whole subtree out of layout, so a closed
        # dialog costs neither height nor the parent container's flex gap. The
        # stylesheet keeps applying regardless: a ``<style>`` element is live
        # wherever it sits in the document, hidden ancestors included.
        self.view = VBox(
            children=(HTML(value=_STYLE), self._scrim),
            layout=Layout(display="none", width="auto"),
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._on_confirm is not None

    # ------------------------------------------------------------------
    # Opening and answering
    # ------------------------------------------------------------------

    def ask(
        self,
        message: str,
        on_confirm: Callable[[], None],
        *,
        title: str = "Are you sure?",
        confirm_label: str = "Confirm",
        cancel_label: str = "Cancel",
        danger: bool = True,
    ) -> bool:
        """Open the dialog. ``True`` if it opened, ``False`` if one was already up.

        ``message`` and ``title`` are inserted as HTML so a caller can emphasise
        the name of what is being destroyed; callers are responsible for
        escaping anything user-supplied (see :meth:`escape_name`).

        Refusing to open over an existing dialog is what keeps a second click on
        the same button from stacking two confirmations for the same action.
        """
        if self.is_open:
            return False

        self.title_label.value = title
        self.message_label.value = message
        self.confirm_button.description = confirm_label
        self.confirm_button.button_style = "danger" if danger else "primary"
        self.cancel_button.description = cancel_label
        self._on_confirm = on_confirm
        self.view.layout.display = ""
        return True

    def confirm(self) -> None:
        """Run the pending callback and close. Safe to call when closed."""
        callback = self._on_confirm
        # Close first: the callback may itself want to open a dialog, and a
        # callback that raises must not leave the scrim covering the UI.
        self.close()
        if callback is None:
            return
        try:
            callback()
        except Exception:  # pragma: no cover - defensive; surfaced in the log
            logger.exception("Confirmed action failed.")

    def cancel(self) -> None:
        """Discard the pending callback and close."""
        self.close()

    def close(self) -> None:
        self._on_confirm = None
        self.view.layout.display = "none"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def escape_name(name: str) -> str:
        """Escape a user-supplied name for inclusion in a message."""
        return escape(str(name))

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_confirm_clicked(self, _button) -> None:
        self.confirm()

    def _on_cancel_clicked(self, _button) -> None:
        self.cancel()
