"""The in-line selection bar above the image.

One row, shared with the **Lasso Select** toggle: a line of text and four
buttons. The text says what is selected (count, source, how many in the FOV
shown) or carries a message (lasso armed, why a loop found nothing). The buttons
send the current selection to the image, the Gallery or the plots, and a second
click on a linked (✓) button takes it back out of that view; ✕ clears it.

It is a plain object, not a ``Widget`` subclass: ``save_widget_states`` walks
``vars(ui_component)`` and persists every widget's ``.value``, and the bar's
text is state of this session, not a setting to restore.
"""

from __future__ import annotations

import html
import re
from typing import Callable, Iterable, Optional

from ipywidgets import HTML, Button, HBox, Layout

#: Fixed row height, so the row never reflows when the bar fills or empties.
ROW_HEIGHT = "32px"

_MUTED = "var(--jp-ui-font-color2, #616161)"
_WARN = "var(--jp-warn-color0, #b45309)"


def _button(description: str, tooltip: str) -> Button:
    return Button(
        description=description,
        tooltip=tooltip,
        layout=Layout(width="auto", height="26px", padding="0 8px"),
    )


class SelectionBar:
    """Text plus Image / Gallery / Plots / clear buttons; see the module doc."""

    LABELS = {"image": "Image", "gallery": "Gallery", "plots": "Plots"}
    _PLACES = {"image": "image", "gallery": "Gallery", "plots": "plots"}
    _SEND_TIPS = {
        "image": "Outline the selected cells in the image",
        "gallery": "Show the selected cells in the Gallery",
        "plots": "Highlight the selected cells in the scatter plot and histogram",
    }

    def __init__(self, toggle) -> None:
        self.text = HTML(
            value="",
            layout=Layout(flex="1 1 auto", min_width="0", overflow="hidden", margin="0 0 0 8px"),
        )
        self.buttons = {target: _button(label, self._SEND_TIPS[target]) for target, label in self.LABELS.items()}
        self.clear_button = _button("✕", "Unselect everything (asks first)")
        self.actions = HBox(
            [*self.buttons.values(), self.clear_button],
            layout=Layout(display="none", gap="4px", flex="0 0 auto"),
        )
        self.row = HBox(
            [toggle, self.text, self.actions],
            layout=Layout(
                align_items="center",
                height=ROW_HEIGHT,
                min_height=ROW_HEIGHT,
                padding="0 0 4px 0",
                width="100%",
            ),
        )
        self._callbacks = {}
        self._message = ""
        self._message_is_warning = False
        self._summary = ""

    # -- wiring ----------------------------------------------------------
    def on_action(self, target: str, callback: Callable[[], None]) -> None:
        """Run *callback* when the button for *target* (or ``"clear"``) is clicked."""
        self._callbacks[target] = callback
        button = self.clear_button if target == "clear" else self.buttons[target]
        button.on_click(lambda _button, t=target: self.trigger(t))

    def trigger(self, target: str) -> None:
        """What a click on the *target* button does."""
        callback = self._callbacks.get(target)
        if callback is not None:
            callback()

    # -- content ---------------------------------------------------------
    @property
    def message(self) -> str:
        return self._message

    def set_message(self, text: str, *, warning: bool = False) -> None:
        """Show *text* (HTML) instead of the selection summary; ``""`` removes it."""
        self._message = text or ""
        self._message_is_warning = bool(warning)
        self._render()

    def show_selection(
        self,
        summary: str,
        delivered: Iterable[str] = (),
        has_selection: Optional[bool] = None,
        source_target: str = "",
    ) -> None:
        """Show the selection *summary* and the state of each target button.

        A target in *delivered* is linked: its button shows ✓ and a click takes
        the selection back out of that view. *source_target* is the view the
        selection was made in; its button shows ✓ too but is disabled, since
        unlinking a selection from its own source would delete it (✕ does that).
        """
        self._summary = summary or ""
        delivered = set(delivered)
        for target, button in self.buttons.items():
            label = self.LABELS[target]
            is_source = target == source_target
            linked = is_source or target in delivered
            button.description = f"{label} ✓" if linked else label
            button.button_style = "success" if linked else ""
            button.disabled = is_source
            if is_source:
                button.tooltip = f"Selected in the {self._PLACES[target]}"
            elif linked:
                button.tooltip = f"Remove the selection from the {self._PLACES[target]}"
            else:
                button.tooltip = self._SEND_TIPS[target]
        active = bool(self._summary) if has_selection is None else bool(has_selection)
        self.actions.layout.display = "flex" if active else "none"
        self._render()

    def _render(self) -> None:
        if self._message:
            color, body = (_WARN if self._message_is_warning else _MUTED), self._message
        elif self._summary:
            color, body = _MUTED, self._summary
        else:
            self.text.value = ""
            return
        # One line, cut with an ellipsis if the column is narrow; the full text
        # is the hover title.
        title = html.escape(html.unescape(re.sub(r"<[^>]+>", "", body)), quote=True)
        self.text.value = (
            f'<div title="{title}" style="color: {color}; font-size: 12px; line-height: 1.3; '
            f'white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{body}</div>'
        )
