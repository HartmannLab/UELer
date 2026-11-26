"""The in-line selection bar above the image.

One row, shared with the **Lasso Select** toggle: ◀ ▶, a line of text and the
action buttons. ◀ ▶ step through the session's selections. The text says what
is selected (count, source, how many in the FOV shown) or carries a message
(lasso armed, why a loop found nothing). The buttons send the current selection
to the image, the Gallery or the plots, and a second click on a linked (✓)
button takes it back out of that view; **Save as set…** keeps it as a named cell
set, and ✕ clears it.

**Save as set…** swaps the text and buttons for a name field, a tags field and
Save / Cancel in the same row, so the bar keeps its one-line height.

It is a plain object, not a ``Widget`` subclass: ``save_widget_states`` walks
``vars(ui_component)`` and persists every widget's ``.value``, and the bar's
text is state of this session, not a setting to restore.
"""

from __future__ import annotations

import html
import re
from typing import Callable, Iterable, Optional

from ipywidgets import HTML, Button, HBox, Layout, Text

#: Fixed row height, so the row never reflows when the bar fills or empties.
ROW_HEIGHT = "32px"

_MUTED = "var(--jp-ui-font-color2, #616161)"
_WARN = "var(--jp-warn-color0, #b45309)"


def _button(description: str, tooltip: str, **layout) -> Button:
    return Button(
        description=description,
        tooltip=tooltip,
        layout=Layout(width="auto", height="26px", padding="0 8px", **layout),
    )


def _show(widget, visible: bool, display: str = "flex") -> None:
    widget.layout.display = display if visible else "none"


class SelectionBar:
    """◀ ▶, text, Image / Gallery / Plots / Save as set… / ✕; see the module doc."""

    LABELS = {"image": "Image", "gallery": "Gallery", "plots": "Plots"}
    _PLACES = {"image": "image", "gallery": "Gallery", "plots": "plots"}
    _SEND_TIPS = {
        "image": "Outline the selected cells in the image",
        "gallery": "Show the selected cells in the Gallery",
        "plots": "Highlight the selected cells in the scatter plot and histogram",
    }

    def __init__(self, toggle) -> None:
        self.back_button = _button("◀", "Previous selection", min_width="28px")
        self.forward_button = _button("▶", "Next selection", min_width="28px")
        self.back_button.disabled = True
        self.forward_button.disabled = True
        self.text = HTML(
            value="",
            layout=Layout(flex="1 1 auto", min_width="0", overflow="hidden", margin="0 0 0 8px"),
        )
        self.buttons = {target: _button(label, self._SEND_TIPS[target]) for target, label in self.LABELS.items()}
        self.save_button = _button("Save as set…", "Keep the selection as a named cell set", display="none")
        self.clear_button = _button("✕", "Unselect everything (asks first)")
        self.actions = HBox(
            [*self.buttons.values(), self.save_button, self.clear_button],
            layout=Layout(display="none", gap="4px", flex="0 0 auto"),
        )
        self.name_input = Text(
            value="",
            placeholder="Name of the cell set",
            layout=Layout(width="180px", height="26px"),
        )
        self.tags_input = Text(
            value="",
            placeholder="Tags: key:value, … (optional)",
            layout=Layout(flex="1 1 auto", min_width="120px", height="26px"),
        )
        self.save_confirm = _button("Save", "Save the selection as a cell set")
        self.save_confirm.button_style = "primary"
        self.save_cancel = _button("Cancel", "Do not save")
        self.save_status = HTML(value="", layout=Layout(flex="0 1 auto", min_width="0", overflow="hidden"))
        self.save_form = HBox(
            [self.name_input, self.tags_input, self.save_confirm, self.save_cancel, self.save_status],
            layout=Layout(display="none", gap="4px", flex="1 1 auto", align_items="center", margin="0 0 0 8px"),
        )
        self.row = HBox(
            [toggle, self.back_button, self.forward_button, self.text, self.actions, self.save_form],
            layout=Layout(
                align_items="center",
                height=ROW_HEIGHT,
                min_height=ROW_HEIGHT,
                padding="0 0 4px 0",
                width="100%",
            ),
        )
        self._callbacks = {}
        self._save_callback: Optional[Callable[[str, str], str]] = None
        self._message = ""
        self._message_is_warning = False
        self._summary = ""
        self._saving = False
        self._has_selection = False
        self.save_button.on_click(lambda _button: self.begin_save())
        self.save_confirm.on_click(lambda _button: self.submit_save())
        self.save_cancel.on_click(lambda _button: self.end_save())

    # -- wiring ----------------------------------------------------------
    def on_action(self, target: str, callback: Callable[[], None]) -> None:
        """Run *callback* when the button for *target* is clicked.

        *target* is a view (``"image"``, ``"gallery"``, ``"plots"``), ``"clear"``
        (✕), ``"back"`` (◀) or ``"forward"`` (▶).
        """
        self._callbacks[target] = callback
        extra = {"clear": self.clear_button, "back": self.back_button, "forward": self.forward_button}
        button = extra[target] if target in extra else self.buttons[target]
        button.on_click(lambda _button, t=target: self.trigger(t))

    def on_save(self, callback: Callable[[str, str], str]) -> None:
        """Offer **Save as set…**; *callback(name, tags)* returns an error, or ``""`` when saved."""
        self._save_callback = callback
        self._refresh_save_button()

    def trigger(self, target: str) -> None:
        """What a click on the *target* button does."""
        callback = self._callbacks.get(target)
        if callback is not None:
            callback()

    # -- saving ----------------------------------------------------------
    @property
    def saving(self) -> bool:
        return self._saving

    def begin_save(self) -> None:
        """Swap the text and buttons for the name and tags fields."""
        if self._save_callback is None:
            return
        self._saving = True
        self.name_input.value = ""
        self.tags_input.value = ""
        self.save_status.value = ""
        _show(self.text, False, "block")
        _show(self.actions, False)
        _show(self.save_form, True)

    def submit_save(self) -> None:
        if self._save_callback is None:
            return
        error = self._save_callback(self.name_input.value, self.tags_input.value)
        if error:
            self.save_status.value = f'<span style="color: {_WARN}; font-size: 12px;">{error}</span>'
            return
        self.end_save()

    def end_save(self) -> None:
        self._saving = False
        self.save_status.value = ""
        _show(self.save_form, False)
        _show(self.text, True, "block")
        _show(self.actions, bool(self._has_selection))

    def _refresh_save_button(self) -> None:
        _show(self.save_button, self._save_callback is not None, "inline-flex")

    # -- history ---------------------------------------------------------
    def set_history(self, can_back: bool, can_forward: bool) -> None:
        """Enable ◀ ▶ when there is a selection to go back or forward to."""
        self.back_button.disabled = not can_back
        self.forward_button.disabled = not can_forward

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
        self._has_selection = active
        if self._saving and not active:
            self.end_save()  # nothing left to save
        if not self._saving:
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
