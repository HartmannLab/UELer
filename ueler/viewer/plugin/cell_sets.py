"""Cell sets: keep selections as named, tagged populations, collect cells in carts, and write labels to the cell table.

The plugin has four parts:

* **Save the current selection**: a name and free-form tags; the set keeps
  the cells, how the selection was made and the session's selection history
  (see :mod:`ueler.viewer.selection` and :mod:`ueler.viewer.cell_set_store`).
  The selection bar's **Save as set…** does the same from above the image.
  **New empty cart** makes a set with no cells, to fill later.
* **Active cart**: one set chosen to collect cells. While it is set, the
  selection bar shows **+ Cart** / **− Cart**, and Shift-click on a cell in the
  image puts it in the cart or takes it out. Without one, Shift-click says so in
  the bar and does nothing else.
* **Write to the cell table**: put a value into a column for the cells of the
  current selection, whichever view made it. This replaces the Cell Table
  Editor, which could only act on cells selected in the image. A write changes
  the cell table in memory and is noted in the session log; saving the selection
  as a set tagged ``column:value`` is what keeps it.
* **The Gallery** asks this plugin what it is showing (:meth:`gallery_source`)
  and calls it for its ★, remove, + Cart and Save shown controls, so the
  Gallery holds no copy of any set; after every change this plugin tells the
  Gallery (``on_cell_sets_change``).
* **Saved sets**: filter by name or tag, select a set's cells (or its ★
  favourites), star or unstar the selected cells in it, make it the active
  cart, take a frozen **Snapshot** of it, edit its name, tags and note, or delete
  it with its stars (asks first).

Selecting a set moves the view to its cells unless the FOV shown already holds
some of them: to the FOV with most of its cells, or to the stitched map when all
its FOVs belong to one. The cells are outlined on the mask layer they were saved
on.
"""

from __future__ import annotations

import html
import logging
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from datetime import datetime

import pandas as pd
from ipywidgets import HTML, Button, Combobox, Dropdown, HBox, Layout, Select, Text, Textarea, VBox

from ueler.viewer.selection import CellSelection
from ueler.viewer.cell_set_store import (
    FAVOURITES,
    CellSetStore,
    cells_from_indices,
    format_tags,
    indices_from_cells,
    parse_tags,
    system_columns,
    write_column,
)

from .plugin_base import PluginBase

_logger = logging.getLogger(__name__)

_OK = "var(--jp-success-color1, #2e7d32)"
_WARN = "var(--jp-warn-color0, #b45309)"
_ERROR = "var(--jp-error-color1, #c62828)"
_MUTED = "var(--jp-ui-font-color2, #616161)"


def _heading(text: str) -> HTML:
    return HTML(value=f"<b style='font-size: 12px;'>{text}</b>", layout=Layout(margin="6px 0 0 0"))


def _cells(count: int) -> str:
    return f"{count:,} {'cell' if count == 1 else 'cells'}"


class CellSetsPlugin(PluginBase):
    """Save, select, tag and combine cell populations; see the module doc."""

    def __init__(self, main_viewer, width: int = 6, height: int = 3) -> None:
        super().__init__(main_viewer, width, height)
        self.displayed_name = "Cell sets"
        self.SidePlots_id = "cell_sets_output"
        self.main_viewer = main_viewer
        self.store: Optional[CellSetStore] = None
        self._sets: List[Dict[str, Any]] = []
        self.active_cart: Optional[str] = None
        self._updating_cart = False
        self.ui_component = SimpleNamespace()
        self._build_widgets()
        self._build_layout()
        # No setup_widget_observers: names, tags and values typed here belong to
        # this session, not to the saved widget states.
        self.initialized = True

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        ui = self.ui_component
        full = dict(width="calc(100% - 5px)", max_width="calc(100% - 5px)")
        auto = {"description_width": "auto"}

        ui.current_label = HTML(value=self._current_text())

        ui.save_name = Text(value="", placeholder="Name of the cell set", description="Name:", style=auto, layout=Layout(**full))
        ui.save_tags = Text(value="", placeholder="Tags, comma-separated, e.g. pilot, figure 2", description="Tags:", style=auto, layout=Layout(**full))
        ui.save_button = Button(description="Save as set", icon="save", button_style="primary", disabled=True, layout=Layout(width="auto"),
                                tooltip="Keep the current selection as a named cell set")
        ui.save_button.on_click(lambda _b: self._on_save_clicked())
        ui.new_cart_button = Button(description="New empty cart", icon="shopping-cart", disabled=True, layout=Layout(width="auto"),
                                    tooltip="Make a set with no cells under this name and make it the active cart")
        ui.new_cart_button.on_click(lambda _b: self._on_new_cart_clicked())

        ui.cart_dropdown = Dropdown(options=[("None", None)], value=None, description="Active cart:", style=auto, layout=Layout(**full))
        ui.cart_dropdown.observe(lambda change: self.set_active_cart(change["new"]), names="value")

        ui.column_combo = Combobox(options=self._column_options(), value="", placeholder="Column (new or existing)", description="Column:",
                                   ensure_option=False, style=auto, layout=Layout(**full))
        ui.value_input = Text(value="", placeholder="Value to write", description="Value:", style=auto, layout=Layout(**full))
        ui.write_button = Button(description="Write to current selection", icon="pencil", disabled=True, layout=Layout(width="auto"),
                                 tooltip="Write the value into the column for every cell of the current selection")
        ui.write_button.on_click(lambda _b: self.write_to_selection())

        ui.filter_input = Text(value="", placeholder="Filter by name or tag", description="Filter:", style=auto, layout=Layout(**full))
        ui.filter_input.observe(lambda _change: self.refresh_sets(), names="value")
        ui.set_list = Select(options=[], rows=8, layout=Layout(**full))
        ui.set_list.observe(lambda _change: self._on_set_picked(), names="value")

        ui.edit_name = Text(value="", description="Name:", style=auto, layout=Layout(**full))
        ui.edit_tags = Text(value="", placeholder="Tags, comma-separated", description="Tags:", style=auto, layout=Layout(**full))
        ui.edit_note = Textarea(value="", placeholder="Note", description="Note:", style=auto, layout=Layout(height="48px", **full))
        ui.select_button = Button(description="Select cells", icon="crosshairs", button_style="info", disabled=True, layout=Layout(width="auto"),
                                  tooltip="Make this set the current selection and outline it in the image")
        ui.update_button = Button(description="Save changes", disabled=True, layout=Layout(width="auto"),
                                  tooltip="Save the name, tags and note (the cells never change)")
        ui.delete_button = Button(description="Delete", icon="trash", button_style="danger", disabled=True, layout=Layout(width="auto"),
                                  tooltip="Delete this set (asks first)")
        ui.select_stars_button = Button(description="Select ★", disabled=True, layout=Layout(width="auto"),
                                        tooltip="Make this set's starred cells the current selection")
        ui.star_button = Button(description="★ Star selection", disabled=True, layout=Layout(width="auto"),
                                tooltip="Star the selected cells in this set (cells not in it are added)")
        ui.unstar_button = Button(description="☆ Unstar", disabled=True, layout=Layout(width="auto"),
                                  tooltip="Take the star off the selected cells; they stay in the set")
        ui.make_cart_button = Button(description="Make active cart", icon="shopping-cart", disabled=True, layout=Layout(width="auto"),
                                     tooltip="Collect cells in this set with + Cart and Shift-click")
        ui.snapshot_button = Button(description="Snapshot", icon="camera", disabled=True, layout=Layout(width="auto"),
                                    tooltip="Save a frozen copy of this set, with its stars")
        ui.select_button.on_click(lambda _b: self.select_set(self._picked_id()))
        ui.select_stars_button.on_click(lambda _b: self.select_stars(self._picked_id()))
        ui.star_button.on_click(lambda _b: self.star_selection(self._picked_id()))
        ui.unstar_button.on_click(lambda _b: self.star_selection(self._picked_id(), unstar=True))
        ui.make_cart_button.on_click(lambda _b: self._make_picked_active())
        ui.snapshot_button.on_click(lambda _b: self.snapshot_set(self._picked_id()))
        ui.update_button.on_click(lambda _b: self._on_update_clicked())
        ui.delete_button.on_click(lambda _b: self.ask_delete_set(self._picked_id()))
        ui.details = HTML(value="")

        ui.status = HTML(value="")

    def _build_layout(self) -> None:
        ui = self.ui_component
        row = dict(gap="4px", flex_flow="row wrap")
        self.ui = VBox(
            [
                ui.current_label,
                _heading("Save the current selection"),
                ui.save_name,
                ui.save_tags,
                HBox([ui.save_button, ui.new_cart_button], layout=Layout(**row)),
                _heading("Active cart"),
                ui.cart_dropdown,
                _heading("Write to the cell table"),
                ui.column_combo,
                ui.value_input,
                ui.write_button,
                _heading("Saved sets"),
                ui.filter_input,
                ui.set_list,
                ui.details,
                ui.edit_name,
                ui.edit_tags,
                ui.edit_note,
                HBox([ui.select_button, ui.select_stars_button, ui.star_button, ui.unstar_button], layout=Layout(**row)),
                HBox([ui.make_cart_button, ui.snapshot_button, ui.update_button, ui.delete_button], layout=Layout(**row)),
                ui.status,
            ],
            layout=Layout(width="100%", gap="4px", padding="4px"),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def after_all_plugins_loaded(self) -> None:
        """Open the store and connect to the selection hub and bar.

        Does not call ``super()``: nothing here is restored from widget states.
        """
        viewer = self.main_viewer
        base_folder = getattr(viewer, "base_folder", None)
        if base_folder:
            try:
                self.store = CellSetStore(base_folder, storage_root=getattr(viewer, "settings_root", None))
            except OSError as exc:
                self._status(f"Cell sets cannot be saved: {html.escape(str(exc))}", _ERROR)
        hub = getattr(viewer, "selection_hub", None)
        if hub is not None:
            hub.subscribe(lambda _selection: self._on_selection())
        bar = self._bar
        if bar is not None and self.store is not None and hasattr(bar, "on_save"):
            bar.on_save(self.save_selection)
            if hasattr(bar, "set_cart"):
                bar.on_action("cart_add", self.add_selection_to_cart)
                bar.on_action("cart_remove", lambda: self.add_selection_to_cart(remove=True))
        display = getattr(viewer, "image_display", None)
        if display is not None and self.store is not None:
            display.shift_click_handler = self.toggle_cell_in_cart
        self._on_selection()
        self.refresh_sets()

    def on_cell_table_change(self) -> None:
        self.ui_component.column_combo.options = self._column_options()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def _hub(self):
        return getattr(self.main_viewer, "selection_hub", None)

    @property
    def _bar(self):
        return getattr(getattr(self.main_viewer, "ui_component", None), "selection_bar", None)

    def _tell(self, text: str, color: str = _OK, *, bar: bool = True) -> None:
        """Say *text* in the plugin, and in the selection bar until the next selection."""
        self._status(text, color)
        if bar and self._bar is not None:
            self._bar.set_message(text, warning=color != _OK)

    def _current(self):
        hub = self._hub
        return hub.current if hub is not None else None

    def _current_text(self) -> str:
        current = self._current()
        if current is None or not current.indices:
            return f"<span style='color: {_MUTED}; font-size: 12px;'>No cells selected. Select cells in the image or a plot, or pick a saved set below.</span>"
        return f"<span style='font-size: 12px;'>Current selection: <b>{_cells(len(current))}</b> · {html.escape(current.source_label)}</span>"

    def _column_options(self) -> List[str]:
        table = getattr(self.main_viewer, "cell_table", None)
        if table is None:
            return []
        reserved = system_columns(self.main_viewer)
        return [str(c) for c in table.columns if str(c) not in reserved]

    def _status(self, text: str, color: str = _MUTED) -> None:
        self.ui_component.status.value = f"<span style='color: {color}; font-size: 12px;'>{text}</span>" if text else ""

    def _on_selection(self) -> None:
        ui = self.ui_component
        current = self._current()
        has = bool(current is not None and current.indices)
        ui.current_label.value = self._current_text()
        ui.save_button.disabled = not has or self.store is None
        ui.new_cart_button.disabled = self.store is None
        ui.write_button.disabled = not has
        self._refresh_set_buttons()

    def _image_masks(self, current) -> Optional[Dict]:
        """``(fov, label) -> mask`` for cells selected in the image, if it shows the selection."""
        if current is None or not current.shows_in("image"):
            return None
        selected = getattr(getattr(self.main_viewer, "image_display", None), "selected_masks_label", None) or ()
        return {(str(s.fov), int(s.mask_id)): str(s.mask) for s in selected}

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save_selection(self, name: str, tags_text: str = "", note: str = "") -> str:
        """Save the current selection as a set; return an error message, or ``""``."""
        if self.store is None:
            return "Cell sets cannot be saved without a dataset folder."
        current = self._current()
        if current is None or not current.indices:
            return "Nothing is selected."
        tags, problems = parse_tags(tags_text)
        if problems:
            return "Tags: " + "; ".join(html.escape(p) for p in problems)
        cells = cells_from_indices(self.main_viewer, current.indices, self._image_masks(current))
        if not len(cells):
            return "None of the selected cells is in the cell table."
        parents = []
        set_id = current.recipe.get("set_id") or current.recipe.get("recipe", {}).get("set_id")
        if set_id:
            parents.append(set_id)
        hub = self._hub
        try:
            meta = self.store.save_set(
                name,
                cells,
                tags=tags,
                note=note,
                recipe=current.recipe,
                source=current.source,
                parents=parents,
                trail=hub.trail() if hub is not None else (),
            )
        except (ValueError, OSError) as exc:
            return html.escape(str(exc))
        if hub is not None:
            hub.note("saved as set", {"set_id": meta["id"], "name": meta["name"], "tags": meta["tags"]}, count=meta["count"])
        self._after_change(select=meta["id"])
        saved = f"Saved {_cells(meta['count'])} as the cell set <b>{html.escape(meta['name'])}</b>."
        self._status(saved, _OK)
        bar = getattr(getattr(self.main_viewer, "ui_component", None), "selection_bar", None)
        if bar is not None:
            bar.set_message(saved)  # until the next selection
        return ""

    def _on_save_clicked(self) -> None:
        ui = self.ui_component
        error = self.save_selection(ui.save_name.value, ui.save_tags.value)
        if error:
            self._status(error, _ERROR)
            return
        ui.save_name.value = ""
        ui.save_tags.value = ""

    # ------------------------------------------------------------------
    # Writing to the cell table
    # ------------------------------------------------------------------
    def write_to_selection(self) -> int:
        """Write the Value into the Column for the current selection's rows."""
        ui = self.ui_component
        current = self._current()
        if current is None or not current.indices:
            self._status("No cells selected.", _ERROR)
            return 0
        if getattr(self.main_viewer, "cell_table", None) is None:
            self._status("No cell table loaded.", _ERROR)
            return 0
        column = (ui.column_combo.value or "").strip()
        value = ui.value_input.value
        try:
            written = write_column(self.main_viewer, current.indices, column, value)
        except ValueError as exc:
            self._status(html.escape(str(exc)), _ERROR)
            return 0
        if not written:
            self._status("None of the selected cells is in the cell table.", _WARN)
            return 0
        hub = self._hub
        if hub is not None:
            hub.note("wrote to the cell table", {"column": column, "value": value}, count=written)
        inform = getattr(self.main_viewer, "inform_plugins", None)
        if callable(inform):
            inform("on_cell_table_change")
        # Offer the write as a tag, so saving the selection keeps it.
        if value and ":" not in column:
            tag = f"{column}:{value}"
            tags, _ = parse_tags(ui.save_tags.value)
            if tag not in tags:
                ui.save_tags.value = format_tags([*tags, tag])
        self._status(
            f"Wrote <b>{html.escape(value)}</b> into <b>{html.escape(column)}</b> for {_cells(written)}. "
            "The cell table changes for this session; save the selection as a set to keep it.",
            _OK,
        )
        return written

    # ------------------------------------------------------------------
    # The list of sets
    # ------------------------------------------------------------------
    def _picked_id(self) -> Optional[str]:
        return self.ui_component.set_list.value or None

    def _entry(self, set_id: Optional[str]) -> Optional[Dict[str, Any]]:
        return next((entry for entry in self._sets if entry.get("id") == set_id), None)

    def _picked(self) -> Optional[Dict[str, Any]]:
        return self._entry(self._picked_id())

    def _display_name(self, entry: Dict[str, Any]) -> str:
        """A set's name; a ★ favourites child is named after its set ("★ CD8 T")."""
        if entry.get("role") == FAVOURITES:
            parent = self._entry(entry.get("container"))
            return f"★ {parent.get('name', '?')}" if parent else str(entry.get("name", "?"))
        return str(entry.get("name", "?"))

    def _path_name(self, entry: Dict[str, Any]) -> str:
        """``parent / child`` for a sub-group, the name for a top-level set."""
        parent = self._entry(entry.get("container")) if entry.get("container") else None
        name = str(entry.get("name", "?"))
        return f"{parent.get('name', '?')} / {name}" if parent else name

    def _stars(self, set_id: Optional[str]) -> Optional[Dict[str, Any]]:
        return next((e for e in self._sets if e.get("container") == set_id and e.get("role") == FAVOURITES), None)

    def _row_label(self, entry: Dict[str, Any]) -> str:
        stars = self._stars(entry.get("id"))
        marks = ""
        if entry.get("id") == self.active_cart:
            marks += "🛒 "
        if entry.get("frozen"):
            marks += "🔒 "
        star_text = f" · ★ {int(stars.get('count', 0))}" if stars and stars.get("count") else ""
        tags = f" · {format_tags(entry.get('tags', []))}" if entry.get("tags") else ""
        return f"{marks}{entry.get('name', '?')} — {_cells(int(entry.get('count', 0)))}{star_text}{tags}"

    def _after_change(self, select: Optional[str] = None) -> None:
        """After a set changed: refresh the list and tell the Gallery."""
        self.refresh_sets(select=select)
        gallery = getattr(getattr(self.main_viewer, "SidePlots", None), "cell_gallery_output", None)
        notify = getattr(gallery, "on_cell_sets_change", None)
        if callable(notify):
            try:
                notify()
            except Exception:  # the Gallery must not break a set edit
                _logger.warning("The Gallery could not follow a cell set change", exc_info=True)

    def refresh_sets(self, select: Optional[str] = None) -> None:
        """Re-read the index and show the top-level sets that match the filter."""
        ui = self.ui_component
        self._sets = self.store.list_sets() if self.store is not None else []
        if self.active_cart and self._entry(self.active_cart) is None:
            self.active_cart = None
        needle = (ui.filter_input.value or "").strip().casefold()

        def matches(entry):
            return (
                not needle
                or needle in str(entry.get("name", "")).casefold()
                or any(needle in str(tag).casefold() for tag in entry.get("tags", []))
            )

        # Top-level sets, each followed by its sub-groups (not its ★, which is a count).
        options = []
        for entry in (e for e in self._sets if e.get("container") is None):
            subs = [e for e in self._sets if e.get("container") == entry["id"] and e.get("role") != FAVOURITES]
            shown_subs = subs if matches(entry) else [e for e in subs if matches(e)]
            if matches(entry) or shown_subs:
                options.append((self._row_label(entry), entry["id"]))
                options.extend(("    ↳ " + self._row_label(sub), sub["id"]) for sub in shown_subs)
        keep = select or self._picked_id()
        ui.set_list.options = options
        ids = [value for _label, value in options]
        ui.set_list.value = keep if keep in ids else None
        self._refresh_cart_choices()
        self._on_set_picked()

    def _refresh_cart_choices(self) -> None:
        """The active-cart dropdown: the top-level sets that are not snapshots."""
        ui = self.ui_component
        carts = [e for e in self._sets if e.get("role") != FAVOURITES and not e.get("frozen")]
        options = [("None", None)] + [(self._path_name(e), e["id"]) for e in carts]
        self._updating_cart = True
        try:
            ui.cart_dropdown.options = options
            ui.cart_dropdown.value = self.active_cart if self.active_cart in [o[1] for o in options] else None
        finally:
            self._updating_cart = False
        bar = self._bar
        if bar is not None and hasattr(bar, "set_cart"):
            cart = self._entry(self.active_cart)
            bar.set_cart(cart.get("name") if cart else None)

    def _refresh_set_buttons(self) -> None:
        ui = self.ui_component
        entry = self._picked()
        current = self._current()
        has_selection = bool(current is not None and current.indices)
        editable = entry is not None and not entry.get("frozen")
        for button in (ui.select_button, ui.update_button, ui.delete_button, ui.snapshot_button):
            button.disabled = entry is None
        stars = self._stars(entry.get("id")) if entry else None
        ui.select_stars_button.disabled = not (stars and stars.get("count"))
        ui.star_button.disabled = not (editable and has_selection)
        ui.unstar_button.disabled = not (editable and has_selection and stars and stars.get("count"))
        ui.make_cart_button.disabled = not editable or entry.get("id") == self.active_cart

    def _on_set_picked(self) -> None:
        ui = self.ui_component
        entry = self._picked()
        self._refresh_set_buttons()
        if entry is None:
            ui.edit_name.value = ui.edit_tags.value = ui.edit_note.value = ""
            ui.details.value = ""
            return
        ui.edit_name.value = entry.get("name", "")
        ui.edit_tags.value = format_tags(entry.get("tags", []))
        ui.edit_note.value = entry.get("note", "")
        steps = sum(1 for step in entry.get("trail", []) if step.get("kind") == "selection")
        made = html.escape(CellSelection(frozenset(), str(entry.get("source", "")), recipe=entry.get("recipe") or {}).source_label)
        fovs = int(entry.get("fovs", 0))
        stars = self._stars(entry.get("id"))
        parts = [
            f"{_cells(int(entry.get('count', 0)))} in {fovs} {'FOV' if fovs == 1 else 'FOVs'}",
            f"★ {int(stars.get('count', 0))}" if stars and stars.get("count") else "",
            f"from {made or 'a selection'}",
            f"saved {html.escape(str(entry.get('created', '')))}",
            f"{steps} selection {'step' if steps == 1 else 'steps'} recorded",
        ]
        edits = entry.get("edits") or []
        if edits:
            parts.append(f"edited {len(edits)} {'time' if len(edits) == 1 else 'times'}, last {html.escape(str(edits[-1].get('time', '')))}")
        if entry.get("frozen"):
            parts.append("snapshot: its cells cannot change")
        if entry.get("id") == self.active_cart:
            parts.append("the active cart")
        ui.details.value = f"<span style='color: {_MUTED}; font-size: 12px;'>{' · '.join(p for p in parts if p)}</span>"

    def _on_update_clicked(self) -> None:
        ui = self.ui_component
        entry = self._picked()
        if entry is None or self.store is None:
            return
        tags, problems = parse_tags(ui.edit_tags.value)
        if problems:
            self._status("Tags: " + "; ".join(html.escape(p) for p in problems), _ERROR)
            return
        try:
            meta = self.store.update_set(entry["id"], name=ui.edit_name.value, tags=tags, note=ui.edit_note.value)
        except (ValueError, KeyError, OSError) as exc:
            self._status(html.escape(str(exc)), _ERROR)
            return
        self._after_change(select=meta["id"])
        self._status(f"Saved the changes to <b>{html.escape(meta['name'])}</b>.", _OK)

    def ask_delete_set(self, set_id: Optional[str]) -> None:
        entry = self._entry(set_id)
        if entry is None or self.store is None:
            return
        stars = self._stars(set_id)
        star_text = f", its {int(stars.get('count', 0))} stars" if stars and stars.get("count") else ""
        subs = [e for e in self._sets if e.get("container") == set_id and e.get("role") != FAVOURITES]
        if subs:
            star_text += f", its {len(subs)} sub-group{'s' if len(subs) != 1 else ''}"
        asked = self.confirm(
            f"The cell set <b>{html.escape(entry['name'])}</b> ({_cells(int(entry.get('count', 0)))}){star_text} "
            "and its recorded history will be deleted from disk. This cannot be undone.",
            lambda: self._delete_confirmed(set_id),
            title="Delete cell set?",
            confirm_label="Delete",
        )
        if not asked:
            self._status("Not deleted: there is no dialog to confirm with.", _WARN)

    def _delete_confirmed(self, set_id: str) -> None:
        entry = self._entry(set_id)
        try:
            self.store.delete_set(set_id)
        except (KeyError, OSError) as exc:
            self._status(html.escape(str(exc)), _ERROR)
            return
        if self.active_cart == set_id:
            self.active_cart = None
        self._after_change()
        self._status(f"Deleted <b>{html.escape(entry['name'] if entry else set_id)}</b>.", _OK)

    # ------------------------------------------------------------------
    # Carts
    # ------------------------------------------------------------------
    def set_active_cart(self, set_id: Optional[str]) -> None:
        """Collect cells in *set_id* with + Cart / − Cart and Shift-click; ``None`` stops."""
        if getattr(self, "_updating_cart", False):
            return
        entry = self._entry(set_id) if set_id else None
        if entry is not None and entry.get("frozen"):
            self._status(f"<b>{html.escape(entry['name'])}</b> is a snapshot; its cells cannot change.", _ERROR)
            entry = None
        self.active_cart = entry["id"] if entry else None
        self._after_change()
        if entry:
            self._status(f"Collecting cells in <b>{html.escape(entry['name'])}</b>: + Cart in the bar, or Shift-click a cell in the image.", _OK)

    def _make_picked_active(self) -> None:
        self.set_active_cart(self._picked_id())

    def _on_new_cart_clicked(self) -> None:
        ui = self.ui_component
        error = self.create_cart(ui.save_name.value, ui.save_tags.value)
        if error:
            self._status(error, _ERROR)
            return
        ui.save_name.value = ""
        ui.save_tags.value = ""

    def create_cart(self, name: str, tags_text: str = "") -> str:
        """Make an empty set named *name* and make it the active cart; return an error, or ``""``."""
        if self.store is None:
            return "Cell sets cannot be saved without a dataset folder."
        tags, _ = parse_tags(tags_text)
        try:
            meta = self.store.save_set(name, None, tags=tags, source="cart", allow_empty=True)
        except (ValueError, OSError) as exc:
            return html.escape(str(exc))
        hub = self._hub
        if hub is not None:
            hub.note("new cart", {"set_id": meta["id"], "name": meta["name"]}, count=0)
        self.active_cart = meta["id"]
        self._after_change(select=meta["id"])
        self._status(f"Made the empty cart <b>{html.escape(meta['name'])}</b> and made it active.", _OK)
        return ""

    def _edit_cells(self, set_id: str, cells, recipe: Dict[str, Any], *, remove: bool = False) -> Optional[int]:
        """Add *cells* to the set or remove them; note it in the session log. ``None`` on failure."""
        entry = self._entry(set_id)
        try:
            if remove:
                changed = self.store.remove_cells(set_id, cells, recipe)
            else:
                changed = self.store.add_cells(set_id, cells, recipe)
        except (ValueError, KeyError, OSError) as exc:
            self._tell(html.escape(str(exc)), _ERROR)
            return None
        hub = self._hub
        if hub is not None:
            hub.note("removed from cart" if remove else "added to cart", {"set_id": set_id, "name": entry["name"] if entry else ""}, count=changed)
        self._after_change()
        return changed

    def add_selection_to_cart(self, remove: bool = False) -> Optional[int]:
        """**+ Cart** / **− Cart**: put the current selection in the active cart or take it out."""
        cart = self._entry(self.active_cart)
        if cart is None or self.store is None:
            self._tell("No active cart. Choose one in the Cell sets plugin.", _WARN)
            return None
        current = self._current()
        if current is None or not current.indices:
            self._tell("Nothing is selected.", _WARN)
            return None
        cells = cells_from_indices(self.main_viewer, current.indices, self._image_masks(current))
        if not len(cells):
            self._tell("None of the selected cells is in the cell table.", _WARN)
            return None
        recipe = {"source": current.source, "recipe": dict(current.recipe or {})}
        changed = self._edit_cells(cart["id"], cells, recipe, remove=remove)
        if changed is None:
            return None
        total = _cells(int((self._entry(cart["id"]) or {}).get("count", 0)))
        name = html.escape(cart["name"])
        if remove:
            self._tell(f"Took {_cells(changed)} out of the cart <b>{name}</b> (now {total}).")
        elif changed < len(cells):
            self._tell(f"Added {_cells(changed)} to the cart <b>{name}</b> ({len(cells) - changed:,} already in it; now {total}).")
        else:
            self._tell(f"Added {_cells(changed)} to the cart <b>{name}</b> (now {total}).")
        return changed

    def toggle_cell_in_cart(self, hit) -> Optional[str]:
        """Shift-click in the image: put the cell in the active cart, or take it out.

        Returns ``"added"`` or ``"removed"``, or ``None`` when nothing changed.
        """
        cart = self._entry(self.active_cart)
        if cart is None or self.store is None:
            self._tell("Shift-click puts a cell in the active cart, and there is none. Choose one in the Cell sets plugin.", _WARN)
            return None
        cell = pd.DataFrame({"fov": [str(hit.fov_name)], "mask": [str(hit.mask_name)], "label": [int(hit.mask_id)]})
        try:
            inside = bool(self.store.contains(cart["id"], cell).iloc[0])
        except (KeyError, OSError, ValueError) as exc:
            self._tell(html.escape(str(exc)), _ERROR)
            return None
        recipe = {"source": "image shift-click", "fov": str(hit.fov_name), "mask": str(hit.mask_name), "label": int(hit.mask_id)}
        if self._edit_cells(cart["id"], cell, recipe, remove=inside) is None:
            return None
        total = _cells(int((self._entry(cart["id"]) or {}).get("count", 0)))
        what = f"cell {int(hit.mask_id)} ({html.escape(str(hit.fov_name))})"
        name = html.escape(cart["name"])
        if inside:
            self._tell(f"Took {what} out of the cart <b>{name}</b> (now {total}).")
            return "removed"
        self._tell(f"Added {what} to the cart <b>{name}</b> (now {total}).")
        return "added"

    # ------------------------------------------------------------------
    # Stars and snapshots
    # ------------------------------------------------------------------
    def star_selection(self, set_id: Optional[str], unstar: bool = False) -> Optional[int]:
        """Star (or unstar) the current selection's cells in the set."""
        entry = self._entry(set_id)
        current = self._current()
        if entry is None or self.store is None or current is None or not current.indices:
            return None
        cells = cells_from_indices(self.main_viewer, current.indices, self._image_masks(current))
        recipe = {"source": current.source, "recipe": dict(current.recipe or {})}
        try:
            changed = self.store.unstar(set_id, cells, recipe) if unstar else self.store.star(set_id, cells, recipe)
        except (ValueError, KeyError, OSError) as exc:
            self._status(html.escape(str(exc)), _ERROR)
            return None
        hub = self._hub
        if hub is not None:
            hub.note("unstarred" if unstar else "starred", {"set_id": set_id, "name": entry["name"]}, count=changed)
        self._after_change(select=set_id)
        name = html.escape(entry["name"])
        self._status(f"{'Unstarred' if unstar else 'Starred'} {_cells(changed)} in <b>{name}</b>.", _OK)
        return changed

    def select_stars(self, set_id: Optional[str]) -> bool:
        stars = self._stars(set_id)
        return self.select_set(stars["id"]) if stars else False

    def snapshot_set(self, set_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Save a frozen copy of the set, named after it and the time."""
        entry = self._entry(set_id)
        if entry is None or self.store is None:
            return None
        name = f"{entry['name']} (snapshot {datetime.now().strftime('%Y-%m-%d %H:%M')})"
        try:
            meta = self.store.snapshot(set_id, name)
        except (ValueError, KeyError, OSError) as exc:
            self._status(html.escape(str(exc)), _ERROR)
            return None
        hub = self._hub
        if hub is not None:
            hub.note("snapshot", {"set_id": meta["id"], "of": set_id, "name": meta["name"]}, count=meta.get("count", 0))
        self._after_change(select=meta["id"])
        self._status(f"Saved the snapshot <b>{html.escape(meta['name'])}</b>; rename it below if you like.", _OK)
        return meta

    # ------------------------------------------------------------------
    # The Gallery: what it shows, and its ★ / remove / cart / save controls
    # ------------------------------------------------------------------
    @staticmethod
    def _set_id_of(selection) -> Optional[str]:
        """The set a selection came from: selected directly, or restored with ◀."""
        recipe = getattr(selection, "recipe", None) or {}
        return recipe.get("set_id") or (recipe.get("recipe") or {}).get("set_id")

    def rows_of(self, set_id: Optional[str]) -> Set:
        """The cell-table rows of the set's cells now (cells gone from the table left out)."""
        if not set_id or self.store is None:
            return set()
        try:
            cells = self.store.read_cells(set_id)
        except (KeyError, OSError, ValueError):
            return set()
        rows, _missing = indices_from_cells(self.main_viewer, cells)
        return rows

    def gallery_source(self, selection) -> Dict[str, Any]:
        """Describe what the Gallery shows for *selection* (``None``: cells sent without one).

        Keys: ``kind`` (``"set"``, ``"stars"``, ``"snapshot"`` or
        ``"selection"``); ``label`` (header text, HTML); ``set_id``; ``rows``
        (the set's rows now, or ``None`` to use the cells the Gallery was given);
        ``star_set`` (the set whose stars the tiles show and toggle, or ``None``);
        ``starred`` (its starred rows); ``can_star``; ``remove_set`` (the set ✕
        removes from, or ``None``); ``container`` (where **Save shown** puts its
        set); ``cart`` (the active cart's id, for **+ Cart**).
        """
        cart = self._entry(self.active_cart)
        source: Dict[str, Any] = {
            "kind": "selection",
            "set_id": None,
            "rows": None,
            "star_set": None,
            "starred": set(),
            "can_star": False,
            "remove_set": None,
            "container": None,
            "cart": cart["id"] if cart else None,
        }
        set_id = self._set_id_of(selection) if selection is not None else None
        entry = self.store.get(set_id) if (set_id and self.store is not None) else None
        if entry is None:
            made = html.escape(selection.source_label) if selection is not None and selection.source_label else "cells sent to the Gallery"
            source["label"] = f"Selection: {made}"
            if cart:
                source.update(star_set=cart["id"], can_star=True)
                source["label"] += f" · ★ goes to the cart <b>{html.escape(cart['name'])}</b>"
        elif entry.get("role") == FAVOURITES:
            parent = self.store.get(entry.get("container")) or {}
            source.update(
                kind="stars",
                set_id=entry["id"],
                rows=self.rows_of(entry["id"]),
                star_set=parent.get("id"),
                can_star=bool(parent) and not parent.get("frozen"),
                container=parent.get("id"),
                label=f"★ of <b>{html.escape(str(parent.get('name', '?')))}</b>",
            )
        elif entry.get("frozen"):
            source.update(
                kind="snapshot",
                set_id=entry["id"],
                rows=self.rows_of(entry["id"]),
                star_set=entry["id"],
                container=entry["id"],
                label=f"Snapshot: <b>{html.escape(entry['name'])}</b> (read-only)",
            )
        else:
            word = "Cart" if entry["id"] == self.active_cart else "Set"
            source.update(
                kind="set",
                set_id=entry["id"],
                rows=self.rows_of(entry["id"]),
                star_set=entry["id"],
                can_star=True,
                remove_set=entry["id"],
                container=entry["id"],
                label=f"{word}: <b>{html.escape(entry['name'])}</b>",
            )
        if source["star_set"]:
            stars = self.store.favourites(source["star_set"]) if self.store is not None else None
            source["starred"] = self.rows_of(stars["id"]) if stars else set()
        return source

    def _cells_for_rows(self, rows: Iterable, set_id: Optional[str] = None) -> pd.DataFrame:
        """The cells of *rows*, on the mask layer *set_id* holds them on, if it does."""
        cells = cells_from_indices(self.main_viewer, rows)
        if set_id and len(cells) and self.store is not None:
            try:
                known = self.store.read_cells(set_id)
            except (KeyError, OSError, ValueError):
                known = None
            if known is not None and len(known):
                masks = {(str(f), int(l)): str(m) for f, m, l in known[["fov", "mask", "label"]].itertuples(index=False)}
                cells["mask"] = [masks.get((f, int(l)), m) for f, m, l in cells[["fov", "mask", "label"]].itertuples(index=False)]
        return cells

    def star_rows(self, set_id: Optional[str], rows: Iterable, unstar: bool = False) -> Optional[int]:
        """Star (or unstar) cell-table *rows* in the set; ``None`` when refused."""
        if not set_id or self.store is None:
            return None
        entry = self.store.get(set_id)
        cells = self._cells_for_rows(rows, set_id)
        recipe = {"source": "gallery"}
        try:
            changed = self.store.unstar(set_id, cells, recipe) if unstar else self.store.star(set_id, cells, recipe)
        except (ValueError, KeyError, OSError) as exc:
            self._status(html.escape(str(exc)), _ERROR)
            return None
        hub = self._hub
        if hub is not None:
            hub.note("unstarred" if unstar else "starred", {"set_id": set_id, "name": (entry or {}).get("name", ""), "from": "gallery"}, count=changed)
        self._after_change()
        return changed

    def remove_rows(self, set_id: Optional[str], rows: Iterable) -> Optional[int]:
        """Take cell-table *rows* out of the set (and its stars); ``None`` when refused."""
        if not set_id or self.store is None:
            return None
        changed = self._edit_cells(set_id, self._cells_for_rows(rows, set_id), {"source": "gallery"}, remove=True)
        if changed is not None:
            entry = self._entry(set_id) or {}
            self._status(f"Took {_cells(changed)} out of <b>{html.escape(str(entry.get('name', '')))}</b>.", _OK)
        return changed

    def add_rows_to_cart(self, rows: Iterable) -> Optional[int]:
        """Put cell-table *rows* in the active cart; ``None`` when there is none."""
        cart = self._entry(self.active_cart)
        if cart is None or self.store is None:
            self._status("No active cart. Choose one under Active cart.", _WARN)
            return None
        rows = list(rows)
        changed = self._edit_cells(cart["id"], self._cells_for_rows(rows, cart["id"]), {"source": "gallery", "shown": len(rows)})
        if changed is not None:
            self._status(f"Added {_cells(changed)} from the Gallery to the cart <b>{html.escape(cart['name'])}</b>.", _OK)
        return changed

    def save_rows(
        self,
        name: str,
        rows: Iterable,
        *,
        container: Optional[str] = None,
        sample: Optional[Dict[str, Any]] = None,
        default_name: str = "Gallery cells",
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Save cell-table *rows* as a set (a child of *container* when given).

        An empty *name* becomes *default_name*, numbered if taken. Returns
        ``(metadata, "")`` or ``(None, error)``.
        """
        if self.store is None:
            return None, "Cell sets cannot be saved without a dataset folder."
        cells = self._cells_for_rows(rows, container)
        if not len(cells):
            return None, "None of the shown cells is in the cell table."
        typed = (name or "").strip()
        candidates = [typed] if typed else [default_name] + [f"{default_name} {n}" for n in range(2, 100)]
        hub = self._hub
        error = ""
        for candidate in candidates:
            try:
                meta = self.store.save_set(
                    candidate,
                    cells,
                    source="gallery",
                    recipe={"gallery": dict(sample or {})},
                    container=container,
                    sample=sample,
                    trail=hub.trail() if hub is not None else (),
                )
            except ValueError as exc:
                error = html.escape(str(exc))
                continue
            except (KeyError, OSError) as exc:
                return None, html.escape(str(exc))
            if hub is not None:
                hub.note("saved shown cells as set", {"set_id": meta["id"], "name": meta["name"], "container": container}, count=meta["count"])
            self._after_change(select=container or meta["id"])
            self._status(f"Saved the {_cells(meta['count'])} shown in the Gallery as <b>{html.escape(meta['name'])}</b>.", _OK)
            return meta, ""
        return None, error

    # ------------------------------------------------------------------
    # Selecting a set
    # ------------------------------------------------------------------
    def select_set(self, set_id: Optional[str]) -> bool:
        """Make the set the current selection, move the view to it and outline it."""
        if not set_id or self.store is None:
            return False
        entry = next((e for e in self._sets if e.get("id") == set_id), None) or self.store.get(set_id)
        if entry is None:
            return False
        name = self._display_name(entry)
        viewer = self.main_viewer
        try:
            cells = self.store.read_cells(set_id)
        except (KeyError, OSError, ValueError) as exc:
            self._status(f"Cannot read the set: {html.escape(str(exc))}", _ERROR)
            return False
        if not len(cells):
            self._status(f"<b>{html.escape(name)}</b> has no cells yet.", _WARN)
            return False
        indices, missing = indices_from_cells(viewer, cells)
        if not indices:
            self._status(f"None of the {_cells(len(cells))} of <b>{html.escape(name)}</b> is in the cell table any more.", _ERROR)
            return False
        self.move_view_to(cells)
        masks = entry.get("masks") or sorted(cells["mask"].unique().tolist())
        known = set(getattr(viewer, "mask_names", ()) or ())
        mask = masks[0] if len(masks) == 1 and (not known or masks[0] in known) else None
        recipe = {"set_id": set_id, "name": name}
        if mask:
            recipe["mask"] = mask
        self._hub.publish(indices, "set", (), recipe)
        send = getattr(viewer, "send_selection_to_image", None)
        if callable(send):
            send()
        if missing:
            self._status(f"{_cells(missing)} of <b>{html.escape(name)}</b> are no longer in the cell table.", _WARN)
        else:
            self._status(f"Selected <b>{html.escape(name)}</b>.", _OK)
        return True

    def move_view_to(self, cells) -> None:
        """Show the set's cells, unless the view already shows some of them.

        Goes to the FOV holding most of them, or to the stitched map when all of
        their FOVs belong to one map.
        """
        viewer = self.main_viewer
        ui = getattr(viewer, "ui_component", None)
        counts = cells["fov"].astype(str).value_counts()
        if not len(counts) or ui is None:
            return
        fovs = set(counts.index)
        if getattr(viewer, "_map_mode_active", False):
            descriptor = getattr(viewer, "_map_descriptors", {}).get(getattr(viewer, "_active_map_id", None))
            shown = {str(getattr(spec, "name", spec)) for spec in getattr(descriptor, "fovs", ())}
        else:
            active = viewer.get_active_fov() if hasattr(viewer, "get_active_fov") else None
            shown = {str(active)} if active else set()
        if shown & fovs:
            return
        lookup = getattr(viewer, "_map_id_for_fov", None)
        map_ids = {lookup(fov) for fov in fovs} if callable(lookup) else {None}
        toggle = getattr(ui, "map_mode_toggle", None)
        selector = getattr(ui, "map_selector", None)
        try:
            if len(fovs) > 1 and len(map_ids) == 1 and None not in map_ids and toggle is not None and selector is not None:
                selector.value = next(iter(map_ids))
                toggle.value = True
                return
            target = counts.index[0]
            image_selector = getattr(ui, "image_selector", None)
            options = [str(o[1] if isinstance(o, tuple) else o) for o in getattr(image_selector, "options", ())]
            if image_selector is None or target not in options:
                return
            if toggle is not None and getattr(toggle, "value", False):
                toggle.value = False
            image_selector.value = target
        except Exception:  # a view that cannot move leaves the selection usable
            _logger.debug("Could not move the view to the cell set", exc_info=True)
