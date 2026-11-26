"""Cell sets: keep selections as named, tagged populations and write labels to the cell table.

The plugin has three parts:

* **Save the current selection**: a name and ``key:value`` tags; the set keeps
  the cells, how the selection was made and the session's selection history
  (see :mod:`ueler.viewer.selection` and :mod:`ueler.viewer.cell_set_store`).
  The selection bar's **Save as set…** does the same from above the image.
* **Write to the cell table**: put a value into a column for the cells of the
  current selection, whichever view made it. This replaces the Cell Table
  Editor, which could only act on cells selected in the image. A write changes
  the cell table in memory and is noted in the session log; saving the selection
  as a set tagged ``column:value`` is what keeps it.
* **Saved sets**: filter by name or tag, select a set's cells, edit its name,
  tags and note, or delete it (asks first).

Selecting a set moves the view to its cells unless the FOV shown already holds
some of them: to the FOV with most of its cells, or to the stitched map when all
its FOVs belong to one. The cells are outlined on the mask layer they were saved
on.
"""

from __future__ import annotations

import html
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from ipywidgets import HTML, Button, Combobox, HBox, Layout, Select, Text, Textarea, VBox

from ueler.viewer.selection import CellSelection
from ueler.viewer.cell_set_store import (
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
        ui.save_tags = Text(value="", placeholder="key:value, … e.g. celltype:CD8 T", description="Tags:", style=auto, layout=Layout(**full))
        ui.save_button = Button(description="Save as set", icon="save", button_style="primary", disabled=True, layout=Layout(width="auto"),
                                tooltip="Keep the current selection as a named cell set")
        ui.save_button.on_click(lambda _b: self._on_save_clicked())

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
        ui.edit_tags = Text(value="", placeholder="key:value, …", description="Tags:", style=auto, layout=Layout(**full))
        ui.edit_note = Textarea(value="", placeholder="Note", description="Note:", style=auto, layout=Layout(height="48px", **full))
        ui.select_button = Button(description="Select cells", icon="crosshairs", button_style="info", disabled=True, layout=Layout(width="auto"),
                                  tooltip="Make this set the current selection and outline it in the image")
        ui.update_button = Button(description="Save changes", disabled=True, layout=Layout(width="auto"),
                                  tooltip="Save the name, tags and note (the cells never change)")
        ui.delete_button = Button(description="Delete", icon="trash", button_style="danger", disabled=True, layout=Layout(width="auto"),
                                  tooltip="Delete this set (asks first)")
        ui.select_button.on_click(lambda _b: self.select_set(self._picked_id()))
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
                ui.save_button,
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
                HBox([ui.select_button, ui.update_button, ui.delete_button], layout=Layout(**row)),
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
        bar = getattr(getattr(viewer, "ui_component", None), "selection_bar", None)
        if bar is not None and self.store is not None and hasattr(bar, "on_save"):
            bar.on_save(self.save_selection)
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
        ui.write_button.disabled = not has

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
        self.refresh_sets(select=meta["id"])
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

    def _picked(self) -> Optional[Dict[str, Any]]:
        set_id = self._picked_id()
        return next((entry for entry in self._sets if entry.get("id") == set_id), None)

    @staticmethod
    def _row_label(entry: Dict[str, Any]) -> str:
        tags = f" · {format_tags(entry.get('tags', []))}" if entry.get("tags") else ""
        return f"{entry.get('name', '?')} — {_cells(int(entry.get('count', 0)))}{tags}"

    def refresh_sets(self, select: Optional[str] = None) -> None:
        """Re-read the index and show the sets that match the filter."""
        ui = self.ui_component
        self._sets = self.store.list_sets() if self.store is not None else []
        needle = (ui.filter_input.value or "").strip().casefold()
        shown = [
            entry
            for entry in self._sets
            if not needle
            or needle in str(entry.get("name", "")).casefold()
            or any(needle in str(tag).casefold() for tag in entry.get("tags", []))
        ]
        keep = select or self._picked_id()
        ui.set_list.options = [(self._row_label(entry), entry["id"]) for entry in shown]
        ids = [entry["id"] for entry in shown]
        ui.set_list.value = keep if keep in ids else None
        self._on_set_picked()

    def _on_set_picked(self) -> None:
        ui = self.ui_component
        entry = self._picked()
        for button in (ui.select_button, ui.update_button, ui.delete_button):
            button.disabled = entry is None
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
        ui.details.value = (
            f"<span style='color: {_MUTED}; font-size: 12px;'>{_cells(int(entry.get('count', 0)))} in "
            f"{fovs} {'FOV' if fovs == 1 else 'FOVs'} · from {made or 'a selection'} · saved {html.escape(str(entry.get('created', '')))}"
            f" · {steps} selection {'step' if steps == 1 else 'steps'} recorded</span>"
        )

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
        self.refresh_sets(select=meta["id"])
        self._status(f"Saved the changes to <b>{html.escape(meta['name'])}</b>.", _OK)

    def ask_delete_set(self, set_id: Optional[str]) -> None:
        entry = next((e for e in self._sets if e.get("id") == set_id), None)
        if entry is None or self.store is None:
            return
        asked = self.confirm(
            f"The cell set <b>{html.escape(entry['name'])}</b> ({_cells(int(entry.get('count', 0)))}) "
            "and its recorded history will be deleted from disk. This cannot be undone.",
            lambda: self._delete_confirmed(set_id),
            title="Delete cell set?",
            confirm_label="Delete",
        )
        if not asked:
            self._status("Not deleted: there is no dialog to confirm with.", _WARN)

    def _delete_confirmed(self, set_id: str) -> None:
        entry = next((e for e in self._sets if e.get("id") == set_id), None)
        try:
            self.store.delete_set(set_id)
        except (KeyError, OSError) as exc:
            self._status(html.escape(str(exc)), _ERROR)
            return
        self.refresh_sets()
        self._status(f"Deleted <b>{html.escape(entry['name'] if entry else set_id)}</b>.", _OK)

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
        viewer = self.main_viewer
        try:
            cells = self.store.read_cells(set_id)
        except (KeyError, OSError, ValueError) as exc:
            self._status(f"Cannot read the set: {html.escape(str(exc))}", _ERROR)
            return False
        indices, missing = indices_from_cells(viewer, cells)
        if not indices:
            self._status(f"None of the {_cells(len(cells))} of <b>{html.escape(entry['name'])}</b> is in the cell table any more.", _ERROR)
            return False
        self.move_view_to(cells)
        masks = entry.get("masks") or sorted(cells["mask"].unique().tolist())
        known = set(getattr(viewer, "mask_names", ()) or ())
        mask = masks[0] if len(masks) == 1 and (not known or masks[0] in known) else None
        recipe = {"set_id": set_id, "name": entry["name"]}
        if mask:
            recipe["mask"] = mask
        self._hub.publish(indices, "set", (), recipe)
        send = getattr(viewer, "send_selection_to_image", None)
        if callable(send):
            send()
        if missing:
            self._status(f"{_cells(missing)} of <b>{html.escape(entry['name'])}</b> are no longer in the cell table.", _WARN)
        else:
            self._status(f"Selected <b>{html.escape(entry['name'])}</b>.", _OK)
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
