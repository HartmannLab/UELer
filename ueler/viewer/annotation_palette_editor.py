"""Palette editor widget exposed via the packaged viewer namespace."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

from ipywidgets import (
	Button,
	ColorPicker,
	GridBox,
	HTML,
	HBox,
	Layout,
	Text,
	VBox,
)

from ueler.viewer.color_palettes import DEFAULT_COLOR, normalize_hex_color

__all__ = ["AnnotationPaletteEditor"]


class AnnotationPaletteEditor(VBox):
	"""Widget allowing users to assign colours and labels to annotation classes."""

	def __init__(
		self,
		apply_callback: Callable[[str, Dict[str, str], Dict[str, str]], None],
		close_callback: Optional[Callable[[], None]] = None,
	) -> None:
		super().__init__(layout=Layout(width="100%", gap="8px"))
		self._apply_callback = apply_callback
		self._close_callback = close_callback
		self._current_annotation: Optional[str] = None
		self._color_pickers: Dict[str, ColorPicker] = {}
		self._label_inputs: Dict[str, Text] = {}

		self._header = HTML(value="<b>Annotation palette editor</b>")
		self._description = HTML(value="")
		self._grid = GridBox(
			layout=Layout(grid_template_columns="repeat(3, minmax(120px, 1fr))", gap="6px")
		)

		self._apply_button = Button(description="Apply", button_style="success", icon="check")
		self._cancel_button = Button(description="Cancel", button_style="", icon="times")
		self._apply_button.on_click(self._on_apply_clicked)
		self._cancel_button.on_click(self._on_cancel_clicked)

		self.children = [
			self._header,
			self._description,
			self._grid,
			HBox([self._apply_button, self._cancel_button], layout=Layout(gap="6px")),
		]
		self.layout.display = "none"

	def load(
		self,
		annotation_name: str,
		class_ids: Iterable[int],
		palette: Dict[str, str],
		labels: Optional[Dict[str, str]] = None,
	) -> None:
		self._current_annotation = annotation_name
		self._grid.children = ()
		self._color_pickers.clear()
		self._label_inputs.clear()

		sorted_ids = sorted({int(class_id) for class_id in class_ids if class_id is not None})
		rows = []
		for class_id in sorted_ids:
			key = str(class_id)
			label_value = (labels or {}).get(key, key)
			color_value = palette.get(key, DEFAULT_COLOR)

			id_badge = HTML(value=f"<span style='font-weight:bold'>ID {class_id}</span>")
			label_input = Text(value=label_value, placeholder="Class label")
			color_picker = ColorPicker(layout=Layout(width="110px"), value=color_value)

			self._color_pickers[key] = color_picker
			self._label_inputs[key] = label_input

			rows.extend([id_badge, label_input, color_picker])

		if not rows:
			rows = [HTML(value="No annotation classes detected."), HTML(value=""), HTML(value="")]

		self._grid.children = tuple(rows)
		self._description.value = f"Editing palette for <b>{annotation_name}</b>"
		self.layout.display = ""

	def hide(self) -> None:
		self.layout.display = "none"
		self._current_annotation = None

	def _on_apply_clicked(self, _button) -> None:
		if not self._current_annotation:
			return
		palette_updates: Dict[str, str] = {}
		label_updates: Dict[str, str] = {}
		for key, picker in self._color_pickers.items():
			palette_updates[key] = normalize_hex_color(picker.value) or DEFAULT_COLOR
		for key, text_input in self._label_inputs.items():
			label_updates[key] = text_input.value.strip() or key

		self._apply_callback(self._current_annotation, palette_updates, label_updates)
		self.hide()

	def _on_cancel_clicked(self, _button) -> None:
		self.hide()
		if self._close_callback:
			self._close_callback()
