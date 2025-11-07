"""Observable helper providing a lightweight pub-sub pattern."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["Observable"]


class Observable:
	"""Small utility that notifies registered observers when the value changes."""

	def __init__(self, initial_value: Any | None = None) -> None:
		self._value: Any | None = initial_value
		self._observers: list[Callable[[Any | None], None]] = []

	def add_observer(self, observer: Callable[[Any | None], None]) -> None:
		self._observers.append(observer)

	def remove_observer(self, observer: Callable[[Any | None], None]) -> None:
		self._observers.remove(observer)

	def notify_observers(self) -> None:
		for observer in self._observers:
			observer(self._value)

	@property
	def value(self) -> Any | None:
		return self._value

	@value.setter
	def value(self, new_value: Any | None) -> None:
		self._value = new_value
		self.notify_observers()
