"""Status bar helper decorators for the packaged viewer namespace."""
from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

__all__ = ["update_status_bar"]


def _resolve_viewer(self_obj: Any) -> Optional[Any]:
	if hasattr(self_obj, "main_viewer"):
		return getattr(self_obj, "main_viewer", None)
	return self_obj


def _set_status_bar(viewer: Optional[Any], state: str, *, fmt: str, width: int, height: int) -> None:
	if viewer is None:
		return
	try:
		viewer.ui_component.status_bar.value = viewer._status_image[state]
		viewer.ui_component.status_bar.format = fmt
		viewer.ui_component.status_bar.width = width
		viewer.ui_component.status_bar.height = height
	except Exception as error:  # pragma: no cover - debugging aid only
		if getattr(viewer, "_debug", False):
			print(f"Error updating status bar: {error}")


def update_status_bar(func: F) -> F:
	"""Wrap a method so the viewer status bar reflects processing state."""

	def wrapper(*args: Any, **kwargs: Any):
		viewer = _resolve_viewer(args[0])
		_set_status_bar(viewer, "processing", fmt="gif", width=225, height=30)
		try:
			return func(*args, **kwargs)
		except AttributeError as error:
			print(f"Error: {error}")
		finally:
			_set_status_bar(viewer, "ready", fmt="png", width=30, height=30)

	return wrapper  # type: ignore[return-value]
