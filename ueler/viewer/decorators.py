"""Status bar helper decorators for the packaged viewer namespace."""
from __future__ import annotations

import contextlib
import functools
import logging
from typing import Any, Callable, Optional, TypeVar

_logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

__all__ = ["set_status_ready", "update_status_bar"]


def _resolve_viewer(self_obj: Any) -> Optional[Any]:
	if hasattr(self_obj, "main_viewer"):
		return getattr(self_obj, "main_viewer", None)
	return self_obj


def _set_status_bar(viewer: Optional[Any], state: str, *, fmt: str, width: int, height: int) -> None:
	"""Switch the status image to *state* in one widget update.

	The four traits go to the browser as a single message (``hold_sync``), size
	and format before the image. Sent one by one, the browser applied them as
	four separate updates: it briefly drew the ready checkmark at the spinner's
	225 px width, and a front end that dropped or reordered one of them was left
	showing a state the kernel had already left.
	"""
	if viewer is None:
		return
	try:
		status_bar = viewer.ui_component.status_bar
		hold_sync = getattr(status_bar, "hold_sync", None)
		with (hold_sync() if callable(hold_sync) else contextlib.nullcontext()):
			status_bar.width = width
			status_bar.height = height
			status_bar.format = fmt
			status_bar.value = viewer._status_image[state]
	except Exception as error:  # pragma: no cover - debugging aid only
		_logger.debug("Error updating status bar: %s", error)


def set_status_ready(viewer: Optional[Any]) -> None:
	"""Show the ready checkmark (used once the viewer's first image is in place)."""
	_set_status_bar(viewer, "ready", fmt="png", width=30, height=30)


def update_status_bar(func: F) -> F:
	"""Wrap a method so the viewer status bar reflects processing state."""

	@functools.wraps(func)
	def wrapper(*args: Any, **kwargs: Any):
		viewer = _resolve_viewer(args[0])
		_set_status_bar(viewer, "processing", fmt="gif", width=225, height=30)
		try:
			return func(*args, **kwargs)
		finally:
			set_status_ready(viewer)

	return wrapper  # type: ignore[return-value]
