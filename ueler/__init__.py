"""UELer package skeleton.

This module provides a forward-looking package namespace while continuing to
expose the legacy `viewer` entry points. Runtime behavior remains identical to
the existing top-level imports; symbols are lazily delegated to the current
`viewer` package until the migration is complete.
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

__all__ = [
	"viewer",
	"ImageMaskViewer",
	"create_widgets",
	"display_ui",
]

__version__ = "0.2.0a0"


def __getattr__(name: str) -> Any:
	"""Dynamically resolve compatibility attributes."""

	if name == "viewer":
		return _import_module("ueler.viewer")

	legacy = _import_module("viewer")
	if hasattr(legacy, name):
		return getattr(legacy, name)
	raise AttributeError(f"module 'ueler' has no attribute '{name}'")


def __dir__() -> list[str]:
	return sorted(set(globals().keys()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover - for static type checkers only
	from viewer import ImageMaskViewer, create_widgets, display_ui  # noqa: F401
	from . import viewer as viewer  # noqa: F401
