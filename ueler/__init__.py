"""UELer package skeleton with compatibility shims.

This module keeps the current runtime behavior by delegating imports to the
legacy modules while providing a stable place to register compatibility alias
modules. The helper re-exports allow notebooks to begin using the ``ueler``
namespace without breaking existing code.
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

from ._compat import (
	UTILITY_ALIASES as _UTILITY_ALIASES,
	ensure_aliases_loaded as _ensure_aliases_loaded,
	register_module_aliases as _register_module_aliases,
)

_register_module_aliases(_UTILITY_ALIASES)

__all__ = [
	"viewer",
	"ImageMaskViewer",
	"create_widgets",
	"display_ui",
	"ensure_compat_aliases",
]

__version__ = "0.2.0a0"


def ensure_compat_aliases() -> None:
	"""Ensure all planned alias modules are registered."""

	_ensure_aliases_loaded()


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
