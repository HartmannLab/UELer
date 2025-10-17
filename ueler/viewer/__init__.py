"""Compatibility re-exports for the viewer package.

Imported via ``ueler.viewer`` to allow consumers to begin switching namespaces
without altering runtime behavior. Symbols continue to originate from the
legacy ``viewer`` package until modules are relocated.
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

from .._compat import (
	VIEWER_CORE_ALIASES as _VIEWER_CORE_ALIASES,
	VIEWER_PLUGIN_ALIASES as _VIEWER_PLUGIN_ALIASES,
	register_module_aliases as _register_module_aliases,
)

_register_module_aliases(_VIEWER_CORE_ALIASES)
_register_module_aliases(_VIEWER_PLUGIN_ALIASES)

__all__ = [
	"ImageMaskViewer",
	"create_widgets",
	"display_ui",
]


def __getattr__(name: str) -> Any:
	"""Delegate attribute lookups to the legacy viewer package."""

	legacy = _import_module("viewer")
	if hasattr(legacy, name):
		return getattr(legacy, name)
	raise AttributeError(f"module 'ueler.viewer' has no attribute '{name}'")


def __dir__() -> list[str]:
	return sorted(set(globals().keys()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover - static analysis aid
	from viewer import ImageMaskViewer, create_widgets, display_ui  # noqa: F401
