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
from .runner import load_cell_table, run_viewer, run_viewer_bia

_register_module_aliases(_UTILITY_ALIASES)

__all__ = [
	"viewer",
	"ImageMaskViewer",
	"create_widgets",
	"display_ui",
	"run_viewer",
	"run_viewer_bia",
	"load_cell_table",
	"ensure_compat_aliases",
]

__version__ = "0.4.1"


def ensure_compat_aliases() -> None:
	"""Ensure all planned alias modules are registered."""

	_ensure_aliases_loaded()


def __getattr__(name: str) -> Any:
	"""Dynamically resolve compatibility attributes.

	Must raise ``AttributeError`` (never ``ModuleNotFoundError``) when the legacy
	``viewer`` package is absent: ``import ueler.<submodule>`` resolves the parent
	via ``getattr(ueler, '<submodule>')`` when the submodule isn't bound as an
	attribute, and a leaked ``ModuleNotFoundError`` there aborts otherwise-valid
	submodule imports.
	"""

	if name == "viewer":
		return _import_module("ueler.viewer")

	try:
		legacy = _import_module("viewer")
	except ModuleNotFoundError:
		raise AttributeError(f"module 'ueler' has no attribute '{name}'")
	if hasattr(legacy, name):
		return getattr(legacy, name)
	raise AttributeError(f"module 'ueler' has no attribute '{name}'")


def __dir__() -> list[str]:
	return sorted(set(globals().keys()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover - for static type checkers only
	from ueler.viewer import ImageMaskViewer, create_widgets, display_ui  # noqa: F401
	from ueler import viewer as viewer  # noqa: F401
