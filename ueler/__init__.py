"""UELer package skeleton with compatibility shims.

This module keeps the current runtime behavior by delegating imports to the
legacy modules while providing a stable place to register compatibility alias
modules. The helper re-exports allow notebooks to begin using the ``ueler``
namespace without breaking existing code.
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

from ._compat import (
	LEGACY_PACKAGE_PREFIXES as _LEGACY_PACKAGE_PREFIXES,
	UTILITY_ALIASES as _UTILITY_ALIASES,
	ensure_aliases_loaded as _ensure_aliases_loaded,
	register_module_aliases as _register_module_aliases,
	register_package_prefixes as _register_package_prefixes,
)
from .runner import load_cell_table, run_viewer, run_viewer_bia

_register_module_aliases(_UTILITY_ALIASES)
# Registering the legacy ``viewer`` namespace here (rather than only in
# ``ensure_compat_aliases``) keeps ``import ueler`` sufficient for notebooks that
# still do ``from viewer.main_viewer import ImageMaskViewer``. The finder is lazy,
# so this costs nothing until such an import happens.
_register_package_prefixes(_LEGACY_PACKAGE_PREFIXES)

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

__version__ = "0.4.3"


def ensure_compat_aliases() -> None:
	"""Ensure all planned alias modules are registered."""

	_ensure_aliases_loaded()


def __getattr__(name: str) -> Any:
	"""Resolve the viewer subpackage lazily.

	Must raise ``AttributeError`` (never ``ModuleNotFoundError``) for unknown
	names: ``import ueler.<submodule>`` resolves the parent via
	``getattr(ueler, '<submodule>')`` when the submodule isn't bound as an
	attribute, and a leaked ``ModuleNotFoundError`` there aborts otherwise-valid
	submodule imports.
	"""

	if name == "viewer":
		return _import_module("ueler.viewer")

	if name in {"ImageMaskViewer", "create_widgets", "display_ui"}:
		return getattr(_import_module("ueler.viewer"), name)

	raise AttributeError(f"module 'ueler' has no attribute '{name}'")


def __dir__() -> list[str]:
	return sorted(set(globals().keys()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover - for static type checkers only
	from ueler.viewer import ImageMaskViewer, create_widgets, display_ui  # noqa: F401
	from ueler import viewer as viewer  # noqa: F401
