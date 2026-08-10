"""The viewer package.

This is the canonical home of the viewer modules. The top-level entry points are
re-exported lazily so that ``from ueler.viewer import ImageMaskViewer`` works
without importing the (heavy) UI modules when only a submodule is needed.
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

__all__ = [
	"ImageMaskViewer",
	"create_widgets",
	"display_ui",
]

_LAZY_EXPORTS = {
	"ImageMaskViewer": "ueler.viewer.main_viewer",
	"create_widgets": "ueler.viewer.ui_components",
	"display_ui": "ueler.viewer.ui_components",
}


def __getattr__(name: str) -> Any:
	"""Resolve the package's public entry points on first access.

	Must raise ``AttributeError`` (never ``ModuleNotFoundError``) for unknown
	names: ``from ueler.viewer import <submodule>`` probes the package with
	``hasattr`` before importing the submodule, and ``hasattr`` only swallows
	``AttributeError`` — a leaked ``ModuleNotFoundError`` aborts the whole
	``from`` import even though the submodule exists.
	"""

	module_name = _LAZY_EXPORTS.get(name)
	if module_name is None:
		raise AttributeError(f"module 'ueler.viewer' has no attribute '{name}'")
	return getattr(_import_module(module_name), name)


def __dir__() -> list[str]:
	return sorted(set(globals().keys()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover - static analysis aid
	from ueler.viewer import ImageMaskViewer, create_widgets, display_ui  # noqa: F401
