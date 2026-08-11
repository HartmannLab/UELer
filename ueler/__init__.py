"""UELer: the public ``ueler`` namespace.

Importing this package must stay free of global side effects: it registers no
``sys.meta_path`` finders and claims no top-level module names, so ``import
ueler`` cannot change how any other import in the session resolves. The heavy UI
modules are pulled in lazily via ``__getattr__``.

The pre-0.2 compatibility layer that aliased the legacy top-level ``viewer``,
``constants``, ``data_loader`` and ``image_utils`` names onto their packaged
counterparts has been removed. Notebooks must import from ``ueler.*``.
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

from .runner import load_cell_table, run_viewer, run_viewer_bia

__all__ = [
	"viewer",
	"ImageMaskViewer",
	"create_widgets",
	"display_ui",
	"run_viewer",
	"run_viewer_bia",
	"load_cell_table",
]

__version__ = "0.5.0a1"


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
