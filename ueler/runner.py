"""Notebook-friendly entry points for launching the UELer viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING, Union

from ._compat import ensure_aliases_loaded

__all__ = ["run_viewer"]

PathLike = Union[str, Path]

if TYPE_CHECKING:
	from .viewer.main_viewer import ImageMaskViewer
else:  # pragma: no cover - runtime placeholder to avoid heavy imports
	ImageMaskViewer = Any  # type: ignore[assignment]



class _ViewerFactory(Protocol):  # pragma: no cover - structural typing helper
	def __call__(
		self,
		base_folder: str,
		*,
		masks_folder: Optional[str] = ...,
		annotations_folder: Optional[str] = ...,
		**kwargs,
	) -> "ImageMaskViewer":
		...


def _normalise_directory(path: PathLike, *, argument: str) -> str:
	"""Expand and validate a required directory argument."""

	directory = Path(path).expanduser()
	if not directory.exists():
		raise FileNotFoundError(f"{argument} '{directory}' does not exist")
	if not directory.is_dir():
		raise NotADirectoryError(f"{argument} '{directory}' is not a directory")
	return str(directory)


def _normalise_optional_directory(path: Optional[PathLike], *, argument: str) -> Optional[str]:
	"""Expand optional directory arguments if they are provided."""

	if path is None:
		return None
	directory = Path(path).expanduser()
	if not directory.exists():
		raise FileNotFoundError(f"{argument} '{directory}' does not exist")
	if not directory.is_dir():
		raise NotADirectoryError(f"{argument} '{directory}' is not a directory")
	return str(directory)


def _load_viewer_factory() -> _ViewerFactory:
	from .viewer.main_viewer import ImageMaskViewer as _ImageMaskViewer

	return _ImageMaskViewer


def _load_display_helpers() -> tuple[Callable[["ImageMaskViewer"], None], Callable[["ImageMaskViewer"], None]]:
	from .viewer.ui_components import display_ui as _display_ui, update_wide_plugin_panel as _update_panel

	return _display_ui, _update_panel


def run_viewer(
	base_folder: PathLike,
	*,
	masks_folder: Optional[PathLike] = None,
	annotations_folder: Optional[PathLike] = None,
	auto_display: bool = True,
	ensure_aliases: bool = True,
	after_plugins: bool = True,
	viewer_factory: Optional[_ViewerFactory] = None,
	display_callback: Optional[Callable[["ImageMaskViewer"], None]] = None,
	**viewer_kwargs,
) -> "ImageMaskViewer":
	"""Initialise the viewer with sensible defaults for notebook use.

	Parameters
	----------
	base_folder:
		Root directory containing the dataset (must exist).
	masks_folder, annotations_folder:
		Optional directories providing masks and annotation rasters. When omitted,
		the viewer's default discovery rules apply.
	auto_display:
		If ``True`` (default) render the widget tree immediately via
		``display_ui``.
	ensure_aliases:
		When ``True`` (default) register the compatibility shims before
		instantiating the viewer so legacy modules stay importable.
	after_plugins:
		Call ``viewer.after_all_plugins_loaded()`` after displaying the UI when
		available (defaults to ``True``).
	viewer_factory:
		Optional factory to construct the viewer instance. Defaults to
		:class:`~ueler.viewer.main_viewer.ImageMaskViewer` and is primarily
		intended for testing.
	display_callback:
		Override the display helper (defaults to
		:func:`~ueler.viewer.ui_components.display_ui`).
	**viewer_kwargs:
		Additional keyword arguments forwarded to the viewer factory.

	Returns
	-------
	ueler.viewer.main_viewer.ImageMaskViewer
		The constructed viewer instance for continued interaction within the
		notebook session.
	"""

	if ensure_aliases:
		ensure_aliases_loaded()

	base_dir = _normalise_directory(base_folder, argument="base_folder")
	masks_dir = _normalise_optional_directory(masks_folder, argument="masks_folder")
	annotations_dir = _normalise_optional_directory(
		annotations_folder, argument="annotations_folder"
	)

	if viewer_factory is None:
		factory: _ViewerFactory = _load_viewer_factory()
	else:
		factory = viewer_factory

	viewer: "ImageMaskViewer" = factory(
		base_dir,
		masks_folder=masks_dir,
		annotations_folder=annotations_dir,
		**viewer_kwargs,
	)

	if auto_display:
		default_display, update_panel = _load_display_helpers()
		display_fn = display_callback or default_display
		display_fn(viewer)
		update_panel(viewer)

	if after_plugins:
		post_loader = getattr(viewer, "after_all_plugins_loaded", None)
		if callable(post_loader):
			post_loader()

	return viewer
