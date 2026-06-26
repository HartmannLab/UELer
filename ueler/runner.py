"""Notebook-friendly entry points for launching the UELer viewer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING, Union

from ._compat import ensure_aliases_loaded

_logger = logging.getLogger(__name__)

__all__ = ["run_viewer", "load_cell_table"]

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


def _normalise_file(path: PathLike, *, argument: str) -> str:
	"""Validate that *path* points to an existing file and normalise it."""

	file_path = Path(path).expanduser()
	if not file_path.exists():
		raise FileNotFoundError(f"{argument} '{file_path}' does not exist")
	if not file_path.is_file():
		raise FileNotFoundError(f"{argument} '{file_path}' is not a file")
	return str(file_path)


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


def _refresh_viewer_state(viewer: "ImageMaskViewer") -> None:
	"""Refresh viewer controls after mutating underlying data."""
	_logger.debug("[runner] _refresh_viewer_state: update_marker_set_dropdown")
	viewer.update_marker_set_dropdown()

	_logger.debug("[runner] _refresh_viewer_state: update_controls")
	viewer.update_controls(None)

	_logger.debug("[runner] _refresh_viewer_state: on_image_change")
	try:
		viewer.on_image_change(None)
	except AttributeError:
		_logger.debug("[runner] Skipping on_image_change refresh; side plots not fully initialised.")

	_logger.debug("[runner] _refresh_viewer_state: on_image_change done")

	# In map mode, run the same activation path used by map-selector changes.
	# load_cell_table -> on_image_change mutates FOV-derived state even when the
	# stitched map is the active viewport; re-activating the current map keeps
	# map canvas and viewport state in sync before the next render.
	map_mode_active = getattr(viewer, "_map_mode_active", False)
	if isinstance(map_mode_active, bool) and map_mode_active:
		active_map_id = getattr(viewer, "_active_map_id", None)
		if not active_map_id:
			ui = getattr(viewer, "ui_component", None)
			selector = getattr(ui, "map_selector", None)
			active_map_id = getattr(selector, "value", None)
		activate = getattr(viewer, "_activate_map_mode", None)
		if active_map_id is not None and callable(activate):
			_logger.debug("[runner] _refresh_viewer_state: _activate_map_mode(%r)", active_map_id)
			try:
				activate(str(active_map_id))
			except Exception:
				_logger.debug("[runner] Failed to re-activate map mode during refresh.")
		refresh_map_controls = getattr(viewer, "_refresh_map_controls", None)
		if callable(refresh_map_controls):
			_logger.debug("[runner] _refresh_viewer_state: _refresh_map_controls")
			refresh_map_controls()
	_logger.debug("[runner] _refresh_viewer_state: update_display")
	viewer.update_display(viewer.current_downsample_factor)
	_logger.debug("[runner] _refresh_viewer_state: update_keys")
	viewer.update_keys(None)
	_logger.debug("[runner] _refresh_viewer_state: refresh_bottom_panel")
	viewer.refresh_bottom_panel()
	_logger.debug("[runner] _refresh_viewer_state: inform_plugins")
	viewer.inform_plugins('refresh_roi_table')
	_logger.debug("[runner] _refresh_viewer_state: DONE")


def load_cell_table(
	viewer: "ImageMaskViewer",
	*,
	cell_table_path: Optional[PathLike] = None,
	cell_table: Any = None,
	auto_display: bool = True,
	after_plugins: bool = True,
) -> "ImageMaskViewer":
	"""Attach a cell table to an existing viewer and re-render the UI.

	Exactly one of ``cell_table_path`` or ``cell_table`` must be provided. The
	former loads the CSV on demand; the latter is forwarded to
	:meth:`ImageMaskViewer.set_cell_table`.
	"""

	if viewer is None:
		raise ValueError("viewer must be provided")
	if (cell_table_path is None) == (cell_table is None):
		raise ValueError("Provide exactly one of cell_table_path or cell_table")

	if cell_table_path is not None:
		file_path = _normalise_file(cell_table_path, argument="cell_table_path")
		load = getattr(viewer, "load_cell_table_from_path", None)
		if not callable(load):
			raise AttributeError("viewer does not support loading a cell table from path")
		load(file_path)
	else:
		setter = getattr(viewer, "set_cell_table", None)
		if not callable(setter):
			raise AttributeError("viewer does not support setting a cell table directly")
		setter(cell_table)

	_refresh_viewer_state(viewer)

	if auto_display:
		display_fn, update_panel = _load_display_helpers()
		display_fn(viewer)
		update_panel(viewer)

	if after_plugins:
		post_loader = getattr(viewer, "after_all_plugins_loaded", None)
		if callable(post_loader):
			post_loader()

	return viewer
