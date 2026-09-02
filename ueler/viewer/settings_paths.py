"""Resolve the location of the ``.UELer`` settings/cache folder (issue #137).

By default ``.UELer`` lives inside ``base_folder``. When a ``settings_path``
override is given, it moves to ``<settings_path>/<base_folder name>/.UELer``
instead, so datasets on read-only or pipeline-managed storage can keep their
UELer state elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

__all__ = ["resolve_settings_root", "resolve_settings_folder", "viewer_settings_folder"]


def resolve_settings_root(base_folder: PathLike, settings_path: Optional[PathLike] = None) -> Path:
    """Return the directory that should contain ``.UELer`` for this dataset."""

    base_path = Path(base_folder).expanduser()
    if settings_path is None:
        return base_path
    return Path(settings_path).expanduser() / base_path.name


def resolve_settings_folder(base_folder: PathLike, settings_path: Optional[PathLike] = None) -> Path:
    """Return the ``.UELer`` folder itself, honoring an optional ``settings_path`` override."""

    return resolve_settings_root(base_folder, settings_path) / ".UELer"


def viewer_settings_folder(viewer) -> Optional[Path]:
    """Return ``viewer.settings_folder``, falling back to deriving it from ``viewer.base_folder``.

    The fallback keeps lightweight test doubles that only set ``base_folder``
    (e.g. ``_ViewerStub`` in ``tests/test_export_fovs_mask_customization.py``)
    working unchanged.
    """

    settings_folder = getattr(viewer, "settings_folder", None)
    if settings_folder is not None:
        return Path(settings_folder)
    base_folder = getattr(viewer, "base_folder", None)
    if not base_folder:
        return None
    return resolve_settings_folder(base_folder)
