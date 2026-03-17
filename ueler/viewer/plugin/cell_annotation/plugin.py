"""CellAnnotationPlugin – orchestrator for the Cell Annotation workflow.

Lifecycle
---------
The plugin is registered by ``ImageMaskViewer.dynamically_load_plugins``
**only** when the ``ENABLE_CELL_ANNOTATION`` environment variable is set
to a truthy value (``"1"``, ``"true"``, ``"yes"``).  When the flag is
absent the plugin is never imported and the viewer behaves as before.

Event hooks
-----------
``on_dataset_opened(base_folder)``
    Called when a dataset directory has been successfully loaded.  Creates
    the per-dataset ``.UELer`` folder tree and logs the store path.

``on_dataset_closed()``
    Called when the dataset is unloaded (future: teardown logic).

The plugin deliberately has **no UI widgets** at this stage – browser
dialogs and selection UI are tracked in later sub-issues.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from .manifest import Manifest
from .store import DatasetStore

logger = logging.getLogger(__name__)


def _flag_enabled() -> bool:
    """Return ``True`` when ``ENABLE_CELL_ANNOTATION`` is set to a truthy value."""
    val = os.environ.get("ENABLE_CELL_ANNOTATION", "").strip().lower()
    return val in {"1", "true", "yes"}


class CellAnnotationPlugin:
    """Orchestrator for the Cell Annotation workflow.

    This class intentionally does **not** extend ``PluginBase`` because it
    owns no UI accordion panel.  It is registered as a plain attribute on the
    viewer and invoked through direct method calls rather than the generic
    ``inform_plugins`` loop.

    Parameters
    ----------
    viewer:
        The ``ImageMaskViewer`` instance that owns this plugin.
    """

    #: Attribute name used when attaching the plugin to the viewer.
    REGISTRY_KEY = "cell_annotation_plugin"

    def __init__(self, viewer: object) -> None:
        self._viewer = viewer
        self._store: Optional[DatasetStore] = None
        self._manifest: Optional[Manifest] = None
        logger.debug("CellAnnotationPlugin instantiated")

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def on_dataset_opened(self, base_folder: str | Path) -> None:
        """Initialise per-dataset storage when a dataset directory is loaded.

        Creates the directory tree::

            <base_folder>/.UELer/dataset_<id>/
            <base_folder>/.UELer/dataset_<id>/checkpoints/
            <base_folder>/.UELer/dataset_<id>/thumbnails/
            <base_folder>/.UELer/dataset_<id>/selections/

        and loads the manifest if one already exists.

        Parameters
        ----------
        base_folder:
            Path to the dataset root directory.
        """
        self._store = DatasetStore(base_folder)
        self._store.ensure_dirs()

        self._manifest = Manifest(self._store.store_path)
        self._manifest.load()

        logger.info(
            "[CellAnnotation] dataset store ready: %s",
            self._store.store_path,
        )

    def on_dataset_closed(self) -> None:
        """Release per-dataset resources when the dataset is unloaded."""
        logger.debug("[CellAnnotation] dataset closed; releasing store reference")
        self._store = None
        self._manifest = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def store(self) -> Optional[DatasetStore]:
        """The active :class:`~.store.DatasetStore`, or ``None`` before open."""
        return self._store

    @property
    def manifest(self) -> Optional[Manifest]:
        """The active :class:`~.manifest.Manifest`, or ``None`` before open."""
        return self._manifest
