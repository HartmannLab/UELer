"""Atomic checkpoint persistence for the heatmap annotation workflow.

Checkpoints are `.h5ad` files stored under::

    <dataset_root>/.UELer/dataset_<sha1>/checkpoints/

A ``manifest.json`` in the same directory provides a fast index so the
``CellAnnotationPlugin`` can render the checkpoint tree without reading every
`.h5ad` file.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import ueler

__all__ = ["CheckpointStore", "empty_manifest"]


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dataset_hash(root: Path) -> str:
    """Return a short stable identifier derived from the resolved dataset root."""
    canonical = str(root.resolve())
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def empty_manifest(dataset_id: str) -> dict:
    """Return a new empty manifest dict."""
    return {
        "dataset_id": dataset_id,
        "updated_at": _now_utc(),
        "checkpoints": [],
    }


class CheckpointStore:
    """Read/write heatmap checkpoints atomically under ``<root>/.UELer/dataset_*/``."""

    def __init__(self, dataset_root: "str | Path") -> None:
        root = Path(dataset_root).expanduser().resolve()
        dataset_id = f"dataset_{_dataset_hash(root)}"
        self._dataset_dir = root / ".UELer" / dataset_id
        self._checkpoints_dir = self._dataset_dir / "checkpoints"
        self._manifest_path = self._dataset_dir / "manifest.json"
        self._dataset_id = dataset_id
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def load_manifest(self) -> dict:
        """Load and return the manifest dict; returns an empty manifest if missing or corrupt."""
        if not self._manifest_path.exists():
            return empty_manifest(self._dataset_id)
        try:
            with self._manifest_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict) or "checkpoints" not in data:
                return empty_manifest(self._dataset_id)
            return data
        except Exception:
            return empty_manifest(self._dataset_id)

    def _save_manifest(self, manifest: dict) -> None:
        """Write the manifest atomically."""
        manifest["updated_at"] = _now_utc()
        partial = self._manifest_path.with_suffix(".json.partial")
        with partial.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(partial, self._manifest_path)

    # ------------------------------------------------------------------
    # Checkpoint CRUD
    # ------------------------------------------------------------------

    def write_checkpoint(
        self,
        adata: "Any",
        *,
        parent_id: Optional[str] = None,
        op: str = "initial",
        step_id: str = "",
        description: str = "",
    ) -> str:
        """Persist *adata* as a new checkpoint and return the checkpoint ID.

        The `.h5ad` file is written atomically (``*.partial`` → fsync → rename).
        The manifest is updated after a successful write.
        """
        import anndata

        checkpoint_id = str(uuid.uuid4())
        safe_step = step_id.replace("/", "_").replace(" ", "_") or "step"
        filename = f"{checkpoint_id}__{safe_step}.h5ad"
        target = self._checkpoints_dir / filename
        partial = target.with_suffix(".h5ad.partial")

        # Stamp checkpoint metadata into uns.  HDF5 cannot store None values so
        # parent_id is serialised as an empty string when absent.
        adata.uns["checkpoint"] = {
            "id": checkpoint_id,
            "parent_id": parent_id or "",
            "op": op,
            "step_id": step_id,
            "description": description,
            "created_at": _now_utc(),
            "producer": {"package": "ueler", "version": ueler.__version__},
        }

        adata.write_h5ad(partial, compression="gzip")
        # fsync the directory so the rename is durable
        with partial.open("rb"):
            pass
        os.replace(partial, target)

        # Build manifest entry (lightweight metadata only — no matrix data)
        n_clusters = adata.n_obs
        n_markers = adata.n_vars
        entry: Dict = {
            "id": checkpoint_id,
            "parent_id": parent_id,
            "op": op,
            "step_id": step_id,
            "description": description,
            "path": f"checkpoints/{filename}",
            "n_clusters": n_clusters,
            "n_markers": n_markers,
            "created_at": adata.uns["checkpoint"]["created_at"],
        }

        manifest = self.load_manifest()
        manifest["checkpoints"].append(entry)
        self._save_manifest(manifest)

        return checkpoint_id

    def read_checkpoint(self, checkpoint_id: str) -> "Any":
        """Load and return the AnnData for *checkpoint_id*.

        Raises ``FileNotFoundError`` if the checkpoint is not in the manifest.
        """
        import anndata

        manifest = self.load_manifest()
        for entry in manifest.get("checkpoints", []):
            if entry.get("id") == checkpoint_id:
                path = self._dataset_dir / entry["path"]
                if not path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint file missing: {path}"
                    )
                return anndata.read_h5ad(path)

        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_id}' not found in manifest."
        )

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Remove the checkpoint file and its manifest entry."""
        manifest = self.load_manifest()
        new_entries = []
        deleted_path: Optional[Path] = None
        for entry in manifest.get("checkpoints", []):
            if entry.get("id") == checkpoint_id:
                deleted_path = self._dataset_dir / entry["path"]
            else:
                new_entries.append(entry)

        manifest["checkpoints"] = new_entries
        self._save_manifest(manifest)

        if deleted_path is not None and deleted_path.exists():
            try:
                deleted_path.unlink()
            except OSError:
                pass

    def list_checkpoints(self) -> List[dict]:
        """Return the list of checkpoint metadata entries from the manifest."""
        return self.load_manifest().get("checkpoints", [])
