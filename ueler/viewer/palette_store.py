"""Shared helpers for persisting color palette definitions."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Mapping, Optional

__all__ = [
    "PaletteStoreError",
    "slugify_name",
    "write_palette_file",
    "read_palette_file",
    "load_registry",
    "save_registry",
    "resolve_palette_path",
]


class PaletteStoreError(Exception):
    """Raised when saving or loading palette data fails."""


_slug_pattern = re.compile(r"[^a-zA-Z0-9-_]+")
_multiple_dash_pattern = re.compile(r"-+")


def slugify_name(name: str, default_slug: str = "palette") -> str:
    """Convert an arbitrary name into a filesystem-friendly slug."""
    if not isinstance(name, str):
        return default_slug
    slug = _slug_pattern.sub("-", name.strip().lower())
    slug = _multiple_dash_pattern.sub("-", slug).strip("-")
    return slug or default_slug


def write_palette_file(path: Path, payload: Mapping[str, object]) -> Path:
    """Write a palette payload to disk with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    return path


def read_palette_file(path: Path) -> Dict[str, object]:
    """Read a palette payload from disk."""
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):  # pragma: no cover - defensive guard
        raise PaletteStoreError(f"Palette file '{path}' does not contain a JSON object.")
    return data


def _registry_path(folder: Path, filename: str) -> Path:
    return folder / filename


def load_registry(folder: Path, filename: str) -> Dict[str, Dict[str, str]]:
    """Load palette registry metadata for the given folder."""
    index_path = _registry_path(folder, filename)
    if not index_path.exists():
        return {}
    with index_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):  # pragma: no cover - defensive guard
        raise PaletteStoreError(f"Registry file '{index_path}' is invalid.")
    return {str(name): record for name, record in data.items() if isinstance(record, dict)}


def save_registry(folder: Path, filename: str, records: Mapping[str, Mapping[str, str]]) -> Path:
    """Persist palette registry metadata to disk."""
    folder.mkdir(parents=True, exist_ok=True)
    index_path = _registry_path(folder, filename)
    with index_path.open("w", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2, ensure_ascii=False)
    return index_path


def resolve_palette_path(
    folder: Path,
    name: str,
    file_suffix: str,
    *,
    default_slug: str = "palette",
) -> Path:
    """Return the full path for a palette payload using the provided naming rules."""
    slug = slugify_name(name, default_slug=default_slug)
    return (folder / f"{slug}{file_suffix}").resolve()
