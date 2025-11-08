"""Loader utilities for map-based FOV descriptors (issue #3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import json
import logging


logger = logging.getLogger(__name__)


class MapDescriptorError(RuntimeError):
    """Raised when a map descriptor fails validation."""


_DISALLOWED_GEOMETRY_FIELDS = {
    "rotation",
    "transform",
    "affineTransform",
    "affine",
    "matrix",
}


@dataclass(frozen=True)
class MapFOVSpec:
    """Normalized representation of an FOV entry inside a map descriptor."""

    name: str
    slide_id: str
    center_um: Tuple[float, float]
    frame_size_px: Tuple[int, int]
    fov_size_um: Optional[float]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SlideDescriptor:
    """Group of FOVs belonging to the same slide identifier."""

    slide_id: str
    source_path: Path
    export_datetime: Optional[str]
    fovs: Tuple[MapFOVSpec, ...]


@dataclass(frozen=True)
class MapDescriptorResult:
    """Outcome of scanning one or more descriptor files."""

    slides: Mapping[str, SlideDescriptor]
    warnings: Tuple[str, ...]
    errors: Tuple[str, ...]


def _coerce_float(value: Any, *, field: str, context: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        raise MapDescriptorError(f"{context}: field '{field}' must be a number, got {value!r}.")


def _coerce_int(value: Any, *, field: str, context: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        raise MapDescriptorError(f"{context}: field '{field}' must be an integer, got {value!r}.")
    if result <= 0:
        raise MapDescriptorError(f"{context}: field '{field}' must be > 0, got {result}.")
    return result


def _validate_translation_only(entry: Mapping[str, Any], *, context: str) -> None:
    disallowed = [field for field in _DISALLOWED_GEOMETRY_FIELDS if entry.get(field) not in (None, 0, {}, [])]
    if disallowed:
        raise MapDescriptorError(
            f"{context}: translation-only mode does not accept fields {', '.join(disallowed)}."
        )
    if entry.get("centerPointPixels") not in (None, {}):
        raise MapDescriptorError(
            f"{context}: mixed coordinate units detected (centerPointPixels); only micron coordinates are supported."
        )


def _normalize_fov(entry: Mapping[str, Any], *, index: int, source: Path) -> MapFOVSpec:
    context = f"{source.name} fovs[{index}]"
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise MapDescriptorError(f"{context}: field 'name' must be a non-empty string.")

    slide_id_raw = entry.get("slideId")
    if slide_id_raw is None:
        raise MapDescriptorError(f"{context}: field 'slideId' is required.")
    slide_id = str(slide_id_raw)

    center = entry.get("centerPointMicrons")
    if not isinstance(center, Mapping):
        raise MapDescriptorError(f"{context}: field 'centerPointMicrons' must be an object with 'x'/'y'.")
    cx = _coerce_float(center.get("x"), field="centerPointMicrons.x", context=context)
    cy = _coerce_float(center.get("y"), field="centerPointMicrons.y", context=context)

    frame = entry.get("frameSizePixels")
    if not isinstance(frame, Mapping):
        raise MapDescriptorError(f"{context}: field 'frameSizePixels' must be an object with 'width'/'height'.")
    width_px = _coerce_int(frame.get("width"), field="frameSizePixels.width", context=context)
    height_px = _coerce_int(frame.get("height"), field="frameSizePixels.height", context=context)

    fov_size_raw = entry.get("fovSizeMicrons")
    fov_size_um: Optional[float]
    if fov_size_raw is None:
        fov_size_um = None
    else:
        fov_size_um = _coerce_float(fov_size_raw, field="fovSizeMicrons", context=context)
        if fov_size_um <= 0:
            raise MapDescriptorError(f"{context}: field 'fovSizeMicrons' must be > 0, got {fov_size_um}.")

    _validate_translation_only(entry, context=context)

    reserved = {
        "name",
        "slideId",
        "centerPointMicrons",
        "centerPointPixels",
        "frameSizePixels",
        "fovSizeMicrons",
    }
    metadata = {key: entry[key] for key in entry.keys() - reserved}

    return MapFOVSpec(
        name=name.strip(),
        slide_id=slide_id,
        center_um=(cx, cy),
        frame_size_px=(width_px, height_px),
        fov_size_um=fov_size_um,
        metadata=metadata,
    )


class MapDescriptorLoader:
    """Loads and validates map descriptors for the map-based tiled mode."""

    def __init__(self, *, file_pattern: str = "*.json") -> None:
        self.file_pattern = file_pattern

    def load_from_directory(self, directory: Path) -> MapDescriptorResult:
        """Scan a directory for descriptor files and return normalized metadata."""

        slides: MutableMapping[str, SlideDescriptor] = {}
        warnings: List[str] = []
        errors: List[str] = []

        if not directory.exists():
            warnings.append(f"Descriptor directory '{directory}' does not exist.")
            return MapDescriptorResult(slides={}, warnings=tuple(warnings), errors=tuple(errors))

        files = sorted(directory.rglob(self.file_pattern))
        if not files:
            warnings.append(f"No descriptor files matching '{self.file_pattern}' found under '{directory}'.")
            return MapDescriptorResult(slides={}, warnings=tuple(warnings), errors=tuple(errors))

        for path in files:
            try:
                slides.update(self._load_file(path, warnings))
            except MapDescriptorError as exc:
                message = f"{path.name}: {exc}"
                errors.append(message)
                logger.warning("Map descriptor rejected: %s", message)
            except Exception as exc:  # pragma: no cover - safety net
                message = f"{path.name}: unexpected error {exc}"  # type: ignore[str-format]
                errors.append(message)
                logger.exception("Unexpected error while reading map descriptor %s", path)

        slides = {slide_id: slides[slide_id] for slide_id in sorted(slides)}
        return MapDescriptorResult(slides=slides, warnings=tuple(warnings), errors=tuple(errors))

    def _load_file(self, path: Path, warnings: List[str]) -> Dict[str, SlideDescriptor]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, Mapping):
            raise MapDescriptorError("descriptor root must be a JSON object.")

        fovs = data.get("fovs")
        if not isinstance(fovs, list):
            raise MapDescriptorError("descriptor must define 'fovs' as a list.")

        export_datetime = data.get("exportDateTime") if isinstance(data.get("exportDateTime"), str) else None

        slide_map: Dict[str, List[MapFOVSpec]] = {}
        seen_fov_keys: Dict[str, Path] = {}

        for index, entry in enumerate(fovs):
            if not isinstance(entry, Mapping):
                raise MapDescriptorError(f"fovs[{index}] must be a JSON object.")

            spec = _normalize_fov(entry, index=index, source=path)
            slide_entries = slide_map.setdefault(spec.slide_id, [])
            if any(existing.name == spec.name for existing in slide_entries):
                duplicate_msg = (
                    f"{path.name}: duplicate FOV '{spec.name}' detected for slide '{spec.slide_id}'."
                )
                warnings.append(duplicate_msg)
                continue

            fov_key = spec.name.lower()
            if fov_key in seen_fov_keys:
                warnings.append(
                    f"{path.name}: FOV '{spec.name}' already defined in '{seen_fov_keys[fov_key].name}', skipping duplicate."
                )
                continue

            seen_fov_keys[fov_key] = path
            slide_entries.append(spec)

        result: Dict[str, SlideDescriptor] = {}
        for slide_id, entries in slide_map.items():
            ordered = tuple(sorted(entries, key=lambda item: item.name.lower()))
            result[slide_id] = SlideDescriptor(
                slide_id=slide_id,
                source_path=path,
                export_datetime=export_datetime,
                fovs=ordered,
            )

        return result
