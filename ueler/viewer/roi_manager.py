"""ROI manager utilities exposed via the packaged viewer namespace."""
from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .observable import Observable

__all__ = [
    "ROI_COLUMNS",
    "ROI_KIND_SHAPE",
    "ROI_KIND_VIEW",
    "ROIManager",
    "build_shape_fields",
    "format_roi_label",
    "geometry_bounds",
    "is_shape_record",
    "parse_geometry",
    "polyline_length",
    "serialize_geometry",
    "shape_display_kind",
]

#: ``roi_kind`` value for the viewport bookmarks the manager has always stored.
#: An empty ``roi_kind`` means the same thing, so CSVs written before shape ROIs
#: existed keep their meaning without migration.
ROI_KIND_VIEW = "view"
#: ``roi_kind`` value for a drawn shape.  Open polylines and closed polygons
#: share this kind and are told apart by ``geometry["closed"]`` — one kind keeps
#: filtering to a single predicate while the UI still names them separately.
ROI_KIND_SHAPE = "line"

#: A straight horizontal or vertical line has a zero-extent bounding box, and
#: the thumbnail renderers reject ``x_max <= x_min``.  Every shape bounding box
#: is grown to at least this many pixels on each axis.
SHAPE_BBOX_MIN_EXTENT = 8.0
#: Extra breathing room around a shape so it is not flush against the tile edge.
SHAPE_BBOX_PADDING = 4.0


def format_roi_label(record: dict) -> str:
    """Return a consistent display label for an ROI record."""
    fov = str(record.get("fov") or "")
    map_id = str(record.get("map_id") or "")
    marker = record.get("marker_set") or "—"
    tags = record.get("tags") or ""
    tag_display = f" [{tags}]" if tags else ""
    roi_id = str(record.get("roi_id") or "")
    location = fov if fov else (f"[MAP:{map_id}]" if map_id else "—")
    kind = shape_display_kind(record)
    if kind:
        location = f"{location} · {kind}"
    name = str(record.get("name") or "").strip()
    suffix = name if name else roi_id[:8]
    return f"{location} · {marker}{tag_display} · {suffix}"


# ----------------------------------------------------------------------
# Shape geometry helpers
# ----------------------------------------------------------------------
def serialize_geometry(points: Sequence[Sequence[float]], closed: bool = False) -> str:
    """Return the JSON payload stored in the ``geometry`` column.

    Coordinates follow the same convention as the bounding-box columns: pixels
    local to ``fov`` when the record names one, stitched-canvas pixels when it
    names a ``map_id``.
    """
    cleaned = _clean_points(points)
    payload = {
        "type": "polygon" if closed else "polyline",
        "closed": bool(closed),
        "points": [[round(x, 2), round(y, 2)] for x, y in cleaned],
    }
    return json.dumps(payload, separators=(",", ":"))


def parse_geometry(payload: object) -> Optional[Dict[str, object]]:
    """Parse a ``geometry`` payload into ``{"type", "closed", "points"}``.

    Returns ``None`` for anything that does not yield at least one usable
    vertex — missing values, ``NaN``, empty strings, malformed JSON, or a point
    list with no finite coordinate pairs left after cleaning.
    """
    if payload is None:
        return None

    data: object = payload
    if isinstance(payload, str):
        text = payload.strip()
        if not text or text.lower() == "nan":
            return None
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return None
    elif isinstance(payload, float) and math.isnan(payload):
        return None

    if not isinstance(data, dict):
        return None

    points = _clean_points(data.get("points"))
    if not points:
        return None

    closed = bool(data.get("closed", str(data.get("type") or "").lower() == "polygon"))
    return {
        "type": "polygon" if closed else "polyline",
        "closed": closed,
        "points": points,
    }


def geometry_bounds(
    points: Sequence[Sequence[float]],
    *,
    min_extent: float = SHAPE_BBOX_MIN_EXTENT,
    padding: float = SHAPE_BBOX_PADDING,
    limit: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, float]]:
    """Return the padded bounding box of ``points`` as ROI table fields.

    The box is what lets a shape ROI travel through every existing consumer
    unchanged — ``center_on_roi``, the thumbnail renderers and the batch-export
    job builder all read the box and never need to know about the vertices.

    ``limit`` optionally clamps the box to a ``(width, height)`` canvas; the
    lower bounds are always clamped at zero, since a negative pixel coordinate
    is meaningless in both FOV-local and stitched-canvas space.
    """
    cleaned = _clean_points(points)
    if not cleaned:
        return None

    xs = [x for x, _ in cleaned]
    ys = [y for _, y in cleaned]
    x_min, x_max = _pad_span(min(xs), max(xs), padding, min_extent, limit[0] if limit else None)
    y_min, y_max = _pad_span(min(ys), max(ys), padding, min_extent, limit[1] if limit else None)

    return {
        "x": (x_min + x_max) / 2.0,
        "y": (y_min + y_max) / 2.0,
        "width": x_max - x_min,
        "height": y_max - y_min,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }


def polyline_length(points: Sequence[Sequence[float]], closed: bool = False) -> float:
    """Return the path length of ``points`` in pixels (perimeter when closed)."""
    cleaned = _clean_points(points)
    if len(cleaned) < 2:
        return 0.0

    segments = list(zip(cleaned, cleaned[1:]))
    if closed:
        segments.append((cleaned[-1], cleaned[0]))
    return float(sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in segments))


def is_shape_record(record: object) -> bool:
    """Return ``True`` when ``record`` is a drawn shape rather than a viewport.

    A record whose ``roi_kind`` is unset but which carries parseable geometry is
    still treated as a shape, so a hand-edited CSV behaves sensibly.
    """
    if not hasattr(record, "get"):
        return False
    kind = str(record.get("roi_kind") or "").strip().lower()  # type: ignore[union-attr]
    if kind == ROI_KIND_SHAPE:
        return True
    if kind:
        return False
    return parse_geometry(record.get("geometry")) is not None  # type: ignore[union-attr]


def shape_display_kind(record: object) -> str:
    """Return ``"polygon"``/``"line"`` for a shape record, ``""`` otherwise."""
    if not is_shape_record(record):
        return ""
    geometry = parse_geometry(record.get("geometry"))  # type: ignore[union-attr]
    if geometry is None:
        return ROI_KIND_SHAPE
    return "polygon" if geometry.get("closed") else "line"


def build_shape_fields(
    points: Sequence[Sequence[float]],
    closed: bool = False,
    *,
    limit: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, object]]:
    """Return the full set of ROI columns describing a drawn shape.

    ``None`` when ``points`` holds nothing usable, so callers can reject an
    empty gesture with a single check.
    """
    bounds = geometry_bounds(points, limit=limit)
    if bounds is None:
        return None

    fields: Dict[str, object] = {
        "roi_kind": ROI_KIND_SHAPE,
        "geometry": serialize_geometry(points, closed),
    }
    fields.update(bounds)
    return fields


def _clean_points(points: object) -> List[Tuple[float, float]]:
    """Return ``points`` as a list of finite ``(x, y)`` float pairs."""
    if not isinstance(points, (list, tuple)):
        return []

    cleaned: List[Tuple[float, float]] = []
    for entry in points:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            x = float(entry[0])
            y = float(entry[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            cleaned.append((x, y))
    return cleaned


def _pad_span(
    low: float,
    high: float,
    padding: float,
    min_extent: float,
    limit: Optional[float],
) -> Tuple[float, float]:
    """Pad ``[low, high]`` to at least ``min_extent``, clamped into the canvas."""
    low -= padding
    high += padding
    if low < 0.0:
        high += -low
        low = 0.0
    if limit is not None and high > limit:
        low = max(0.0, low - (high - limit))
        high = limit

    deficit = min_extent - (high - low)
    if deficit > 0:
        low -= deficit / 2.0
        high += deficit / 2.0
        if low < 0.0:
            high += -low
            low = 0.0
        if limit is not None and high > limit:
            low = max(0.0, low - (high - limit))
            high = limit
    return low, high


ROI_COLUMNS = [
    "roi_id",
    "name",
    "fov",
    "map_id",
    "x",
    "y",
    "width",
    "height",
    "zoom",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "marker_set",
    "tags",
    "annotation_palette",
    "mask_color_set",
    "mask_visibility",
    "mask_painter_state",
    "comment",
    "roi_kind",
    "geometry",
    "created_at",
    "updated_at",
]

#: Columns holding text rather than numbers.  Missing ones are back-filled with
#: ``""`` instead of ``0.0``, which is what keeps ``geometry`` readable when an
#: older CSV is loaded.
_STRING_COLUMNS = (
    "name",
    "fov",
    "marker_set",
    "tags",
    "annotation_palette",
    "mask_color_set",
    "mask_visibility",
    "mask_painter_state",
    "comment",
    "map_id",
    "roi_kind",
    "geometry",
)


def _ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in ROI_COLUMNS if col not in df.columns]
    for col in missing:
        df[col] = "" if col in _STRING_COLUMNS else 0.0
    df = df[ROI_COLUMNS]
    for col in _STRING_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("")
                .apply(lambda value: "" if str(value).strip().lower() == "nan" else str(value).strip())
            )
    return df


class ROIManager:
    """Manage Region-of-Interest records with persistence."""

    def __init__(self, base_folder: str):
        self.base_folder = base_folder
        self.storage_dir = os.path.join(base_folder, ".UELer")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.storage_path = os.path.join(self.storage_dir, "roi_manager.csv")

        self._table = Observable(self._load_initial_table())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_initial_table(self) -> pd.DataFrame:
        if os.path.exists(self.storage_path):
            try:
                df = pd.read_csv(self.storage_path)
            except Exception:
                df = pd.DataFrame(columns=ROI_COLUMNS)
        else:
            df = pd.DataFrame(columns=ROI_COLUMNS)

        return _ensure_dataframe(df)

    def _set_table(self, df: pd.DataFrame, persist: bool = True) -> None:
        df = _ensure_dataframe(df.copy())
        if persist:
            df.to_csv(self.storage_path, index=False)
        self._table.value = df

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalise_tags(tags: Optional[Iterable[str]]) -> str:
        if tags is None:
            return ""
        if isinstance(tags, str):
            return tags
        clean = [str(tag).strip() for tag in tags if str(tag).strip()]
        return ",".join(dict.fromkeys(clean))

    def _default_record(self) -> Dict[str, object]:
        ts = self._timestamp()
        return {
            "roi_id": str(uuid.uuid4()),
            "fov": "",
            "map_id": "",
            "x": 0.0,
            "y": 0.0,
            "width": 0.0,
            "height": 0.0,
            "zoom": 0.0,
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "marker_set": "",
            "tags": "",
            "annotation_palette": "",
            "mask_color_set": "",
            "mask_visibility": "",
            "mask_painter_state": "",
            "comment": "",
            "roi_kind": "",
            "geometry": "",
            "created_at": ts,
            "updated_at": ts,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def table(self) -> pd.DataFrame:
        return self._table.value.copy()

    @property
    def observable(self) -> Observable:
        return self._table

    def add_roi(self, record: Dict[str, object]) -> Dict[str, object]:
        df = self.table
        base = self._default_record()
        base.update(record)
        base["tags"] = self._normalise_tags(base.get("tags"))
        base["updated_at"] = base["created_at"]
        df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)
        self._set_table(df)
        return base

    def update_roi(self, roi_id: str, updates: Dict[str, object]) -> Optional[Dict[str, object]]:
        if not roi_id:
            return None

        df = self.table
        mask = df["roi_id"] == roi_id
        if not mask.any():
            return None

        updates = updates.copy()
        if "tags" in updates:
            updates["tags"] = self._normalise_tags(updates["tags"])
        updates["updated_at"] = self._timestamp()
        for key, value in updates.items():
            if key in df.columns:
                df.loc[mask, key] = value

        self._set_table(df)
        return df.loc[mask].iloc[0].to_dict()

    def delete_roi(self, roi_id: str) -> bool:
        if not roi_id:
            return False
        df = self.table
        new_df = df[df["roi_id"] != roi_id]
        if len(new_df) == len(df):
            return False
        self._set_table(new_df)
        return True

    def get_roi(self, roi_id: str) -> Optional[Dict[str, object]]:
        if not roi_id:
            return None
        df = self.table
        matches = df[df["roi_id"] == roi_id]
        if matches.empty:
            return None
        return matches.iloc[0].to_dict()

    def list_rois(self, fov: Optional[str] = None) -> pd.DataFrame:
        df = self.table
        if fov:
            df = df[df["fov"] == fov]
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Import/export helpers
    # ------------------------------------------------------------------
    def export_to_csv(self, path: Optional[str] = None) -> str:
        target = path or self.storage_path
        df = self.table
        df.to_csv(target, index=False)
        return target

    def import_from_csv(self, path: str, merge: bool = True) -> None:
        df = pd.read_csv(path)
        df = _ensure_dataframe(df)
        current = self.table if merge else pd.DataFrame(columns=ROI_COLUMNS)

        if merge and not current.empty:
            # Avoid ID collisions by regenerating IDs that already exist
            existing_ids = set(current["roi_id"].astype(str))
            for idx, roi_id in enumerate(df["roi_id"].astype(str)):
                if roi_id in existing_ids:
                    df.at[df.index[idx], "roi_id"] = str(uuid.uuid4())

        combined = pd.concat([current, df], ignore_index=True)
        self._set_table(combined)
