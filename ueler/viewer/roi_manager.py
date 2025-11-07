"""ROI manager utilities exposed via the packaged viewer namespace."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional

import pandas as pd

from .observable import Observable

__all__ = ["ROI_COLUMNS", "ROIManager"]

ROI_COLUMNS = [
    "roi_id",
    "fov",
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
    "comment",
    "created_at",
    "updated_at",
]


def _ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in ROI_COLUMNS if col not in df.columns]
    for col in missing:
        df[col] = (
            ""
            if col
            in {
                "marker_set",
                "tags",
                "annotation_palette",
                "mask_color_set",
                "mask_visibility",
                "comment",
            }
            else 0.0
        )
    df = df[ROI_COLUMNS]
    for col in ["marker_set", "tags", "annotation_palette", "mask_color_set", "mask_visibility", "comment"]:
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
            "comment": "",
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
