"""Rendering helpers for batch export workflows.

These helpers accept explicit render parameters so callers outside the
interactive viewer can reuse the compositing logic without depending on widget
state or side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


ColorTuple = Tuple[float, float, float]
Region = Tuple[int, int, int, int]


def _to_numpy(array) -> np.ndarray:
    """Materialise array-like inputs (NumPy or Dask) as ``float32`` arrays."""
    try:
        materialised = array.compute()
    except AttributeError:
        materialised = np.asarray(array)
    return np.asarray(materialised, dtype=np.float32)


def _slice_and_materialise(array, region: Region, downsample: int) -> np.ndarray:
    ymin, ymax = region[2], region[3]
    xmin, xmax = region[0], region[1]
    y_slice = slice(ymin, ymax, downsample)
    x_slice = slice(xmin, xmax, downsample)
    try:
        sliced = array[y_slice, x_slice]
    except TypeError:  # pragma: no cover - defensive guard for non-sliceable inputs
        sliced = array
    return _to_numpy(sliced)


def _slice_downsampled(array, region_ds: Region) -> np.ndarray:
    ymin, ymax = region_ds[2], region_ds[3]
    xmin, xmax = region_ds[0], region_ds[1]
    y_slice = slice(ymin, ymax)
    x_slice = slice(xmin, xmax)
    try:
        sliced = array[y_slice, x_slice]
    except TypeError:  # pragma: no cover - defensive guard for non-sliceable inputs
        sliced = array
    try:
        return sliced.compute()
    except AttributeError:
        return np.asarray(sliced)


@dataclass(frozen=True)
class ChannelRenderSettings:
    color: ColorTuple
    contrast_min: float
    contrast_max: float


@dataclass(frozen=True)
class AnnotationRenderSettings:
    array: np.ndarray
    colormap: np.ndarray
    alpha: float
    mode: str = "combined"


@dataclass(frozen=True)
class MaskRenderSettings:
    array: np.ndarray
    color: ColorTuple


def _infer_region(channel_arrays: Mapping[str, object], selected: Sequence[str]) -> Region:
    for channel in selected:
        candidate = channel_arrays.get(channel)
        if candidate is None:
            continue
        shape = getattr(candidate, "shape", None)
        if shape and len(shape) >= 2:
            height, width = int(shape[0]), int(shape[1])
            return (0, width, 0, height)
        # Materialise to infer shape when metadata is missing
        materialised = _to_numpy(candidate)
        return (0, materialised.shape[1], 0, materialised.shape[0])
    raise ValueError("No channel data available to infer region dimensions.")


def _ensure_region_within_bounds(region: Region, bounds: Region) -> Region:
    xmin = max(bounds[0], min(bounds[1], region[0]))
    xmax = max(xmin + 1, min(bounds[1], region[1]))
    ymin = max(bounds[2], min(bounds[3], region[2]))
    ymax = max(ymin + 1, min(bounds[3], region[3]))
    return (xmin, xmax, ymin, ymax)


def _derive_downsampled_region(region: Region, downsample: int) -> Region:
    xmin, xmax, ymin, ymax = region
    xmin_ds = xmin // downsample
    xmax_ds = max(xmin_ds + 1, xmax // downsample)
    ymin_ds = ymin // downsample
    ymax_ds = max(ymin_ds + 1, ymax // downsample)
    return (xmin_ds, xmax_ds, ymin_ds, ymax_ds)


def _normalise_color(color: ColorTuple) -> np.ndarray:
    return np.asarray(color, dtype=np.float32).reshape(1, 1, 3)


def _composite_channels(
    channel_arrays: Mapping[str, object],
    selected_channels: Sequence[str],
    channel_settings: Mapping[str, ChannelRenderSettings],
    region_xy: Region,
    downsample_factor: int,
    canvas_shape: Tuple[int, int],
) -> np.ndarray:
    composite = np.zeros((canvas_shape[0], canvas_shape[1], 3), dtype=np.float32)

    for channel in selected_channels:
        settings = channel_settings.get(channel)
        if settings is None:
            raise KeyError(f"Missing render settings for channel '{channel}'")

        region_array = _slice_and_materialise(channel_arrays[channel], region_xy, downsample_factor)
        min_val = float(settings.contrast_min)
        max_val = float(settings.contrast_max)
        if not np.isfinite(min_val):
            min_val = 0.0
        if not np.isfinite(max_val):
            max_val = min_val + 1.0
        scale = max(max_val - min_val, np.finfo(np.float32).eps)
        normalised = np.clip((region_array - min_val) / scale, 0.0, 1.0)
        composite += normalised[..., np.newaxis] * _normalise_color(settings.color)

    return np.clip(composite, 0.0, 1.0)


def _apply_annotation_overlay(
    composite: np.ndarray,
    annotation: Optional[AnnotationRenderSettings],
    region_ds: Region,
) -> np.ndarray:
    if annotation is None or annotation.mode not in {"annotation", "combined"}:
        return composite

    annotation_region = _slice_downsampled(annotation.array, region_ds)
    if not annotation_region.size:
        return composite

    annotation_int = annotation_region.astype(np.int64, copy=False)
    colormap = np.asarray(annotation.colormap, dtype=np.float32)
    if colormap.ndim != 2 or colormap.shape[1] != 3:
        raise ValueError("annotation colormap must have shape (N, 3)")

    max_index = colormap.shape[0] - 1
    safe_indices = np.clip(annotation_int, 0, max_index)
    overlay_rgb = colormap[safe_indices]
    alpha = float(np.clip(annotation.alpha, 0.0, 1.0))
    blended = (1.0 - alpha) * composite + alpha * overlay_rgb
    return np.clip(blended, 0.0, 1.0)


def _apply_mask_overlays(
    composite: np.ndarray,
    masks: Optional[Iterable[MaskRenderSettings]],
    region_ds: Region,
) -> np.ndarray:
    if not masks:
        return composite

    result = composite.copy()
    for mask in masks:
        mask_region = _slice_downsampled(mask.array, region_ds)
        if mask_region.shape[:2] != composite.shape[:2]:
            raise ValueError(
                "Mask dimensions do not match composite output for the requested region"
            )
        mask_bool = mask_region.astype(bool, copy=False)
        if np.any(mask_bool):
            result[mask_bool] = _normalise_color(mask.color)
    return result


def render_fov_to_array(
    fov_name: str,
    channel_arrays: Mapping[str, object],
    selected_channels: Sequence[str],
    channel_settings: Mapping[str, ChannelRenderSettings],
    *,
    downsample_factor: int,
    region_xy: Optional[Region] = None,
    region_ds: Optional[Region] = None,
    annotation: Optional[AnnotationRenderSettings] = None,
    masks: Optional[Iterable[MaskRenderSettings]] = None,
) -> np.ndarray:
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be >= 1")

    missing_channels = [ch for ch in selected_channels if ch not in channel_arrays]
    if missing_channels:
        raise KeyError(
            f"FOV '{fov_name}' does not provide channels: {', '.join(missing_channels)}"
        )

    bounds = _infer_region(channel_arrays, selected_channels)
    region_xy = _ensure_region_within_bounds(region_xy or bounds, bounds)
    region_ds = region_ds or _derive_downsampled_region(region_xy, downsample_factor)

    ymin_ds, ymax_ds = region_ds[2], region_ds[3]
    xmin_ds, xmax_ds = region_ds[0], region_ds[1]
    height = max(1, ymax_ds - ymin_ds)
    width = max(1, xmax_ds - xmin_ds)

    composite = _composite_channels(
        channel_arrays,
        selected_channels,
        channel_settings,
        region_xy,
        downsample_factor,
        (height, width),
    )
    composite = _apply_annotation_overlay(composite, annotation, region_ds)
    composite = _apply_mask_overlays(composite, masks, region_ds)
    return composite.astype(np.float32, copy=False)


def render_crop_to_array(
    fov_name: str,
    channel_arrays: Mapping[str, object],
    selected_channels: Sequence[str],
    channel_settings: Mapping[str, ChannelRenderSettings],
    *,
    center_xy: Tuple[float, float],
    size_px: int,
    downsample_factor: int,
    annotation: Optional[AnnotationRenderSettings] = None,
    masks: Optional[Iterable[MaskRenderSettings]] = None,
) -> np.ndarray:
    bounds = _infer_region(channel_arrays, selected_channels)
    half_size = max(1, int(size_px) // 2)
    center_x = int(round(center_xy[0]))
    center_y = int(round(center_xy[1]))
    region_xy = (
        center_x - half_size,
        center_x + half_size,
        center_y - half_size,
        center_y + half_size,
    )
    region_xy = _ensure_region_within_bounds(region_xy, bounds)
    region_ds = _derive_downsampled_region(region_xy, downsample_factor)
    return render_fov_to_array(
        fov_name,
        channel_arrays,
        selected_channels,
        channel_settings,
        downsample_factor=downsample_factor,
        region_xy=region_xy,
        region_ds=region_ds,
        annotation=annotation,
        masks=masks,
    )


def render_roi_to_array(
    fov_name: str,
    channel_arrays: Mapping[str, object],
    selected_channels: Sequence[str],
    channel_settings: Mapping[str, ChannelRenderSettings],
    *,
    roi_definition: Mapping[str, float],
    downsample_factor: int,
    annotation: Optional[AnnotationRenderSettings] = None,
    masks: Optional[Iterable[MaskRenderSettings]] = None,
) -> np.ndarray:
    if {
        "x_min",
        "x_max",
        "y_min",
        "y_max",
    }.issubset(roi_definition):
        region_xy = (
            int(roi_definition["x_min"]),
            int(roi_definition["x_max"]),
            int(roi_definition["y_min"]),
            int(roi_definition["y_max"]),
        )
    elif {"x", "y", "width", "height"}.issubset(roi_definition):
        x = float(roi_definition["x"])
        y = float(roi_definition["y"])
        width = float(roi_definition["width"])
        height = float(roi_definition["height"])
        region_xy = (
            int(round(x - width / 2.0)),
            int(round(x + width / 2.0)),
            int(round(y - height / 2.0)),
            int(round(y + height / 2.0)),
        )
    else:
        raise ValueError(
            "roi_definition must include either (x_min, x_max, y_min, y_max) or (x, y, width, height)"
        )

    region_ds = _derive_downsampled_region(region_xy, downsample_factor)
    return render_fov_to_array(
        fov_name,
        channel_arrays,
        selected_channels,
        channel_settings,
        downsample_factor=downsample_factor,
        region_xy=region_xy,
        region_ds=region_ds,
        annotation=annotation,
        masks=masks,
    )


__all__ = [
    "AnnotationRenderSettings",
    "ChannelRenderSettings",
    "MaskRenderSettings",
    "render_crop_to_array",
    "render_fov_to_array",
    "render_roi_to_array",
]
