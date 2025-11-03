"""Shared rendering engine for viewer, plugins, and exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import math

try:  # pragma: no cover - optional dependency is validated via higher-level tests
    from skimage.segmentation import find_boundaries  # type: ignore
except Exception:  # pragma: no cover - allow environments without skimage
    find_boundaries = None  # type: ignore[assignment]

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
    alpha: float = 1.0
    mode: str = "fill"
    outline_thickness: int = 1


@dataclass(frozen=True)
class AnnotationOverlaySnapshot:
    name: str
    alpha: float
    mode: str
    palette: Mapping[str, str]


@dataclass(frozen=True)
class MaskOverlaySnapshot:
    name: str
    color: ColorTuple
    alpha: float = 1.0
    mode: str = "outline"
    outline_thickness: int = 1


@dataclass(frozen=True)
class OverlaySnapshot:
    include_annotations: bool
    include_masks: bool
    annotation: Optional[AnnotationOverlaySnapshot] = None
    masks: Tuple[MaskOverlaySnapshot, ...] = ()


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


def _coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _normalise_roi_region(
    x_min: Optional[float],
    x_max: Optional[float],
    y_min: Optional[float],
    y_max: Optional[float],
    fallback: Region,
) -> Region:
    if None in (x_min, x_max, y_min, y_max):
        return fallback

    xmin = int(math.floor(min(x_min, x_max)))
    xmax = int(math.ceil(max(x_min, x_max)))
    ymin = int(math.floor(min(y_min, y_max)))
    ymax = int(math.ceil(max(y_min, y_max)))

    if xmax <= xmin:
        xmax = xmin + 1
    if ymax <= ymin:
        ymax = ymin + 1
    return (xmin, xmax, ymin, ymax)


def _resolve_roi_region(roi_definition: Mapping[str, float], fallback: Region) -> Region:
    x_min = _coerce_float(roi_definition.get("x_min"))
    x_max = _coerce_float(roi_definition.get("x_max"))
    y_min = _coerce_float(roi_definition.get("y_min"))
    y_max = _coerce_float(roi_definition.get("y_max"))

    if None not in (x_min, x_max, y_min, y_max):
        return _normalise_roi_region(x_min, x_max, y_min, y_max, fallback)

    x_center = _coerce_float(roi_definition.get("x"))
    y_center = _coerce_float(roi_definition.get("y"))
    width = _coerce_float(roi_definition.get("width"))
    height = _coerce_float(roi_definition.get("height"))

    if None not in (x_center, y_center, width, height) and width > 0 and height > 0:
        half_w = width / 2.0
        half_h = height / 2.0
        return _normalise_roi_region(
            x_center - half_w,
            x_center + half_w,
            y_center - half_h,
            y_center + half_h,
            fallback,
        )

    return fallback


def _derive_downsampled_region(region: Region, downsample: int) -> Region:
    xmin, xmax, ymin, ymax = region
    xmin_ds = xmin // downsample
    ymin_ds = ymin // downsample

    width = max(1, int(math.ceil(max(0, xmax - xmin) / downsample)))
    height = max(1, int(math.ceil(max(0, ymax - ymin) / downsample)))

    xmax_ds = xmin_ds + width
    ymax_ds = ymin_ds + height
    return (xmin_ds, xmax_ds, ymin_ds, ymax_ds)


def _normalise_color(color: ColorTuple) -> np.ndarray:
    return np.asarray(color, dtype=np.float32).reshape(1, 1, 3)


def _label_boundaries(mask_labels: np.ndarray) -> np.ndarray:
    labels = mask_labels.astype(np.int64, copy=False)
    boundaries = np.zeros(labels.shape, dtype=bool)
    if labels.size == 0:
        return boundaries

    interior = labels != 0

    north = np.zeros_like(boundaries)
    north[1:, :] = labels[1:, :] != labels[:-1, :]

    south = np.zeros_like(boundaries)
    south[:-1, :] = labels[:-1, :] != labels[1:, :]

    east = np.zeros_like(boundaries)
    east[:, :-1] = labels[:, :-1] != labels[:, 1:]

    west = np.zeros_like(boundaries)
    west[:, 1:] = labels[:, 1:] != labels[:, :-1]

    boundaries |= (north | south | east | west) & interior

    if labels.shape[0] > 0:
        boundaries[0, :] |= labels[0, :] != 0
        boundaries[-1, :] |= labels[-1, :] != 0
    if labels.shape[1] > 0:
        boundaries[:, 0] |= labels[:, 0] != 0
        boundaries[:, -1] |= labels[:, -1] != 0

    return boundaries


def _binary_dilation_4(mask_bool: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask_bool
    result = mask_bool.astype(bool, copy=False)
    if result.size == 0:
        return result
    for _ in range(iterations):
        expanded = result.copy()
        if result.shape[0] > 1:
            expanded[0, :] |= result[1, :]
            expanded[-1, :] |= result[-2, :]
        if result.shape[1] > 1:
            expanded[:, 0] |= result[:, 1]
            expanded[:, -1] |= result[:, -2]
        if result.shape[0] > 2 and result.shape[1] > 2:
            expanded[1:-1, 1:-1] |= result[:-2, 1:-1]
            expanded[1:-1, 1:-1] |= result[2:, 1:-1]
            expanded[1:-1, 1:-1] |= result[1:-1, :-2]
            expanded[1:-1, 1:-1] |= result[1:-1, 2:]
        result = expanded
    return result


def _resolve_mask_pixels(mask_array: np.ndarray, mask: MaskRenderSettings) -> np.ndarray:
    mode = (mask.mode or "fill").lower()
    if mode == "outline":
        labels = mask_array.astype(np.int64, copy=False)
        baseline = (
            find_boundaries(labels, mode="inner")
            if find_boundaries is not None
            else _label_boundaries(labels)
        )
        thickness = max(1, int(getattr(mask, "outline_thickness", 1)))
        if thickness > 1:
            baseline = _binary_dilation_4(baseline, thickness - 1)
        return baseline
    if mode == "fill":
        return mask_array.astype(bool, copy=False)
    raise ValueError(f"Unsupported mask render mode: {mask.mode}")


def _blend_mask_pixels(result: np.ndarray, affected: np.ndarray, colour: np.ndarray, alpha: float) -> None:
    if alpha >= 1.0:
        result[affected] = colour
        return
    existing = result[affected]
    result[affected] = (1.0 - alpha) * existing + alpha * colour


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
        if not np.any(mask_region):
            continue
        affected = _resolve_mask_pixels(mask_region, mask)
        if not np.any(affected):
            continue
        colour = _normalise_color(mask.color)
        alpha = float(np.clip(mask.alpha, 0.0, 1.0))
        _blend_mask_pixels(result, affected, colour, alpha)
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
    has_bounds = {"x_min", "x_max", "y_min", "y_max"}.issubset(roi_definition)
    has_center = {"x", "y", "width", "height"}.issubset(roi_definition)
    if not has_bounds and not has_center:
        raise ValueError(
            "roi_definition must include either (x_min, x_max, y_min, y_max) or (x, y, width, height)"
        )

    bounds = _infer_region(channel_arrays, selected_channels)
    region_xy = _resolve_roi_region(roi_definition, bounds)
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


__all__ = [
    "AnnotationOverlaySnapshot",
    "AnnotationRenderSettings",
    "ChannelRenderSettings",
    "MaskOverlaySnapshot",
    "MaskRenderSettings",
    "OverlaySnapshot",
    "render_crop_to_array",
    "render_fov_to_array",
    "render_roi_to_array",
]
