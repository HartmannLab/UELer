"""Helpers for applying per-cell mask colors across viewer contexts."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
from matplotlib.colors import to_rgb
from skimage.segmentation import find_boundaries

from ueler.rendering import get_all_cell_colors_for_fov
from ueler.rendering.engine import scale_outline_thickness, thicken_outline

Region = Tuple[int, int, int, int]


def compute_crop_regions(
    center_xy: Tuple[float, float],
    size_px: int,
    bounds: Region,
    downsample_factor: int,
) -> Tuple[Region, Region]:
    """Return spatial regions for crop rendering and downsampled overlays."""
    half_size = max(1, int(size_px) // 2)
    center_x = int(round(center_xy[0]))
    center_y = int(round(center_xy[1]))

    xmin = center_x - half_size
    xmax = center_x + half_size
    ymin = center_y - half_size
    ymax = center_y + half_size

    xmin = max(bounds[0], xmin)
    xmax = min(bounds[1], xmax)
    ymin = max(bounds[2], ymin)
    ymax = min(bounds[3], ymax)

    region_xy = (xmin, xmax, ymin, ymax)
    region_ds = derive_downsampled_region(region_xy, downsample_factor)
    return region_xy, region_ds


def derive_downsampled_region(region_xy: Region, downsample_factor: int) -> Region:
    xmin, xmax, ymin, ymax = region_xy
    safe_downsample = max(1, int(downsample_factor))

    xmin_ds = xmin // safe_downsample
    ymin_ds = ymin // safe_downsample

    width = max(1, int(math.ceil(max(0, xmax - xmin) / safe_downsample)))
    height = max(1, int(math.ceil(max(0, ymax - ymin) / safe_downsample)))

    xmax_ds = xmin_ds + width
    ymax_ds = ymin_ds + height
    return (xmin_ds, xmax_ds, ymin_ds, ymax_ds)


def collect_mask_regions(
    label_cache: Mapping[str, Mapping[int, np.ndarray]],
    mask_names: Iterable[str],
    downsample_factor: int,
    region_ds: Region,
) -> Dict[str, np.ndarray]:
    """Extract downsampled mask regions for the requested slice."""
    masks: Dict[str, np.ndarray] = {}
    xmin_ds, xmax_ds, ymin_ds, ymax_ds = region_ds

    for mask_name in mask_names:
        per_factor = label_cache.get(mask_name)
        if not per_factor:
            continue
        mask_ds = per_factor.get(downsample_factor)
        if mask_ds is None:
            continue
        try:
            mask_array = mask_ds.compute()
        except AttributeError:
            mask_array = np.asarray(mask_ds)
        if mask_array.size == 0:
            continue

        y0 = max(0, min(mask_array.shape[0], ymin_ds))
        y1 = max(0, min(mask_array.shape[0], ymax_ds))
        x0 = max(0, min(mask_array.shape[1], xmin_ds))
        x1 = max(0, min(mask_array.shape[1], xmax_ds))
        if y0 >= y1 or x0 >= x1:
            continue

        masks[mask_name] = mask_array[y0:y1, x0:x1]

    return masks


def apply_registry_colors(
    image: np.ndarray,
    *,
    fov: str,
    mask_regions: Mapping[str, np.ndarray],
    outline_thickness: int,
    downsample_factor: int,
    color_map: Optional[Mapping[int, str]] = None,
    enable: bool = True,
    exclude_ids: Optional[set] = None,
) -> np.ndarray:
    """Overlay painted mask colors onto an image array.
    
    Args:
        image: Base image to overlay colors onto
        fov: FOV name for looking up registry colors
        mask_regions: Dictionary of mask name -> mask array
        outline_thickness: Thickness of mask outlines
        downsample_factor: Current downsample factor
        color_map: Optional explicit color mapping (overrides registry)
        enable: Whether to apply colors at all
        exclude_ids: Set of mask IDs to skip (e.g., currently selected cells)
    """
    if not enable or not mask_regions:
        return image

    registry = dict(color_map or get_all_cell_colors_for_fov(fov))
    if not registry:
        return image

    dilation = _resolve_outline_dilation(outline_thickness, downsample_factor)
    result = np.array(image, copy=True)
    excluded = exclude_ids or set()

    for region in mask_regions.values():
        _apply_region_colors(result, np.asarray(region), registry, dilation, excluded)

    return result


def _resolve_outline_dilation(thickness: int, downsample_factor: int) -> int:
    try:
        effective = max(1, int(scale_outline_thickness(thickness, downsample_factor)))
    except Exception:
        effective = 1
    return max(0, effective - 1)


def _apply_region_colors(
    canvas: np.ndarray,
    region_array: np.ndarray,
    registry: Mapping[int, str],
    dilation: int,
    exclude_ids: set,
) -> None:
    if region_array.size == 0:
        return

    for raw_mask_id, colour_hex in _iter_mask_region_ids(region_array, registry):
        # Skip excluded IDs (e.g., currently selected cells)
        if raw_mask_id in exclude_ids:
            continue
            
        rgb = _to_rgb_safe(colour_hex)
        if rgb is None:
            continue

        mask_bool = region_array == raw_mask_id
        if not np.any(mask_bool):
            continue

        edges = find_boundaries(mask_bool, mode="inner")
        if dilation > 0:
            edges = thicken_outline(edges, dilation)
        if np.any(edges):
            canvas[edges] = rgb


def _iter_mask_region_ids(
    region_array: np.ndarray,
    registry: Mapping[int, str],
):
    for raw_mask_id in np.unique(region_array):
        if not raw_mask_id:
            continue
        try:
            mask_id = int(raw_mask_id)
        except (TypeError, ValueError):
            continue
        colour_hex = registry.get(mask_id)
        if colour_hex:
            yield raw_mask_id, colour_hex


def _to_rgb_safe(colour_hex: str) -> Optional[Tuple[float, float, float]]:
    try:
        return to_rgb(colour_hex)
    except (ValueError, TypeError):
        return None


__all__ = [
    "apply_registry_colors",
    "collect_mask_regions",
    "compute_crop_regions",
    "derive_downsampled_region",
]
