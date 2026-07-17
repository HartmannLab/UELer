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


FILL_ALPHA_DEFAULT: float = 0.35


def apply_registry_colors(
    image: np.ndarray,
    *,
    fov: str,
    mask_regions: Mapping[str, np.ndarray],
    outline_thickness: int,
    downsample_factor: int,
    color_map: Optional[Mapping[int, str]] = None,
    border_color_map: Optional[Mapping[int, str]] = None,
    enable: bool = True,
    exclude_ids: Optional[set] = None,
    mode_map: Optional[Mapping[int, str]] = None,
    opacity_map: Optional[Mapping[int, float]] = None,
    fill_alpha: float = FILL_ALPHA_DEFAULT,
    show_borders_on_filled: bool = False,
) -> np.ndarray:
    """Overlay painted mask colors onto an image array.

    Args:
        image: Base image to overlay colors onto
        fov: FOV name for looking up registry colors
        mask_regions: Dictionary of mask name -> mask array
        outline_thickness: Thickness of mask outlines
        downsample_factor: Current downsample factor
        color_map: Optional explicit color mapping (overrides registry)
        border_color_map: Optional explicit border color mapping. Cells absent
            from this mapping fall back to ``color_map`` / registry colors.
        enable: Whether to apply colors at all
        exclude_ids: Set of mask IDs to skip (e.g., currently selected cells)
        mode_map: Optional per-cell render mode mapping (mask_id -> "outline" | "fill").
            Cells absent from this mapping default to "outline".
        opacity_map: Optional per-cell fill alpha mapping (mask_id -> 0-1).
            Cells absent from this mapping fall back to ``fill_alpha``.
        fill_alpha: Alpha used when blending filled cells onto the image (0–1).
        show_borders_on_filled: Whether filled masks should also render an outline.
    """
    if not enable or not mask_regions:
        return image

    registry = dict(color_map or get_all_cell_colors_for_fov(fov))
    if not registry:
        return image

    border_registry = dict(border_color_map or {})

    dilation = _resolve_outline_dilation(outline_thickness, downsample_factor)
    result = np.array(image, copy=True)
    excluded = exclude_ids or set()
    resolved_mode_map: Mapping[int, str] = mode_map or {}
    resolved_opacity_map: Mapping[int, float] = opacity_map or {}

    for region in mask_regions.values():
        _apply_region_colors(
            result,
            np.asarray(region),
            registry,
            border_registry,
            dilation,
            excluded,
            resolved_mode_map,
            resolved_opacity_map,
            fill_alpha,
            show_borders_on_filled,
        )

    return result


def _resolve_outline_dilation(thickness: int, downsample_factor: int) -> int:
    try:
        effective = max(1, int(scale_outline_thickness(thickness, downsample_factor)))
    except Exception:
        effective = 1
    return max(0, effective - 1)


def _region_colored_ids(region_array: np.ndarray, registry: Mapping[int, str], exclude_ids: set) -> list:
    """Distinct, non-excluded mask ids in the region that have a registry color."""
    colored = []
    for raw in np.unique(region_array):
        if not raw:
            continue
        try:
            mask_id = int(raw)
        except (TypeError, ValueError):
            continue
        if mask_id in exclude_ids:
            continue
        if registry.get(mask_id):
            colored.append(mask_id)
    return colored


def _can_vectorize_fill(
    colored_ids: list,
    mode_map: Mapping[int, str],
    opacity_map: Mapping[int, float],
    fill_alpha: float,
    show_borders_on_filled: bool,
) -> bool:
    """Fast-path guard: every colored cell is opaque-ish fill with no border.

    This is the common continuous-coloring case (all cells filled, uniform-ish
    opacity, no outlines). Anything needing outlines/borders/zero-alpha fallback
    falls back to the per-cell loop so categorical behavior is untouched.
    """
    if show_borders_on_filled:
        return False
    for mask_id in colored_ids:
        if mode_map.get(mask_id) != "fill":
            return False
        try:
            alpha = float(opacity_map.get(mask_id, fill_alpha))
        except (TypeError, ValueError):
            alpha = fill_alpha
        if alpha <= 0.0:
            return False
    return True


def _apply_region_colors_fill_vectorized(
    canvas: np.ndarray,
    region_array: np.ndarray,
    registry: Mapping[int, str],
    colored_ids: list,
    opacity_map: Mapping[int, float],
    fill_alpha: float,
) -> None:
    """Vectorized fill recolor via a per-id lookup table (O(pixels), not O(cells×pixels))."""
    region_idx = region_array if np.issubdtype(region_array.dtype, np.integer) else region_array.astype(np.intp)
    max_id = int(region_idx.max())
    color_lut = np.zeros((max_id + 1, 3), dtype=np.float32)
    alpha_lut = np.zeros((max_id + 1,), dtype=np.float32)
    active = np.zeros((max_id + 1,), dtype=bool)
    for mask_id in colored_ids:
        if mask_id > max_id:
            continue
        rgb = _to_rgb_safe(registry.get(mask_id))
        if rgb is None:
            continue
        try:
            alpha = float(opacity_map.get(mask_id, fill_alpha))
        except (TypeError, ValueError):
            alpha = fill_alpha
        alpha = max(0.0, min(1.0, alpha))
        color_lut[mask_id] = rgb
        alpha_lut[mask_id] = alpha
        active[mask_id] = True

    fill_mask = active[region_idx]
    if not fill_mask.any():
        return
    rgb_px = color_lut[region_idx]
    alpha_px = alpha_lut[region_idx][..., None]
    blended = (1.0 - alpha_px) * canvas + alpha_px * rgb_px
    canvas[fill_mask] = blended[fill_mask].astype(canvas.dtype, copy=False)


def _apply_region_colors(
    canvas: np.ndarray,
    region_array: np.ndarray,
    registry: Mapping[int, str],
    border_registry: Mapping[int, str],
    dilation: int,
    exclude_ids: set,
    mode_map: Mapping[int, str],
    opacity_map: Mapping[int, float],
    fill_alpha: float,
    show_borders_on_filled: bool,
) -> None:
    if region_array.size == 0:
        return

    # Fast vectorized path for the common all-fill, no-border case (continuous
    # coloring). Falls back to the per-cell loop for outlines/borders/mixed modes.
    colored_ids = _region_colored_ids(region_array, registry, exclude_ids)
    if not colored_ids:
        return
    if _can_vectorize_fill(colored_ids, mode_map, opacity_map, fill_alpha, show_borders_on_filled):
        _apply_region_colors_fill_vectorized(
            canvas, region_array, registry, colored_ids, opacity_map, fill_alpha
        )
        return

    pending_edges: list[tuple[np.ndarray, np.ndarray]] = []

    for raw_mask_id, colour_hex in _iter_mask_region_ids(region_array, registry):
        # Skip excluded IDs (e.g., currently selected cells)
        if raw_mask_id in exclude_ids:
            continue

        rgb = _to_rgb_safe(colour_hex)
        if rgb is None:
            continue
        border_rgb = _to_rgb_safe(border_registry.get(int(raw_mask_id), colour_hex))
        if border_rgb is None:
            border_rgb = rgb

        mask_bool = region_array == raw_mask_id
        if not np.any(mask_bool):
            continue

        render_mode = mode_map.get(int(raw_mask_id), "outline")
        if render_mode == "fill":
            resolved_alpha = opacity_map.get(int(raw_mask_id), fill_alpha)
            try:
                resolved_alpha = max(0.0, min(1.0, float(resolved_alpha)))
            except (TypeError, ValueError):
                resolved_alpha = fill_alpha
            rgb_arr = np.array(rgb, dtype=np.float32)
            if resolved_alpha > 0.0:
                canvas[mask_bool] = (
                    (1.0 - resolved_alpha) * canvas[mask_bool] + resolved_alpha * rgb_arr
                ).astype(canvas.dtype)
            if show_borders_on_filled or resolved_alpha <= 0.0:
                edges = find_boundaries(mask_bool, mode="inner")
                if dilation > 0:
                    edges = thicken_outline(edges, dilation)
                edges = np.logical_and(edges, mask_bool)
                if np.any(edges):
                    pending_edges.append((edges, np.array(border_rgb, dtype=np.float32)))
        else:
            edges = find_boundaries(mask_bool, mode="inner")
            if dilation > 0:
                edges = thicken_outline(edges, dilation)
            if np.any(edges):
                pending_edges.append((edges, np.array(rgb, dtype=np.float32)))

    for edges, edge_rgb in pending_edges:
        canvas[edges] = edge_rgb.astype(canvas.dtype, copy=False)


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
