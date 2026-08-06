"""Helpers for applying per-cell mask colors across viewer contexts."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
from matplotlib.colors import to_rgb
from skimage.segmentation import find_boundaries

from ueler.rendering import get_all_cell_colors_for_fov
from ueler.rendering.engine import scale_outline_thickness

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


def _region_index(region_array: np.ndarray) -> np.ndarray:
    """The region as an integer array usable to index a per-id lookup table."""
    if np.issubdtype(region_array.dtype, np.integer):
        return region_array
    return region_array.astype(np.intp)


def _region_colored_ids(
    unique_ids: np.ndarray, registry: Mapping[int, str], exclude_ids: set
) -> list:
    """Distinct, non-excluded mask ids in the region that have a registry color."""
    colored = []
    for raw in unique_ids:
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


def _max_dilate_labels_4(labels: np.ndarray, iterations: int) -> np.ndarray:
    """4-connected dilation of a *label* image; the largest label wins overlaps.

    Mirrors ``ueler.rendering.engine._binary_dilation_4`` shift for shift — including its
    border handling, where edge rows/columns receive only a partial update — so a batched
    outline is geometrically identical to the per-cell ``thicken_outline`` it replaces.

    Substituting ``np.maximum`` for ``|`` keeps every thickened pixel attributed to the cell
    it grew from, which is what lets one dilation replace N per-cell ones. It also resolves
    collisions exactly as the old code did: ``pending_edges`` was painted in ascending id
    order and the last write won, i.e. the highest mask id won.
    """
    if iterations <= 0:
        return labels
    result = labels
    for _ in range(iterations):
        expanded = result.copy()
        if result.shape[0] > 1:
            np.maximum(expanded[0, :], result[1, :], out=expanded[0, :])
            np.maximum(expanded[-1, :], result[-2, :], out=expanded[-1, :])
        if result.shape[1] > 1:
            np.maximum(expanded[:, 0], result[:, 1], out=expanded[:, 0])
            np.maximum(expanded[:, -1], result[:, -2], out=expanded[:, -1])
        if result.shape[0] > 2 and result.shape[1] > 2:
            core = expanded[1:-1, 1:-1]
            np.maximum(core, result[:-2, 1:-1], out=core)
            np.maximum(core, result[2:, 1:-1], out=core)
            np.maximum(core, result[1:-1, :-2], out=core)
            np.maximum(core, result[1:-1, 2:], out=core)
        result = expanded
    return result


def _dilate_within_labels(
    seeds: np.ndarray, labels: np.ndarray, iterations: int
) -> np.ndarray:
    """4-connected dilation of ``seeds`` that never crosses a label boundary.

    This is the batched equivalent of the old "dilate the cell's edges freely, then clip the
    result back to that cell". The two agree exactly: the shortest path from a pixel to its
    own cell's boundary never leaves that cell, because the step before it would exit is
    itself a boundary pixel and would have terminated the path sooner. Restricting
    propagation up front is what lets one pass serve every cell — a free dilation would let a
    neighbouring cell's pixels win ground that the clip then discards, erasing borders the
    per-cell version drew.

    Shares the shift pattern (and border handling) of ``engine._binary_dilation_4``.
    """
    if iterations <= 0:
        return seeds
    result = seeds
    for _ in range(iterations):
        expanded = result.copy()
        if result.shape[0] > 1:
            expanded[0, :] |= result[1, :] & (labels[1, :] == labels[0, :])
            expanded[-1, :] |= result[-2, :] & (labels[-2, :] == labels[-1, :])
        if result.shape[1] > 1:
            expanded[:, 0] |= result[:, 1] & (labels[:, 1] == labels[:, 0])
            expanded[:, -1] |= result[:, -2] & (labels[:, -2] == labels[:, -1])
        if result.shape[0] > 2 and result.shape[1] > 2:
            core = expanded[1:-1, 1:-1]
            lab = labels[1:-1, 1:-1]
            core |= result[:-2, 1:-1] & (labels[:-2, 1:-1] == lab)
            core |= result[2:, 1:-1] & (labels[2:, 1:-1] == lab)
            core |= result[1:-1, :-2] & (labels[1:-1, :-2] == lab)
            core |= result[1:-1, 2:] & (labels[1:-1, 2:] == lab)
        result = expanded
    return result


def _resolve_alpha(mask_id: int, opacity_map: Mapping[int, float], fill_alpha: float) -> float:
    try:
        alpha = float(opacity_map.get(mask_id, fill_alpha))
    except (TypeError, ValueError):
        alpha = fill_alpha
    return max(0.0, min(1.0, alpha))


def _paint_border_group(
    canvas: np.ndarray,
    region_idx: np.ndarray,
    boundaries: np.ndarray,
    active: np.ndarray,
    color_lut: np.ndarray,
    dilation: int,
    clip_to_own_cell: bool,
) -> None:
    """Paint one group of borders in a single vectorised pass.

    ``active`` selects which ids belong to this group; ``color_lut`` holds their colors.
    Thickening dilates an owner-label image so each grown pixel still knows which cell it
    came from. ``clip_to_own_cell`` drops pixels that grew past their owner — required for
    filled cells, whose borders must not dim a neighbour's fill (issue #91), and deliberately
    off for outline mode, which has always allowed thick outlines to spill.
    """
    seeds = boundaries & active[region_idx]
    if not seeds.any():
        return

    if clip_to_own_cell:
        # Confined borders: every painted pixel belongs to the cell it sits in, so the
        # region itself is the owner map.
        painted = _dilate_within_labels(seeds, region_idx, dilation)
        owner = region_idx
    else:
        owner = np.where(seeds, region_idx, 0)
        if dilation > 0:
            owner = _max_dilate_labels_4(owner, dilation)
            painted = owner > 0
        else:
            painted = seeds

    if not painted.any():
        return
    canvas[painted] = color_lut[owner[painted]].astype(canvas.dtype, copy=False)


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
    """Composite painted cells onto ``canvas`` in O(pixels), independent of cell count.

    Every mode is vectorised through per-id lookup tables. Boundaries are computed **once**
    on the label image: ``find_boundaries(labels, "inner")`` restricted to one cell equals
    ``find_boundaries(labels == cell, "inner")``, so this is an exact rewrite of the former
    per-cell loop rather than an approximation (issue #131).

    Fills are blended before any border is drawn, preserving the ordering fix from issue #91.
    """
    if region_array.size == 0:
        return

    region_idx = _region_index(region_array)
    colored_ids = _region_colored_ids(np.unique(region_idx), registry, exclude_ids)
    if not colored_ids:
        return

    max_id = int(region_idx.max())
    lut_size = max_id + 1

    fill_colors = np.zeros((lut_size, 3), dtype=np.float32)
    fill_alphas = np.zeros((lut_size,), dtype=np.float32)
    fill_active = np.zeros((lut_size,), dtype=bool)
    # Outline-mode borders: colored with the cell color, free to spill when thickened.
    outline_colors = np.zeros((lut_size, 3), dtype=np.float32)
    outline_active = np.zeros((lut_size,), dtype=bool)
    # Filled-cell borders: colored with the border color, clipped to the owning cell.
    border_colors = np.zeros((lut_size, 3), dtype=np.float32)
    border_active = np.zeros((lut_size,), dtype=bool)

    for mask_id in colored_ids:
        if mask_id > max_id:
            continue
        colour_hex = registry.get(mask_id)
        rgb = _to_rgb_safe(colour_hex)
        if rgb is None:
            continue

        if mode_map.get(mask_id, "outline") != "fill":
            outline_colors[mask_id] = rgb
            outline_active[mask_id] = True
            continue

        alpha = _resolve_alpha(mask_id, opacity_map, fill_alpha)
        if alpha > 0.0:
            fill_colors[mask_id] = rgb
            fill_alphas[mask_id] = alpha
            fill_active[mask_id] = True
        if show_borders_on_filled or alpha <= 0.0:
            border_rgb = _to_rgb_safe(border_registry.get(mask_id, colour_hex)) or rgb
            border_colors[mask_id] = border_rgb
            border_active[mask_id] = True

    fill_px = fill_active[region_idx]
    if fill_px.any():
        alpha_px = fill_alphas[region_idx][..., None]
        blended = (1.0 - alpha_px) * canvas + alpha_px * fill_colors[region_idx]
        canvas[fill_px] = blended[fill_px].astype(canvas.dtype, copy=False)

    if not (outline_active.any() or border_active.any()):
        return

    # One boundary computation for every cell in the region, instead of one per cell.
    boundaries = np.asarray(find_boundaries(region_idx, mode="inner"), dtype=bool)

    _paint_border_group(
        canvas, region_idx, boundaries, outline_active, outline_colors, dilation,
        clip_to_own_cell=False,
    )
    _paint_border_group(
        canvas, region_idx, boundaries, border_active, border_colors, dilation,
        clip_to_own_cell=True,
    )


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
