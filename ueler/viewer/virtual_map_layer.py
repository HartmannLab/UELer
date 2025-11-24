"""Virtual map layer for composing stitched map-mode images."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .map_descriptor_loader import MapFOVSpec, SlideDescriptor


@dataclass(frozen=True)
class _Tile:
    """Viewport-aligned metadata for an individual FOV tile."""

    name: str
    pixel_size_um: float
    width_px: int
    height_px: int
    x_min_um: float
    x_max_um: float
    y_min_um: float
    y_max_um: float


class VirtualMapLayer:
    """Compose stitched RGB tiles for a map descriptor."""

    def __init__(
        self,
        viewer,
        descriptor: SlideDescriptor,
        allowed_downsample: Sequence[int],
        cache: Optional[MutableMapping] = None,
        *,
        cache_capacity: int = 6,
    ) -> None:
        self._viewer = viewer
        self.descriptor = descriptor
        self._map_id = descriptor.slide_id
        self._last_visible_fovs: Tuple[str, ...] = ()

        parsed = []
        for factor in allowed_downsample:
            try:
                value = int(factor)
            except (TypeError, ValueError):
                continue
            if value > 0:
                parsed.append(value)
        self._allowed_downsample = tuple(sorted(set(parsed)))
        if not self._allowed_downsample:
            raise ValueError("allowed_downsample must contain at least one positive factor")

        self._cache = cache if cache is not None else OrderedDict()
        self._cache_capacity = max(1, int(cache_capacity))

        self._tiles = []
        self._base_pixel_size_um = 1.0
        self._map_bounds = (0.0, 0.0, 0.0, 0.0)
        self._viewport: Optional[Tuple[float, float, float, float, int]] = None

        self._build_tile_index(descriptor.fovs)

    def set_viewport(
        self,
        xmin_um: float,
        xmax_um: float,
        ymin_um: float,
        ymax_um: float,
        *,
        downsample_factor: int,
    ) -> None:
        if downsample_factor not in self._allowed_downsample:
            raise ValueError(f"Unsupported downsample factor: {downsample_factor}")
        if xmin_um >= xmax_um or ymin_um >= ymax_um:
            raise ValueError("Viewport must have positive width and height")
        self._viewport = (
            float(xmin_um),
            float(xmax_um),
            float(ymin_um),
            float(ymax_um),
            int(downsample_factor),
        )

    def render(self, selected_channels: Iterable[str]) -> np.ndarray:
        if self._viewport is None:
            raise RuntimeError("Viewport not configured")
        if not self._tiles:
            return np.zeros((1, 1, 3), dtype=np.float32)

        xmin_um, xmax_um, ymin_um, ymax_um, ds_factor = self._viewport

        visible_tiles = self._collect_visible_tiles(xmin_um, xmax_um, ymin_um, ymax_um)
        self._last_visible_fovs = tuple(tile.name for tile, _ in visible_tiles)
        if not visible_tiles:
            empty_width = max(
                1,
                int(math.ceil((xmax_um - xmin_um) / (self._base_pixel_size_um * ds_factor))),
            )
            empty_height = max(
                1,
                int(math.ceil((ymax_um - ymin_um) / (self._base_pixel_size_um * ds_factor))),
            )
            return np.zeros((empty_height, empty_width, 3), dtype=np.float32)

        canvas = self._allocate_canvas(xmin_um, xmax_um, ymin_um, ymax_um, ds_factor)
        channels_tuple = tuple(selected_channels)
        state_signature = None
        signature_provider = getattr(self._viewer, "_map_state_signature", None)
        if callable(signature_provider):
            try:
                state_signature = signature_provider(channels_tuple, ds_factor)
            except Exception:  # pragma: no cover - viewer-provided signature is optional
                state_signature = None

        for tile, intersection in visible_tiles:
            region = self._compute_tile_region(tile, intersection, ds_factor)
            if region is None:
                continue
            region_xy, region_ds = region
            cache_key = (
                "tile",
                self._map_id,
                tile.name,
                ds_factor,
                channels_tuple,
                region_xy,
                state_signature,
            )
            image = self._cache_lookup(cache_key)
            if image is None:
                image = self._viewer._render_fov_region(  # pylint: disable=protected-access
                    tile.name,
                    channels_tuple,
                    ds_factor,
                    region_xy,
                    region_ds,
                )
                self._cache_store(cache_key, image)
            self._blit_tile(canvas, image, intersection, xmin_um, ymin_um, ds_factor)

        return canvas

    def invalidate_for_fov(self, fov_name: str) -> None:
        keys = list(self._cache.keys())
        remove_keys = []
        for key in keys:
            if not isinstance(key, tuple):
                continue
            if key and key[0] == "tile":
                tile_name_index = 2
            else:
                tile_name_index = 0
            try:
                tile_name = key[tile_name_index]
            except IndexError:
                continue
            if tile_name == fov_name:
                remove_keys.append(key)
        for key in remove_keys:
            self._cache.pop(key, None)

    def last_visible_fovs(self) -> Tuple[str, ...]:
        return self._last_visible_fovs

    def map_bounds(self) -> Tuple[float, float, float, float]:
        return self._map_bounds

    def base_pixel_size_um(self) -> float:
        return self._base_pixel_size_um

    def _build_tile_index(self, entries: Sequence[MapFOVSpec]) -> None:
        tiles = []
        base_pixel_size: Optional[float] = None
        x_values = []
        y_values = []

        for spec in entries:
            tile_info = self._tile_from_spec(spec, base_pixel_size)
            if tile_info is None:
                continue
            tile, pixel_size_um = tile_info
            if base_pixel_size is None:
                base_pixel_size = pixel_size_um
            tiles.append(tile)
            x_values.extend([tile.x_min_um, tile.x_max_um])
            y_values.extend([tile.y_min_um, tile.y_max_um])

        if base_pixel_size is not None:
            self._base_pixel_size_um = base_pixel_size
        self._tiles = tiles
        if x_values and y_values:
            self._map_bounds = (
                min(x_values),
                max(x_values),
                min(y_values),
                max(y_values),
            )

    def _tile_from_spec(
        self,
        spec: MapFOVSpec,
        base_pixel_size: Optional[float],
    ) -> Optional[Tuple[_Tile, float]]:
        if spec.fov_size_um is None:
            return None

        width_px, height_px = spec.frame_size_px
        if width_px <= 0 or height_px <= 0:
            return None

        pixel_size_um = spec.fov_size_um / float(width_px)
        if pixel_size_um <= 0:
            return None

        if base_pixel_size is not None and not math.isclose(
            pixel_size_um,
            base_pixel_size,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            return None

        half_width_um = pixel_size_um * width_px * 0.5
        half_height_um = pixel_size_um * height_px * 0.5
        x_center, y_center = spec.center_um
        tile = _Tile(
            name=spec.name,
            pixel_size_um=pixel_size_um,
            width_px=width_px,
            height_px=height_px,
            x_min_um=x_center - half_width_um,
            x_max_um=x_center + half_width_um,
            y_min_um=y_center - half_height_um,
            y_max_um=y_center + half_height_um,
        )
        return tile, pixel_size_um

    def _collect_visible_tiles(
        self,
        xmin_um: float,
        xmax_um: float,
        ymin_um: float,
        ymax_um: float,
    ) -> Sequence[Tuple[_Tile, Tuple[float, float, float, float]]]:
        visible = []
        for tile in self._tiles:
            ix_min = max(xmin_um, tile.x_min_um)
            ix_max = min(xmax_um, tile.x_max_um)
            iy_min = max(ymin_um, tile.y_min_um)
            iy_max = min(ymax_um, tile.y_max_um)
            if ix_min >= ix_max or iy_min >= iy_max:
                continue
            visible.append((tile, (ix_min, ix_max, iy_min, iy_max)))
        return visible

    def _allocate_canvas(
        self,
        xmin_um: float,
        xmax_um: float,
        ymin_um: float,
        ymax_um: float,
        downsample_factor: int,
    ) -> np.ndarray:
        pixel_size = self._base_pixel_size_um * downsample_factor
        width_px = max(1, int(math.ceil((xmax_um - xmin_um) / pixel_size)))
        height_px = max(1, int(math.ceil((ymax_um - ymin_um) / pixel_size)))
        return np.zeros((height_px, width_px, 3), dtype=np.float32)

    def _compute_tile_region(
        self,
        tile: _Tile,
        intersection: Tuple[float, float, float, float],
        downsample_factor: int,
    ) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
        ix_min, ix_max, iy_min, iy_max = intersection

        x_min_px = max(0, int(math.floor((ix_min - tile.x_min_um) / tile.pixel_size_um)))
        x_max_px = min(tile.width_px, int(math.ceil((ix_max - tile.x_min_um) / tile.pixel_size_um)))
        y_min_px = max(0, int(math.floor((iy_min - tile.y_min_um) / tile.pixel_size_um)))
        y_max_px = min(tile.height_px, int(math.ceil((iy_max - tile.y_min_um) / tile.pixel_size_um)))

        if x_min_px >= x_max_px or y_min_px >= y_max_px:
            return None

        ds = max(1, downsample_factor)
        xmin_ds = x_min_px // ds
        ymin_ds = y_min_px // ds

        width_ds = max(1, int(math.ceil((x_max_px - x_min_px) / ds)))
        height_ds = max(1, int(math.ceil((y_max_px - y_min_px) / ds)))

        xmax_ds = xmin_ds + width_ds
        ymax_ds = ymin_ds + height_ds

        region_xy = (x_min_px, x_max_px, y_min_px, y_max_px)
        region_ds = (xmin_ds, xmax_ds, ymin_ds, ymax_ds)
        return region_xy, region_ds

    def _blit_tile(
        self,
        canvas: np.ndarray,
        tile_image: np.ndarray,
        intersection: Tuple[float, float, float, float],
        viewport_xmin_um: float,
        viewport_ymin_um: float,
        downsample_factor: int,
    ) -> None:
        ix_min, _, iy_min, _ = intersection
        pixel_size_global = self._base_pixel_size_um * downsample_factor

        x_start = int(math.floor((ix_min - viewport_xmin_um) / pixel_size_global))
        y_start = int(math.floor((iy_min - viewport_ymin_um) / pixel_size_global))
        x_end = x_start + tile_image.shape[1]
        y_end = y_start + tile_image.shape[0]

        dest_x0 = max(0, x_start)
        dest_y0 = max(0, y_start)
        dest_x1 = min(canvas.shape[1], x_end)
        dest_y1 = min(canvas.shape[0], y_end)
        if dest_x0 >= dest_x1 or dest_y0 >= dest_y1:
            return

        src_x0 = dest_x0 - x_start
        src_y0 = dest_y0 - y_start
        src_x1 = src_x0 + (dest_x1 - dest_x0)
        src_y1 = src_y0 + (dest_y1 - dest_y0)

        canvas[dest_y0:dest_y1, dest_x0:dest_x1, :] = tile_image[src_y0:src_y1, src_x0:src_x1, :]

    def _cache_lookup(self, key):
        image = self._cache.get(key)
        if image is not None and hasattr(self._cache, "move_to_end"):
            self._cache.move_to_end(key)
        return image

    def _cache_store(self, key, image: np.ndarray) -> None:
        self._cache[key] = image
        if hasattr(self._cache, "move_to_end"):
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_capacity:
                self._cache.popitem(last=False)