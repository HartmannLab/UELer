"""Data loading helpers for the UELer viewer package."""

from __future__ import annotations

import glob
import logging
import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np

TIFF_PATTERNS: Tuple[str, ...] = ("*.tiff", "*.tif")
_CHUNK_SIZE = (1024, 1024)


logger = logging.getLogger(__name__)


def _list_tiff_files(folder: str) -> List[str]:
    files: List[str] = []
    for pattern in TIFF_PATTERNS:
        files.extend(glob.glob(os.path.join(folder, pattern)))
    files.sort()
    return files


def _ensure_dask():
    try:
        import dask  # type: ignore
        import dask.array as da  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "dask and dask[array] are required to load UELer imagery."
        ) from exc
    return dask, da


def _ensure_imread():
    try:
        from dask_image.imread import imread  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "dask-image is required to read TIFF assets for UELer."
        ) from exc
    return imread


def _ensure_measure():
    try:
        from skimage import measure  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "scikit-image is required to post-process mask rasters for UELer."
        ) from exc
    return measure


def _coerce_float(value, fallback: float | None = None) -> float | None:
    if value is None:
        return fallback
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result


def _previous_channel_record(record, dtype_fallback: float) -> Tuple[float, float]:
    if isinstance(record, dict):
        prev_display = _coerce_float(record.get("display_max"), 0.0) or 0.0
        prev_dtype = _coerce_float(record.get("dtype_max"), dtype_fallback) or dtype_fallback
        return prev_display, prev_dtype
    if record is not None and np.isfinite(record):
        prev_display = float(record)
        return prev_display, max(prev_display, dtype_fallback)
    return 0.0, dtype_fallback


def load_channel_struct_fov(fov_name: str, base_folder: str) -> Dict[str, None] | None:
    fov_path = os.path.join(base_folder, fov_name)
    rescaled_path = os.path.join(fov_path, "rescaled")
    channel_folder = rescaled_path if os.path.isdir(rescaled_path) else fov_path

    tiff_files = _list_tiff_files(channel_folder)
    if not tiff_files:
        print(f"No TIFF files found in {channel_folder}. Skipping {fov_name}.")
        return None

    channels: Dict[str, None] = {}
    for tiff_file in tiff_files:
        channel_name = os.path.splitext(os.path.basename(tiff_file))[0]
        channels[channel_name] = None
    return channels


def merge_channel_max(channel_name, channel_max_values, display_max, dtype_max):
    if channel_name is None or channel_max_values is None:
        return False

    display_candidate = _coerce_float(display_max)
    if display_candidate is None or not np.isfinite(display_candidate):
        return False

    dtype_candidate = _coerce_float(dtype_max, display_candidate)
    if dtype_candidate is None or not np.isfinite(dtype_candidate):
        dtype_candidate = display_candidate

    prev_display, prev_dtype = _previous_channel_record(channel_max_values.get(channel_name), dtype_candidate)

    new_display = max(prev_display, display_candidate)
    new_dtype = max(prev_dtype, dtype_candidate, new_display)

    if isinstance(channel_max_values.get(channel_name), dict) and (
        new_display == prev_display and new_dtype == prev_dtype
    ):
        return False

    channel_max_values[channel_name] = {
        "display_max": new_display,
        "dtype_max": new_dtype,
    }
    return True


def _update_channel_max(channel_name, channel_image, channel_max_values) -> None:
    if channel_image is None:
        return

    dask, da = _ensure_dask()

    axes = tuple(range(channel_image.ndim))

    try:
        percentile_task = da.nanpercentile(channel_image, 99.9, axis=axes)
    except (ValueError, NotImplementedError):
        percentile_task = da.nanmax(channel_image, axis=axes)

    max_task = da.nanmax(channel_image, axis=axes)
    percentile_value, absolute_max = dask.compute(percentile_task, max_task)

    if percentile_value is None or np.isnan(percentile_value):
        percentile_value = absolute_max

    dtype = getattr(channel_image, "dtype", None)
    if dtype is not None and np.issubdtype(dtype, np.integer):
        dtype_limit = float(np.iinfo(dtype).max)
    elif dtype is not None and np.issubdtype(dtype, np.floating):
        dtype_limit = float(np.finfo(dtype).max)
    else:
        dtype_limit = float(absolute_max) if absolute_max is not None else 65535.0

    display_max = float(np.clip(percentile_value, 0, dtype_limit))
    if not np.isfinite(display_max) or display_max <= 0:
        display_max = float(np.clip(absolute_max, 0, dtype_limit))
        if display_max == 0:
            display_max = min(dtype_limit, 1.0)

    merge_channel_max(channel_name, channel_max_values, display_max, dtype_limit)


def _ensure_channel_folder(base_folder: str, fov_name: str) -> str:
    fov_path = os.path.join(base_folder, fov_name)
    rescaled_path = os.path.join(fov_path, "rescaled")
    return rescaled_path if os.path.isdir(rescaled_path) else fov_path


def load_one_channel_fov(fov_name, base_folder, channel_max_values, requested_channel):
    channel_folder = _ensure_channel_folder(base_folder, fov_name)
    tiff_files = _list_tiff_files(channel_folder)
    if not tiff_files:
        print(f"No TIFF files found in {channel_folder}. Skipping {fov_name}.")
        return None

    imread = _ensure_imread()
    channel_image = None

    for tiff_file in tiff_files:
        channel_name = os.path.splitext(os.path.basename(tiff_file))[0]
        if channel_name not in requested_channel:
            continue

        channel_image = imread(tiff_file)
        if getattr(channel_image, "ndim", 0) == 3 and channel_image.shape[0] == 1:
            channel_image = channel_image[-1, :, :]
        channel_image = channel_image.rechunk(_CHUNK_SIZE)

        _update_channel_max(channel_name, channel_image, channel_max_values)

    return channel_image


def load_images_for_fov(fov_name, base_folder, channel_max_values, requested_channels=None):
    channel_folder = _ensure_channel_folder(base_folder, fov_name)
    tiff_files = _list_tiff_files(channel_folder)
    if not tiff_files:
        print(f"No TIFF files found in {channel_folder}. Skipping {fov_name}.")
        return None

    imread = _ensure_imread()
    channels: Dict[str, object] = {}

    if requested_channels is None:
        requested_channels = [os.path.splitext(os.path.basename(tiff_files[0]))[0]]

    for tiff_file in tiff_files:
        channel_name = os.path.splitext(os.path.basename(tiff_file))[0]
        channel_image = None
        if channel_name in requested_channels:
            channel_image = imread(tiff_file)
            if getattr(channel_image, "ndim", 0) == 3 and channel_image.shape[0] == 1:
                channel_image = channel_image[-1, :, :]
            channel_image = channel_image.rechunk(_CHUNK_SIZE)
            _update_channel_max(channel_name, channel_image, channel_max_values)
        channels[channel_name] = channel_image

    return channels


def load_masks_for_fov(fov_name, masks_folder, mask_names_set):
    measure = _ensure_measure()
    mask_dict: Dict[str, object] = {}

    pattern = os.path.join(masks_folder, f"{fov_name}_*")
    mask_files: List[str] = []
    for ext in (".tiff", ".tif"):
        mask_files.extend(glob.glob(pattern + ext))

    imread = _ensure_imread()

    for mask_file in mask_files:
        filename = os.path.basename(mask_file)
        name_without_ext = os.path.splitext(filename)[0]

        prefix = f"{fov_name}_"
        if not name_without_ext.startswith(prefix):
            continue
        mask_name = name_without_ext[len(prefix):]

        mask_image = imread(mask_file)
        if getattr(mask_image, "ndim", 0) == 3 and mask_image.shape[0] == 1:
            mask_image = mask_image[-1, :, :]
        elif getattr(mask_image, "ndim", 0) == 3 and mask_image.shape[2] == 1:
            mask_image = mask_image[-1, :, -1]
        elif getattr(mask_image, "ndim", 0) != 2:
            print(
                f"Warning: Mask '{mask_name}' in FOV '{fov_name}' has unexpected dimensions {mask_image.shape}."
            )

        if getattr(mask_image, "max", lambda: 2)() <= 1:
            mask_image = measure.label(mask_image)

        mask_dict[mask_name] = mask_image
        mask_names_set.add(mask_name)

    return mask_dict


def load_annotations_for_fov(fov_name, annotations_folder, annotation_names_set):
    annotation_dict: Dict[str, object] = {}

    if not annotations_folder or not os.path.isdir(annotations_folder):
        return annotation_dict

    imread = _ensure_imread()

    pattern = os.path.join(annotations_folder, f"{fov_name}_*")
    annotation_files: List[str] = []
    for ext in (".tiff", ".tif"):
        annotation_files.extend(glob.glob(pattern + ext))

    for annotation_file in annotation_files:
        filename = os.path.basename(annotation_file)
        stem, _ = os.path.splitext(filename)
        prefix = f"{fov_name}_"
        if not stem.startswith(prefix):
            continue
        annotation_name = stem[len(prefix):]
        array = imread(annotation_file)

        if getattr(array, "ndim", 0) == 3 and array.shape[0] == 1:
            array = array[0]
        elif getattr(array, "ndim", 0) == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        elif getattr(array, "ndim", 0) != 2:
            print(
                f"Warning: Annotation '{annotation_name}' in FOV '{fov_name}' has unexpected dimensions {array.shape}."
            )

        if not np.issubdtype(getattr(array, "dtype", np.int32), np.integer):
            array = array.astype(np.int32)

        annotation_dict[annotation_name] = array
        annotation_names_set.add(annotation_name)

    return annotation_dict


def _ensure_tifffile():
    try:
        import tifffile  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "tifffile is required to read OME-TIFF metadata for UELer."
        ) from exc
    return tifffile


def extract_ome_channel_names(path: str) -> List[str]:
    tifffile = _ensure_tifffile()
    try:
        with tifffile.TiffFile(path) as tif:
            xml_str = tif.ome_metadata
    except Exception:
        return []

    if not xml_str:
        return []

    try:
        root = ET.fromstring(xml_str)
        # Try standard OME namespace
        ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
        channels = root.findall(".//ome:Channel", ns)
        if not channels:
             # Try without namespace or older
             channels = [elem for elem in root.iter() if elem.tag.endswith("Channel")]
        
        channel_names = []
        for i, ch in enumerate(channels):
            name = ch.attrib.get("Name")
            if not name:
                name = ch.attrib.get("ID", f"Channel_{i}")
            channel_names.append(name)
        return channel_names
    except Exception:
        return []


def open_ome_tiff_as_dask(path: str):
    imread = _ensure_imread()
    return imread(path)


class OMEFovWrapper:
    def __init__(self, path: str, ds_factor: int):
        self.path = path
        self.ds_factor = max(1, int(ds_factor) or 1)
        self.is_ome_tiff = True
        self._channel_cache: Dict[Tuple[str, int], object] = {}
        self._level_specs: List[Dict[str, object]] = []
        self._level_count = 0
        self._series_index = 0
        self._closed = False

        tifffile = _ensure_tifffile()
        self._tif = tifffile.TiffFile(path)
        self._series = self._tif.series[self._series_index]
        self._init_levels()

        self.channel_names = extract_ome_channel_names(path)
        n_channels = self._infer_channel_count()
        if not self.channel_names or len(self.channel_names) != n_channels:
            self.channel_names = [f"Channel_{i}" for i in range(n_channels)]
        elif len(self.channel_names) > n_channels:
            self.channel_names = self.channel_names[:n_channels]
        elif len(self.channel_names) < n_channels:
            self.channel_names.extend(
                f"Channel_{i}" for i in range(len(self.channel_names), n_channels)
            )

        self._name_to_index = {name: idx for idx, name in enumerate(self.channel_names)}

    def _init_levels(self) -> None:
        levels = list(getattr(self._series, "levels", ()))
        if not levels:
            levels = [self._series]
        base_axes = getattr(levels[0], "axes", getattr(self._series, "axes", ""))
        base_shape = getattr(levels[0], "shape", self._series.shape)
        base_y = self._axis_size(base_shape, base_axes, "Y") or 1

        for idx, level in enumerate(levels):
            axes = getattr(level, "axes", base_axes)
            shape = tuple(int(dim) for dim in getattr(level, "shape", base_shape))
            level_y = self._axis_size(shape, axes, "Y") or base_y
            scale = max(1, int(round(base_y / level_y)))
            self._level_specs.append(
                {
                    "level_index": idx,
                    "axes": axes,
                    "shape": shape,
                    "scale": scale,
                    "array": None,
                }
            )

        self._level_count = len(self._level_specs)

    @staticmethod
    def _axis_size(shape: Tuple[int, ...], axes: str, label: str) -> Optional[int]:
        try:
            idx = axes.index(label)
        except ValueError:
            return None
        return int(shape[idx]) if 0 <= idx < len(shape) else None

    def _infer_channel_count(self) -> int:
        if not self._level_specs:
            return 1
        axes = self._level_specs[0]["axes"]
        shape = self._level_specs[0]["shape"]
        idx = self._axis_index(axes, "C")
        if idx is not None:
            return int(shape[idx])
        if len(shape) == 2:
            return 1
        return int(shape[0])

    @staticmethod
    def _axis_index(axes: str, label: str) -> Optional[int]:
        try:
            return axes.index(label)
        except ValueError:
            return None

    @property
    def shape(self) -> Tuple[int, int]:
        if not self._level_specs:
            return (0, 0)
        level0 = self._level_specs[0]
        shape = level0["shape"]
        axes = level0["axes"]

        y_idx = self._axis_index(axes, "Y")
        x_idx = self._axis_index(axes, "X")

        # Fallback if axes not found (assume Y, X order if 2D, or similar)
        if y_idx is None:
            y_idx = 0 if len(shape) >= 2 else 0
        if x_idx is None:
            x_idx = 1 if len(shape) >= 2 else 0

        h = int(shape[y_idx]) if y_idx < len(shape) else 0
        w = int(shape[x_idx]) if x_idx < len(shape) else 0
        return (h, w)

    def get_channel_names(self) -> List[str]:
        return self.channel_names

    def set_downsample_factor(self, ds_factor: int) -> None:
        ds = max(1, int(ds_factor) or 1)
        if ds != self.ds_factor:
            self.ds_factor = ds
            self._channel_cache.clear()

    def _select_level(self, ds_factor: int) -> Tuple[Dict[str, object], int]:
        ds = max(1, int(ds_factor) or 1)
        best = self._level_specs[0]
        for level in self._level_specs:
            if level["scale"] <= ds and level["scale"] >= best["scale"]:
                best = level
        residual = max(1, int(math.ceil(ds / best["scale"])))
        return best, residual

    def _get_level_array(self, level: Dict[str, object]):
        cached = level.get("array")
        if cached is not None:
            return cached
        _, da = _ensure_dask()
        level_param = level["level_index"] if self._level_count > 1 else None
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[OMEFovWrapper] materializing pyramid level %s (scale=%s) for %s",
                level.get("level_index"),
                level.get("scale"),
                self.path,
            )
        store = self._series.aszarr(level=level_param)
        array = da.from_zarr(store)
        level["array"] = array
        return array

    def _slice_channel(self, level: Dict[str, object], channel_idx: int, residual: int):
        arr = self._get_level_array(level)
        axes = level["axes"]
        slices = []
        for axis in axes:
            if axis == "C":
                slices.append(channel_idx)
            elif axis == "Y":
                slices.append(slice(None, None, residual))
            elif axis == "X":
                slices.append(slice(None, None, residual))
            elif axis in {"Z", "T", "S"}:
                slices.append(0)
            else:
                slices.append(slice(None))

        if "C" not in axes and arr.ndim >= 3:
            slices[0] = channel_idx

        return arr[tuple(slices)]

    def close(self) -> None:
        if self._closed:
            return
        self._channel_cache.clear()
        for level in self._level_specs:
            level["array"] = None
        try:
            self._tif.close()
        except Exception:
            pass
        self._closed = True

    def __del__(self):
        self.close()

    def get(self, key, default=None):
        if key in self._name_to_index:
            return self[key]
        return default

    def keys(self):
        return self.channel_names

    def __getitem__(self, channel_name: str):
        if channel_name not in self._name_to_index:
            return None

        cache_key = (channel_name, self.ds_factor)
        cached = self._channel_cache.get(cache_key)
        if cached is not None:
            return cached

        level, residual = self._select_level(self.ds_factor)
        idx = self._name_to_index[channel_name]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[OMEFovWrapper] slicing channel %s (ds=%s, level_scale=%s) for %s",
                channel_name,
                self.ds_factor,
                level.get("scale"),
                self.path,
            )
        sliced = self._slice_channel(level, idx, residual)
        self._channel_cache[cache_key] = sliced
        return sliced

    def values(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[OMEFovWrapper] values() requested for %s (%d channels); this may materialize arrays",
                self.path,
                len(self.channel_names),
            )
        return [self[name] for name in self.channel_names]

    def items(self):
        return [(name, self[name]) for name in self.channel_names]

    def __contains__(self, key):
        return key in self._name_to_index

