"""Data loading helpers for the UELer viewer package."""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

import numpy as np

TIFF_PATTERNS: Tuple[str, ...] = ("*.tiff", "*.tif")
_CHUNK_SIZE = (1024, 1024)


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
