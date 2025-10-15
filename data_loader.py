# data_loader.py

import os
import glob
import dask
import dask.array as da
from dask_image.imread import imread
# from dask.distributed import LocalCluster, Client
import numpy as np

def load_channel_struct_fov(fov_name, base_folder):
    """
    Load requested images (channels) for a specific Field of View (FOV).

    Parameters:
    fov_name (str): The name of the Field of View (FOV) to load.
    base_folder (str): The base directory containing the FOV folders.
    channel_max_values (dict): A dictionary containing the maximum values for each channel.

    Returns:
    dict: A dictionary where keys are channel names and values are None if no channels are loaded.
          Returns None if no TIFF files are found in the specified FOV folder.
    """
    fov_path = os.path.join(base_folder, fov_name)

    # Check for 'rescaled' subfolder
    rescaled_path = os.path.join(fov_path, 'rescaled')
    if os.path.isdir(rescaled_path):
        channel_folder = rescaled_path
    else:
        channel_folder = fov_path

    # Get list of TIFF files in the channel folder
    tiff_files = glob.glob(os.path.join(channel_folder, '*.tiff'))
    tiff_files += glob.glob(os.path.join(channel_folder, '*.tif'))  # Include .tif files

    if not tiff_files:
        print(f"No TIFF files found in {channel_folder}. Skipping {fov_name}.")
        return None

    # Sort files to maintain consistent channel order
    tiff_files.sort()

    # Load channel structures
    channels = {}

    # If requested_channels is None, load the first channel
    for tiff_file in tiff_files:
        channel_name = os.path.splitext(os.path.basename(tiff_file))[0]
        channels[channel_name] = None
    return channels
    
def merge_channel_max(channel_name, channel_max_values, display_max, dtype_max):
    """Ensure the cached channel maxima grow monotonically."""
    if channel_name is None or channel_max_values is None:
        return False

    try:
        display_candidate = float(display_max)
    except (TypeError, ValueError):
        return False

    if not np.isfinite(display_candidate):
        return False

    try:
        dtype_candidate = float(dtype_max) if dtype_max is not None else display_candidate
    except (TypeError, ValueError):
        dtype_candidate = display_candidate

    if not np.isfinite(dtype_candidate):
        dtype_candidate = display_candidate

    record = channel_max_values.get(channel_name)
    if isinstance(record, dict):
        prev_display = float(record.get("display_max", 0.0) or 0.0)
        prev_dtype = float(record.get("dtype_max", dtype_candidate) or dtype_candidate)
    elif record is not None:
        prev_display = float(record) if np.isfinite(record) else 0.0
        prev_dtype = max(prev_display, dtype_candidate)
    else:
        prev_display = 0.0
        prev_dtype = dtype_candidate

    new_display = max(prev_display, display_candidate)
    new_dtype = max(prev_dtype, dtype_candidate, new_display)

    changed = (
        not isinstance(record, dict)
        or new_display != prev_display
        or new_dtype != prev_dtype
    )

    if changed:
        channel_max_values[channel_name] = {
            "display_max": new_display,
            "dtype_max": new_dtype
        }

    return changed


def _update_channel_max(channel_name, channel_image, channel_max_values):
    if channel_image is None:
        return

    axes = tuple(range(channel_image.ndim))

    try:
        percentile_task = da.nanpercentile(channel_image, 99.9, axis=axes)
    except (ValueError, NotImplementedError):
        percentile_task = da.nanmax(channel_image, axis=axes)

    max_task = da.nanmax(channel_image, axis=axes)
    percentile_value, absolute_max = dask.compute(percentile_task, max_task)

    if percentile_value is None or np.isnan(percentile_value):
        percentile_value = absolute_max

    dtype = channel_image.dtype
    if np.issubdtype(dtype, np.integer):
        dtype_limit = float(np.iinfo(dtype).max)
    elif np.issubdtype(dtype, np.floating):
        dtype_limit = float(np.finfo(dtype).max)
    else:
        dtype_limit = float(absolute_max) if absolute_max is not None else 65535.0

    display_max = float(np.clip(percentile_value, 0, dtype_limit))
    if not np.isfinite(display_max) or display_max <= 0:
        display_max = float(np.clip(absolute_max, 0, dtype_limit))
        if display_max == 0:
            display_max = min(dtype_limit, 1.0)

    merge_channel_max(channel_name, channel_max_values, display_max, dtype_limit)


def load_one_channel_fov(fov_name, base_folder, channel_max_values, requested_channel):
    """
    Load requested images (channels) for a specific FOV.
    """
    fov_path = os.path.join(base_folder, fov_name)

    # Check for 'rescaled' subfolder
    rescaled_path = os.path.join(fov_path, 'rescaled')
    if os.path.isdir(rescaled_path):
        channel_folder = rescaled_path
    else:
        channel_folder = fov_path

    # Get list of TIFF files in the channel folder
    tiff_files = glob.glob(os.path.join(channel_folder, '*.tiff'))
    tiff_files += glob.glob(os.path.join(channel_folder, '*.tif'))  # Include .tif files

    if not tiff_files:
        print(f"No TIFF files found in {channel_folder}. Skipping {fov_name}.")
        return None

    # Sort files to maintain consistent channel order
    tiff_files.sort()

    # Load the channel
    for tiff_file in tiff_files:
        # Extract channel name from filename
        channel_name = os.path.splitext(os.path.basename(tiff_file))[0]
        # Load image preserving original data type
        if channel_name in requested_channel:
            channel_image = imread(tiff_file)
            if channel_image.ndim == 3 and channel_image.shape[0] == 1:
                channel_image = channel_image[-1, :, :]
            channel_image = channel_image.rechunk((1024,1024))
        
            if channel_image is not None:
                _update_channel_max(channel_name, channel_image, channel_max_values)

    return channel_image

def load_images_for_fov(fov_name, base_folder, channel_max_values, requested_channels=None):
    """
    Load requested images (channels) for a specific FOV.
    """
    fov_path = os.path.join(base_folder, fov_name)

    # Check for 'rescaled' subfolder
    rescaled_path = os.path.join(fov_path, 'rescaled')
    if os.path.isdir(rescaled_path):
        channel_folder = rescaled_path
    else:
        channel_folder = fov_path

    # Get list of TIFF files in the channel folder
    tiff_files = glob.glob(os.path.join(channel_folder, '*.tiff'))
    tiff_files += glob.glob(os.path.join(channel_folder, '*.tif'))  # Include .tif files

    if not tiff_files:
        print(f"No TIFF files found in {channel_folder}. Skipping {fov_name}.")
        return None

    # Sort files to maintain consistent channel order
    tiff_files.sort()

    # Load channels
    channels = {}

    # If requested_channels is None, load the first channel
    if requested_channels is None:
        requested_channels = [os.path.splitext(os.path.basename(tiff_files[0]))[0]]
    for tiff_file in tiff_files:
        # Extract channel name from filename
        channel_name = os.path.splitext(os.path.basename(tiff_file))[0]
        # Load image preserving original data type
        if channel_name in requested_channels:
            channel_image = imread(tiff_file)
            if channel_image.ndim == 3 and channel_image.shape[0] == 1:
                channel_image = channel_image[-1, :, :]
            channels[channel_name] = channel_image.rechunk((1024,1024))
        else:
            channel_image = None
            channels[channel_name] = channel_image
        # Update channel maximum value
        if channel_image is not None:
            _update_channel_max(channel_name, channel_image, channel_max_values)

    return channels

# def load_images_for_fov(fov_name, base_folder, channel_max_values):
#     """
#     Load images for a specific FOV.
#     """
#     fov_path = os.path.join(base_folder, fov_name)

#     # Check for 'rescaled' subfolder
#     rescaled_path = os.path.join(fov_path, 'rescaled')
#     if os.path.isdir(rescaled_path):
#         channel_folder = rescaled_path
#     else:
#         channel_folder = fov_path

#     # Get list of TIFF files in the channel folder
#     tiff_files = glob.glob(os.path.join(channel_folder, '*.tiff'))
#     tiff_files += glob.glob(os.path.join(channel_folder, '*.tif'))  # Include .tif files

#     if not tiff_files:
#         print(f"No TIFF files found in {channel_folder}. Skipping {fov_name}.")
#         return None

#     # Sort files to maintain consistent channel order
#     tiff_files.sort()

#     # Load channels
#     channels = {}
#     for tiff_file in tiff_files:
#         # Load image preserving original data type
#         channel_image = imread(tiff_file)

#         if channel_image.ndim == 3 and channel_image.shape[0] == 1:
#             channel_image = channel_image[-1, :, :]

#         # Extract channel name from filename
#         channel_name = os.path.splitext(os.path.basename(tiff_file))[0]
#         channels[channel_name] = channel_image.rechunk((1024,1024))

#         # Update channel maximum value
#         max_value = channel_image.max()
#         if channel_name in channel_max_values:
#             channel_max_values[channel_name] = max(channel_max_values[channel_name], max_value)
#         else:
#             channel_max_values[channel_name] = max_value

#     return channels

def load_masks_for_fov(fov_name, masks_folder, mask_names_set):
    """
    Load masks for a specific FOV, accommodating mask suffixes with underscores.
    Ensures that masks are label images.
    """
    from skimage import measure

    mask_dict = {}

    # Get list of mask files for this FOV
    mask_files = glob.glob(os.path.join(masks_folder, fov_name + '_*.tiff'))
    mask_files += glob.glob(os.path.join(masks_folder, fov_name + '_*.tif'))

    for mask_file in mask_files:
        filename = os.path.basename(mask_file)
        name_without_ext = os.path.splitext(filename)[0]

        # Extract mask name by removing the FOV name and the underscore
        prefix = fov_name + '_'
        if name_without_ext.startswith(prefix):
            mask_name = name_without_ext[len(prefix):]  # Extract mask name after fov_name + '_'

            # Load mask
            mask_image = imread(mask_file)

            # Remove extra dimensions if present
            if mask_image.ndim == 3 and mask_image.shape[0] == 1:
                mask_image = mask_image[-1, :, :]
            elif mask_image.ndim == 3 and mask_image.shape[2] == 1:
                mask_image = mask_image[-1, :, -1]
            elif mask_image.ndim != 2:
                print(f"Warning: Mask '{mask_name}' in FOV '{fov_name}' has unexpected dimensions {mask_image.shape}.")

            # Ensure mask is a label image
            if mask_image.max() <= 1:
                # Convert binary mask to label image
                mask_image = measure.label(mask_image)

            mask_dict[mask_name] = mask_image
            mask_names_set.add(mask_name)  # Collect mask names
        else:
            # Filename does not match expected pattern
            continue

    return mask_dict


def load_annotations_for_fov(fov_name, annotations_folder, annotation_names_set):
    """Load annotation rasters for a specific FOV.

    Parameters
    ----------
    fov_name: str
        Name of the field of view.
    annotations_folder: str
        Folder containing annotation images following the pattern
        ``<fov_name>_<annotation_type>.tif(f)``.
    annotation_names_set: set[str]
        Collective set capturing all discovered annotation types.

    Returns
    -------
    dict[str, dask.array]
        Mapping of annotation name to lazily loaded dask arrays.
    """

    annotation_dict = {}

    if not annotations_folder or not os.path.isdir(annotations_folder):
        return annotation_dict

    pattern_tiff = os.path.join(annotations_folder, f"{fov_name}_*.tiff")
    pattern_tif = os.path.join(annotations_folder, f"{fov_name}_*.tif")
    annotation_files = glob.glob(pattern_tiff)
    annotation_files += glob.glob(pattern_tif)

    for annotation_file in annotation_files:
        filename = os.path.basename(annotation_file)
        stem, _ = os.path.splitext(filename)
        prefix = f"{fov_name}_"
        if not stem.startswith(prefix):
            continue
        annotation_name = stem[len(prefix):]
        array = imread(annotation_file)

        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        elif array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        elif array.ndim != 2:
            print(
                f"Warning: Annotation '{annotation_name}' in FOV '{fov_name}' "
                f"has unexpected dimensions {array.shape}."
            )

        # Ensure integer dtype for downstream indexing operations
        if not np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.int32)

        annotation_dict[annotation_name] = array
        annotation_names_set.add(annotation_name)

    return annotation_dict
