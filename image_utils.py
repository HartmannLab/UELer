# image_utils.py

from skimage.segmentation import find_boundaries
from dask import delayed

from ueler.rendering.engine import scale_outline_thickness, thicken_outline


def generate_edges(mask, *, thickness: int = 1, downsample: int = 1):
    """Generate edges from a mask with an optional thickness adjustment."""

    effective = scale_outline_thickness(thickness, downsample)
    iterations = max(0, int(effective) - 1)

    def _compute_edges(mask_array, extra_iterations):
        edges = find_boundaries(mask_array, mode="inner")
        if extra_iterations <= 0:
            return edges
        return thicken_outline(edges, extra_iterations)

    return delayed(_compute_edges)(mask, iterations)

def calculate_downsample_factor(width, height, ignore_zoom=False, max_dimension=512):
    """
    Calculate the downsample factor based on the current zoom level.
    The downsample factor inversely scales with the zoom level.
    """
    if ignore_zoom:
        return 1
    largest_dimension = max(width, height)
    factor = 1
    while (largest_dimension / factor) > max_dimension:
        factor *= 2
    return factor


def select_downsample_factor(
    width,
    height,
    *,
    max_dimension=512,
    allowed_factors=None,
    minimum=1,
):
    """Return the nearest permitted downsample factor for the given image size.

    The helper reuses :func:`calculate_downsample_factor` to determine the
    smallest power-of-two factor that keeps the longest edge within
    ``max_dimension`` pixels. When ``allowed_factors`` is provided, the result
    is clamped to the largest allowed factor that does not exceed the computed
    baseline. If no permitted value satisfies that condition, the smallest
    allowed factor is returned instead. When ``allowed_factors`` is omitted, the
    raw baseline factor is used directly.
    """

    try:
        base_factor = int(
            calculate_downsample_factor(width, height, ignore_zoom=False, max_dimension=max_dimension)
        )
    except Exception:
        base_factor = 1

    base_factor = max(1, base_factor)
    minimum = max(1, int(minimum or 1))

    if not allowed_factors:
        return max(minimum, base_factor)

    try:
        candidates = sorted({int(factor) for factor in allowed_factors if int(factor) >= minimum})
    except Exception:
        candidates = []

    if not candidates:
        return max(minimum, base_factor)

    at_most_base = [factor for factor in candidates if factor <= base_factor]
    if at_most_base:
        return at_most_base[-1]

    return candidates[0]


import os
import numpy as np
import pandas as pd
import seaborn_image as isns
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
from skimage import exposure
from skimage.transform import resize
from skimage.segmentation import find_boundaries
import cv2

import tifffile

def process_single_crop(FOV, marker2display, crop_position, crop_width, file_source='', subfolder='',
                         mask_suffix=None, mask_source='', mask_ID = None, image_viewer=None):
    """
    Process and merge images for visualization, with cropping.

    Args:
        FOV (str): Single FOV (Field of View) to process.
        marker2display (dict): Dictionary mapping channel names to display markers.
        crop_position (tuple): (x, y) position for the top-left corner of the crop.
        crop_width (int): Width of the crop (assumed to be square).
        file_source (str, optional): Source directory for the image files. Defaults to ''.
        subfolder (str, optional): Subfolder within the FOV directory. Defaults to ''.
        mask_suffix (str, optional): Suffix for the mask files. Defaults to None.
        mask_source (str, optional): Source directory for the mask files. Defaults to ''.
        mask_ID (int, optional): Specific mask ID to highlight. Defaults to None.
        image_viewer (object, optional): An `ImageMaskViewer` object for displaying MIBI images. Defaults to None.

    Returns:
        img_stack (np.array): Stacked channels of size [crop_width, crop_width, num_channels].

    Raises:
        FileNotFoundError: If the image file is not found.

    """
    img_stack = None

    x, y = crop_position

    for i, channel in enumerate(marker2display.keys()):
        path_fov = os.path.join(file_source, FOV)
        fname = os.path.join(path_fov, subfolder, channel + '.tiff')

        try:
            if image_viewer is None:
                with tifffile.TiffFile(fname) as tif:
                    img = crop_image_center(tif, x, y, crop_width)
            else:
                image_viewer.load_fov(FOV, [channel])
                tif = image_viewer.image_cache[FOV][channel]
                img = crop_image_center(tif, x, y, crop_width)

        except FileNotFoundError:
            raise FileNotFoundError(f"Image file '{fname}' not found.")

        # stack img of different channels
        if img_stack is None:
            img_stack = img
        else:
            img_stack = np.dstack((img_stack, img))

    if mask_suffix is not None:
        fname = os.path.join(mask_source, f"{FOV}_{mask_suffix}" + '.tiff')
        try:
            if image_viewer is None:
                with tifffile.TiffFile(fname) as tif:
                    img = crop_image_center(tif, x, y, crop_width)
                    # convert masks in img to outlines
                    img_bound = find_boundaries(img, mode='thick')  # 'thick' or 'inner' based on preference
            else:
                tif = image_viewer.mask_cache[FOV][mask_suffix]
                img = crop_image_center(tif, x, y, crop_width)
                # convert masks in img to outlines
                img_bound = find_boundaries(img, mode='thick')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file '{fname}' not found.")
        
        img_stack = np.dstack((img_stack, img_bound))

        if mask_ID is not None:
            img[img != mask_ID] = 0
            img = find_boundaries(img, mode='thick')  # 'thick' or 'inner' based on preference
           
            img_stack = np.dstack((img_stack, img))

    return img_stack

def crop_image_center(tif, x, y, crop_width):
    """
    Crop a square region from the center of the specified coordinates in a TIFF image.

    Parameters:
    tif (TiffFile): The TIFF file object containing the image.
    x (int): The x-coordinate of the center of the crop region.
    y (int): The y-coordinate of the center of the crop region.
    crop_width (int): The width of the square crop region.

    Returns:
    numpy.ndarray: The cropped image as a 2D numpy array.
    """
    import numpy as np
    
    # If the tif is a multi-page TIFF, read the first page
    # If it is an array, do nothing
    if isinstance(tif, tifffile.TiffFile):
        page = tif.pages[0]
    else:
        page = tif
    img_width, img_height = page.shape

    img = np.zeros((crop_width, crop_width))

    # Ensure the crop coordinates are within the image boundaries
    halfwidth = np.round(crop_width / 2).astype(int)
    x1 = max(0, x - halfwidth)
    y1 = max(0, y - halfwidth)
    x2 = min(img_width, x + halfwidth)
    y2 = min(img_height, y + halfwidth)

    # Calculate offsets for the destination image to center the crop
    offset_x1 = max(0, halfwidth - x)
    offset_y1 = max(0, halfwidth - y)
    offset_x2 = crop_width - max(0, (x + halfwidth) - img_width)
    offset_y2 = crop_width - max(0, (y + halfwidth) - img_height)

    # Adjust the destination coordinates based on the calculated offsets
    # If `page` is tif page, convert it to numpy array
    if isinstance(page, tifffile.TiffPage):
        img[offset_x1:offset_x2, offset_y1:offset_y2] = page.asarray()[x1:x2, y1:y2]
    else:
        img[offset_x1:offset_x2, offset_y1:offset_y2] = page[x1:x2, y1:y2]

    return img

def estimate_color_range(img_stack, channels, pt=99, est_lb=False):
    """
    Estimate the color range for each channel in an image stack.

    Parameters:
    - img_stack (ndarray): The image stack.
    - channels (list): The list of channel names.
    - pt (int, optional): The percentile value to use for color range estimation. Default is 99.
    - est_lb (bool, optional): Whether to estimate the lower bound of the color range. Default is False.

    Returns:
    - color_range (dict): A dictionary containing the color range for each channel.
    """
    color_range = {}
    for i, channel in enumerate(channels):
        img = img_stack[:, :, i]
        # linearize img
        img = img.flatten()
        img = img[img<255]
        img = img[img>0]

        if est_lb:
            color_range[channel] = [np.percentile(img, 1), np.percentile(img, pt)]
        else:
            color_range[channel] = [0, np.percentile(img, pt)]
    return color_range

def color_one_image(image_stack, marker2display, color_range):
    """
    Colorizes a single image based on the given marker-to-display mapping and color range.

    Args:
        image_stack (ndarray): The input image stack.
        marker2display (dict): A dictionary mapping marker names to display colors.
        color_range (dict): A dictionary mapping marker names to color range values.

    Returns:
        ndarray: The colorized image.

    """
    img_colored = None
    for i, channel in enumerate(marker2display.keys()):
        img = image_stack[:, :, i]
        channel_color_range = color_range[channel]
        lb = channel_color_range[0]
        ub = channel_color_range[1]
        
        img = img / ub
        img[img > 1] = 1

        if lb > 0:
            img = img - np.percentile(img, lb)
            img[img < 0] = 0
            img = img / np.max(img)

        # Create 3D array from 2D image
        i3d = np.atleast_3d(img)

        # Colourize image
        ic = i3d * marker2display[channel]

        if img_colored is None:
            img_colored = ic
        else:
            img_colored = img_colored + ic

    img_colored = (img_colored - np.min(img_colored))
    # img_colored = img_colored / np.percentile(img_colored, pt if not isinstance(pt, dict) else 99)
    img_colored[img_colored > 1] = 1

    img_colored = (img_colored * 255).astype(np.uint8)

    return img_colored

def process_single_crop_and_color(FOV, marker2display, lbs, gains, gammas, pt, crop_position, crop_width, file_source='', subfolder='', default_gamma=1.0, ubs=None):
    """
    Process and merge images for visualization, with cropping.

    Args:
        FOV (str): Single FOV (Field of View) to process.
        marker2display (dict): Dictionary mapping channel names to display markers.
        lbs (dict): Dictionary mapping channel names to lower bounds for intensity normalization.
        gains (dict): Dictionary mapping channel names to gain values for gamma adjustment.
        gammas (dict): Dictionary mapping channel names to gamma values for gamma adjustment.
        pt (float or dict): Percentile value(s) for intensity normalization. Can be a single float or a dictionary of floats for each channel.
        crop_position (tuple): (x, y) position for the top-left corner of the crop.
        crop_width (int): Width of the crop (assumed to be square).
        file_source (str, optional): Source directory for the image files. Defaults to ''.
        subfolder (str, optional): Subfolder within the FOV directory. Defaults to ''.
        default_gamma (float, optional): Default gamma value for gamma adjustment. Defaults to 1.0.
        ubs (dict, optional): Dictionary mapping channel names to upper bounds. If provided, these values override the pt.

    Returns:
        img_colored (np.array): Processed and merged image for visualization.
        ubs (list): List of upper bounds for each channel.

    Raises:
        FileNotFoundError: If the image file is not found.

    """
    ubs = [] if ubs is None else ubs
    img_colored = None

    x, y = crop_position

    for i, channel in enumerate(marker2display.keys()):
        path_fov = os.path.join(file_source, FOV)
        fname = os.path.join(path_fov, subfolder, channel + '.tiff')

        try:
            with tifffile.TiffFile(fname) as tif:
                page = tif.pages[0]
                img_width, img_height = page.shape

                img = np.zeros((crop_width, crop_width))

                # Ensure the crop coordinates are within the image boundaries
                halfwidth = np.round(crop_width / 2).astype(int)
                x1 = max(0, x - halfwidth)
                y1 = max(0, y - halfwidth)
                x2 = min(img_width, x + halfwidth)
                y2 = min(img_height, y + halfwidth)

                # Calculate offsets for the destination image to center the crop
                offset_x1 = max(0, halfwidth - x)
                offset_y1 = max(0, halfwidth - y)
                offset_x2 = crop_width - max(0, (x + halfwidth) - img_width)
                offset_y2 = crop_width - max(0, (y + halfwidth) - img_height)

                # Adjust the destination coordinates based on the calculated offsets
                img[offset_x1:offset_x2, offset_y1:offset_y2] = page.asarray()[x1:x2, y1:y2]

        except FileNotFoundError:
            raise FileNotFoundError(f"Image file '{fname}' not found.")

        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        # Determine the upper bound for this channel
        if isinstance(ubs, dict) and channel in ubs:
            ub = ubs[channel]
        else:
            # Determine the percentile value for this channel
            if isinstance(pt, dict):
                pt_value = pt.get(channel, 99)  # Use channel-specific value or default to 99 if not provided
            else:
                pt_value = pt  # Use the single provided value

            if pt_value <= 100:
                if img[img>0].size == 0:
                    ub = 1
                else:
                    ub  = np.percentile(img[img>0], pt_value)
            else:
                ub = np.max(img)*(10**(pt_value/100-1))

        ubs.append(ub)
        
        img = img / ub
        img[img > 1] = 1

        if channel in lbs:
            lb = lbs[channel]
            img = img - np.percentile(img, lb)
            img[img < 0] = 0
            img = img / np.max(img)

        gain = gains.get(channel, 1)
        gamma = gammas.get(channel, default_gamma)

        # Adjust image gamma
        img = exposure.adjust_gamma(img, gamma, gain)

        # Create 3D array from 2D image
        i3d = np.atleast_3d(img)

        # Colourize image
        ic = i3d * marker2display[channel]

        if img_colored is None:
            img_colored = ic
        else:
            img_colored = img_colored + ic

    img_colored = (img_colored - np.min(img_colored))
    # img_colored = img_colored / np.percentile(img_colored, pt if not isinstance(pt, dict) else 99)
    img_colored[img_colored > 1] = 1

    img_colored = (img_colored * 255).astype(np.uint8)

    return img_colored, ubs


def process_single_image(FOV, marker2display, lbs, gains, gammas, pt, file_source='', subfolder='', default_gamma=1.0, ubs=None):
    """
    Process and merge images for visualization.

    Args:
        FOV (str): Single FOV (Field of View) to process.
        marker2display (dict): Dictionary mapping channel names to display markers.
        lbs (dict): Dictionary mapping channel names to lower bounds for intensity normalization.
        gains (dict): Dictionary mapping channel names to gain values for gamma adjustment.
        gammas (dict): Dictionary mapping channel names to gamma values for gamma adjustment.
        pt (float or dict): Percentile value(s) for intensity normalization. Can be a single float or a dictionary of floats for each channel.
        file_source (str, optional): Source directory for the image files. Defaults to ''.
        subfolder (str, optional): Subfolder within the FOV directory. Defaults to ''.
        default_gamma (float, optional): Default gamma value for gamma adjustment. Defaults to 1.0.
        ubs (dict, optional): Dictionary mapping channel names to upper bounds. If provided, these values override the pt.

    Returns:
        img_colored (np.array): Processed and merged image for visualization.
        ubs (list): List of upper bounds for each channel.

    Raises:
        FileNotFoundError: If the image file is not found.

    """
    ubs = [] if ubs is None else ubs
    img_colored = None

    for i, channel in enumerate(marker2display.keys()):
        path_fov = os.path.join(file_source, FOV)
        fname = os.path.join(path_fov, subfolder, channel + '.tiff')

        try:
            img = imread(fname)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file '{fname}' not found.")

        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        # Determine the upper bound for this channel
        if isinstance(ubs, dict) and channel in ubs:
            ub = ubs[channel]
        else:
            # Determine the percentile value for this channel
            if isinstance(pt, dict):
                pt_value = pt.get(channel, 99)  # Use channel-specific value or default to 99 if not provided
            else:
                pt_value = pt  # Use the single provided value

            if pt_value <= 100:
                ub = np.percentile(img[img>0], pt_value)
            else:
                ub = np.max(img)*(10**(pt_value/100-1))
            

        ubs.append(ub)
        
        img = img / ub
        img[img > 1] = 1

        if channel in lbs:
            lb = lbs[channel]
            img = img - np.percentile(img, lb)
            img[img < 0] = 0
            img = img / np.max(img)

        gain = gains.get(channel, 1)
        gamma = gammas.get(channel, default_gamma)

        # Adjust image gamma
        img = exposure.adjust_gamma(img, gamma, gain)

        # Create 3D array from 2D image
        i3d = np.atleast_3d(img)

        # Colourize image
        ic = i3d * marker2display[channel]

        if img_colored is None:
            img_colored = ic
        else:
            img_colored = img_colored + ic

    img_colored = (img_colored - np.min(img_colored))
    # img_colored = img_colored / np.percentile(img_colored, pt if not isinstance(pt, dict) else 99)
    img_colored[img_colored > 1] = 1

    img_colored = (img_colored * 255).astype(np.uint8)

    return img_colored, ubs



def process_images(FOVs, text, marker2display, lbs, gains, gammas, pt, base_dir, project_name, set_id, file_source='', subfolder='', n=5, m=5, default_gamma=1.0):
    """
    Process and merge images for visualization.

    Args:
        FOVs (list): List of FOVs (Field of Views) to process.
        text (list): List of text labels to add to each component image.
        marker2display (dict): Dictionary mapping channel names to display markers.
        lbs (dict): Dictionary mapping channel names to lower bounds for intensity normalization.
        gains (dict): Dictionary mapping channel names to gain values for gamma adjustment.
        gammas (dict): Dictionary mapping channel names to gamma values for gamma adjustment.
        pt (float): Percentile value for intensity normalization.
        base_dir (str): Base directory for saving the visualization.
        project_name (str): Name of the project.
        set_id (str): ID of the image set.
        file_source (str, optional): Source directory for the image files. Defaults to ''.
        subfolder (str, optional): Subfolder within each FOV directory. Defaults to ''.
        n (int, optional): Number of rows in the image grid. Defaults to 5.
        m (int, optional): Number of columns in the image grid. Defaults to 5.
        default_gamma (float, optional): Default gamma value for gamma adjustment. Defaults to 1.0.

    Returns:
        None
    """
    for i, channel in enumerate(list(marker2display.keys())):
        t = 0
        img_grid = []
        row_images = []
        for idx, FOV in enumerate(FOVs):
            path_fov = os.path.join(file_source, FOV)
            fname = os.path.join(path_fov, subfolder, channel + '.tiff')

            img = imread(fname)
            row_images.append(img)
            if (idx + 1) % m == 0:  # When we have enough images, add the row to the grid
                img_row = np.concatenate(row_images, axis=1)
                img_grid.append(img_row)
                row_images = []

        # If there are any leftover images that didn't form a complete row, add them now
        if row_images:
            while len(row_images) < m:  # Pad the row with blank images until it has m images
                blank_img = np.zeros_like(row_images[0])
                row_images.append(blank_img)
            img_row = np.concatenate(row_images, axis=1)
            img_grid.append(img_row)

        img_tiled = np.concatenate(img_grid, axis=0)

        img_tiled = (img_tiled - np.min(img_tiled)) / (np.max(img_tiled) - np.min(img_tiled))

        img_tiled = img_tiled / np.percentile(img_tiled, pt)
        img_tiled[img_tiled > 1] = 1

        if channel in lbs.keys():
            lb = lbs[channel]
            img_tiled = img_tiled - np.percentile(img_tiled, lb)
            img_tiled[img_tiled < 0] = 0
            img_tiled = img_tiled / np.max(img_tiled)

        if channel in gains.keys():
            gain = gains[channel]
        else:
            gain = 1

        if channel in gains.keys():
            gamma = gammas[channel]
        else:
            gamma = default_gamma

        # adjust image gamma
        img_tiled = exposure.adjust_gamma(img_tiled, gamma, gain)

        # create 3d array from 2d image
        i3d = np.atleast_3d(img_tiled)

        # colourize image
        ic = i3d * marker2display[channel]

        if i == 0:
            img_tiled_colored = ic
        else:
            img_tiled_colored = img_tiled_colored + ic

    img_tiled_colored = (img_tiled_colored - np.min(img_tiled_colored))
    img_tiled_colored = img_tiled_colored / np.percentile(img_tiled, pt)
    img_tiled_colored[img_tiled_colored > 1] = 1

    img_tiled_colored = (img_tiled_colored * 255).astype(np.uint8)

    # adding corresponding text to each component image
    for idx in range(len(text)):
        current_row = idx // m
        current_col = idx % m
        cv2.putText(img_tiled_colored, text[idx], (10 + 2048 * current_col, 250 + 2048 * current_row), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 255, 255), 20)  # Add this line

    visualization_dir = os.path.join(base_dir, project_name, "visualization")
    if not os.path.exists(visualization_dir):
        os.mkdir(visualization_dir)
    imsave(os.path.join(visualization_dir, project_name + '_' + set_id + '_merge.png'), img_tiled_colored, check_contrast=False)

def color_annotations(pixel_annotation_dir, pixel_annotation_viz_dir, color_map):
    """
    Process each TIFF image in the specified directory by mapping each label to a color
    according to the provided color map and saving the resulting image as a TIFF file.

    Args:
        pixel_annotation_dir (str): The directory containing the TIFF images to be processed.
        pixel_annotation_viz_dir (str): The directory where the processed images will be saved.
        color_map (dict): A dictionary mapping each label to a color.

    Returns:
        None
    """
    # Process each TIFF image in dir1
    for filename in os.listdir(pixel_annotation_dir):
        if filename.endswith('.tiff'):
            # Load the image
            img = imread(os.path.join(pixel_annotation_dir, filename))
            # Create a new RGB image
            new_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            # Map each label to a color
            for label, color in color_map.items():
                new_img[img == label] = color
            # Save the new image as a TIFF file
            imsave(os.path.join(pixel_annotation_viz_dir, filename), new_img)

def color_annotations_single_FOV(FOV, file_source, color_map):
    """
    Process each TIFF image in the specified directory by mapping each label to a color
    according to the provided color map and saving the resulting image as a TIFF file.

    Args:
        FOV (str): Single FOV (Field of View) to process.
        file_source (str): The directory containing the TIFF images to be processed.
        color_map (dict): A dictionary mapping each label to a color.

    Returns:
        None
    """
    # Process each TIFF image in dir1
    filename = FOV + '_Simple Segmentation.tiff'
    # Load the image
    img = imread(os.path.join(file_source, filename))
    # Create a new RGB image
    new_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # Map each label to a color
    for label, color in color_map.items():
        new_img[img == label] = color
    # Save the new image as a TIFF file
    return new_img


import numpy as np
from skimage import color, io, measure
from skimage.io import imread
import cv2  # Ensure OpenCV is installed
import os

def draw_contour(image, contour, thickness, color):
    """
    Draw a single contour on the image with the specified thickness and color.

    Args:
        image (ndarray): The original image.
        contour (ndarray): Contour points.
        thickness (int): Thickness of the contour.
        color (list): Color of the contour in [R, G, B].

    Returns:
        ndarray: The image with the contour drawn on it.
    """
    for point in contour:
        rr, cc = draw.circle_perimeter(int(point[0]), int(point[1]), radius=thickness, shape=image.shape)
        valid = (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
        image[rr[valid], cc[valid]] = color
    return image

def overlay_masks_single_FOV(FOV, image_source, mask_source, output_dir, source_surfix, mask_surfix, thickness = 1, color = [255, 255, 255]):
    """
    Overlay masks on the original image for a single FOV.

    Args:
        FOV (str): Single FOV (Field of View) to process.
        image_source (str): Source directory for the original image files.
        mask_source (str): Source directory for the mask files.
        mask_surfix (str): Surfix for the mask files.

    Returns:
        None
    """
    # Load the image
    img_name = f"{FOV}_{source_surfix}.tiff" if source_surfix else f"{FOV}.tiff"
    img = imread(os.path.join(image_source, img_name))
    
    # Load the mask
    mask_name = f"{FOV}_{mask_surfix}.tiff" if mask_surfix else f"{FOV}.tiff"
    mask = imread(os.path.join(mask_source, mask_name))
    
    # Ensure the mask is in 2D
    if mask.ndim == 3:
        mask = mask[-1,:,:]
    
    # Convert original image to RGB if it's grayscale
    if img.ndim == 2:
        img = color.gray2rgb(img)
    
    unique_ids = np.unique(mask)
    # Convert image to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    for cell_id in unique_ids:
        binary_mask = np.uint8(mask == cell_id)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_bgr, contours, -1, color, thickness)
    
    # Convert back to RGB if saving with skimage.io
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Save the modified image
    image_name = f"{FOV}_{mask_surfix}_overlay.png"
    io.imsave(os.path.join(output_dir, image_name), img_rgb)


def overlay_masks(image_source, mask_source, output_dir, source_surfix = None, mask_surfix=None, thickness = 1, color = [255, 255, 255]):
    """
    Overlay masks on the original image for all FOVs by calling overlay_masks_single_FOV().
    Args:
        image_source (str): Source directory for the original image files.
        mask_source (str): Source directory for the mask files.
        mask_surfix (str): Surfix for the mask files.
        output_dir (str): Output directory for the overlay images.
    Returns:
        None
    """
    # Process each TIFF image in image_source
    for filename in os.listdir(image_source):
        if filename.endswith('.tiff'):
            FOV = filename.split('.')[0]
            overlay_masks_single_FOV(FOV, image_source, mask_source, output_dir, source_surfix, mask_surfix, thickness, color)

def plot_color_palette(color_palette, title):
    fig, ax = plt.subplots(1, 1, figsize=(0.6, 0.6*len(color_palette)+0.3),
                           dpi=300, facecolor='w', edgecolor='k')

    for sp in ax.spines.values():
        sp.set_visible(False)

    plt.yticks([])
    plt.xticks([])

    # Reverse the order of the keys and values in the color palette dictionary
    keys = list(color_palette.keys())[::-1]
    values = list(color_palette.values())[::-1]

    # If the values are larger than 1, divide by 255
    # Adjusted to handle RGB values correctly
    corrected_values = []
    for v in values:
        if isinstance(v, (list, tuple)) and np.max(v) > 1:
            corrected_values.append([c/255 for c in v])
        elif np.max(v) > 1:
            corrected_values.append(v/255)
        else:
            corrected_values.append(v)
    values = corrected_values

    ax.barh(keys, [0.1]*len(color_palette), color=values, height=0.8)  # Adjust the height parameter as needed

    # Add the names of the colors to the right of the bars
    for i, key in enumerate(keys):
        ax.text(0.12, i, key, ha='left', va='center', fontsize=16, color='black')

    plt.title(title)
    plt.show()

    return(fig)

def get_axis_limits_with_padding(self, downsample_factor):
    """
    Calculate the axis limits with padding and adjust for downsampling.

    This function retrieves the current axis limits from the image display,
    ensures they are within the bounds of the image dimensions, adds minimal
    padding to ensure at least a minimal area is displayed, and adjusts the
    limits according to the specified downsample factor.

    Parameters:
    self (object): The instance of the class containing image display and dimensions.
    downsample_factor (int): The factor by which the image is downsampled.

    Returns:
    tuple: A tuple containing the following values:
        - xmin (int): The minimum x-axis limit.
        - xmax (int): The maximum x-axis limit.
        - ymin (int): The minimum y-axis limit.
        - ymax (int): The maximum y-axis limit.
        - xmin_ds (int): The minimum x-axis limit adjusted for downsampling.
        - xmax_ds (int): The maximum x-axis limit adjusted for downsampling.
        - ymin_ds (int): The minimum y-axis limit adjusted for downsampling.
        - ymax_ds (int): The maximum y-axis limit adjusted for downsampling.
    """
    # Get current axis limits
    xlim = self.image_display.ax.get_xlim()
    ylim = self.image_display.ax.get_ylim()
    xmin, xmax = int(max(xlim[0], 0)), int(min(xlim[1], self.width))
    ymin, ymax = int(max(ylim[0], 0)), int(min(ylim[1], self.height))

    # Correct for inverted y-axis in images
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    # Add padding to ensure at least a minimal area is displayed
    if xmax - xmin < 1:
        xmax = xmin + 1
    if ymax - ymin < 1:
        ymax = ymin + 1

    

    # Adjust for downsample factor
    xmin_ds = xmin // downsample_factor
    xmin = xmin_ds * downsample_factor
    xmax_ = range(xmin,xmax,downsample_factor)[-1]
    xmax_ds = xmax_ // downsample_factor +1
    
    ymin_ds = ymin // downsample_factor
    ymin = ymin_ds * downsample_factor
    ymax_ = range(ymin,ymax,downsample_factor)[-1]
    ymax_ds = ymax_ // downsample_factor +1

    # Ensure indices are within bounds
    xmin = max(0, xmin)
    xmax = min(self.width, xmax)
    ymin = max(0, ymin)
    ymax = min(self.height, ymax)

    # Downsampled image dimensions
    img_ds_shape = (self.height // downsample_factor, self.width // downsample_factor)
    xmin_ds = max(0, xmin_ds)
    xmax_ds = min(img_ds_shape[1], xmax_ds)
    xmax_ds = range(xmin_ds,xmax_ds,downsample_factor)[-1]
    ymin_ds = max(0, ymin_ds)
    ymax_ds = min(img_ds_shape[0], ymax_ds)
    ymax_ds = range(ymin_ds,ymax_ds,downsample_factor)[-1]

    xmax = xmax_ds * downsample_factor
    ymax = ymax_ds * downsample_factor

    return xmin, xmax, ymin, ymax, xmin_ds, xmax_ds, ymin_ds, ymax_ds