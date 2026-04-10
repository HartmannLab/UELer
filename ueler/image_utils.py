"""Image utility helpers used by the packaged viewer code.

This module is the canonical home for image-processing helpers that were
historically imported from the repository root as ``image_utils``.
"""

from __future__ import annotations

import math
import os

import numpy as np
import tifffile
from dask import delayed
from skimage.segmentation import find_boundaries

from ueler.rendering.engine import scale_outline_thickness, thicken_outline


def generate_edges(mask, *, thickness: int = 1, downsample: int = 1):
	"""Generate mask outlines with optional thickness scaling for downsampling."""

	effective = scale_outline_thickness(thickness, downsample)
	iterations = max(0, int(effective) - 1)

	def _compute_edges(mask_array, extra_iterations):
		edges = find_boundaries(mask_array, mode="inner")
		if extra_iterations <= 0:
			return edges
		return thicken_outline(edges, extra_iterations)

	return delayed(_compute_edges)(mask, iterations)


def calculate_downsample_factor(width, height, ignore_zoom=False, max_dimension=512):
	"""Return the smallest power-of-two factor that fits within ``max_dimension``."""

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
	"""Return the nearest permitted downsample factor for the given image size."""

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


def crop_image_center(tif, x, y, crop_width):
	"""Crop a square region centered on ``(x, y)`` from a TIFF page or array."""

	if isinstance(tif, tifffile.TiffFile):
		page = tif.pages[0]
	else:
		page = tif

	img_width, img_height = page.shape
	img = np.zeros((crop_width, crop_width))

	halfwidth = np.round(crop_width / 2).astype(int)
	x1 = max(0, x - halfwidth)
	y1 = max(0, y - halfwidth)
	x2 = min(img_width, x + halfwidth)
	y2 = min(img_height, y + halfwidth)

	offset_x1 = max(0, halfwidth - x)
	offset_y1 = max(0, halfwidth - y)
	offset_x2 = crop_width - max(0, (x + halfwidth) - img_width)
	offset_y2 = crop_width - max(0, (y + halfwidth) - img_height)

	if isinstance(page, tifffile.TiffPage):
		img[offset_x1:offset_x2, offset_y1:offset_y2] = page.asarray()[x1:x2, y1:y2]
	else:
		img[offset_x1:offset_x2, offset_y1:offset_y2] = page[x1:x2, y1:y2]

	return img


def process_single_crop(
	FOV,
	marker2display,
	crop_position,
	crop_width,
	file_source="",
	subfolder="",
	mask_suffix=None,
	mask_source="",
	mask_ID=None,
	image_viewer=None,
):
	"""Load and crop the requested FOV channels, optionally including mask outlines."""

	img_stack = None
	x, y = crop_position

	for channel in marker2display.keys():
		path_fov = os.path.join(file_source, FOV)
		fname = os.path.join(path_fov, subfolder, channel + ".tiff")

		try:
			if image_viewer is None:
				with tifffile.TiffFile(fname) as tif:
					img = crop_image_center(tif, x, y, crop_width)
			else:
				image_viewer.load_fov(FOV, [channel])
				tif = image_viewer.image_cache[FOV][channel]
				img = crop_image_center(tif, x, y, crop_width)
		except FileNotFoundError as exc:
			raise FileNotFoundError(f"Image file '{fname}' not found.") from exc

		if img_stack is None:
			img_stack = img
		else:
			img_stack = np.dstack((img_stack, img))

	if mask_suffix is not None:
		fname = os.path.join(mask_source, f"{FOV}_{mask_suffix}.tiff")
		try:
			if image_viewer is None:
				with tifffile.TiffFile(fname) as tif:
					img = crop_image_center(tif, x, y, crop_width)
					img_bound = find_boundaries(img, mode="thick")
			else:
				tif = image_viewer.mask_cache[FOV][mask_suffix]
				img = crop_image_center(tif, x, y, crop_width)
				img_bound = find_boundaries(img, mode="thick")
		except FileNotFoundError as exc:
			raise FileNotFoundError(f"Image file '{fname}' not found.") from exc

		img_stack = np.dstack((img_stack, img_bound))

		if mask_ID is not None:
			img = np.array(img, copy=True)
			img[img != mask_ID] = 0
			img = find_boundaries(img, mode="thick")
			img_stack = np.dstack((img_stack, img))

	return img_stack


def estimate_color_range(img_stack, channels, pt=99, est_lb=False):
	"""Estimate per-channel lower and upper display bounds from an image stack."""

	color_range = {}
	for index, channel in enumerate(channels):
		img = img_stack[:, :, index].flatten()
		img = img[img < 255]
		img = img[img > 0]

		if est_lb:
			color_range[channel] = [np.percentile(img, 1), np.percentile(img, pt)]
		else:
			color_range[channel] = [0, np.percentile(img, pt)]
	return color_range


def color_one_image(image_stack, marker2display, color_range):
	"""Combine a multi-channel stack into a single RGB image using channel colors."""

	img_colored = None
	for index, channel in enumerate(marker2display.keys()):
		img = image_stack[:, :, index]
		lb, ub = color_range[channel]

		img = img / ub
		img[img > 1] = 1

		if lb > 0:
			img = img - np.percentile(img, lb)
			img[img < 0] = 0
			img = img / np.max(img)

		channel_rgb = np.atleast_3d(img) * marker2display[channel]
		if img_colored is None:
			img_colored = channel_rgb
		else:
			img_colored = img_colored + channel_rgb

	img_colored = img_colored - np.min(img_colored)
	img_colored[img_colored > 1] = 1
	return (img_colored * 255).astype(np.uint8)


def get_axis_limits_with_padding(self, downsample_factor):
	"""Return padded axis limits aligned to the requested downsample factor."""

	width = max(1, int(getattr(self, "width", 1)))
	height = max(1, int(getattr(self, "height", 1)))
	ds = max(1, int(downsample_factor or 1))

	ax = getattr(self.image_display, "ax", None)
	if ax is None:
		return 0, width, 0, height, 0, math.ceil(width / ds), 0, math.ceil(height / ds)

	xlim = getattr(ax, "get_xlim", lambda: (0.0, float(width)))()
	ylim = getattr(ax, "get_ylim", lambda: (0.0, float(height)))()

	xmin_raw, xmax_raw = float(min(xlim)), float(max(xlim))
	ymin_raw, ymax_raw = float(min(ylim)), float(max(ylim))

	xmin = max(0, math.floor(xmin_raw))
	xmax = min(width, math.ceil(xmax_raw))
	if xmax <= xmin:
		xmax = min(width, xmin + 1)

	ymin = max(0, math.floor(ymin_raw))
	ymax = min(height, math.ceil(ymax_raw))
	if ymax <= ymin:
		ymax = min(height, ymin + 1)

	xmin_ds = max(0, xmin // ds)
	xmax_ds = math.ceil(xmax / ds)
	width_ds = max(1, math.ceil(width / ds))
	xmax_ds = min(width_ds, max(xmin_ds + 1, xmax_ds))

	ymin_ds = max(0, ymin // ds)
	ymax_ds = math.ceil(ymax / ds)
	height_ds = max(1, math.ceil(height / ds))
	ymax_ds = min(height_ds, max(ymin_ds + 1, ymax_ds))

	xmin_aligned = xmin_ds * ds
	xmax_aligned = min(width, xmax_ds * ds)
	ymin_aligned = ymin_ds * ds
	ymax_aligned = min(height, ymax_ds * ds)

	return (
		xmin_aligned,
		xmax_aligned,
		ymin_aligned,
		ymax_aligned,
		xmin_ds,
		xmax_ds,
		ymin_ds,
		ymax_ds,
	)


__all__ = [
	"calculate_downsample_factor",
	"color_one_image",
	"crop_image_center",
	"estimate_color_range",
	"generate_edges",
	"get_axis_limits_with_padding",
	"process_single_crop",
	"select_downsample_factor",
]