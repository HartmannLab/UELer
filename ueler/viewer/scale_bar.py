"""Utility helpers for computing and rendering scale bars.

Phase 4 introduces a shared helper so both the interactive viewer and the
batch export pipeline can present consistent scale bars sized to the rendered
image and pixel spacing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

try:  # pragma: no cover - modules may be stubbed in test environments
    from matplotlib.font_manager import FontProperties  # type: ignore
except Exception:  # pragma: no cover - allow lazy Matplotlib usage
    FontProperties = None  # type: ignore

try:  # pragma: no cover - modules may be stubbed in test environments
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar  # type: ignore
except Exception:  # pragma: no cover - allow lazy Matplotlib usage
    AnchoredSizeBar = None  # type: ignore

__all__ = [
    "ScaleBarSpec",
    "add_scale_bar",
    "compute_scale_bar_spec",
    "effective_pixel_size_nm",
]


@dataclass(frozen=True)
class ScaleBarSpec:
    """Describes the geometry and label for a rendered scale bar."""

    pixel_length: float
    physical_length_um: float
    label: str


def effective_pixel_size_nm(pixel_size_nm: float, downsample: int) -> float:
    """Return the physical size covered by one rendered pixel in nanometres."""

    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    return float(pixel_size_nm) * float(downsample)


def compute_scale_bar_spec(
    *,
    image_width_px: int,
    pixel_size_nm: float,
    max_fraction: float = 0.1,
    rounding_sequence: Sequence[float] = (1.0, 2.0, 5.0),
) -> ScaleBarSpec:
    """Compute a tidy scale bar specification for the given image.

    Parameters
    ----------
    image_width_px:
        Width of the rendered image (in pixels) that will host the scale bar.
    pixel_size_nm:
        Physical size represented by a single rendered pixel (nanometres).
    max_fraction:
        Maximum fraction of the image width that the scale bar should occupy.
    rounding_sequence:
        Base magnitudes to consider when rounding the bar length (defaults to
        ``1, 2, 5`` style engineering increments).

    Returns
    -------
    ScaleBarSpec
        Data class describing the chosen bar length (in pixels and microns)
        along with a formatted label suitable for display.
    """

    if image_width_px <= 0:
        raise ValueError("image_width_px must be > 0")
    if pixel_size_nm <= 0:
        raise ValueError("pixel_size_nm must be > 0")
    if max_fraction <= 0:
        raise ValueError("max_fraction must be > 0")
    if not rounding_sequence:
        raise ValueError("rounding_sequence must not be empty")

    pixel_size_um = float(pixel_size_nm) / 1000.0
    max_pixels = float(image_width_px) * float(max_fraction)
    if max_pixels <= 0:
        raise ValueError("Computed maximum pixel span must be > 0")

    max_physical_um = max_pixels * pixel_size_um
    if max_physical_um <= 0:
        raise ValueError("Computed maximum physical size must be > 0")

    candidate = _select_candidate_length(max_physical_um, rounding_sequence)
    pixel_length = max(candidate / pixel_size_um, 1.0)
    label_value, label_unit = _format_length(candidate)
    label = f"{label_value} {label_unit}"
    return ScaleBarSpec(pixel_length=pixel_length, physical_length_um=candidate, label=label)


def add_scale_bar(
    ax,
    spec: ScaleBarSpec,
    *,
    loc: str = "lower right",
    color: str = "white",
    pad: float = 0.5,
    size_vertical: float = 2.0,
    font_size: float = 12.0,
    frameon: bool = False,
):
    """Attach an anchored scale bar to the supplied Matplotlib axes."""

    if AnchoredSizeBar is None or FontProperties is None:
        raise RuntimeError("Matplotlib scale bar support is unavailable in this environment")

    fontprops = FontProperties(size=font_size)
    bar = AnchoredSizeBar(
        ax.transData,
        spec.pixel_length,
        spec.label,
        loc,
        pad=pad,
        color=color,
        frameon=frameon,
        size_vertical=size_vertical,
        fontproperties=fontprops,
    )
    ax.add_artist(bar)
    return bar


def _select_candidate_length(max_physical_um: float, rounding_sequence: Sequence[float]) -> float:
    max_physical_um = float(max_physical_um)
    exponent = math.floor(math.log10(max_physical_um)) if max_physical_um > 0 else 0
    for offset in range(6):
        current_exponent = exponent - offset
        for base in sorted(rounding_sequence, reverse=True):
            candidate = base * (10 ** current_exponent)
            if candidate <= max_physical_um and candidate > 0:
                return candidate
    # Fallback to the maximum available value if all rounded options exceed the limit
    return max_physical_um


def _format_length(length_um: float) -> tuple[str, str]:
    abs_length = abs(length_um)
    if abs_length >= 1000.0:
        value = length_um / 1000.0
        unit = "mm"
    elif abs_length >= 1.0:
        value = length_um
        unit = "µm"
    else:
        value = length_um * 1000.0
        unit = "nm"

    abs_value = abs(value)
    if abs_value >= 100:
        formatted = f"{value:.0f}"
    elif abs_value >= 10:
        formatted = f"{value:.1f}"
    elif abs_value >= 1:
        formatted = f"{value:.2f}"
    elif abs_value >= 0.1:
        formatted = f"{value:.3f}"
    else:
        formatted = f"{value:.4f}"

    formatted = formatted.rstrip("0").rstrip(".") if "." in formatted else formatted
    return formatted, unit
