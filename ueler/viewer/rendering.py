"""Compatibility wrapper for the shared rendering engine."""

from __future__ import annotations

from ueler.rendering.engine import (  # noqa: F401 - re-export for legacy imports
    AnnotationOverlaySnapshot,
    AnnotationRenderSettings,
    ChannelRenderSettings,
    MaskOverlaySnapshot,
    MaskRenderSettings,
    OverlaySnapshot,
    find_boundaries,
    render_crop_to_array,
    render_fov_to_array,
    render_roi_to_array,
    _binary_dilation_4,
    _label_boundaries,
)

__all__ = [
    "AnnotationOverlaySnapshot",
    "AnnotationRenderSettings",
    "ChannelRenderSettings",
    "MaskOverlaySnapshot",
    "MaskRenderSettings",
    "OverlaySnapshot",
    "find_boundaries",
    "render_crop_to_array",
    "render_fov_to_array",
    "render_roi_to_array",
    "_binary_dilation_4",
    "_label_boundaries",
]
