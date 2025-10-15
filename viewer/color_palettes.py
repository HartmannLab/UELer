"""Shared helpers for palette management across viewer components."""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from matplotlib.colors import to_rgb

DEFAULT_COLOR = "#A0A0A0"


def normalize_hex_color(color: str | None) -> str:
    if not isinstance(color, str):
        return ""
    value = color.strip()
    if not value:
        return ""
    if not value.startswith("#"):
        value = f"#{value}"
    return value.lower()


def colors_match(a: str | None, b: str | None) -> bool:
    return normalize_hex_color(a) == normalize_hex_color(b)


def ensure_palette_keys(
    palette: Mapping[str | int, str],
    default_color: str = DEFAULT_COLOR,
) -> "OrderedDict[str, str]":
    resolved: "OrderedDict[str, str]" = OrderedDict()
    for key, value in palette.items():
        normalized_key = str(key)
        normalized_value = normalize_hex_color(value) or default_color
        resolved[normalized_key] = normalized_value
    return resolved


def apply_color_defaults(
    class_ids: Sequence[int],
    palette: Mapping[str | int, str],
    default_color: str = DEFAULT_COLOR,
) -> "OrderedDict[str, str]":
    resolved = ensure_palette_keys(palette, default_color=default_color)
    ordered: "OrderedDict[str, str]" = OrderedDict()
    for class_id in class_ids:
        key = str(int(class_id))
        ordered[key] = resolved.get(key, default_color)
    return ordered


def hex_to_rgb_unit(color: str) -> np.ndarray:
    return np.array(to_rgb(color or DEFAULT_COLOR), dtype=np.float32)


def build_discrete_colormap(
    class_ids: Iterable[int],
    palette: Mapping[str | int, str],
    default_color: str = DEFAULT_COLOR,
) -> np.ndarray:
    ids = sorted({int(c) for c in class_ids if c is not None})
    if not ids:
        return np.tile(hex_to_rgb_unit(default_color), (1, 1))

    max_id = max(ids)
    table = np.tile(hex_to_rgb_unit(default_color), (max_id + 1, 1))
    resolved = ensure_palette_keys(palette, default_color=default_color)
    for class_id in ids:
        key = str(class_id)
        table[class_id] = hex_to_rgb_unit(resolved.get(key, default_color))
    return table


def merge_palette_updates(
    target: MutableMapping[str, str],
    updates: Mapping[str, str],
    default_color: str = DEFAULT_COLOR,
) -> None:
    for key, value in updates.items():
        target[str(key)] = normalize_hex_color(value) or default_color
