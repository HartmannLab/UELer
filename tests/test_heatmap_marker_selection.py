"""Tests for the heatmap plugin's shared marker-selection UI/UX (issue #117).

The heatmap plugin now reuses the same channel-selector bundle as the scatter and
histogram plugins (``_chart_common.build_channel_selector``): a ``TagsInput`` plus a
"Marker set:" dropdown + "Load set" button. These tests assert that integration —
that ``channel_selector`` is aliased to the bundle's tags widget, and that loading a
predefined marker set populates the picker locally (numeric-only, filtered) without
disturbing the left-panel channel selector.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

import pandas as pd

from ueler.viewer.plugin import _chart_common
from ueler.viewer.plugin.heatmap import UiComponent


def _cell_table() -> "pd.DataFrame":
    return pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov2"],
            "label": [1, 2, 3],
            "intensity": [1.0, 5.0, 9.0],
            "area": [10.0, 20.0, 30.0],
        }
    )


def _make_ui_component():
    """Construct a real heatmap ``UiComponent`` against a mock parent.

    ``UiComponent`` only needs ``parent.main_viewer.cell_table`` (real pandas) and a
    grab-bag of callback attributes for widget observers; a ``MagicMock`` parent
    supplies the callables while we override ``main_viewer`` with a real namespace.
    """
    viewer = SimpleNamespace(
        cell_table=_cell_table(),
        marker_sets={},
        # A left-panel selector we can assert is NOT mutated by loading a set.
        ui_component=SimpleNamespace(channel_selector=SimpleNamespace(value=("untouched",))),
    )
    parent = MagicMock()
    parent.main_viewer = viewer
    ui = UiComponent(parent)
    return ui, viewer


class TestHeatmapChannelSelector(unittest.TestCase):
    def test_channel_selector_is_shared_bundle(self):
        ui, _ = _make_ui_component()
        self.assertIs(ui.channel_selector, ui.channel_selector_bundle.tags)

    def test_available_channels_are_numeric_only(self):
        ui, _ = _make_ui_component()
        # "fov" (object) and "label"/"intensity"/"area" — only numeric survive.
        self.assertEqual(
            sorted(ui.channel_selector_bundle.available),
            ["area", "intensity", "label"],
        )

    def test_load_marker_set_populates_channels_locally(self):
        ui, viewer = _make_ui_component()
        viewer.marker_sets = {"T cells": {"selected_channels": ["intensity", "area"]}}
        bundle = ui.channel_selector_bundle
        _chart_common.refresh_marker_set_options(bundle, viewer)
        bundle.marker_set_dropdown.value = "T cells"

        _chart_common.apply_marker_set_to_selector(bundle, viewer)
        self.assertEqual(list(bundle.tags.value), ["intensity", "area"])
        # Loading a set into the plugin must not disturb the left-panel selector.
        self.assertEqual(viewer.ui_component.channel_selector.value, ("untouched",))

    def test_load_marker_set_filters_unknown_and_non_numeric(self):
        ui, viewer = _make_ui_component()
        viewer.marker_sets = {
            "mixed": {"selected_channels": ["intensity", "does_not_exist", "fov"]}
        }
        bundle = ui.channel_selector_bundle
        _chart_common.refresh_marker_set_options(bundle, viewer)
        bundle.marker_set_dropdown.value = "mixed"

        _chart_common.apply_marker_set_to_selector(bundle, viewer)
        # "fov" (object) and the absent channel are filtered out.
        self.assertEqual(list(bundle.tags.value), ["intensity"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
