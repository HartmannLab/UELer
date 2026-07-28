"""Tests for the searchable, scrollable marker/feature picker (issue #125).

The marker/feature pickers used to be ipywidgets ``TagsInput`` widgets, which expose
their options through a native ``<datalist>`` popup: browser-drawn, height-capped,
unscrollable and clipped in embedded notebook hosts, so long marker lists were
unusable.  ``ChannelPickerWidget`` replaces it with an in-DOM list we control.

These tests cover
  * the Python-side contract that keeps it a drop-in ``TagsInput`` replacement
    (``value`` / ``allowed_tags``, observers, forgiving normalisation),
  * the front-end contract that actually fixes the bug (a height-capped,
    ``overflow-y: auto`` option list and no ``<datalist>`` anywhere), and
  * the wiring of every marker/feature picker in the viewer and its plugins.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

import pandas as pd

from ueler.viewer.plugin import _chart_common
from ueler.viewer.plugin.channel_picker_widget import (
    DEFAULT_LIST_MAX_HEIGHT,
    ChannelPickerWidget,
    _CSS,
    _ESM,
    _normalise_options,
    _normalise_selection,
    build_channel_picker,
)


def _cell_table() -> "pd.DataFrame":
    return pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov2"],
            "label": [1, 2, 3],
            "CD45": [1.0, 5.0, 9.0],
            "CD3": [10.0, 20.0, 30.0],
            "area": [4.0, 5.0, 6.0],
        }
    )


def _viewer(marker_sets=None):
    return SimpleNamespace(
        cell_table=_cell_table(),
        marker_sets=marker_sets or {},
        ui_component=SimpleNamespace(channel_selector=SimpleNamespace(value=("untouched",))),
    )


class NormalisationTestCase(unittest.TestCase):
    def test_options_are_stringified_and_deduplicated(self):
        self.assertEqual(_normalise_options(["b", "a", "b"]), ["b", "a"])
        self.assertEqual(_normalise_options((1, 2, 1)), ["1", "2"])
        self.assertEqual(_normalise_options("CD45"), ["CD45"])
        self.assertEqual(_normalise_options(None), [])

    def test_selection_is_filtered_against_allowed_options(self):
        self.assertEqual(_normalise_selection(["a", "zz"], ["a", "b"]), ["a"])

    def test_selection_keeps_order_and_drops_duplicates(self):
        self.assertEqual(_normalise_selection(("b", "a", "b"), ["a", "b"]), ["b", "a"])

    def test_selection_is_untouched_when_no_options_are_known_yet(self):
        # ``TagsInput`` behaved the same way: an empty ``allowed_tags`` allows anything.
        self.assertEqual(_normalise_selection(["a"], []), ["a"])


class ChannelPickerWidgetTestCase(unittest.TestCase):
    def test_value_and_allowed_tags_round_trip(self):
        picker = ChannelPickerWidget(allowed_tags=["CD45", "CD3"], value=["CD3"])
        self.assertEqual(list(picker.allowed_tags), ["CD45", "CD3"])
        self.assertEqual(list(picker.value), ["CD3"])

    def test_tuple_assignment_is_accepted(self):
        # ``main_viewer.on_image_change`` assigns tuples.
        picker = ChannelPickerWidget(allowed_tags=["CD45", "CD3"])
        picker.value = ("CD45",)
        self.assertEqual(list(picker.value), ["CD45"])

    def test_bare_string_assignment_is_wrapped(self):
        # ``run_flowsom`` seeds the picker with a single column name.
        picker = ChannelPickerWidget(allowed_tags=["CD45"], value="CD45")
        self.assertEqual(list(picker.value), ["CD45"])

    def test_unknown_names_are_dropped_instead_of_raising(self):
        # ``TagsInput`` raised TraitError here, which broke checkpoint restores of
        # marker lists whose columns no longer exist.
        picker = ChannelPickerWidget(allowed_tags=["CD45"])
        picker.value = ("CD45", "gone")
        self.assertEqual(list(picker.value), ["CD45"])

    def test_duplicates_are_collapsed(self):
        picker = ChannelPickerWidget(allowed_tags=["CD45", "CD3"])
        picker.value = ["CD3", "CD3", "CD45"]
        self.assertEqual(list(picker.value), ["CD3", "CD45"])

    def test_non_string_option_labels_are_coerced(self):
        picker = ChannelPickerWidget(allowed_tags=[1, 2])
        picker.value = [1]
        self.assertEqual(list(picker.allowed_tags), ["1", "2"])
        self.assertEqual(list(picker.value), ["1"])

    def test_value_observers_fire(self):
        picker = ChannelPickerWidget(allowed_tags=["CD45", "CD3"])
        seen = []
        picker.observe(lambda change: seen.append(list(change["new"])), names="value")
        picker.value = ["CD3"]
        self.assertEqual(seen, [["CD3"]])

    def test_defaults(self):
        picker = ChannelPickerWidget()
        self.assertEqual(list(picker.value), [])
        self.assertEqual(list(picker.allowed_tags), [])
        self.assertFalse(picker.disabled)
        self.assertEqual(int(picker.list_max_height), DEFAULT_LIST_MAX_HEIGHT)


class FrontEndContractTestCase(unittest.TestCase):
    """Guard the properties that actually fix #125 in the browser."""

    def test_no_native_datalist_is_used(self):
        # The whole point: the option list is ours, not a browser popup.
        self.assertNotIn("datalist", _ESM.lower())

    def test_option_list_is_scrollable_and_height_capped(self):
        list_rule = _CSS.split(".ucp-list {", 1)[1].split("}", 1)[0]
        self.assertIn("overflow-y: auto", list_rule)
        self.assertIn("maxHeight", _ESM)
        self.assertIn("list_max_height", _ESM)

    def test_option_list_is_in_flow_not_a_floating_overlay(self):
        # An in-flow panel cannot be clipped by an ancestor overflow, a stacking
        # context, or an iframe/host boundary — unlike a positioned overlay.
        panel_rule = _CSS.split(".ucp-panel {", 1)[1].split("}", 1)[0]
        self.assertNotIn("position:", panel_rule)
        self.assertNotIn("z-index", panel_rule)

    def test_every_option_is_reachable_from_the_filter_box(self):
        self.assertIn("toLowerCase().indexOf(q)", _ESM)      # substring filter
        self.assertIn("Select all shown", _ESM)              # bulk-select filtered
        self.assertIn("shown", _ESM)                         # "N of M shown" footer

    def test_keyboard_navigation_is_wired(self):
        for key in ("ArrowDown", "ArrowUp", "Enter", "Escape", "Backspace"):
            self.assertIn(key, _ESM)
        self.assertIn("scrollIntoView", _ESM)

    def test_python_side_changes_are_reflected(self):
        for trait in ("value", "allowed_tags", "disabled", "list_max_height"):
            self.assertIn("change:" + trait, _ESM)

    def test_document_listener_is_cleaned_up(self):
        self.assertIn("removeEventListener('mousedown'", _ESM)


class BuildChannelPickerTestCase(unittest.TestCase):
    def test_returns_the_picker_widget(self):
        picker = build_channel_picker(allowed_tags=["CD45"], value=["CD45"])
        self.assertIsInstance(picker, ChannelPickerWidget)

    def test_tags_input_only_kwargs_are_ignored(self):
        picker = build_channel_picker(
            allowed_tags=["CD45"],
            allow_duplicates=False,
            style={"description_width": "auto"},
        )
        self.assertIsInstance(picker, ChannelPickerWidget)

    def test_value_is_normalised_before_construction(self):
        picker = build_channel_picker(allowed_tags=["CD45", "CD3"], value="CD3")
        self.assertEqual(list(picker.value), ["CD3"])

    def test_layout_is_forwarded(self):
        layout = SimpleNamespace(width="100%")
        picker = build_channel_picker(allowed_tags=[], layout=layout)
        self.assertIs(picker.layout, layout)

    def test_layout_defaults_to_a_mutable_placeholder(self):
        picker = build_channel_picker(allowed_tags=[])
        picker.layout.width = "50%"  # must not raise
        self.assertEqual(picker.layout.width, "50%")

    def test_description_and_placeholder_are_stored(self):
        picker = build_channel_picker(
            allowed_tags=[], description="Channels:", placeholder="filter..."
        )
        self.assertEqual(picker.description, "Channels:")
        self.assertEqual(picker.placeholder, "filter...")


class ChartCommonIntegrationTestCase(unittest.TestCase):
    """The Scatter plot, Histogram and Heatmap plugins share this bundle."""

    def test_bundle_tags_is_the_new_picker(self):
        bundle = _chart_common.build_channel_selector(_viewer())
        self.assertIsInstance(bundle.tags, ChannelPickerWidget)

    def test_options_are_the_numeric_columns(self):
        viewer = _viewer()
        bundle = _chart_common.build_channel_selector(viewer)
        expected = _chart_common.numeric_columns(viewer)
        self.assertEqual(list(bundle.tags.allowed_tags), expected)
        self.assertEqual(bundle.available, expected)

    def test_starts_empty(self):
        bundle = _chart_common.build_channel_selector(_viewer())
        self.assertEqual(list(bundle.tags.value), [])

    def test_marker_set_loading_still_populates_the_picker(self):
        viewer = _viewer(
            marker_sets={"panel": {"selected_channels": ["CD3", "fov", "CD45"]}}
        )
        bundle = _chart_common.build_channel_selector(viewer)
        _chart_common.refresh_marker_set_options(bundle, viewer)
        bundle.marker_set_dropdown.value = "panel"
        _chart_common.apply_marker_set_to_selector(bundle, viewer)
        # Non-numeric ``fov`` is filtered out; order is preserved.
        self.assertEqual(list(bundle.tags.value), ["CD3", "CD45"])
        # The left-panel selector must not be touched by a plugin-local load.
        self.assertEqual(viewer.ui_component.channel_selector.value, ("untouched",))


class PluginWiringTestCase(unittest.TestCase):
    def test_heatmap_channel_selector_uses_the_picker(self):
        from ueler.viewer.plugin.heatmap import UiComponent

        parent = MagicMock()
        parent.main_viewer = _viewer()
        ui = UiComponent(parent)
        self.assertIs(ui.channel_selector, ui.channel_selector_bundle.tags)
        self.assertIsInstance(ui.channel_selector, ChannelPickerWidget)

    def test_left_panel_channel_selector_uses_the_picker(self):
        from ueler.viewer import ui_components

        self.assertIs(ui_components.build_channel_picker, build_channel_picker)

    def test_flowsom_feature_picker_uses_the_picker(self):
        from ueler.viewer.plugin import run_flowsom

        self.assertIs(run_flowsom.build_channel_picker, build_channel_picker)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
