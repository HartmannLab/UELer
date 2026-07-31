"""Tests for drag-reordering the selected-channel chips (issue #126).

``TagsInput`` — the widget ``ChannelPickerWidget`` replaced in #125 — let users drag
its tags to reorder them.  The replacement's chips were static, so the ordering that
``value`` carries could no longer be changed from the UI.  That order is not cosmetic:
:meth:`ImageMaskViewer.update_controls` rebuilds the per-channel colour/contrast rows
straight from ``channel_selector.value``, and channel compositing follows it too.

Three things are covered here:

* **The drop arithmetic** — the ESM's ``reorderSelection`` helper is extracted from the
  widget source and executed under ``node``, so what the browser actually runs is what
  is asserted (skipped when ``node`` is unavailable).
* **The front-end contract** — the chips are draggable, every drag event is wired, the
  drop indicator cannot reflow the row, and a keyboard path exists.
* **The consequence the issue asks for** — "the order should be reflected in the channel
  color and scale UI": a permuted ``value`` reorders ``channel_controls_box.children``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

from ueler.viewer.plugin.channel_picker_widget import (
    ChannelPickerWidget,
    _CSS,
    _ESM,
)

_NODE = shutil.which("node") or shutil.which("nodejs")

_HELPER_START = "// --- reorder helper (#126) ---"
_HELPER_END = "// --- end reorder helper ---"


def _reorder_helper_source() -> str:
    """Return the ESM's ``reorderSelection`` function, verbatim."""
    _, _, rest = _ESM.partition(_HELPER_START)
    body, marker, _ = rest.partition(_HELPER_END)
    if not marker:
        raise AssertionError(
            "the reorder helper markers are missing from _ESM; the node-backed "
            "tests below can no longer reach the code the browser runs"
        )
    return body


def _run_reorder(cases):
    """Execute ``reorderSelection`` under node for every case; return the results.

    Each case is ``(selection, name, target, after)``; ``target`` may be ``None``
    (append).  Running the real ESM source keeps the assertions honest — a Python
    re-implementation could drift from the shipped JavaScript without a test noticing.
    """
    script = (
        _reorder_helper_source()
        + "\nconst cases = "
        + json.dumps(cases)
        + ";\n"
        + "const out = cases.map(function (c) {\n"
        + "  return reorderSelection(c[0], c[1], c[2], c[3]);\n"
        + "});\n"
        + "process.stdout.write(JSON.stringify(out));\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "reorder.mjs"
        path.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            [_NODE, str(path)], capture_output=True, text=True, timeout=60
        )
    if proc.returncode != 0:
        raise AssertionError(f"node failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


@unittest.skipUnless(_NODE, "node is required to execute the widget's ESM helper")
class ReorderArithmeticTestCase(unittest.TestCase):
    """Run the shipped ``reorderSelection`` and check where chips land."""

    SELECTION = ["CD45", "CD3", "CD4", "CD8"]

    def _reorder(self, name, target, after=False, selection=None):
        selection = self.SELECTION if selection is None else selection
        return _run_reorder([[selection, name, target, after]])[0]

    def test_moving_left_drops_before_the_target(self):
        # Drag CD8 onto the left half of CD3.
        self.assertEqual(
            self._reorder("CD8", "CD3", after=False),
            ["CD45", "CD8", "CD3", "CD4"],
        )

    def test_moving_left_onto_the_right_half_drops_after_the_target(self):
        self.assertEqual(
            self._reorder("CD8", "CD3", after=True),
            ["CD45", "CD3", "CD8", "CD4"],
        )

    def test_moving_right_drops_after_the_target(self):
        # The regression a naive implementation gets wrong: resolving the target
        # index *before* splicing the source out lands CD45 one slot too far right
        # (["CD3", "CD4", "CD8", "CD45"]) because removal shifts everything left.
        self.assertEqual(
            self._reorder("CD45", "CD4", after=True),
            ["CD3", "CD4", "CD45", "CD8"],
        )

    def test_moving_right_onto_the_left_half_drops_before_the_target(self):
        self.assertEqual(
            self._reorder("CD45", "CD4", after=False),
            ["CD3", "CD45", "CD4", "CD8"],
        )

    def test_a_null_target_appends(self):
        # Dropping in the empty space after the last chip.
        self.assertEqual(
            self._reorder("CD45", None),
            ["CD3", "CD4", "CD8", "CD45"],
        )

    def test_first_to_last_and_last_to_first_are_both_reachable(self):
        results = _run_reorder(
            [
                [self.SELECTION, "CD45", "CD8", True],
                [self.SELECTION, "CD8", "CD45", False],
            ]
        )
        self.assertEqual(results[0], ["CD3", "CD4", "CD8", "CD45"])
        self.assertEqual(results[1], ["CD8", "CD45", "CD3", "CD4"])

    def test_dropping_a_chip_on_itself_is_a_no_op(self):
        self.assertEqual(self._reorder("CD3", "CD3"), self.SELECTION)

    def test_reordering_never_adds_or_drops_a_channel(self):
        cases = []
        for name in self.SELECTION:
            for target in self.SELECTION + [None]:
                for after in (False, True):
                    cases.append([self.SELECTION, name, target, after])
        for case, result in zip(cases, _run_reorder(cases)):
            self.assertEqual(
                sorted(result), sorted(self.SELECTION), msg=f"case {case}"
            )
            self.assertEqual(len(result), len(set(result)), msg=f"case {case}")

    def test_an_unknown_name_is_ignored(self):
        # Defensive: a stale drag payload must not corrupt the selection.
        self.assertEqual(self._reorder("gone", "CD3"), self.SELECTION)

    def test_an_empty_selection_is_handled(self):
        self.assertEqual(self._reorder("CD3", None, selection=[]), [])


@unittest.skipUnless(_NODE, "node is required to parse the widget's ESM")
class EsmSyntaxTestCase(unittest.TestCase):
    def test_the_esm_is_valid_javascript(self):
        """A syntax error here breaks every picker in the viewer at render time.

        Nothing else catches it: the ESM is an opaque string to Python, and the
        headless test fallback never evaluates it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "picker.mjs"
            path.write_text(_ESM, encoding="utf-8")
            proc = subprocess.run(
                [_NODE, "--check", str(path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr.strip())


class FrontEndContractTestCase(unittest.TestCase):
    """Guard the wiring that makes the chips draggable in the browser."""

    def test_chips_are_draggable_and_carry_their_name(self):
        self.assertIn("chip.draggable = true", _ESM)
        self.assertIn("chip.dataset.name = name", _ESM)

    def test_every_drag_event_is_handled(self):
        for event in ("dragstart", "dragover", "dragleave", "drop", "dragend"):
            self.assertIn("'" + event + "'", _ESM, msg=f"{event} is not wired")

    def test_the_drop_side_follows_the_pointer(self):
        # Without a midpoint test only one drop side is reachable per target.
        self.assertIn("getBoundingClientRect", _ESM)
        self.assertIn("rect.width / 2", _ESM)
        self.assertIn("drop-after", _ESM)
        self.assertIn("drop-before", _ESM)

    def test_the_reorder_is_committed_to_the_value_trait(self):
        # This is what propagates the new order to update_controls.
        self.assertIn("reorderSelection(selection()", _ESM)
        self.assertIn("commit(next)", _ESM)

    def test_the_remove_button_does_not_start_a_drag(self):
        self.assertIn("x.draggable = false", _ESM)

    def test_the_drop_indicator_cannot_reflow_the_chip_row(self):
        # A border/margin marker would resize the chips mid-drag and make the drop
        # target move out from under the pointer; absolute pseudo-elements do not.
        rule = _CSS.split(".ucp-chip.drop-before::before,", 1)[1].split("}", 1)[0]
        self.assertIn("position: absolute", rule)
        self.assertNotIn("border-width", rule)
        self.assertNotIn("margin", rule)

    def test_a_drag_does_not_select_the_chip_text(self):
        chip_rule = _CSS.split(".ucp-chip {", 1)[1].split("}", 1)[0]
        self.assertIn("user-select: none", chip_rule)
        self.assertIn("cursor: grab", chip_rule)

    def test_a_grip_advertises_the_affordance(self):
        self.assertIn("ucp-grip", _CSS)
        self.assertIn("Drag to reorder", _ESM)

    def test_a_keyboard_path_exists(self):
        # Pointer-only reordering is unreachable in hosts that swallow drag events.
        self.assertIn("chip.tabIndex = 0", _ESM)
        self.assertIn("ArrowLeft", _ESM)
        self.assertIn("ArrowRight", _ESM)
        self.assertIn("focusChip", _ESM)

    def test_dropping_past_the_last_chip_appends(self):
        self.assertIn("moveChip(moved, null, false)", _ESM)


class ValueTraitTestCase(unittest.TestCase):
    """The Python side has to accept a permutation as a real change."""

    def test_a_permutation_is_preserved_not_normalised_away(self):
        picker = ChannelPickerWidget(allowed_tags=["CD45", "CD3", "CD4"])
        picker.value = ["CD45", "CD3", "CD4"]
        picker.value = ["CD4", "CD45", "CD3"]
        self.assertEqual(list(picker.value), ["CD4", "CD45", "CD3"])

    def test_a_permutation_notifies_observers(self):
        # ``on_channel_selection_change`` is what rebuilds the colour/scale rows.
        picker = ChannelPickerWidget(allowed_tags=["CD45", "CD3"], value=["CD45", "CD3"])
        seen = []
        picker.observe(lambda change: seen.append(list(change["new"])), names="value")
        picker.value = ["CD3", "CD45"]
        self.assertEqual(seen, [["CD3", "CD45"]])

    def test_allowed_tags_order_does_not_constrain_the_selection_order(self):
        picker = ChannelPickerWidget(allowed_tags=["CD45", "CD3"], value=["CD3", "CD45"])
        self.assertEqual(list(picker.value), ["CD3", "CD45"])


class ChannelControlOrderTestCase(unittest.TestCase):
    """"The order should be reflected in the channel color and scale UI." """

    @staticmethod
    def _viewer(channels):
        """Minimal viewer that can run ``update_controls``.

        Follows the stub in ``tests/test_annotation_palettes.AnnotationLayoutTests``,
        trimmed to the channel section (no masks, no pixel annotations).
        """
        import sys

        from ueler.viewer.main_viewer import ImageMaskViewer

        widgets = sys.modules["ipywidgets"]

        viewer = ImageMaskViewer.__new__(ImageMaskViewer)
        viewer.predefined_colors = {"Red": "#FF0000", "Green": "#00FF00"}
        viewer.current_downsample_factor = 1
        viewer.masks_available = False
        viewer.annotations_available = False
        viewer._control_section_titles = []

        viewer.ui_component = SimpleNamespace(
            image_selector=SimpleNamespace(value="FOV1"),
            channel_selector=SimpleNamespace(value=tuple(channels)),
            color_controls={},
            channel_visibility_controls={},
            contrast_min_controls={},
            contrast_max_controls={},
            channel_controls_box=widgets.VBox(),
            channel_section_panel=widgets.VBox(),
            mask_controls_box=widgets.VBox(),
            annotation_controls_box=widgets.VBox(),
            annotation_palette_tab=widgets.Tab(children=()),
            no_channels_label=widgets.HTML(),
            no_annotations_label=widgets.HTML(),
            empty_controls_placeholder=widgets.HTML(),
        )
        tab_layout = getattr(viewer.ui_component.annotation_palette_tab, "layout", None)
        if tab_layout is None:
            viewer.ui_component.annotation_palette_tab.layout = SimpleNamespace(display="")

        accordion = widgets.Accordion()
        accordion.children = ()
        accordion.selected_index = 0
        accordion.set_title = lambda idx, title: None
        viewer.ui_component.control_sections = accordion

        viewer._ensure_channel_max_computed = lambda *a, **k: None
        viewer._get_channel_stats = lambda channel: (100.0, 0.0)
        viewer._calculate_slider_step = lambda max_value: 0.1
        viewer._slider_readout_format = lambda max_value: ".2f"
        viewer._refresh_annotation_control_states = lambda: None
        viewer.update_display = lambda *a, **k: None
        return viewer

    @staticmethod
    def _row_labels(viewer):
        """Return the marker name shown in each channel-control row, in order."""
        labels = []
        for group in viewer.ui_component.channel_controls_box.children:
            header = group.children[0]
            labels.append(header.children[1].value)
        return labels

    def test_control_rows_follow_the_selection_order(self):
        viewer = self._viewer(["CD45", "CD3", "CD4"])
        viewer.update_controls(None)
        rows = self._row_labels(viewer)
        self.assertEqual(len(rows), 3)
        for expected, rendered in zip(["CD45", "CD3", "CD4"], rows):
            self.assertIn(expected, rendered)

    def test_reordering_the_selection_reorders_the_control_rows(self):
        viewer = self._viewer(["CD45", "CD3", "CD4"])
        viewer.update_controls(None)
        sliders_before = dict(viewer.ui_component.contrast_min_controls)

        # What a chip drag produces: the same channels in a new order.
        viewer.ui_component.channel_selector.value = ("CD4", "CD45", "CD3")
        viewer.update_controls(None)

        for expected, rendered in zip(["CD4", "CD45", "CD3"], self._row_labels(viewer)):
            self.assertIn(expected, rendered)
        # Reordering must not rebuild the controls: a user's contrast settings and
        # colour choices have to survive a reorder.
        self.assertEqual(viewer.ui_component.contrast_min_controls, sliders_before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
