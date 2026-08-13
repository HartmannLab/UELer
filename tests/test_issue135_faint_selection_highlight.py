"""A selection too small to see as a bar tints its bins instead (#135 reply).

``Follow main viewer`` (#135) puts the cells clicked in the image into the
histogram's "Selected" distribution.  That overlay shares the y-axis with the
"All" bars, so a handful of cells among tens of thousands is drawn a fraction of
a pixel high and the user sees nothing happen.  Below a visibility threshold the
plugin therefore switches encoding: every bin that holds selected cells is tinted
over its **full** height, which answers "where are my cells" — the only question
a five-cell selection can answer.

These tests cover the threshold rule as pure arithmetic and then its effect on
real Bokeh sources, including that the honest overlay is never altered to
achieve the visibility.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

import numpy as np
import pandas as pd

# Reuses the stub widgets / viewer fakes the histogram suite already installs.
from tests.test_histogram_plugin import (  # noqa: E402
    _bokeh_available,
    _make_histogram,
    _make_viewer,
)

from ueler.viewer.plugin.histogram import (  # noqa: E402
    _FAINT_FRACTION,
    _faint_selection_tops,
)


def _peaked_table(n: int = 1000) -> "pd.DataFrame":
    """A table whose histogram has a real peak — where the problem actually bites.

    A *uniform* channel cannot reproduce the bug: spread 1000 cells over 50 bins
    and a single selected cell is already 5% of a 20-cell bar. Real marker
    distributions are peaked, so here 900 of the 1000 cells pile into the low
    range (tallest ``intensity`` bar: 360) and 100 trail off into a sparse tail.
    One cell out of the peak is 0.3% of that bar — the #135 case.

    ``area`` deliberately bins differently (10 flat values, 100 cells each) so
    one selection can be faint on one channel and readable on the other.
    """
    peak_size = 900
    return pd.DataFrame(
        {
            "fov": ["fov1"] * n,
            "label": list(range(1, n + 1)),
            "intensity": np.concatenate(
                [np.linspace(0.0, 5.0, peak_size), np.linspace(50.0, 100.0, n - peak_size)]
            ),
            "area": [float(i % 10) for i in range(n)],
        }
    )


class FaintSelectionRuleTestCase(unittest.TestCase):
    """The threshold rule, as arithmetic — no Bokeh, no widgets."""

    def test_faint_selection_marks_exactly_the_occupied_bins(self):
        full = np.array([100, 200, 300, 400])
        counts = np.array([0, 1, 0, 2])  # peak 2 vs 400 → 0.5% of the tallest bar
        self.assertEqual(_faint_selection_tops(counts, full).tolist(), [0, 200, 0, 400])

    def test_marked_bins_get_the_full_bar_height(self):
        """The point of the tint: one selected cell colours the whole bar."""
        full = np.array([1000, 1000])
        counts = np.array([1, 0])
        self.assertEqual(_faint_selection_tops(counts, full).tolist(), [1000, 0])

    def test_a_readable_selection_is_left_alone(self):
        """At or above the threshold the proportional overlay speaks for itself."""
        full = np.array([100, 200])
        counts = np.array([0, int(_FAINT_FRACTION * 200)])  # exactly 5% of the peak
        self.assertEqual(_faint_selection_tops(counts, full).tolist(), [0, 0])

    def test_just_below_the_threshold_is_marked(self):
        full = np.array([100, 200])
        counts = np.array([0, int(_FAINT_FRACTION * 200) - 1])
        self.assertEqual(_faint_selection_tops(counts, full).tolist(), [0, 200])

    def test_empty_selection_marks_nothing(self):
        full = np.array([100, 200, 300])
        self.assertEqual(_faint_selection_tops(np.zeros(3, dtype=int), full).tolist(), [0, 0, 0])

    def test_empty_data_is_safe(self):
        """No bars at all must not divide by (or compare against) an empty peak."""
        self.assertEqual(_faint_selection_tops(np.array([]), np.array([])).tolist(), [])
        self.assertEqual(
            _faint_selection_tops(np.zeros(2, dtype=int), np.zeros(2, dtype=int)).tolist(),
            [0, 0],
        )

    def test_visibility_is_measured_on_bar_heights_not_cell_counts(self):
        """A small *share of the cells* concentrated in one bin needs no help.

        3% of the cells all in one bin out of a flat 100-bin histogram makes a bar
        three times the height of its neighbours — visible, so untinted. Counting
        cells instead of bar heights would have tinted it (see the docstring of
        ``_faint_selection_tops``).
        """
        full = np.full(100, 100)
        counts = np.zeros(100, dtype=int)
        counts[7] = 300  # 3% of 10 000 cells, but 3x the tallest bar
        self.assertEqual(_faint_selection_tops(counts, full).tolist(), [0] * 100)


class FaintSelectionOverlayTestCase(unittest.TestCase):
    """The rule as it reaches the Bokeh sources the figures are drawn from."""

    def setUp(self):
        if not _bokeh_available():
            self.skipTest("bokeh not available in this environment")
        self.viewer = _make_viewer(_peaked_table())
        self.hist = _make_histogram(self.viewer)
        self.hist._plot_data = self.viewer.cell_table.copy()
        self.hist._channels = ["intensity", "area"]
        _layout, self.hist._sources, self.hist._spans = self.hist._build_figures()

    def _tops(self, channel: str, key: str):
        return list(self.hist._sources[channel][key].data["top"])

    def test_a_single_cell_selection_tints_one_bin(self):
        """The #135 case: one cell clicked in the image is now findable."""
        self.hist.show_external_selection([0], push_highlight=False)
        tinted = [t for t in self._tops("intensity", "hits") if t]
        self.assertEqual(len(tinted), 1)
        # Tinted to the full height of the bar it sits under (360 cells), not to
        # its own count of 1 — which is the whole point.
        self.assertEqual(tinted[0], 360)

    def test_the_proportional_overlay_still_reports_the_true_count(self):
        """Visibility is bought with a second glyph, never by inflating the first."""
        self.hist.show_external_selection([0], push_highlight=False)
        self.assertEqual(sum(self._tops("intensity", "selected")), 1)

    def test_a_large_selection_is_not_tinted(self):
        """Half the cells read fine as a distribution; tinting would just hide it."""
        self.hist.show_external_selection(range(500), push_highlight=False)
        self.assertEqual(sum(self._tops("intensity", "hits")), 0)
        self.assertEqual(sum(self._tops("intensity", "selected")), 500)

    def test_the_decision_is_taken_per_channel(self):
        """One selection, faint on one channel and readable on the other.

        These ten rows share an ``area`` value, so on that channel they are ten
        cells in a 100-cell bar — 10%, readable, left alone. The same ten sit
        inside ``intensity``'s 360-cell peak, where they are 2.8% and invisible,
        so only that channel is tinted.
        """
        self.hist.show_external_selection(range(0, 100, 10), push_highlight=False)
        self.assertEqual(sum(self._tops("intensity", "hits")), 360)
        self.assertEqual(sum(self._tops("area", "hits")), 0)
        # Both channels still report the same ten cells honestly.
        self.assertEqual(sum(self._tops("intensity", "selected")), 10)
        self.assertEqual(sum(self._tops("area", "selected")), 10)

    def test_unticking_the_checkbox_drops_the_tint(self):
        self.hist.show_external_selection([0], push_highlight=False)
        self.assertGreater(sum(self._tops("intensity", "hits")), 0)
        self.hist.ui_component.faint_highlight_checkbox.value = False
        self.hist._on_faint_highlight_change(SimpleNamespace(new=False))
        self.assertEqual(sum(self._tops("intensity", "hits")), 0)
        # The selection itself is untouched — only how it is drawn changed.
        self.assertEqual(sum(self._tops("intensity", "selected")), 1)

    def test_reticking_the_checkbox_restores_the_tint(self):
        self.hist.ui_component.faint_highlight_checkbox.value = False
        self.hist.show_external_selection([0], push_highlight=False)
        self.assertEqual(sum(self._tops("intensity", "hits")), 0)
        self.hist.ui_component.faint_highlight_checkbox.value = True
        self.hist._on_faint_highlight_change(SimpleNamespace(new=True))
        self.assertGreater(sum(self._tops("intensity", "hits")), 0)

    def test_toggling_the_checkbox_never_replots(self):
        """Same invariant as every other selection-side update (#127)."""
        self.hist._render_calls = 0
        self.hist.show_external_selection([0], push_highlight=False)
        self.hist._on_faint_highlight_change(SimpleNamespace(new=True))
        self.assertEqual(self.hist._render_calls, 0)

    def test_clearing_the_selection_clears_the_tint(self):
        self.hist.show_external_selection([0], push_highlight=False)
        self.hist.clear_selection()
        self.assertEqual(sum(self._tops("intensity", "hits")), 0)

    def test_a_gate_selection_is_tinted_on_the_same_rule(self):
        """The tint is a property of the overlay, not of where the selection came from."""
        self.hist.handle_range("intensity", 0.0, 0.02)  # 4 of 1000 cells
        self.assertGreater(sum(self._tops("intensity", "hits")), 0)

    def test_the_checkbox_is_observed(self):
        """``_wire_events`` really registers the handler, so toggling acts at once.

        The stub widgets swallow ``observe``, so the registration is checked by
        substituting a recorder — the same shape as the #129 link-checkbox test.
        """
        recorded: list = []

        class _Recorder:
            value = True

            def observe(self, callback, names=None):
                recorded.append((callback, names))

        self.hist.ui_component.faint_highlight_checkbox = _Recorder()
        self.hist._wire_events()
        self.assertIn((self.hist._on_faint_highlight_change, "value"), recorded)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
