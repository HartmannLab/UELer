"""Grouping and timing rules of the live latency timer."""

import unittest

from ueler.viewer.latency_timer import format_summary, summarise_interactions


def _in(t, kind):
    return {"t": t, "type": "input", "kind": kind}


def _frame(t, kb=100.0, **extra):
    return {"t": t, "type": "frame", "kb": kb, **extra}


class DragTests(unittest.TestCase):
    def test_drag_steps_and_release_frame(self):
        events = [
            _in(0.000, "press"),
            _in(0.010, "drag"), _in(0.020, "drag"), _frame(0.050, 150),
            _in(0.060, "drag"), _frame(0.090, 160),
            _in(0.100, "drag"), _frame(0.140, 170),
            _in(0.150, "release"), _frame(0.350, 1000),
        ]
        (it,) = summarise_interactions(events)
        self.assertEqual(it["kind"], "drag")
        self.assertEqual(it["drag_frames"], 3)
        self.assertEqual(it["drag_frame_interval_ms"], 45.0)  # 40 and 50 ms
        self.assertEqual(it["drag_lag_ms"], 40.0)  # 40, 30, 40 ms from the oldest unanswered move
        self.assertEqual(it["drag_frame_kb"], 160.0)
        self.assertEqual(it["release_to_frame_ms"], 200.0)
        self.assertEqual(it["release_frame_kb"], 1000)

    def test_release_without_a_frame(self):
        events = [_in(0.0, "press"), _in(0.01, "drag"), _frame(0.04), _in(0.05, "release")]
        (it,) = summarise_interactions(events)
        self.assertIsNone(it["release_to_frame_ms"])


class GroupingTests(unittest.TestCase):
    def test_next_start_input_separates_interactions(self):
        events = [_in(0.0, "widget"), _frame(0.3), _in(5.0, "widget"), _frame(5.5, 50), _frame(5.6, 200)]
        first, second = summarise_interactions(events)
        self.assertEqual((first["kind"], first["first_frame_ms"], first["last_frame_ms"]), ("widget", 300.0, 300.0))
        self.assertEqual((second["frames"], second["first_frame_ms"], second["last_frame_ms"]), (2, 500.0, 600.0))

    def test_a_slow_sharp_frame_still_belongs_to_its_drag(self):
        # On a slow link the full-size frame lands seconds after the release.
        events = [_in(0.0, "press"), _in(0.05, "drag"), _frame(0.4, 220), _in(0.5, "release"),
                  _frame(0.9, 230), _frame(6.0, 2500), _frame(6.3, 3)]
        (it,) = summarise_interactions(events)
        self.assertEqual(it["release_to_frame_ms"], 5500.0)  # the largest frame, not the tooltip after it
        self.assertEqual(it["release_frame_kb"], 2500)

    def test_bursts_of_widget_or_scroll_inputs_count_once(self):
        events = [_in(0.0, "scroll"), _in(0.1, "scroll"), _in(0.2, "scroll"), _frame(0.5), _in(3.0, "widget"), _in(3.2, "widget"), _frame(3.6)]
        scroll, widget = summarise_interactions(events)
        self.assertEqual((scroll["kind"], scroll["last_frame_ms"]), ("scroll", 500.0))
        self.assertEqual((widget["kind"], widget["first_frame_ms"]), ("widget", 600.0))

    def test_frames_before_any_input_or_long_after_are_ignored(self):
        events = [_frame(0.0), _frame(0.2), _in(3.0, "scroll"), _frame(3.1), _frame(100.0, 5000)]
        (it,) = summarise_interactions(events)
        self.assertEqual((it["kind"], it["frames"], it["last_frame_ms"]), ("scroll", 1, 100.0))

    def test_unsorted_events_are_ordered_first(self):
        events = [_frame(0.2), _in(0.0, "widget")]
        (it,) = summarise_interactions(events)
        self.assertEqual(it["first_frame_ms"], 200.0)


class SummaryTests(unittest.TestCase):
    def test_medians_per_kind(self):
        its = summarise_interactions([
            _in(0.0, "widget"), _frame(0.2, render_ms=50.0, encode_ms=20.0),
            _in(5.0, "widget"), _frame(5.4, render_ms=70.0, encode_ms=30.0),
        ])
        text = format_summary(its, "### header")
        self.assertIn("Interactions: widget × 2", text)
        self.assertIn("| Widget change: input → first frame | 300 ms |", text)
        self.assertIn("| Render per frame | 60 ms |", text)
        self.assertNotIn("Drag:", text)


if __name__ == "__main__":
    unittest.main()
