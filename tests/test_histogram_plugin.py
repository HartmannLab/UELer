"""Tests for the standalone Histogram plugin (issue #112).

Covers feature-parity with the old histogram (cutoff → above/below highlight →
cell-gallery forwarding) plus the new linked-brushing behaviour (a range brush
selects cells and feeds ``selected_indices``, which drives the cross-histogram
overlay and the cell gallery).
"""
from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

# ---------------------------------------------------------------------------
# Lightweight stubs for optional UI / scientific dependencies
# ---------------------------------------------------------------------------
try:
    import ipywidgets as _ipywidgets  # type: ignore
except ImportError:  # pragma: no cover - stub fallback
    _ipywidgets = types.ModuleType("ipywidgets")
    sys.modules["ipywidgets"] = _ipywidgets


class _Layout:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Widget:
    def __init__(self, *args, **kwargs):
        children = kwargs.get("children")
        if children is None and args:
            children = args[0]
        self.children = tuple(children or ())
        self.value = kwargs.get("value")
        self.options = kwargs.get("options", [])
        self.layout = kwargs.get("layout", _Layout())
        self.description = kwargs.get("description", "")
        self._observers: list = []

    def observe(self, callback, names=None):
        self._observers.append((callback, names))

    def _trigger(self, new_value):
        self.value = new_value
        for cb, _ in self._observers:
            cb(SimpleNamespace(new=new_value))

    def on_click(self, *_, **__):
        return None

    def clear_output(self, *_, **__):
        return None

    def set_title(self, *_, **__):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


for _wname in [
    "Button", "Checkbox", "Dropdown", "FloatSlider", "HBox", "HTML",
    "IntSlider", "IntText", "Layout", "Output", "SelectMultiple", "Tab",
    "TagsInput", "Text", "ToggleButtons", "VBox", "Widget",
]:
    if not hasattr(_ipywidgets, _wname):
        setattr(_ipywidgets, _wname, _Widget)

if not hasattr(_ipywidgets, "Layout"):
    _ipywidgets.Layout = _Layout  # type: ignore[attr-defined]

for _mod in ["seaborn_image", "tifffile", "cv2", "dask"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

import numpy as np
import pandas as pd

from ueler.viewer.plugin.histogram import HistogramDisplay


def _bokeh_available() -> bool:
    """True when Bokeh is importable (enough to build a figure/layout)."""
    from ueler.viewer.plugin import histogram as _h

    return bool(_h._BOKEH_OK)


def _bokeh_stack_available() -> bool:
    """True when both bokeh and jupyter_bokeh are importable (full interactive render)."""
    from ueler.viewer.plugin import histogram as _h

    return bool(_h._BOKEH_OK and _h._JBOKEH_OK)


# ---------------------------------------------------------------------------
# Minimal viewer stub (mirrors test_chart_cell_gallery_link)
# ---------------------------------------------------------------------------

class _FakeImageDisplay:
    def __init__(self):
        self.last_mask_ids: list = []
        self.last_fov_mask_pairs = None
        # Distinguishes "never asked to highlight" from "asked to clear" (#129).
        self.set_mask_ids_calls = 0

    def set_mask_ids(self, *, mask_name, mask_ids, fov_mask_pairs=None):
        self.set_mask_ids_calls += 1
        # Mirror the real ImageDisplay: replacing the highlight drops the
        # FOV-independent record, which `sync_mask_highlights_from_selection`
        # re-arms straight after when the selection is non-empty (#119).
        viewer = getattr(self, "main_viewer", None)
        if viewer is not None:
            viewer.linked_selection_indices = None
        if fov_mask_pairs is not None:
            self.last_fov_mask_pairs = list(fov_mask_pairs)
            self.last_mask_ids = []
        else:
            self.last_mask_ids = list(mask_ids)
            self.last_fov_mask_pairs = None


class _FakeCellGallery:
    def __init__(self):
        self.received: object = None

    def set_selected_cells(self, indices):
        self.received = indices


def _make_viewer(cell_table: "pd.DataFrame") -> SimpleNamespace:
    gallery = _FakeCellGallery()
    image_display = _FakeImageDisplay()
    ui_component = SimpleNamespace(image_selector=SimpleNamespace(value="fov1"))
    side_plots = SimpleNamespace(cell_gallery_output=gallery)
    viewer = SimpleNamespace(
        cell_table=cell_table,
        fov_key="fov",
        label_key="label",
        mask_key="cells",
        ui_component=ui_component,
        image_display=image_display,
        SidePlots=side_plots,
        linked_selection_indices=None,
        get_active_fov=lambda: ui_component.image_selector.value,
    )
    image_display.main_viewer = viewer
    return viewer


def _make_histogram(viewer: SimpleNamespace, *, patch_render=True) -> HistogramDisplay:
    hist = HistogramDisplay(viewer, width=4, height=3)
    hist.setup_observe()
    if patch_render:
        # Decouple selection logic from real Matplotlib figure building.
        hist._render_calls = 0

        def _fake_render():
            hist._render_calls += 1

        hist._render = _fake_render
    return hist


def _two_fov_table() -> "pd.DataFrame":
    return pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov1", "fov2", "fov2"],
            "label": [1, 2, 3, 4, 5],
            "intensity": [1.0, 5.0, 9.0, 3.0, 7.0],
            "area": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


# ---------------------------------------------------------------------------
# Cutoff-mode parity tests
# ---------------------------------------------------------------------------

class TestHistogramCutoffGalleryLink(unittest.TestCase):
    """highlight_cells() → selected_indices → cell gallery forwarding (parity)."""

    def setUp(self):
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer)
        self.gallery: _FakeCellGallery = self.viewer.SidePlots.cell_gallery_output
        self.hist._active_histogram_column = "intensity"
        self.hist.ui_component.above_below_buttons.value = "above"
        self.hist.ui_component.cell_gallery_linked_checkbox.value = True
        self.hist.cutoff = 4.0

    def test_gallery_receives_all_fov_indices_above_cutoff(self):
        self.hist.highlight_cells(push_to_gallery=True)
        self.assertIsNotNone(self.gallery.received)
        table = self.viewer.cell_table
        expected = set(table.loc[table["intensity"] > 4.0].index)
        self.assertEqual(set(self.gallery.received), expected)

    def test_gallery_not_updated_when_checkbox_off(self):
        self.hist.ui_component.cell_gallery_linked_checkbox.value = False
        self.hist.highlight_cells(push_to_gallery=True)
        self.assertIsNone(self.gallery.received)

    def test_gallery_receives_indices_below_cutoff(self):
        self.hist.ui_component.above_below_buttons.value = "below"
        self.hist.highlight_cells(push_to_gallery=True)
        table = self.viewer.cell_table
        expected = set(table.loc[table["intensity"] < 4.0].index)
        self.assertEqual(set(self.gallery.received), expected)

    def test_image_display_limited_to_current_fov(self):
        # Highlighting the viewer requires the "Main viewer" link (#129).
        self.hist.ui_component.mv_linked_checkbox.value = True
        self.hist.highlight_cells(push_to_gallery=True)
        img = self.viewer.image_display
        self.assertNotIn(1, img.last_mask_ids)
        self.assertIn(2, img.last_mask_ids)
        self.assertIn(3, img.last_mask_ids)
        for label in (4, 5):
            self.assertNotIn(label, img.last_mask_ids)

    def test_map_mode_uses_fov_mask_pairs(self):
        self.hist.ui_component.mv_linked_checkbox.value = True
        self.viewer.get_active_fov = lambda: None
        self.hist.highlight_cells(push_to_gallery=True)
        pairs = self.viewer.image_display.last_fov_mask_pairs
        self.assertIsNotNone(pairs)
        # intensity > 4.0 → labels 2,3 (fov1) and 5 (fov2)
        self.assertIn(("fov1", 2), pairs)
        self.assertIn(("fov2", 5), pairs)

    def test_no_crash_when_cutoff_is_none(self):
        self.hist.cutoff = None
        self.hist.highlight_cells(push_to_gallery=True)
        self.assertIsNone(self.gallery.received)

    def test_no_crash_when_no_active_channel(self):
        self.hist._active_histogram_column = None
        self.hist.highlight_cells(push_to_gallery=True)
        self.assertIsNone(self.gallery.received)

    def test_auto_rerender_does_not_update_gallery(self):
        self.gallery.received = {99}
        self.hist.highlight_cells(push_to_gallery=False)
        self.assertEqual(self.gallery.received, {99})


# ---------------------------------------------------------------------------
# Brush-mode / linked-selection tests
# ---------------------------------------------------------------------------

class TestHistogramBrushLinking(unittest.TestCase):
    def setUp(self):
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer)
        self.gallery: _FakeCellGallery = self.viewer.SidePlots.cell_gallery_output
        self.hist._plot_data = self.viewer.cell_table.copy()
        self.hist._channels = ["intensity", "area"]

    def test_cells_in_range_returns_indices_within_bounds(self):
        idx = self.hist._cells_in_range("intensity", 4.0, 8.0)
        table = self.viewer.cell_table
        expected = set(table.loc[table["intensity"].between(4.0, 8.0)].index)
        self.assertEqual(idx, expected)

    def test_cells_in_range_handles_reversed_bounds(self):
        forward = self.hist._cells_in_range("intensity", 4.0, 8.0)
        reversed_ = self.hist._cells_in_range("intensity", 8.0, 4.0)
        self.assertEqual(forward, reversed_)

    def test_brush_publishes_selected_indices(self):
        self.hist._on_brush("intensity", 4.0, 8.0)
        table = self.viewer.cell_table
        expected = set(table.loc[table["intensity"].between(4.0, 8.0)].index)
        self.assertEqual(set(self.hist.selected_indices.value), expected)

    def test_brush_forwards_to_gallery_when_linked(self):
        self.hist.ui_component.cell_gallery_linked_checkbox.value = True
        # 3 cells in range so the single-point guard does not suppress forwarding.
        self.hist._on_brush("area", 15.0, 55.0)
        self.assertIsNotNone(self.gallery.received)
        self.assertEqual(set(self.gallery.received), {1, 2, 3, 4})

    def test_brush_highlights_viewer_when_mv_linked(self):
        self.hist.ui_component.mv_linked_checkbox.value = True
        self.hist._on_brush("intensity", 4.0, 10.0)
        # fov1 labels with intensity in [4,10]: labels 2 (5.0), 3 (9.0)
        self.assertIn(2, self.viewer.image_display.last_mask_ids)
        self.assertIn(3, self.viewer.image_display.last_mask_ids)

    def test_handle_range_is_the_brush_alias(self):
        """`_on_brush` delegates to the public `handle_range` (same effect)."""
        self.hist._on_brush("intensity", 4.0, 8.0)
        via_alias = set(self.hist.selected_indices.value)
        self.hist.clear_selection()
        self.hist.handle_range("intensity", 4.0, 8.0)
        self.assertEqual(set(self.hist.selected_indices.value), via_alias)

    def test_clear_selection_empties_indices(self):
        self.hist._on_brush("intensity", 4.0, 8.0)
        self.assertTrue(self.hist.selected_indices.value)
        self.hist.clear_selection()
        self.assertEqual(self.hist.selected_indices.value, set())

    def test_show_external_selection_publishes_indices(self):
        """A linked plugin (e.g. heatmap #114) can push a selection in."""
        self.hist.show_external_selection([2, 4])
        self.assertEqual(set(self.hist.selected_indices.value), {2, 4})

    def test_show_external_selection_forwards_to_gallery_when_linked(self):
        self.hist.ui_component.cell_gallery_linked_checkbox.value = True
        self.hist.show_external_selection([1, 2, 3])
        self.assertIsNotNone(self.gallery.received)
        self.assertEqual(set(self.gallery.received), {1, 2, 3})

    def test_show_external_selection_highlights_viewer_when_mv_linked(self):
        self.hist.ui_component.mv_linked_checkbox.value = True
        # Rows 1 and 2 are both in fov1, with labels 2 and 3.
        self.hist.show_external_selection([1, 2])
        self.assertIn(2, self.viewer.image_display.last_mask_ids)
        self.assertIn(3, self.viewer.image_display.last_mask_ids)

    @unittest.skipUnless(_bokeh_available(), "bokeh not available")
    def test_show_external_selection_drives_overlay(self):
        """The pushed selection is drawn as the 'Selected' overlay distribution."""
        layout, sources, spans = self.hist._build_figures()
        self.hist._sources, self.hist._spans = sources, spans
        # Two valid row indices → two overlaid cells across the bins.
        self.hist.show_external_selection([2, 4])
        total = sum(sources["intensity"]["selected"].data["top"])
        self.assertEqual(total, 2)

    def test_bin_counts_matches_numpy_histogram(self):
        from ueler.viewer.plugin.histogram import bin_counts

        edges = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        counts = bin_counts(self.viewer.cell_table["intensity"], edges)
        expected, _ = np.histogram(self.viewer.cell_table["intensity"], bins=edges)
        self.assertTrue(np.array_equal(counts, expected))

    @unittest.skipUnless(_bokeh_available(), "bokeh not available")
    def test_overlay_source_counts_match_selection(self):
        """After a brush, each figure's 'selected' source counts the selected cells."""
        layout, sources, spans = self.hist._build_figures()
        self.hist._sources, self.hist._spans = sources, spans
        self.hist.handle_range("intensity", 4.0, 8.0)
        # intensity in [4,8] → rows with 5.0 and 7.0 → 2 cells.
        total = sum(sources["intensity"]["selected"].data["top"])
        self.assertEqual(total, 2)

    def test_bin_edges_span_full_data_range(self):
        """Edges cover the full column range and have bins+1 entries (#112 reply)."""
        edges = self.hist._histogram_bin_edges("intensity", 20)
        col = self.viewer.cell_table["intensity"]
        self.assertEqual(len(edges), 21)
        self.assertLessEqual(edges[0], col.min())
        self.assertGreaterEqual(edges[-1], col.max())

    def test_bin_edges_independent_of_selection(self):
        """The overlay must reuse the full-data edges, not the subset's own range.

        Regression for the #112 reply: a narrow subset selection must not change
        the bin grid, otherwise the overlay is squeezed into its own bins and is
        not comparable to the full histogram.
        """
        before = self.hist._histogram_bin_edges("intensity", 15)
        # Select a narrow subset, then recompute — edges must be unchanged.
        self.hist._on_brush("intensity", 4.5, 5.5)
        after = self.hist._histogram_bin_edges("intensity", 15)
        self.assertTrue(np.array_equal(before, after))


# ---------------------------------------------------------------------------
# "Main viewer" link gates every highlight push (#129)
# ---------------------------------------------------------------------------

class TestHistogramMainViewerLink(unittest.TestCase):
    """With the link off, nothing this plugin does may highlight the main viewer.

    Cutoff mode used to push highlights unconditionally (``highlight=True`` in
    ``highlight_cells``), and unchecking the box left the previous outlines on the
    canvas — both read as "the histogram still affects the main viewer" (#129).
    """

    def setUp(self):
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer)
        self.img: _FakeImageDisplay = self.viewer.image_display
        self.hist._plot_data = self.viewer.cell_table.copy()
        self.hist._channels = ["intensity", "area"]

    def _set_cutoff(self, value=4.0, direction="above"):
        self.hist._active_histogram_column = "intensity"
        self.hist.ui_component.above_below_buttons.value = direction
        self.hist.cutoff = value

    def _toggle_link(self, value: bool):
        """Flip the checkbox the way a user click does.

        The shared ipywidgets stub ignores ``observe`` registrations, so the
        observer is invoked explicitly — ``_on_mv_link_change`` reads the
        checkbox rather than the change payload, so this is the same code path
        the real widget takes. ``test_link_checkbox_is_observed`` covers the
        wiring itself.
        """
        self.hist.ui_component.mv_linked_checkbox.value = value
        self.hist._on_mv_link_change(SimpleNamespace(name="value", new=value))

    # -- cutoff mode ---------------------------------------------------------
    def test_cutoff_does_not_highlight_when_unlinked(self):
        self._set_cutoff()
        self.hist.highlight_cells(push_to_gallery=True)
        # The gate is live (published) but the viewer was never touched.
        self.assertTrue(self.hist.selected_indices.value)
        self.assertEqual(self.img.set_mask_ids_calls, 0)

    def test_cutoff_highlights_when_linked(self):
        self.hist.ui_component.mv_linked_checkbox.value = True
        self._set_cutoff()
        self.hist.highlight_cells(push_to_gallery=True)
        self.assertIn(2, self.img.last_mask_ids)
        self.assertIn(3, self.img.last_mask_ids)

    def test_above_below_flip_does_not_highlight_when_unlinked(self):
        self._set_cutoff()
        self.hist.highlight_cells(push_to_gallery=True)
        self.hist.ui_component.above_below_buttons.value = "below"
        self.hist._on_above_below_change(None)
        self.assertEqual(self.img.set_mask_ids_calls, 0)

    def test_fov_reapply_does_not_highlight_when_unlinked(self):
        """The viewer re-triggers ``highlight_cells()`` on every FOV change (#119)."""
        self._set_cutoff()
        self.hist.highlight_cells(push_to_gallery=True)
        self.hist.highlight_cells()  # what ImageMaskViewer calls after a FOV switch
        self.assertEqual(self.img.set_mask_ids_calls, 0)

    def test_brush_does_not_highlight_when_unlinked(self):
        self.hist.handle_range("intensity", 4.0, 10.0)
        self.assertTrue(self.hist.selected_indices.value)
        self.assertEqual(self.img.set_mask_ids_calls, 0)

    # -- toggling the checkbox ----------------------------------------------
    def test_unchecking_withdraws_the_highlight(self):
        self.hist.ui_component.mv_linked_checkbox.value = True
        self.hist.handle_range("intensity", 4.0, 10.0)
        self.assertTrue(self.img.last_mask_ids)
        self._toggle_link(False)
        self.assertEqual(self.img.last_mask_ids, [])
        # The FOV-independent record is dropped too, so a FOV switch cannot
        # resurrect the outlines (#119).
        self.assertIsNone(getattr(self.viewer, "linked_selection_indices", None))

    def test_rechecking_restores_the_highlight(self):
        self.hist.handle_range("intensity", 4.0, 10.0)
        self.assertEqual(self.img.set_mask_ids_calls, 0)
        self._toggle_link(True)
        self.assertIn(2, self.img.last_mask_ids)
        self.assertIn(3, self.img.last_mask_ids)

    def test_toggling_without_a_selection_is_a_no_op(self):
        """Toggling an idle histogram must not wipe another plugin's highlight."""
        self._toggle_link(True)
        self._toggle_link(False)
        self.assertEqual(self.img.set_mask_ids_calls, 0)

    def test_link_checkbox_is_observed(self):
        """``_wire_events`` really registers the handler on the checkbox (#129)."""
        recorded: list = []

        class _Recorder:
            value = False

            def observe(self, callback, names=None):
                recorded.append((callback, names))

        self.hist.ui_component.mv_linked_checkbox = _Recorder()
        self.hist._wire_events()
        self.assertIn(
            (self.hist._on_mv_link_change, "value"),
            recorded,
        )


# ---------------------------------------------------------------------------
# Multi-channel gating: ranges AND cutoffs intersect, per-histogram state (#127)
# ---------------------------------------------------------------------------

class TestHistogramGating(unittest.TestCase):
    """Every gated channel contributes a term; the selection is their intersection.

    Reference table (row index → intensity / area):
        0 → 1.0 / 10.0   (fov1)
        1 → 5.0 / 20.0   (fov1)
        2 → 9.0 / 30.0   (fov1)
        3 → 3.0 / 40.0   (fov2)
        4 → 7.0 / 50.0   (fov2)
    """

    def setUp(self):
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer)
        self.gallery: _FakeCellGallery = self.viewer.SidePlots.cell_gallery_output
        self.hist._plot_data = self.viewer.cell_table.copy()
        self.hist._channels = ["intensity", "area"]

    def _set_cutoff(self, channel, value, direction="above"):
        self.hist._active_histogram_column = channel
        self.hist.cutoff = value
        self.hist.ui_component.above_below_buttons.value = direction
        self.hist.highlight_cells(push_to_gallery=True)

    # -- intersection across channels ---------------------------------------
    def test_two_brushes_intersect_instead_of_replacing(self):
        self.hist.handle_range("intensity", 4.0, 8.0)      # {1, 4}
        self.assertEqual(set(self.hist.selected_indices.value), {1, 4})
        self.hist.handle_range("area", 15.0, 35.0)         # {1, 2}
        self.assertEqual(set(self.hist.selected_indices.value), {1})

    def test_two_cutoffs_intersect(self):
        self._set_cutoff("intensity", 4.0, "above")        # {1, 2, 4}
        self._set_cutoff("area", 35.0, "below")            # {0, 1, 2}
        self.assertEqual(set(self.hist.selected_indices.value), {1, 2})

    def test_range_and_cutoff_intersect(self):
        self.hist.handle_range("intensity", 4.0, 8.0)      # {1, 4}
        self._set_cutoff("area", 45.0, "above")            # {4}
        self.assertEqual(set(self.hist.selected_indices.value), {4})

    def test_cutoff_then_range_on_other_channel_keeps_both_terms(self):
        self._set_cutoff("intensity", 4.0, "above")        # {1, 2, 4}
        self.hist.handle_range("area", 15.0, 35.0)         # {1, 2}
        self.assertEqual(set(self.hist.selected_indices.value), {1, 2})
        self.assertEqual(set(self.hist._gates), {"intensity", "area"})

    def test_gated_indices_matches_published_selection(self):
        self.hist.handle_range("intensity", 4.0, 8.0)
        self._set_cutoff("area", 45.0, "above")
        self.assertEqual(
            set(self.hist.gated_indices()), set(self.hist.selected_indices.value)
        )

    # -- per-channel replacement -------------------------------------------
    def test_rebrushing_replaces_only_that_channels_term(self):
        self.hist.handle_range("intensity", 4.0, 8.0)
        self.hist.handle_range("area", 15.0, 35.0)
        self.hist.handle_range("intensity", 0.0, 10.0)     # widen intensity only
        # area's [15, 35] term still gates → {1, 2}
        self.assertEqual(set(self.hist.selected_indices.value), {1, 2})

    def test_brush_supersedes_a_cutoff_on_the_same_channel(self):
        self._set_cutoff("intensity", 4.0, "above")        # {1, 2, 4}
        self.hist.handle_range("intensity", 0.0, 2.0)      # {0}
        self.assertEqual(set(self.hist.selected_indices.value), {0})
        self.assertEqual(self.hist._gates["intensity"][0], "range")
        self.assertIsNone(self.hist.cutoff)

    def test_above_below_toggle_flips_only_the_last_cutoff(self):
        self._set_cutoff("intensity", 4.0, "above")        # {1, 2, 4}
        self._set_cutoff("area", 35.0, "below")            # {0, 1, 2} → ∩ {1, 2}
        # Flipping the toggle re-applies the *area* cutoff as "above" → {3, 4}.
        # The observer is invoked directly so the test holds with either the real
        # ipywidgets stack or the stub widgets.
        self.hist.ui_component.above_below_buttons.value = "above"
        self.hist._on_above_below_change(None)
        self.assertEqual(set(self.hist.selected_indices.value), {4})
        self.assertEqual(self.hist._gates["intensity"], ("cutoff", "above", 4.0))

    # -- clearing ----------------------------------------------------------
    def test_clear_gate_drops_one_term_and_keeps_the_rest(self):
        self.hist.handle_range("intensity", 4.0, 8.0)
        self.hist.handle_range("area", 15.0, 35.0)
        self.hist.clear_gate("area")
        self.assertEqual(set(self.hist.selected_indices.value), {1, 4})
        self.assertEqual(set(self.hist._gates), {"intensity"})

    def test_clear_gate_on_an_ungated_channel_is_a_noop(self):
        self.hist.handle_range("intensity", 4.0, 8.0)
        self.hist.clear_gate("area")
        self.assertEqual(set(self.hist.selected_indices.value), {1, 4})

    def test_clear_selection_empties_all_gates_and_cutoff_state(self):
        self.hist.handle_range("intensity", 4.0, 8.0)
        self._set_cutoff("area", 45.0, "above")
        self.hist.clear_selection()
        self.assertEqual(self.hist._gates, {})
        self.assertIsNone(self.hist.cutoff)
        self.assertIsNone(self.hist._active_histogram_column)
        self.assertEqual(self.hist.selected_indices.value, set())

    def test_replotting_without_a_channel_drops_its_term(self):
        self.hist.handle_range("intensity", 4.0, 8.0)
        self.hist.handle_range("area", 15.0, 35.0)
        self.hist.ui_component.channel_selector.value = ("intensity",)
        self.hist.plot_histograms(None)
        self.assertEqual(set(self.hist._gates), {"intensity"})
        self.assertEqual(set(self.hist.selected_indices.value), {1, 4})

    def test_external_selection_replaces_the_gate(self):
        self.hist.handle_range("intensity", 4.0, 8.0)
        self.hist.show_external_selection([0])
        self.assertEqual(self.hist._gates, {})
        self.assertEqual(set(self.hist.selected_indices.value), {0})

    # -- "don't refresh on selection" --------------------------------------
    def test_gating_never_replots(self):
        """Selection touches sources/annotations only — never a figure rebuild (#127)."""
        self.hist._render_calls = 0
        self.hist.handle_range("intensity", 4.0, 8.0)
        self._set_cutoff("area", 45.0, "above")
        self.hist.clear_gate("area")
        self.hist.clear_selection()
        self.assertEqual(self.hist._render_calls, 0)

    # -- subset consistency ------------------------------------------------
    def test_cutoff_gate_respects_the_plotted_subset(self):
        """A cutoff is evaluated on the plotted frame, like a brush (#127)."""
        table = self.viewer.cell_table
        self.hist._plot_data = table.loc[table["fov"] == "fov1"].copy()
        self._set_cutoff("intensity", 4.0, "above")
        # fov1 rows above 4.0 → {1, 2}; row 4 (fov2, 7.0) is outside the subset.
        self.assertEqual(set(self.hist.selected_indices.value), {1, 2})

    def test_cutoff_falls_back_to_the_cell_table_before_any_plot(self):
        self.hist._plot_data = None
        self._set_cutoff("intensity", 4.0, "above")
        self.assertEqual(set(self.hist.selected_indices.value), {1, 2, 4})

    # -- readability -------------------------------------------------------
    def test_gate_description_lists_every_term(self):
        self.hist.handle_range("intensity", 4.0, 8.0)
        self._set_cutoff("area", 45.0, "above")
        text = self.hist.gate_description()
        self.assertIn("intensity ∈ [4, 8]", text)
        self.assertIn("area > 45", text)
        self.assertIn("AND", text)

    def test_gate_description_when_nothing_is_gated(self):
        self.assertIn("No gate", self.hist.gate_description())


# ---------------------------------------------------------------------------
# Multi-channel plot state
# ---------------------------------------------------------------------------

class TestHistogramMultiChannel(unittest.TestCase):
    def setUp(self):
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer)

    def test_plot_histograms_records_selected_channels(self):
        self.hist.ui_component.channel_selector.value = ("intensity", "area")
        self.hist.plot_histograms(None)
        self.assertEqual(self.hist._channels, ["intensity", "area"])
        self.assertIsNotNone(self.hist._plot_data)
        self.assertGreater(self.hist._render_calls, 0)

    def test_plot_histograms_no_channels_is_noop(self):
        self.hist.ui_component.channel_selector.value = ()
        self.hist.plot_histograms(None)
        self.assertEqual(self.hist._channels, [])

    def test_ensure_bokehjs_is_noop_outside_a_kernel(self):
        """Preloading BokehJS must not raise or mark loaded when there's no IPython kernel."""
        from ueler.viewer.plugin import histogram as h

        h._bokehjs_loaded = False
        h._ensure_bokehjs()  # unit tests have no interactive kernel → no-op
        self.assertFalse(h._bokehjs_loaded)

    def test_scroll_height_kicks_in_only_when_stack_is_tall(self):
        """`_scroll_height` returns a fixed px height once the stack exceeds the cap.

        The scroll is applied to the BokehModel in `_render`; ipywidgets 8 removed
        the per-axis overflow traits, and a `max-height` on the parent VBox does not
        clip the Bokeh column, so the height must live on the model itself (#112 reply 2).
        """
        from ueler.viewer.plugin.histogram import (
            _FIGURE_HEIGHT, _MAX_PLOT_HEIGHT, _ROW_OVERHEAD,
        )

        per = _FIGURE_HEIGHT + _ROW_OVERHEAD
        few = max(1, _MAX_PLOT_HEIGHT // per)          # fits within the cap
        many = (_MAX_PLOT_HEIGHT // per) + 2           # exceeds the cap

        self.hist._channels = ["c%d" % i for i in range(few)]
        self.assertIsNone(self.hist._scroll_height())

        self.hist._channels = ["c%d" % i for i in range(many)]
        self.assertEqual(self.hist._scroll_height(), f"{_MAX_PLOT_HEIGHT}px")


class TestHistogramChannelSelector(unittest.TestCase):
    """Left-panel-consistent channel selector + marker-set loading (#113)."""

    def setUp(self):
        self.viewer = _make_viewer(_two_fov_table())
        # A left-panel channel selector we can assert is NOT mutated by loading.
        self.viewer.ui_component.channel_selector = SimpleNamespace(value=("untouched",))
        self.viewer.marker_sets = {}
        self.hist = _make_histogram(self.viewer)

    def test_channel_selector_is_shared_bundle(self):
        bundle = self.hist.ui_component.channel_selector_bundle
        self.assertIs(self.hist.ui_component.channel_selector, bundle.tags)

    def test_load_marker_set_populates_channels_locally(self):
        self.viewer.marker_sets = {
            "T cells": {"selected_channels": ["intensity", "area"]}
        }
        self.hist.on_marker_sets_changed()
        bundle = self.hist.ui_component.channel_selector_bundle
        bundle.marker_set_dropdown.value = "T cells"
        from ueler.viewer.plugin import _chart_common

        _chart_common.apply_marker_set_to_selector(bundle, self.viewer)
        self.assertEqual(list(bundle.tags.value), ["intensity", "area"])
        # Loading a set into the plugin must not disturb the left-panel selector.
        self.assertEqual(self.viewer.ui_component.channel_selector.value, ("untouched",))

    def test_load_marker_set_filters_unknown_channels(self):
        self.viewer.marker_sets = {
            "mixed": {"selected_channels": ["intensity", "does_not_exist", "fov"]}
        }
        self.hist.on_marker_sets_changed()
        bundle = self.hist.ui_component.channel_selector_bundle
        bundle.marker_set_dropdown.value = "mixed"
        from ueler.viewer.plugin import _chart_common

        _chart_common.apply_marker_set_to_selector(bundle, self.viewer)
        # Only numeric cell-table columns survive; "fov" (object) and the absent
        # channel are filtered out.
        self.assertEqual(list(bundle.tags.value), ["intensity"])


class TestHistogramBokehLayout(unittest.TestCase):
    """Build the Bokeh layout (bokeh only; no jupyter_bokeh needed)."""

    def setUp(self):
        if not _bokeh_available():
            self.skipTest("bokeh not available in this environment")
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer)
        self.hist._plot_data = self.viewer.cell_table.copy()

    def test_build_figures_one_per_channel_with_shared_edges(self):
        self.hist._channels = ["intensity", "area"]
        layout, sources, spans = self.hist._build_figures()
        # A figure (and sources/spans) per channel.
        self.assertEqual(set(sources), {"intensity", "area"})
        self.assertEqual(set(spans), {"intensity", "area"})
        # The selected overlay shares the same bin edges as the full histogram.
        edges = sources["intensity"]["edges"]
        self.assertEqual(
            sources["intensity"]["selected"].data["left"], edges[:-1].tolist()
        )

    def test_cutoff_span_shows_only_on_gated_channels(self):
        """Each gated channel shows its own cutoff line; ungated ones show none (#127)."""
        self.hist._channels = ["intensity", "area"]
        _layout, sources, spans = self.hist._build_figures()
        self.hist._sources, self.hist._spans = sources, spans
        self.hist._active_histogram_column = "intensity"
        self.hist.cutoff = 5.0
        self.hist.highlight_cells(push_to_gallery=True)
        self.assertTrue(spans["intensity"].visible)
        self.assertEqual(spans["intensity"].location, 5.0)
        self.assertFalse(spans["area"].visible)

        # A second cutoff gates `area` too — the first channel keeps its line.
        self.hist._active_histogram_column = "area"
        self.hist.cutoff = 25.0
        self.hist.highlight_cells(push_to_gallery=True)
        self.assertTrue(spans["intensity"].visible)
        self.assertEqual(spans["intensity"].location, 5.0)
        self.assertTrue(spans["area"].visible)
        self.assertEqual(spans["area"].location, 25.0)

    def test_range_band_marks_the_brushed_channel_only(self):
        """A brushed range is drawn as our own persistent band (#127)."""
        self.hist._channels = ["intensity", "area"]
        _layout, sources, spans = self.hist._build_figures()
        self.hist._sources, self.hist._spans = sources, spans
        self.hist.handle_range("intensity", 4.0, 8.0)
        band = sources["intensity"]["band"]
        self.assertTrue(band.visible)
        self.assertEqual((band.left, band.right), (4.0, 8.0))
        self.assertFalse(sources["area"]["band"].visible)

    def test_selection_does_not_mute_bars_on_box_select(self):
        """Bokeh's own (non)selection glyphs are pinned to the base glyph (#127).

        Otherwise a box-select gesture greys out the non-selected bars of the
        histogram being brushed, which reads as the other channels' state changing.
        """
        self.hist.ui_component.interaction_mode.value = "Brush"
        self.hist._channels = ["intensity"]
        layout, _sources, _spans = self.hist._build_figures()
        for fig in layout.children:
            for renderer in fig.renderers:
                glyph = getattr(renderer, "glyph", None)
                if glyph is None:
                    continue
                self.assertIs(renderer.selection_glyph, glyph)
                self.assertIs(renderer.nonselection_glyph, glyph)

    def test_brush_mode_activates_box_select_drag(self):
        """Brush mode must set the BoxSelectTool as the active drag gesture (#112 reply).

        Without this, click-drag falls back to pan and no range can be brushed.
        """
        from bokeh.models import BoxSelectTool

        self.hist.ui_component.interaction_mode.value = "Brush"
        self.hist._channels = ["intensity", "area"]
        layout, _sources, _spans = self.hist._build_figures()
        for fig in layout.children:
            self.assertIsInstance(fig.toolbar.active_drag, BoxSelectTool)

    def test_cutoff_mode_does_not_activate_box_select(self):
        """Cutoff mode leaves drag as pan/auto so tapping to set a cutoff still works."""
        from bokeh.models import BoxSelectTool

        self.hist.ui_component.interaction_mode.value = "Cutoff"
        self.hist._channels = ["intensity"]
        layout, _sources, _spans = self.hist._build_figures()
        for fig in layout.children:
            self.assertNotIsInstance(fig.toolbar.active_drag, BoxSelectTool)


class TestHistogramInteractionModeSwitch(unittest.TestCase):
    """Cutoff ↔ Brush switches the gesture in place, never the plot (#127 reply)."""

    def setUp(self):
        if not _bokeh_available():
            self.skipTest("bokeh not available in this environment")
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer)
        self.hist._plot_data = self.viewer.cell_table.copy()
        self.hist._channels = ["intensity", "area"]
        self.hist.ui_component.interaction_mode.value = "Cutoff"
        _layout, sources, spans = self.hist._build_figures()
        self.hist._sources, self.hist._spans = sources, spans
        self.hist._render_calls = 0

    def _switch_to(self, mode):
        self.hist.ui_component.interaction_mode.value = mode
        # Direct call as well, so the assertion holds under the widget stubs
        # (no traitlets) as well as with real ipywidgets.
        self.hist._on_interaction_mode_change(None)

    def test_switching_mode_never_replots(self):
        self._switch_to("Brush")
        self._switch_to("Cutoff")
        self.assertEqual(self.hist._render_calls, 0)

    def test_switching_mode_keeps_the_same_figures(self):
        """The very objects holding the user's zoom/pan must survive the switch."""
        before = dict(self.hist._figures)
        self._switch_to("Brush")
        for channel, fig in before.items():
            self.assertIs(self.hist._figures[channel], fig)

    def test_switching_flips_the_active_drag_tool_in_place(self):
        from bokeh.models import BoxSelectTool

        self._switch_to("Brush")
        for channel, fig in self.hist._figures.items():
            self.assertIs(fig.toolbar.active_drag, self.hist._box_tools[channel])
        self._switch_to("Cutoff")
        for fig in self.hist._figures.values():
            self.assertNotIsInstance(fig.toolbar.active_drag, BoxSelectTool)

    def test_both_gestures_are_wired_in_either_mode(self):
        """Handlers stay registered, so the toggle has nothing to rebuild."""
        from bokeh.events import SelectionGeometry, Tap

        for fig in self.hist._figures.values():
            self.assertTrue(fig._event_callbacks.get(SelectionGeometry.event_name))
            self.assertTrue(fig._event_callbacks.get(Tap.event_name))

    def test_switching_mode_keeps_the_gate_and_its_markers(self):
        self.hist.handle_range("intensity", 4.0, 8.0)
        selected = set(self.hist.selected_indices.value)
        self._switch_to("Brush")
        self.assertEqual(self.hist.selected_indices.value, selected)
        self.assertTrue(self.hist._sources["intensity"]["band"].visible)
        self.assertEqual(self.hist.gate_description(), "Gate: intensity ∈ [4, 8]")

    def test_tap_is_ignored_while_brushing(self):
        """A bare click during a brush must not silently set a cutoff."""
        self.hist.ui_component.interaction_mode.value = "Brush"
        self.hist._make_tap_handler("intensity")(SimpleNamespace(x=5.0))
        self.assertIsNone(self.hist.cutoff)
        self.assertEqual(self.hist._gates, {})

    def test_tap_sets_a_cutoff_in_cutoff_mode(self):
        self.hist.ui_component.interaction_mode.value = "Cutoff"
        self.hist._make_tap_handler("intensity")(SimpleNamespace(x=5.0))
        self.assertEqual(self.hist.cutoff, 5.0)
        self.assertIn("intensity", self.hist._gates)


class TestHistogramRendering(unittest.TestCase):
    """Exercise the full interactive render path (skipped without the Bokeh stack)."""

    def setUp(self):
        if not _bokeh_stack_available():
            self.skipTest("bokeh + jupyter_bokeh not available in this environment")
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer, patch_render=False)

    def test_render_hosts_a_single_bokeh_model(self):
        self.hist.ui_component.channel_selector.value = ("intensity", "area")
        self.hist.plot_histograms(None)
        # A single BokehModel widget hosting the multi-figure column is swapped in.
        self.assertEqual(len(self.hist._plot_host.children), 1)
        self.assertIs(self.hist._plot_host.children[0], self.hist._bokeh_model)

    def test_render_with_narrow_subset_overlay_does_not_crash(self):
        """Rendering the shared-edge overlay for a narrow subset builds cleanly (#112)."""
        self.hist.ui_component.channel_selector.value = ("intensity",)
        self.hist.plot_histograms(None)
        self.hist.selected_indices.value = {1}  # intensity 5.0 (range is 1..9)
        self.hist._render()
        self.assertEqual(len(self.hist._plot_host.children), 1)

    def test_tall_stack_applies_scroll_to_the_model(self):
        """A tall histogram stack sets a fixed height + overflow on the BokehModel (#112 reply 2)."""
        from ueler.viewer.plugin.histogram import _MAX_PLOT_HEIGHT

        # Enough numeric channels to exceed the scroll cap.
        table = pd.DataFrame(
            {
                "fov": ["f1", "f1", "f1"],
                "label": [1, 2, 3],
                **{f"m{i}": [1.0, 2.0, 3.0] for i in range(5)},
            }
        )
        viewer = _make_viewer(table)
        hist = _make_histogram(viewer, patch_render=False)
        hist.ui_component.channel_selector.value = tuple(f"m{i}" for i in range(5))
        hist.plot_histograms(None)
        self.assertEqual(hist._bokeh_model.layout.height, f"{_MAX_PLOT_HEIGHT}px")
        self.assertIn("auto", hist._bokeh_model.layout.overflow)

    def test_short_stack_leaves_model_unconstrained(self):
        """A single histogram renders at natural height (no scroll height on the model)."""
        self.hist.ui_component.channel_selector.value = ("intensity",)
        self.hist.plot_histograms(None)
        self.assertIsNone(self.hist._bokeh_model.layout.height)


class TestHistogramSideOnly(unittest.TestCase):
    """The histogram lives in the side accordion, not the wide footer (#121 reply).

    The first pass at #121 briefly moved the histogram into the footer; the reply
    corrected that — the histogram belongs in the side panel and the heatmap is the
    plugin that gets the permanent wide-footer allocation instead.
    """

    def setUp(self):
        self.viewer = _make_viewer(_two_fov_table())
        self.hist = _make_histogram(self.viewer)

    def test_histogram_is_not_footer_only(self):
        self.assertFalse(self.hist.footer_only)

    def test_wide_panel_layout_is_none(self):
        self.assertIsNone(self.hist.wide_panel_layout())


if __name__ == "__main__":
    unittest.main()
