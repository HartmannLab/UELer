"""Tests for histogram cutoff → cell gallery linking (issue #77).

Verifies that calling highlight_cells() updates selected_indices and
consequently triggers the cell-gallery observer when the linking checkbox
is enabled.
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
if "anywidget" not in sys.modules:
    sys.modules["anywidget"] = types.ModuleType("anywidget")

if "jscatter" not in sys.modules:
    jscatter_stub = types.ModuleType("jscatter")

    class _StubScatter:
        def __init__(self, *_, **__):
            self.widget = SimpleNamespace(
                layout=None,
                observe=lambda *a, **k: None,
                mouse_mode=None,
            )
            self._selection: list = []

        def axes(self, *_, **__): return None
        def height(self, *_, **__): return None
        def size(self, *_, **__): return None
        def color(self, *_, **__): return None
        def tooltip(self, *_, **__): return None
        def show(self, *_, **__): return self.widget

        def selection(self, values=None):
            if values is not None:
                self._selection = list(values)
            return self._selection

    def _stub_compose(*entries, **_kwargs):
        return entries

    jscatter_stub.Scatter = _StubScatter
    jscatter_stub.compose = _stub_compose
    sys.modules["jscatter"] = jscatter_stub

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
        """Helper for tests: simulate a widget value change."""
        self.value = new_value
        for cb, _ in self._observers:
            cb(SimpleNamespace(new=new_value))

    def on_click(self, *_, **__):
        return None

    def clear_output(self, *_, **__):
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

if "matplotlib.pyplot" not in sys.modules:
    _mpl = types.ModuleType("matplotlib")
    _plt = types.ModuleType("matplotlib.pyplot")

    class _Canvas:
        def mpl_connect(self, *_, **__): return None
        def draw_idle(self, *_, **__): return None

    class _Figure:
        def __init__(self):
            self.canvas = _Canvas()
        def tight_layout(self): return None

    class _Axes:
        def __init__(self, fig):
            self.figure = fig
        def hist(self, *_, **__): return None
        def set_xlabel(self, *_, **__): return None
        def set_ylabel(self, *_, **__): return None
        def axvline(self, *_, **__):
            return SimpleNamespace(remove=lambda: None)

    def _subplots(*_, **__):
        fig = _Figure()
        return fig, _Axes(fig)

    _plt.subplots = _subplots
    _plt.show = lambda *_, **__: None
    _mpl.pyplot = _plt
    _font = types.ModuleType("matplotlib.font_manager")
    _font.FontProperties = type("FontProperties", (), {"__init__": lambda self, *_, **__: None})
    _mpl.font_manager = _font
    sys.modules["matplotlib"] = _mpl
    sys.modules["matplotlib.pyplot"] = _plt
    sys.modules["matplotlib.font_manager"] = _font

if "matplotlib.font_manager" not in sys.modules:
    _font2 = types.ModuleType("matplotlib.font_manager")
    _font2.FontProperties = type("FontProperties", (), {"__init__": lambda self, *_, **__: None})
    sys.modules["matplotlib.font_manager"] = _font2

for _mod in ["seaborn_image", "tifffile", "cv2", "dask"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

if "skimage" not in sys.modules:
    _ski = types.ModuleType("skimage")
    for _sub, _attrs in [
        ("skimage.segmentation", {"find_boundaries": lambda *_, **__: None}),
        ("skimage.io", {"imread": lambda *_, **__: None, "imsave": lambda *_, **__: None}),
        ("skimage.exposure", {"rescale_intensity": lambda img, **_: img}),
        ("skimage.transform", {"resize": lambda *_, **__: None}),
        ("skimage.color", {"label2rgb": lambda *_, **__: None}),
        ("skimage.measure", {"regionprops_table": lambda *_, **__: {}}),
        ("skimage.draw", {"circle_perimeter": lambda *_, **__: ([], [])}),
    ]:
        _m = types.ModuleType(_sub)
        for k, v in _attrs.items():
            setattr(_m, k, v)
        sys.modules[_sub] = _m
    sys.modules["skimage"] = _ski

import numpy as np
import pandas as pd

from ueler.viewer.plugin.chart import ChartDisplay


# ---------------------------------------------------------------------------
# Minimal viewer stub
# ---------------------------------------------------------------------------

class _FakeImageDisplay:
    def __init__(self):
        self.last_mask_ids: list = []

    def set_mask_ids(self, *, mask_name, mask_ids, fov_mask_pairs=None):
        self.last_mask_ids = list(mask_ids)


class _FakeCellGallery:
    def __init__(self):
        self.received: object = None

    def set_selected_cells(self, indices):
        self.received = indices


def _make_viewer(cell_table: "pd.DataFrame") -> SimpleNamespace:
    """Return a minimal fake viewer that satisfies ChartDisplay requirements."""
    gallery = _FakeCellGallery()
    image_display = _FakeImageDisplay()

    ui_component = SimpleNamespace(
        image_selector=SimpleNamespace(value="fov1"),
    )
    side_plots = SimpleNamespace(cell_gallery_output=gallery)

    viewer = SimpleNamespace(
        cell_table=cell_table,
        fov_key="fov",
        label_key="label",
        mask_key="cells",
        ui_component=ui_component,
        image_display=image_display,
        SidePlots=side_plots,
        get_active_fov=lambda: ui_component.image_selector.value,
    )
    return viewer


def _make_chart(viewer: SimpleNamespace) -> ChartDisplay:
    """Instantiate ChartDisplay with the fake viewer."""
    chart = ChartDisplay(viewer, width=4, height=3)
    chart.setup_observe()
    return chart


def _two_fov_table() -> "pd.DataFrame":
    """Cell table spanning two FOVs with a numeric 'intensity' column."""
    return pd.DataFrame(
        {
            "fov": ["fov1", "fov1", "fov1", "fov2", "fov2"],
            "label": [1, 2, 3, 4, 5],
            "intensity": [1.0, 5.0, 9.0, 3.0, 7.0],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHistogramCutoffGalleryLink(unittest.TestCase):
    """highlight_cells() → selected_indices → cell gallery forwarding."""

    def setUp(self):
        table = _two_fov_table()
        self.viewer = _make_viewer(table)
        self.chart = _make_chart(self.viewer)
        self.gallery: _FakeCellGallery = self.viewer.SidePlots.cell_gallery_output

        # Configure chart state to simulate a histogram cutoff
        self.chart.ui_component.x_axis_selector.value = "intensity"
        self.chart.ui_component.above_below_buttons.value = "above"
        self.chart.ui_component.cell_gallery_linked_checkbox.value = True
        self.chart.cutoff = 4.0

    def test_gallery_receives_all_fov_indices_above_cutoff(self):
        """Gallery should get row indices from ALL fovs whose intensity > cutoff."""
        self.chart.highlight_cells(push_to_gallery=True)
        # intensity > 4.0: rows with intensity 5.0, 9.0, 7.0 (indices 1, 2, 4)
        self.assertIsNotNone(self.gallery.received)
        received = set(self.gallery.received)
        table = self.viewer.cell_table
        expected = set(table.loc[table["intensity"] > 4.0].index)
        self.assertEqual(received, expected)

    def test_gallery_not_updated_when_checkbox_off(self):
        """When linking checkbox is False, gallery must not receive new indices."""
        self.chart.ui_component.cell_gallery_linked_checkbox.value = False
        self.chart.highlight_cells(push_to_gallery=True)
        self.assertIsNone(self.gallery.received)

    def test_gallery_receives_indices_below_cutoff(self):
        """Selecting 'below' sends indices whose intensity < cutoff to gallery."""
        self.chart.ui_component.above_below_buttons.value = "below"
        self.chart.highlight_cells(push_to_gallery=True)
        table = self.viewer.cell_table
        expected = set(table.loc[table["intensity"] < 4.0].index)
        received = set(self.gallery.received)
        self.assertEqual(received, expected)

    def test_gallery_uses_all_fovs_not_just_current(self):
        """Gallery indices must include cells from fov2, not only from fov1."""
        self.chart.highlight_cells(push_to_gallery=True)
        table = self.viewer.cell_table
        fov2_indices = set(table.loc[table["fov"] == "fov2"].index)
        received = set(self.gallery.received)
        # At least one fov2 row (intensity 7.0 > 4.0) must appear
        self.assertTrue(received & fov2_indices, "gallery should include fov2 cells")

    def test_image_display_still_limited_to_current_fov(self):
        """Image highlighting should still be restricted to the current FOV."""
        self.chart.highlight_cells(push_to_gallery=True)
        image_display: _FakeImageDisplay = self.viewer.image_display
        # fov1 has labels 1, 2, 3; only 2 and 3 have intensity > 4.0
        self.assertNotIn(1, image_display.last_mask_ids)
        self.assertIn(2, image_display.last_mask_ids)
        self.assertIn(3, image_display.last_mask_ids)
        # fov2 labels must NOT appear in image highlight
        for label in [4, 5]:
            self.assertNotIn(label, image_display.last_mask_ids)

    def test_no_crash_when_cutoff_is_none(self):
        """highlight_cells() with no cutoff set should return silently."""
        self.chart.cutoff = None
        self.chart.highlight_cells(push_to_gallery=True)  # should not raise
        self.assertIsNone(self.gallery.received)

    def test_no_crash_when_x_col_is_none(self):
        """highlight_cells() with x_axis set to 'None' should return silently."""
        self.chart.ui_component.x_axis_selector.value = "None"
        self.chart.highlight_cells(push_to_gallery=True)
        self.assertIsNone(self.gallery.received)

    def test_auto_rerender_does_not_update_gallery(self):
        """Automatic histogram re-renders (FOV/bin change) must NOT override gallery."""
        # Simulate a scatter-driven gallery update first
        self.gallery.received = {99}  # pretend scatter already set indices
        # Auto re-render path (push_to_gallery=False, the default)
        self.chart.highlight_cells(push_to_gallery=False)
        # Gallery must be unchanged
        self.assertEqual(self.gallery.received, {99})


class TestAboveBelowToggleAutoUpdate(unittest.TestCase):
    """Toggling above_below_buttons should re-trigger highlight_cells()."""

    def setUp(self):
        table = _two_fov_table()
        self.viewer = _make_viewer(table)
        chart = ChartDisplay(self.viewer, width=4, height=3)

        # Replace with a controllable stub BEFORE setup_observe() so the
        # observer lambda is registered on our stub rather than on the real
        # ToggleButtons (which doesn't fire in a non-notebook test context).
        class _ToggleStub:
            def __init__(self, value: str):
                self.value = value
                self._cbs: list = []

            def observe(self, callback, names=None):
                self._cbs.append(callback)

            def _trigger(self, new_value: str):
                self.value = new_value
                for cb in self._cbs:
                    cb(SimpleNamespace(new=new_value))

        chart.ui_component.above_below_buttons = _ToggleStub("above")
        chart.setup_observe()
        self.chart = chart
        self.chart.ui_component.x_axis_selector.value = "intensity"
        self.chart.ui_component.cell_gallery_linked_checkbox.value = True
        self.chart.cutoff = 4.0
        self.gallery: _FakeCellGallery = self.viewer.SidePlots.cell_gallery_output

    def test_toggle_below_updates_gallery_automatically(self):
        """Switching toggle from 'above' to 'below' should push new indices."""
        # Simulate initial state via explicit apply
        self.chart.highlight_cells(push_to_gallery=True)
        table = self.viewer.cell_table
        above_set = set(table.loc[table["intensity"] > 4.0].index)
        self.assertEqual(set(self.gallery.received), above_set)

        # Simulate widget toggle (above_below_buttons is a _Widget with observe support)
        self.chart.ui_component.above_below_buttons._trigger("below")

        below_set = set(table.loc[table["intensity"] < 4.0].index)
        self.assertEqual(set(self.gallery.received), below_set)

    def test_toggle_above_updates_gallery_automatically(self):
        """After switching to 'below', switching back to 'above' updates gallery."""
        self.chart.ui_component.above_below_buttons._trigger("below")
        self.chart.ui_component.above_below_buttons._trigger("above")
        table = self.viewer.cell_table
        expected = set(table.loc[table["intensity"] > 4.0].index)
        self.assertEqual(set(self.gallery.received), expected)

    def test_toggle_no_op_when_cutoff_not_set(self):
        """Toggle before any cutoff is set should not crash and gallery unchanged."""
        self.chart.cutoff = None
        prev = self.gallery.received
        self.chart.ui_component.above_below_buttons._trigger("below")
        self.assertEqual(self.gallery.received, prev)


if __name__ == "__main__":
    unittest.main()
