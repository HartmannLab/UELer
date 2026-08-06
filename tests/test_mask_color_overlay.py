"""Tests for the mask color registry optimisations introduced in issue #82.

Covers:
- Nested-dict registry structure (O(1) per-FOV access)
- set_cell_colors_bulk vectorised write
- Per-class dirty tracking in apply_colors_to_masks
"""

import unittest
from unittest.mock import MagicMock, patch, call

import pandas as pd
import numpy as np

from ueler.rendering import (
    set_cell_color,
    set_cell_colors_bulk,
    get_cell_color,
    get_all_cell_colors_for_fov,
    clear_cell_colors,
)
from ueler.rendering.engine import _CELL_COLOR_REGISTRY
from ueler.viewer.mask_color_overlay import apply_registry_colors


# ---------------------------------------------------------------------------
# Registry structure tests
# ---------------------------------------------------------------------------

class NestedRegistryTests(unittest.TestCase):
    """Verify the restructured nested dict registry behaves correctly."""

    def setUp(self):
        clear_cell_colors()

    def tearDown(self):
        clear_cell_colors()

    def test_set_cell_color_creates_nested_entry(self):
        set_cell_color("FOV_001", 42, "#FF0000")
        self.assertIn("FOV_001", _CELL_COLOR_REGISTRY)
        self.assertEqual(_CELL_COLOR_REGISTRY["FOV_001"][42], "#FF0000")

    def test_get_cell_color_o1_lookup(self):
        set_cell_color("FOV_001", 7, "#00FF00")
        self.assertEqual(get_cell_color("FOV_001", 7), "#00FF00")

    def test_get_cell_color_missing_fov_returns_none(self):
        self.assertIsNone(get_cell_color("NO_SUCH_FOV", 1))

    def test_get_cell_color_missing_mask_id_returns_none(self):
        set_cell_color("FOV_001", 1, "#FFFFFF")
        self.assertIsNone(get_cell_color("FOV_001", 999))

    def test_get_all_cell_colors_for_fov_o1_lookup(self):
        set_cell_color("FOV_001", 1, "#FF0000")
        set_cell_color("FOV_001", 2, "#00FF00")
        set_cell_color("FOV_002", 3, "#0000FF")  # different FOV — must not appear
        result = get_all_cell_colors_for_fov("FOV_001")
        self.assertEqual(result, {1: "#FF0000", 2: "#00FF00"})

    def test_get_all_cell_colors_for_fov_missing_fov_returns_empty(self):
        self.assertEqual(get_all_cell_colors_for_fov("NO_SUCH_FOV"), {})

    def test_clear_cell_colors_specific_fov(self):
        set_cell_color("FOV_001", 1, "#FF0000")
        set_cell_color("FOV_002", 2, "#00FF00")
        clear_cell_colors("FOV_001")
        self.assertIsNone(get_cell_color("FOV_001", 1))
        self.assertEqual(get_cell_color("FOV_002", 2), "#00FF00")

    def test_clear_cell_colors_all(self):
        set_cell_color("FOV_001", 1, "#FF0000")
        set_cell_color("FOV_002", 2, "#00FF00")
        clear_cell_colors()
        self.assertEqual(get_all_cell_colors_for_fov("FOV_001"), {})
        self.assertEqual(get_all_cell_colors_for_fov("FOV_002"), {})

    def test_get_all_cell_colors_returns_copy(self):
        """Mutating the returned dict must not affect the registry."""
        set_cell_color("FOV_001", 1, "#FF0000")
        result = get_all_cell_colors_for_fov("FOV_001")
        result[99] = "#AABBCC"
        self.assertNotIn(99, _CELL_COLOR_REGISTRY.get("FOV_001", {}))


# ---------------------------------------------------------------------------
# set_cell_colors_bulk tests
# ---------------------------------------------------------------------------

class BulkWriteTests(unittest.TestCase):
    """Verify set_cell_colors_bulk writes the nested registry correctly."""

    def setUp(self):
        clear_cell_colors()

    def tearDown(self):
        clear_cell_colors()

    def test_bulk_write_single_fov(self):
        entries = {"FOV_001": {1: "#FF0000", 2: "#00FF00", 3: "#0000FF"}}
        set_cell_colors_bulk(entries)
        self.assertEqual(get_cell_color("FOV_001", 1), "#FF0000")
        self.assertEqual(get_cell_color("FOV_001", 2), "#00FF00")
        self.assertEqual(get_cell_color("FOV_001", 3), "#0000FF")

    def test_bulk_write_multiple_fovs(self):
        entries = {
            "FOV_001": {10: "#AAAAAA"},
            "FOV_002": {20: "#BBBBBB"},
            "FOV_003": {30: "#CCCCCC"},
        }
        set_cell_colors_bulk(entries)
        self.assertEqual(get_cell_color("FOV_001", 10), "#AAAAAA")
        self.assertEqual(get_cell_color("FOV_002", 20), "#BBBBBB")
        self.assertEqual(get_cell_color("FOV_003", 30), "#CCCCCC")

    def test_bulk_write_merges_with_existing(self):
        set_cell_color("FOV_001", 1, "#FF0000")
        set_cell_colors_bulk({"FOV_001": {2: "#00FF00"}})
        # old entry preserved, new entry added
        self.assertEqual(get_cell_color("FOV_001", 1), "#FF0000")
        self.assertEqual(get_cell_color("FOV_001", 2), "#00FF00")

    def test_bulk_write_overwrites_existing_entry(self):
        set_cell_color("FOV_001", 5, "#OLD000")
        set_cell_colors_bulk({"FOV_001": {5: "#NEW000"}})
        self.assertEqual(get_cell_color("FOV_001", 5), "#NEW000")

    def test_bulk_write_empty_entries_is_noop(self):
        set_cell_colors_bulk({})
        self.assertEqual(_CELL_COLOR_REGISTRY, {})


class OverlayFillRenderingTests(unittest.TestCase):
    """Verify painter overlay rendering supports per-cell fill opacity and borders."""

    def test_apply_registry_colors_uses_per_cell_opacity(self):
        image = np.zeros((3, 3, 3), dtype=np.float32)
        region = np.array(
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.int32,
        )

        result = apply_registry_colors(
            image,
            fov="FOV_001",
            mask_regions={"cell": region},
            outline_thickness=1,
            downsample_factor=1,
            color_map={1: "#FF0000"},
            mode_map={1: "fill"},
            opacity_map={1: 0.5},
        )

        self.assertAlmostEqual(float(result[1, 1, 0]), 0.5, places=4)
        self.assertAlmostEqual(float(result[1, 1, 1]), 0.0, places=4)
        self.assertAlmostEqual(float(result[1, 1, 2]), 0.0, places=4)

    def test_vectorized_fill_distinct_colors_no_cross_contamination(self):
        """The all-fill fast path (continuous coloring) colors each cell exactly
        with its own color at alpha 1.0, with no bleed between neighbors."""
        image = np.zeros((2, 4, 3), dtype=np.float32)
        region = np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
            ],
            dtype=np.int32,
        )
        result = apply_registry_colors(
            image,
            fov="FOV_001",
            mask_regions={"cell": region},
            outline_thickness=1,
            downsample_factor=1,
            color_map={1: "#FF0000", 2: "#00FF00"},
            mode_map={1: "fill", 2: "fill"},
            opacity_map={1: 1.0, 2: 1.0},
        )
        # Cell 1 → pure red, cell 2 → pure green.
        self.assertTrue(np.allclose(result[:, :2], np.array([1.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(result[:, 2:], np.array([0.0, 1.0, 0.0])))

    def test_vectorized_fill_leaves_background_untouched(self):
        """Background (id 0) and unregistered ids keep the base image."""
        image = np.full((2, 3, 3), 0.2, dtype=np.float32)
        region = np.array([[0, 1, 9], [0, 1, 9]], dtype=np.int32)
        result = apply_registry_colors(
            image,
            fov="FOV_001",
            mask_regions={"cell": region},
            outline_thickness=1,
            downsample_factor=1,
            color_map={1: "#FF0000"},  # id 9 has no color
            mode_map={1: "fill"},
            opacity_map={1: 1.0},
        )
        self.assertTrue(np.allclose(result[:, 0], 0.2))  # background
        self.assertTrue(np.allclose(result[:, 2], 0.2))  # unregistered id 9
        self.assertTrue(np.allclose(result[:, 1], np.array([1.0, 0.0, 0.0])))

    def test_zero_fill_opacity_falls_back_to_outline_only(self):
        image = np.zeros((5, 5, 3), dtype=np.float32)
        region = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        # The real inner boundary of the 3x3 block is its ring; [2, 2] is interior.
        result = apply_registry_colors(
            image,
            fov="FOV_001",
            mask_regions={"cell": region},
            outline_thickness=1,
            downsample_factor=1,
            color_map={1: "#FF0000"},
            mode_map={1: "fill"},
            opacity_map={1: 0.0},
            show_borders_on_filled=False,
        )

        self.assertAlmostEqual(float(result[2, 2, 0]), 0.0, places=4)
        self.assertTrue(np.any(result[:, :, 0] == 1.0))

    def test_fill_with_border_preserves_outline_on_top(self):
        image = np.zeros((5, 5, 3), dtype=np.float32)
        region = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        result = apply_registry_colors(
            image,
            fov="FOV_001",
            mask_regions={"cell": region},
            outline_thickness=1,
            downsample_factor=1,
            color_map={1: "#FF0000"},
            mode_map={1: "fill"},
            opacity_map={1: 0.5},
            show_borders_on_filled=True,
        )

        # [2, 2] is interior → blended fill; [1, 1] is on the ring → solid border.
        self.assertAlmostEqual(float(result[2, 2, 0]), 0.5, places=4)
        self.assertAlmostEqual(float(result[1, 1, 0]), 1.0, places=4)

    def test_fill_with_border_can_use_distinct_border_color(self):
        image = np.zeros((5, 5, 3), dtype=np.float32)
        region = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        result = apply_registry_colors(
            image,
            fov="FOV_001",
            mask_regions={"cell": region},
            outline_thickness=1,
            downsample_factor=1,
            color_map={1: "#FF0000"},
            border_color_map={1: "#00FF00"},
            mode_map={1: "fill"},
            opacity_map={1: 0.5},
            show_borders_on_filled=True,
        )

        self.assertAlmostEqual(float(result[2, 2, 0]), 0.5, places=4)
        self.assertAlmostEqual(float(result[2, 2, 1]), 0.0, places=4)
        self.assertAlmostEqual(float(result[1, 1, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(result[1, 1, 1]), 1.0, places=4)

    def test_thickened_fill_border_does_not_reblend_neighbor_fill(self):
        """A thickened filled border must stay inside its own cell (issue #91).

        Two touching 7x7 cells with ``outline_thickness=3`` (dilation 2): cell 1's border
        would reach two columns into cell 2 if it were not clipped to the owning cell.
        Cell 2's centre sits 3 steps from its own ring, so it stays pure fill.
        """
        image = np.zeros((9, 16, 3), dtype=np.float32)
        region = np.zeros((9, 16), dtype=np.int32)
        region[1:8, 1:8] = 1
        region[1:8, 8:15] = 2

        result = apply_registry_colors(
            image,
            fov="FOV_001",
            mask_regions={"cell": region},
            outline_thickness=3,
            downsample_factor=1,
            color_map={1: "#FF0000", 2: "#0000FF"},
            mode_map={1: "fill", 2: "fill"},
            opacity_map={1: 0.5, 2: 0.5},
            show_borders_on_filled=True,
        )

        cell2 = region == 2
        # Nothing red may appear anywhere in cell 2's footprint.
        self.assertTrue(np.allclose(result[cell2][:, 0], 0.0))
        # The one pixel of cell 2 that its own thickened border does not reach keeps the fill blend.
        self.assertAlmostEqual(float(result[4, 11, 2]), 0.5, places=4)
        self.assertAlmostEqual(float(result[4, 11, 0]), 0.0, places=4)


def _reference_apply_region_colors(
    image,
    region,
    *,
    registry,
    border_registry=None,
    dilation=0,
    exclude_ids=frozenset(),
    mode_map=None,
    opacity_map=None,
    fill_alpha=0.35,
    show_borders_on_filled=False,
):
    """The pre-#131 per-cell loop, kept as an executable oracle.

    ``_apply_region_colors`` was rewritten from O(cells x pixels) to O(pixels). This
    reproduces the original algorithm verbatim so the parity tests below compare the new
    implementation against what it replaced, rather than against hand-written expectations.
    """
    from matplotlib.colors import to_rgb
    from skimage.segmentation import find_boundaries
    from ueler.rendering.engine import thicken_outline

    border_registry = border_registry or {}
    mode_map = mode_map or {}
    opacity_map = opacity_map or {}

    canvas = np.array(image, copy=True)
    pending = []
    for raw in np.unique(region):
        if not raw:
            continue
        mask_id = int(raw)
        if mask_id in exclude_ids:
            continue
        colour_hex = registry.get(mask_id)
        if not colour_hex:
            continue
        rgb = np.array(to_rgb(colour_hex), dtype=np.float32)
        border_rgb = np.array(to_rgb(border_registry.get(mask_id, colour_hex)), dtype=np.float32)
        mask_bool = region == raw

        if mode_map.get(mask_id, "outline") == "fill":
            alpha = max(0.0, min(1.0, float(opacity_map.get(mask_id, fill_alpha))))
            if alpha > 0.0:
                canvas[mask_bool] = (
                    (1.0 - alpha) * canvas[mask_bool] + alpha * rgb
                ).astype(canvas.dtype)
            if show_borders_on_filled or alpha <= 0.0:
                edges = find_boundaries(mask_bool, mode="inner")
                if dilation > 0:
                    edges = thicken_outline(edges, dilation)
                edges = np.logical_and(edges, mask_bool)
                if np.any(edges):
                    pending.append((edges, border_rgb))
        else:
            edges = find_boundaries(mask_bool, mode="inner")
            if dilation > 0:
                edges = thicken_outline(edges, dilation)
            if np.any(edges):
                pending.append((edges, rgb))

    for edges, colour in pending:
        canvas[edges] = colour.astype(canvas.dtype, copy=False)
    return canvas


class VectorizedOverlayParityTests(unittest.TestCase):
    """Issue #131: the batched overlay must match the per-cell loop it replaced."""

    def _blocky_labels(self, height, width, n_cells):
        side = int(np.ceil(np.sqrt(n_cells)))
        ys = np.linspace(0, side, height, endpoint=False).astype(np.int64)
        xs = np.linspace(0, side, width, endpoint=False).astype(np.int64)
        labels = (ys[:, None] * side + xs[None, :] + 1).astype(np.int32)
        labels[labels > n_cells] = 0
        return labels

    def _run_both(self, region, *, thickness=1, **kwargs):
        from ueler.viewer.mask_color_overlay import _resolve_outline_dilation

        rng = np.random.default_rng(7)
        image = rng.random(region.shape + (3,)).astype(np.float32)

        actual = apply_registry_colors(
            image,
            fov="FOV_001",
            mask_regions={"cell": region},
            outline_thickness=thickness,
            downsample_factor=1,
            color_map=kwargs.get("registry"),
            border_color_map=kwargs.get("border_registry"),
            exclude_ids=set(kwargs.get("exclude_ids", frozenset())),
            mode_map=kwargs.get("mode_map"),
            opacity_map=kwargs.get("opacity_map"),
            show_borders_on_filled=kwargs.get("show_borders_on_filled", False),
        )
        expected = _reference_apply_region_colors(
            image,
            region,
            dilation=_resolve_outline_dilation(thickness, 1),
            **kwargs,
        )
        return actual, expected

    def test_outline_mode_matches_reference(self):
        region = self._blocky_labels(24, 24, 16)
        ids = [int(v) for v in np.unique(region) if v]
        actual, expected = self._run_both(
            region,
            registry={i: "#FF0000" if i % 2 else "#00FF00" for i in ids},
            mode_map={i: "outline" for i in ids},
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_thick_outline_matches_reference(self):
        region = self._blocky_labels(30, 30, 9)
        ids = [int(v) for v in np.unique(region) if v]
        actual, expected = self._run_both(
            region,
            thickness=3,
            registry={i: "#123456" for i in ids},
            mode_map={i: "outline" for i in ids},
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_mixed_modes_match_reference(self):
        region = self._blocky_labels(24, 24, 16)
        ids = [int(v) for v in np.unique(region) if v]
        actual, expected = self._run_both(
            region,
            registry={i: "#FF0000" for i in ids},
            border_registry={i: "#00FF00" for i in ids if i % 3 == 0},
            mode_map={i: ("fill" if i % 2 else "outline") for i in ids},
            opacity_map={i: (0.0 if i % 4 == 0 else 0.6) for i in ids},
            show_borders_on_filled=True,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_excluded_and_uncolored_ids_match_reference(self):
        region = self._blocky_labels(24, 24, 16)
        ids = [int(v) for v in np.unique(region) if v]
        actual, expected = self._run_both(
            region,
            registry={i: "#FF00FF" for i in ids if i % 3},  # some ids have no color
            exclude_ids=frozenset({ids[0], ids[-1]}),
            mode_map={i: "outline" for i in ids},
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_thick_fill_borders_match_reference(self):
        region = self._blocky_labels(30, 30, 9)
        ids = [int(v) for v in np.unique(region) if v]
        actual, expected = self._run_both(
            region,
            thickness=3,
            registry={i: "#FF0000" for i in ids},
            border_registry={i: "#0000FF" for i in ids},
            mode_map={i: "fill" for i in ids},
            opacity_map={i: 0.4 for i in ids},
            show_borders_on_filled=True,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_boundaries_computed_once_per_region(self):
        """The whole point of #131: cost must not scale with the number of cells."""
        import ueler.viewer.mask_color_overlay as overlay

        for n_cells in (4, 400):
            region = self._blocky_labels(40, 40, n_cells)
            ids = [int(v) for v in np.unique(region) if v]
            calls = []
            real = overlay.find_boundaries

            def _counting(array, mode=None, _real=real, _calls=calls):
                _calls.append(array.shape)
                return _real(array, mode=mode)

            with patch.object(overlay, "find_boundaries", side_effect=_counting):
                apply_registry_colors(
                    np.zeros(region.shape + (3,), dtype=np.float32),
                    fov="FOV_001",
                    mask_regions={"cell": region},
                    outline_thickness=1,
                    downsample_factor=1,
                    color_map={i: "#FF0000" for i in ids},
                    mode_map={i: "outline" for i in ids},
                )
            self.assertEqual(
                len(calls), 1, f"{n_cells} cells triggered {len(calls)} boundary passes"
            )

    def test_thickened_outlines_resolve_overlap_to_highest_id(self):
        """The old loop painted ascending by id, so the largest id won an overlap."""
        region = np.zeros((7, 9), dtype=np.int32)
        region[3, 1] = 5
        region[3, 7] = 9
        region[3, 2:7] = 0  # a gap both outlines grow into

        actual, expected = self._run_both(
            region,
            thickness=7,
            registry={5: "#FF0000", 9: "#0000FF"},
            mode_map={5: "outline", 9: "outline"},
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# _register_color_globally: bulk write path (mask_painter integration)
# ---------------------------------------------------------------------------

class RegisterColorGloballyBulkTests(unittest.TestCase):
    """Verify that apply_colors_to_masks uses set_cell_colors_bulk, not iterrows."""

    def _make_viewer(self, cell_table):
        import types

        viewer = types.SimpleNamespace()
        viewer.cell_table = cell_table
        viewer.fov_key = "fov"
        viewer.label_key = "label"
        viewer.mask_key = "cell"
        viewer.get_active_fov = lambda: "FOV_001"

        image_display = types.SimpleNamespace()
        image_display.set_mask_colors_current_fov = MagicMock()
        viewer.image_display = image_display
        return viewer

    def _setup_painter(self, painter, classes):
        """Configure painter with a set of visible classes."""
        import ipywidgets

        painter.ui_component.identifier_dropdown.options = ["cell_type"]
        painter.ui_component.identifier_dropdown.value = "cell_type"
        painter.selected_classes = list(classes)
        painter.ui_component.sorting_items_tagsinput.value = tuple(classes)
        painter.ui_component.show_all_checkbox.value = False
        painter.class_color_controls = {
            "A": ipywidgets.ColorPicker(value="#FF0000"),
            "B": ipywidgets.ColorPicker(value="#00FF00"),
        }

    def setUp(self):
        clear_cell_colors()

    def tearDown(self):
        clear_cell_colors()

    def test_register_color_globally_calls_bulk_write_not_iterrows(self):
        """set_cell_colors_bulk must be called; iterrows must not be called."""
        cell_table = pd.DataFrame({
            "fov": ["FOV_001", "FOV_001", "FOV_002", "FOV_002"],
            "label": [1, 2, 3, 4],
            "cell_type": ["A", "B", "A", "B"],
        })
        viewer = self._make_viewer(cell_table)

        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay

        painter = MaskPainterDisplay(viewer, width=400, height=300)
        self._setup_painter(painter, ["A", "B"])

        with patch("ueler.viewer.plugin.mask_painter.set_cell_colors_bulk") as mock_bulk:
            painter.apply_colors_to_masks(None, register_globally=True)
            self.assertTrue(mock_bulk.called, "set_cell_colors_bulk was not called")

    def test_register_color_globally_populates_registry_for_all_fovs(self):
        """After apply_colors_to_masks, registry has entries for every FOV."""
        import ipywidgets

        cell_table = pd.DataFrame({
            "fov": ["FOV_001", "FOV_002", "FOV_003"],
            "label": [10, 20, 30],
            "cell_type": ["A", "A", "A"],
        })
        viewer = self._make_viewer(cell_table)

        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay

        painter = MaskPainterDisplay(viewer, width=400, height=300)
        painter.ui_component.identifier_dropdown.options = ["cell_type"]
        painter.ui_component.identifier_dropdown.value = "cell_type"
        painter.selected_classes = ["A"]
        painter.ui_component.sorting_items_tagsinput.value = ("A",)
        painter.ui_component.show_all_checkbox.value = False
        painter.class_color_controls = {
            "A": ipywidgets.ColorPicker(value="#ABCDEF"),
        }

        painter.apply_colors_to_masks(None, register_globally=True)

        self.assertEqual(get_cell_color("FOV_001", 10), "#ABCDEF")
        self.assertEqual(get_cell_color("FOV_002", 20), "#ABCDEF")
        self.assertEqual(get_cell_color("FOV_003", 30), "#ABCDEF")


# ---------------------------------------------------------------------------
# Per-class dirty tracking tests
# ---------------------------------------------------------------------------

class PerClassDirtyTrackingTests(unittest.TestCase):
    """Verify that _register_color_globally is skipped for unchanged classes."""

    def _make_painter(self, cell_table):
        import types

        viewer = types.SimpleNamespace()
        viewer.cell_table = cell_table
        viewer.fov_key = "fov"
        viewer.label_key = "label"
        viewer.mask_key = "cell"
        viewer.get_active_fov = lambda: None  # map mode → always register_globally=True

        image_display = types.SimpleNamespace()
        image_display.set_mask_colors_current_fov = MagicMock()
        viewer.image_display = image_display

        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay
        import ipywidgets

        painter = MaskPainterDisplay(viewer, width=400, height=300)
        painter.ui_component.identifier_dropdown.options = ["cell_type"]
        painter.ui_component.identifier_dropdown.value = "cell_type"
        painter.selected_classes = ["A", "B"]
        painter.ui_component.sorting_items_tagsinput.value = ("A", "B")
        painter.ui_component.show_all_checkbox.value = False
        painter.class_color_controls = {
            "A": ipywidgets.ColorPicker(value="#FF0000"),
            "B": ipywidgets.ColorPicker(value="#00FF00"),
        }
        return painter

    def setUp(self):
        clear_cell_colors()

    def tearDown(self):
        clear_cell_colors()

    def test_first_apply_registers_all_classes(self):
        cell_table = pd.DataFrame({
            "fov": ["FOV_001", "FOV_002"],
            "label": [1, 2],
            "cell_type": ["A", "B"],
        })
        painter = self._make_painter(cell_table)

        with patch("ueler.viewer.plugin.mask_painter.set_cell_colors_bulk") as mock_bulk:
            painter.apply_colors_to_masks(None, register_globally=True)
        # Both classes must have triggered a bulk write (called once each)
        self.assertEqual(mock_bulk.call_count, 2)

    def test_second_apply_same_colors_skips_all_registration(self):
        """Re-applying with unchanged colors must skip set_cell_colors_bulk entirely."""
        cell_table = pd.DataFrame({
            "fov": ["FOV_001", "FOV_002"],
            "label": [1, 2],
            "cell_type": ["A", "B"],
        })
        painter = self._make_painter(cell_table)

        # First apply populates the dirty cache
        painter.apply_colors_to_masks(None, register_globally=True)

        with patch("ueler.viewer.plugin.mask_painter.set_cell_colors_bulk") as mock_bulk:
            painter.apply_colors_to_masks(None, register_globally=True)
        mock_bulk.assert_not_called()

    def test_color_change_triggers_re_registration_only_for_changed_class(self):
        """Only the class whose color changed should re-register."""
        import ipywidgets

        cell_table = pd.DataFrame({
            "fov": ["FOV_001", "FOV_001"],
            "label": [1, 2],
            "cell_type": ["A", "B"],
        })
        painter = self._make_painter(cell_table)

        # First apply — populates dirty cache
        painter.apply_colors_to_masks(None, register_globally=True)

        # Change only class A's color
        painter.class_color_controls["A"].value = "#FFFFFF"

        with patch("ueler.viewer.plugin.mask_painter.set_cell_colors_bulk") as mock_bulk:
            painter.apply_colors_to_masks(None, register_globally=True)

        # Exactly one call (for class A only; class B color unchanged)
        self.assertEqual(mock_bulk.call_count, 1)

    def test_cell_table_change_resets_dirty_cache(self):
        cell_table = pd.DataFrame({
            "fov": ["FOV_001"],
            "label": [1],
            "cell_type": ["A"],
        })
        painter = self._make_painter(cell_table)
        painter.apply_colors_to_masks(None, register_globally=True)
        self.assertNotEqual(painter._last_applied_class_colors, {})

        painter.on_cell_table_change()
        self.assertEqual(painter._last_applied_class_colors, {})


if __name__ == "__main__":
    unittest.main()
