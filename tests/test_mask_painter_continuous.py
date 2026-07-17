"""Tests for continuous (gradient) cell-mask coloring in the Mask Painter (issue #115)."""

import types
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ueler.rendering import get_cell_color, clear_cell_colors, MaskPainterSnapshot
from ueler.viewer.plugin.mask_painter import (
    build_painter_state_maps_for_fov,
    compute_continuous_colors,
    resolve_continuous_range,
)


def _cmap_hex(colormap, t):
    """Expected hex for a normalized position t in [0, 1] of a matplotlib colormap."""
    import matplotlib
    from matplotlib.colors import to_hex

    return to_hex(matplotlib.colormaps[colormap](t))


def _make_viewer():
    """Minimal mock viewer with a float feature column for continuous coloring."""
    cell_table = pd.DataFrame({
        "fov": ["FOV_001", "FOV_001", "FOV_002", "FOV_002"],
        "label": [1, 2, 3, 4],
        "cell_type": ["TypeA", "TypeB", "TypeA", "TypeB"],
        "CD3": [0.0, 5.0, 10.0, 2.5],
    })
    viewer = types.SimpleNamespace()
    viewer.cell_table = cell_table
    viewer.fov_key = "fov"
    viewer.label_key = "label"
    viewer.mask_key = "cell"
    viewer.base_folder = Path.cwd()
    viewer.mask_outline_thickness = 1
    viewer.current_downsample_factor = 1
    viewer.update_display = lambda ds=1: None

    ui_component = types.SimpleNamespace()
    ui_component.image_selector = types.SimpleNamespace()
    ui_component.image_selector.value = "FOV_001"
    ui_component.mask_color_controls = {"cell": types.SimpleNamespace(value="Green")}
    viewer.ui_component = ui_component
    viewer.predefined_colors = {"Green": "#00FF00", "White": "#FFFFFF"}

    image_display = types.SimpleNamespace()
    image_display.set_mask_colors_current_fov = lambda **kwargs: None
    viewer.image_display = image_display

    viewer.get_active_fov = lambda: ui_component.image_selector.value
    return viewer


class TestComputeContinuousColors(unittest.TestCase):
    def test_basic_mapping(self):
        colors = compute_continuous_colors(
            [0.0, 5.0, 10.0], [1, 2, 3], colormap="viridis", vmin=0.0, vmax=10.0
        )
        self.assertEqual(set(colors), {1, 2, 3})
        self.assertEqual(colors[1], _cmap_hex("viridis", 0.0))
        self.assertEqual(colors[3], _cmap_hex("viridis", 1.0))
        for hex_value in colors.values():
            self.assertRegex(hex_value, r"^#[0-9a-fA-F]{6}$")

    def test_nan_values_skipped(self):
        colors = compute_continuous_colors(
            [0.0, np.nan, 10.0], [1, 2, 3], colormap="viridis", vmin=0.0, vmax=10.0
        )
        self.assertNotIn(2, colors)
        self.assertEqual(set(colors), {1, 3})

    def test_all_nan_returns_empty(self):
        colors = compute_continuous_colors(
            [np.nan, np.nan], [1, 2], colormap="viridis", vmin=0.0, vmax=1.0
        )
        self.assertEqual(colors, {})

    def test_arcsinh_negatives_monotonic(self):
        values = [-3.0, 0.0, 3.0]
        vmin, vmax = resolve_continuous_range(values, arcsinh=True, cofactor=1.0)
        colors = compute_continuous_colors(
            values, [1, 2, 3], colormap="viridis", vmin=vmin, vmax=vmax, arcsinh=True, cofactor=1.0
        )
        self.assertEqual(set(colors), {1, 2, 3})
        # Distinct colors across a monotonic transform.
        self.assertEqual(len(set(colors.values())), 3)

    def test_clipping_out_of_range(self):
        # Values beyond vmax clip to the top of the colormap.
        colors = compute_continuous_colors(
            [100.0], [7], colormap="viridis", vmin=0.0, vmax=10.0
        )
        self.assertEqual(colors[7], _cmap_hex("viridis", 1.0))


class TestResolveContinuousRange(unittest.TestCase):
    def test_constant_column_widened(self):
        vmin, vmax = resolve_continuous_range([4.0, 4.0, 4.0])
        self.assertGreater(vmax, vmin)

    def test_all_nan_fallback(self):
        vmin, vmax = resolve_continuous_range([np.nan, np.nan])
        self.assertTrue(np.isfinite(vmin) and np.isfinite(vmax))
        self.assertEqual((vmin, vmax), (0.0, 1.0))

    def test_percentile_range(self):
        vmin, vmax = resolve_continuous_range(list(range(101)), lo_pct=1.0, hi_pct=99.0)
        self.assertAlmostEqual(vmin, 1.0, places=6)
        self.assertAlmostEqual(vmax, 99.0, places=6)


class TestBuildStateMapsContinuous(unittest.TestCase):
    def _table(self):
        return pd.DataFrame({
            "fov": ["FOV_001", "FOV_001", "FOV_002"],
            "label": [1, 2, 3],
            "CD3": [0.0, 10.0, 5.0],
        })

    def test_continuous_branch_keys_to_fov(self):
        spec = {
            "column": "CD3", "colormap": "viridis", "vmin": 0.0, "vmax": 10.0,
            "arcsinh": False, "cofactor": 5.0, "opacity": 100, "fill": True,
        }
        color_map, border_map, mode_map, opacity_map = build_painter_state_maps_for_fov(
            cell_table=self._table(), fov_key="fov", label_key="label", fov="FOV_001",
            identifier=None, active_classes=(), class_colors={}, class_visible={},
            class_fill={}, class_opacity={}, default_color="#FFFFFF", global_fill_opacity=35,
            continuous=spec,
        )
        self.assertEqual(set(color_map), {1, 2})
        self.assertEqual(color_map[1], _cmap_hex("viridis", 0.0))
        self.assertEqual(color_map[2], _cmap_hex("viridis", 1.0))
        self.assertTrue(all(m == "fill" for m in mode_map.values()))
        self.assertTrue(all(abs(a - 1.0) < 1e-9 for a in opacity_map.values()))
        self.assertEqual(border_map, {})

    def test_outline_mode_has_no_opacity(self):
        spec = {
            "column": "CD3", "colormap": "viridis", "vmin": 0.0, "vmax": 10.0,
            "arcsinh": False, "cofactor": 5.0, "opacity": 50, "fill": False,
        }
        _, _, mode_map, opacity_map = build_painter_state_maps_for_fov(
            cell_table=self._table(), fov_key="fov", label_key="label", fov="FOV_002",
            identifier=None, active_classes=(), class_colors={}, class_visible={},
            class_fill={}, class_opacity={}, default_color="#FFFFFF", global_fill_opacity=35,
            continuous=spec,
        )
        self.assertTrue(all(m == "outline" for m in mode_map.values()))
        self.assertEqual(opacity_map, {})


class TestSnapshotRoundtrip(unittest.TestCase):
    def test_snapshot_carries_continuous_fields(self):
        snap = MaskPainterSnapshot(
            mask_name="cell", identifier="", active_classes=(), class_colors={},
            class_visible={}, class_fill={}, class_opacity={}, default_color="#FFFFFF",
            color_mode="continuous", continuous_column="CD3", colormap="magma",
            vmin=1.0, vmax=9.0, arcsinh=True, arcsinh_cofactor=5.0,
            continuous_opacity=80, continuous_fill=True,
        )
        data = asdict(snap)
        self.assertEqual(data["color_mode"], "continuous")
        self.assertEqual(data["continuous_column"], "CD3")
        self.assertEqual(data["colormap"], "magma")
        rebuilt = MaskPainterSnapshot(**data)
        self.assertEqual(rebuilt.vmin, 1.0)
        self.assertEqual(rebuilt.vmax, 9.0)
        self.assertTrue(rebuilt.arcsinh)

    def test_replay_matches_direct_compute(self):
        table = pd.DataFrame({
            "fov": ["F1", "F1"], "label": [1, 2], "CD3": [2.0, 8.0],
        })
        spec = {
            "column": "CD3", "colormap": "plasma", "vmin": 0.0, "vmax": 10.0,
            "arcsinh": False, "cofactor": 5.0, "opacity": 100, "fill": True,
        }
        replayed, _, _, _ = build_painter_state_maps_for_fov(
            cell_table=table, fov_key="fov", label_key="label", fov="F1",
            identifier=None, active_classes=(), class_colors={}, class_visible={},
            class_fill={}, class_opacity={}, default_color="#FFFFFF", global_fill_opacity=35,
            continuous=spec,
        )
        direct = compute_continuous_colors(
            [2.0, 8.0], [1, 2], colormap="plasma", vmin=0.0, vmax=10.0
        )
        self.assertEqual(replayed, direct)


class TestPainterContinuousMode(unittest.TestCase):
    def setUp(self):
        clear_cell_colors()
        self.viewer = _make_viewer()

    def tearDown(self):
        clear_cell_colors()

    def _make_painter(self):
        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay

        return MaskPainterDisplay(self.viewer, width=400, height=300)

    def test_continuous_options_are_float_columns(self):
        painter = self._make_painter()
        self.assertIn("CD3", list(painter.ui_component.continuous_column_dropdown.options))
        # cell_type (categorical) must NOT appear in continuous options.
        self.assertNotIn("cell_type", list(painter.ui_component.continuous_column_dropdown.options))

    def test_toggle_hides_categorical_layout(self):
        painter = self._make_painter()
        # Stub widgets don't fire observers, so invoke the handler directly.
        painter._on_color_mode_change({"new": "continuous"})
        self.assertEqual(painter.ui_component.categorical_layout.layout.display, "none")
        self.assertEqual(painter.ui_component.continuous_layout.layout.display, "block")
        painter._on_color_mode_change({"new": "categorical"})
        self.assertEqual(painter.ui_component.categorical_layout.layout.display, "block")
        self.assertEqual(painter.ui_component.continuous_layout.layout.display, "none")

    def _setup_continuous(self, painter, colormap="viridis", vmin=0.0, vmax=10.0):
        painter.ui_component.color_mode_toggle.value = "continuous"
        painter.ui_component.continuous_column_dropdown.value = "CD3"
        painter.ui_component.colormap_dropdown.value = colormap
        painter.ui_component.auto_range_checkbox.value = False
        painter.ui_component.vmin_input.value = vmin
        painter.ui_component.vmax_input.value = vmax

    def test_apply_continuous_does_not_populate_global_registry(self):
        """Issue #115 reply: continuous apply must not register every cell in every
        FOV (that was the multi-minute stall). Colors are resolved on demand."""
        import ueler.viewer.plugin.mask_painter as mp
        from unittest.mock import patch

        painter = self._make_painter()
        self._setup_continuous(painter)

        with patch.object(mp, "set_cell_colors_bulk") as bulk:
            painter.apply_colors_to_masks(None, notify_cell_gallery=False, register_globally=True)
            bulk.assert_not_called()

        # Registry is not populated for any FOV (unregistered cells → falsy).
        self.assertFalse(get_cell_color("FOV_001", 1))
        self.assertFalse(get_cell_color("FOV_002", 3))

    def test_continuous_colors_resolve_on_demand(self):
        """Per-FOV colors come from the effective-state-maps path, not the registry."""
        painter = self._make_painter()
        self._setup_continuous(painter)
        painter.apply_colors_to_masks(None, notify_cell_gallery=False, register_globally=True)

        color_map, _, mode_map, _ = painter.get_effective_state_maps_for_fov("FOV_002")
        # FOV_002 label 3 has CD3 == 10.0 → top of viridis; label 4 == 2.5 → 0.25.
        self.assertEqual(color_map[3], _cmap_hex("viridis", 1.0))
        self.assertEqual(color_map[4], _cmap_hex("viridis", 0.25))
        self.assertTrue(all(m == "fill" for m in mode_map.values()))

    def test_auto_range_cached_off_hot_path(self):
        """resolve_continuous_range runs once per (column, arcsinh, cofactor); the
        render hot path (_build_continuous_spec) reuses the cache and writes no widgets."""
        import ueler.viewer.plugin.mask_painter as mp
        from unittest.mock import patch

        painter = self._make_painter()
        painter.ui_component.color_mode_toggle.value = "continuous"
        painter.ui_component.continuous_column_dropdown.value = "CD3"
        painter.ui_component.auto_range_checkbox.value = True

        with patch.object(mp, "resolve_continuous_range", wraps=mp.resolve_continuous_range) as rng:
            painter._refresh_auto_range_fields()  # handler → computes + caches once
            self.assertEqual(rng.call_count, 1)
            # Repeated hot-path spec builds must not recompute the range.
            painter._build_continuous_spec()
            painter._build_continuous_spec()
            self.assertEqual(rng.call_count, 1)

        # _build_continuous_spec must not mutate the vmin/vmax widgets.
        painter.ui_component.vmin_input.value = 123.0
        painter._build_continuous_spec()
        self.assertEqual(painter.ui_component.vmin_input.value, 123.0)

    def test_apply_sets_busy_state(self):
        """apply_colors_to_masks toggles the status bar to processing then ready."""
        import ueler.viewer.decorators as dec
        from unittest.mock import patch

        painter = self._make_painter()
        self._setup_continuous(painter)

        calls = []
        with patch.object(dec, "_set_status_bar", side_effect=lambda v, state, **k: calls.append(state)):
            painter.apply_colors_to_masks(None, notify_cell_gallery=False)
        self.assertEqual(calls[0], "processing")
        self.assertEqual(calls[-1], "ready")

    def test_palette_payload_roundtrip(self):
        painter = self._make_painter()
        painter.ui_component.color_mode_toggle.value = "continuous"
        painter.ui_component.continuous_column_dropdown.value = "CD3"
        painter.ui_component.colormap_dropdown.value = "magma"
        painter.ui_component.arcsinh_checkbox.value = True
        painter.ui_component.arcsinh_cofactor_input.value = 5.0
        painter.ui_component.auto_range_checkbox.value = False
        painter.ui_component.vmin_input.value = 1.0
        painter.ui_component.vmax_input.value = 7.0
        painter.ui_component.continuous_opacity_input.value = 75
        painter.ui_component.continuous_fill_checkbox.value = False

        payload = painter._build_color_set_payload("my_gradient", "2026-01-01T00:00:00Z")
        self.assertEqual(payload["color_mode"], "continuous")
        cont = payload["continuous"]
        self.assertEqual(cont["column"], "CD3")
        self.assertEqual(cont["colormap"], "magma")
        self.assertTrue(cont["arcsinh"])

        # Reset then restore from payload.
        painter.ui_component.color_mode_toggle.value = "categorical"
        painter._apply_continuous_payload(payload, Path("dummy.maskcolors.json"))
        self.assertEqual(painter.ui_component.color_mode_toggle.value, "continuous")
        self.assertEqual(painter.ui_component.continuous_column_dropdown.value, "CD3")
        self.assertEqual(painter.ui_component.colormap_dropdown.value, "magma")
        self.assertTrue(painter.ui_component.arcsinh_checkbox.value)
        self.assertEqual(painter.ui_component.vmin_input.value, 1.0)
        self.assertEqual(painter.ui_component.vmax_input.value, 7.0)
        self.assertEqual(painter.ui_component.continuous_opacity_input.value, 75)
        self.assertFalse(painter.ui_component.continuous_fill_checkbox.value)


if __name__ == "__main__":
    unittest.main()
