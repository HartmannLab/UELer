from pathlib import Path
import unittest

import numpy as np

import tests.bootstrap  # noqa: F401

from ueler.viewer.map_descriptor_loader import MapFOVSpec, SlideDescriptor
from ueler.viewer.virtual_map_layer import VirtualMapLayer


class DummyViewer:
    def __init__(self):
        self.calls = []
        self.signature = 0

    def _render_fov_region(self, fov_name, channels, downsample, region_xy, region_ds):
        self.calls.append((fov_name, channels, downsample, region_xy, region_ds))
        width = max(1, region_ds[1] - region_ds[0])
        height = max(1, region_ds[3] - region_ds[2])
        scale = 1.0 if fov_name == "FOV_A" else 2.0
        return np.full((height, width, 3), scale, dtype=np.float32)

    def _map_state_signature(self, channels, downsample):  # pragma: no cover - simple tuple hash
        return ("sig", tuple(channels), int(downsample), self.signature)


def _build_descriptor():
    specs = (
        MapFOVSpec(
            name="FOV_A",
            slide_id="slide-1",
            center_um=(100.0, 100.0),
            frame_size_px=(100, 100),
            fov_size_um=100.0,
            metadata={},
        ),
        MapFOVSpec(
            name="FOV_B",
            slide_id="slide-1",
            center_um=(250.0, 100.0),
            frame_size_px=(100, 100),
            fov_size_um=100.0,
            metadata={},
        ),
    )
    return SlideDescriptor(
        slide_id="slide-1",
        source_path=Path("dummy.json"),
        export_datetime=None,
        fovs=specs,
    )


class VirtualMapLayerTests(unittest.TestCase):
    def test_render_stitches_tiles_with_gap(self):
        viewer = DummyViewer()
        descriptor = _build_descriptor()
        layer = VirtualMapLayer(viewer, descriptor, allowed_downsample=[1, 2])

        layer.set_viewport(50.0, 300.0, 50.0, 150.0, downsample_factor=1)
        output = layer.render(["DAPI"])

        self.assertEqual(output.shape, (100, 250, 3))
        np.testing.assert_allclose(output[:, :100, :], 1.0)
        np.testing.assert_allclose(output[:, 150:, :], 2.0)
        np.testing.assert_allclose(output[:, 100:150, :], 0.0)
        self.assertEqual({call[0] for call in viewer.calls}, {"FOV_A", "FOV_B"})

    def test_cache_reuses_rendered_tiles(self):
        viewer = DummyViewer()
        descriptor = _build_descriptor()
        layer = VirtualMapLayer(viewer, descriptor, allowed_downsample=[1, 2])

        layer.set_viewport(50.0, 300.0, 50.0, 150.0, downsample_factor=1)
        layer.render(["CD3"])
        initial_call_count = len(viewer.calls)

        layer.render(["CD3"])
        self.assertEqual(len(viewer.calls), initial_call_count)

    def test_invalidate_forces_rerender(self):
        viewer = DummyViewer()
        descriptor = _build_descriptor()
        layer = VirtualMapLayer(viewer, descriptor, allowed_downsample=[1, 2])

        layer.set_viewport(50.0, 300.0, 50.0, 150.0, downsample_factor=1)
        layer.render(["DAPI", "CD3"])
        layer.invalidate_for_fov("FOV_A")
        layer.render(["DAPI", "CD3"])

        calls_for_a = [call for call in viewer.calls if call[0] == "FOV_A"]
        self.assertEqual(len(calls_for_a), 2)

    def test_set_viewport_rejects_invalid_factor(self):
        viewer = DummyViewer()
        descriptor = _build_descriptor()
        layer = VirtualMapLayer(viewer, descriptor, allowed_downsample=[2])

        with self.assertRaises(ValueError):
            layer.set_viewport(0.0, 10.0, 0.0, 10.0, downsample_factor=1)

    def test_signature_change_busts_cache(self):
        viewer = DummyViewer()
        descriptor = _build_descriptor()
        layer = VirtualMapLayer(viewer, descriptor, allowed_downsample=[1])

        layer.set_viewport(50.0, 200.0, 50.0, 150.0, downsample_factor=1)
        viewer.signature = 1
        layer.render(["DAPI"])
        initial_count = len(viewer.calls)
        layer.render(["DAPI"])
        self.assertEqual(len(viewer.calls), initial_count)

        viewer.signature = 2
        layer.render(["DAPI"])
        self.assertGreater(len(viewer.calls), initial_count)

    def test_render_handles_partial_downsample_tiles(self):
        viewer = DummyViewer()
        descriptor = _build_descriptor()
        layer = VirtualMapLayer(viewer, descriptor, allowed_downsample=[1, 2, 4, 8])

        layer.set_viewport(130.0, 270.0, 70.0, 170.0, downsample_factor=8)
        output = layer.render(["CD4"])

        self.assertEqual(output.shape, (13, 18, 3))

        calls_by_fov = {call[0]: call for call in viewer.calls}
        self.assertIn("FOV_A", calls_by_fov)
        self.assertIn("FOV_B", calls_by_fov)

        _, _, _, _, region_ds_a = calls_by_fov["FOV_A"]
        _, _, _, _, region_ds_b = calls_by_fov["FOV_B"]

        width_a = region_ds_a[1] - region_ds_a[0]
        height_a = region_ds_a[3] - region_ds_a[2]
        width_b = region_ds_b[1] - region_ds_b[0]
        height_b = region_ds_b[3] - region_ds_b[2]

        self.assertEqual((height_a, width_a), (10, 3))
        self.assertEqual((height_b, width_b), (10, 9))


def _build_row_descriptor(n_tiles: int) -> "SlideDescriptor":
    """Return a descriptor with *n_tiles* 100×100 µm tiles in a horizontal row."""
    specs = tuple(
        MapFOVSpec(
            name=f"FOV_{i}",
            slide_id="slide-row",
            center_um=(50.0 + i * 100.0, 50.0),
            frame_size_px=(100, 100),
            fov_size_um=100.0,
            metadata={},
        )
        for i in range(n_tiles)
    )
    return SlideDescriptor(
        slide_id="slide-row",
        source_path=Path("dummy_row.json"),
        export_datetime=None,
        fovs=specs,
    )


class TileCapTests(unittest.TestCase):
    """Tests for the cache-aware, ds-scaled tile render cap."""

    def test_render_all_cached_tiles_beyond_limit(self):
        """Cached tiles always render regardless of the uncached tile cap."""
        viewer = DummyViewer()
        viewer._map_render_tile_limit = 2  # very low cap for testing

        layer = VirtualMapLayer(viewer, _build_row_descriptor(5), allowed_downsample=[1])
        layer.set_viewport(0.0, 500.0, 0.0, 100.0, downsample_factor=1)

        # Repeated renders warm the cache: at limit=2, each pass adds ≤2 new tiles.
        # After 3 passes all 5 tiles are cached; 10 passes ensures we are stable.
        for _ in range(10):
            layer.render(["DAPI"])

        # All 5 tiles must appear in the last render's visible set.
        self.assertEqual(set(layer._last_visible_fovs), {f"FOV_{i}" for i in range(5)})

        # A further render must not call _render_fov_region again (all cached).
        calls_before = len(viewer.calls)
        layer.render(["DAPI"])
        self.assertEqual(len(viewer.calls), calls_before)

    def test_render_limits_uncached_tiles_scales_with_ds(self):
        """At ds=1 only 2 tiles render (limit=2); at ds=4 all 5 fit within limit=8."""
        viewer = DummyViewer()
        viewer._map_render_tile_limit = 2  # base limit; effective = 2 * ds_factor

        layer = VirtualMapLayer(viewer, _build_row_descriptor(5), allowed_downsample=[1, 4])

        # ds=1: effective limit = 2 — only 2 of the 5 uncached tiles render.
        layer.set_viewport(0.0, 500.0, 0.0, 100.0, downsample_factor=1)
        layer.render(["DAPI"])
        self.assertEqual(len(viewer.calls), 2)

        # ds=4: effective limit = 8 — all 5 tiles are visible and none are cached at
        # this ds level, so 5 ≤ 8 → all 5 render.
        layer.set_viewport(0.0, 500.0, 0.0, 100.0, downsample_factor=4)
        layer.render(["DAPI"])
        ds4_calls = len(viewer.calls) - 2
        self.assertEqual(ds4_calls, 5)

    def test_allocate_canvas_shape_never_under_allocated(self):
        """Canvas dimensions are always ≥ ceil((extent) / pixel_size)."""
        import math as _math

        viewer = DummyViewer()
        layer = VirtualMapLayer(viewer, _build_row_descriptor(1), allowed_downsample=[1, 2, 4, 8])

        test_cases = [
            # (xmin, xmax, ymin, ymax, ds)
            (0.0, 100.0, 0.0, 100.0, 1),
            (0.0, 100.1, 0.0, 99.9, 1),   # fractional extents
            (0.0, 800.0, 0.0, 600.0, 8),  # large downsampled view
            (0.3, 100.7, 0.2, 50.8, 2),   # sub-pixel offsets
        ]
        for xmin, xmax, ymin, ymax, ds in test_cases:
            # _allocate_canvas is a pure computation; no need to call set_viewport
            canvas = layer._allocate_canvas(xmin, xmax, ymin, ymax, ds)
            pixel_size = 1.0 * ds  # base_pixel_size_um == 1.0 for these fixtures
            min_w = _math.ceil((xmax - xmin) / pixel_size)
            min_h = _math.ceil((ymax - ymin) / pixel_size)
            self.assertGreaterEqual(canvas.shape[1], min_w, msg=f"width too small for {(xmin,xmax,ymin,ymax,ds)}")
            self.assertGreaterEqual(canvas.shape[0], min_h, msg=f"height too small for {(xmin,xmax,ymin,ymax,ds)}")


if __name__ == "__main__":
    unittest.main()