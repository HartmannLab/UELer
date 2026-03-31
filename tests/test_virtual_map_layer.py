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


if __name__ == "__main__":
    unittest.main()