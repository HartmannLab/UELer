"""Regression tests for the package-wide logging sweep.

Verifies that the plugins' status-label / log helpers mirror their messages
into the ``ueler`` logger (which feeds the bottom log console), in addition to
updating their inline widgets.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import tests.bootstrap  # noqa: F401 — installs the ipywidgets stub

import ipywidgets as widgets


class TestStatusHelpersMirrorToLog(unittest.TestCase):
    def test_cell_annotation_set_status(self):
        from ueler.viewer.plugin.cell_annotation import CellAnnotationPlugin

        plugin = CellAnnotationPlugin.__new__(CellAnnotationPlugin)
        plugin.status_label = SimpleNamespace(value="")
        with self.assertLogs("ueler", level="INFO") as cm:
            plugin._set_status("hello world")
        self.assertTrue(any("hello world" in line for line in cm.output))
        with self.assertLogs("ueler", level="WARNING") as cm:
            plugin._set_status("bad thing", error=True)
        self.assertTrue(any("bad thing" in line for line in cm.output))

    def test_mask_painter_log(self):
        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay

        plugin = MaskPainterDisplay.__new__(MaskPainterDisplay)
        plugin.ui_component = SimpleNamespace(feedback_label=SimpleNamespace(value=""))
        with self.assertLogs("ueler", level="INFO") as cm:
            plugin._log("painted ok")
        self.assertTrue(any("painted ok" in line for line in cm.output))
        with self.assertLogs("ueler", level="WARNING") as cm:
            plugin._log("painter problem", error=True)
        self.assertTrue(any("painter problem" in line for line in cm.output))

    def test_roi_manager_set_status(self):
        from ueler.viewer.plugin.roi_manager_plugin import ROIManagerPlugin

        plugin = ROIManagerPlugin.__new__(ROIManagerPlugin)
        plugin.ui_component = SimpleNamespace(status=SimpleNamespace(value=""))
        with self.assertLogs("ueler", level="WARNING") as cm:
            plugin.set_status("roi warning", level="warning")
        self.assertTrue(any("roi warning" in line for line in cm.output))

    def test_export_fovs_log(self):
        from ueler.viewer.plugin.export_fovs import BatchExportPlugin

        plugin = BatchExportPlugin.__new__(BatchExportPlugin)
        plugin.ui_component = SimpleNamespace(log_output=widgets.Output())
        with self.assertLogs("ueler", level="INFO") as cm:
            plugin._log("export step")
        self.assertTrue(any("export step" in line for line in cm.output))


if __name__ == "__main__":
    unittest.main()
