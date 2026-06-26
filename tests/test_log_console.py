"""Unit tests for the bottom-docked UELer log console (debug mode)."""

from __future__ import annotations

import logging
import unittest

import tests.bootstrap  # noqa: F401 — installs the ipywidgets stub

import ueler.viewer.log_console as log_console
from ueler.viewer.log_console import (
    OutputWidgetHandler,
    build_log_console_panel,
    disable_log_console,
    enable_log_console,
    get_log_console_handler,
)


def _make_record(msg: str, level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord(
        name="ueler.viewer.plugin.heatmap_layers",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


class TestOutputWidgetHandler(unittest.TestCase):
    def setUp(self):
        self.handler = OutputWidgetHandler()
        self.handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    def test_emit_prepends_newest_first_and_formats(self):
        self.handler.emit(_make_record("first"))
        self.handler.emit(_make_record("second"))
        texts = [o["text"] for o in self.handler.out.outputs]
        self.assertEqual(len(texts), 2)
        self.assertIn("second", texts[0])  # newest on top
        self.assertIn("first", texts[1])
        self.assertTrue(texts[0].startswith("[INFO] "))

    def test_emit_respects_max_entries(self):
        original = log_console._MAX_ENTRIES
        log_console._MAX_ENTRIES = 3
        try:
            for i in range(10):
                self.handler.emit(_make_record(f"msg{i}"))
            self.assertEqual(len(self.handler.out.outputs), 3)
            # Newest retained, oldest dropped.
            self.assertIn("msg9", self.handler.out.outputs[0]["text"])
        finally:
            log_console._MAX_ENTRIES = original

    def test_clear_logs_empties(self):
        self.handler.emit(_make_record("something"))
        self.assertTrue(self.handler.out.outputs)
        self.handler.clear_logs()
        self.assertEqual(self.handler.out.outputs, ())


class TestEnableDisable(unittest.TestCase):
    def setUp(self):
        self._logger = logging.getLogger("ueler")
        self._saved_handlers = list(self._logger.handlers)
        self._saved_level = self._logger.level
        self._saved_propagate = self._logger.propagate
        log_console._HANDLER = None  # reset singleton

    def tearDown(self):
        self._logger.handlers = self._saved_handlers
        self._logger.setLevel(self._saved_level)
        self._logger.propagate = self._saved_propagate
        log_console._HANDLER = None

    def test_enable_attaches_sets_propagate_and_level(self):
        handler = enable_log_console()
        self.assertIn(handler, self._logger.handlers)
        self.assertFalse(self._logger.propagate)
        self.assertLessEqual(self._logger.level, logging.DEBUG)

    def test_enable_is_idempotent(self):
        h1 = enable_log_console()
        h2 = enable_log_console()
        self.assertIs(h1, h2)
        self.assertEqual(self._logger.handlers.count(h1), 1)

    def test_disable_detaches_and_restores_propagate(self):
        handler = enable_log_console()
        disable_log_console()
        self.assertNotIn(handler, self._logger.handlers)
        self.assertTrue(self._logger.propagate)

    def test_get_handler_is_singleton(self):
        self.assertIs(get_log_console_handler(), get_log_console_handler())


class TestBuildPanel(unittest.TestCase):
    def test_panel_contains_output_widget(self):
        handler = OutputWidgetHandler()
        panel = build_log_console_panel(handler)
        # handler.out is docked directly under the panel (header, out).
        self.assertIn(handler.out, panel.children)


if __name__ == "__main__":
    unittest.main()
