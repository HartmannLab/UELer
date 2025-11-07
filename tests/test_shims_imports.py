import importlib
import sys
import unittest

from ueler import ensure_compat_aliases
from ueler._compat import SHIM_ALIAS_MAP


class TestShimImportCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._restored_modules = {}
        # Clear stub modules without origins so the real modules reload for these checks.
        for target in SHIM_ALIAS_MAP.values():
            module = sys.modules.get(target)
            if module is None:
                continue
            if getattr(module, "__file__", None):
                continue
            cls._restored_modules[target] = module
            sys.modules.pop(target, None)

        ensure_compat_aliases()

    @classmethod
    def tearDownClass(cls):
        for name, module in getattr(cls, "_restored_modules", {}).items():
            if name not in sys.modules:
                sys.modules[name] = module

    def test_aliases_mirror_legacy_modules(self):
        for alias, target in SHIM_ALIAS_MAP.items():
            with self.subTest(alias=alias, target=target):
                target_module = None
                target_error = None
                try:
                    target_module = importlib.import_module(target)
                except Exception as exc:  # pragma: no cover - defensive guard
                    target_error = exc

                if target_error is not None:
                    with self.assertRaises(target_error.__class__):
                        importlib.import_module(alias)
                    continue

                alias_module = importlib.import_module(alias)
                self.assertIs(alias_module, sys.modules.get(alias))
                self.assertIs(target_module, sys.modules.get(target))
                if alias_module is target_module:
                    continue
                self.assertEqual(alias_module.__name__, target_module.__name__)

    def test_from_import_core_symbol(self):
        try:
            from viewer.main_viewer import ImageMaskViewer as LegacyMaskViewer
        except Exception as exc:  # pragma: no cover - environment guard
            self.skipTest(f"viewer.main_viewer unavailable: {exc!r}")

        from ueler.viewer.main_viewer import ImageMaskViewer  # type: ignore[import-error]

        self.assertIs(ImageMaskViewer, LegacyMaskViewer)

    def test_from_import_plugin_symbol(self):
        try:
            from viewer.plugin.chart import ChartDisplay as LegacyChartDisplay
        except Exception as exc:  # pragma: no cover - environment guard
            self.skipTest(f"viewer.plugin.chart unavailable: {exc!r}")

        module = sys.modules.get("viewer.plugin.chart")
        if getattr(module, "__file__", None) is None:
            self.skipTest("viewer.plugin.chart is stubbed; alias equality not enforceable")

        from ueler.viewer.plugin.chart import ChartDisplay  # type: ignore[import-error]
        if ChartDisplay.__module__ != LegacyChartDisplay.__module__:
            self.skipTest("ChartDisplay replaced by a test stub")

        self.assertIs(ChartDisplay, LegacyChartDisplay)

    def test_top_level_utility_alias(self):
        legacy_constants = importlib.import_module("constants")
        shim_constants = importlib.import_module("ueler.constants")

        self.assertIs(shim_constants, legacy_constants)


if __name__ == "__main__":  # pragma: no cover - unittest entrypoint
    unittest.main()
