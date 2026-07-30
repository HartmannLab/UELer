import importlib
import sys
import unittest

from ueler import ensure_compat_aliases
from ueler._compat import SHIM_ALIAS_MAP


class TestShimImportCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._restored_modules = {}
        # Clear stub modules without origins so the real modules reload for these
        # checks. Both sides matter: other test modules inject stubs under the
        # legacy alias names as well as under the canonical ones.
        for name in (*SHIM_ALIAS_MAP.keys(), *SHIM_ALIAS_MAP.values()):
            module = sys.modules.get(name)
            if module is None:
                continue
            if getattr(module, "__file__", None):
                continue
            cls._restored_modules[name] = module
            sys.modules.pop(name, None)

        ensure_compat_aliases()

    @classmethod
    def tearDownClass(cls):
        # Put the stubs back exactly as they were so later test modules keep the
        # lightweight versions they installed.
        for name, module in getattr(cls, "_restored_modules", {}).items():
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
                # The shim must hand back the very same module object: a second,
                # separately-executed copy would duplicate every class in it.
                self.assertIs(alias_module, target_module)

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

    def test_image_utils_alias_and_canonical_module(self):
        legacy_image_utils = importlib.import_module("image_utils")
        shim_image_utils = importlib.import_module("ueler.image_utils")

        self.assertIs(shim_image_utils, legacy_image_utils)
        self.assertTrue(callable(shim_image_utils.calculate_downsample_factor))


if __name__ == "__main__":  # pragma: no cover - unittest entrypoint
    unittest.main()
