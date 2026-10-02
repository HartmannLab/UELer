"""Tests for the ``anywidget`` browser-extension startup check.

The user-visible failure is a frontend notification — ``Unable to find widget
'anywidget' version '~0.9.*' from configured widget sources ["local"]`` — with
nothing in the kernel log, so these tests pin both halves of the contract: the
three genuinely broken layouts are reported, and *every* other case is silent.
The silence half is the one that matters most: a false alarm on a healthy
JupyterLab install would be worse than the problem being diagnosed.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from ueler.viewer import widget_frontend_check as wfc
from ueler.viewer.widget_frontend_check import (
    FrontendStatus,
    check_anywidget_frontend,
    warn_about_widget_frontend,
)


def _fake_anywidget(version="0.9.13", required="~0.9.*"):
    """A stand-in for a real ``anywidget`` install.

    The suite's bootstrap replaces ``anywidget`` with a placeholder that has no
    ``AnyWidget``, so every test that needs the populated path injects this.
    """
    module = ModuleType("anywidget")
    module.__version__ = version
    module.AnyWidget = type(
        "AnyWidget",
        (),
        {"_model_module_version": SimpleNamespace(default_value=required)},
    )
    return module


def _make_assets(root, kinds, version=None):
    """Create ``<root>/share/jupyter/<kind>/anywidget`` trees.

    Returns ``{kind: <search path entry>}``, mirroring what ``jupyter_path``
    yields — the directory *containing* the per-module folders.
    """
    roots = {}
    for kind in kinds:
        directory = Path(root) / "share" / "jupyter" / kind / "anywidget"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.js").write_text("// asset", encoding="utf-8")
        if version is not None and kind == "labextensions":
            (directory / "package.json").write_text(
                json.dumps({"name": "anywidget", "version": version}), encoding="utf-8"
            )
        roots[kind] = str(directory.parent)
    return roots


class VersionMatchingTests(unittest.TestCase):
    def test_npm_tilde_ranges_match_on_the_fixed_prefix(self):
        self.assertTrue(wfc._satisfies("0.9.13", "~0.9.*"))
        self.assertTrue(wfc._satisfies("0.11.0", "~0.11.*"))
        self.assertFalse(wfc._satisfies("0.11.0", "~0.9.*"))
        self.assertFalse(wfc._satisfies("0.9.0", "~0.11.*"))

    def test_anything_unrecognised_counts_as_satisfied(self):
        """The silence rule: never warn on a range shape we cannot parse."""
        self.assertTrue(wfc._satisfies("1.2.3", "not-a-range"))
        self.assertTrue(wfc._satisfies("1.2.3", ""))
        self.assertTrue(wfc._satisfies(None, "~0.9.*"))
        self.assertTrue(wfc._satisfies("1.2.3", None))

    def test_version_prefix_stops_at_the_wildcard(self):
        self.assertEqual(wfc._version_prefix("~0.9.*"), (0, 9))
        self.assertEqual(wfc._version_prefix("^1.2.x"), (1, 2))
        self.assertIsNone(wfc._version_prefix("*"))
        self.assertIsNone(wfc._version_prefix("latest"))


class SysPrefixTests(unittest.TestCase):
    def test_paths_inside_the_running_prefix_are_recognised(self):
        inside = str(Path(sys.prefix) / "share" / "jupyter" / "nbextensions")
        self.assertTrue(wfc._under_sys_prefix(inside))

    def test_paths_outside_the_running_prefix_are_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            self.assertFalse(wfc._under_sys_prefix(outside))


class CheckAnywidgetFrontendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _run(self, roots, prefix, anywidget=None):
        """Run the check against a synthetic search path and ``sys.prefix``."""
        module = anywidget if anywidget is not None else _fake_anywidget()
        with mock.patch.dict(sys.modules, {"anywidget": module}):
            with mock.patch.object(sys, "prefix", str(prefix)):
                with mock.patch(
                    "jupyter_core.paths.jupyter_path",
                    lambda kind: list(roots.get(kind, [])),
                ):
                    return check_anywidget_frontend()

    def test_assets_in_the_kernel_environment_are_ok(self):
        env = self.root / "env"
        roots = _make_assets(env, ["labextensions", "nbextensions"], version="0.9.13")
        status = self._run({k: [v] for k, v in roots.items()}, env)
        self.assertEqual(status.code, "ok")
        self.assertFalse(status.is_problem)

    def test_install_outside_the_kernel_environment_is_flagged(self):
        """The reported case: ``pip install --user`` into a shared env.

        JupyterLab reads the whole ``jupyter_path()`` chain and works; VS Code's
        'local' source only reads ``<sys.prefix>/share/jupyter/nbextensions``
        and does not.
        """
        user = self.root / "home_local"
        env = self.root / "env"
        env.mkdir()
        roots = _make_assets(user, ["labextensions", "nbextensions"], version="0.9.13")
        status = self._run({k: [v] for k, v in roots.items()}, env)

        self.assertEqual(status.code, "outside-env")
        self.assertTrue(status.is_problem)
        # The message has to carry both remedies and the path actually found,
        # because the frontend notification carries none of them.
        self.assertIn("home_local", status.message)
        self.assertIn("nbextensions", status.message)
        self.assertIn("jupyter.widgetScriptSources", status.message)

    def test_no_assets_anywhere_is_flagged(self):
        env = self.root / "env"
        env.mkdir()
        status = self._run({}, env)
        self.assertEqual(status.code, "missing-assets")
        self.assertTrue(status.is_problem)

    def test_version_skew_between_python_and_frontend_is_flagged(self):
        """Two overlapping installs: the asset is local but the wrong version."""
        env = self.root / "env"
        roots = _make_assets(env, ["labextensions", "nbextensions"], version="0.9.13")
        status = self._run(
            {k: [v] for k, v in roots.items()},
            env,
            anywidget=_fake_anywidget(version="0.11.0", required="~0.11.*"),
        )
        self.assertEqual(status.code, "version-skew")
        self.assertTrue(status.is_problem)
        self.assertIn("0.9.13", status.message)

    def test_matching_local_asset_wins_over_a_skewed_one_elsewhere(self):
        """A stale copy on the search path must not trigger a false alarm."""
        env = self.root / "env"
        stale = self.root / "stale"
        env_roots = _make_assets(
            env, ["labextensions", "nbextensions"], version="0.11.0"
        )
        stale_roots = _make_assets(stale, ["labextensions"], version="0.9.13")
        roots = {
            "labextensions": [stale_roots["labextensions"], env_roots["labextensions"]],
            "nbextensions": [env_roots["nbextensions"]],
        }
        status = self._run(
            roots, env, anywidget=_fake_anywidget(version="0.11.0", required="~0.11.*")
        )
        self.assertEqual(status.code, "ok")

    def test_unreadable_manifest_does_not_warn(self):
        env = self.root / "env"
        roots = _make_assets(env, ["labextensions", "nbextensions"], version=None)
        status = self._run({k: [v] for k, v in roots.items()}, env)
        # No version to compare against, so no claim is made.
        self.assertEqual(status.code, "ok")

    def test_missing_jupyter_core_is_undetermined_not_a_problem(self):
        env = self.root / "env"
        with mock.patch.dict(sys.modules, {"anywidget": _fake_anywidget()}):
            with mock.patch.object(sys, "prefix", str(env)):
                with mock.patch.dict(sys.modules, {"jupyter_core.paths": None}):
                    status = check_anywidget_frontend()
        self.assertEqual(status.code, "undetermined")
        self.assertFalse(status.is_problem)


class AbsentAnywidgetTests(unittest.TestCase):
    def test_the_suite_bootstrap_stub_is_absent_and_silent(self):
        """The suite replaces anywidget with a placeholder; that is not a fault.

        The plugins' own ``ANYWIDGET_AVAILABLE`` fallbacks cover this case, so
        the check must not add a warning on top of them.
        """
        status = check_anywidget_frontend()
        self.assertEqual(status.code, "absent")
        self.assertFalse(status.is_problem)

    def test_uninstalled_anywidget_is_absent(self):
        # ``None`` in sys.modules is what makes ``import anywidget`` raise.
        with mock.patch.dict(sys.modules, {"anywidget": None}):
            status = check_anywidget_frontend()
        self.assertEqual(status.code, "absent")
        self.assertFalse(status.is_problem)


class WarnAboutWidgetFrontendTests(unittest.TestCase):
    def test_a_problem_is_logged_as_a_warning(self):
        status = FrontendStatus("outside-env", "assets are in the wrong place")
        with self.assertLogs("ueler.viewer.widget_frontend_check", level="WARNING") as logs:
            returned = warn_about_widget_frontend(status)
        self.assertIs(returned, status)
        self.assertIn("assets are in the wrong place", logs.output[0])

    def test_a_healthy_check_logs_nothing_above_debug(self):
        status = FrontendStatus("ok")
        with self.assertLogs("ueler.viewer.widget_frontend_check", level="DEBUG") as logs:
            warn_about_widget_frontend(status)
        self.assertTrue(all("WARNING" not in line for line in logs.output))

    def test_a_failing_check_cannot_stop_the_viewer_opening(self):
        with mock.patch.object(
            wfc, "check_anywidget_frontend", side_effect=RuntimeError("boom")
        ):
            status = warn_about_widget_frontend()
        self.assertEqual(status.code, "undetermined")
        self.assertFalse(status.is_problem)


class ViewerRunsTheCheckTests(unittest.TestCase):
    def test_main_viewer_calls_the_check_during_init(self):
        import inspect

        from ueler.viewer import main_viewer

        source = inspect.getsource(main_viewer.ImageMaskViewer.__init__)
        self.assertIn("warn_about_widget_frontend()", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
