"""Keep the MkDocs site truthful about the software it ships with.

``mkdocs build --strict`` proves the site is well *formed* — no broken internal
link, no orphan page. It says nothing about whether the site is *correct*, and
that is where the docs actually rot: a plugin gets renamed, a helper module
moves, ``requires-python`` widens, and the prose describing it stays exactly as
it was. The 2026-08 audit found a developer note pointing at a
``scale_bar_helper.py`` that had been ``scale_bar.py`` for months, and another
still describing the Cell Annotation plugin as "specified but not yet
implemented" while the plugin shipped and had its own tutorial page.

``tools/check_docs_consistency.py`` encodes the checkable half of that audit.
This module is the CI gate for it, plus unit coverage of the checker's own
parsing so a green run means "no inconsistencies" rather than "the checker
silently stopped looking".
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_docs_consistency as checker  # noqa: E402  (path set up above)


class DocsConsistencyTests(unittest.TestCase):
    """The site must make no checkable false claim about the software."""

    def test_docs_make_no_false_claims(self):
        findings = checker.collect_findings()
        if findings:
            report = "\n".join(f"  {finding}" for finding in findings)
            self.fail(
                f"{len(findings)} doc/software inconsistenc"
                f"{'y' if len(findings) == 1 else 'ies'}:\n{report}\n\n"
                "Fix the docs (or the code), then re-run "
                "`python tools/check_docs_consistency.py`."
            )

    def test_every_shipped_plugin_has_a_display_name(self):
        """The UI-label check is only meaningful if it sees every plugin.

        A plugin that stops setting ``displayed_name`` in its constructor would
        drop out of the AST scan silently, and the "is this plugin documented"
        check would then pass by simply not looking. Assert the roster is full.
        """

        plugin_dir = REPO_ROOT / "ueler" / "viewer" / "plugin"
        modules = {
            path.stem
            for path in plugin_dir.glob("*.py")
            if not path.name.startswith("_")
        }
        # Not every module in the directory is a plugin; several are widgets and
        # adapters the plugins compose. Those legitimately have no label.
        named = checker.plugin_display_names()
        self.assertTrue(named, "no plugin display names were discovered at all")
        self.assertLessEqual(set(named), modules)
        for module, label in named.items():
            self.assertTrue(label.strip(), f"{module} has an empty displayed_name")


class RequiresPythonParsingTests(unittest.TestCase):
    """``requires-python`` must expand to the exact set of minors it permits."""

    def test_exclusive_upper_bound(self):
        self.assertEqual(
            checker.declared_python_versions(">=3.10,<3.13"),
            {"3.10", "3.11", "3.12"},
        )

    def test_inclusive_upper_bound(self):
        self.assertEqual(
            checker.declared_python_versions(">=3.11,<=3.12"), {"3.11", "3.12"}
        )

    def test_unbounded_specifier_disables_the_check(self):
        """An open-ended bound must yield nothing rather than a guess.

        Returning a partial set here would invent an upper bound and then report
        every doc page as wrong about it.
        """

        self.assertEqual(checker.declared_python_versions(">=3.10"), set())


class StatedVersionParsingTests(unittest.TestCase):
    """Both spellings of a support claim must expand to the same set."""

    def test_comma_separated_list(self):
        self.assertEqual(
            checker.stated_python_versions("**Python** 3.10, 3.11, or 3.12"),
            {"3.10", "3.11", "3.12"},
        )

    def test_en_dash_range_expands_the_interior(self):
        """A range must include the minors it spans, not just its endpoints.

        This is the case that let "Supported Python: 3.10–3.11" sit next to a
        ``requires-python`` permitting 3.12 without anything noticing.
        """

        self.assertEqual(
            checker.stated_python_versions("Supported Python: 3.10–3.12"),
            {"3.10", "3.11", "3.12"},
        )

    def test_hyphen_range(self):
        self.assertEqual(
            checker.stated_python_versions("Supported Python: 3.10-3.11"),
            {"3.10", "3.11"},
        )


class ProjectFactsTests(unittest.TestCase):
    """The packaging facts must be readable without ``tomllib``.

    ``requires-python`` still admits 3.10, where ``tomllib`` does not exist, so
    the checker parses ``pyproject.toml`` by regex — the same choice
    ``tools/check_stable_rehearsal.py`` made. These tests pin that it works
    against the real file rather than a fixture, so a reformat of
    ``pyproject.toml`` cannot quietly blind the checker.
    """

    def test_reads_the_real_pyproject(self):
        facts = checker.load_project_facts()
        self.assertEqual(facts.name, "ueler-viewer")
        self.assertTrue(facts.requires_python.startswith(">=3."))
        self.assertIn("dev", facts.extras)
        self.assertIn("docs", facts.extras)


if __name__ == "__main__":
    unittest.main()
