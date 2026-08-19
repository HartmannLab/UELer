"""Tests for tools/check_stable_rehearsal.py — the rc -> stable promotion gate.

These build a throwaway git repository rather than mocking git, because the
subtle part of the check is exactly how a real diff looks: the rc and the stable
release *cannot* be the same commit (both must declare their own version), so the
check has to accept a diff that changes the version lines and nothing else. A
mock would let that distinction pass untested.

The TestPyPI lookup is injected rather than performed, so the suite needs no
network.
"""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rehearsal = _load("check_stable_rehearsal")
RehearsalError = rehearsal.RehearsalError

PYPROJECT = """\
[project]
name = "ueler-viewer"
version = "{version}"
dependencies = [
    "scikit-image>=0.19",
    "numpy",
]
"""

INIT = '''\
"""UELer."""

__version__ = "{version}"

from ueler.viewer import MainViewer  # noqa: F401
'''

VIEWER = '''\
class MainViewer:
    """The viewer.{extra}"""
'''


class _Repo:
    """A minimal repository shaped like this one: `ueler/`, pyproject, docs."""

    def __init__(self, root: Path):
        self.root = root
        self._git("init", "--quiet")
        self._git("config", "user.email", "test@example.invalid")
        self._git("config", "user.name", "Test")
        self._git("config", "commit.gpgsign", "false")

    def _git(self, *args):
        return subprocess.run(
            ("git", *args), cwd=self.root, check=True,
            capture_output=True, text=True,
        ).stdout

    def write(self, relative: str, text: str):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def commit(self, message: str):
        self._git("add", "-A")
        self._git("commit", "--quiet", "-m", message)

    def tag(self, name: str):
        self._git("tag", name)

    def write_release(self, version: str, *, extra: str = "", deps_extra: str = ""):
        pyproject = PYPROJECT.format(version=version)
        if deps_extra:
            pyproject = pyproject.replace('    "numpy",', f'    "numpy",\n{deps_extra}')
        self.write("pyproject.toml", pyproject)
        self.write("ueler/__init__.py", INIT.format(version=version))
        self.write("ueler/viewer.py", VIEWER.format(extra=extra))


class RehearsalTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = _Repo(Path(self._tmp.name))

    def candidate_release(self, version="0.5.0-rc1", tag="v0.5.0-rc1", **kwargs):
        self.repo.write_release(version, **kwargs)
        self.repo.write("doc/log.md", "# Log\n\n## 0.5.0-rc1\n")
        self.repo.commit(f"release {version}")
        self.repo.tag(tag)

    def promote(self, version="0.5.0", tag="v0.5.0", **kwargs):
        self.repo.write_release(version, **kwargs)
        # Documentation is expected to change on the way to a stable release.
        self.repo.write("doc/log.md", "# Log\n\n## 0.5.0\n\nReleased.\n")
        self.repo.commit(f"release {version}")
        self.repo.tag(tag)

    def check(self, tag="v0.5.0", published=frozenset({"0.5.0rc1"})):
        return rehearsal.check_rehearsal(
            tag, self.repo.root, published=set(published)
        )


class PromotionAcceptedTests(RehearsalTestCase):
    def test_version_only_bump_is_accepted(self):
        self.candidate_release()
        self.promote()
        rc_tag, rc_version = self.check()
        self.assertEqual(rc_tag, "v0.5.0-rc1")
        self.assertEqual(str(rc_version), "0.5.0rc1")

    def test_documentation_and_tests_may_change_freely(self):
        self.candidate_release()
        self.repo.write_release("0.5.0")
        self.repo.write("doc/log.md", "# Log\n\nrewritten\n")
        self.repo.write("README.md", "# UELer\n")
        self.repo.write("tests/test_new.py", "def test_x():\n    pass\n")
        self.repo.write("dev_note/notes.md", "notes\n")
        self.repo.commit("release 0.5.0")
        self.repo.tag("v0.5.0")
        self.assertEqual(self.check()[0], "v0.5.0-rc1")

    def test_highest_candidate_is_the_one_that_vouches(self):
        self.candidate_release()
        # rc2 changes the code, then the stable tag promotes rc2 unchanged.
        self.repo.write_release("0.5.0-rc2", extra=" Fixed.")
        self.repo.commit("release 0.5.0-rc2")
        self.repo.tag("v0.5.0-rc2")
        self.promote(extra=" Fixed.")
        rc_tag, _ = self.check(published={"0.5.0rc1", "0.5.0rc2"})
        self.assertEqual(rc_tag, "v0.5.0-rc2")

    def test_dotted_candidate_spelling_is_accepted(self):
        self.candidate_release(version="0.5.0.rc1", tag="v0.5.0.rc1")
        self.promote()
        self.assertEqual(self.check()[0], "v0.5.0.rc1")


class PromotionRejectedTests(RehearsalTestCase):
    def test_no_candidate_at_all(self):
        self.repo.write_release("0.5.0")
        self.repo.commit("release 0.5.0")
        self.repo.tag("v0.5.0")
        with self.assertRaisesRegex(RehearsalError, "no release candidate"):
            self.check()

    def test_an_alpha_is_not_a_rehearsal(self):
        self.candidate_release(version="0.5.0-alpha2", tag="v0.5.0-alpha2")
        self.promote()
        with self.assertRaisesRegex(RehearsalError, "no release candidate"):
            self.check(published={"0.5.0a2"})

    def test_candidate_absent_from_testpypi(self):
        self.candidate_release()
        self.promote()
        with self.assertRaisesRegex(RehearsalError, "TestPyPI does not serve"):
            self.check(published=set())

    def test_shipped_code_changed_after_the_candidate(self):
        self.candidate_release()
        self.promote(extra=" One more fix.")
        with self.assertRaisesRegex(RehearsalError, r"ueler/viewer\.py"):
            self.check()

    def test_dependency_changed_after_the_candidate(self):
        # No Python source moved, but the wheel's metadata did — the rehearsal
        # no longer describes what a resolver would install.
        self.candidate_release()
        self.promote(deps_extra='    "pandas>=2.0",')
        with self.assertRaisesRegex(RehearsalError, "beyond its `version` line"):
            self.check()

    def test_version_line_moved_to_the_wrong_version(self):
        self.candidate_release()
        self.promote(version="0.5.1", tag="v0.5.1")
        with self.assertRaisesRegex(RehearsalError, "no release candidate"):
            self.check(tag="v0.5.1")

    def test_stable_tag_required(self):
        self.candidate_release()
        with self.assertRaisesRegex(RehearsalError, "is a pre-release"):
            self.check(tag="v0.5.0-rc1")

    def test_a_higher_unpublished_candidate_blocks_the_release(self):
        # rc2 was tagged but its upload failed; promoting on rc1's evidence
        # would ship code no index ever served.
        self.candidate_release()
        self.repo.write_release("0.5.0-rc2", extra=" Fixed.")
        self.repo.commit("release 0.5.0-rc2")
        self.repo.tag("v0.5.0-rc2")
        self.promote(extra=" Fixed.")
        with self.assertRaisesRegex(RehearsalError, "TestPyPI does not serve"):
            self.check(published={"0.5.0rc1"})


class CandidateSelectionTests(unittest.TestCase):
    """find_release_candidate in isolation — ordering and spelling."""

    def setUp(self):
        self.parse = rehearsal.parse_tag
        self.find = rehearsal.find_release_candidate

    def test_numeric_not_lexical_ordering(self):
        tags = ["v0.5.0-rc1", "v0.5.0-rc2", "v0.5.0-rc10"]
        tag, version = self.find(self.parse("v0.5.0"), tags)
        self.assertEqual(tag, "v0.5.0-rc10")
        self.assertEqual(str(version), "0.5.0rc10")

    def test_other_versions_are_ignored(self):
        tags = ["v0.4.9-rc3", "v0.5.0-rc1", "v0.6.0-rc9", "not-a-tag", "v0.5.0-alpha1"]
        tag, _ = self.find(self.parse("v0.5.0"), tags)
        self.assertEqual(tag, "v0.5.0-rc1")

    def test_spellings_collapse(self):
        for spelling in ("v0.5.0-rc1", "v0.5.0.rc1", "v0.5.0rc1", "v0.5.0-c1"):
            with self.subTest(spelling=spelling):
                _, version = self.find(self.parse("v0.5.0"), [spelling])
                self.assertEqual(str(version), "0.5.0rc1")

    def test_normalise_dist_name(self):
        self.assertEqual(rehearsal.normalise_dist_name("UELer_Viewer"), "ueler-viewer")


if __name__ == "__main__":
    unittest.main()
