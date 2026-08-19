"""Tests for tools/release_channel.py — which index a release run uploads to.

PyPI is append-only, so a wrong answer from this predicate burns a version
number permanently. The corpus below is the whole contract: the tag spellings
this project actually writes, plus the ones it has not written yet.
"""

import importlib.util
import unittest
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_channel = _load("release_channel")
ChannelError = release_channel.ChannelError
decide = release_channel.decide


class TagCorpusRoutingTests(unittest.TestCase):
    """The existing tags as a corpus of real spellings, not as releases to redo.

    They cost nothing to check and they are the spellings this project actually
    writes, so they are the best available evidence that the predicate reads a
    tag the way a human would.
    """

    CORPUS = [
        ("v0.1.10", "pypi"),
        ("v0.1.10-rc2", "testpypi"),
        ("v0.2.0", "pypi"),
        ("v0.2.0-alpha", "testpypi"),
        ("v0.2.0-rc1", "testpypi"),
        ("v0.4.0", "pypi"),
        ("v0.4.1", "pypi"),
        ("v0.5.0-alpha1", "testpypi"),
        ("v0.5.0-alpha2", "testpypi"),
    ]

    def test_every_existing_tag_spelling(self):
        for tag, expected in self.CORPUS:
            with self.subTest(tag=tag):
                outputs = decide("push", "tag", tag, "")
                self.assertEqual(outputs["channel"], expected)


class SpellingTests(unittest.TestCase):
    """Routing is on the normalised PEP 440 version, not on the tag text."""

    def test_prerelease_spellings_go_to_testpypi(self):
        for tag in ("v0.5.0-rc1", "v0.5.0.rc1", "v0.5.0rc1", "v0.5.0-c1",
                    "v0.5.0-beta1", "v0.5.0b1", "v0.5.0-dev1", "v0.5.0.dev1"):
            with self.subTest(tag=tag):
                self.assertEqual(decide("push", "tag", tag, "")["channel"], "testpypi")

    def test_stable_and_post_releases_go_to_pypi(self):
        # A post-release of a stable version is not a pre-release, so it must
        # reach the real index rather than being quietly diverted.
        for tag in ("v1.0.0", "v1.0.0.post1", "1.0.0", "v0.6.0.1"):
            with self.subTest(tag=tag):
                self.assertEqual(decide("push", "tag", tag, "")["channel"], "pypi")

    def test_version_is_reported_normalised(self):
        outputs = decide("push", "tag", "v0.5.0-alpha2", "")
        self.assertEqual(outputs["version"], "0.5.0a2")
        self.assertEqual(outputs["prerelease"], "true")


class RejectionTests(unittest.TestCase):
    def test_unparseable_tag(self):
        with self.assertRaises(ChannelError):
            decide("push", "tag", "vnonsense", "")

    def test_local_version_segment_is_rejected(self):
        # PyPI refuses local versions outright; catching it here fails the run
        # in seconds instead of at the upload step.
        with self.assertRaises(ChannelError):
            decide("push", "tag", "v0.5.0+local", "")

    def test_push_to_a_branch_is_rejected(self):
        with self.assertRaises(ChannelError):
            decide("push", "branch", "main", "")

    def test_unknown_event(self):
        with self.assertRaises(ChannelError):
            decide("schedule", "tag", "v0.5.0", "")


class DispatchTests(unittest.TestCase):
    """The manual path is kept as the way to re-drive an upload."""

    def test_dispatch_honours_the_choice(self):
        for choice in ("none", "testpypi", "pypi"):
            with self.subTest(choice=choice):
                outputs = decide("workflow_dispatch", "tag", "v0.6.0", choice)
                self.assertEqual(outputs["channel"], choice)

    def test_dispatch_can_send_a_prerelease_to_pypi_only_deliberately(self):
        # Routing does not override an explicit choice: the operator asked for
        # PyPI from a tag, and `verify` still has to pass on that tag.
        outputs = decide("workflow_dispatch", "tag", "v0.6.0-rc1", "pypi")
        self.assertEqual(outputs["channel"], "pypi")

    def test_pypi_requires_a_tag(self):
        with self.assertRaises(ChannelError):
            decide("workflow_dispatch", "branch", "main", "pypi")

    def test_branch_dispatch_without_upload_is_allowed(self):
        outputs = decide("workflow_dispatch", "branch", "nightly", "none")
        self.assertEqual(outputs["channel"], "none")
        self.assertEqual(outputs["version"], "")

    def test_unknown_choice(self):
        with self.assertRaises(ChannelError):
            decide("workflow_dispatch", "tag", "v0.6.0", "somewhere-else")


if __name__ == "__main__":
    unittest.main()
