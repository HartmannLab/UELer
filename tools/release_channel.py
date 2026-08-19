#!/usr/bin/env python3
"""Decide which package index a release run should upload to.

The tag already says everything needed, so the routing rule is one predicate::

    packaging.version.Version(tag).is_prerelease   ->  TestPyPI
    otherwise                                      ->  PyPI

That agrees with every tag this repository has ever pushed, without
special-casing: the SemVer spellings the repo uses (``-alpha``, ``-alphaN``,
``-rcN``) normalise to PEP 440 pre-release segments (``a0``, ``a1``, ``rc1``)
while a plain ``vX.Y.Z`` does not. It also behaves for spellings not yet used --
``-beta1`` and ``-dev1`` are pre-releases, and ``v1.0.0.post1`` is *not*, so a
post-release of a stable version correctly routes to PyPI.

Why a script instead of an expression inside the workflow YAML: PyPI is
append-only, so a wrong answer here burns a version number permanently. The
predicate is worth having somewhere it can be unit-tested, which is
``tests/test_release_channel.py``.

The channel names the *final* index for the run. Every publishing run passes
through TestPyPI first regardless (see ``.github/workflows/release.yml``), so
``channel=pypi`` means "TestPyPI, then PyPI", not "PyPI instead".

Reads the GitHub context from the environment and writes ``key=value`` lines for
``$GITHUB_OUTPUT``::

    EVENT=push REF_TYPE=tag TAG=v0.6.0 python tools/release_channel.py
    channel=pypi
    version=0.6.0
    prerelease=false
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

CHANNELS = ("none", "testpypi", "pypi")


class ChannelError(ValueError):
    """The context does not describe a releasable ref."""


def parse_tag(tag: str) -> Version:
    """Normalise a release tag to a PEP 440 version.

    Accepts the repository's ``v``-prefixed SemVer spelling. A local version
    segment is rejected here rather than at the upload: PyPI refuses local
    versions outright, and a tag carrying one is always a mistake.
    """
    if not tag:
        raise ChannelError("no tag given, but this ref is a tag")
    raw = tag[1:] if tag.startswith("v") else tag
    try:
        version = Version(raw)
    except InvalidVersion as exc:
        raise ChannelError(
            f"tag {tag!r} is not a valid PEP 440 version ({exc}). "
            "Release tags look like v0.5.0, v0.5.0-rc1 or v0.5.0-alpha2."
        ) from exc
    if version.local:
        raise ChannelError(
            f"tag {tag!r} carries the local version segment "
            f"'+{version.local}'. PyPI rejects local versions; retag without it."
        )
    return version


def decide(event: str, ref_type: str, tag: str, choice: str) -> dict[str, str]:
    """Return the ``$GITHUB_OUTPUT`` mapping for one release run."""
    if event == "push":
        if ref_type != "tag":
            raise ChannelError(
                f"a push to a {ref_type or 'ref'} reached the release workflow; "
                "it is only meant to trigger on `v*` tags"
            )
        version = parse_tag(tag)
        channel = "testpypi" if version.is_prerelease else "pypi"
        return {
            "channel": channel,
            "version": str(version),
            "prerelease": str(version.is_prerelease).lower(),
        }

    if event == "workflow_dispatch":
        if choice not in CHANNELS:
            raise ChannelError(
                f"publish_to={choice!r} is not one of {', '.join(CHANNELS)}"
            )
        # The pre-existing rule, kept: a manual upload to PyPI is only ever
        # allowed from a tag, so `verify` has matched the artifacts against one.
        if choice == "pypi" and ref_type != "tag":
            raise ChannelError(
                "publish_to=pypi requires the run to be started from a tag, so "
                f"the upload is tied to a verified release (this ref is a {ref_type})"
            )
        if ref_type == "tag":
            version = parse_tag(tag)
            return {
                "channel": choice,
                "version": str(version),
                "prerelease": str(version.is_prerelease).lower(),
            }
        return {"channel": choice, "version": "", "prerelease": ""}

    raise ChannelError(f"unsupported event {event!r}")


def _describe(outputs: dict[str, str]) -> str:
    channel = outputs["channel"]
    version = outputs["version"] or "(no tag on this ref)"
    if channel == "none":
        return f"{version}: building and verifying only, no upload"
    if channel == "testpypi":
        return f"{version} is a pre-release -> TestPyPI"
    return f"{version} is a stable release -> TestPyPI, then PyPI"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event", default=os.environ.get("EVENT", ""))
    parser.add_argument("--ref-type", default=os.environ.get("REF_TYPE", ""))
    parser.add_argument("--tag", default=os.environ.get("TAG", ""))
    parser.add_argument("--choice", default=os.environ.get("CHOICE", ""))
    args = parser.parse_args(argv)

    try:
        outputs = decide(args.event, args.ref_type, args.tag, args.choice)
    except ChannelError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    verdict = _describe(outputs)
    lines = [f"{key}={value}" for key, value in outputs.items()]

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    # The run page should state where this is heading before anything uploads.
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(f"**Release channel:** {verdict}\n")
    print(f"\n{verdict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
