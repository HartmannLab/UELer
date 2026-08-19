#!/usr/bin/env python3
"""Check that a stable release is the promotion of a rehearsed release candidate.

PyPI is append-only: a version can be yanked but never replaced or reused. So a
stable upload should never be the first time the artifact meets a real index and
a real resolver. This enforces that discipline mechanically -- ``vX.Y.Z`` may
only reach PyPI when

1. a release-candidate tag exists for the same ``X.Y.Z``,
2. that candidate -- the *highest* rc, not merely some rc -- was actually
   published to TestPyPI, and
3. everything that lands in the wheel is unchanged between the two tags, apart
   from the version declarations themselves.

**On (3), and why it is not "the same commit".** The rc and the stable release
differ by construction. ``tools/check_release_tag.py`` requires the tag,
``pyproject.toml [project] version``, ``ueler.__version__`` and both artifact
filenames to describe one release: at the rc those all read ``0.5.0rc1`` and at
the stable tag they must all read ``0.5.0``. One commit cannot declare both, so
the two tags are *necessarily* different commits and the artifacts are
*necessarily* not byte-identical -- the wheel filename, ``METADATA``'s
``Version:`` and ``__version__`` all move. Comparing whole trees or whole files
would therefore fail on every correct release.

The comparison is instead scoped to what ships in the wheel, with the
version-bearing lines named explicitly:

===================  ==========================================================
``ueler/**``         unchanged, except ``ueler/__init__.py`` whose diff may
                     contain only its ``__version__`` line
``pyproject.toml``   unchanged, except its ``version`` line
===================  ==========================================================

and on each exception the removed value must normalise to the rc version and the
added value to the stable version -- otherwise "only the version line changed"
would be satisfied by a version line changed to anything at all.

Freezing ``pyproject.toml`` matters as much as freezing the code: a dependency
floor edited after the rc changes the wheel's metadata and invalidates the
rehearsal even though no Python source moved. Everything else is deliberately
free -- ``doc/log.md``, ``README.md``, ``docs/**``, ``dev_note/**`` and
``tests/**`` are all expected to change on the way to a stable release and none
of them affect what a user installs. That allowed set is exactly what the
version-bump skill touches on an rc -> stable bump.

Ancestry (``git merge-base --is-ancestor``) is reported but not enforced. The
content comparison above is strictly stronger for this purpose -- it proves the
shipped bytes match -- while ancestry can fail for reasons that say nothing about
the artifact: a rebase, a merge topology where the candidate and the release sit
on lines joined later, or a shallow fetch. As a gate it would add a failure mode
without adding evidence, so it prints as a note instead: worth seeing, not worth
blocking on.

Usage, from the repository root (needs the tags: CI checks out with
``fetch-depth: 0``)::

    python tools/check_stable_rehearsal.py v0.6.0
    python tools/check_stable_rehearsal.py v0.6.0 --skip-index-check
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

from packaging.version import InvalidVersion, Version

# Everything that lands in the wheel. Kept in step with
# [tool.setuptools.packages.find] and [tool.setuptools.package-data].
FROZEN_PATHS: tuple[str, ...] = ("ueler", "pyproject.toml")

# The files inside FROZEN_PATHS that must change, and the single key whose line
# is allowed to differ in each.
VERSION_BEARING: dict[str, str] = {
    "ueler/__init__.py": "__version__",
    "pyproject.toml": "version",
}

TESTPYPI_JSON = "https://test.pypi.org/pypi/{name}/json"

_ASSIGNMENT_RE = re.compile(r"""^\s*(?P<key>[A-Za-z_][\w-]*)\s*=\s*["'](?P<value>[^"']+)["']""")
_PYPROJECT_NAME_RE = re.compile(r"""^\s*name\s*=\s*["']([^"']+)["']""", re.MULTILINE)


class RehearsalError(Exception):
    """The stable tag is not a promotion of a published release candidate."""


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RehearsalError(
            f"git {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


def normalise_dist_name(name: str) -> str:
    """PEP 503 normalisation, for building the index URL."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_dist_name(root: Path) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    # Deliberately the first `name = "..."` line: it is the [project] one, and
    # this must keep working on Python 3.10 where tomllib does not exist.
    match = _PYPROJECT_NAME_RE.search(text)
    if not match:
        raise RehearsalError("pyproject.toml: could not find the [project] name")
    return match.group(1)


def parse_tag(tag: str) -> Version:
    raw = tag[1:] if tag.startswith("v") else tag
    try:
        return Version(raw)
    except InvalidVersion as exc:
        raise RehearsalError(f"tag {tag!r} is not a valid PEP 440 version ({exc})") from exc


def list_tags(root: Path) -> list[str]:
    return [line.strip() for line in _git(root, "tag", "--list").splitlines() if line.strip()]


def find_release_candidate(
    stable: Version, tags: Iterable[str]
) -> tuple[str, Version]:
    """Return the highest rc tag for ``stable``'s release tuple.

    An alpha or beta does not count. A preview says "look at this"; a release
    candidate says "this is what I intend to ship", and only the latter is a
    rehearsal of a specific stable release. Tag spelling is normalised rather
    than matched textually, so ``v0.5.0.rc1``, ``v0.5.0-rc1``, ``v0.5.0rc1`` and
    ``v0.5.0-c1`` are all the same candidate.
    """
    candidates: list[tuple[Version, str]] = []
    for tag in tags:
        try:
            version = parse_tag(tag)
        except RehearsalError:
            continue  # not a version tag; not our business
        if version.release != stable.release:
            continue
        if version.pre is None or version.pre[0] != "rc":
            continue
        if version.local or version.is_devrelease or version.is_postrelease:
            continue
        candidates.append((version, tag))

    if not candidates:
        raise RehearsalError(
            f"no release candidate found for {stable}. A stable release must be the "
            f"promotion of a candidate that TestPyPI has already served: tag and "
            f"publish v{stable}-rc1 first, then tag v{stable}."
        )
    version, tag = max(candidates, key=lambda item: item[0])
    return tag, version


def published_versions(dist_name: str, url_template: str = TESTPYPI_JSON) -> set[str]:
    """The versions TestPyPI actually serves for ``dist_name``.

    A tag existing is not evidence that its upload succeeded -- the publish step
    could have failed, or the tag could predate the workflow. So this asks the
    index. It fails closed: a network problem is an error, not a pass.
    """
    url = url_template.format(name=normalise_dist_name(dist_name))
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RehearsalError(
                f"{url} returned 404: TestPyPI has no project {dist_name!r} at all, "
                "so no release candidate can have been published."
            ) from exc
        raise RehearsalError(f"{url} returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RehearsalError(
            f"could not read {url} ({exc}). This check fails closed; re-run it "
            "rather than skipping it."
        ) from exc

    found: set[str] = set()
    for raw in payload.get("releases", {}):
        try:
            found.add(str(Version(raw)))
        except InvalidVersion:
            continue
    return found


def _diff_paths(root: Path, rc_tag: str, stable_tag: str) -> list[str]:
    output = _git(
        root, "diff", "--name-only", rc_tag, stable_tag, "--", *FROZEN_PATHS
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def _version_line_changes(
    root: Path,
    rc_tag: str,
    stable_tag: str,
    path: str,
    key: str,
    rc_version: Version,
    stable_version: Version,
) -> None:
    """Fail unless ``path``'s diff consists solely of its version assignment."""
    diff = _git(
        root,
        "diff",
        "--unified=0",
        "--no-color",
        rc_tag,
        stable_tag,
        "--",
        path,
    )
    removed: list[str] = []
    added: list[str] = []
    offending: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---", "@@", "diff ", "index ")):
            continue
        if not line.startswith(("+", "-")):
            continue
        body = line[1:]
        match = _ASSIGNMENT_RE.match(body)
        if not match or match.group("key") != key:
            offending.append(line)
            continue
        (added if line.startswith("+") else removed).append(match.group("value"))

    if offending:
        listing = "\n    ".join(offending)
        raise RehearsalError(
            f"{path} changed beyond its `{key}` line between {rc_tag} and "
            f"{stable_tag}. It ships in the wheel, so the rehearsal no longer "
            f"describes what would be uploaded:\n    {listing}"
        )
    if not (removed and added):
        raise RehearsalError(
            f"{path} is reported as changed but no `{key}` assignment moved; "
            "the diff could not be interpreted."
        )
    for label, values, expected in (
        ("removed", removed, rc_version),
        ("added", added, stable_version),
    ):
        for value in values:
            try:
                actual = Version(value)
            except InvalidVersion as exc:
                raise RehearsalError(
                    f"{path}: {label} `{key} = \"{value}\"` is not a valid PEP 440 "
                    f"version ({exc})"
                ) from exc
            if actual != expected:
                raise RehearsalError(
                    f"{path}: the {label} `{key}` is {actual}, expected {expected}. "
                    "The only difference between a candidate and its promotion may "
                    "be the rc version giving way to the stable one."
                )


def check_rehearsal(
    stable_tag: str,
    root: Path,
    dist_name: str | None = None,
    tags: Sequence[str] | None = None,
    published: set[str] | None = None,
) -> tuple[str, Version]:
    """Run every rehearsal check. Returns the candidate that vouches for the tag.

    ``tags`` and ``published`` exist to be injected by the tests; in CI they are
    read from git and from TestPyPI respectively.
    """
    stable = parse_tag(stable_tag)
    if stable.is_prerelease:
        raise RehearsalError(
            f"{stable_tag} is a pre-release ({stable}); this check only applies to "
            "a stable release heading for PyPI."
        )

    tag_list = list(tags) if tags is not None else list_tags(root)
    rc_tag, rc_version = find_release_candidate(stable, tag_list)
    print(f"candidate: {rc_tag} -> {rc_version}")

    if published is None:
        published = published_versions(dist_name or read_dist_name(root))
    if str(rc_version) not in published:
        raise RehearsalError(
            f"{rc_version} is tagged as {rc_tag} but TestPyPI does not serve it. "
            "The candidate's release run must have failed before the upload; fix "
            "that and let the rc publish before promoting it."
        )
    print(f"TestPyPI serves {rc_version}")

    changed = _diff_paths(root, rc_tag, stable_tag)
    unexpected = [path for path in changed if path not in VERSION_BEARING]
    if unexpected:
        listing = "\n    ".join(unexpected)
        raise RehearsalError(
            f"these files ship in the wheel and changed between {rc_tag} and "
            f"{stable_tag}:\n    {listing}\n"
            f"The rehearsal no longer covers what would be uploaded. Tag the next "
            f"candidate (v{stable}-rc<N+1>), let it publish, then promote that."
        )
    for path in changed:
        _version_line_changes(
            root, rc_tag, stable_tag, path, VERSION_BEARING[path],
            rc_version, stable,
        )
    print(f"wheel contents identical to {rc_tag} apart from the version lines")

    # A note, not a gate -- see the module docstring.
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", rc_tag, stable_tag),
        cwd=root,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        print(
            f"note: {rc_tag} is not an ancestor of {stable_tag}. Not a failure — "
            "the content check above is what proves the promotion."
        )
    return rc_tag, rc_version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", help="the stable release tag being published (e.g. v0.6.0)")
    parser.add_argument(
        "--root", default=".", help="repository root (default: the current directory)"
    )
    parser.add_argument(
        "--dist-name",
        default=None,
        help="distribution name to look up (default: read [project] name)",
    )
    parser.add_argument(
        "--skip-index-check",
        action="store_true",
        help=(
            "do not ask TestPyPI whether the candidate was published. For local "
            "dry runs only -- CI must not pass this."
        ),
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    published: set[str] | None = None
    if args.skip_index_check:
        print(
            "WARNING: skipping the TestPyPI check; publication is NOT verified",
            file=sys.stderr,
        )
        stable = parse_tag(args.tag)
        try:
            _, rc_version = find_release_candidate(stable, list_tags(root))
        except RehearsalError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        published = {str(rc_version)}

    try:
        rc_tag, rc_version = check_rehearsal(
            args.tag, root, dist_name=args.dist_name, published=published
        )
    except RehearsalError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 1

    print(f"\nOK: {args.tag} is the promotion of {rc_tag} ({rc_version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
