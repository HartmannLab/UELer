#!/usr/bin/env python3
"""Check that a release tag agrees with every version UELer declares and built.

PyPI is append-only: a version can be yanked but never replaced or reused, so a
tag that does not match the artifact is a mistake that outlives the mistake.
This compares four places at once —

* the git tag being released (``v0.5.0-alpha``),
* ``pyproject.toml`` ``[project] version`` (the canonical string),
* ``ueler.__version__`` (what a user sees at runtime),
* the built ``dist/`` filenames (what would actually be uploaded),

— and fails on the first disagreement.

Comparison is on **PEP 440-normalised** versions, so the repository's SemVer tag
spelling and the packaging spelling compare equal: ``v0.5.0-alpha``,
``v0.5.0-a0`` and ``0.5.0a0`` are all the same release. That is deliberate — the
repo tags in SemVer style (`v0.2.0-alpha` is an existing tag) while setuptools
normalises to `0.5.0a0`, and a plain string comparison would reject a correct
tag.

Usage, from the repository root::

    python tools/check_release_tag.py v0.5.0-alpha
    python tools/check_release_tag.py v0.5.0-alpha --dist dist
    python tools/check_release_tag.py --no-tag        # sources only, no tag yet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 path
    tomllib = None  # type: ignore[assignment]

from packaging.version import InvalidVersion, Version

_PYPROJECT_VERSION_RE = re.compile(
    r"^\s*version\s*=\s*[\"']([^\"']+)[\"']", re.MULTILINE
)
_DUNDER_VERSION_RE = re.compile(
    r"^\s*__version__\s*=\s*[\"']([^\"']+)[\"']", re.MULTILINE
)


def _read_pyproject_version(root: Path) -> str:
    path = root / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    if tomllib is not None:
        data = tomllib.loads(text)
        version = data.get("project", {}).get("version")
        if version:
            return str(version)
        raise SystemExit(f"{path}: [project] has no `version` key")
    # 3.10 has no tomllib and this script must not need a dependency to run
    # locally before tagging. The version line is a single quoted literal.
    match = _PYPROJECT_VERSION_RE.search(text)
    if not match:
        raise SystemExit(f"{path}: could not find a `version = \"...\"` line")
    return match.group(1)


def _read_package_version(root: Path) -> str:
    path = root / "ueler" / "__init__.py"
    match = _DUNDER_VERSION_RE.search(path.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"{path}: could not find a `__version__ = \"...\"` line")
    return match.group(1)


def _dist_versions(dist_dir: Path) -> dict[str, str]:
    """Map each artifact filename to the version encoded in it."""
    found: dict[str, str] = {}
    for wheel in sorted(dist_dir.glob("*.whl")):
        # name-version-python-abi-platform.whl
        parts = wheel.name.split("-")
        if len(parts) < 3:
            raise SystemExit(f"{wheel.name}: not a parseable wheel filename")
        found[wheel.name] = parts[1]
    for sdist in sorted(dist_dir.glob("*.tar.gz")):
        stem = sdist.name[: -len(".tar.gz")]
        if "-" not in stem:
            raise SystemExit(f"{sdist.name}: not a parseable sdist filename")
        found[sdist.name] = stem.rsplit("-", 1)[1]
    return found


def _normalise(label: str, raw: str) -> Version:
    try:
        return Version(raw)
    except InvalidVersion as exc:  # pragma: no cover - defensive
        raise SystemExit(f"{label}: {raw!r} is not a valid PEP 440 version ({exc})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "tag",
        nargs="?",
        help="the release tag, with or without the leading 'v' (e.g. v0.5.0-alpha)",
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="only cross-check the sources and dist/ (use before a tag exists)",
    )
    parser.add_argument(
        "--dist",
        default="dist",
        help="directory holding the built artifacts (default: dist)",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="repository root (default: the current directory)",
    )
    parser.add_argument(
        "--require-dist",
        action="store_true",
        help="fail if dist/ is missing or empty instead of skipping that check",
    )
    args = parser.parse_args(argv)

    if not args.tag and not args.no_tag:
        parser.error("give a tag, or pass --no-tag to check the sources only")

    root = Path(args.root).resolve()
    checked: list[tuple[str, str, Version]] = []

    pyproject_raw = _read_pyproject_version(root)
    checked.append(("pyproject.toml [project] version", pyproject_raw,
                    _normalise("pyproject.toml", pyproject_raw)))

    package_raw = _read_package_version(root)
    checked.append(("ueler.__version__", package_raw,
                    _normalise("ueler/__init__.py", package_raw)))

    if args.tag:
        tag_raw = args.tag[1:] if args.tag.startswith("v") else args.tag
        checked.append((f"git tag {args.tag}", tag_raw,
                        _normalise("git tag", tag_raw)))

    dist_dir = root / args.dist
    dist_versions = _dist_versions(dist_dir) if dist_dir.is_dir() else {}
    if dist_versions:
        for filename, raw in dist_versions.items():
            checked.append((f"dist/{filename}", raw, _normalise(filename, raw)))
    elif args.require_dist:
        raise SystemExit(f"{dist_dir}: no wheel or sdist found (build first)")
    else:
        print(f"note: no artifacts in {dist_dir}, skipping the dist check")

    width = max(len(label) for label, _, _ in checked)
    for label, raw, version in checked:
        print(f"  {label:<{width}}  {raw:<14} -> {version}")

    distinct = {version for _, _, version in checked}
    if len(distinct) > 1:
        print(
            "\nFAIL: these do not describe the same release. PyPI cannot reuse a "
            "version number, so fix the mismatch before uploading — the "
            "version-bump skill's `check` subcommand syncs the source locations, "
            "and `make build` refreshes dist/.",
            file=sys.stderr,
        )
        return 1

    print(f"\nOK: everything describes {distinct.pop()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
