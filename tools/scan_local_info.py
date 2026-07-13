#!/usr/bin/env python3
"""Scan the codebase for local / machine-specific information that should not ship.

Two scopes are supported:

* ``package``  -- the importable ``ueler`` package and its subpackages only
                  (this is what actually gets built into the wheel/sdist).
* ``project``  -- the entire repository working tree (source, docs, notebooks,
                  planning notes, configs, ...).

Findings are grouped by category and severity so the output is actionable:

* HIGH   -- almost certainly a leak: hardcoded secrets, absolute machine paths.
* MEDIUM -- worth a human look: emails, personal/user identifiers, network literals.
* LOW    -- informational: home-relative (``~/``) defaults, developer notes.

The rules are tuned to avoid the common false positives (e.g. a lexer ``Token``
class or a ``cache_token`` variable is NOT a secret) by requiring assignment
context with a real string literal for secret detection.

Suppression: append ``# noqa: scan`` (or ``# noqa: scan:<category>``) to a line
to exclude it from the report.

Git awareness: by default only files that are **tracked by git** are scanned
(untracked and gitignored files can't be committed or shipped, so they'd only
add noise). Pass ``--include-untracked`` to scan every text file on disk. If the
root is not inside a git repository, the scan falls back to all files.

Exit code: 0 if no HIGH findings, 1 otherwise (so it can gate CI / releases).

Usage:
    python tools/scan_local_info.py                 # scan both scopes
    python tools/scan_local_info.py --scope package
    python tools/scan_local_info.py --scope project
    python tools/scan_local_info.py --min-severity medium
    python tools/scan_local_info.py --include-untracked
    python tools/scan_local_info.py --root /path/to/repo
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Pattern, Set

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Directories never worth scanning, regardless of scope.
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "graphify-out",  # generated RAG artifact, not shipped
}
SKIP_DIR_SUFFIXES = (".egg-info",)

# Only these extensions are treated as text and scanned. Notebooks (.ipynb) are
# JSON and handled as text too.
TEXT_EXTENSIONS = {
    ".py", ".pyi", ".ipynb", ".txt", ".md", ".rst", ".cfg", ".ini", ".toml",
    ".json", ".yaml", ".yml", ".csv", ".tsv", ".sh", ".bash", ".make", ".env",
    ".html", ".js", ".css",
}
# Extension-less files worth scanning by name.
TEXT_FILENAMES = {"Makefile", "makefile", "LICENSE", "LICENSE.txt", ".env"}

MAX_BYTES = 5_000_000  # skip files larger than ~5 MB (data blobs, not code)

SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1}


@dataclass
class Rule:
    name: str
    severity: str
    pattern: Pattern
    hint: str


def _rules() -> List[Rule]:
    r: List[Rule] = []

    # -- HIGH: hardcoded secrets. Requires assignment + a non-empty string value,
    #    so `Token(...)`, `cache_token`, `token = source.strip()` do NOT match.
    r.append(Rule(
        "secret-assignment", "high",
        re.compile(
            r"""(?ix)
            \b(pass(word|wd)?|secret|api[_-]?key|apikey|access[_-]?key
               |auth[_-]?token|access[_-]?token|client[_-]?secret
               |private[_-]?key)\b
            \s*[:=]\s*
            ['"][^'"\s]{4,}['"]
            """,
        ),
        "Possible hardcoded credential value.",
    ))
    r.append(Rule(
        "private-key-block", "high",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
        "Embedded private key.",
    ))
    r.append(Rule(
        "ssh-public-key", "high",
        re.compile(r"ssh-(?:rsa|ed25519|dss) AAAA[0-9A-Za-z+/]{20,}"),
        "Embedded SSH key.",
    ))
    r.append(Rule(
        "aws-access-key", "high",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        "AWS access key id.",
    ))

    # -- HIGH: absolute machine paths that leak the developer's filesystem layout.
    r.append(Rule(
        "machine-abs-path", "high",
        re.compile(
            r"""(?x)
            (?<![\w~.])(
                /omics/            |
                /home/[A-Za-z0-9]  |
                /Users/[A-Za-z0-9] |
                /mnt/[A-Za-z0-9]   |
                /media/[A-Za-z0-9] |
                /scratch/          |
                /data/[A-Za-z0-9]  |
                /srv/[A-Za-z0-9]
            )
            """,
        ),
        "Absolute path tied to a specific machine/user.",
    ))
    r.append(Rule(
        "windows-abs-path", "high",
        re.compile(r"\b[C-Zc-z]:\\\\?[A-Za-z0-9_]"),
        "Absolute Windows path.",
    ))

    # -- MEDIUM: personal / project-internal identifiers.
    r.append(Rule(
        "email-address", "medium",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "Email address.",
    ))
    r.append(Rule(
        "personal-identifier", "medium",
        re.compile(r"(?i)\b(ywu|yu[-_]?le|dkfz|/omics/groups)\b"),
        "Personal or institution-internal identifier.",
    ))

    # -- MEDIUM: hardcoded network endpoints.
    r.append(Rule(
        "network-literal", "medium",
        re.compile(r"(?i)\b(localhost|127\.0\.0\.1|0\.0\.0\.0"
                   r"|\b(?:10|192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))"
                   r"(?:\.\d{1,3}){2}\b)"),
        "Hardcoded network address.",
    ))

    # -- LOW: informational only.
    r.append(Rule(
        "home-relative-path", "low",
        re.compile(r"(?<![\w])~/[A-Za-z0-9._]"),
        "Home-relative path (usually fine as a portable default).",
    ))
    r.append(Rule(
        "dev-note", "low",
        re.compile(r"(?i)\b(TODO|FIXME|XXX|HACK)\b|@copilot"),
        "Developer note left in source.",
    ))
    return r


RULES = _rules()


@dataclass
class Finding:
    path: Path
    line_no: int
    rule: Rule
    text: str


@dataclass
class ScanResult:
    scope: str
    root: Path
    files_scanned: int = 0
    findings: List[Finding] = field(default_factory=list)
    git_filtered: bool = False  # True if the scan was restricted to tracked files


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #

def _is_text_file(path: Path) -> bool:
    if path.name in TEXT_FILENAMES:
        return True
    return path.suffix.lower() in TEXT_EXTENSIONS


def git_tracked_files(root: Path) -> Optional[Set[Path]]:
    """Return the set of git-tracked files under ``root`` (resolved absolute
    paths), or ``None`` if ``root`` is not inside a git working tree or git is
    unavailable."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    tracked: Set[Path] = set()
    for rel in proc.stdout.split(b"\0"):
        if not rel:
            continue
        tracked.add((root / rel.decode("utf-8", "surrogateescape")).resolve())
    return tracked


def _iter_files(root: Path, tracked: Optional[Set[Path]]) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        parts = set(path.parts)
        if parts & SKIP_DIRS:
            continue
        if any(part.endswith(SKIP_DIR_SUFFIXES) for part in path.parts):
            continue
        if not _is_text_file(path):
            continue
        if tracked is not None and path.resolve() not in tracked:
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
        except OSError:
            continue
        yield path


_SUPPRESS_RE = re.compile(r"#\s*noqa:\s*scan(?::([\w-]+))?")


def _suppressed(line: str, rule_name: str) -> bool:
    m = _SUPPRESS_RE.search(line)
    if not m:
        return False
    target = m.group(1)
    return target is None or target == rule_name


def _scan_file(path: Path, findings: List[Finding]) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    for line_no, line in enumerate(text.splitlines(), start=1):
        for rule in RULES:
            if rule.pattern.search(line):
                if _suppressed(line, rule.name):
                    continue
                findings.append(Finding(path, line_no, rule, line.strip()))


def scan(scope: str, root: Path, git_only: bool) -> ScanResult:
    tracked = git_tracked_files(root) if git_only else None
    result = ScanResult(scope=scope, root=root, git_filtered=tracked is not None)
    for path in _iter_files(root, tracked):
        result.files_scanned += 1
        _scan_file(path, result.findings)
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

_COLOR = {
    "high": "\033[91m", "medium": "\033[93m", "low": "\033[96m",
    "bold": "\033[1m", "dim": "\033[2m", "reset": "\033[0m",
}


def _c(key: str, use_color: bool) -> str:
    return _COLOR[key] if use_color else ""


def _report(result: ScanResult, repo_root: Path, min_sev: str, use_color: bool) -> int:
    threshold = SEVERITY_ORDER[min_sev]
    shown = [f for f in result.findings
             if SEVERITY_ORDER[f.rule.severity] >= threshold]
    shown.sort(key=lambda f: (-SEVERITY_ORDER[f.rule.severity], str(f.path), f.line_no))

    r = _c("reset", use_color)
    print(f"\n{_c('bold', use_color)}== scope: {result.scope} =={r}")
    print(f"{_c('dim', use_color)}root: {result.root}{r}")
    filt = "git-tracked only" if result.git_filtered else "all files on disk"
    print(f"{_c('dim', use_color)}files scanned: {result.files_scanned} "
          f"({filt}){r}")

    counts = {"high": 0, "medium": 0, "low": 0}
    for f in result.findings:
        counts[f.rule.severity] += 1

    if not shown:
        print(f"{_c('dim', use_color)}no findings at/above severity "
              f"'{min_sev}'.{r}")
    else:
        current = None
        for f in shown:
            if f.rule.severity != current:
                current = f.rule.severity
                col = _c(current, use_color)
                print(f"\n  {col}{_c('bold', use_color)}[{current.upper()}]{r}")
            rel = f.path.relative_to(repo_root)
            col = _c(f.rule.severity, use_color)
            print(f"    {col}{rel}:{f.line_no}{r}  "
                  f"{_c('dim', use_color)}({f.rule.name}){r}")
            snippet = f.text if len(f.text) <= 160 else f.text[:157] + "..."
            print(f"        {snippet}")

    print(f"\n  summary: "
          f"{_c('high', use_color)}{counts['high']} high{r}, "
          f"{_c('medium', use_color)}{counts['medium']} medium{r}, "
          f"{_c('low', use_color)}{counts['low']} low{r}")
    return counts["high"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan for local/machine-specific info that should not ship.",
    )
    parser.add_argument(
        "--scope", choices=["package", "project", "both"], default="both",
        help="'package' = ueler/ only; 'project' = whole repo; default 'both'.",
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="Repository root (default: parent of this script's tools/ dir).",
    )
    parser.add_argument(
        "--min-severity", choices=["low", "medium", "high"], default="low",
        help="Only report findings at/above this severity (default: low).",
    )
    parser.add_argument(
        "--include-untracked", action="store_true",
        help="Scan every text file on disk, not just git-tracked ones "
             "(default: only git-tracked files are scanned).",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable ANSI colors.",
    )
    args = parser.parse_args(argv)

    repo_root = (args.root or Path(__file__).resolve().parent.parent).resolve()
    if not repo_root.exists():
        parser.error(f"root does not exist: {repo_root}")

    use_color = (not args.no_color) and sys.stdout.isatty()
    git_only = not args.include_untracked

    scopes: List[tuple[str, Path]] = []
    if args.scope in ("package", "both"):
        pkg = repo_root / "ueler"
        if not pkg.is_dir():
            parser.error(f"ueler package not found under {repo_root}")
        scopes.append(("package (ueler/)", pkg))
    if args.scope in ("project", "both"):
        scopes.append(("project (whole repo)", repo_root))

    if git_only and git_tracked_files(repo_root) is None:
        print(f"{_c('medium', use_color)}note{_c('reset', use_color)}: "
              f"{repo_root} is not a git repository — scanning all files.")

    total_high = 0
    for scope_name, scope_root in scopes:
        result = scan(scope_name, scope_root, git_only)
        total_high += _report(result, repo_root, args.min_severity, use_color)

    print()
    if total_high:
        print(f"{_c('high', use_color)}FAIL{_c('reset', use_color)}: "
              f"{total_high} high-severity finding(s). Review before packaging.")
        return 1
    print(f"{_c('low', use_color)}OK{_c('reset', use_color)}: "
          f"no high-severity findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
