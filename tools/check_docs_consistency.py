#!/usr/bin/env python
"""Check that the MkDocs site still describes the software it ships with.

The docs make three kinds of factual claim about UELer, and they rot at three
different speeds:

1. **Packaging claims** — the distribution name, the supported Python range, the
   extras you can install. These live in ``pyproject.toml`` and change on a
   release.
2. **Repository claims** — "see ``tests/test_x.py``", "run ``make test-ci``",
   "computed in ``scale_bar.py``". These break silently whenever a file is
   renamed.
3. **UI claims** — the plugin names a user reads off the right-hand accordion.
   These live in each plugin's ``displayed_name`` and change when a plugin is
   renamed, added, or removed.

None of the three can be caught by ``mkdocs build --strict``, which only
validates that the *site* is well formed. This script validates that the site is
*true*. It is deliberately lenient where prose is legitimately vague and strict
only where a claim is checkable, so that a failure here always means a real
defect rather than a style disagreement.

Run it directly for a report, or via ``tests/test_docs_consistency.py`` in CI::

    python tools/check_docs_consistency.py
    python tools/check_docs_consistency.py --quiet   # exit status only
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
PACKAGE_DIR = REPO_ROOT / "ueler"

# Directories a backticked path may point into. A token that starts with one of
# these is treated as a repository-path claim and must resolve.
REPO_TOP_LEVEL = (
    "ueler/",
    "tests/",
    "tools/",
    "docs/",
    "dev_note/",
    "doc/",
    "script/",
    ".github/",
    ".binder/",
)

# Tokens that look like code or paths but legitimately are not, and so are never
# resolved against the tree. Keep this list short and justified — every entry is
# a hole in the check.
NOT_A_CLAIM = frozenset(
    {
        # Placeholders in worked examples, not real paths. `my_plugin.py` is the
        # filename the plugin-development guide tells the reader to create, so it
        # must *not* exist — the day it does, the guide has a bug.
        "base_folder/",
        "<base_folder>/exports",
        "cells.h5ad",
        "cells_annotated.h5ad",
        "my_plugin.py",
        # Interpreter and tooling names that are not UELer symbols.
        "PYTHONPATH",
        "PATH",
        "MANIFEST.in",
    }
)

# A claim about code that has been *removed* cannot be resolved against the tree
# by definition, and the developer notes deliberately narrate removals — the
# whole point of "Compatibility shims removed" is to name what is gone so nobody
# reintroduces it. Recognising the removal language keeps those paragraphs
# writable without an author-maintained allowlist that would inevitably rot.
REMOVAL_NARRATION = re.compile(
    r"\b(remov\w+|delet\w+|dropp?\w*|no longer|used to|former\w*|legacy|"
    r"replaced|renamed|obsolete|gone)\b",
    re.IGNORECASE,
)

# Environment variables the docs may mention. Each must be read somewhere in the
# shipped code or the test harness, otherwise the docs are documenting a knob
# that no longer exists.
ENV_VAR_PATTERN = re.compile(r"\b(UELER_[A-Z0-9_]+|ENABLE_MAP_MODE)\b")

# Where an environment variable may legitimately be read.
ENV_VAR_SEARCH_ROOTS = ("ueler", "tests", "tools", "sitecustomize.py", "Makefile")

# A backticked token shaped like a code identifier rather than a UI label.
CODE_TOKEN_PATTERNS = (
    re.compile(r"^_[A-Za-z][A-Za-z0-9_]*$"),          # _private_name
    re.compile(r"^[a-z_][a-z0-9_]*\(\)$"),            # snake_case()
    re.compile(r"^[A-Z][A-Za-z0-9]+\.[A-Za-z_]+$"),   # ClassName.attribute
    re.compile(r"^[A-Z][A-Z0-9_]{3,}$"),              # CONSTANT_NAME
    re.compile(r"^[a-z_][a-z0-9_]*\.py$"),            # module_file.py
)

# Code claims are only enforced on pages whose convention is to name real
# symbols. Tutorials name *widgets*, which share the snake_case shape by
# accident, so enforcing there would produce noise rather than findings.
CODE_CLAIM_PAGES = ("develop-notes/",)

BACKTICK = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"^```+\s*([A-Za-z0-9_+-]*)\s*$")


@dataclass(frozen=True)
class Finding:
    """One inconsistency between the docs and the software."""

    check: str
    page: str
    line: int
    message: str

    def __str__(self) -> str:
        location = f"{self.page}:{self.line}" if self.line else self.page
        return f"[{self.check}] {location}: {self.message}"


# --------------------------------------------------------------------------- #
# Fact collection — what the software actually is
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProjectFacts:
    """The three packaging facts the docs make claims about."""

    name: str
    requires_python: str
    extras: frozenset[str]


def load_project_facts() -> ProjectFacts:
    """Read ``pyproject.toml`` with regexes rather than a TOML parser.

    ``tomllib`` only exists on Python 3.11+, and ``requires-python`` still admits
    3.10 — the same constraint that made ``check_stable_rehearsal.py`` parse by
    hand. Three scalars and a table's key list do not justify a dependency.
    """

    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    def scalar(key: str) -> str:
        match = re.search(rf'^{key}\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
        if not match:
            raise SystemExit(f"pyproject.toml: could not find `{key}`")
        return match.group(1)

    extras: set[str] = set()
    optional = re.search(
        r"^\[project\.optional-dependencies\]\s*$(.*?)(?=^\[|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if optional:
        extras = set(re.findall(r"^([A-Za-z0-9_-]+)\s*=", optional.group(1), flags=re.MULTILINE))

    return ProjectFacts(scalar("name"), scalar("requires-python"), frozenset(extras))


def declared_python_versions(requires_python: str) -> set[str]:
    """Expand a ``requires-python`` specifier into the 3.x minors it permits.

    Only the ``>=`` / ``<`` / ``<=`` forms UELer uses are handled; anything else
    returns an empty set, which disables the check rather than guessing.
    """

    lower, upper, upper_inclusive = None, None, False
    for part in (piece.strip() for piece in requires_python.split(",")):
        if part.startswith(">="):
            lower = part[2:].strip()
        elif part.startswith("<="):
            upper, upper_inclusive = part[2:].strip(), True
        elif part.startswith("<"):
            upper = part[1:].strip()
    if not lower or not upper:
        return set()

    def minor(spec: str) -> int:
        return int(spec.split(".")[1])

    last = minor(upper) if upper_inclusive else minor(upper) - 1
    return {f"3.{index}" for index in range(minor(lower), last + 1)}


def stated_python_versions(line: str) -> set[str]:
    """Expand the Python minors a prose line claims support for.

    Two spellings are in use and both have to expand to the same set, or a range
    written as "3.10–3.12" would silently pass while omitting 3.11:

    * a list — "Python 3.10, 3.11, or 3.12"
    * a range — "Supported Python: 3.10–3.11" (en dash, em dash or hyphen)
    """

    versions: set[str] = set()
    for low, high in re.findall(r"3\.(\d+)\s*[–—-]\s*3\.(\d+)", line):
        versions.update(f"3.{index}" for index in range(int(low), int(high) + 1))
    if versions:
        return versions
    return set(re.findall(r"3\.\d+", line))


def plugin_display_names() -> dict[str, str]:
    """Map each plugin module to the label the viewer shows for it.

    The viewer discovers plugins by scanning ``ueler/viewer/plugin`` for
    ``PluginBase`` subclasses (``ImageMaskViewer.dynamically_load_plugins``), and
    each one sets ``self.displayed_name`` in its constructor. That string is what
    titles the accordion section, and what the docs must therefore call it.

    Parsed with :mod:`ast` rather than imported: importing a plugin drags in the
    whole widget stack, which is far too heavy for a docs check.
    """

    names: dict[str, str] = {}
    for path in sorted((PACKAGE_DIR / "viewer" / "plugin").glob("*.py")):
        if path.name.startswith("_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "displayed_name"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                    and node.value.value
                ):
                    names[path.stem] = node.value.value
    return names


def source_identifiers() -> set[str]:
    """Every identifier-shaped token appearing anywhere in the project's Python.

    A flat token set, not a resolved symbol table: the question a docs check has
    to answer is "does this name still exist at all", and a name that has been
    renamed away vanishes from the token set entirely. Resolving scopes properly
    would reject legitimate references to attributes set on other objects.

    ``tests/`` and ``tools/`` count as project code, because the developer notes
    legitimately name the test harness and the release scripts —
    ``_ensure_matplotlib_stub()`` is a real symbol that simply does not live in
    the shipped package.
    """

    tokens: set[str] = set()
    word = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    for root in (PACKAGE_DIR, REPO_ROOT / "tests", REPO_ROOT / "tools"):
        for path in root.rglob("*.py"):
            tokens.update(word.findall(path.read_text(encoding="utf-8")))
    return tokens


def source_file_basenames() -> set[str]:
    return {path.name for path in REPO_ROOT.rglob("*.py") if ".git" not in path.parts}


def make_targets() -> set[str]:
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Za-z][A-Za-z0-9_-]*):", text, flags=re.MULTILINE))


def env_var_mentions_in_code() -> set[str]:
    found: set[str] = set()
    for root in ENV_VAR_SEARCH_ROOTS:
        target = REPO_ROOT / root
        if target.is_file():
            found.update(ENV_VAR_PATTERN.findall(target.read_text(encoding="utf-8")))
        elif target.is_dir():
            for path in target.rglob("*"):
                if path.suffix in {".py", ".cfg", ".toml"} and path.is_file():
                    found.update(ENV_VAR_PATTERN.findall(path.read_text(encoding="utf-8")))
    return found


def public_api_names() -> set[str]:
    """Names importable from ``ueler`` per its ``__init__``, read without importing."""

    tree = ast.parse((PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        names.update(
                            element.value
                            for element in node.value.elts
                            if isinstance(element, ast.Constant)
                            and isinstance(element.value, str)
                        )
                elif isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


# --------------------------------------------------------------------------- #
# Doc parsing helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Page:
    path: Path
    rel: str
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def is_code_claim_page(self) -> bool:
        return any(marker in self.rel for marker in CODE_CLAIM_PAGES)


def load_pages() -> list[Page]:
    pages = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        rel = path.relative_to(DOCS_DIR).as_posix()
        pages.append(Page(path, rel, path.read_text(encoding="utf-8").splitlines()))
    return pages


def prose_lines(page: Page):
    """Yield ``(line_number, text)`` for lines outside fenced code blocks.

    Claims inside a shell or Python fence are checked by the fence-specific
    checks; treating them as prose too would double-report every install command.
    """

    in_fence = False
    for number, line in enumerate(page.lines, start=1):
        if FENCE.match(line.strip()):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield number, line


def fenced_blocks(page: Page):
    """Yield ``(language, start_line, body)`` for each fenced code block."""

    language, start, buffer = None, 0, []
    for number, line in enumerate(page.lines, start=1):
        match = FENCE.match(line.strip())
        if match and language is None:
            language, start, buffer = match.group(1) or "", number, []
        elif match:
            yield language, start, "\n".join(buffer)
            language = None
        elif language is not None:
            buffer.append(line)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_distribution_name(pages: list[Page], project: ProjectFacts) -> list[Finding]:
    """Install commands must name the distribution, not the import package."""

    dist = project.name
    findings = []
    # `pip install ueler` is correct in exactly one place: the note telling users
    # of the old name to remove it first.
    uninstall_context = re.compile(r"pip uninstall")
    bad = re.compile(r"pip install\s+(?:--?\S+\s+)*(?P<name>ueler)\b(?!-)")
    for page in pages:
        for number, line in prose_lines(page):
            if uninstall_context.search(line):
                continue
            if bad.search(line):
                findings.append(
                    Finding(
                        "dist-name",
                        page.rel,
                        number,
                        f"`pip install ueler` names the import package; the distribution is `{dist}`",
                    )
                )
    return findings


def check_python_versions(pages: list[Page], project: ProjectFacts) -> list[Finding]:
    """A stated list of supported Python minors must match ``requires-python``."""

    requires = project.requires_python
    permitted = declared_python_versions(requires)
    if not permitted:
        return []

    findings = []
    # Only lines that read as a support statement are checked; a passing mention
    # of a version ("the 3.12 leg", "pandas 3 needs 3.11") is not a claim.
    claim = re.compile(
        r"\*\*Python\*\*|\bPython\s+3\.\d+\s*,|\bSupported Python\b", re.IGNORECASE
    )
    # A line may quote the specifier it is describing. Those digits are the
    # authority, not a claim about it, so they must not be read as one.
    quoted_specifier = re.compile(r'requires-python\s*=\s*"[^"]*"')
    for page in pages:
        for number, line in prose_lines(page):
            if not claim.search(line):
                continue
            stated = stated_python_versions(quoted_specifier.sub("", line))
            if not stated:
                continue
            if stated != permitted:
                findings.append(
                    Finding(
                        "python-range",
                        page.rel,
                        number,
                        "states Python "
                        + ", ".join(sorted(stated))
                        + f" but `requires-python = \"{requires}\"` permits "
                        + ", ".join(sorted(permitted)),
                    )
                )
    return findings


def check_extras(pages: list[Page], project: ProjectFacts) -> list[Finding]:
    """Every documented extra must exist in ``[project.optional-dependencies]``."""

    declared = set(project.extras)
    pattern = re.compile(r"(?:ueler-viewer|\.)\[([A-Za-z0-9,_-]+)\]")
    findings = []
    for page in pages:
        for number, line in enumerate(page.lines, start=1):
            for group in pattern.findall(line):
                for extra in (piece.strip() for piece in group.split(",")):
                    if extra and extra not in declared:
                        findings.append(
                            Finding(
                                "extras",
                                page.rel,
                                number,
                                f"documents extra `[{extra}]`, which is not declared in pyproject.toml "
                                f"(declared: {', '.join(sorted(declared)) or 'none'})",
                            )
                        )
    return findings


def check_repo_paths(pages: list[Page]) -> list[Finding]:
    """Backticked and linked repository paths must resolve."""

    findings = []
    blob_link = re.compile(r"blob/(?:main|develop)/([^)\s#]+)")
    for page in pages:
        for number, line in enumerate(page.lines, start=1):
            if REMOVAL_NARRATION.search(line):
                continue
            candidates = set()
            for token in BACKTICK.findall(line):
                # A backticked command carries its arguments; only the program
                # path is a claim about the tree ("tools/x.py --max-skips 0").
                token = token.strip().split()[0] if token.strip() else ""
                if not token or token in NOT_A_CLAIM or "<" in token or "*" in token:
                    continue
                if token.startswith(REPO_TOP_LEVEL):
                    candidates.add(token)
            candidates.update(blob_link.findall(line))
            for candidate in candidates:
                if candidate in NOT_A_CLAIM:
                    continue
                target = REPO_ROOT / candidate.rstrip("/")
                if not target.exists():
                    findings.append(
                        Finding("repo-path", page.rel, number, f"`{candidate}` does not exist")
                    )
    return findings


def check_make_targets(pages: list[Page], targets: set[str]) -> list[Finding]:
    """Documented ``make`` invocations must name a real target.

    Only backticked and fenced occurrences count: "the stubs make an
    already-complete environment fast" is English, not a command.
    """

    findings = []
    pattern = re.compile(r"\bmake\s+([a-z][a-z0-9_-]*)\b")
    for page in pages:
        commands: list[tuple[int, str]] = []
        for number, line in prose_lines(page):
            commands.extend((number, token) for token in BACKTICK.findall(line))
        for language, start, body in fenced_blocks(page):
            if language in {"shell", "sh", "bash", "console", ""}:
                commands.extend(
                    (start + offset + 1, text)
                    for offset, text in enumerate(body.splitlines())
                )
        for number, command in commands:
            for target in pattern.findall(command):
                if target not in targets:
                    findings.append(
                        Finding(
                            "make-target",
                            page.rel,
                            number,
                            f"`make {target}` is not a target in the Makefile",
                        )
                    )
    return findings


def check_env_vars(pages: list[Page], in_code: set[str]) -> list[Finding]:
    findings = []
    seen: set[tuple[str, str]] = set()
    for page in pages:
        for number, line in enumerate(page.lines, start=1):
            for name in ENV_VAR_PATTERN.findall(line):
                if name in in_code or (page.rel, name) in seen:
                    continue
                seen.add((page.rel, name))
                findings.append(
                    Finding(
                        "env-var",
                        page.rel,
                        number,
                        f"`{name}` is documented but is not read anywhere in the code",
                    )
                )
    return findings


def check_code_identifiers(
    pages: list[Page], identifiers: set[str], basenames: set[str]
) -> list[Finding]:
    """Symbols named in the developer notes must still exist in the package."""

    findings = []
    for page in pages:
        if not page.is_code_claim_page():
            continue
        for number, line in prose_lines(page):
            if REMOVAL_NARRATION.search(line):
                continue
            for token in BACKTICK.findall(line):
                token = token.strip()
                if token in NOT_A_CLAIM or not any(
                    pattern.match(token) for pattern in CODE_TOKEN_PATTERNS
                ):
                    continue
                if token.endswith(".py"):
                    if token not in basenames:
                        findings.append(
                            Finding(
                                "code-symbol",
                                page.rel,
                                number,
                                f"`{token}` is not a module in this repository",
                            )
                        )
                    continue
                stem = token.removesuffix("()").split(".")[-1]
                if stem not in identifiers:
                    findings.append(
                        Finding(
                            "code-symbol",
                            page.rel,
                            number,
                            f"`{token}` does not appear anywhere in `ueler/`",
                        )
                    )
    return findings


def ui_label_vocabulary() -> set[str]:
    """Every human-visible string the viewer puts on screen.

    Widget labels (``description=``), tab and accordion titles
    (``set_title(index, ...)``) and plugin ``displayed_name`` values together are
    the vocabulary a tutorial is allowed to bold. Collected by AST so that the
    heavy widget stack never has to be imported.
    """

    labels: set[str] = set()
    for path in PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if (
                        keyword.arg in {"description", "placeholder", "title", "tooltip"}
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        labels.add(keyword.value.value)
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "set_title"
                ):
                    labels.update(
                        argument.value
                        for argument in node.args
                        if isinstance(argument, ast.Constant)
                        and isinstance(argument.value, str)
                    )
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str) and any(
                    isinstance(target, ast.Attribute)
                    and target.attr in {"displayed_name", "description"}
                    for target in node.targets
                ):
                    labels.add(node.value.value)
    return {label.strip() for label in labels if label.strip()}


def check_ui_labels(
    pages: list[Page], display_names: dict[str, str], vocabulary: set[str]
) -> list[Finding]:
    """Docs must name UI elements exactly as the UI names them.

    Two failure modes, both of which have happened here:

    1. A plugin ships and no page mentions it at all.
    2. A page bolds a label that differs from the real one only in case —
       "**ROI Manager**" for a section the viewer titles "ROI manager". Only case
       variants are flagged, and only when the string is otherwise a real UI
       label: that is precise enough to never fire on ordinary bolded prose,
       which is why "**Cell gallery**" (a genuine checkbox) stays silent.
    """

    findings = []
    corpus = [page.text for page in pages]

    for module, label in sorted(display_names.items()):
        if not any(label in text for text in corpus):
            findings.append(
                Finding(
                    "ui-label",
                    "docs/",
                    0,
                    f'plugin `{module}` shows as "{label}" but no page mentions that label',
                )
            )

    by_case = {label.casefold(): label for label in vocabulary}
    for page in pages:
        for number, line in prose_lines(page):
            for bold in re.findall(r"\*\*([^*\n]+)\*\*", line):
                candidate = bold.strip().rstrip(":")
                if candidate in vocabulary:
                    continue
                real = by_case.get(candidate.casefold())
                if real and real != candidate:
                    findings.append(
                        Finding(
                            "ui-label",
                            page.rel,
                            number,
                            f'writes "{candidate}" but the UI shows "{real}"',
                        )
                    )
    return findings


def check_python_fences(pages: list[Page], api: set[str]) -> list[Finding]:
    """Python examples must parse, and must import names ``ueler`` really exports."""

    findings = []
    for page in pages:
        for language, start, body in fenced_blocks(page):
            if language not in {"python", "py"}:
                continue
            # Notebook magics are valid in the target environment but not in the
            # grammar, so strip them before parsing. Dedent too: a fence nested
            # in a list item is indented by the list, not by Python.
            source = textwrap.dedent(
                "\n".join(
                    line for line in body.splitlines() if not line.lstrip().startswith("%")
                )
            )
            try:
                tree = ast.parse(source)
            except SyntaxError as error:
                findings.append(
                    Finding(
                        "python-example",
                        page.rel,
                        start + (error.lineno or 0),
                        f"example does not parse: {error.msg}",
                    )
                )
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "ueler":
                    for alias in node.names:
                        if alias.name not in api:
                            findings.append(
                                Finding(
                                    "python-example",
                                    page.rel,
                                    start + node.lineno,
                                    f"`from ueler import {alias.name}` — "
                                    "not exported by ueler/__init__.py",
                                )
                            )
                elif (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "ueler"
                    and node.attr not in api
                ):
                    findings.append(
                        Finding(
                            "python-example",
                            page.rel,
                            start + node.lineno,
                            f"`ueler.{node.attr}` — not exported by ueler/__init__.py",
                        )
                    )
    return findings


def check_nav_coverage(pages: list[Page]) -> list[Finding]:
    """Every page must be reachable from the nav; every nav entry must exist."""

    config = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    nav_start = config.index("\nnav:")
    nav = config[nav_start:]
    referenced = set(re.findall(r"([A-Za-z0-9_./-]+\.md)", nav))

    findings = []
    for page in pages:
        if page.rel not in referenced:
            findings.append(
                Finding("nav", page.rel, 0, "page exists but is not in the mkdocs nav")
            )
    for entry in sorted(referenced):
        if not (DOCS_DIR / entry).exists():
            findings.append(
                Finding("nav", "mkdocs.yml", 0, f"nav points at `{entry}`, which does not exist")
            )
    return findings


def check_source_links(pages: list[Page]) -> list[Finding]:
    """Each developer note cites a ``dev_note/`` source; it has to be there."""

    findings = []
    pattern = re.compile(r"^>\s*Source:\s*\[`([^`]+)`\]")
    for page in pages:
        if not page.is_code_claim_page():
            continue
        for number, line in enumerate(page.lines, start=1):
            match = pattern.match(line.strip())
            if match and not (REPO_ROOT / match.group(1)).exists():
                findings.append(
                    Finding(
                        "source-link",
                        page.rel,
                        number,
                        f"cites `{match.group(1)}`, which does not exist",
                    )
                )
    return findings


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def collect_findings() -> list[Finding]:
    project = load_project_facts()
    pages = load_pages()
    findings: list[Finding] = []
    findings += check_distribution_name(pages, project)
    findings += check_python_versions(pages, project)
    findings += check_extras(pages, project)
    findings += check_repo_paths(pages)
    findings += check_make_targets(pages, make_targets())
    findings += check_env_vars(pages, env_var_mentions_in_code())
    findings += check_code_identifiers(pages, source_identifiers(), source_file_basenames())
    findings += check_ui_labels(pages, plugin_display_names(), ui_label_vocabulary())
    findings += check_python_fences(pages, public_api_names())
    findings += check_nav_coverage(pages)
    findings += check_source_links(pages)
    return sorted(findings, key=lambda item: (item.check, item.page, item.line))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="print nothing; exit status only")
    args = parser.parse_args()

    findings = collect_findings()
    if not args.quiet:
        if findings:
            for finding in findings:
                print(finding)
            print(f"\n{len(findings)} inconsistenc{'y' if len(findings) == 1 else 'ies'}.")
        else:
            print("Docs are consistent with the software.")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
