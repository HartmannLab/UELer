# viewer/widget_frontend_check.py
"""Startup check for the browser-side half of ``anywidget``.

Six UELer plugins render their whole UI as an ``AnyWidget`` subclass — the
channel picker, the ROI expression editor, the Mask Painter class list, the
gallery tile grid, the cell-annotation checkpoint tree, and (via
``jupyter-scatter``) the scatter plot.  Each needs two halves: the Python
package, and a JavaScript asset the *frontend* has to resolve by itself.  When
only the first is present the widgets come up as empty boxes, the kernel logs
nothing, and the user's only clue is a frontend notification naming a package
they never installed deliberately:

    Unable to find widget 'anywidget' version '~0.9.*' from configured widget
    sources ["local"]

That message is the VS Code Jupyter extension's, and ``["local"]`` is the
value of its ``jupyter.widgetScriptSources`` setting — every CDN source
disabled, on-disk lookup the only one left.  Its local provider searches
exactly one directory, ``<sys.prefix>/share/jupyter/nbextensions`` of the
kernel interpreter, whereas JupyterLab reads federated extensions from the
whole ``jupyter_path()`` chain.  So a ``pip install --user`` into a shared,
read-only environment — the normal move on an HPC cluster — puts the package on
``sys.path`` but its data files under ``~/.local/share/jupyter``, and
JupyterLab then works while VS Code does not, from one identical environment.

Nothing here repairs the frontend; that is not reachable from the kernel.  What
it does is turn a silent, near-undiagnosable failure into one warning that
names the condition and the fix, using facts that are all available in-process.
``channel_picker_widget.ANYWIDGET_AVAILABLE`` answers a different question —
*can Python import it* — and is ``True`` in exactly this failure.

The governing rule for every ambiguous case is **stay silent**: a false alarm
on a healthy install would be worse than the problem being reported, so
anything unparseable or unexpected yields a non-warning code.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

_logger = logging.getLogger(__name__)

#: Widget module name, as it appears in both the frontend error and the traits.
MODULE = "anywidget"

#: Codes that mean "a user is actually affected"; everything else stays quiet.
PROBLEM_CODES = frozenset({"missing-assets", "outside-env", "version-skew"})


@dataclass(frozen=True)
class FrontendStatus:
    """Outcome of :func:`check_anywidget_frontend`.

    ``labextensions``/``nbextensions`` list what was found as
    ``(path, version)`` pairs; an nbextension carries no ``package.json``, so
    its version is the Python package's — they ship from the same wheel.
    """

    code: str
    message: str = ""
    python_version: Optional[str] = None
    required_range: Optional[str] = None
    labextensions: Tuple[Tuple[str, Optional[str]], ...] = field(default_factory=tuple)
    nbextensions: Tuple[Tuple[str, Optional[str]], ...] = field(default_factory=tuple)

    @property
    def is_problem(self) -> bool:
        """``True`` only for the codes a user should be told about."""
        return self.code in PROBLEM_CODES


def _version_prefix(spec: str) -> Optional[Tuple[int, ...]]:
    """Return the fixed leading components of an npm range like ``~0.9.*``.

    ``~0.9.*`` → ``(0, 9)``; ``~0.11.*`` → ``(0, 11)``.  ``None`` means "shape
    not recognised", which callers treat as *satisfied* rather than guessing.
    """
    if not spec:
        return None
    cleaned = spec.strip().lstrip("~^=v ")
    parts: List[int] = []
    for chunk in cleaned.split("."):
        if chunk in ("*", "x", "X", ""):
            break
        if not chunk.isdigit():
            return None
        parts.append(int(chunk))
    return tuple(parts) or None


def _satisfies(version: Optional[str], spec: Optional[str]) -> bool:
    """Whether ``version`` falls inside the npm range ``spec``.

    Unknown on either side counts as satisfied — see the silence rule.
    """
    if not version or not spec:
        return True
    wanted = _version_prefix(spec)
    found = _version_prefix(version)
    if wanted is None or found is None:
        return True
    return found[: len(wanted)] == wanted


def _asset_version(directory: Path) -> Optional[str]:
    """Read ``package.json``'s ``version`` from a federated labextension dir."""
    manifest = directory / "package.json"
    try:
        import json

        return json.loads(manifest.read_text(encoding="utf-8")).get("version")
    except Exception:  # pragma: no cover - unreadable or absent manifest
        return None


def _scan(kind: str, jupyter_path) -> Tuple[Tuple[str, Optional[str]], ...]:
    """Find every ``anywidget`` asset directory of ``kind`` on the search path."""
    found: List[Tuple[str, Optional[str]]] = []
    try:
        roots = list(jupyter_path(kind))
    except Exception:  # pragma: no cover - jupyter_core misbehaving
        return ()
    for root in roots:
        try:
            directory = Path(root) / MODULE
            if directory.is_dir():
                found.append((str(directory), _asset_version(directory)))
        except Exception:  # pragma: no cover - unreadable path entry
            continue
    return tuple(found)


def _under_sys_prefix(path: str) -> bool:
    """Whether ``path`` lives inside the running interpreter's prefix.

    This is the single directory VS Code's ``local`` widget source consults.
    """
    try:
        return Path(path).resolve().is_relative_to(Path(sys.prefix).resolve())
    except Exception:  # pragma: no cover - exotic paths
        return str(path).startswith(str(sys.prefix))


def check_anywidget_frontend() -> FrontendStatus:
    """Compare what the kernel will ask the browser for against what is on disk.

    Pure and cheap: a handful of ``is_dir()`` calls and at most one small JSON
    read per hit.  Never raises.
    """
    try:
        import anywidget  # type: ignore[import]
    except Exception:
        return FrontendStatus("absent", "anywidget is not installed")

    widget_class = getattr(anywidget, "AnyWidget", None)
    if widget_class is None:
        # The unit-test bootstrap installs a placeholder module with no
        # AnyWidget; the plugins' own fallbacks cover that case.
        return FrontendStatus("absent", "anywidget is a placeholder module")

    required = None
    trait = getattr(widget_class, "_model_module_version", None)
    if trait is not None:
        required = getattr(trait, "default_value", None)
    python_version = getattr(anywidget, "__version__", None)

    try:
        from jupyter_core.paths import jupyter_path
    except Exception:  # pragma: no cover - jupyter_core always ships with ipykernel
        return FrontendStatus(
            "undetermined",
            "jupyter_core is unavailable, cannot locate frontend assets",
            python_version,
            required,
        )

    labextensions = _scan("labextensions", jupyter_path)
    nbextensions = _scan("nbextensions", jupyter_path)
    common = dict(
        python_version=python_version,
        required_range=required,
        labextensions=labextensions,
        nbextensions=nbextensions,
    )

    if not labextensions and not nbextensions:
        return FrontendStatus(
            "missing-assets",
            (
                f"the {MODULE} browser extension is not installed anywhere on this "
                "machine's Jupyter search path, so UELer's channel picker, ROI "
                "expression editor, Mask Painter class list, gallery tiles and "
                "scatter plot will render as empty boxes. Reinstall it into the "
                f"environment this kernel runs from ({sys.prefix}): "
                f"`pip install --force-reinstall {MODULE}`."
            ),
            **common,
        )

    # A labextension carries its version; an nbextension does not, but ships
    # from the same wheel as the Python package, so it is the Python version.
    local = [
        (path, version)
        for path, version in labextensions
        if version is not None and _under_sys_prefix(path)
    ]
    if local and not any(_satisfies(version, required) for _, version in local):
        paths = ", ".join(f"{path} (v{version})" for path, version in local)
        return FrontendStatus(
            "version-skew",
            (
                f"the installed {MODULE} browser extension does not match the "
                f"{MODULE} Python package: the kernel will ask the browser for "
                f"'{required}' but the only extension found is {paths}. This "
                "usually means two installs are overlapping — reinstall with "
                f"`pip install --force-reinstall {MODULE}`."
            ),
            **common,
        )

    if any(_under_sys_prefix(path) for path, _ in nbextensions):
        return FrontendStatus("ok", "", **common)

    located = ", ".join(path for path, _ in labextensions + nbextensions)
    return FrontendStatus(
        "outside-env",
        (
            f"the {MODULE} browser extension is installed at {located}, which is "
            f"outside this kernel's environment ({sys.prefix}). JupyterLab will "
            "find it there, but VS Code's 'local' widget source only searches "
            "<environment>/share/jupyter/nbextensions, so in VS Code UELer's "
            "channel picker, ROI expression editor, Mask Painter class list, "
            "gallery tiles and scatter plot will render as empty boxes with the "
            f"notification \"Unable to find widget '{MODULE}' version "
            f"'{required}'\". Fix it either by reinstalling into this "
            f"environment rather than with `pip install --user`, or by adding "
            '"jsdelivr.com" to the VS Code setting "jupyter.widgetScriptSources".'
        ),
        **common,
    )


def warn_about_widget_frontend(status: Optional[FrontendStatus] = None) -> FrontendStatus:
    """Run the check and log one warning if a user is actually affected.

    Returns the status either way.  Never raises: a diagnostic must not be able
    to prevent the viewer from opening.
    """
    try:
        if status is None:
            status = check_anywidget_frontend()
    except Exception:  # pragma: no cover - defensive, the check is already total
        return FrontendStatus("undetermined", "frontend check failed")

    if status.is_problem:
        _logger.warning("UELer widget frontend: %s", status.message)
    else:
        _logger.debug("UELer widget frontend check: %s", status.code)
    return status
