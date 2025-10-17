"""User-level site customisations for the test suite.

Python imports ``usercustomize`` automatically after ``sitecustomize`` when the
module is present on ``sys.path``. We use the hook to add the repository root to
``sys.path`` and initialise the shared test bootstrap so unittest discovery that
imports test modules as top-level modules still benefits from the dependency
shims.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

if os.environ.get("UELER_SKIP_TEST_BOOTSTRAP") == "1":  # pragma: no cover
    pass
else:  # pragma: no cover - simple bootstrap wiring
    try:
        from tests import bootstrap  # noqa: F401

        if hasattr(bootstrap, "initialize"):
            bootstrap.initialize()
    except Exception:
        # Importing the bootstrap should never be fatal to runtime usage. If it
        # fails (for example, tests package missing), swallow the error and
        # continue with normal interpreter startup.
        pass
