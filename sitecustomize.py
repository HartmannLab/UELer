"""Project-wide site customizations for the test environment.

When the repository root is on ``sys.path`` Python automatically imports this
module. We use the hook to initialise the shared test bootstrap so that
``unittest`` discovery (which may import test modules as top-level modules) still
gets the dependency stubs and safety patches.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

# Allow opting out (for example, production notebooks) by setting the
# environment variable ``UELER_SKIP_TEST_BOOTSTRAP=1``.
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