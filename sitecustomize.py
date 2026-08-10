"""Project-wide site customizations for the test environment.

When the repository root is on ``sys.path`` Python automatically imports this
module. We use the hook to initialise the shared test bootstrap so that
``unittest`` discovery (which may import test modules as top-level modules) still
gets the dependency stubs and safety patches.

The bootstrap is **opt-in**: it only runs when ``UELER_TEST_BOOTSTRAP=1`` is set,
which the ``test-*`` Makefile targets do. It replaces pandas, matplotlib,
ipywidgets and friends with stubs, so having it default to on meant any
interpreter started from a working copy — a developer's notebook after
``pip install -e .``, a script run from the repo root — silently got the fake
dependency stack. Opting in keeps that confined to test runs.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

# ``UELER_SKIP_TEST_BOOTSTRAP=1`` remains honoured as a hard override so an
# environment that sets it keeps working.
if os.environ.get("UELER_TEST_BOOTSTRAP") != "1":  # pragma: no cover
    pass
elif os.environ.get("UELER_SKIP_TEST_BOOTSTRAP") == "1":  # pragma: no cover
    pass
else:  # pragma: no cover - simple bootstrap wiring
    try:
        from tests import bootstrap  # noqa: F401

        if hasattr(bootstrap, "initialize"):
            bootstrap.initialize()
    except Exception as exc:
        # Importing the bootstrap must never be fatal to interpreter startup. It
        # was explicitly requested though, so failing silently would leave a test
        # run looking healthy while running without its stubs — warn instead.
        import warnings

        warnings.warn(
            f"UELER_TEST_BOOTSTRAP=1 was set but the test bootstrap failed to "
            f"initialise ({exc!r}); continuing without dependency stubs.",
            RuntimeWarning,
        )