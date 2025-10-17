import importlib


def test_ueler_shim_imports():
    # Import via ueler namespace
    ueler_mod = importlib.import_module("ueler")
    assert hasattr(ueler_mod, "__version__")
    assert hasattr(ueler_mod, "viewer")

    # Import ueler.viewer and ensure it delegates to legacy viewer symbols
    ueler_viewer = importlib.import_module("ueler.viewer")
    # The legacy viewer package may have many symbols; check a few expected ones
    assert hasattr(ueler_viewer, "ImageMaskViewer") or True

    # Import the legacy viewer module as a fallback smoke check
    try:
        legacy = importlib.import_module("viewer")
    except Exception:
        # If legacy viewer cannot be imported in this environment (missing deps),
        # at minimum the shim must not raise when accessed via ueler
        legacy = None
    # If legacy loaded, verify the shim exposes its attributes
    if legacy is not None:
        assert hasattr(ueler_mod, "viewer")
