"""Helpers for testing actions guarded by the confirmation modal (#139 reply 1).

Five destructive actions now route through :meth:`ImageMaskViewer.confirm`, each
from a different plugin with its own viewer stub. Rather than teach every stub
how to hold a dialog, these helpers attach the real :class:`ConfirmDialog` and
:func:`ueler.viewer.confirm_dialog.confirm_via` -- the same function the viewer's
own ``confirm`` delegates to -- to whatever object the test already has. Using
the real ones matters: a ``MagicMock`` viewer would return a truthy mock from
``confirm`` and never invoke the callback, so a test would pass while the
deletion silently never happened.
"""

from types import SimpleNamespace

from ueler.viewer.confirm_dialog import ConfirmDialog, confirm_via


def attach_dialog(viewer):
    """Give ``viewer`` a real dialog and the real ``confirm``; return the dialog."""
    ui_component = getattr(viewer, "ui_component", None)
    if ui_component is None:
        ui_component = SimpleNamespace()
        viewer.ui_component = ui_component
    dialog = ConfirmDialog()
    ui_component.confirm_dialog = dialog
    viewer.confirm = lambda *args, **kwargs: confirm_via(
        viewer.ui_component, *args, **kwargs
    )
    return dialog


def without_dialog(viewer):
    """Make ``viewer`` a host that cannot ask, the way the widget shim leaves it."""
    ui_component = getattr(viewer, "ui_component", None)
    if ui_component is not None and hasattr(ui_component, "confirm_dialog"):
        del ui_component.confirm_dialog
    viewer.confirm = lambda *args, **kwargs: confirm_via(
        getattr(viewer, "ui_component", None), *args, **kwargs
    )
    return viewer


def answer(dialog, which):
    """Press ``confirm`` or ``cancel`` on ``dialog``.

    ``tests.bootstrap`` replaces ``ipywidgets`` with a stub whose ``on_click`` is
    a no-op and whose buttons have no ``click()``, so a real click cannot be
    simulated; invoking the handler runs the same code path.
    """
    getattr(dialog, f"_on_{which}_clicked")(None)
