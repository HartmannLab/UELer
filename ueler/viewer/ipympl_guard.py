# viewer/ipympl_guard.py
"""Containment for an upstream ipympl crash.

``ipympl.backend_nbagg.Canvas._handle_message`` dereferences ``self.manager``
without checking it, but ``FigureCanvasBase.manager`` is legitimately ``None``
in two windows:

* for the duration of ``savefig`` — ``print_figure`` wraps itself in
  ``cbook._setattr_cm(self, manager=None)`` so that writing a file cannot
  resize the GUI widget;
* between ``Canvas(figure)`` and ``FigureManager(canvas, num)`` in
  ``new_figure_manager_given_figure``, by which point the widget's comm is
  already open.

A comm message delivered inside either window raises
``AttributeError: 'NoneType' object has no attribute 'handle_json'`` from the
kernel's message handler, with no application frames to point at.  UELer's own
trigger for this was pyplot use on the export worker thread, fixed at the
source in ``plugin/export_fovs.py``; this guard stays because UELer does not
own every path into that line — a notebook reconnecting to a stale widget model
after a kernel restart reaches it too.

Dropping the message is the only available response: one that arrives while the
manager is ``None`` cannot be dispatched at all, so the choice is between
discarding it and raising out of the comm handler.
"""

import logging

_logger = logging.getLogger(__name__)

_GUARD_FLAG = "_ueler_none_manager_guard"


def install_canvas_message_guard() -> bool:
    """Make ipympl's canvas message handler ignore messages that arrive while
    the figure manager is unset.

    Idempotent, and a no-op when ipympl is not installed.  Returns ``True`` if
    the guard is in place afterwards, ``False`` if it could not be installed.
    """
    try:
        from ipympl.backend_nbagg import Canvas
    except Exception:  # pragma: no cover - ipympl is optional at import time
        return False

    handler = getattr(Canvas, "_handle_message", None)
    if handler is None:  # pragma: no cover - upstream rename
        _logger.debug("ipympl Canvas has no _handle_message; guard not installed")
        return False
    if getattr(handler, _GUARD_FLAG, False):
        return True

    def _handle_message(self, object, content, buffers):
        if getattr(self, "manager", None) is None:
            # 'closing' is the one message the upstream handler services without
            # touching the manager, so let it through to keep _closed accurate.
            if isinstance(content, dict) and content.get("type") == "closing":
                self._closed = True
                return
            _logger.debug(
                "ipympl message dropped, figure manager unset: %r",
                content.get("type") if isinstance(content, dict) else content,
            )
            return
        return handler(self, object, content, buffers)

    setattr(_handle_message, _GUARD_FLAG, True)
    Canvas._handle_message = _handle_message
    _logger.debug("ipympl none-manager guard installed on Canvas._handle_message")
    return True
