"""Run a callback before a canvas rasterises, so view-dependent renders land in the same frame.

UELer re-renders the image for the current view (pan, zoom, FOV switch). That
used to happen in a ``draw_event`` handler, but matplotlib fires ``draw_event``
*after* the Agg renderer has rasterised the figure. Under ipympl the PNG sent
for that draw therefore showed the previous render, and the new render only
reached the browser through a second draw (IPython's ``post_execute`` hook,
``plt.draw_all``), one PNG later. Every view change cost two full draws and two
PNG transfers, and the first one was outdated.

:func:`install_pre_draw_hook` wraps ``canvas.draw`` on the instance so the
render runs first and the single frame that follows already contains it. Every
ipympl draw goes through ``canvas.draw`` (``handle_draw`` calls it), and so do
Agg's ``draw_idle`` and ``FigureCanvasBase.draw_idle`` in tests.

While the callback runs, ``canvas.draw_idle`` is a no-op: the draw about to
happen picks up everything the callback changed, so a redraw request issued
from inside it (``update_display`` ends with one) would only queue a second,
identical frame — or, on Agg, recurse into a nested draw.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

_logger = logging.getLogger(__name__)

_HOOKS_ATTR = "_ueler_pre_draw_hooks"


def _noop_draw_idle(*_args: Any, **_kwargs: Any) -> None:
    return None


def install_pre_draw_hook(canvas: Any, callback: Callable[[], Any]) -> None:
    """Call *callback* at the start of every ``canvas.draw()``, before rasterising.

    Several hooks may be installed on one canvas; they run in installation
    order. Exceptions are logged and do not prevent the draw.
    """
    hooks = getattr(canvas, _HOOKS_ATTR, None)
    if hooks is not None:
        hooks.append(callback)
        return

    hooks = [callback]
    original_draw = canvas.draw

    def draw(*args: Any, **kwargs: Any):
        # Shadow draw_idle on the instance for the duration of the hooks only.
        had_instance_attr = "draw_idle" in vars(canvas)
        previous = vars(canvas).get("draw_idle")
        canvas.draw_idle = _noop_draw_idle
        try:
            for hook in list(hooks):
                try:
                    hook()
                except Exception:
                    _logger.exception("[viewer] pre-draw render failed")
        finally:
            if had_instance_attr:
                canvas.draw_idle = previous
            else:
                try:
                    del canvas.draw_idle
                except AttributeError:
                    pass
        return original_draw(*args, **kwargs)

    try:
        setattr(canvas, _HOOKS_ATTR, hooks)
        canvas.draw = draw
    except Exception:  # pragma: no cover - canvases that refuse instance attributes
        _logger.debug("[viewer] could not install pre-draw hook on %r", canvas)
