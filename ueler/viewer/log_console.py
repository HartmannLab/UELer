"""Bottom-docked log console for the UELer viewer (debug mode).

A :class:`logging.Handler` that renders ``ueler.*`` log records into an
``ipywidgets.Output`` widget shown at the bottom of the viewer UI when
``ImageMaskViewer(debug=True)``.

Rationale: ipykernel captures ``sys.stdout``/``sys.stderr`` (and the OS file
descriptors) and routes them to iopub, so plain ``print()`` and ordinary stream
handlers land in the notebook cell or inside a plugin's ``Output`` widget — not
in a place suited to Python-end log messages. Rendering records into a dedicated
``Output`` widget side-steps all of that: the messages live in their own
scrollable, copyable, clearable panel driven entirely by the standard
``logging`` module.

The module loggers across the package use ``logging.getLogger(__name__)`` (e.g.
``ueler.viewer.plugin.heatmap_layers``), which propagate to ``logging.getLogger(
"ueler")``. Attaching this handler there captures all of them.
"""

from __future__ import annotations

import logging

import ipywidgets as widgets

_PKG_LOGGER = "ueler"
_MAX_ENTRIES = 1000  # cap retained records to bound widget memory
_HANDLER: "OutputWidgetHandler | None" = None  # module singleton


class OutputWidgetHandler(logging.Handler):
    """Logging handler that renders records into an ``ipywidgets.Output``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                height="200px",
                overflow="auto",
                border="1px solid var(--jp-border-color2, #cccccc)",
            )
        )

    def emit(self, record):
        try:
            formatted = self.format(record)
        except Exception:  # pragma: no cover - defensive, mirrors logging.Handler
            self.handleError(record)
            return
        entry = {
            "name": "stdout",
            "output_type": "stream",
            "text": formatted + "\n",
        }
        # Newest on top; cap to bound memory.
        self.out.outputs = ((entry,) + tuple(self.out.outputs))[:_MAX_ENTRIES]

    def clear_logs(self):
        """Empty the console."""
        self.out.clear_output()


def get_log_console_handler() -> OutputWidgetHandler:
    """Return the module-singleton console handler, creating it on first use."""
    global _HANDLER
    if _HANDLER is None:
        _HANDLER = OutputWidgetHandler()
        _HANDLER.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    return _HANDLER


def enable_log_console() -> OutputWidgetHandler:
    """Attach the console handler to the ``ueler`` logger (idempotent).

    Sets ``propagate=False`` so records land only in the console widget (not the
    notebook cell), and ensures the logger level is at most ``DEBUG``.
    """
    handler = get_log_console_handler()
    logger = logging.getLogger(_PKG_LOGGER)
    if handler not in logger.handlers:
        logger.addHandler(handler)
    logger.propagate = False
    if logger.level == logging.NOTSET or logger.level > logging.DEBUG:
        logger.setLevel(logging.DEBUG)
    return handler


def disable_log_console() -> None:
    """Detach the console handler and restore default propagation."""
    logger = logging.getLogger(_PKG_LOGGER)
    if _HANDLER is not None and _HANDLER in logger.handlers:
        logger.removeHandler(_HANDLER)
    logger.propagate = True


def build_log_console_panel(handler: OutputWidgetHandler):
    """Return a bottom-dock panel: a header (title + Clear) above the Output."""
    clear_btn = widgets.Button(
        description="Clear",
        icon="trash",
        layout=widgets.Layout(width="90px"),
    )
    clear_btn.on_click(lambda _btn: handler.clear_logs())
    header = widgets.HBox(
        [widgets.HTML("<b>UELer Log Console</b>"), clear_btn],
        layout=widgets.Layout(
            justify_content="space-between", align_items="center"
        ),
    )
    return widgets.VBox(
        [header, handler.out],
        layout=widgets.Layout(
            width="100%",
            margin="12px 0 0 0",
            border="1px solid var(--jp-border-color2, #cccccc)",
            padding="8px",
            gap="4px",
        ),
    )
