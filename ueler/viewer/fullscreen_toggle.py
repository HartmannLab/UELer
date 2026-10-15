"""Full-screen toggle for the whole viewer (#104).

A single icon button, mounted at the top-left of the viewer's root box, that
takes the viewer full screen and back.

How it finds what to enlarge
----------------------------
The button is an ``anywidget`` rendered *inside* the root ``VBox``. Python tags
that box with :data:`HOST_CLASS`, and the ESM resolves its host with
``el.closest('.ueler-root')``. Nothing walks the DOM by size or by framework
class names — every earlier attempt at #104 that did so picked the wrong node in
VS Code (the button's own wrapper, or an empty box), and fullscreened that.

Strategies, in order
--------------------
1. **Native** — ``host.requestFullscreen()``. Used wherever the browser allows
   it, which is JupyterLab and Notebook in practice.
2. When the Fullscreen API is missing or refuses:

   * **Expand** (VS Code) — reached from a notice, not directly: in VS Code
     the button first explains that full screen is not supported there and
     points to JupyterLab / JupyterHub, offering this as the alternative.
     The VS Code output webview may not go full screen,
     and the code-cell editors are drawn by VS Code on a layer *above* that
     webview, so an overlay inside it is covered by them (markdown cells render
     inside the webview, which is why only code cells showed through). The host
     instead gets :data:`EXPANDED_CLASS` and stays in the notebook flow, so
     the cells below are pushed down rather than drawn over it. Its frame is
     about one editor-screen tall -- ``min(innerHeight, screen.availHeight -
     160)``, taken once on entry, because the webview is sized to the notebook
     content and its ``innerHeight`` grows with the viewer -- and content
     taller than that scrolls inside the frame. The frame is not anchored:
     an output can neither set nor read the notebook's scroll position (see
     the issue #104 notes).
   * **Maximize** (any other page) — the host gets :data:`MAXIMIZED_CLASS`, a
     ``position: fixed`` overlay over the page.

Esc leaves every mode. The resolved mode is synced back to Python as
:attr:`FullscreenToggle.mode`, and the space the viewer now has as
:attr:`FullscreenToggle.viewport`.

The button also reports the root's width in every mode
(:attr:`FullscreenToggle.host_width`). :func:`attach_canvas_fit` uses it for
the default layout -- a square image filling the width the panels leave, so
the side column sits at the right edge -- and the viewport while enlarged.

The ESM holds only transport logic; the sizing decisions are in Python, where
the unit tests reach them.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

_logger = logging.getLogger(__name__)

#: Class Python adds to the viewer's root box; the ESM's ``closest()`` target.
HOST_CLASS = "ueler-root"
#: Class the ESM adds to the host for the fixed-overlay fallback.
MAXIMIZED_CLASS = "ueler-maximized"
#: Class the ESM adds to the host for the in-flow fallback used in VS Code.
EXPANDED_CLASS = "ueler-expanded"
#: The values :attr:`FullscreenToggle.mode` can take.
MODES = ("off", "native", "maximized", "expanded")

_ENTER_ICON = (
    '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" '
    'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">'
    '<path d="M2 6V2h4M10 2h4v4M14 10v4h-4M6 14H2v-4"/></svg>'
)
_EXIT_ICON = (
    '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" '
    'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">'
    '<path d="M6 2v4H2M14 6h-4V2M10 14v-4h4M2 10h4v4"/></svg>'
)

_ESM = """
const ENTER_ICON = __ENTER_ICON__;
const EXIT_ICON = __EXIT_ICON__;

// Room VS Code's own chrome takes from the screen: title bar, tabs,
// breadcrumbs, the notebook toolbar and the status bar.
const VSCODE_CHROME_PX = 160;
const MIN_EXPANDED_PX = 400;
// Debounce for width reports, so a window drag refits the canvas once.
const WIDTH_SETTLE_MS = 150;
// A root narrower than this is hidden or still laying out; not reported.
const MIN_REPORTED_WIDTH = 200;

// VS Code's notebook webview marks its <body> with the theme kind.
function isVSCodeWebview() {
  const body = document.body;
  if (!body) return false;
  if (body.hasAttribute && body.hasAttribute("data-vscode-theme-kind")) return true;
  return Array.from(body.classList || []).some((c) => c.startsWith("vscode-"));
}

function render({ model, el }) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "ueler-fs-btn";
  el.appendChild(btn);

  const host = () => el.closest("." + model.get("host_class"));
  const maximizedClass = model.get("maximized_class");
  const expandedClass = model.get("expanded_class");

  // Canvases (ipympl, jupyter-scatter, bqplot) measure their container on
  // window resize, so announce the new size once the layout has settled.
  const announceResize = () =>
    requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));

  function sync() {
    const mode = model.get("mode");
    const on = mode !== "off";
    btn.innerHTML = on ? EXIT_ICON : ENTER_ICON;
    btn.title = on ? "Exit full screen (Esc)" : "Full screen";
    btn.setAttribute("aria-label", btn.title);
    btn.setAttribute("aria-pressed", on ? "true" : "false");
  }

  function setMode(mode) {
    if (model.get("mode") === mode) return;
    model.set("mode", mode);
    model.save_changes();
  }

  function setViewport(size) {
    const old = model.get("viewport") || [];
    if (old.length === size.length && old.every((v, i) => v === size[i])) return;
    model.set("viewport", size);
    model.save_changes();
  }

  // The node this view enlarged, so a second view of the same model (the
  // viewer displayed twice) never tears down or reports on the other one.
  let active = null;

  // Height for the in-flow "expanded" mode: about one editor-screen. VS Code
  // sizes its output webview to the notebook *content*, so ``innerHeight``
  // grows with the viewer itself and cannot be the editor height on its own;
  // the screen bounds it. It is taken once, on entry -- re-measuring would
  // feed the viewer's own growth back into its size.
  let expandedHeight = 0;
  function computeExpandedHeight() {
    const screenHeight = (window.screen && window.screen.availHeight) || window.innerHeight;
    const height = Math.min(window.innerHeight, screenHeight - VSCODE_CHROME_PX);
    return Math.max(MIN_EXPANDED_PX, Math.round(height));
  }

  // Report the space the viewer now occupies, for Python to fit the canvas.
  function measure() {
    if (!active) return;
    requestAnimationFrame(() => {
      if (!active) return;
      const rect = active.getBoundingClientRect();
      const height = model.get("mode") === "expanded" ? expandedHeight : rect.height;
      setViewport([Math.round(rect.width), Math.round(height)]);
    });
  }

  function applyClass(node, cls, mode) {
    node.classList.add(cls);
    active = node;
    setMode(mode);
    measure();
    announceResize();
  }

  // No scroll anchoring in "expanded". A widget has no way to set the
  // notebook's scroll position, and every indirect route failed in VS Code:
  // vetoing or rewriting wheel events stopped unevenly or bounced, and
  // clicking an in-page link scrolled the webview's own document instead,
  // which blanked every output in it. See issue104_fullscreen_mode.md.

  function fallback(node) {
    if (isVSCodeWebview()) {
      expandedHeight = computeExpandedHeight();
      node.style.setProperty("--ueler-expanded-height", expandedHeight + "px");
      applyClass(node, expandedClass, "expanded");
    } else {
      applyClass(node, maximizedClass, "maximized");
    }
  }

  function enter() {
    const node = host();
    if (!node) return;
    const native = document.fullscreenEnabled && typeof node.requestFullscreen === "function";
    if (!native) {
      fallback(node);
      return;
    }
    node.requestFullscreen().then(
      () => { active = node; setMode("native"); measure(); announceResize(); },
      () => fallback(node),
    );
  }

  // Undo whatever this view applied. Runs on every switch to "off", whether
  // the button, Esc, or Python (``toggle.mode = "off"``) asked for it.
  function teardown() {
    const node = active;
    active = null;
    if (!node) return;
    node.classList.remove(maximizedClass);
    node.classList.remove(expandedClass);
    node.style.removeProperty("--ueler-expanded-height");
    if (document.fullscreenElement === node) {
      document.exitFullscreen().catch(() => {});
    }
    setViewport([]);
    announceResize();
  }

  // VS Code refuses full screen to notebook outputs and draws code cells
  // above them, so instead of trying, the button says so and points to
  // JupyterLab / JupyterHub, offering grow-in-place as what VS Code allows.
  // The notice is inline, under the button: VS Code's output webview is as
  // tall as the whole notebook, so a centred overlay could land far off
  // screen, and it would sit under the code cells anyway.
  let notice = null;
  function closeNotice() {
    if (!notice) return;
    notice.remove();
    notice = null;
  }
  function noticeButton(label, onClick) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ueler-fs-notice-btn";
    button.textContent = label;
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      onClick();
    });
    return button;
  }
  function showNotice() {
    if (notice) return;
    notice = document.createElement("div");
    notice.className = "ueler-fs-notice";
    notice.setAttribute("role", "dialog");
    notice.setAttribute("aria-label", "Full screen is not supported in VS Code");
    const title = document.createElement("div");
    title.className = "ueler-fs-notice-title";
    title.textContent = "Full screen is not supported in VS Code";
    const text = document.createElement("div");
    text.className = "ueler-fs-notice-text";
    text.textContent = "VS Code does not let notebook outputs go full screen. "
      + "If you need it, open this notebook in JupyterLab or JupyterHub, "
      + "where this button shows the viewer full screen.";
    const actions = document.createElement("div");
    actions.className = "ueler-fs-notice-actions";
    const grow = noticeButton("Enlarge in place instead", () => {
      closeNotice();
      const node = host();
      if (node) fallback(node);
    });
    const close = noticeButton("Close", closeNotice);
    actions.appendChild(grow);
    actions.appendChild(close);
    notice.appendChild(title);
    notice.appendChild(text);
    notice.appendChild(actions);
    el.appendChild(notice);
    if (typeof close.focus === "function") close.focus();
  }

  btn.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (model.get("mode") !== "off") setMode("off");
    else if (notice) closeNotice();
    else if (isVSCodeWebview()) showNotice();
    else enter();
  });

  // Esc in native mode is handled by the browser; this keeps the trait honest.
  const onFullscreenChange = () => {
    if (active && model.get("mode") === "native" && document.fullscreenElement !== active) {
      setMode("off");
    }
  };
  // Esc in the fallback modes is ours to handle.
  const onKeydown = (event) => {
    if (event.key === "Escape" && notice) {
      event.stopPropagation();
      closeNotice();
      return;
    }
    const mode = model.get("mode");
    if (event.key === "Escape" && active && (mode === "maximized" || mode === "expanded")) {
      event.stopPropagation();
      setMode("off");
    }
  };
  // Window resized while enlarged: re-measure, debounced. Not in "expanded"
  // mode, where the window grows *because* the viewer did (see above).
  let resizeTimer = null;
  const onWindowResize = () => {
    if (!active || model.get("mode") === "expanded") return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(measure, 200);
  };
  document.addEventListener("fullscreenchange", onFullscreenChange);
  document.addEventListener("keydown", onKeydown, true);
  window.addEventListener("resize", onWindowResize);

  const onModeChange = () => {
    if (model.get("mode") === "off") teardown();
    sync();
  };
  model.on("change:mode", onModeChange);
  sync();

  // Always report the root's width, in every mode: Python fits the image to
  // it, so the side column sits at the right edge (``attach_canvas_fit``).
  // Width is safe to follow where height was not: the output webview is as
  // wide as the editor, and the root's width (100 % of the output) does not
  // depend on what is inside it, so refitting the canvas cannot feed back.
  // The root may not be attached yet when ``render`` runs, hence the retry.
  let widthObserver = null;
  let widthTimer = null;
  let attachTries = 0;
  function reportWidth() {
    const node = host();
    if (!node) return;
    const width = Math.round(node.getBoundingClientRect().width);
    if (width < MIN_REPORTED_WIDTH) return;
    if (Math.abs(width - (model.get("host_width") || 0)) < 2) return;
    model.set("host_width", width);
    model.save_changes();
  }
  function watchWidth() {
    const node = host();
    if (!node) {
      if (attachTries++ < 60) requestAnimationFrame(watchWidth);
      return;
    }
    if (typeof ResizeObserver !== "function") {
      reportWidth();
      return;
    }
    widthObserver = new ResizeObserver(() => {
      clearTimeout(widthTimer);
      widthTimer = setTimeout(reportWidth, WIDTH_SETTLE_MS);
    });
    widthObserver.observe(node);
  }
  watchWidth();

  return () => {
    if (widthObserver) widthObserver.disconnect();
    clearTimeout(widthTimer);
    document.removeEventListener("fullscreenchange", onFullscreenChange);
    document.removeEventListener("keydown", onKeydown, true);
    window.removeEventListener("resize", onWindowResize);
    clearTimeout(resizeTimer);
    model.off("change:mode", onModeChange);
    closeNotice();
    teardown();
  };
}

export default { render };
""".replace("__ENTER_ICON__", repr(_ENTER_ICON)).replace("__EXIT_ICON__", repr(_EXIT_ICON))

_CSS = """
.ueler-fs-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  padding: 0;
  border: 1px solid var(--jp-border-color1, #c0c0c0);
  border-radius: 4px;
  background: var(--jp-layout-color1, #ffffff);
  color: var(--jp-ui-font-color1, #333333);
  cursor: pointer;
}
.ueler-fs-btn:hover {
  background: var(--jp-layout-color2, #eeeeee);
}
.ueler-fs-btn:focus-visible {
  outline: 2px solid var(--jp-brand-color1, #1976d2);
  outline-offset: 1px;
}

/* VS Code notice: inline under the button, in the page flow. */
.ueler-fs-notice {
  box-sizing: border-box;
  max-width: 440px;
  margin-top: 6px;
  padding: 10px 12px;
  border: 1px solid var(--jp-border-color1, var(--vscode-editorWidget-border, #c0c0c0));
  border-radius: 6px;
  background: var(--jp-layout-color1, var(--vscode-editorWidget-background, #ffffff));
  color: var(--jp-ui-font-color1, var(--vscode-foreground, #333333));
  font-size: 13px;
  line-height: 1.4;
}
.ueler-fs-notice-title {
  font-weight: 600;
  margin-bottom: 4px;
}
.ueler-fs-notice-actions {
  display: flex;
  gap: 8px;
  margin-top: 10px;
}
.ueler-fs-notice-btn {
  padding: 3px 10px;
  border: 1px solid var(--jp-border-color1, var(--vscode-button-border, #c0c0c0));
  border-radius: 4px;
  background: var(--jp-layout-color2, var(--vscode-button-secondaryBackground, #eeeeee));
  color: inherit;
  cursor: pointer;
}

/* Native and maximize: an opaque, scrollable page for the viewer. */
.ueler-root:fullscreen,
.ueler-root.ueler-maximized {
  box-sizing: border-box !important;
  width: 100% !important;
  max-width: none !important;
  height: 100% !important;
  overflow: auto !important;
  padding: 8px 12px !important;
  background: var(--jp-layout-color0, var(--vscode-editor-background, #ffffff)) !important;
}
.ueler-root::backdrop {
  background: var(--jp-layout-color0, var(--vscode-editor-background, #ffffff));
}

/* Maximize fallback: a fixed overlay over the page. */
.ueler-root.ueler-maximized {
  position: fixed !important;
  inset: 0 !important;
  z-index: 9000 !important;
  margin: 0 !important;
}

/* Expand fallback (VS Code): one editor-screen tall, in the notebook flow. */
.ueler-root.ueler-expanded {
  box-sizing: border-box !important;
  width: 100% !important;
  max-width: none !important;
  height: var(--ueler-expanded-height, 100vh) !important;
  overflow: auto !important;
}
/* The frame is the editor's height; content taller than that scrolls inside
   it as one page rather than the flex column squeezing its rows to fit. */
.ueler-root.ueler-expanded > * {
  flex-shrink: 0 !important;
}
"""


def _anywidget_module():
    """Return the imported ``anywidget`` module, or ``None`` if unavailable."""
    try:
        import anywidget  # type: ignore[import]
    except ImportError:
        return None
    return anywidget


_ANYWIDGET = _anywidget_module()
#: ``True`` when a *real* anywidget is installed, i.e. the button can render.
ANYWIDGET_AVAILABLE = bool(_ANYWIDGET is not None and hasattr(_ANYWIDGET, "AnyWidget"))

import traitlets  # type: ignore[import]  # noqa: E402


def _validate_mode(proposal):
    mode = proposal["value"]
    return mode if mode in MODES else "off"


if ANYWIDGET_AVAILABLE:

    class FullscreenToggle(_ANYWIDGET.AnyWidget):  # type: ignore[misc,name-defined]
        """Icon button that takes the enclosing ``.ueler-root`` box full screen."""

        mode: str = traitlets.Unicode("off").tag(sync=True)
        #: ``[width, height]`` in CSS pixels while enlarged, ``[]`` otherwise.
        viewport: list = traitlets.List(traitlets.Int()).tag(sync=True)
        #: Width of the root box in CSS pixels, reported in every mode.
        host_width: int = traitlets.Int(0).tag(sync=True)
        host_class: str = traitlets.Unicode(HOST_CLASS).tag(sync=True)
        maximized_class: str = traitlets.Unicode(MAXIMIZED_CLASS).tag(sync=True)
        expanded_class: str = traitlets.Unicode(EXPANDED_CLASS).tag(sync=True)

        _css = _CSS
        _esm = _ESM

        @traitlets.validate("mode")
        def _validate_mode(self, proposal):
            return _validate_mode(proposal)

else:  # pragma: no cover - exercised by the headless unit-test bootstrap

    class FullscreenToggle(traitlets.HasTraits):  # type: ignore[no-redef]
        """Headless stand-in with the same traits (no anywidget available)."""

        mode: str = traitlets.Unicode("off")
        viewport: list = traitlets.List(traitlets.Int())
        host_width: int = traitlets.Int(0)
        host_class: str = traitlets.Unicode(HOST_CLASS)
        maximized_class: str = traitlets.Unicode(MAXIMIZED_CLASS)
        expanded_class: str = traitlets.Unicode(EXPANDED_CLASS)

        _css = _CSS
        _esm = _ESM

        def __init__(self, **kwargs):
            layout = kwargs.pop("layout", None)
            super().__init__(**kwargs)
            self.layout = layout if layout is not None else SimpleNamespace()

        @traitlets.validate("mode")
        def _validate_mode(self, proposal):
            return _validate_mode(proposal)


def build_fullscreen_toggle():
    """Return a :class:`FullscreenToggle`, or ``None`` when it cannot render.

    Without a real ``anywidget`` the button would be an empty box, so the viewer
    simply leaves it out; the layout is otherwise unchanged.
    """
    if not ANYWIDGET_AVAILABLE:
        return None
    return FullscreenToggle()


# --- Canvas fitting -----------------------------------------------------------

#: Width of the left control panel (``display_ui``'s ``left_panel``).
LEFT_PANEL_PX = 350
#: Gaps, padding and scrollbars around the canvas, horizontally.
CHROME_WIDTH_PX = 64
#: Toolbar row, Lasso row, gaps and padding above and below the canvas.
CHROME_HEIGHT_PX = 120
#: Never shrink the canvas below this, whatever the viewport.
MIN_CANVAS_INCHES = 3.0
#: Never trust a viewport side larger than this (a 4K screen is 3840 CSS px).
#: A frontend bug that reports a runaway size must not produce a figure too
#: large for ipympl to send, which shows as a broken image.
MAX_VIEWPORT_PX = 4096


def fit_canvas_inches(viewport, *, side_width_px, css_dpi):
    """Return the ``(width, height)`` in inches that fills ``viewport``.

    ``viewport`` is ``[width, height]`` in CSS pixels, as the ESM reports it;
    ``None`` is returned when it is empty or malformed. ``css_dpi`` is how many
    CSS pixels the figure draws per inch (ipympl scales ``fig.dpi`` by the
    device pixel ratio, so this is the figure's *original* dpi).
    """
    try:
        width_px, height_px = (int(v) for v in viewport)
    except (TypeError, ValueError):
        return None
    if width_px <= 0 or height_px <= 0 or not css_dpi or css_dpi <= 0:
        return None
    width_px = min(width_px, MAX_VIEWPORT_PX)
    height_px = min(height_px, MAX_VIEWPORT_PX)
    usable_w = width_px - LEFT_PANEL_PX - max(0, int(side_width_px)) - CHROME_WIDTH_PX
    usable_h = height_px - CHROME_HEIGHT_PX
    return (
        max(MIN_CANVAS_INCHES, usable_w / css_dpi),
        max(MIN_CANVAS_INCHES, usable_h / css_dpi),
    )


def fit_square_inches(width_px, *, side_width_px, css_dpi):
    """Return the square ``(side, side)`` in inches filling the root's width.

    The default layout: the image takes whatever width the left panel and the
    side column leave, so the side column sits at the right edge, and stays
    square. ``None`` when the width is unknown or malformed.
    """
    try:
        width_px = int(width_px)
    except (TypeError, ValueError):
        return None
    if width_px <= 0 or not css_dpi or css_dpi <= 0:
        return None
    width_px = min(width_px, MAX_VIEWPORT_PX)
    usable = width_px - LEFT_PANEL_PX - max(0, int(side_width_px)) - CHROME_WIDTH_PX
    side = max(MIN_CANVAS_INCHES, usable / css_dpi)
    return (side, side)


def attach_canvas_fit(toggle, viewer, *, side_width_px=0):
    """Keep the viewer's image figure sized to the space ``toggle`` reports.

    * Enlarged (``mode != "off"`` with a ``viewport``): fill the screen,
      :func:`fit_canvas_inches`.
    * Otherwise, once the root's width is known: a square filling the width,
      :func:`fit_square_inches` -- the default layout.
    * Neither known: the figure's own size, as created.

    Failures are logged, never raised: a canvas that keeps its size is not
    worth breaking the viewer for.
    """
    state = {"original": None}

    def _figure():
        display = getattr(viewer, "image_display", None)
        return getattr(display, "fig", None)

    def _target(fig):
        css_dpi = getattr(fig, "_original_dpi", None) or getattr(fig, "dpi", None)
        if toggle.mode != "off":
            size = fit_canvas_inches(toggle.viewport, side_width_px=side_width_px, css_dpi=css_dpi)
            if size is not None:
                return size
        size = fit_square_inches(toggle.host_width, side_width_px=side_width_px, css_dpi=css_dpi)
        if size is not None:
            return size
        return state["original"]

    def _refresh(_change=None):
        fig = _figure()
        if fig is None:
            return
        current = tuple(float(v) for v in fig.get_size_inches())
        if state["original"] is None:
            state["original"] = current
        size = _target(fig)
        if size is None:
            return
        # Width reports arrive on every resize; skip ones that change nothing.
        if all(abs(a - b) < 0.02 for a, b in zip(size, current)):
            return
        try:
            fig.set_size_inches(*size, forward=True)
            factor = getattr(viewer, "current_downsample_factor", None)
            if factor is not None and hasattr(viewer, "update_display"):
                viewer.update_display(factor)
            fig.canvas.draw_idle()
        except Exception:  # pragma: no cover - defensive; depends on the backend
            _logger.debug("[fullscreen] could not resize the canvas", exc_info=True)

    toggle.observe(_refresh, names=["viewport", "mode", "host_width"])
