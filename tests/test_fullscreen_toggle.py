"""Tests for the full-screen toggle (issue #104).

Every earlier attempt at #104 failed in VS Code by fullscreening the wrong DOM
node, so these tests pin down *which* node is enlarged as much as *whether*
anything is:

* **The Python contract** — traits, mode validation, and the root box being
  tagged with the class the ESM looks for.
* **The browser behaviour** — the ESM is executed under ``node`` against a
  minimal fake DOM, covering native fullscreen, the maximize fallback, Esc, a
  Python-side reset and unmount (skipped when ``node`` is unavailable).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tests.bootstrap  # noqa: F401  # Ensure shared test bootstrap runs

from ueler.viewer import fullscreen_toggle
from ueler.viewer.fullscreen_toggle import (
    CHROME_HEIGHT_PX,
    CHROME_WIDTH_PX,
    EXPANDED_CLASS,
    HOST_CLASS,
    LEFT_PANEL_PX,
    MAXIMIZED_CLASS,
    MAX_VIEWPORT_PX,
    MIN_CANVAS_INCHES,
    FullscreenToggle,
    _CSS,
    _ESM,
    attach_canvas_fit,
    build_fullscreen_toggle,
    fit_canvas_inches,
    fit_square_inches,
)

_NODE = shutil.which("node") or shutil.which("nodejs")


class FullscreenToggleContractTests(unittest.TestCase):
    def test_defaults(self):
        toggle = FullscreenToggle()
        self.assertEqual(toggle.mode, "off")
        self.assertEqual(toggle.host_class, HOST_CLASS)
        self.assertEqual(toggle.maximized_class, MAXIMIZED_CLASS)
        self.assertEqual(toggle.expanded_class, EXPANDED_CLASS)
        self.assertEqual(toggle.viewport, [])

    def test_unknown_mode_is_coerced_to_off(self):
        toggle = FullscreenToggle()
        toggle.mode = "maximized"
        self.assertEqual(toggle.mode, "maximized")
        toggle.mode = "sideways"
        self.assertEqual(toggle.mode, "off")

    def test_builder_omits_button_without_anywidget(self):
        with mock.patch.object(fullscreen_toggle, "ANYWIDGET_AVAILABLE", False):
            self.assertIsNone(build_fullscreen_toggle())

    def test_esm_resolves_host_by_class_not_by_dom_walk(self):
        self.assertIn('el.closest("." + model.get("host_class"))', _ESM)
        # Only the wheel guard walks up the tree, and only from an event target.
        self.assertNotIn("el.parentElement", _ESM)

    def test_css_styles_both_modes_on_the_host(self):
        self.assertIn(f".{HOST_CLASS}:fullscreen", _CSS)
        self.assertIn(f".{HOST_CLASS}.{MAXIMIZED_CLASS}", _CSS)
        self.assertIn("position: fixed", _CSS)
        self.assertIn(f".{HOST_CLASS}.{EXPANDED_CLASS}", _CSS)
        self.assertIn("var(--ueler-expanded-height", _CSS)
        # Content taller than the editor scrolls inside the frame, unsqueezed.
        self.assertIn(f".{HOST_CLASS}.{EXPANDED_CLASS} > *", _CSS)
        self.assertIn("flex-shrink: 0", _CSS)


class DisplayUiMountTests(unittest.TestCase):
    """``display_ui`` must tag the root, or the ESM finds no host."""

    def test_root_carries_host_class_and_button_is_first(self):
        from ueler.viewer import ui_components

        captured = {}

        class _Root:
            def __init__(self, children, layout=None):
                self.children = list(children)
                self.classes = []

            def add_class(self, name):
                self.classes.append(name)

        real_vbox = ui_components.VBox

        def fake_vbox(children=(), *args, **kwargs):
            if kwargs.get("layout") is not None and getattr(kwargs["layout"], "gap", None) == "12px":
                root = _Root(children)
                captured["root"] = root
                return root
            return real_vbox(children, *args, **kwargs)

        sentinel = object()
        viewer = _minimal_viewer()
        with mock.patch.object(ui_components, "VBox", side_effect=fake_vbox), \
                mock.patch.object(ui_components, "build_fullscreen_toggle", return_value=sentinel), \
                mock.patch.object(ui_components, "attach_canvas_fit") as attach, \
                mock.patch.object(ui_components, "HBox", side_effect=lambda children, **kw: ("hbox", tuple(children))), \
                mock.patch.object(ui_components, "display"):
            ui_components.display_ui(viewer)

        root = captured["root"]
        self.assertIn(HOST_CLASS, root.classes)
        self.assertEqual(root.children[0], ("hbox", (sentinel,)))
        self.assertIs(viewer.fullscreen_toggle, sentinel)
        attach.assert_called_once_with(sentinel, viewer, side_width_px=0)


class CanvasFitTests(unittest.TestCase):
    def test_fills_the_space_left_by_the_panels(self):
        w, h = fit_canvas_inches([1920, 1080], side_width_px=576, css_dpi=100)
        self.assertAlmostEqual(w, (1920 - LEFT_PANEL_PX - 576 - CHROME_WIDTH_PX) / 100)
        self.assertAlmostEqual(h, (1080 - CHROME_HEIGHT_PX) / 100)

    def test_runaway_viewport_is_capped(self):
        w, h = fit_canvas_inches([1414, 250000], side_width_px=0, css_dpi=100)
        self.assertAlmostEqual(h, (MAX_VIEWPORT_PX - CHROME_HEIGHT_PX) / 100)

    def test_never_below_the_minimum(self):
        self.assertEqual(
            fit_canvas_inches([500, 150], side_width_px=576, css_dpi=100),
            (MIN_CANVAS_INCHES, MIN_CANVAS_INCHES),
        )

    def test_empty_or_malformed_viewport_is_ignored(self):
        for viewport in ([], [0, 900], ["x", 1], None, [1, 2, 3]):
            self.assertIsNone(fit_canvas_inches(viewport, side_width_px=0, css_dpi=100))
        self.assertIsNone(fit_canvas_inches([1000, 800], side_width_px=0, css_dpi=0))

    def _rig(self):
        class _Fig:
            dpi = 200            # HiDPI: ipympl doubles dpi ...
            _original_dpi = 100  # ... but draws at the original CSS scale.

            def __init__(self):
                self.size = (6.0, 6.0)
                self.canvas = mock.Mock()

            def get_size_inches(self):
                return self.size

            def set_size_inches(self, w, h, forward=False):
                self.size = (w, h)

        viewer = mock.Mock()
        viewer.image_display.fig = _Fig()
        viewer.current_downsample_factor = 4
        toggle = FullscreenToggle()
        attach_canvas_fit(toggle, viewer, side_width_px=0)
        return toggle, viewer, viewer.image_display.fig

    def test_enlarges_on_viewport_and_restores_on_exit(self):
        toggle, viewer, fig = self._rig()
        toggle.mode = "expanded"
        toggle.viewport = [1414, 920]
        self.assertEqual(fig.size, fit_canvas_inches([1414, 920], side_width_px=0, css_dpi=100))
        viewer.update_display.assert_called_with(4)
        toggle.viewport = [1614, 920]   # editor resized while enlarged
        toggle.mode = "off"
        self.assertEqual(fig.size, (6.0, 6.0))

    def test_viewport_while_off_is_ignored(self):
        toggle, viewer, fig = self._rig()
        toggle.viewport = [1414, 920]
        self.assertEqual(fig.size, (6.0, 6.0))
        viewer.update_display.assert_not_called()

    def test_square_fills_the_width_left_by_the_panels(self):
        side = (1414 - LEFT_PANEL_PX - 576 - CHROME_WIDTH_PX) / 100
        self.assertEqual(fit_square_inches(1414, side_width_px=576, css_dpi=100), (side, side))
        self.assertEqual(
            fit_square_inches(700, side_width_px=576, css_dpi=100),
            (MIN_CANVAS_INCHES, MIN_CANVAS_INCHES),
        )
        for bad in (0, None, "x"):
            self.assertIsNone(fit_square_inches(bad, side_width_px=0, css_dpi=100))

    def test_default_layout_is_a_square_filling_the_width(self):
        toggle, viewer, fig = self._rig()
        toggle.host_width = 1414
        expected = fit_square_inches(1414, side_width_px=0, css_dpi=100)
        self.assertEqual(fig.size, expected)
        self.assertEqual(fig.size[0], fig.size[1])
        viewer.update_display.assert_called_with(4)
        # The window is resized: the square follows the width.
        toggle.host_width = 1800
        self.assertEqual(fig.size, fit_square_inches(1800, side_width_px=0, css_dpi=100))

    def test_a_width_change_too_small_to_matter_does_not_refit(self):
        toggle, viewer, fig = self._rig()
        toggle.host_width = 1414
        calls = viewer.update_display.call_count
        toggle.host_width = 1415
        self.assertEqual(viewer.update_display.call_count, calls)

    def test_leaving_full_screen_returns_to_the_square(self):
        toggle, viewer, fig = self._rig()
        toggle.host_width = 1414
        square = fig.size
        toggle.mode = "native"
        toggle.viewport = [1920, 1080]
        self.assertEqual(fig.size, fit_canvas_inches([1920, 1080], side_width_px=0, css_dpi=100))
        toggle.mode = "off"
        self.assertEqual(fig.size, square)


def _minimal_viewer():
    from types import SimpleNamespace

    from ueler.viewer import ui_components

    ui = SimpleNamespace(
        control_sections=ui_components.Output(),
        annotation_editor_host=ui_components.Output(),
        image_selector=ui_components.Output(),
        map_controls_box=ui_components.Output(),
        advanced_settings_accordion=ui_components.Output(),
        status_bar=ui_components.Output(),
        lasso_select_toggle=ui_components.Output(),
    )
    return SimpleNamespace(
        cell_table=None,
        ui_component=ui,
        image_output=ui_components.Output(),
        grid_output=ui_components.Output(),
        dynamically_load_plugins=lambda **_kw: None,
        _debug=False,
    )


# --- Browser behaviour under node ------------------------------------------

_HARNESS = r"""
import { pathToFileURL } from "node:url";

const scenario = process.argv[3];

// ---- minimal DOM ----
const listeners = {};
const classSet = () => {
  const s = new Set();
  return { add: (c) => s.add(c), remove: (c) => s.delete(c), contains: (c) => s.has(c), _s: s };
};
const styleVars = {};
const host = {
  id: "host", classList: classSet(), fullscreenCalls: 0,
  style: { setProperty: (k, v) => { styleVars[k] = v; }, removeProperty: (k) => { delete styleVars[k]; } },
  getBoundingClientRect: () => ({ width: hostWidth, height: 5000, top: 215 }),
  nodeType: 1, _overflow: "auto", scrollTop: 0, scrollHeight: 3000, clientHeight: 920,
  parentElement: null, wheelHandlers: new Set(), kids: [],
  appendChild(c) { c.parentNode = this; this.kids.push(c); },
  removeChild(c) { this.kids = this.kids.filter((k) => k !== c); c.parentNode = null; },
  id: "",
  removeAttribute(name) { if (name === "id") this.id = ""; },
  addEventListener(t, fn) { if (t === "wheel") this.wheelHandlers.add(fn); },
  removeEventListener(t, fn) { if (t === "wheel") this.wheelHandlers.delete(fn); },
};
// IntersectionObserver stand-in: the test decides what is "on screen".
const observers = new Set();
globalThis.IntersectionObserver = class {
  constructor(cb, options) { this.cb = cb; this.options = options; this.targets = []; }
  observe(t) { this.targets.push(t); observers.add(this); }
  disconnect() { observers.delete(this); }
};
// ResizeObserver stand-in: the test fires it after changing the host's width.
const resizeObservers = new Set();
globalThis.ResizeObserver = class {
  constructor(cb) { this.cb = cb; }
  observe(t) { this.target = t; resizeObservers.add(this); }
  disconnect() { resizeObservers.delete(this); }
};
let hostWidth = 1414.4;
const resizeHost = (w) => { hostWidth = w; for (const ro of resizeObservers) ro.cb([]); };
globalThis.getComputedStyle = (node) => ({ overflowY: node._overflow ?? "visible" });
// A 900px VS Code editor ("vscode" scenario only); ``y`` is the viewer's top
// relative to the editor's top, in px.
const editor = { E: 900, y: 200, snaps: 0, hrefs: [] };
const viewerHeight = () => parseFloat(styleVars["--ueler-expanded-height"]);
const report = () => {
  const h = viewerHeight();
  const visible = Math.max(0, Math.min(editor.E, editor.y + h) - Math.max(0, editor.y));
  for (const io of observers) {
    if (!io.targets.includes(host)) continue;
    io.cb([{ target: host, isIntersecting: visible > 0, intersectionRatio: visible / h }]);
  }
};
// VS Code's handling of an in-page link click: scroll the target to the top.
const onLinkClick = (href) => {
  editor.hrefs.push(href);
  if (scenario !== "vscode" || href !== "#" + host.id) return;
  editor.snaps += 1;
  editor.y = 0;
  report();
};
host.classList.add("ueler-root");
const wrapper = { id: "wrapper", classList: classSet() };  // the anywidget's own box
const el = {
  children: [],
  appendChild(c) { this.children.push(c); },
  closest(sel) { return sel === ".ueler-root" ? (scenario === "no-host" ? null : host) : null; },
  parentElement: wrapper,
};
let resizes = 0;
let exitCalls = 0;
globalThis.Event = class { constructor(t) { this.type = t; } };
globalThis.requestAnimationFrame = (fn) => fn();
const windowListeners = {};
globalThis.window = {
  // VS Code sizes the webview to the notebook content, so innerHeight can be
  // far larger than the screen ("vscode-runaway").
  innerHeight: scenario === "vscode-runaway" ? 20000 : 920,
  screen: { availHeight: 1080 },
  dispatchEvent: () => { resizes += 1; },
  addEventListener(t, fn) { (windowListeners[t] ||= new Set()).add(fn); },
  removeEventListener(t, fn) { windowListeners[t]?.delete(fn); },
};
const bodyClasses = scenario.startsWith("vscode") ? ["vscode-dark"] : ["jp-LabShell"];
globalThis.document = {
  fullscreenEnabled: scenario !== "api-disabled",
  fullscreenElement: null,
  body: { classList: bodyClasses, hasAttribute: () => false },
  scrollingElement: { scrollHeight: 5000, scrollTop: 0 },
  createElement() {
    const btn = { attrs: {}, handlers: {}, innerHTML: "", title: "" };
    btn.setAttribute = (k, v) => { btn.attrs[k] = v; };
    btn.addEventListener = (t, fn) => { btn.handlers[t] = fn; };
    btn.click = () => btn.handlers.click
      ? btn.handlers.click({ preventDefault() {}, stopPropagation() {} })
      : onLinkClick(btn.href);
    btn.style = {};
    btn.remove = () => {};
    return btn;
  },
  addEventListener(t, fn) { (listeners[t] ||= new Set()).add(fn); },
  removeEventListener(t, fn) { listeners[t]?.delete(fn); },
  exitFullscreen() { exitCalls += 1; document.fullscreenElement = null; return Promise.resolve(); },
};
host.requestFullscreen = () => {
  host.fullscreenCalls += 1;
  if (scenario === "native-rejected" || scenario.startsWith("vscode")) return Promise.reject(new Error("denied"));
  document.fullscreenElement = host;
  return Promise.resolve();
};
const fire = (t, ev = {}) => { for (const fn of listeners[t] ?? []) fn({ stopPropagation() {}, ...ev }); };

// ---- minimal anywidget model ----
const state = {
  mode: "off", viewport: [], host_width: 0, host_class: "ueler-root",
  maximized_class: "ueler-maximized", expanded_class: "ueler-expanded",
};
const modelHandlers = {};
let saves = 0;
const model = {
  get: (k) => state[k],
  set(k, v) { const old = state[k]; state[k] = v; if (old !== v) for (const fn of modelHandlers["change:" + k] ?? []) fn(); },
  save_changes() { saves += 1; },
  on(evt, fn) { (modelHandlers[evt] ||= new Set()).add(fn); },
  off(evt, fn) { modelHandlers[evt]?.delete(fn); },
};

const mod = await import(pathToFileURL(process.argv[2]).href);
const cleanup = mod.default.render({ model, el });
const btn = el.children[0];
const tick = () => new Promise((r) => setTimeout(r, 0));
const snap = () => ({
  mode: state.mode,
  maximized: host.classList.contains("ueler-maximized"),
  expanded: host.classList.contains("ueler-expanded"),
  heightVar: styleVars["--ueler-expanded-height"] ?? null,
  viewport: state.viewport,
  wrapperTouched: wrapper.classList._s.size > 0,
  fullscreenElement: document.fullscreenElement === host ? "host" : (document.fullscreenElement ? "other" : null),
  fullscreenCalls: host.fullscreenCalls,
  exitCalls,
  pressed: btn.attrs["aria-pressed"],
});

const out = { initial: snap() };
btn.click(); await tick();
out.entered = snap();

if (scenario === "native-ok") {
  btn.click(); await tick();
  out.exited = snap();
  btn.click(); await tick();
  document.fullscreenElement = null;   // browser Esc
  fire("fullscreenchange"); await tick();
  out.browserEsc = snap();
} else if (scenario === "width") {
  const settle = () => new Promise((r) => setTimeout(r, 250));
  out.watching = resizeObservers.size;
  resizeHost(1414.4); await settle();
  out.first = state.host_width;
  resizeHost(1800); await settle();
  out.wider = state.host_width;
  const saved = saves;
  resizeHost(1801); await settle();        // under 2px: not reported
  out.tiny = state.host_width;
  out.tinySaves = saves - saved;
  resizeHost(0); await settle();            // hidden: not reported
  out.hidden = state.host_width;
  cleanup();
  out.observersLeft = resizeObservers.size;
} else if (scenario === "vscode-runaway") {
  // The viewer grew, so the webview grew: this must not feed back.
  window.innerHeight += 5000;
  for (const fn of windowListeners.resize ?? []) fn();
  await new Promise((r) => setTimeout(r, 300));
  out.afterGrowth = snap();
} else if (scenario === "vscode") {
  await new Promise((r) => setTimeout(r, 300));
  out.linkClicks = editor.hrefs.length;
  out.anchorId = host.id;
  fire("keydown", { key: "Escape" }); await tick();
  out.esc = snap();

  btn.click(); await tick();
  cleanup();
  out.unmounted = snap();
  out.idAfterExit = host.id;
  out.listenersLeft = Object.values(listeners).reduce((n, s) => n + s.size, 0)
    + Object.values(windowListeners).reduce((n, s) => n + s.size, 0)
    + observers.size + resizeObservers.size;
} else if (scenario === "native-rejected" || scenario === "api-disabled") {
  fire("keydown", { key: "Escape" }); await tick();
  out.esc = snap();
  btn.click(); await tick();
  model.set("mode", "off");            // Python resets the trait
  await tick();
  out.pythonReset = snap();
  btn.click(); await tick();
  cleanup();
  out.unmounted = snap();
  out.listenersLeft = Object.values(listeners).reduce((n, s) => n + s.size, 0)
    + Object.values(windowListeners).reduce((n, s) => n + s.size, 0);
}
out.saves = saves;
out.resizes = resizes;
console.log(JSON.stringify(out));
process.exit(0);  // a probe left running must not keep node alive
"""


@unittest.skipUnless(_NODE, "node is required to execute the widget ESM")
class FullscreenToggleEsmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls._esm_path = tmp / "widget.mjs"
        cls._esm_path.write_text(_ESM, encoding="utf-8")
        cls._harness_path = tmp / "harness.mjs"
        cls._harness_path.write_text(_HARNESS, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run(self, scenario):
        result = subprocess.run(
            [_NODE, str(self._harness_path), str(self._esm_path), scenario],
            capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_native_fullscreen_targets_the_root_and_exits(self):
        out = self._run("native-ok")
        self.assertEqual(out["initial"]["pressed"], "false")
        entered = out["entered"]
        self.assertEqual(entered["mode"], "native")
        self.assertEqual(entered["fullscreenElement"], "host")
        self.assertFalse(entered["maximized"])
        self.assertFalse(entered["wrapperTouched"])
        self.assertEqual(entered["pressed"], "true")
        self.assertEqual(entered["viewport"], [1414, 5000])

        exited = out["exited"]
        self.assertEqual(exited["mode"], "off")
        self.assertIsNone(exited["fullscreenElement"])
        self.assertEqual(exited["exitCalls"], 1)
        self.assertEqual(exited["viewport"], [])

        # Leaving through the browser (Esc) resets the trait without a second exit call.
        self.assertEqual(out["browserEsc"]["mode"], "off")
        self.assertEqual(out["browserEsc"]["exitCalls"], 1)
        self.assertGreater(out["resizes"], 0)

    def test_rejected_native_request_falls_back_to_maximize(self):
        out = self._run("native-rejected")
        entered = out["entered"]
        self.assertEqual(entered["fullscreenCalls"], 1)
        self.assertEqual(entered["mode"], "maximized")
        self.assertTrue(entered["maximized"])
        self.assertFalse(entered["expanded"])
        self.assertFalse(entered["wrapperTouched"])
        self.assertEqual(out["esc"]["mode"], "off")
        self.assertFalse(out["esc"]["maximized"])

    def test_vscode_refusal_expands_in_flow_instead_of_overlaying(self):
        out = self._run("vscode")
        entered = out["entered"]
        self.assertEqual(entered["fullscreenCalls"], 1)
        self.assertEqual(entered["mode"], "expanded")
        self.assertTrue(entered["expanded"])
        self.assertFalse(entered["maximized"])
        # One editor-screen: min(innerHeight 920, 1080 available - 160).
        self.assertEqual(entered["heightVar"], "920px")
        self.assertEqual(entered["viewport"], [1414, 920])
        esc = out["esc"]
        self.assertEqual(esc["mode"], "off")
        self.assertFalse(esc["expanded"])
        self.assertIsNone(esc["heightVar"])
        self.assertEqual(esc["viewport"], [])
        self.assertFalse(out["unmounted"]["expanded"])
        self.assertEqual(out["listenersLeft"], 0)

    def test_vscode_never_clicks_links_or_tags_the_root(self):
        out = self._run("vscode")
        self.assertEqual(out["linkClicks"], 0)
        self.assertEqual(out["anchorId"], "")
        self.assertNotIn("link.click", _ESM)
        self.assertNotIn("WheelEvent", _ESM)
        self.assertNotIn('"wheel"', _ESM)
        for call in ("scrollTo(", "scrollBy(", "scrollIntoView(", "scrollTop ="):
            self.assertNotIn(call, _ESM)

    def test_root_width_is_reported_in_every_mode(self):
        out = self._run("width")
        # Reported without entering any mode -- this is the default layout.
        self.assertEqual(out["initial"]["mode"], "off")
        self.assertEqual(out["watching"], 1)
        self.assertEqual(out["first"], 1414)
        self.assertEqual(out["wider"], 1800)
        self.assertEqual(out["tiny"], 1800)
        self.assertEqual(out["tinySaves"], 0)
        self.assertEqual(out["hidden"], 1800)
        self.assertEqual(out["observersLeft"], 0)

    def test_vscode_height_is_bounded_by_screen_and_does_not_feed_back(self):
        out = self._run("vscode-runaway")
        entered = out["entered"]
        self.assertEqual(entered["mode"], "expanded")
        # 1080 available - 160 VS Code chrome, not the 20000px webview.
        self.assertEqual(entered["heightVar"], "920px")
        self.assertEqual(entered["viewport"], [1414, 920])
        after = out["afterGrowth"]
        self.assertEqual(after["heightVar"], "920px")
        self.assertEqual(after["viewport"], [1414, 920])

    def test_missing_api_maximizes_and_python_reset_and_unmount_clean_up(self):
        out = self._run("api-disabled")
        self.assertEqual(out["entered"]["fullscreenCalls"], 0)
        self.assertTrue(out["entered"]["maximized"])
        self.assertEqual(out["pythonReset"]["mode"], "off")
        self.assertFalse(out["pythonReset"]["maximized"])
        self.assertFalse(out["unmounted"]["maximized"])
        self.assertEqual(out["listenersLeft"], 0)

    def test_without_host_nothing_happens(self):
        out = self._run("no-host")
        self.assertEqual(out["entered"]["mode"], "off")
        self.assertEqual(out["entered"]["fullscreenCalls"], 0)
        self.assertFalse(out["entered"]["maximized"])
        self.assertEqual(out["saves"], 0)


if __name__ == "__main__":
    unittest.main()
