"""Measure how long the viewer takes to respond, on the device you use it on.

The benchmark harness (``tools/benchmark``) times the kernel only. What a user
feels also includes sending each frame to the browser and the browser decoding
it, which depend on the network and the client device. ``LatencyTimer`` times
both ends of a live session:

* **Browser side** (a small script in the cell's output): from each input
  (press, drag, release, scroll, a widget change) until the resulting frame of
  the main canvas has been decoded, plus each frame's size and decode time.
  This is the delay the user sees.
* **Kernel side**: from each input message arriving in the kernel until the
  frame leaves it, plus the render and PNG-encode time of each frame.

The difference between the two is network transfer plus the browser's work.

Usage in a notebook::

    from ueler.viewer.latency_timer import LatencyTimer
    timer = LatencyTimer(viewer)   # shows the browser panel below the cell
    # ... pan, zoom, switch FOV, add a channel ...
    timer.report()                 # kernel-side summary (Markdown)
    timer.stop()                   # remove the hooks

Inputs are grouped into interactions: a press-to-release drag, a scroll, or a
widget change, closed after ``quiet_s`` seconds without input or frame. Mouse
movement without a pressed button (hover) is not an input.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from typing import Dict, List, Optional

__all__ = ["LatencyTimer", "summarise_interactions"]

QUIET_S = 1.5


def _median(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 1) if values else None


def summarise_interactions(events: List[Dict], quiet_s: float = QUIET_S) -> List[Dict]:
    """Group a time-ordered event log into interactions and time each one.

    Each event is ``{"t": seconds, "type": "input" | "frame", ...}``. Inputs
    carry ``kind`` (``press``, ``drag``, ``release``, ``scroll`` or
    ``widget``); frames carry ``kb`` and may carry ``render_ms`` and
    ``encode_ms``. Frames outside any interaction (e.g. a tooltip redraw) are
    ignored. The browser panel implements the same rules in JavaScript.
    """
    groups: List[List[Dict]] = []
    current: List[Dict] = []
    last_t = None
    for ev in sorted(events, key=lambda e: e["t"]):
        if last_t is not None and ev["t"] - last_t > quiet_s and current:
            groups.append(current)
            current = []
        if not current and ev["type"] != "input":
            continue  # a frame nobody asked for
        current.append(ev)
        last_t = ev["t"]
    if current:
        groups.append(current)
    return [_summarise_group(g) for g in groups]


def _summarise_group(group: List[Dict]) -> Dict:
    inputs = [e for e in group if e["type"] == "input"]
    frames = [e for e in group if e["type"] == "frame"]
    kinds = [e["kind"] for e in inputs]
    t0 = inputs[0]["t"]
    out: Dict = {
        "frames": len(frames),
        "kb": round(sum(f.get("kb", 0) for f in frames), 1),
        "render_ms": _median(f.get("render_ms") for f in frames),
        "encode_ms": _median(f.get("encode_ms") for f in frames),
        "decode_ms": _median(f.get("decode_ms") for f in frames),
    }
    if "press" in kinds and "release" in kinds:
        release_t = next(e["t"] for e in inputs if e["kind"] == "release")
        steps = [f for f in frames if f["t"] <= release_t]
        after = [f for f in frames if f["t"] > release_t]
        lags = []
        answered = t0
        for f in steps:
            # Oldest drag input not yet answered by an earlier frame.
            pending = [e["t"] for e in inputs if e["kind"] == "drag" and answered <= e["t"] <= f["t"]]
            if pending:
                lags.append((f["t"] - pending[0]) * 1000)
            answered = f["t"]
        intervals = [(b["t"] - a["t"]) * 1000 for a, b in zip(steps, steps[1:])]
        out.update(
            kind="drag",
            drag_frames=len(steps),
            drag_frame_interval_ms=_median(intervals),
            drag_lag_ms=_median(lags),
            drag_frame_kb=_median(f.get("kb") for f in steps),
            release_to_frame_ms=round((after[-1]["t"] - release_t) * 1000, 1) if after else None,
            release_frame_kb=after[-1].get("kb") if after else None,
        )
    else:
        out.update(
            kind="scroll" if kinds[0] == "scroll" else ("widget" if kinds[0] == "widget" else "click"),
            first_frame_ms=round((frames[0]["t"] - t0) * 1000, 1) if frames else None,
            last_frame_ms=round((frames[-1]["t"] - t0) * 1000, 1) if frames else None,
        )
    return out


_ROWS = [
    # (label, kind, key, unit)
    ("Drag: frame interval while dragging", "drag", "drag_frame_interval_ms", "ms"),
    ("Drag: input → frame while dragging", "drag", "drag_lag_ms", "ms"),
    ("Drag: frame size while dragging", "drag", "drag_frame_kb", "KB"),
    ("Drag: release → last frame", "drag", "release_to_frame_ms", "ms"),
    ("Drag: last frame size", "drag", "release_frame_kb", "KB"),
    ("Widget change: input → first frame", "widget", "first_frame_ms", "ms"),
    ("Widget change: input → last frame", "widget", "last_frame_ms", "ms"),
    ("Widget change: KB sent", "widget", "kb", "KB"),
    ("Scroll: input → last frame", "scroll", "last_frame_ms", "ms"),
    ("Render per frame", None, "render_ms", "ms"),
    ("PNG encode per frame", None, "encode_ms", "ms"),
]


def format_summary(interactions: List[Dict], header: str) -> str:
    """Markdown table of medians per interaction kind."""
    counts = {}
    for it in interactions:
        counts[it["kind"]] = counts.get(it["kind"], 0) + 1
    lines = [header, "", "Interactions: " + (", ".join(f"{k} × {n}" for k, n in sorted(counts.items())) or "none"), "",
             "| Measure | Median |", "| --- | --- |"]
    for label, kind, key, unit in _ROWS:
        value = _median(it.get(key) for it in interactions if kind is None or it["kind"] == kind)
        if value is not None:
            lines.append(f"| {label} | {value:g} {unit} |")
    return "\n".join(lines)


class LatencyTimer:
    """Hook a live viewer and time its interactions; see the module docstring."""

    def __init__(self, viewer, *, show_browser_panel: bool = True, quiet_s: float = QUIET_S):
        self.viewer = viewer
        self.canvas = viewer.image_display.fig.canvas
        self.quiet_s = quiet_s
        self.events: List[Dict] = []
        self._restore = []
        self._frame_render_ms = 0.0
        self._frame_encode_ms = 0.0
        self.started = time.time()
        self._install()
        if show_browser_panel:
            self.show_browser_panel()

    # -- hooks ------------------------------------------------------------
    def _wrap_instance(self, obj, name, make):
        original = getattr(obj, name)
        had_own = name in vars(obj)
        setattr(obj, name, make(original))
        self._restore.append((obj, name, original if had_own else None))

    def _install(self):
        timer = self
        canvas = self.canvas

        def timed(original, attr):
            def wrapper(*args, **kwargs):
                t = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    setattr(timer, attr, getattr(timer, attr) + (time.perf_counter() - t) * 1000)
            return wrapper

        self._wrap_instance(self.viewer, "update_display", lambda f: timed(f, "_frame_render_ms"))
        self._wrap_instance(canvas, "get_diff_image", lambda f: timed(f, "_frame_encode_ms"))

        def on_send_binary(original):
            def wrapper(data, *args, **kwargs):
                try:
                    timer.events.append({
                        "t": time.perf_counter(), "type": "frame", "kb": round(len(data) / 1024, 1),
                        "render_ms": round(timer._frame_render_ms, 1), "encode_ms": round(timer._frame_encode_ms, 1),
                    })
                finally:
                    timer._frame_render_ms = timer._frame_encode_ms = 0.0
                return original(data, *args, **kwargs)
            return wrapper

        self._wrap_instance(canvas, "send_binary", on_send_binary)

        def on_handle_event(original):
            def wrapper(event, *args, **kwargs):
                kind = None
                etype = event.get("type") if isinstance(event, dict) else None
                if etype == "button_press":
                    kind = "press"
                elif etype == "button_release":
                    kind = "release"
                elif etype == "motion_notify" and event.get("buttons"):
                    kind = "drag"
                elif etype == "scroll":
                    kind = "scroll"
                if kind:
                    timer.events.append({"t": time.perf_counter(), "type": "input", "kind": kind})
                return original(event, *args, **kwargs)
            return wrapper

        self._wrap_instance(canvas, "handle_event", on_handle_event)

        # Widget changes (FOV, channels, sliders) arrive as comm messages to
        # other widgets; hook the comm class, which dispatches every message.
        comm = getattr(canvas, "comm", None)
        comm_cls = None
        for cls in type(comm).__mro__ if comm is not None else ():
            if "handle_msg" in vars(cls):
                comm_cls = cls
                break
        if comm_cls is not None:
            original = comm_cls.handle_msg

            def handle_msg(comm_self, msg, *args, **kwargs):
                try:
                    data = msg.get("content", {}).get("data", {})
                    if comm_self is not getattr(canvas, "comm", None) and data.get("method") == "update":
                        timer.events.append({"t": time.perf_counter(), "type": "input", "kind": "widget"})
                except Exception:
                    pass
                return original(comm_self, msg, *args, **kwargs)

            comm_cls.handle_msg = handle_msg
            self._restore.append((comm_cls, "handle_msg", original))

    def stop(self):
        """Remove every hook; the event log is kept for ``report``."""
        for obj, name, original in reversed(self._restore):
            if original is None:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
            else:
                setattr(obj, name, original)
        self._restore = []

    # -- results ----------------------------------------------------------
    def environment(self) -> Dict:
        import matplotlib

        try:
            from ueler import __version__ as ueler_version
        except Exception:  # pragma: no cover
            ueler_version = "?"
        try:
            device_px = list(self.canvas.get_width_height(physical=True))
            css_px = list(self.canvas.get_width_height())
        except Exception:
            device_px = css_px = None
        return {
            "kernel_platform": platform.platform(),
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "ueler": ueler_version,
            "canvas_css_px": css_px,
            "canvas_device_px": device_px,
            "device_pixel_ratio": getattr(self.canvas, "device_pixel_ratio", None),
            "drag_preview_px": getattr(self.viewer, "drag_preview_px", None),
        }

    def interactions(self) -> List[Dict]:
        return summarise_interactions(self.events, self.quiet_s)

    def report(self, print_it: bool = True) -> str:
        env = self.environment()
        text = format_summary(self.interactions(), "### UELer latency — kernel side")
        text += "\n\n" + ", ".join(f"{k}: {v}" for k, v in env.items())
        if print_it:
            print(text)
        return text

    # -- browser ----------------------------------------------------------
    def show_browser_panel(self):
        from IPython.display import HTML, display

        try:
            w, h = self.canvas.get_width_height(physical=True)
        except Exception:
            w = h = 0
        config = {"mainCanvasDevicePx": [int(w), int(h)], "quietMs": int(self.quiet_s * 1000)}
        panel_id = f"ueler-latency-{int(time.time() * 1000)}"
        display(HTML(_BROWSER_PANEL.replace("__PANEL_ID__", panel_id).replace("__CONFIG__", json.dumps(config))))


_BROWSER_PANEL = r"""
<div id="__PANEL_ID__" style="font: 12px/1.4 sans-serif; border: 1px solid #8884; padding: 6px; max-width: 640px">
  <b>UELer latency — browser side</b> <span class="status">(waiting for input; if this never changes, scripts in outputs are disabled here)</span>
  <div class="table"></div>
  <textarea class="copy" rows="6" style="width: 100%; font: 11px monospace; margin-top: 4px" readonly></textarea>
</div>
<script>
(function () {
  const config = __CONFIG__;
  const root = document.getElementById("__PANEL_ID__");
  const T = window.__uelerLatency || (window.__uelerLatency = { events: [], pending: new Map(), panels: [] });
  T.config = config;
  T.panels.push(root);

  if (!T.installed) {
    T.installed = true;
    const now = () => performance.now();
    // Frames: ipympl turns each PNG into a blob URL and assigns it to an Image.
    const URLs = window.URL || window.webkitURL;
    const create = URLs.createObjectURL.bind(URLs);
    URLs.createObjectURL = function (obj) {
      const url = create(obj);
      try {
        if (obj instanceof Blob && obj.type === "image/png") T.pending.set(url, { arrive: now(), kb: obj.size / 1024 });
      } catch (e) {}
      return url;
    };
    const desc = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, "src");
    Object.defineProperty(HTMLImageElement.prototype, "src", {
      configurable: true, enumerable: desc.enumerable,
      get() { return desc.get.call(this); },
      set(value) {
        const p = T.pending.get(value);
        if (p) {
          T.pending.delete(value);
          this.addEventListener("load", () => {
            const [w, h] = T.config.mainCanvasDevicePx || [0, 0];
            // Only frames of the main viewer canvas (other figures are smaller).
            if (w && h && this.naturalWidth * this.naturalHeight < 0.5 * w * h) return;
            const t = now();
            T.events.push({ t: t / 1000, type: "frame", kb: Math.round(p.kb * 10) / 10, decode_ms: Math.round((t - p.arrive) * 10) / 10 });
            T.schedule();
          }, { once: true });
        }
        desc.set.call(this, value);
      },
    });
    // Inputs, timed when the browser received them.
    const onCanvas = (el) => el && el.closest && el.closest(".jupyter-matplotlib-figure");
    const onWidget = (el) => el && el.closest && el.closest(".jupyter-widgets, .widget-box, .lm-Widget, .p-Widget");
    let pressed = false;
    const add = (e, kind) => { T.events.push({ t: e.timeStamp / 1000, type: "input", kind }); T.schedule(); };
    document.addEventListener("pointerdown", (e) => { if (onCanvas(e.target)) { pressed = true; add(e, "press"); } }, true);
    document.addEventListener("pointermove", (e) => { if (pressed && e.buttons) add(e, "drag"); }, true);
    document.addEventListener("pointerup", (e) => {
      if (pressed) { pressed = false; add(e, "release"); }
      else if (onWidget(e.target) && !onCanvas(e.target)) add(e, "widget");
    }, true);
    document.addEventListener("wheel", (e) => { if (onCanvas(e.target)) add(e, "scroll"); }, { capture: true, passive: true });
    document.addEventListener("change", (e) => { if (onWidget(e.target) && !onCanvas(e.target)) add(e, "widget"); }, true);
    document.addEventListener("keydown", (e) => { if (e.key === "Enter" && onWidget(e.target) && !onCanvas(e.target)) add(e, "widget"); }, true);

    // Same grouping rules as ueler.viewer.latency_timer.summarise_interactions.
    const median = (xs) => {
      xs = xs.filter((x) => x !== null && x !== undefined && !Number.isNaN(x)).sort((a, b) => a - b);
      if (!xs.length) return null;
      const m = xs.length >> 1;
      return Math.round((xs.length % 2 ? xs[m] : (xs[m - 1] + xs[m]) / 2) * 10) / 10;
    };
    T.summarise = function () {
      const quiet = T.config.quietMs / 1000;
      const evs = T.events.slice().sort((a, b) => a.t - b.t);
      const groups = []; let cur = []; let last = null;
      for (const ev of evs) {
        if (last !== null && ev.t - last > quiet && cur.length) { groups.push(cur); cur = []; }
        if (!cur.length && ev.type !== "input") continue;
        cur.push(ev); last = ev.t;
      }
      if (cur.length) groups.push(cur);
      return groups.map((g) => {
        const inputs = g.filter((e) => e.type === "input"), frames = g.filter((e) => e.type === "frame");
        const kinds = inputs.map((e) => e.kind), t0 = inputs[0].t;
        const out = { frames: frames.length, kb: Math.round(frames.reduce((s, f) => s + f.kb, 0) * 10) / 10, decode_ms: median(frames.map((f) => f.decode_ms)) };
        if (kinds.includes("press") && kinds.includes("release")) {
          const rel = inputs.find((e) => e.kind === "release").t;
          const steps = frames.filter((f) => f.t <= rel), after = frames.filter((f) => f.t > rel);
          const lags = []; let answered = t0;
          for (const f of steps) {
            const p = inputs.filter((e) => e.kind === "drag" && e.t >= answered && e.t <= f.t);
            if (p.length) lags.push((f.t - p[0].t) * 1000);
            answered = f.t;
          }
          const iv = []; for (let i = 1; i < steps.length; i++) iv.push((steps[i].t - steps[i - 1].t) * 1000);
          Object.assign(out, { kind: "drag", drag_frames: steps.length, drag_frame_interval_ms: median(iv), drag_lag_ms: median(lags),
            drag_frame_kb: median(steps.map((f) => f.kb)),
            release_to_frame_ms: after.length ? Math.round((after[after.length - 1].t - rel) * 10000) / 10 : null,
            release_frame_kb: after.length ? after[after.length - 1].kb : null });
        } else {
          Object.assign(out, { kind: kinds[0] === "scroll" ? "scroll" : (kinds[0] === "widget" ? "widget" : "click"),
            first_frame_ms: frames.length ? Math.round((frames[0].t - t0) * 10000) / 10 : null,
            last_frame_ms: frames.length ? Math.round((frames[frames.length - 1].t - t0) * 10000) / 10 : null });
        }
        return out;
      });
    };
    const ROWS = [
      ["Drag: frame interval while dragging", "drag", "drag_frame_interval_ms", "ms"],
      ["Drag: input → frame on screen while dragging", "drag", "drag_lag_ms", "ms"],
      ["Drag: frame size while dragging", "drag", "drag_frame_kb", "KB"],
      ["Drag: release → last frame on screen", "drag", "release_to_frame_ms", "ms"],
      ["Drag: last frame size", "drag", "release_frame_kb", "KB"],
      ["Widget change: input → first frame on screen", "widget", "first_frame_ms", "ms"],
      ["Widget change: input → last frame on screen", "widget", "last_frame_ms", "ms"],
      ["Widget change: KB received", "widget", "kb", "KB"],
      ["Scroll: input → last frame on screen", "scroll", "last_frame_ms", "ms"],
      ["Browser decode per frame", null, "decode_ms", "ms"],
    ];
    T.report = function () {
      const its = T.summarise();
      const counts = {}; its.forEach((i) => (counts[i.kind] = (counts[i.kind] || 0) + 1));
      const lines = ["### UELer latency — browser side", "",
        "Interactions: " + (Object.keys(counts).sort().map((k) => k + " × " + counts[k]).join(", ") || "none"), "",
        "| Measure | Median |", "| --- | --- |"];
      for (const [label, kind, key, unit] of ROWS) {
        const v = median(its.filter((i) => kind === null || i.kind === kind).map((i) => i[key]));
        if (v !== null) lines.push("| " + label + " | " + v + " " + unit + " |");
      }
      lines.push("", "devicePixelRatio: " + window.devicePixelRatio + ", main canvas device px: " + JSON.stringify(T.config.mainCanvasDevicePx) + ", browser: " + navigator.userAgent);
      return lines.join("\n");
    };
    let timer = null;
    T.schedule = function () {
      if (timer) return;
      timer = setTimeout(() => {
        timer = null;
        const text = T.report();
        for (const panel of T.panels) {
          if (!panel.isConnected) continue;
          const rows = text.split("\n").filter((l) => l.startsWith("| ") && !l.startsWith("| Measure") && !l.startsWith("| ---"));
          panel.querySelector(".table").innerHTML = "<table>" + rows.map((r) => "<tr>" + r.split("|").slice(1, -1).map((c) => "<td style='padding:0 8px'>" + c.trim() + "</td>").join("") + "</tr>").join("") + "</table>";
          panel.querySelector(".status").textContent = "(" + T.events.filter((e) => e.type === "input").length + " inputs, " + T.events.filter((e) => e.type === "frame").length + " frames recorded)";
          panel.querySelector(".copy").value = text;
        }
      }, 300);
    };
  }
  T.schedule();
})();
</script>
"""
