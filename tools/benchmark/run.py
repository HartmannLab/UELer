"""Benchmark UELer's interactive render path, headless.

Builds the real ``ImageMaskViewer`` from this repository (not an installed copy)
on an Agg-backed ipympl canvas, then drives it through the same message path a
browser uses: mouse events go in through ``canvas.handle_event`` and the browser
side is replaced by a model of ipympl's ``mpl_widget.js``:

* a ``draw`` request from the kernel is answered with a ``draw`` message only if
  the browser is not already waiting for an image (``waiting_for_image``);
  otherwise it is dropped;
* receiving a PNG clears the waiting flag and becomes the frame on screen.

That model makes two things measurable that wall time alone hides: how many
renders and PNG frames one interaction costs, and whether the frame left on
screen when the interaction ends shows the latest render (``stale``).

Not measured: network round-trip time, browser-side PNG decode and paint, and
widget rendering in the front end. Run on the target machine to include its
storage; network latency adds to every frame on remote setups.

Usage::

    python tools/benchmark/synth.py /tmp/ueler_bench/small --fovs 4 --size 1024 --channels 5
    python tools/benchmark/run.py --data /tmp/ueler_bench/small --out results.json
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# ``--ueler-src`` (see ``main``) can put another checkout in front of this one,
# so the same harness measures an older version (``compare.py`` does that).
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Stage timing
# ---------------------------------------------------------------------------
class Stages:
    """Wall time per instrumented function.

    Stages nest — matplotlib fires ``draw_event`` inside the Agg draw, so a
    re-render triggered from ``on_draw`` runs *within* ``agg_draw``. Each stage
    therefore reports ``ms`` (inclusive) and ``self_ms`` (excluding time spent in
    nested instrumented stages). ``self_ms`` values add up to the instrumented
    total without double counting.
    """

    def __init__(self):
        self.totals = collections.defaultdict(float)
        self.self_totals = collections.defaultdict(float)
        self.counts = collections.Counter()
        self._stack = []  # child time accumulated per open frame
        self._undo = []

    def wrap(self, owner, attr, name):
        original = getattr(owner, attr)
        stages = self

        def timed(*args, **kwargs):
            stages._stack.append(0.0)
            t = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t
                children = stages._stack.pop()
                if stages._stack:
                    stages._stack[-1] += elapsed
                stages.totals[name] += elapsed
                stages.self_totals[name] += elapsed - children
                stages.counts[name] += 1

        setattr(owner, attr, timed)
        self._undo.append((owner, attr, original))

    def reset(self):
        self.totals.clear()
        self.self_totals.clear()
        self.counts.clear()

    def snapshot(self):
        order = sorted(self.self_totals, key=lambda k: -self.self_totals[k])
        return {k: {"self_ms": round(self.self_totals[k] * 1000, 2), "ms": round(self.totals[k] * 1000, 2), "calls": self.counts[k]} for k in order}


class Browser:
    """Model of ipympl's mpl_widget.js draw/image handshake."""

    def __init__(self):
        self.waiting = False
        self.outbox = collections.deque()
        self.on_screen = None
        self.reset()

    def reset(self):
        self.pngs = 0
        self.png_bytes = 0
        self.dropped = 0
        self.stale_pngs = 0
        self.stale_png_bytes = 0


class Session:
    def __init__(self, data, masks=None, settings=None, interactive=True, dpr=1.0, canvas_px=None):
        import matplotlib

        # ``%matplotlib widget`` turns interactive mode on; it decides whether
        # the post-message hook (see ``post_execute``) redraws stale figures.
        matplotlib.interactive(interactive)
        from ueler.viewer.decorators import update_status_bar  # noqa: F401  (import before patching)
        import ueler.rendering.engine as engine
        import ueler.viewer.decorators as decorators
        from matplotlib.backends import backend_agg
        from ueler.runner import run_viewer

        self.stages = Stages()
        st = self.stages
        for attr in ("_slice_and_materialise", "_slice_downsampled"):
            st.wrap(engine, attr, "read_channels")
        st.wrap(engine, "_composite_channels", "composite_channels")
        st.wrap(engine, "_apply_mask_overlays", "mask_overlay")
        st.wrap(engine, "_apply_annotation_overlay", "annotation_overlay")
        st.wrap(engine, "render_fov_to_array", "render_fov")
        st.wrap(decorators, "_set_status_bar", "status_bar")
        st.wrap(backend_agg.FigureCanvasAgg, "draw", "agg_draw")

        t = time.perf_counter()
        self.viewer = v = run_viewer(data, masks_folder=masks, settings_path=settings, auto_display=False)
        self.construct_ms = (time.perf_counter() - t) * 1000
        v._widget_displayed = True  # display_ui() would set this

        for attr, name in (("update_display", "update_display"), ("load_fov", "load_fov"), ("inform_plugins", "plugin_hooks"), ("update_scale_bar", "scale_bar")):
            st.wrap(v, attr, name)
        st.wrap(v.image_display, "update_patches", "selection_outlines")
        canvas = self.canvas = v.image_display.fig.canvas
        st.wrap(canvas, "get_diff_image", "png_encode")

        self.version = 0
        self.drawn_version = None
        self.frames = []  # (send time, image version rasterised, bytes) per PNG
        self.inputs = []  # per-input latency records, see ``_timed``
        img = v.image_display.img_display
        set_data = img.set_data

        def counting_set_data(a):
            self.version += 1
            return set_data(a)

        img.set_data = counting_set_data

        b = self.browser = Browser()

        def send_json(content):
            if content.get("type") == "draw":
                if b.waiting:
                    b.dropped += 1
                else:
                    b.waiting = True
                    b.outbox.append({"type": "draw"})

        def send_binary(data):
            self.frames.append((time.perf_counter(), self.drawn_version, len(data)))
            b.waiting = False
            b.pngs += 1
            b.png_bytes += len(data)
            b.on_screen = self.drawn_version
            # The Agg buffer was rasterised before on_draw installed a new
            # render: this frame shows an outdated image.
            if self.drawn_version != self.version:
                b.stale_pngs += 1
                b.stale_png_bytes += len(data)

        canvas.send_json = send_json
        canvas.send_binary = send_binary
        # Record which image version is rasterised at the point Agg actually
        # renders the figure (``Figure.draw``), not when ``canvas.draw`` is
        # entered: the viewer may render before rasterising (pre-draw hook) or
        # after it (``draw_event``), and only the former reaches this frame.
        fig = canvas.figure
        fig_draw = fig.draw

        def marking_fig_draw(renderer, *args, **kwargs):
            self.drawn_version = self.version
            return fig_draw(renderer, *args, **kwargs)

        fig.draw = marking_fig_draw

        # There is no event loop here, so matplotlib timers would never fire.
        # Record them instead and fire them explicitly after an interaction
        # (see ``measure``) — e.g. the viewer's 300 ms hover timer, which
        # triggers a full redraw each time it fires.
        self.timers = []

        class FakeTimer:
            def __init__(timer, interval=None, callbacks=None):
                timer.interval = interval or 0
                timer.single_shot = False
                timer.callbacks = list(callbacks or [])
                timer.active = False

            def add_callback(timer, func, *args, **kwargs):
                timer.callbacks.append((func, args, kwargs))
                return func

            def start(timer, interval=None):
                if interval is not None:
                    timer.interval = interval
                timer.active = True
                self.timers.append(timer)

            def stop(timer):
                timer.active = False

            def fire(timer):
                timer.active = False
                for func, args, kwargs in timer.callbacks:
                    func(*args, **kwargs)

        canvas.new_timer = lambda *a, **k: FakeTimer(*a, **k)

        # Display: CSS size of the square canvas and the browser's device pixel
        # ratio (2 on a Retina screen). ipympl renders at CSS size × ratio, so
        # the ratio sets the Agg draw, PNG encode and transfer size.
        if canvas_px:
            fig.set_size_inches(canvas_px / fig.dpi, canvas_px / fig.dpi)
        if dpr and dpr != 1:
            canvas.handle_event({"type": "set_device_pixel_ratio", "device_pixel_ratio": dpr})
        b.outbox.clear()
        self.display = {"canvas_css_px": list(canvas.get_width_height()), "device_pixel_ratio": canvas.device_pixel_ratio, "canvas_device_px": list(canvas.get_width_height(physical=True))}
        t = time.perf_counter()
        b.waiting = True  # the browser asks for the first frame like any other
        self.comm({"type": "draw"})
        self.pump()
        self.first_frame_ms = (time.perf_counter() - t) * 1000

    # -- kernel / browser loop ---------------------------------------------
    @staticmethod
    def post_execute():
        """What ipykernel does after *every* comm message, not only cell runs.

        ipykernel triggers IPython's ``post_execute`` event around comm-message
        handlers, and ``%matplotlib widget`` hooks matplotlib's
        ``_draw_all_if_interactive`` onto it: every figure left stale gets a
        ``draw_idle()``. This is the request that, in a real session, arrives
        right after the PNG and delivers a render installed during the draw.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        if matplotlib.is_interactive():
            plt.draw_all()

    def comm(self, message):
        """Deliver one browser message to the kernel, as ipykernel would."""
        self.canvas.handle_event(message)
        self.post_execute()

    def pump(self):
        b = self.browser
        while b.outbox:
            self.comm(b.outbox.popleft())

    def _timed(self, kind, deliver):
        """Deliver one user input and time it until the correct image is sent.

        *Latency* is the kernel time from the input until the PNG that shows the
        latest render goes out; frames sent before it show an outdated image.
        A real browser adds, per frame before the correct one, its decode and
        one round trip (the next draw request), and per frame its transfer.
        """
        n0 = len(self.frames)
        t0 = time.perf_counter()
        deliver()
        sent = self.frames[n0:]
        rec = {"kind": kind, "latency_ms": None, "frames_until_correct": None, "kb_until_correct": None, "frames": len(sent)}
        for i, (t, drawn, size) in enumerate(sent):
            if drawn == self.version:
                rec["latency_ms"] = round((t - t0) * 1000, 2)
                rec["frames_until_correct"] = i + 1
                rec["kb_until_correct"] = round(sum(x[2] for x in sent[: i + 1]) / 1024, 1)
                break
        self.inputs.append(rec)
        return rec

    def _mouse(self, kind, x, y, buttons):
        h = self.canvas.get_renderer().height

        def deliver():
            self.comm({"type": kind, "x": x, "y": h - y, "button": 0, "buttons": buttons, "modifiers": [], "guiEvent": None})
            self.pump()

        return self._timed(kind, deliver)

    @contextlib.contextmanager
    def measure(self, record):
        self.stages.reset()
        self.browser.reset()
        self.inputs = []
        t = time.perf_counter()
        yield
        record["wall_ms"] = round((time.perf_counter() - t) * 1000, 2)
        record["inputs"] = list(self.inputs)
        summarise_latency(record)
        # A render is one update_display (re-read + composite); set_data is
        # called more often (the selection repaint re-installs the array).
        record["renders"] = self.stages.counts["update_display"]
        record["png_frames"] = self.browser.pngs
        record["png_kb"] = round(self.browser.png_bytes / 1024, 1)
        record["dropped_draw_requests"] = self.browser.dropped
        record["stale_png_frames"] = self.browser.stale_pngs
        record["stale_png_kb"] = round(self.browser.stale_png_bytes / 1024, 1)
        record["stale_at_end"] = self.browser.on_screen != self.version
        record["stages"] = self.stages.snapshot()
        # Fire pending timers in the order a real event loop would, and note
        # which one (if any) finally puts the latest render on screen.
        record["fresh_after_timer_ms"] = None
        pending = sorted((tm for tm in self.timers if tm.active), key=lambda tm: tm.interval)
        self.timers.clear()
        for tm in pending:
            if not tm.active:
                continue
            tm.fire()
            self.pump()
            if record["stale_at_end"] and record["fresh_after_timer_ms"] is None and self.browser.on_screen == self.version:
                record["fresh_after_timer_ms"] = tm.interval

    # -- interactions -------------------------------------------------------
    def drag(self, tool, steps, dx_frac, dy_frac, start=(0.3, 0.3)):
        tb = self.canvas.toolbar
        getattr(tb, tool)()
        try:
            bb = self.viewer.image_display.ax.bbox
            x0, y0 = bb.x0 + bb.width * start[0], bb.y0 + bb.height * start[1]
            dx, dy = bb.width * dx_frac / steps, bb.height * dy_frac / steps
            rec = {}
            with self.measure(rec):
                self._mouse("button_press", x0, y0, 1)
                for i in range(1, steps + 1):
                    self._mouse("motion_notify", x0 + dx * i, y0 + dy * i, 1)
                self._mouse("button_release", x0 + dx * steps, y0 + dy * steps, 0)
            rec["motion_events"] = steps
            rec["ms_per_motion_event"] = round(rec["wall_ms"] / steps, 2)
            return rec
        finally:
            getattr(tb, tool)()  # toggle the tool off again

    def set_view(self, frac):
        """Zoom to the central *frac* of the image (1.0 = whole FOV) and settle."""
        ax = self.viewer.image_display.ax
        w, h = self.viewer.width, self.viewer.height
        cx, cy = w / 2, h / 2
        ax.set_xlim(cx - w * frac / 2, cx + w * frac / 2)
        ax.set_ylim(cy + h * frac / 2, cy - h * frac / 2)
        self.canvas.draw_idle()
        self.pump()

    def set_trait(self, widget, value):
        rec = {}

        def deliver():
            widget.value = value
            self.post_execute()  # the change arrived as a comm message
            self.pump()

        with self.measure(rec):
            self._timed("widget", deliver)
        return rec


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def rss_mb():
    import psutil

    return round(psutil.Process().memory_info().rss / 2**20, 1)


def peak_rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 2**20 if sys.platform == "darwin" else peak / 1024, 1)


def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 2) if values else None


def summarise_latency(record):
    """Per-interaction latency: the input that the user waits on.

    For a drag that is each motion step (pan) and the release (rectangle zoom,
    which only renders on release); for a widget change, the change itself.
    """
    inputs = record["inputs"]
    steps = [i for i in inputs if i["kind"] == "motion_notify"]
    release = [i for i in inputs if i["kind"] == "button_release"]
    widget = [i for i in inputs if i["kind"] == "widget"]
    target = widget or (release if not any(i["frames"] for i in steps) else steps)
    record["latency_ms"] = _median(i["latency_ms"] for i in target)
    record["frames_until_correct"] = _median(i["frames_until_correct"] for i in target)
    record["kb_until_correct"] = _median(i["kb_until_correct"] for i in target)


def median_record(records):
    out = dict(records[-1])
    for key in ("wall_ms", "ms_per_motion_event", "latency_ms", "kb_until_correct", "frames_until_correct"):
        if key in out:
            out[key] = round(statistics.median(r[key] for r in records), 2)
    out["repeats"] = len(records)
    return out


def run_scenarios(s, channels, repeats):
    v = s.viewer
    ui = v.ui_component
    res = {}
    fovs = list(ui.image_selector.options)
    picker = ui.channel_selector
    all_channels = list(getattr(picker, "allowed_tags", None) or getattr(picker, "options", ()))
    chosen = channels or all_channels

    # Add channels one by one (first time each channel is read from disk),
    # starting from an empty selection so the first add is measured too.
    s.set_trait(picker, [])
    adds = []
    for k in range(1, len(chosen) + 1):
        r = s.set_trait(picker, list(chosen[:k]))
        r["channels_after"] = k
        adds.append(r)
    res["add_channel"] = adds
    res["n_channels"] = len(chosen)

    for frac, label in ((1.0, "full_view"), (0.25, "zoomed_4x")):
        s.set_view(frac)
        res[f"pan_{label}"] = median_record([s.drag("pan", 10, 0.04 * (1 if i % 2 == 0 else -1), 0.0) for i in range(repeats)])
    s.set_view(1.0)
    res["zoom_rectangle"] = median_record([(s.drag("zoom", 5, 0.4, 0.4), s.set_view(1.0))[0] for _ in range(repeats)])

    # Every scenario below starts from the full view: pan repeats alternate
    # direction but an odd count leaves the view shifted, and a FOV switch
    # behaves differently when the view is not at home (see the zoomed case).
    masks = getattr(ui, "mask_display_controls", {}) or {}
    if masks:
        name, box = next(iter(masks.items()))
        s.set_view(1.0)
        res["mask_overlay_on"] = s.set_trait(box, True)
        res["pan_full_view_with_masks"] = median_record([s.drag("pan", 10, 0.04 * (1 if i % 2 == 0 else -1), 0.0) for i in range(repeats)])
        s.set_trait(box, False)

    if len(fovs) >= 3:
        s.set_view(1.0)
        res["fov_switch_cold"] = s.set_trait(ui.image_selector, fovs[1])
        s.set_view(0.25)
        res["fov_switch_cold_while_zoomed"] = s.set_trait(ui.image_selector, fovs[2])
        s.set_view(1.0)
        res["fov_switch_warm"] = s.set_trait(ui.image_selector, fovs[1])
    return res


def environment(data):
    import psutil

    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    def fs_type(path):
        path = os.path.realpath(path)
        best = None
        for part in psutil.disk_partitions(all=True):
            if path.startswith(part.mountpoint) and (best is None or len(part.mountpoint) > len(best.mountpoint)):
                best = part
        return None if best is None else {"mountpoint": best.mountpoint, "fstype": best.fstype}

    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "ueler"], capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, dirty = None, None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(),
        "packages": {m: ver(m) for m in ("numpy", "dask", "matplotlib", "ipympl", "ipywidgets", "tifffile", "anndata")},
        "ueler_commit": commit,
        "ueler_dirty": dirty,
        # Only whether this repo's ueler/ was used, never the absolute path: results get committed.
        "ueler_src": "working tree" if "ueler" in sys.modules and Path(sys.modules["ueler"].__file__).resolve().is_relative_to(REPO) else "external",
        "data_storage": fs_type(data),
    }


def to_markdown(result):
    lines = [f"# UELer render benchmark — {result['dataset_label']}", ""]
    env = result["environment"]
    lines.append(f"{env['platform']} · {env['cpu_count']} CPUs · {env['ram_gb']} GB RAM · Python {env['python']} · UELer {result.get('ueler_ref') or (env['ueler_commit'] + (' (dirty)' if env['ueler_dirty'] else ''))}")
    lines.append(f"Storage: {env['data_storage']}")
    if result.get("dataset"):
        lines.append(f"Dataset: {result['dataset']}")
    lines += ["", f"Viewer construction {result['construct_ms']:.0f} ms · first frame {result['first_frame_ms']:.0f} ms · RSS {result['rss_mb_end']} MB (peak {result['peak_rss_mb']} MB)", ""]
    disp = result.get("display") or {}
    if disp:
        lines += [f"Canvas {disp['canvas_css_px'][0]} CSS px × device pixel ratio {disp['device_pixel_ratio']} = {disp['canvas_device_px'][0]} px", ""]
    lines += ["| Scenario | Latency ms | Frames until correct (KB) | Wall ms | ms / motion | Renders | PNG frames | PNG KB | Stale PNG frames (KB) | Latest render shown | Top stages, self time (ms) |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]

    def row(name, r):
        top = ", ".join(f"{k} {v['self_ms']:.0f}" for k, v in list(r["stages"].items())[:4])
        lat = "–" if r.get("latency_ms") is None else f"{r['latency_ms']:.0f}"
        until = "–" if r.get("frames_until_correct") is None else f"{r['frames_until_correct']:g} ({r['kb_until_correct']})"
        lines.append(f"| {name} | {lat} | {until} | {r['wall_ms']:.0f} | {r.get('ms_per_motion_event', '')} | {r['renders']} | {r['png_frames']} | {r['png_kb']} | {r['stale_png_frames']} ({r['stale_png_kb']}) | {stale(r)} | {top} |")

    def stale(r):
        if not r["stale_at_end"]:
            return "no"
        if r.get("fresh_after_timer_ms") is not None:
            return f"**+{r['fresh_after_timer_ms']:.0f} ms** (timer)"
        return "**until next event**"

    for name, r in result["scenarios"].items():
        if name == "add_channel":
            for a in r:
                row(f"add channel → {a['channels_after']}", a)
        elif isinstance(r, dict):
            row(name, r)
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="Benchmark UELer's interactive render path headless.")
    p.add_argument("--data", required=True, help="dataset root (synth.py layout: images/ + masks/) or an image folder")
    p.add_argument("--masks", help="mask folder (default: <data>/masks when it exists)")
    p.add_argument("--channels", nargs="*", help="channel names to use, in order (default: all)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--label", help="label for the report (default: data folder name)")
    p.add_argument("--out", help="write JSON here; a .md report is written next to it")
    p.add_argument("--profile", help="write a cProfile dump of the whole run here")
    p.add_argument("--ueler-src", help="import ueler from this checkout instead of this repository (e.g. an older commit extracted by compare.py)")
    p.add_argument("--ueler-ref", help="label for --ueler-src in the report (e.g. the commit it was extracted from)")
    p.add_argument("--dpr", type=float, default=1.0, help="browser device pixel ratio (2 on a Retina screen)")
    p.add_argument("--canvas-px", type=int, help="square canvas edge in CSS px (default: the viewer's 600)")
    p.add_argument("--no-interactive", action="store_true", help="matplotlib interactive mode off (default on, as under %%matplotlib widget)")
    a = p.parse_args(argv)
    if a.ueler_src:
        sys.path.insert(0, str(Path(a.ueler_src).resolve()))

    data = Path(a.data)
    images = data / "images" if (data / "images").is_dir() else data
    masks = a.masks or (str(data / "masks") if (data / "masks").is_dir() else None)
    spec_file = data / "dataset.json"

    import tempfile

    profiler = None
    if a.profile:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()
    with tempfile.TemporaryDirectory() as settings:
        rss0 = rss_mb()
        s = Session(str(images), masks=masks, settings=settings, interactive=not a.no_interactive, dpr=a.dpr, canvas_px=a.canvas_px)
        scenarios = run_scenarios(s, a.channels, a.repeats)
        result = {
            "dataset_label": a.label or data.name,
            "matplotlib_interactive": not a.no_interactive,
            "dataset": json.loads(spec_file.read_text()) if spec_file.exists() else str(data),
            "environment": environment(str(images)),
            "ueler_ref": a.ueler_ref,
            "display": s.display,
            "construct_ms": round(s.construct_ms, 1),
            "first_frame_ms": round(s.first_frame_ms, 1),
            "rss_mb_start": rss0,
            "rss_mb_end": rss_mb(),
            "peak_rss_mb": peak_rss_mb(),
            "scenarios": scenarios,
        }
    if profiler:
        profiler.disable()
        profiler.dump_stats(a.profile)

    md = to_markdown(result)
    print(md)
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str))
        out.with_suffix(".md").write_text(md)
    return result


if __name__ == "__main__":
    main()
