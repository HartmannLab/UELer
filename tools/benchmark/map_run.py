"""Benchmark map mode: activation, zooming in and back out to the whole map.

Lays the FOVs of a ``synth.py`` dataset out on a square grid, writes a map
descriptor for them, opens the viewer with map mode enabled and times, with the
same simulated browser as ``run.py``:

* activating the map (first render of every tile),
* zooming in 4x at three places and back out to the whole map,
* zooming in 2x and back out,
* a contrast change while the whole map is shown.

Example::

    python tools/benchmark/synth.py /tmp/ueler_bench/map25 --fovs 25 --size 2048 --channels 6 --cells 2000
    python tools/benchmark/map_run.py --data /tmp/ueler_bench/map25

Each line reports the latency until the correct frame is sent, the KB sent, the
stages with the most self time, the channel cache's size and hit/miss counts,
and the number of rendered tiles cached.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run as R  # noqa: E402  (sets up the import of ueler from this repository)


def write_descriptor(images: Path, settings: Path, fov_um: float) -> int:
    """One descriptor placing every FOV folder of *images* on a square grid."""
    import tifffile

    names = sorted(p.name for p in images.iterdir() if p.is_dir())
    first = next((images / names[0]).glob("*.tif*"))
    height, width = tifffile.imread(first).shape[:2]
    cols = math.ceil(math.sqrt(len(names)))
    fovs = []
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        fovs.append({
            "name": name,
            "slideId": "bench_slide",
            "centerPointMicrons": {"x": fov_um / 2 + c * fov_um, "y": fov_um / 2 + r * fov_um},
            "frameSizePixels": {"width": int(width), "height": int(height)},
            "fovSizeMicrons": fov_um,
        })
    # The viewer reads descriptors from <settings root>/.UELer/maps, and the
    # settings root for an image folder is that folder's name below --settings.
    for root in (settings, settings / images.name):
        (root / ".UELer" / "maps").mkdir(parents=True, exist_ok=True)
        (root / ".UELer" / "maps" / "bench.json").write_text(json.dumps({"fovs": fovs}))
    return len(names)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="synth.py dataset root (images/ + masks/)")
    p.add_argument("--dpr", type=float, default=1.0)
    p.add_argument("--canvas-px", type=int)
    p.add_argument("--fov-um", type=float, default=800.0, help="FOV edge in microns on the map grid")
    p.add_argument("--channels", type=int, default=6, help="number of channels to show")
    a = p.parse_args(argv)

    os.environ["ENABLE_MAP_MODE"] = "1"
    data = Path(a.data)
    images = data / "images" if (data / "images").is_dir() else data
    masks = str(data / "masks") if (data / "masks").is_dir() else None
    with tempfile.TemporaryDirectory() as tmp:
        n = write_descriptor(images, Path(tmp), a.fov_um)
        s = R.Session(str(images), masks=masks, settings=tmp, dpr=a.dpr, canvas_px=a.canvas_px)
        v = s.viewer
        ui = v.ui_component
        print(f"{n} FOVs on the map; maps found: {list(v._map_descriptors)}")
        s.set_trait(ui.channel_selector, list(ui.channel_selector.allowed_tags)[: a.channels])

        def report(label, rec):
            top = ", ".join(f"{k} {x['self_ms']:.0f}" for k, x in list(rec["stages"].items())[:5])
            cc = v.channel_level_cache
            print(f"{label:34s} {rec['latency_ms']} ms, {rec['kb_until_correct']} KB | {top} | "
                  f"channel cache {cc.nbytes >> 20} MiB, hits {cc.hits}, misses {cc.misses} | tiles cached {len(v._map_tile_cache)}")

        def view(frac, cx=0.5, cy=0.5):
            ax = v.image_display.ax
            w, h = v._map_canvas_size
            rec = {}
            with s.measure(rec):
                def deliver():
                    ax.set_xlim(w * cx - w * frac / 2, w * cx + w * frac / 2)
                    ax.set_ylim(h * cy + h * frac / 2, h * cy - h * frac / 2)
                    s.canvas.draw_idle()
                    s.pump()
                s._timed("widget", deliver)
            return rec

        report("activate map (whole map)", s.set_trait(ui.map_mode_toggle, True))
        print(f"map canvas {v._map_canvas_size} px, downsample factor {v.current_downsample_factor}")
        for cx, cy in ((0.3, 0.3), (0.7, 0.3), (0.5, 0.7)):
            report(f"zoom in 4x at ({cx}, {cy})", view(0.25, cx, cy))
        report("zoom out to the whole map", view(1.0))
        report("zoom in 2x", view(0.5))
        report("zoom out to the whole map", view(1.0))
        ch = list(ui.channel_selector.value)[0]
        slider = ui.contrast_max_controls[ch]
        report("contrast change, whole map", s.set_trait(slider, slider.value * 0.8))


if __name__ == "__main__":
    main()
