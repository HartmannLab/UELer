"""Before/after comparison of two UELer versions with the render benchmark.

Extracts the *before* version's ``ueler/`` package from a git ref (default: the
baseline 3c80506, the v0.5.1-era nightly) and runs ``run.py`` against it and against the
working tree, on the same datasets, display settings and machine. Runs are
interleaved (before, after, before, ...) so drift in machine load or
temperature affects both versions alike, and each number is the median over
``--runs`` separate processes.

The result is a Markdown table of the latency the user waits on per operation,
plus every raw run as JSON::

    python tools/benchmark/compare.py \\
        --data /tmp/ueler_bench/small /tmp/ueler_bench/medium \\
        --dpr 1 2 --runs 3 --out /tmp/ueler_bench/compare
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tarfile
import io
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# (label, scenario key, how to pick the record from that scenario)
OPERATIONS = [
    ("Pan, one step (whole FOV in view)", "pan_full_view", None),
    ("Pan, one step (zoomed in 4×)", "pan_zoomed_4x", None),
    ("Pan, one step (mask outlines shown)", "pan_full_view_with_masks", None),
    ("Pan, sharp frame after letting go (whole FOV)", "pan_full_view", "release"),
    ("Zoom to a rectangle", "zoom_rectangle", None),
    ("Add a marker channel", "add_channel", "median_after_first"),
    ("Show mask outlines", "mask_overlay_on", None),
    ("Switch FOV (first visit)", "fov_switch_cold", None),
    ("Switch FOV while zoomed in (first visit)", "fov_switch_cold_while_zoomed", None),
    ("Switch FOV (visited before)", "fov_switch_warm", None),
]


def extract(ref, dest):
    """Write ``ueler/`` as of *ref* into *dest* (no worktree bookkeeping)."""
    dest.mkdir(parents=True, exist_ok=True)
    blob = subprocess.run(["git", "-C", str(REPO), "archive", ref, "ueler"], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(dest)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", ref], check=True, capture_output=True, text=True).stdout.strip()
    return commit


def pick(scenarios, key, how):
    r = scenarios.get(key)
    if r is None:
        return None
    if how == "median_after_first":
        # Adding a channel to an existing selection: the k-th add for k ≥ 2,
        # each the first read of that channel. Median over the adds.
        adds = [a for a in r if a.get("channels_after", 0) >= 2] or r
        return {
            "latency_ms": statistics.median(a["latency_ms"] for a in adds if a.get("latency_ms") is not None),
            "frames_until_correct": statistics.median(a["frames_until_correct"] for a in adds if a.get("frames_until_correct") is not None),
            "kb_until_correct": statistics.median(a["kb_until_correct"] for a in adds if a.get("kb_until_correct") is not None),
        }
    if how == "release":
        # Only a pan that previews coarse frames sends a frame on release.
        if r.get("release_latency_ms") is None:
            return None
        return {"latency_ms": r["release_latency_ms"], "frames_until_correct": 1, "kb_until_correct": r["release_kb"]}
    return r


def summarise(runs):
    """Median over runs of latency, frames and KB until the correct image, per operation."""
    out = {}
    for label, key, how in OPERATIONS:
        recs = [pick(run["scenarios"], key, how) for run in runs]
        recs = [r for r in recs if r and r.get("latency_ms") is not None]
        if not recs:
            continue
        out[label] = {
            "latency_ms": round(statistics.median(r["latency_ms"] for r in recs), 1),
            "latency_ms_runs": [r["latency_ms"] for r in recs],
            "frames_until_correct": statistics.median(r["frames_until_correct"] for r in recs),
            "kb_until_correct": round(statistics.median(r["kb_until_correct"] for r in recs), 1),
        }
    return out


def dataset_label(data):
    spec = Path(data) / "dataset.json"
    if spec.exists():
        d = json.loads(spec.read_text())
        return f"{d['n_fovs']} FOVs, {d['size']}×{d['size']} px, {d['n_channels']} channels, {d['cells_per_fov']:,} cells/FOV"
    return Path(data).name


def to_markdown(result):
    env = result["environment"]
    lines = [
        f"# UELer before/after — {result['before']['ref']} vs working tree",
        "",
        f"{env['platform']} · {env['cpu_count']} CPUs · {env['ram_gb']} GB RAM · Python {env['python']} · matplotlib {env['packages'].get('matplotlib')} · ipympl {env['packages'].get('ipympl')}",
        f"Median of {result['runs']} runs per version, interleaved. Latency = kernel time from the user's input until the PNG showing the correct image is sent; network transfer and browser decode come on top.",
        "",
    ]
    for block in result["blocks"]:
        lines += [f"## {block['dataset']} · canvas {block['display']}", ""]
        lines += ["| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after | KB sent until correct, before → after |", "| --- | --- | --- | --- | --- | --- |"]
        for label, _key, _how in OPERATIONS:
            b, a = block["before"].get(label), block["after"].get(label)
            if not a:
                continue
            if not b:
                # e.g. the release frame of a pan, which only a coarse drag preview sends.
                lines.append(f"| {label} | — | {a['latency_ms']:.0f} | — | — → {a['frames_until_correct']:g} | — → {a['kb_until_correct']:.0f} |")
                continue
            change = (a["latency_ms"] - b["latency_ms"]) / b["latency_ms"] * 100
            lines.append(f"| {label} | {b['latency_ms']:.0f} | {a['latency_ms']:.0f} | {change:+.0f} % | {b['frames_until_correct']:g} → {a['frames_until_correct']:g} | {b['kb_until_correct']:.0f} → {a['kb_until_correct']:.0f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare UELer render latency before/after, with tools/benchmark/run.py.")
    p.add_argument("--data", nargs="+", required=True, help="dataset roots (synth.py layout)")
    p.add_argument("--before", default="3c80506", help="git ref of the 'before' version (default: baseline 3c80506)")
    p.add_argument("--dpr", nargs="+", type=float, default=[2.0], help="device pixel ratios to test (2 = Retina)")
    p.add_argument("--canvas-px", type=int, help="square canvas edge in CSS px (default: the viewer's 600)")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--repeats", type=int, default=3, help="drag repeats within one run (passed to run.py)")
    p.add_argument("--frame-format", choices=["png", "jpeg", "webp"], help="frame format for both versions (passed to run.py; a version without the setting ignores it)")
    p.add_argument("--out", required=True, help="output folder")
    a = p.parse_args(argv)

    out = Path(a.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    src = out / f"src_{a.before}"
    commit = extract(a.before, src)

    blocks = []
    environment = None
    for data in a.data:
        for dpr in a.dpr:
            runs = {"before": [], "after": []}
            for i in range(a.runs):
                for side in ("before", "after"):
                    tag = f"{Path(data).name}_dpr{dpr:g}_{side}_{i}"
                    cmd = [sys.executable, str(HERE / "run.py"), "--data", data, "--repeats", str(a.repeats), "--dpr", str(dpr), "--label", tag, "--out", str(raw / f"{tag}.json")]
                    if a.canvas_px:
                        cmd += ["--canvas-px", str(a.canvas_px)]
                    if a.frame_format:
                        cmd += ["--frame-format", a.frame_format]
                    if side == "before":
                        cmd += ["--ueler-src", str(src), "--ueler-ref", commit]
                    print("running", tag, flush=True)
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                    result = json.loads((raw / f"{tag}.json").read_text())
                    environment = environment or result["environment"]
                    runs[side].append(result)
            disp = runs["after"][0]["display"]
            blocks.append({
                "dataset": dataset_label(data),
                "display": f"{disp['canvas_css_px'][0]} CSS px × {disp['device_pixel_ratio']:g} = {disp['canvas_device_px'][0]} px",
                "before": summarise(runs["before"]),
                "after": summarise(runs["after"]),
            })

    result = {"before": {"ref": a.before, "commit": commit}, "runs": a.runs, "environment": environment, "blocks": blocks}
    (out / "compare.json").write_text(json.dumps(result, indent=2))
    md = to_markdown(result)
    (out / "compare.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
