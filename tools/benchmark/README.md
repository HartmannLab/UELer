# UELer render benchmark

Measures how long UELer's interactive operations take and where the time goes, without a browser. It covers load time, pan/zoom latency, memory and scaling, and serves as the before/after yardstick for render-path optimisations. Design notes on the render path: [`dev_note/main_viewer.md`](../../dev_note/main_viewer.md).

## Quick start

```bash
micromamba activate ueler
python tools/benchmark/synth.py /tmp/ueler_bench/small --fovs 4 --size 1024 --channels 5 --cells 3000
python tools/benchmark/run.py --data /tmp/ueler_bench/small --out /tmp/ueler_bench/results/small.json
```

`run.py` prints a Markdown table and, with `--out`, writes the full JSON plus a `.md` copy next to it. It always imports `ueler` from this repository, not from an installed copy, so results reflect the working tree (the JSON records the commit and whether `ueler/` had uncommitted changes).

For a before/after table, `compare.py` runs the same scenarios against an older commit and the working tree:

```bash
python tools/benchmark/compare.py --data /tmp/ueler_bench/small /tmp/ueler_bench/medium --dpr 1 2 --runs 3 --out /tmp/ueler_bench/compare
```

It extracts `ueler/` at `--before` (default `3c80506`, the v0.5.1-era nightly) with `git archive`, runs `run.py --ueler-src` against it and against the working tree, interleaving the two so machine drift hits both alike, and writes `compare.md` (median latency per operation over `--runs` processes) plus `compare.json` and every raw run.

Real datasets work too: `--data` takes either a `synth.py`-style root (`images/` + `masks/`) or a plain image folder, with `--masks` for the mask folder and `--channels` to choose channels.

## What it does

`run.py` builds the real `ImageMaskViewer` on an Agg-backed ipympl canvas and drives it through the same message path a browser uses: mouse events enter through `canvas.handle_event`, exactly as ipympl's front end sends them. The browser itself is replaced by a model of ipympl's `mpl_widget.js` handshake:

- when the kernel sends a `draw` request, the browser answers with a `draw` message only if it is not already waiting for an image (`waiting_for_image`), otherwise the request is dropped;
- receiving a PNG clears the waiting flag and becomes the frame on screen.

After every browser message (mouse event, draw request, widget change) the harness also runs what ipykernel runs after each comm message: IPython's `post_execute` event, onto which `%matplotlib widget` hooks a redraw of every stale figure. In a real session this hook is what delivers a render installed *during* a draw, one frame later. Interactive mode is on by default, as under `%matplotlib widget`; `--no-interactive` turns it off.

There is no event loop, so matplotlib timers (`canvas.new_timer`) are recorded instead of scheduled; after each interaction the pending ones are fired in interval order, as a real event loop would.

That makes measurable what wall time alone hides: how many renders and PNG frames one interaction costs, how many of those frames show an outdated image, and whether the latest render reaches the screen at all (**Latest render shown**). Dropped draw requests are kept in the JSON (`dropped_draw_requests`).

## Scenarios

| Scenario | What it does |
| --- | --- |
| add channel → k | Adds the k-th channel to the selection (first read of that channel) |
| pan_full_view / pan_zoomed_4x | 10-step pan drag with the pan tool, whole FOV / central quarter |
| zoom_rectangle | Rectangle-zoom drag into 40 % of the view |
| mask_overlay_on | Ticks the first mask's display checkbox |
| pan_full_view_with_masks | Pan drag with the mask outlines shown |
| fov_switch_cold / warm | From the full view, switches to an unvisited FOV / back to a cached one |
| fov_switch_cold_while_zoomed | Switches to an unvisited FOV while zoomed in 4× (the switch resets the view) |

Drags report the median of `--repeats` runs.

## Display

`--canvas-px` sets the square canvas edge in CSS pixels (default: the viewer's 600) and `--dpr` the browser's device pixel ratio (2 on a Retina screen). ipympl renders at CSS size × ratio, so the ratio sets the Agg draw, PNG encode and transfer size. The ratio is applied the way the browser does it, with a `set_device_pixel_ratio` message.

## Columns

- **Latency ms**: kernel time from the user's input until the PNG that shows the correct (latest) image is sent. For a pan this is per motion step, for a rectangle zoom the release, for a widget change the change itself; drags report the median over steps and repeats. Network transfer and browser decode come on top, as does one browser round trip for every frame sent before the correct one. **Frames until correct (KB)**: how many PNGs, and how much data, the browser receives up to and including the correct one.
- **Wall ms**: kernel time for the whole interaction. **ms / motion**: per mouse-motion event, i.e. the kernel-side cost of one pan frame.
- **Renders**: `update_display` calls (re-read + composite). **PNG frames / PNG KB**: what ipympl would send to the browser.
- **Stale PNG frames (KB)**: frames whose image was rasterised before the render installed during the same draw — the browser shows an outdated image until the next frame arrives.
- **Latest render shown**: whether the last frame sent at the end of the interaction contains the latest render. `no` means it does. Otherwise the browser is showing an outdated image: `+N ms (timer)` means a pending matplotlib timer (the viewer's 300 ms hover timer) triggers the draw that fixes it, and `until next event` means nothing pending will.
- **Stages**: each instrumented function's *self* time (excluding nested stages). matplotlib fires `draw_event` inside the Agg draw, so a re-render triggered from `on_draw` runs within `agg_draw`; the JSON also has inclusive `ms` and call counts.

| Stage | Function |
| --- | --- |
| read_channels | `engine._slice_and_materialise` / `_slice_downsampled` — slicing and materialising channel arrays (disk read + decode for lazy arrays) |
| composite_channels | `engine._composite_channels` — contrast scaling and colour blending |
| mask_overlay / annotation_overlay | `engine._apply_mask_overlays` / `_apply_annotation_overlay` |
| render_fov | `engine.render_fov_to_array` |
| update_display, load_fov, plugin_hooks, scale_bar | the viewer methods of those names (`plugin_hooks` = `inform_plugins`) |
| selection_outlines | `ImageDisplay.update_patches` |
| status_bar | `decorators._set_status_bar` |
| agg_draw | `FigureCanvasAgg.draw` — rasterising the figure |
| png_encode | `canvas.get_diff_image` — the PNG ipympl sends |

## Not measured

Network round-trip time, browser-side PNG decode and paint, and front-end widget rendering. On a remote setup each PNG frame additionally costs its transfer time (PNG KB ÷ bandwidth) plus one round trip. Run the benchmark on the machine that holds the data to include its storage.
