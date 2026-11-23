# Render-path performance and benchmarking

Pan, zoom, adding a marker and switching FOV feel slow on large FOVs, and the repo had no benchmark tooling: no number to state, and no baseline to measure fixes against. This document holds the investigation of the render path, the benchmark harness under `tools/benchmark/`, the fix order, and the record of each fix.

---

## Decisions

- **Speed up the existing renderer; no second renderer.** A WebGL renderer (vizarr) next to the matplotlib/ipympl path was considered and dropped. The work goes into the current path, and its remaining floor (one Agg draw, one PNG encode and one websocket round trip per settled frame) is documented as the architecture's limit.
- **Measure first, then fix in the order the measurements give.** Every fix is compared against the baseline commit 3c80506 with the same harness, datasets and machine.

---

## How a pan or zoom is rendered (as of 3c80506)

Every matplotlib `draw_event` calls `ImageDisplay.on_draw` (`ueler/viewer/image_display.py`), which compares the viewport with the previous one and, if it moved, calls `ImageMaskViewer.update_display` (`ueler/viewer/main_viewer.py`). That re-reads each visible channel, composites them (`ueler/rendering/engine.py`), recomputes mask outlines, notifies plugins through `inform_plugins('on_mv_update_display')`, and calls `draw_idle()` again. ipympl then rasterises the figure with Agg, encodes it as PNG and sends it to the browser.

### Suspected costs (from reading the code, before measuring)

1. **Rendering while dragging.** If ipympl redraws on each motion event during a pan, every step triggers a full `update_display`, and renders queue behind the mouse.
2. **Channels re-read from disk on every render.** In local folder mode `load_one_channel_fov` keeps each channel as a lazy Dask array over the TIFF, so every render reads and decompresses the file again. A first micro-benchmark (5 channels, 1024², zlib, local SSD) measured 20–22 ms per render lazy versus 0.2–2.4 ms in memory; on network storage the gap is larger.
3. **Channels materialised one at a time** — one `.compute()` per channel instead of a single `dask.compute(*channels)`.
4. **Full re-composite on single-channel changes** (adding a marker, changing one contrast), plus the statistics computed on first channel load.
5. **Mask outlines recomputed per frame** in `_label_boundaries`.
6. **Plugin hooks on every frame** (`on_mv_update_display` in Mask Painter, Export, Cell Table Editor).
7. **Small leaks:** `on_mouse_move` calls `draw_idle()` on every motion outside the axes; the canvas pixel size (HiDPI) sets the PNG payload.

---

## Benchmark harness (`tools/benchmark/`)

- `synth.py` generates synthetic datasets with controllable FOV count, image size, channel count, cell count and compression (zlib TIFFs, Voronoi cell masks).
- `run.py` builds the real viewer headless (Agg-backed ipympl canvas, no front end), models the browser's `waiting_for_image` handshake and IPython's post-message redraw, and times scenarios with a per-stage breakdown (channel reads, composite, overlays, plugin hooks, Agg draw, PNG encode), the frames sent and peak memory. It reports the latency of each input: kernel time from the input until the PNG showing the correct image is sent. Options: `--dpr` (device pixel ratio), `--canvas-px`, `--ueler-src` / `--ueler-ref` to run another checkout. Output is JSON plus Markdown with the machine, storage and package versions.
- `compare.py` extracts `ueler/` at an older commit (default 3c80506) with `git archive`, runs the same scenarios on it and on the working tree in interleaved separate processes, and reports the median over runs as a before/after table. Results are kept in `dev_note/benchmarks/`.

Usage is documented in `tools/benchmark/README.md`.

---

## Baseline (commit 3c80506)

Developer machine: macOS 26.5 arm64, 12 CPUs, 18 GB RAM, local APFS SSD, Python 3.10, matplotlib 3.10.9, ipympl 0.10.0, dask 2026.7.1. Canvas 600 × 600 px. Kernel time only — network and browser time come on top.

| Dataset | Pan frame, full view | Pan frame, zoomed 4× | Pan frame with masks | Add 1 → all channels | Cold / warm FOV switch | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 4 FOVs, 1024², 5 channels, 3 000 cells/FOV | 133 ms | 79 ms | 160 ms | 61 → 202 ms | 182 / 129 ms | ~600 MB |
| 3 FOVs, 2048², 10 channels, 12 000 cells/FOV | 581 ms | 182 ms | 681 ms | 140 → 1 110 ms | 1 470 / 629 ms | 1 010 MB |

Pan frames include the second, post-message draw and PNG (see the first finding): every pan motion event sends two frames, one of them stale.

### Findings

- **Every settled view is drawn and sent twice, and the first frame shows the old image.** Confirmed against a call-stack trace of a real session (`%matplotlib widget`, interactive mode on).
  - `on_draw` runs *inside* the Agg draw (matplotlib fires `draw_event` after rasterising) and installs the new render there. The PNG of that draw was rasterised before the new data existed, so it shows the previous render stretched to the new limits. The `draw_idle()` calls made during the re-render reach the browser while it is still waiting for that image and are dropped (`waiting_for_image` in ipympl's `mpl_widget.js`).
  - What delivers the new render is ipykernel: it triggers IPython's `post_execute` event after **every comm message**, not only after cell runs, and `%matplotlib widget` hooks matplotlib's `_draw_all_if_interactive` onto that event. The figure is stale after the re-render, so a `draw_idle()` goes out right after the PNG; the browser accepts it, and a second draw sends the up-to-date frame.
  - Measured in a real session after one rectangle zoom: `update_display` inside the draw 73 ms, a stale 2.2 MB full-frame PNG (Retina canvas, ~1 188 px), then the post-message draw request; the browser picked it up 310 ms later — busy decoding the large stale frame — and the correct 18 KB diff frame arrived 350 ms after the stale one. That is the "blurry for a moment, then updates" effect seen in the browser.
  - In the harness: a 10-step pan sends 21 PNG frames, 10 of them stale — 3.5 MB of 3.9 MB is outdated images; a rectangle zoom and a FOV switch from a zoomed view each send one stale frame before the correct one.
  - The hover timer adds load as well: each time it fires (300 ms after the last movement, including during drags) `process_hover_event` calls `draw_idle()`, costing a full Agg draw and PNG even when the tooltip does not change.
  - *How this was established.* Two earlier versions of this finding were wrong and were corrected by checks in a real browser. The first harness had no event loop and no post-message hook, so it predicted frames stuck stale "until the next event"; adding timers then made the 300 ms hover timer look like the rescue. A FOV switch refreshed on its own in the browser, which neither version explained. A kernel-side trace recording the call stack of every draw request (a notebook snippet hooking `send_json`, `send_binary`, `update_display` and the comm callbacks) showed the sender: `draw_all <- _draw_all_if_interactive <- trigger <- handle_msg <- comm_msg`. Side finding: under interactive mode matplotlib marks the scale bar, legend and axes stale *during* every draw (`set_offset`, `apply_aspect`), each calling `draw_idle()`; a real browser drops these while waiting for the image, but the harness's first frame originally skipped the waiting state and looped on them — a harness bug, fixed.
- **Compositing at full resolution for a 600 px canvas.** `DOWNSAMPLE_MAX_DIMENSION = 2048` keeps the full-view render at factor 1 for images up to 2048 px, so a 2048² × 10-channel pan frame blends 42 M pixel-channels and Agg then resamples it down to 600 px.
- **Channels re-read and re-decoded on every render.** Even zoomed in 4×, `read_channels` costs ~19 ms (1024² × 5) and ~117 ms (2048² × 10) per render, because the lazy Dask array covers the whole compressed TIFF.
- **Redundant renders.** Adding a channel triggers 2 `update_display` calls; a cold FOV switch triggers 2–3.
- **PNG encode ~25 ms and ~300 KB per frame** at 600 × 600 px; on a remote link the transfer time adds to every frame.
- **Status checkmark reflow.** `decorators._set_status_bar` assigns `value`, `format`, `width` and `height` as four separate widget updates, so the browser briefly shows the green checkmark at the spinner's 225 px width before the size update arrives, reflowing the layout. It also ran on every draw event, even short-circuited ones. Fix: set the four traits in one `hold_sync()` block, size before image.

---

## Fix order

1. **Done (Fix 1).** Render on a view change before the draw, not in `on_draw` after it; tooltip redraws only on change.
2. **Done (Fix 2).** Float32 composite and screen-aware downsample factor (canvas size × device pixel ratio).
3. **Done (Fix 3).** In-memory channel cache with a byte budget: full resolution when a channel fits, the viewed downsampled level when it does not. This is the one fix that adds memory by design; its memory is reported next to its timings.
4. Remove the redundant `update_display` calls on channel add and FOV switch.
5. **Done (Fix 5).** Status bar `hold_sync()`, plus the spinner cleared once the viewer is shown (it could stay on next to a loaded image after Fixes 1–2).
6. Re-measure, then decide on the composite itself (uint8 path), the PNG compression level and the remaining items from the suspected-cost list.

### Still to do on the harness

- Cell-table operations: selection, Cell Gallery, AnnData write-back; tables over 1 M cells.
- A run against remote data (e.g. BioImage Archive) and on the cluster, next to the local-SSD numbers.

---

## **Update Summary: Fix 1 — render before the draw (commit 69e4247)**

Every pan, zoom or FOV change was drawn and sent to the browser twice, and the first frame showed the previous image (see the first finding above).

---

### **Actions Taken**
- **The view-change render now runs before the canvas rasterises.**
  `ueler/viewer/predraw.py` adds `install_pre_draw_hook(canvas, callback)`, which wraps `canvas.draw` on the instance and runs the callback first. `ImageDisplay` installs `_render_if_view_changed` there: the viewport check, downsample-factor update and `update_display` that used to run in `on_draw`. ipympl's `handle_draw` calls `canvas.draw()`, so the one PNG sent for a view change already contains the new render, and the figure is not stale afterwards, so IPython's post-message `draw_all` has nothing left to send.
- **`draw_idle` is a no-op while the hook runs.**
  The draw that follows includes everything the render changed, so the `draw_idle()` at the end of `update_display` would only request a second identical frame (or, on Agg, recurse into a nested draw). The instance attribute is shadowed for the duration of the callbacks only and removed afterwards.
- **`on_draw` stays as a fallback.**
  It calls the same `_render_if_view_changed`, which finds the viewport unchanged after the hook and returns at once. A draw that bypasses `canvas.draw` still gets the old behaviour (render installed one frame late) rather than none. The logic lives in module-level functions rather than methods so the existing `on_draw` tests, which call `ImageDisplay.on_draw` on lightweight stubs, keep working unchanged.
- **The status bar only flips when a render happens.**
  `@update_status_bar` moved from `on_draw` (every draw, including tooltip redraws and short-circuited ones) to `_render_for_view`. This removes most occurrences of the checkmark reflow; the `hold_sync()` fix (item 5) is still needed for the renders themselves.
- **Channel grid view gets the same hook.**
  `ChannelGridDisplay._on_draw` is installed as a pre-draw hook too; its existing viewport check makes the post-draw call a no-op.
- **The hover tooltip redraws only when it changes.**
  `_show_tooltip` / `_hide_tooltip` skip `draw_idle()` when the visibility, text and anchor are already what they would set. Before, the hover timer redrew the whole figure (Agg draw + PNG) every time it fired, and every mouse-motion event outside the axes did the same. The checks compare with `is True` / `is False`, so mocked annotations in tests still take the redraw path. Settle-then-render during drags was not done: with one frame per motion and no stale frame the drag already shows the newest render each time; it can be revisited after the caching fix.
- **Harness: stale-frame detection moved to the rasterisation point.**
  `tools/benchmark/run.py` recorded the image version when `canvas.draw` was entered, which is before the new hook and therefore flagged every fresh frame as stale. It now records it when `Figure.draw` starts. Re-running the baseline with the corrected harness reproduces the earlier baseline numbers exactly (10 stale frames per pan, 1 per zoom), so the baseline table stands.

---

### **Results**

Kernel time per pan motion and frames sent to the browser, same machine and datasets as the baseline:

| Scenario | 1024² × 5: before → after | 2048² × 10: before → after |
| --- | --- | --- |
| Pan, full view: ms / motion | 133 → 118 | 581 → 553 |
| Pan, full view: PNG frames (stale) / KB sent | 21 (10) / 3 950 → 11 (0) / 3 558 | 21 (10) / 7 293 → 11 (0) / 3 556 |
| Pan, zoomed 4×: ms / motion | 79 → 58 | 182 → 163 |
| Pan, zoomed 4×: PNG frames (stale) / KB sent | 21 (10) / 4 974 → 11 (0) / 2 802 | 21 (10) / 4 447 → 11 (0) / 2 287 |
| Pan with masks: PNG KB sent | 7 619 → 6 887 | 15 720 → 7 624 |
| Rectangle zoom: PNG frames (stale) | 2 (1) → 1 (0) | 2 (1) → 1 (0) |
| FOV switch while zoomed: PNG frames (stale) | 2 (1) → 1 (0) | 2 (1) → 1 (0) |

Kernel time drops by the second Agg draw and PNG encode (~15–30 ms per motion). The larger gain is on the browser side, which the harness does not time: half the frames and up to half the bytes per view change, no stale frame to decode first, and no extra round trip before the correct image appears (~350 ms after a zoom in a Retina session).

**Confirmed in a real session** (`%matplotlib widget`, Retina, rectangle zoom). The browser's draw request arrives 16 ms after the mouse release. `update_display` then runs *before* the PNG (686 ms), the Agg draw and PNG encode take ~180 ms, and a single 3 723 KB frame goes out. No second draw request follows. Release to frame sent is ~880 ms, split into the render (~78 %) and rasterising plus encoding a full-size Retina frame (~20 %). The frame is far larger than the harness's 600 px canvas produces (~290 KB) because the viewer fills the notebook width (#104) at device pixel ratio 2 — the harness gained `--dpr` and `--canvas-px` for that case.

`compare.py` output for Fix 1 alone: [`dev_note/benchmarks/compare_fix1_vs_3c80506.md`](../benchmarks/compare_fix1_vs_3c80506.md). Pan steps and rectangle zooms are 6–36 % faster on the kernel side and need one frame instead of two; channel add, mask toggle and FOV switches from the full view are unchanged within run-to-run noise (±8 %), as expected since they never sent a stale frame.

---

### **Files Changed**
- `ueler/viewer/predraw.py` — new: `install_pre_draw_hook`.
- `ueler/viewer/image_display.py` — `_render_if_view_changed` / `_render_for_view` (moved out of `on_draw`), pre-draw hook installed in `_setup_event_connections`, `_show_tooltip` / `_hide_tooltip`.
- `ueler/viewer/channel_grid_view.py` — pre-draw hook for `_on_draw`.
- `tools/benchmark/` — new: `synth.py`, `run.py`, `compare.py`, `README.md`.
- `tests/test_predraw_render.py` — new, 11 tests: hook order, `draw_idle` suppression and restore, failing hook, render-before-rasterise on a real `ImageDisplay`, tooltip redraw-on-change.

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1435 tests, OK

---

## **Update Summary: Fix 2 — float32 composite and screen-aware downsample factor (commit 69e4247)**

After Fix 1 the render itself was 60–80 % of every latency. A check of what the planned screen-aware factor would buy showed it buys little on a Retina screen: a 600 CSS px canvas is 1 200 device px, so a 2048² image needs full resolution to stay sharp. Profiling the composite then showed it was slow for a reason unrelated to resolution, which became the main part of this fix.

---

### **Actions Taken**
- **`_composite_channels` works in float32, in place.**
  `uint16 − python float` promoted every channel to float64, and `normalised[..., None] * color` built an H × W × 3 temporary per channel. The loop now converts each channel once to float32, scales and clips it in place, and adds it only to the colour planes its colour uses (one plane for a pure red, green or blue marker). Microbenchmark at 2048² × 10 channels: ~570 → ~60 ms, maximum difference from the float64 formulation 1.8 × 10⁻⁷. Every render path goes through this function (`render_fov_to_array`: main viewer, map tiles, Cell Gallery, export), so all of them gain.
- **Screen-aware downsample factor.**
  `image_utils.screen_downsample_factor(width, height, screen_w, screen_h, …)` returns the largest power of two that still renders at least one image pixel per device pixel on both axes, and never less than the existing `DOWNSAMPLE_MAX_DIMENSION` (2048) factor. So the render is never blurrier than the screen shows and never larger than the old budget. The screen size is the axes' `bbox`, which under ipympl is in device pixels (the figure dpi already includes the device pixel ratio). A size below 64 px (canvas not laid out) falls back to the budget. Used by `_render_for_view` (main view) and `ChannelGridDisplay._sync_downsample_factor` (grid panes). The initial factor at viewer construction still uses the size budget; the first draw corrects it.
- **Where it helps.** Standard-DPI screens (2048² full view on a 600 px canvas: factor 1 → 2, a quarter of the pixels) and the channel grid, whose panes are small (a 3000 px view in one pane: factor 2 → 8). On Retina with images up to ~2 400 px it changes nothing, by design.
- **Ruled out:** rounding the factor to the *nearest* power of two (e.g. 1024 px rendered for 1 200 device px on Retina). It would have sped up the Retina case too, but at a visible loss of sharpness (an 0.85× upsample), which is a trade-off rather than a fix.
- **Tests.** Two channel-grid tests asserted factor 2 for a 3000 px view (issue #116 threshold); they now assert the screen-aware value and that it is at least the 2048-budget factor, which keeps their #116 intent. New `tests/test_screen_downsample_factor.py` (9 tests): the factor rule (no blur, never above budget, both axes, Downsample off, unknown screen, zoomed-in) and the float32 composite against the float64 reference, source array untouched, shape-mismatch cropping.

---

### **Results**

`compare.py` against 3c80506, median of 3 interleaved runs, latency the user waits on (kernel time from input to the PNG with the correct image); Fix 1 and Fix 2 together. Full output: [`dev_note/benchmarks/compare_fix2_vs_3c80506.md`](../benchmarks/compare_fix2_vs_3c80506.md). Large dataset (3 FOVs, 2048² px, 10 channels, 12 000 cells/FOV):

| Operation | Retina (1 200 px): before → after | Standard (600 px): before → after |
| --- | --- | --- |
| Pan, one step (whole FOV) | 747 → 326 ms (−56 %) | 664 → 164 ms (−75 %) |
| Pan, one step (zoomed 4×) | 312 → 203 ms (−35 %) | 191 → 137 ms (−28 %) |
| Pan, one step (mask outlines) | 883 → 446 ms (−49 %) | 777 → 216 ms (−72 %) |
| Zoom to a rectangle | 345 → 227 ms (−34 %) | 249 → 159 ms (−36 %) |
| Add a marker channel | 905 → 467 ms (−48 %) | 806 → 240 ms (−70 %) |
| Show mask outlines | 795 → 421 ms (−47 %) | 729 → 221 ms (−70 %) |
| Switch FOV (first visit) | 1 676 → 920 ms (−45 %) | 1 521 → 647 ms (−57 %) |
| Switch FOV (visited before) | 720 → 367 ms (−49 %) | 632 → 198 ms (−69 %) |

Small dataset (1024² × 5): −10 to −43 %. Where the remaining Retina pan step goes (326 ms): channel reads ~144 ms (re-reading and decompressing the TIFFs, fix 3), PNG encode ~83 ms and Agg draw ~61 ms (scale with device pixels), composite ~45 ms.

**Memory.** Fix 2 lowers memory rather than adding to it. The composite's transient peak at 2048² × 10 channels drops from 128 MB to 96 MB (same 48 MB result), since there are no float64 or per-channel H × W × 3 temporaries and the final clip is in place. The screen-aware factor is never finer than the old budget factor, so a render never holds more pixels than before, and the extra mask levels it requests are strided views of the full-resolution mask, not copies. Process peak RSS (median of 3): 2048² × 10 at standard DPI 1 059 → 866 MB; on Retina unchanged (~1 055 MB); the small dataset +8 to +14 MB, within run-to-run noise.

---

### **Files Changed**
- `ueler/rendering/engine.py` — float32 in-place `_composite_channels`.
- `ueler/image_utils.py` — `screen_downsample_factor`.
- `ueler/viewer/image_display.py`, `ueler/viewer/channel_grid_view.py` — use it with the axes' device-pixel size.
- `tests/test_screen_downsample_factor.py` — new; `tests/test_channel_grid_view.py` — two expectations updated.
- `dev_note/benchmarks/compare_fix1_vs_3c80506.{md,json}`, `dev_note/benchmarks/compare_fix2_vs_3c80506.{md,json}` — comparison output.

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1444 tests, OK

---

## **Update Summary: Fix 3 — decoded channel cache with a byte budget (commit 0bc36f2)**

After Fix 2, re-reading the channels was the largest single stage of a pan step: ~127 ms of a 319 ms Retina step at 2048² × 10 channels. In folder mode each channel is a Dask array over one compressed TIFF page, and the page is one Dask task, so slicing even a small zoomed-in region decompressed the whole file again on every render. Fix 3 decodes each channel once and serves every later render from memory.

---

### **Actions Taken**
- **`ChannelLevelCache` (`ueler/rendering/channel_cache.py`).**
  A least-recently-used cache keyed by `(fov, channel, factor)` with a total byte budget (512 MiB by default) and a per-entry cap of an eighth of the budget (64 MiB). `read_region(fov, channel, source, region_xy, factor)` returns exactly `source[ymin:ymax:f, xmin:xmax:f]`, or `None` when it does not handle the read and the caller keeps its lazy path.
- **Two kinds of entry.**
  A channel that fits the cap is kept at full resolution, and every region at every factor is a NumPy view of it: no copy, no alignment condition, and changing zoom never decodes again. A larger channel keeps only the downsampled level being viewed (`source[::f, ::f]`); it serves factor-aligned regions, which is what `get_axis_limits_with_padding` produces, and zooming to full resolution on it falls back to the lazy read.
- **Engine hook, not a new render path.**
  `render_fov_to_array` and `_composite_channels` take an optional `channel_reader`; without one they behave as before. `_compose_fov_image` passes a reader bound to the FOV (`_cached_channel_reader`), so the main view, map tiles and the channel grid all use the cache. Cell Gallery and export call the engine with their own arrays and are unchanged.
- **Correctness guards.**
  Each entry keeps the array it was decoded from and is discarded when `image_cache` holds a different one, so a reloaded channel is never served stale. Evicting a FOV from `image_cache` drops its entries. OME-TIFF FOVs (`OMEFovWrapper`, which read their own pyramid levels) and arrays already in memory are not cached. `set_max_bytes(0)` disables the cache.
- **Ruled out:** caching every factor level next to full resolution. With full resolution in memory a level is a strided view, so a second copy would only add memory. Caching the rendered composite instead of the channels would have saved the composite as well, but it is invalidated by every contrast, colour or channel change, which are the interactions that most need to be fast.

---

### **Results**

`compare.py` against 69e4247 (Fix 2), median of 3 interleaved runs, latency the user waits on. Full output: [`dev_note/benchmarks/compare_fix3_vs_69e4247.md`](../benchmarks/compare_fix3_vs_69e4247.md). Large dataset (3 FOVs, 2048² px, 10 channels, 12 000 cells/FOV):

| Operation | Retina (1 200 px): before → after | Standard (600 px): before → after |
| --- | --- | --- |
| Pan, one step (whole FOV) | 319 → 203 ms (−36 %) | 164 → 56 ms (−66 %) |
| Pan, one step (zoomed 4×) | 195 → 94 ms (−52 %) | 135 → 32 ms (−76 %) |
| Pan, one step (mask outlines) | 437 → 310 ms (−29 %) | 215 → 100 ms (−54 %) |
| Zoom to a rectangle | 220 → 116 ms (−47 %) | 153 → 47 ms (−70 %) |
| Add a marker channel | 443 → 310 ms (−30 %) | 237 → 118 ms (−50 %) |
| Show mask outlines | 415 → 288 ms (−31 %) | 210 → 98 ms (−53 %) |
| Switch FOV (first visit) | 888 → 708 ms (−20 %) | 647 → 499 ms (−23 %) |
| Switch FOV (visited before) | 359 → 232 ms (−35 %) | 186 → 74 ms (−60 %) |

Small dataset (1024² × 5): −4 to −39 %. A first FOV visit gains too, because a cold switch renders more than once (fix 4) and only the first render decodes now. Where the Retina pan step goes now (203 ms): PNG encode ~80 ms and Agg draw ~58 ms (both scale with device pixels), composite ~54 ms (up ~13 ms, since it now copies a strided view rather than receiving a fresh array); channel reads are gone from it (~127 ms before).

**Memory.** As planned, this fix adds memory. The cache holds at most its budget (512 MiB by default); a 2048² uint16 channel is 8 MiB at full resolution, so a 10-channel FOV is 80 MiB. Process RSS at the end of a run (median of 3), with the benchmark visiting all three FOVs: 2048² × 10 at standard DPI 819 → 954 MB (+135 MB), on Retina 1 016 → 1 162 MB (+146 MB); the small dataset +1 MB at standard DPI and +25 MB on Retina. The budget bounds the growth when more FOVs are visited; lower it with `viewer.channel_level_cache.set_max_bytes(n)` on a machine with little memory.

**Not done here:** the first visit to a FOV still reads each channel twice, once for the display statistics (`_update_channel_max`) and once for the cache. Computing the statistics from the decoded array would remove the second read; it touches the loader and the map-mode statistics path, so it is left for a separate change.

---

### **Files Changed**
- `ueler/rendering/channel_cache.py` — new; `ChannelLevelCache`.
- `ueler/rendering/engine.py` — optional `channel_reader` on `render_fov_to_array` and `_composite_channels`.
- `ueler/viewer/main_viewer.py` — `channel_level_cache`, `_cached_channel_reader`, entries dropped on FOV eviction.
- `tests/test_channel_level_cache.py` — new (12 tests).
- `dev_note/main_viewer.md` — channel cache paragraph.
- `dev_note/benchmarks/compare_fix3_vs_69e4247.{md,json}` — comparison output.

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1456 tests, OK

---

## **Update Summary: Fix 5 — status spinner left on next to a loaded image, missing on FOV switch; one-message status updates (commit pending)**

After the Fix 1–2 commit (69e4247) the status bar could keep showing the "processing" spinner while the image was already loaded (user report). Fix 5 was planned for the related reflow (the ready checkmark briefly drawn at the spinner's 225 px width) and is done together with it.

---

### **Actions Taken**
- **Cause: the spinner is the status widget's initial value, and Fix 1 changed what clears it.**
  `ui_components` creates `status_bar` showing the spinner. Before Fix 1, `ImageDisplay.on_draw` carried `@update_status_bar`, so the first draw after the viewer appeared, of any kind, switched it to ready. Fix 1 moved the decorator to `_render_for_view`, which runs only when the view has changed since the last render; a first draw that finds the view unchanged leaves the spinner on. Reproduced headless with the notebook's entry point: after `run_viewer(..., auto_display=True)` the widget still held the spinner, and only a later view-changing draw (the first pan or zoom) cleared it.
- **`display_ui` now ends with `set_status_ready(viewer)`,** right after `display(root)`. It is the path both entry points share: `run_viewer(auto_display=True)` calls `display_ui` directly, and `viewer.display()` calls it too. A first attempt put the reset at the end of `viewer.display()` only; that passed headless through `display()` but changed nothing in a notebook started with `run_viewer`, which never calls `display()`. The spinner still shows while the widget tree and plugins are built.
- **FOV switches and channel changes show the spinner again (user report on the first version of this fix).**
  Neither `on_image_change` nor `on_channel_selection_change` was decorated; before Fix 1 the spinner only flashed because the draw that followed went through the decorated `on_draw`. Now that draws no longer toggle the status, both handlers carry `@update_status_bar` themselves, so the spinner covers the actual load and render rather than flashing after it. `update_status_bar` now uses `functools.wraps`, so decorated methods keep their name and expose `__wrapped__`.
- **`_set_status_bar` sends one widget update.**
  The four traits are set inside `status_bar.hold_sync()`, size and format before the image, so the browser receives them as a single message instead of three or four. Checked with ipywidgets 8.1.8: one `send_state` carrying `value`, `format` and `width` (`height` is unchanged between the two states and is not sent). Objects without `hold_sync` (test doubles) are updated as before.
- **Kernel state verified for every interaction.** The benchmark harness's full scenario set (channel adds, pans, zoom, masks, FOV switches) ends with the widget in the ready state on the kernel side; `update_status_bar` resets it in a `finally`, including when the wrapped call raises.

---

### **Files Changed**
- `ueler/viewer/decorators.py` — `hold_sync` in `_set_status_bar`; new `set_status_ready`; `functools.wraps` in `update_status_bar`.
- `ueler/viewer/main_viewer.py` — `@update_status_bar` on `on_image_change` and `on_channel_selection_change`.
- `ueler/viewer/ui_components.py` — `display_ui` clears the spinner once the widget tree is shown.
- `tests/test_status_bar.py` — new (6 tests).

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1462 tests, OK
- Not yet checked in a browser.
