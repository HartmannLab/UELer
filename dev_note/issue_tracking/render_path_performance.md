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
4. **Done (Fix 4).** One render per channel change, FOV switch and start-up: render requests are batched, and a render no longer starts a nested one.
5. **Done (Fix 5).** Status bar `hold_sync()`, plus the spinner cleared once the viewer is shown (it could stay on next to a loaded image after Fixes 1–2).
6. Re-measure, then decide on the composite itself (uint8 path), the PNG compression level and the remaining items from the suspected-cost list.
7. **Done (Fix 7).** Coarse preview while the view is dragged, one sharp frame on release. Aimed at remote sessions and slower clients, where the PNG transfer and browser decode of each full-size frame dominate (0.5–1 s per pan step reported on a Surface Pro). A kernel-side cap on the device pixel ratio was ruled out: ipympl's front end sizes its canvas from the browser's own ratio and draws each frame unscaled, so a smaller frame would fill only part of the canvas.
8. **Done (Fix 8).** Frames that replace most of the canvas can be sent as JPEG or WebP (Advanced Settings → Frame format), after the live timer showed a remote session spending ~4.4 s moving one 2.5 MB PNG frame.
9. **Done (Fix 9).** Map mode: zooming back out to the whole map re-rendered every tile and re-decoded every channel; the tile cache is now budgeted by bytes and the channel cache keeps small level copies for coarse reads.
10. **Done (Fix 10).** Map activation read every channel three times (two renders at different factors, plus the per-tile statistics); it now renders once at the screen factor, reduces the statistics from the decoded copy and decodes a tile's channels in parallel.

### Still to do on the harness

- Real-device numbers: `ueler.viewer.latency_timer.LatencyTimer` (below, after Fix 7) times live sessions in the browser and the kernel; results from the Surface Pro and the remote setup are pending.

- Cell-table operations: selection, Cell Gallery, AnnData write-back; tables over 1 M cells.
- A run against remote data (e.g. BioImage Archive) and on the cluster, next to the local-SSD numbers.
- Map mode now has `map_run.py` (Fix 9). Activation of a 25-tile map (2048² × 6 channels) went from ~5.8 s to ~1.9 s in Fix 10; what remains is decoding each channel once and its statistics.

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

## **Update Summary: Fix 5 — status spinner left on next to a loaded image, missing on FOV switch; one-message status updates (commit bd0549e)**

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
- ✅ Checked in a notebook (VS Code): the checkmark appears once the viewer is shown, and the spinner shows while a FOV switch loads.

---

## **Update Summary: Fix 4 — one render per channel change and FOV switch (commit 5e43164)**

Adding a channel rendered twice, and a first FOV visit rendered twice on the large dataset and whenever the view was zoomed in. After Fix 3 the repeated render no longer decodes anything, so it cost a composite, mask outlines and plugin hooks rather than a file read, but it was still pure overhead. A call-stack trace of every `update_display` in the harness showed three sources.

---

### **Actions Taken**
- **Found the nested render on channel add.**
  The render itself loads a newly selected channel (`_compose_fov_image` → `load_fov`). A first load computes the channel's statistics and `_sync_channel_controls` clamps its contrast sliders to the new range; each slider's observer calls `update_display`, so a second render ran *inside* the first. The outer render reads the slider values after the load and installs its image last, so the nested one was never seen.
- **Found the extra render on a zoomed FOV switch.**
  `on_image_change` resets the axes to the whole image and renders, but at the downsample factor of the previous, zoomed view. The pre-draw hook (Fix 1) then saw a changed view and rendered again at the right factor; the first render was both wasted and the expensive one (full view at factor 1).
- **Found the observer cascade.**
  `update_controls` gives reused sliders the new FOV's range, which can move their values; every moved slider asked for a render, and so did the channel-selection observer that `on_image_change` triggers, before `update_controls` asked again.
- **`_batched_display_updates(viewer, render=True)` (`main_viewer.py`).**
  A context manager around a group of widget changes. Inside it `update_display` only records that a render was asked for and at which factor (last wins); the outermost block renders once on exit. Blocks nest. With `render=False` the requests made inside are dropped, keeping any that were pending before. A block that exits with an error does not render, so a half-updated state is not drawn and the error is not masked by a render error. It is a depth counter, not a flag, so it composes with the existing `_suspend_display_updates` (which still wins: a suspended call records nothing).
- **Applied it in four places.**
  `update_controls` (body moved to `_rebuild_controls`); `on_image_change`, whose work up to and including `update_controls` moved to `_load_selected_fov_and_controls` and is flushed *before* `_reapply_selection_highlights`, which must follow the render; viewer start-up, from the first `update_controls` to the render after the widget states are restored; and, with `render=False`, the `load_fov` calls on the render path (`_compose_fov_image`, and `render_image` when called without a viewport).
- **`adopt_current_view(display)` (`image_display.py`).**
  After `on_image_change` resets the limits it picks the downsample factor for the whole-image view and records that view as rendered, so the batched render uses the right factor and the pre-draw hook finds nothing to redo. It does nothing before start-up finishes: there the first draw must still pick the screen-aware factor (Fix 2) and render. `_render_if_view_changed` was split into `_view_geometry`, `_remember_view` and `_apply_downsample_factor_for_view` so both paths share the same code.
- **Harness counts real renders.**
  `run.py` now counts `render_image` calls as renders, because a batched `update_display` call no longer renders. For the earlier versions the two counts are the same.
- **Ruled out:** a generic re-entrancy guard that defers a nested `update_display` to after the outer render. It would keep the second render (the deferred one), since the guard cannot know whether the outer render already read the changed state. On the render path it does know, because the settings are read after `load_fov`, so dropping is safe exactly there and nowhere else. Removing the slider observers' renders altogether was also ruled out: they are how a user's contrast change reaches the screen.
- **Left as is:** in channel-grid mode `on_image_change` still calls `_update_grid_display` after the highlights, although the batched `update_display` has already routed to it. Grid mode is outside the harness, and the second call follows the highlights, which a grid refresh may depend on; it needs its own check.

---

### **Results**

Renders per operation (harness, all runs and both screens): adding a channel 2 → 1 (from the second channel on); first FOV visit 2 → 1 on the large dataset; first FOV visit while zoomed 2 → 1 on both datasets; a revisit stays at 1.

`compare.py` against bd0549e (Fixes 1–3 and 5), median of 3 interleaved runs. Full output: [`dev_note/benchmarks/compare_fix4_vs_bd0549e.md`](../benchmarks/compare_fix4_vs_bd0549e.md). Large dataset (3 FOVs, 2048² px, 10 channels):

| Operation | Retina (1 200 px): before → after | Standard (600 px): before → after |
| --- | --- | --- |
| Add a marker channel | 317 → 256 ms (−19 %) | 122 → 106 ms (−13 %) |
| Switch FOV (first visit) | 740 → 657 ms (−11 %) | 521 → 501 ms (−4 %) |
| Switch FOV while zoomed in (first visit) | 732 → 648 ms (−11 %) | 568 → 498 ms (−12 %) |
| Switch FOV (visited before) | 244 → 240 ms (−2 %) | 76 → 74 ms (−2 %) |

Small dataset (1024² × 5), standard screen: add a channel −12 %, first FOV visit −19 %, zoomed first visit −26 %; on Retina −5 % and within noise for the switches. Pan and zoom do not go through these paths; their rows move within run-to-run noise (±10 % on the small dataset at standard DPI). Viewer construction is ~5 % faster (large dataset 1 307 → 1 214 ms at standard DPI) because start-up now renders once. The gains are smaller than the render counts suggest because, since Fix 3, the second render was a warm one: the time saved is one composite, outline pass and plugin round, not a file read. A first FOV visit is now dominated by reading the channels (and reading them twice, see Fix 3).

---

### **Files Changed**
- `ueler/viewer/main_viewer.py` — `_batched_display_updates`; the guard in `update_display`; `update_controls` → `_rebuild_controls`; `on_image_change` → `_load_selected_fov_and_controls` plus `adopt_current_view`; batched start-up; absorbed loads in `_compose_fov_image` and `render_image`.
- `ueler/viewer/image_display.py` — `adopt_current_view`; view tracking split into shared helpers.
- `tools/benchmark/run.py`, `tools/benchmark/README.md` — renders counted as `render_image` calls.
- `tests/test_render_batching.py` — new (12 tests).
- `tests/test_initial_display.py`, `tests/test_map_mode_initial_display.py` — the `MagicMock` viewers run the real `_load_selected_fov_and_controls`.
- `dev_note/FOV_load_cycle.md` — batching paragraph under `update_display`.
- `dev_note/benchmarks/compare_fix4_vs_bd0549e.{md,json}` — comparison output.

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1474 tests, OK

---

## **Update Summary: Fix 7 — coarse preview while dragging, sharp frame on release (commit 4ff01d0)**

On a Surface Pro the viewer was reported to lag 0.5–1 s behind each pan step, and on remote sessions too. By then the kernel spent about 200 ms on a Retina pan step of the large dataset, but every step still sent a full-size PNG of about 1–2.4 MB (frames are rendered at CSS size × device pixel ratio). On a remote link, and for a slower client that has to decode each frame, that transfer and decode dominate. A pan step only has to show where the view is going; the detail matters once the view stops.

---

### **Actions Taken**
- **Ruled out first: capping the device pixel ratio in the kernel.**
  ipympl's front end sizes its offscreen canvas from the browser's own `devicePixelRatio` and draws each frame onto it unscaled (`drawImage(image, 0, 0)`), so a frame rendered at ratio 1 would fill a quarter of the canvas, and mouse coordinates would no longer match. Doing it properly means patching ipympl's JavaScript.
- **Measured what a coarse frame saves before building it.**
  1 200 px frames from the 2048² synthetic channels, PNG at matplotlib's level: factor 2 (sharp on Retina) 2 352 KB / 84 ms encode, factor 4 884 KB / 43 ms, factor 8 267 KB / 28 ms. A coarse render upscaled to the canvas repeats pixels, which PNG's filters and deflate compress far better.
- **`image_utils.drag_preview_factor(width, height, max_px)`.**
  The smallest power of two that keeps the longer side of the view at or under `max_px` rendered pixels, with a 1e-6 tolerance so float drift in the limits during a pan (2048.0000001) does not flip the factor mid-drag; that flip was seen in the first trace.
- **Drag state in `ImageDisplay`.**
  A press with the pan tool inside the axes sets `_view_drag_active`; release or leaving the figure clears it. While it is set, `_apply_downsample_factor_for_view` uses the coarser of the screen factor and the preview factor and records `_coarse_view_rendered`. The preview never makes a render sharper than the screen factor, and with Downsample off (native pixels requested) there is no preview.
- **Sharp frame on release.**
  If the last frame was coarse, release sets `_sharp_render_requested`, marks the axes stale and calls `draw_idle()`. `_render_if_view_changed` renders when that flag is set even though the view has not moved. The stale mark matters: if the browser is still waiting for the last preview frame it drops the draw request (`waiting_for_image`), and the stale figure is then redrawn by the post-message `draw_all` after the browser's next message (see the first finding of the baseline).
- **Setting.**
  `viewer.drag_preview_px` (default `DRAG_PREVIEW_PX = 256`); `0` or `None` keeps every drag frame sharp. No widget yet; it can be added to Advanced Settings if it needs to be switched often.
- **Harness.**
  Pan records gain `release_latency_ms` / `release_kb` (the frame sent on release), `compare.md` gains a "sharp frame after letting go" row and a column with the KB sent until the correct image, and `run.py --drag-preview-px` sets the preview (0 = off).

---

### **Results**

`compare.py` against 5e43164 (Fix 4), median of 3 interleaved runs. Full output: [`dev_note/benchmarks/compare_fix7_vs_5e43164.md`](../benchmarks/compare_fix7_vs_5e43164.md). Large dataset (3 FOVs, 2048² px, 10 channels), Retina canvas (1 200 px):

| Operation | Kernel ms, before → after | KB per frame, before → after |
| --- | --- | --- |
| Pan, one step (whole FOV) | 201 → 36 (−82 %) | 1 088 → 156 |
| Pan, one step (zoomed 4×) | 94 → 33 (−65 %) | 686 → 117 |
| Pan, one step (mask outlines) | 310 → 55 (−82 %) | 2 374 → 131 |
| Sharp frame after letting go (whole FOV) | — → 188 | — → 1 060 |

Each drag step now sends about 7–18× fewer bytes and costs a sixth of the kernel time; the full-size frame is sent once per drag instead of once per step. Before, the release sent only a tiny diff frame (1–6 KB), since the last step was already sharp. Operations other than pan are unchanged within noise.

On a standard 600 px canvas the steps are 33–50 % faster but barely smaller (whole FOV 353 → 332 KB): a 256 px preview upscaled 2.3× to the canvas does not repeat pixels in whole blocks, and Agg's resampling leaves little for PNG to exploit. The gain in bytes is a high-DPI effect, which is where the frames were largest. A zoomed-in view whose sharp factor already gives ≤ 256 px of detail gets no preview (small dataset zoomed 4× at standard DPI).

**Not verified yet:** the feel in a real browser, on the Surface and over a remote connection; the harness measures kernel time and bytes, not transfer or decode.

---

### **Files Changed**
- `ueler/image_utils.py` — `drag_preview_factor`.
- `ueler/viewer/image_display.py` — `DRAG_PREVIEW_PX`; drag start/end handlers; preview factor in `_apply_downsample_factor_for_view`; sharp render on request in `_render_if_view_changed`.
- `ueler/viewer/main_viewer.py` — `drag_preview_px` setting.
- `tools/benchmark/run.py`, `tools/benchmark/compare.py`, `tools/benchmark/README.md` — release-frame metrics, KB column, `--drag-preview-px`.
- `tests/test_drag_preview.py` — new (15 tests).
- `dev_note/main_viewer.md` — drag preview paragraph.
- `dev_note/benchmarks/compare_fix7_vs_5e43164.{md,json}` — comparison output.

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1489 tests, OK

---

## **Update Summary: live latency timer for real devices (commit a6ad780)**

The harness measures kernel time and bytes on the development machine. What the 0.5–1 s lag reported on a Surface Pro and on a remote session consists of — kernel, network or the client's decode — cannot be read from it, and a larger canvas makes every frame bigger (on the large dataset at Retina, going from a 600 to a 1 000 CSS px canvas raised a sharp frame from ~1.1 to ~2.8 MB and its PNG encode from ~85 to ~230 ms). `LatencyTimer` measures a live session on the device itself, so the next fix can target the part that dominates there.

---

### **Actions Taken**
- **Browser side.**
  A script in the cell output hooks `URL.createObjectURL` and `HTMLImageElement.src`, which ipympl uses for every PNG frame, and records each frame's size, arrival and load (decoded) time; frames much smaller than the main canvas (other figures) are ignored. Capturing listeners on the document record presses, drags, releases and scrolls on an ipympl figure and changes on other widgets, with the event's own timestamp. It shows a live Markdown summary with a text box to copy, and `window.__uelerLatency.report()` in the console returns the same text.
- **Kernel side.**
  Wraps the canvas's `handle_event` (mouse input), `send_binary` (frame leaves the kernel, with its size), `get_diff_image` (PNG encode) and the viewer's `update_display` (render), and the comm class's `handle_msg` for widget changes. `stop()` restores all of them.
- **Same grouping on both sides.**
  `summarise_interactions` (Python) and its JavaScript twin group inputs into interactions (press-to-release drag, scroll, widget change; closed after 1.5 s quiet) and report for drags the frame interval, input-to-frame lag, frame size and release-to-last-frame time, and for widget changes input to first and last frame. The two were checked against the same event log.
- **Ruled out:** a single clock for both sides. The kernel and the browser may run on different machines, so the network share is read as the difference between the two summaries, not per frame.

---

### **Files Changed**
- `ueler/viewer/latency_timer.py` — new; `LatencyTimer`, `summarise_interactions`, `format_summary`, browser panel.
- `tests/test_latency_timer.py` — new (6 tests).
- `tools/benchmark/README.md` — pointer to the live timer.

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1495 tests, OK
- Kernel side run against the harness's simulated browser (drags and a FOV change through the comm path); the browser script parsed with `node --check` and its summary checked against the Python one on the same events. Not yet run in a real browser.

---

## **Update Summary: Fix 8 — selectable frame format (PNG, JPEG, WebP) and timer grouping for slow links (commit 7a11959)**

The first real-device numbers (`LatencyTimer`, VS Code on a Surface Pro, kernel on a remote Linux server, a 1 605 CSS px canvas at pixel ratio 0.833, so 1 337 device px) located the lag: the browser decoded a frame in 37 ms and the kernel rendered in 15 ms, but the sharp frame after a drag (2.48 MB PNG) reached the screen 5.5 s after the release while the kernel had sent it after 1.1 s. That is ~0.5 MB/s (4–5 Mbit/s) of effective bandwidth, and 225 KB drag frames at that rate account for the 420–500 ms between drag frames on both sides. Bytes per frame are the bottleneck on such a link, and PNG, being lossless, is the wrong format for a full frame of a multiplexed image.

---

### **Actions Taken**
- **Measured the formats at that canvas first.**
  A sharp 1 337 px frame of the synthetic 2048² data: PNG 1 510 KB / 138 ms encode, PNG at compression level 1 3 168 KB / 37 ms, JPEG q85 398 KB / 19 ms, JPEG q70 277 KB / 19 ms, WebP q80 258 KB / 46 ms. For a drag preview frame PNG won (142 KB, against 488 KB JPEG and 313 KB WebP), because its repeated pixels suit PNG's filters and the block edges suit JPEG badly.
- **`ueler/viewer/frame_encoding.py`.**
  `install_frame_encoder(canvas, get_format, is_preview)` replaces the canvas's `get_diff_image` on the instance. With `png` it calls matplotlib's own method unchanged. With `jpeg` or `webp` a frame goes lossy only when it replaces most of the canvas (a forced full frame, a size change, or at least 25 % of pixels changed); it is then sent in `full` image mode, since a lossy frame has no transparency to overlay a diff with, and the same bookkeeping as matplotlib's (`_last_buff`, `_force_full`, `_png_is_old`) is done. Small updates (tooltip, highlight) stay PNG diffs, and drag preview frames (`image_display._coarse_view_rendered`) stay PNG. Any encoder error falls back to PNG.
- **The front end needs no change.**
  ipympl wraps every frame in a `Blob` typed `image/png`, but browsers choose the image decoder from the data's signature, not the declared type, so a JPEG or WebP frame displays like a PNG. This is the one assumption to confirm on each client; the timer shows it (frames keep arriving and decoding).
- **Advanced Settings → Frame format.**
  A dropdown with PNG (lossless, the default), JPEG (q85) and WebP (q80, `method=0`), listing only what the installed Pillow can encode. `on_frame_format_change` sets `viewer.frame_format`, read at every frame, so a change applies to the next one. The value is saved and restored with the other widget states. PNG stays the default because the lossy formats soften thin outlines and text slightly and the decoder assumption is not yet confirmed on every client.
- **Timer grouping fixed for slow links.**
  The first rule closed an interaction after 1.5 s without input or frame, so a sharp frame still in transit for 4 s fell outside its drag (and the report showed a 234 KB "last frame"). An interaction now runs until the next press, scroll or widget change (bursts of scroll or widget inputs under 0.5 s apart count once; frames more than 30 s after the last input are ignored), and its settled frame is the largest one after the release or input, not the last, so a later tooltip update does not count. Python and JavaScript implement the same rule and were checked on the same slow-link event log. The kernel report now includes the frame format.
- **Harness.** `run.py --frame-format` and `compare.py --frame-format` set the format (a version without the setting ignores it).

---

### **Results**

`compare.py` against 4ff01d0 with `--frame-format jpeg`, at the measured client's canvas (1 605 CSS px × 0.833 = 1 337 px), large dataset, median of 3 runs. Full output: [`dev_note/benchmarks/compare_fix8_vs_4ff01d0.md`](../benchmarks/compare_fix8_vs_4ff01d0.md).

| Operation | Kernel ms, before → after | KB sent, before → after |
| --- | --- | --- |
| Sharp frame after letting go of a pan | 214 → 126 | 1 264 → 423 |
| Zoom to a rectangle | 138 → 50 | 988 → 233 |
| Add a marker channel | 273 → 167 | 1 459 → 439 |
| Show mask outlines | 298 → 250 | 1 953 → 1 017 |
| Switch FOV (first visit) | 647 → 553 | 1 405 → 442 |
| Switch FOV (visited before) | 256 → 150 | 1 404 → 442 |

Pan steps are unchanged (PNG previews, 99–131 KB). Full updates send about a third of the bytes and the kernel saves the PNG encode (~90 ms each). At the measured ~0.5 MB/s a sharp frame of this size arrives ~1.7 s sooner; the real data's 2.5 MB PNG frames should shrink by a similar or larger factor. Mask outlines compress least (high-frequency edges), about half. WebP is ~35 % smaller again than JPEG at roughly twice JPEG's encode time.

**Not verified yet:** the frames displaying in the client browsers (VS Code on Windows in particular), and the effect on the real link; to be measured with the timer.

---

### **Files Changed**
- `ueler/viewer/frame_encoding.py` — new; `install_frame_encoder`, `available_frame_formats`, `encode_lossy`.
- `ueler/viewer/main_viewer.py` — encoder installed on the main canvas; `frame_format`; `on_frame_format_change`.
- `ueler/viewer/ui_components.py` — `frame_format_dropdown` in Advanced Settings.
- `ueler/viewer/latency_timer.py` — interaction grouping by start input, largest frame as settled frame, frame format in the environment line.
- `tools/benchmark/run.py`, `tools/benchmark/compare.py` — `--frame-format`.
- `tests/test_frame_encoding.py` — new (9 tests); `tests/test_latency_timer.py` — grouping tests for slow links.
- `dev_note/main_viewer.md` — frame format paragraph.
- `dev_note/benchmarks/compare_fix8_vs_4ff01d0.{md,json}` — comparison output.

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1506 tests, OK
- Frame format saved and restored across two viewer sessions sharing a settings folder.

---

## **Update Summary: Fix 5 follow-up — spinner when switching to map mode or between maps (commit 1ff849b)**

Reported after Fix 8: switching to map mode left the status at the ready checkmark while the map rendered. Fix 5 had moved the spinner from the draw to the handlers that load and render, but only to the FOV and channel handlers; `on_map_mode_toggle` and `on_map_selector_change`, which activate a map and render its tiles, were missed.

---

### **Actions Taken**
- **Decorated both map handlers with `@update_status_bar`.**
  The spinner now covers activating a map, computing its channel statistics and rendering its tiles, and also deactivating it (back to the single FOV). The first draw after activation still renders through `_render_for_view`, which has its own status.

---

### **Files Changed**
- `ueler/viewer/main_viewer.py` — `@update_status_bar` on `on_map_mode_toggle` and `on_map_selector_change`.
- `tests/test_status_bar.py` — the handler check covers both map handlers.

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1506 tests, OK

---

## **Update Summary: Fix 9 — map mode: zooming out to the whole map from memory (commit 1be8820)**

Reported: in map mode, zooming back out to the whole map took a long time. Measured on a synthetic map (`tools/benchmark/map_run.py`, 25 FOVs of 2048² px on a 5 × 5 grid, 6 channels, map canvas 10 240 px, whole-map factor 16): 1.3–1.5 s every time the view returned to the whole map after zooming in, while returning to it with nothing in between took 12 ms. The channel cache counters showed ~150 misses per zoom-out, i.e. every channel of every tile decoded from disk again.

---

### **Actions Taken**
- **Found two caches evicting the overview.**
  The rendered-tile cache (`_map_tile_cache`) held `max(6, number of FOVs)` tiles, one per tile at most, keyed by factor and region; three 4× zoom-ins filled it with zoomed tiles and pushed the 25 overview tiles out. Re-rendering those then needed every channel, and the channel cache (Fix 3) keeps channels at full resolution: 25 × 6 × 8 MiB = 1.2 GB against a 512 MiB budget, so the overview's channels had long been evicted as well.
- **Tile cache budgeted by bytes.**
  `VirtualMapLayer` takes `cache_max_bytes` and evicts least recently used tiles until the shared cache fits; the viewer passes `_map_tile_cache_max_bytes` (256 MiB). A whole-map tile at factor 16 is 128² × 3 float32 = 192 KB, so the overview of even a large map stays cached while zoomed-in tiles come and go. The count limit is kept for layers built without a budget.
- **Level copies in the channel cache.**
  A read at factor ≥ 4 from an aligned region also stores `full[::f, ::f]` as a separate entry in its own share of the budget (a quarter, 128 MiB). When the full-resolution entry has been evicted, the next read at that factor is served from the copy without decoding. A copy is 1/16 of the channel or less (32 KB per channel at factor 16), so the whole-map overview of thousands of channel tiles fits. The total stays within `max_bytes`; `drop_fov`, a replaced source and `set_max_bytes(0)` drop copies as they drop entries.
- **Map benchmark.**
  `tools/benchmark/map_run.py` writes the grid descriptor, enables map mode and times activation, zooms and a contrast change, with the channel cache's hit and miss counts.
- **Ruled out:** making the channel cache store only levels in map mode. Zooming in on a tile would then decode it again at every new factor, where a cached full-resolution channel serves all factors; the copies add the overview without giving that up.

---

### **Results**

`map_run.py`, same machine and dataset, before (1ff849b) → after:

| Action | Before | After |
| --- | --- | --- |
| Zoom out to the whole map after zooming in | 1 305 / 1 520 ms | 42 / 40 ms |
| Zoom in 2× after that | 568 ms | 47 ms |
| Contrast change at the whole-map view | 1 565 ms | 46 ms |
| First zoom-in 4× (new tiles at a new factor) | 390–565 ms | 405–595 ms (unchanged) |
| Map activation (first visit of every tile) | 5.1 s | 5.2 s (unchanged) |

The contrast change re-renders every tile (the tile keys include the contrast) and now takes all 150 channel reads from level copies. What remains slow is reading a tile for the first time: decoding its compressed TIFFs, plus, on activation, the per-tile statistics (`_update_stats_for_visited_tile`, ~1.8 s of the 5.2 s), which decode each channel a second time. Memory: up to 256 MiB more for rendered tiles; the level copies live inside the existing 512 MiB channel budget.

---

### **Files Changed**
- `ueler/viewer/virtual_map_layer.py` — `cache_max_bytes`; byte-based eviction in `_cache_store`.
- `ueler/viewer/main_viewer.py` — `_map_tile_cache_max_bytes`, passed to each layer.
- `ueler/rendering/channel_cache.py` — level copies (`_levels`, `level_fraction`, `LEVEL_COPY_MIN_FACTOR`).
- `tools/benchmark/map_run.py` — new; `tools/benchmark/README.md` — map section.
- `tests/test_channel_level_cache.py` — level copy tests (4); `tests/test_map_tile_cache_budget.py` — new (3 tests).
- `dev_note/main_viewer.md`, `dev_note/FOV_load_cycle.md` — cache descriptions.

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1513 tests, OK

---

## **Update Summary: Fix 10 — map activation reads each channel once (commit TBD)**

Reported: switching to map mode still took a while; the guess was the cost of setting up the Dask arrays. Profiling activation with `map_run.py` (25 FOVs of 2048² px, 6 channels) showed 5.8 s, little of it Dask's own overhead: the channel cache logged 300 misses for 150 channels, and the per-tile statistics read every TIFF a third time.

---

### **Actions Taken**
- **Found the second render.**
  `_activate_map_mode` picked the downsample factor from the canvas size alone (`select_downsample_factor`, factor 8 for a 10 240 px map), and `on_map_mode_toggle` rendered every tile at it. The next draw then found `_map_needs_initial_render` set, chose the screen factor (16 on this canvas) and rendered every tile again. By then the full-resolution channels (25 × 6 × 8 MiB = 1.2 GB) had overflowed the 512 MiB channel cache, so each one was decoded a second time.
- **Activation renders once, at the screen factor, when the widget is on screen.**
  After the canvas is set up, `_activate_map_mode` calls `adopt_current_view` (Fix 4), which sets the screen factor and records the whole-map view, and clears `_map_needs_initial_render`. The caller's render is then the only one; the next draw sees an unchanged view. Before the widget is shown, the flag is kept, since that render never reaches the browser and the first draw has to render.
- **Statistics from the decoded copy.**
  `_update_stats_for_visited_tile` ran `da.nanpercentile`/`da.nanmax` on the lazy channel, which reads the file again (its docstring claimed no I/O). `ChannelLevelCache.peek_full` returns the raster the render has just decoded, and `data_loader._channel_stat_values` reduces it with NumPy: the same 99.9th percentile and max, checked against the Dask tasks in the tests. The channels of a tile are reduced on a shared thread pool (`_channel_thread_pool`), since `np.percentile` releases the GIL (6 channels: 89 ms serial, 24 ms pooled; the Dask path took 53 ms without the read). A channel the cache did not hold keeps the Dask path.
- **Channels of a tile decoded in parallel.**
  `ChannelLevelCache.prefetch` decodes, through the same pool, exactly the channels the following `read_region` calls would decode at full resolution, and stores them; `_compose_fov_image` calls it before `render_fov_to_array`. The gain is modest (6 channels: ~65 → ~46 ms per tile), as decoding these TIFFs holds the GIL part of the time. It applies to every first read of a FOV or tile, not only to activation.
- **Ruled out:** skipping or approximating the per-tile statistics. They set the contrast slider ranges, and a strided or histogram estimate would give different values from single-FOV mode; reducing the decoded copy keeps them identical.

---

### **Results**

`map_run.py`, same machine and dataset, before (7db8e87) → after:

| Action | Before | After |
| --- | --- | --- |
| Map activation (first visit of every tile) | 5.8 s | 1.9 s |
| Channel decodes during activation | 300 | 150 |
| First zoom-in 4× (new tiles) | 410–620 ms | 315–480 ms |
| Zoom in 2× after zooming out | 47 ms | 300 ms |
| Zoom out to the whole map; contrast change | 40–48 ms | 40–46 ms (unchanged) |

The zoom-in 2× was fast before only because activation's wasted first render had left factor-8 tiles in the tile cache; it now costs what any first visit of tiles at a new factor costs. Of the remaining ~1.9 s, about 1.1 s is decoding and downsampling the 150 channels and about 0.6 s the statistics.

---

### **Files Changed**
- `ueler/viewer/main_viewer.py` — `_activate_map_mode` adopts the screen view when on screen; `_update_stats_for_visited_tile` reduces the decoded copy; `_channel_thread_pool`; prefetch in `_compose_fov_image`.
- `ueler/rendering/channel_cache.py` — `prefetch`, `peek_full`.
- `ueler/data_loader.py` — `_channel_stat_values`.
- `tests/test_channel_level_cache.py` — prefetch and `peek_full` tests (4); `tests/test_map_activation_reads.py` — new (5 tests).

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1522 tests, OK
