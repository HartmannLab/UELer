# Display Settings

UELer opens with defaults that suit a MIBI dataset on a wide screen. If your data comes from somewhere else, a few of those defaults are quietly wrong for you — most notably the pixel size, which drives the scale bar in the viewer *and* in every exported image.

This page is a setup walkthrough, not a reference: it tells you what to set, in what order, and why. For the control-by-control map of the left panel, see [User Interface](user-interface.md).

---

## Before You Launch

Three things about the environment, settled once and then forgotten.

**1. Run `%matplotlib widget` once per kernel session**, before you build the viewer:

```python
import ueler
%matplotlib widget

viewer = ueler.run_viewer(base_folder)
```

Without it the canvas renders as a static PNG — no zoom, no pan, no lasso. If you restart the kernel, run it again.

**2. Any Jupyter front end works.** JupyterLab, the classic Notebook, and VS Code all render the full UI, interactive scatter included. You do not need to set `UELER_SCATTER_BACKEND` — the interactive `jupyter-scatter` widget is the default everywhere.

!!! note "Old notebooks may still set `UELER_SCATTER_BACKEND=widget`"
    Earlier versions of UELer dropped to a static Matplotlib scatter under VS Code, and the variable was the way to force the interactive one. That fallback is gone. Delete the line — it does nothing now. The static renderer survives as an opt-out (`UELER_SCATTER_BACKEND=static`), which you are unlikely to want.

**3. Give the notebook some width.** The left control panel is a fixed 350 px column. On a narrow window, or at high browser zoom, the image canvas gets whatever is left over and can end up uncomfortably small. Zooming the browser *out* one step (`Ctrl`/`Cmd` + `-`) buys the canvas real estate without shrinking the controls' legibility much.

---

## The Four Things to Set First

Do these in order the first time you open a dataset. Two of the four are wrong by default for most non-MIBI data.

### 1. Pixel size — set it, or the scale bar lies

**Advanced Settings → Pixel Size (nm):** defaults to **390**, the MIBI pixel pitch. It is applied to *every* dataset, whatever the source, with no warning. The scale bar in the viewer and in every batch-exported image is computed from this number, so if it is wrong, every figure you export carries a wrong scale bar.

| Platform | Typical pixel size |
|---|---|
| MIBI | 390 nm (the default) |
| IMC | 1000 nm (1 µm) |
| Anything else | Read it off your acquisition metadata |

If you don't know the value, set it to **0** — the scale bar is then omitted entirely, which is honest. A wrong scale bar is worse than none.

### 2. Cache size — match it to your memory

**Cache Size:** (top of the left panel, default **100**) is the number of FOVs kept in memory at once. Each cached FOV holds the channels you have actually opened, so the cost is roughly:

```
FOV pixels × channels opened × bytes per pixel × cached FOVs
```

A 1024 × 1024 uint16 FOV with 6 channels is about 12 MB, so 100 of them is ~1.2 GB — fine on a workstation. The same arithmetic on 4000 × 4000 tiles gives ~190 MB per FOV, and the default cache will exhaust most machines. Lower it to 10–20 for large images; raise it if you flip back and forth between the same handful of FOVs and have memory to spare.

### 3. Downsample — leave it on unless you know why not

**Advanced Settings → Downsample** is **on** by default. With it on, UELer decimates the image so the drawn view never exceeds **2048 px** on its longest side, picking the smallest power-of-two factor that achieves this. As you zoom in the visible region shrinks, the factor drops back toward 1, and you get full-resolution pixels again — so you lose nothing but the detail you could not have resolved on screen anyway.

Two consequences worth knowing:

- On images at or below 2048 px (a standard 1024² MIBI FOV, say) the factor is 1 and the toggle changes nothing.
- Turning it **off** forces native resolution at every zoom level. On large OME-TIFFs and in [map mode](map-mode.md) this is the single fastest way to make the viewer feel sluggish.

Turn it off only when you need to be certain that what you are looking at is not an artefact of decimation.

### 4. Channels — pick them, then order them

Add channels in the **Channels:** field; each becomes a chip. Drag a chip by its `⋮⋮` grip, or click it and press ← / →, to reorder.

!!! info "Ordering is for you, not for the renderer"
    Channels are composited **additively** — each channel's colour is scaled by its normalised intensity and summed, then clipped. The result does not depend on the order, so no channel "covers" another. What the chip order *does* control is the order of the per-channel control rows below, the channel legend, and the panes in the [channel grid view](basic-usage.md#7-use-the-channel-grid-view). Put the channels you keep adjusting at the top.

    Because the blend is additive, several bright channels in similar colours saturate to white where they overlap. If a composite looks washed out, that is usually the reason — lower a **Max** or two rather than reaching for the visibility checkboxes.

---

## Getting Contrast Right

When a channel is first shown, UELer computes its display range from the data: **Min** starts at 0, and **Max** starts at the channel's **99.9th percentile**, not its maximum. That is a deliberate auto-contrast — the brightest 0.1 % of pixels (hot pixels, aggregates, detector artefacts) are allowed to saturate so the rest of the dynamic range is visible.

Knowing that, the two common complaints have specific fixes:

- **The channel looks all-black.** The percentile is being dragged up by a bright artefact, or the marker is genuinely dim. Lower **Max**. This is by far the more common case.
- **The channel looks all-white.** Real signal fills most of the frame and the percentile sits near the top of it. Raise **Min** to push the background down, then lower **Max** to taste.

!!! warning "The Max slider stops at the auto value"
    The slider's upper bound *is* the percentile-derived maximum, so you cannot raise **Max** above it to un-saturate that top 0.1 %. The bound is per channel and only ever grows (see below) — it is not a per-FOV reset.

**The range grows as you look around.** The stored maximum for a channel is the running maximum over everything UELer has computed so far. Open a brighter FOV, or pan into a brighter tile in map mode — where per-tile statistics are computed lazily, on first render — and the slider's upper bound can grow mid-session. Your **Min** / **Max** values are preserved; only the room above them changes. This is expected, and it is why contrast set on the first FOV of a map is worth re-checking after you have panned across it.

Once a channel/colour/contrast combination works, save it as a **marker set** (below) — that is the only way to get it back exactly.

---

## Overlays

Overlays are drawn over the finished channel composite, in this order: annotations first, then masks.

- **Masks** — enable a mask with its checkbox in the left panel's **Masks** section and pick a colour from the **Mask &lt;name&gt;** dropdown. **Mask outline px:** sets the outline thickness. This is the quick look.
- **Mask painter** (right panel) — per-class colours, fill versus outline, per-class opacity, and saved `.maskcolors.json` palettes. Use this when a single mask colour is not enough. Requires a cell table.
- **Pixel annotations** — **Show annotation**, pick one from **Annotation:**, and set **Fill alpha:** for transparency. **Edit palette…** customises per-class colours and labels.
- **No image (masks only)** — hides the channel composite and draws overlays on black. Good for checking segmentation boundaries, and cheaper to render since no channel compositing happens.
- **Channel grid view** — one labelled, synchronised pane per visible channel instead of one composite. The honest way to judge whether a marker is really co-expressed or just additively blended into the same white.

---

## Saving and Reusing a View

Two mechanisms, with different jobs.

**Marker sets** are the deliberate, portable unit: a named combination of channels, their colours, and their contrast ranges. Type a name in **Set Name:**, click **Save Marker Set**, and reload it later from the **Marker Set:** dropdown. Use these for the views you want to reproduce — a figure panel, a QC pass, a set of markers you always check first.

**Widget state** is the automatic one. UELer writes `<base_folder>/.UELer/widget_states.json` whenever a control changes, and reloads it when you next open the same `base_folder`, restoring the FOV, channel selection, contrast, overlays, advanced settings, and accordion positions. Plugins keep their own state alongside it, one file each, named `<Plugin Name>_widget_states.json`.

!!! tip "Resetting a bad restored state"
    Because restoration is automatic and covers nearly everything, a session that ended in a strange state comes back in that same strange state. To start clean, close the viewer and delete `<base_folder>/.UELer/widget_states.json`. Your ROIs, palettes, marker sets, and export configs live in separate files under `.UELer/` and are not affected — see [Where does UELer store my work?](../faq.md#where-does-ueler-store-my-work).

---

## When It Feels Slow

In rough order of effect:

1. **Check Downsample is on.** Off plus a large image means every pan re-reads full-resolution data.
2. **Lower Cache Size.** If the kernel is swapping, nothing else you do will help. Use the arithmetic above to sanity-check the cache against your RAM.
3. **Open fewer channels.** Compositing cost is linear in the number of visible channels, and every channel you open is held in memory for each cached FOV. Note that unchecking a channel's visibility skips its compositing but does not release it: loaded channel data stays with its cached FOV until that FOV is evicted from the cache.
4. **In map mode, lower the tile limit.** UELer draws at most 80 uncached tiles per frame, nearest the viewport first. On a slow machine, fewer is smoother:

    ```python
    viewer._map_render_tile_limit = 40
    ```

5. **Expect the first pan across a map to be slower.** Channel statistics are computed per tile on first render, deliberately, rather than up front — computing them for the whole map at launch is what would otherwise exhaust memory on large datasets. It settles once you have covered the ground.

---

## Next Steps

- Capture what you are looking at as a [Region of Interest](roi-manager.md).
- Push the same settings across many FOVs with [Batch Export](export.md).
- Stitch FOVs into one canvas with [Map Mode](map-mode.md).
