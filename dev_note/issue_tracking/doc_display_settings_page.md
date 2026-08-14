# Doc site — "Display Settings" page

> **Implemented.** `docs/tutorials/display-settings.md` is written and linked; see [Outcome](#outcome) at the bottom for what changed relative to this plan.

**Mode:** Random Request (chat-only planning; no `dev_note/github_issues.md` entry, but `doc/log.md` and `README.md` still get updated on completion).

**Goal:** give the docs site one page that answers *"how do I set the viewer up so it shows my data properly?"* — currently there is no such page, and the answer is scattered across five documents plus the FAQ.

---

## Survey — what exists today

### The site

MkDocs Material, config in `mkdocs.yml`, versioned with `mike`. The `nav:` block is explicit, so a new page is invisible until it is listed there. User-facing content lives in `docs/tutorials/`; `docs/develop-notes/` is internal architecture and is not the right home for this.

Current Essentials order: Basic Usage → User Interface → Regions of Interest → Map Mode → Batch Export.

### Where viewing settings are documented now

| Setting | Documented in |
|---|---|
| Channel selection and chip order (chip order = compositing order) | `docs/tutorials/basic-usage.md` §3, `docs/tutorials/user-interface.md` "Channels" |
| Per-channel colour, visibility, Min/Max contrast | `docs/tutorials/basic-usage.md` §5, `docs/tutorials/user-interface.md` |
| Marker sets | `docs/tutorials/basic-usage.md` §4, `docs/tutorials/user-interface.md` "Marker Sets" |
| Mask overlay, outline px, mask painter | `docs/tutorials/basic-usage.md` §6 |
| Annotation fill alpha and palette | `docs/tutorials/basic-usage.md` §6 |
| Cache size | `docs/tutorials/basic-usage.md` §2 |
| Pixel size → scale bar | `docs/tutorials/user-interface.md` "Advanced Settings", `docs/faq.md` "scale bar is missing" |
| Downsample toggle | one clause in `docs/tutorials/user-interface.md` — named, never explained |
| `%matplotlib widget`, scatter backend, map-mode flag | `docs/getting-started.md`, `docs/faq.md` "Viewer & Widgets" (the scatter-backend answers were rewritten under #122 — the interactive backend is now the default everywhere) |
| `.UELer/` persisted state | `docs/faq.md` "Where does UELer store my work?" |

Nothing is *missing* as reference material. What is missing is an ordered, opinionated setup path, and the launch-environment half of the answer only exists as FAQ troubleshooting — i.e. it is written for someone whose viewer is already broken, not for someone setting it up.

### Gaps found in the code that the docs do not cover

- **Pixel size defaults to `390` nm** (`ueler/viewer/ui_components.py:707`). That is a MIBI value applied silently to every dataset. Non-MIBI users get a wrong scale bar in the viewer *and* in every export, with no warning anywhere. This is the single highest-value thing the new page can say.
- **Contrast auto-scaling uses the 99.9th percentile**, not the data max (`ueler/data_loader.py:132`, clipped to the dtype limit at `ueler/data_loader.py:151`). The FAQ's "all-white or all-black" answer just says "move the sliders" — knowing where the default came from is what makes the fix intentional rather than trial and error.
- **Contrast slider ranges grow while you pan in map mode** (`_map_lazy_stats_enabled`, `ueler/viewer/main_viewer.py:333`; stats merge into `channel_max_values` per tile). Undocumented and surprising.
- **Downsample defaults to ON** (`ueler/viewer/ui_components.py:714`) and only has an effect for OME-TIFF sources (`ueler/viewer/main_viewer.py:1986` returns early unless `_fov_mode == "ome-tiff"`). Neither the default nor the scope is stated. *(The scope claim is wrong — that early return only skips the pyramid-level push; the factor itself is applied for every source. See [Outcome](#outcome).)*
- **`widget_states.json` restores the whole UI per `base_folder`** on launch (`ueler/viewer/main_viewer.py:594`) and is written back at `ueler/viewer/main_viewer.py:5392`. The FAQ lists the file but not *when* it is written, *what* it covers, or how to reset a bad restored state (delete the file). Plugins keep their own `<plugin>_widget_states.json` (`ueler/viewer/plugin/plugin_base.py:50`).
- **The control column is a fixed 6in / 350px** (`ueler/viewer/ui_components.py:146`, `:384`), which is why browser zoom and narrow windows matter.

### Do not document

`dev_note/issue_tracking/issue104_fullscreen_mode.md` describes a full-screen toggle in implementation-level detail, but no `fullscreen` / `requestFullscreen` / `ueler-fs-active` code exists anywhere under `ueler/` on this branch. Confirm it actually ships before it goes in the page.

---

## Plan

### New page

`docs/tutorials/display-settings.md`, titled **Display Settings**, placed in *Essentials* between **User Interface** and **Regions of Interest** — after the reader knows the layout, before the feature tutorials.

Style: an opinionated walkthrough ("do this, in this order"), not another reference table. Reference detail stays in `user-interface.md`; this page links to it rather than restating it.

### Outline

1. **Before you launch** — `%matplotlib widget` once per kernel; JupyterLab vs VS Code vs classic Notebook; browser zoom and window width against the fixed 350px control column. Absorbs the "Viewer & Widgets" FAQ answers as a setup step, leaving the FAQ entries as short pointers. Note that as of #122 no backend environment variable is needed — the interactive scatter is the default everywhere, and `UELER_SCATTER_BACKEND=static` is an opt-out only.
2. **The four things to set first** — a checklist, in order:
   1. **Pixel size (nm)** — change it off the 390 nm default, or the scale bar lies.
   2. **Cache size** — what 100 costs in memory, when to lower it.
   3. **Downsample** — on by default, OME-TIFF only, what it trades. *(Corrected during writing — it applies to every source; see [Outcome](#outcome).)*
   4. **Channel order** — chips define compositing order, so the last chip paints on top. *(Wrong — compositing is additive and order-independent; see [Outcome](#outcome).)*
3. **Getting contrast right** — the 99.9th-percentile default and what it implies for dim and saturated channels; when to override Min vs Max; why the slider range grows as you pan in map mode. Takes over the FAQ's all-white/all-black answer.
4. **Overlays** — mask checkbox + outline px for a quick look, mask painter for per-class fill/outline/opacity; annotation fill alpha; "No image (masks only)"; channel grid view for side-by-side comparison.
5. **Saving and reusing a view** — marker sets as the reproducible unit (channels + colours + contrast); what `widget_states.json` restores automatically and when it is written; per-plugin state files; how to reset (delete `<base_folder>/.UELer/widget_states.json`).
6. **When it is slow** — downsample, cache size, `viewer._map_render_tile_limit`, and what map mode's lazy stats mean for the first pan.

### Edits to existing files

- `mkdocs.yml` — add `Display Settings: tutorials/display-settings.md` to the Essentials nav block.
- `docs/tutorials/index.md` — add a row to the Part 1 · Essentials table.
- `docs/tutorials/basic-usage.md` — keep §5 short and link out to the new page for the percentile/auto-contrast explanation.
- `docs/faq.md` — replace the duplicated bodies of "Channel images appear all-white or all-black" and "Nothing renders / the widgets don't show up" with one-line answers that link to the new page. Keep the `.UELer/` table where it is; the new page links *to* it.
- `docs/getting-started.md` — one "Next steps" link to the new page.

### Open judgement calls (need your input before writing)

- **Recommended pixel sizes** to cite as examples — MIBI 390 nm is in the code; is there a house IMC value worth naming, or should the page just say "look it up for your platform"?
- **Cache size guidance** — is there a rule of thumb you use (e.g. FOV pixel count × channels × dtype), or should the page stay qualitative ("lower it if the kernel is swapping")?
- **Screenshots** — the site currently has exactly one image (`docs/GUI_preview.png`). Worth adding cropped shots of the Channels accordion and Advanced Settings, or keep the page text-only for now?

### Verification

Docs-only change, no tests to run. Build the site locally and check the nav, the new page, and every cross-link:

```bash
mkdocs build --strict
```

`--strict` turns broken internal links into build failures, which is the check that matters here.

### Documentation updates on completion

- `doc/log.md` — new entry at the top of the current version section.
- `README.md` — "New Update" summary under the current major version.
- This file — mark the plan as implemented.

---

## Outcome

`docs/tutorials/display-settings.md` was written to the outline above, with the six sections intact. `mkdocs build --strict` passes, and every deep anchor used across the site (`#getting-contrast-right`, `#before-you-launch`, `#7-use-the-channel-grid-view`, `#where-does-ueler-store-my-work`) was checked to resolve in the built HTML — `--strict` validates page links but not fragments.

### Two claims in this plan were wrong and did not reach the page

- **"Chip order = compositing order, so the last chip paints on top" is false.** `_composite_channels` does `composite += normalised[..., None] * colour` for each selected channel and clips once at the end (`ueler/rendering/engine.py:409-421`); there is no second compositing path. The blend is **additive and therefore order-independent** — no channel occludes another. The chip order controls the order of the per-channel control rows, the legend, and the channel-grid panes. The page says this explicitly, and adds the practical corollary (several bright channels in similar colours saturate to white where they overlap, which is what a "washed out" composite usually is).
- **"Downsample only has an effect for OME-TIFF sources" is too narrow.** The factor is recomputed on every draw from the *viewport* size for every source (`calculate_downsample_factor(range_x, range_y, ignore_zoom=not checkbox.value, max_dimension=2048)`, `ueler/viewer/image_display.py:288-293`), and unchecking the box passes `ignore_zoom=True`, pinning the factor at 1 everywhere. What is OME-TIFF-specific is only the *push* of the chosen factor into the pyramid wrapper so a lower resolution level is read instead of decimating after the read (`main_viewer.py:1983-1989`). The page documents the viewport rule, the 2048 px threshold, and that the toggle is a no-op on images at or below it.

### Open judgement calls — resolved

- **Pixel sizes:** the page names MIBI 390 nm (the code default) and IMC 1000 nm as a table, then says to read the value off the acquisition metadata for anything else. It recommends **0** — which omits the scale bar — over guessing.
- **Cache size:** arithmetic the reader applies themselves (`FOV pixels × channels opened × bytes per pixel × cached FOVs`), worked through for a 1024² uint16 FOV (~12 MB → ~1.2 GB at the default 100) and 4000² tiles (~190 MB each), rather than a house number.
- **Screenshots:** deferred, page is text-only. Cropped shots of the Channels accordion and Advanced Settings would help this page more than any other on the site; worth adding once that layout settles.

### Still open

- The **fullscreen toggle** warning above stands — nothing under `ueler/` implements it on this branch, and it is not mentioned anywhere in the new page.
