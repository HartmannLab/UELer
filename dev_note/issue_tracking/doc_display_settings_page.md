# Doc site — "Display Settings" page

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
- **Downsample defaults to ON** (`ueler/viewer/ui_components.py:714`) and only has an effect for OME-TIFF sources (`ueler/viewer/main_viewer.py:1986` returns early unless `_fov_mode == "ome-tiff"`). Neither the default nor the scope is stated.
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
   3. **Downsample** — on by default, OME-TIFF only, what it trades.
   4. **Channel order** — chips define compositing order, so the last chip paints on top.
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
