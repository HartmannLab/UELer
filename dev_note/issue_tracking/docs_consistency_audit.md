# Docs ↔ software consistency audit (2026-08-20)

**Mode:** ad-hoc request ("ensure the mkdocs pages are consistent with the software"). No GitHub issue; no entry in `dev_note/github_issues.md`.

---

## Problem

The site built clean under `mkdocs build --strict` and still asserted things that were not true. Strict mode validates that the site is well *formed* — every nav entry resolves, every internal link lands — and says nothing about whether it is *correct*. Nothing in the repo checked correctness, and nothing in CI built the docs strictly either: `docs.yml` ran `mike deploy` directly, which does not build strictly, so even the formedness guarantee was accidental rather than enforced.

The drift was concentrated exactly where you would predict from `git log`:

| Docs area | Last touched before this pass | Code commits since |
|---|---|---|
| `develop-notes/{map-mode, export-pipeline, roi-gallery, heatmap, ome-tiff, index}` | 2026-04-16 | ~50 |
| `tutorials/{export, map-mode, roi-manager}` | 2026-07-10 | #121, #125–#135 |
| `tutorials/{cell-table, user-interface}` | 2026-07-31 | #131–#135 |
| `installation`, `index`, `packaging` | 2026-08-19/20 | current |

---

## Approach: split the claims by how they can be verified

The expensive part of a docs audit is a human (or a model) reading every page against the code. So everything decidable by a program was pushed into a program — which also converts a one-off cleanup into a standing guard.

**Tier A — machine-checkable invariants.** `tools/check_docs_consistency.py`, gated by `tests/test_docs_consistency.py` and `make check-docs`. Checks: distribution name in install commands; the stated Python range against `requires-python`; documented extras against `[project.optional-dependencies]`; every backticked repo path and GitHub blob link; every `make <target>`; every `UELER_*` / `ENABLE_MAP_MODE` variable against the code that reads it; code symbols named in the developer notes; UI labels against the real `displayed_name` / `description=` / `set_title` strings; Python examples parse and import only names `ueler/__init__.py` exports; nav coverage both ways; and each developer note's `> Source:` link.

**Tier B — anchored prose review.** Pages reviewed in churn order. The review input per page was bounded: the page, its `dev_note/topic_*.md` counterpart, and `git log -p` for the relevant code directory since the page's last commit. Important caveat found on the way: `topic_map_mode_spatial.md`, `topic_export_pipeline.md` and `topic_ome_tiff_loading.md` are themselves *older* (2026-03-31) than the docs pages they source, so for those three the dev_note is not an authority and the review had to go to code. One dev_note was itself wrong and was corrected (see below).

**Tier C — the two content gaps.** `topic_plugin_development.md` (568 lines, current) and `topic_mask_rendering_highlighting_coloring.md` (303 lines) had no published counterpart at all, leaving plugin authoring and mask painting undocumented on the site.

### Why the checker parses instead of importing

`plugin_display_names()` and `ui_label_vocabulary()` read the plugin modules with `ast`, not `import`. Importing a plugin drags in ipywidgets, matplotlib, anywidget and the rest of the stack, which is far too heavy for a docs check and would make the gate fail for reasons unrelated to the docs. `load_project_facts()` parses `pyproject.toml` with regexes for the same reason `tools/check_stable_rehearsal.py` does: `requires-python` still admits 3.10, where `tomllib` does not exist, and three scalars plus a table's key list do not justify a dependency.

### Two false-positive classes the checker has to model

Both were found by running it, not by reasoning about it:

- **Removal narration.** The developer notes deliberately name things that are *gone* — "the `_AliasModuleFinder` / `_PrefixAliasFinder` layer in `ueler/_compat.py` … has been deleted" is the point of that paragraph. `REMOVAL_NARRATION` skips symbol and path claims on a line carrying removal language, which keeps those paragraphs writable without an author-maintained allowlist that would itself rot.
- **Real UI strings that resemble a plugin name.** An early version flagged "**Cell gallery**" as a miscased "Gallery". It is not: `description="Cell gallery"` is a genuine checkbox in `_chart_common.build_link_checkboxes()`. The fix generalised the rule from *plugin labels* to *the whole UI vocabulary* — every `description=`, `set_title` and `displayed_name` string — and flags a bolded doc term only when it is a **case variant** of a real label. That is the precise statement of the actual requirement ("name UI elements as the UI names them") rather than a heuristic approximating it.

---

## What was wrong

### Fixed by the checker's own findings

- `develop-notes/export-pipeline.md` cited `scale_bar_helper.py`; the module is `ueler/viewer/scale_bar.py` and has been for months.
- `develop-notes/map-mode.md` cited a `_RENDER_TILE_LIMIT` that has never existed. The real mechanism is `_BASE_UNCACHED_LIMIT`, read from `viewer._map_render_tile_limit` (default 80), multiplied by `max(1, ds_factor)`, and applied to *uncached* tiles only.
- `develop-notes/packaging.md` claimed "Supported Python: 3.10–3.11 … the classifiers list the same two versions". `requires-python` permits 3.12 and `pyproject.toml` now carries the 3.12 classifier, so both halves were false.
- `index.md` wrote "**ROI Manager**" for a plugin the UI calls "ROI manager".

### Fixed by review

- **Plugin placement was wrong on four pages (#121).** `chart.py` and `heatmap.py` set `footer_only = True`, which means they are skipped when the accordion is assembled and live *only* in the footer. `user-interface.md` listed both as right-panel accordion entries, and it, `clustering-annotation.md` and `scatter-histogram.md` all documented ways of moving them — an "enable **Horizontal layout**" toggle and an "automatically when more than one scatter is active" behaviour — neither of which exists any more. The current `doc/GUI_preview.png` confirms it: ten accordion entries, no Scatter plot or Heatmap among them, and two footer tabs.
- **Mask controls are their own accordion section, not part of Channels.** `_rebuild_control_sections` appends a **Masks** section when `masks_available` and a **Pixel annotations** section when `annotations_available`. `user-interface.md`, `basic-usage.md` and `display-settings.md` all told the reader to find mask checkboxes inside **Channels**. The conditional construction is worth stating too — a missing section is an unreadable folder, not a hidden setting.
- **The channel picker description predated #125/#126.** `basic-usage.md` described a tag input and asserted "not a scrolling list — there is no Shift- or Ctrl-click range selection", contradicting both the code and `user-interface.md` on the same site. `ChannelPickerWidget` is a filterable scrolling list with **Select all shown** / **Clear** and keyboard navigation, and chips drag to reorder.
- **"Follow main viewer" (#135) was undocumented.** It is the *counterpart* of the **Main viewer** checkbox — one pushes the plot's selection outward, the other pulls the image's selection in — and getting them confused is the obvious way to conclude linking is broken. Now a direction table in `scatter-histogram.md`, with the **Trace** one-shot variant, and reflected in `cell-table.md`'s linked-selection diagram and `clustering-annotation.md`.
- **`develop-notes/heatmap.md` described the Cell Annotation plugin as "specified but not yet implemented as a plugin".** It ships as `CellAnnotationPlugin`, with `CheckpointStore`, `parent_id`/`op` lineage and an anywidget tree browser — and has had its own tutorial page for months. The same page called the heatmap a "cell-by-marker matrix"; it is cluster × marker of grouped medians (`df.groupby(cluster_column)[marker_columns].median()`), which is what makes it usable on a large table.
- **`develop-notes/roi-gallery.md` listed 7 ROI columns including a `palette` that does not exist.** `ROI_COLUMNS` has 24, and the interesting ones were the missing `roi_kind` / `geometry` pair: shape ROIs (#c373352) were entirely undocumented, on the developer page and in `tutorials/roi-manager.md` alike, despite a full Draw / Edit / Finish / Undo / Redo / Save shape UI that also reports a physical length.
- **`ENABLE_MAP_MODE` is read at module import time**, not at launch. `_MAP_MODE_FLAG` is a module-level assignment in `main_viewer.py`, captured into `self._map_mode_enabled` at construction, so setting the variable after `import ueler` does nothing even if it is before `run_viewer`. Both `map-mode.md` and `faq.md` said "before launching", which is the failure mode rather than the fix.
- **The Batch export **Cells:** picker silently shows at most 500 rows** (`filtered.head(self._CELL_OPTION_LIMIT)`), with no indication of truncation. Documented, because the consequence is that the query filter is how you choose cells rather than a pre-filter you scroll through.
- **`packaging.md`'s status block contradicted its own open items** — "Remaining: Gate D … then publish `0.5.0-alpha`" against a later line recording that `0.5.0-alpha2` published on 2026-08-19 — and its package tree omitted `data_loader.py`, `cell_table.py`, `bia_loader.py`, `constants.py`, `rendering/` and `export/` while listing `image_utils.py`. Test count refreshed to the current 1109.
- **`develop-notes/ome-tiff.md`** gained the `residual` factor that `_select_level` returns alongside the pyramid level (the level does the cheap part by reading fewer bytes, the residual makes up the difference), and the "incompatible keyframe" fallback.
- **The docs-site screenshot was the pre-accordion one.** `docs/GUI_preview.png` was byte-identical to the *old* `doc/GUI_preview.png`; the v0.5.0-rc2 log recorded this and left it "for the next docs pass". This is that pass — the two are now the same file.

### A dev_note corrected

`dev_note/topic_mask_rendering_highlighting_coloring.md` named `GridChannelDisplay.update_mask_highlights`. No such method exists; the real ones are `_update_grid_patches()` and `clear_patches()`. Found by the checker while validating the new page drafted *from* that note — a useful demonstration that the tool catches errors inherited from the source notes, not just editing slips. Corrected in place with a dated parenthetical rather than silently, per the no-rewriting-history rule.

---

## Mermaid support

Enabled while writing the Tier C pages, since both are structural and diagrams carry that better than prose. Material for MkDocs ships the integration itself; routing the fence to `pymdownx.superfences` with `format: !!python/name:pymdownx.superfences.fence_code_format` is the whole change. No plugin, no `<script>` tag — either one fights the built-in integration, which is theme-aware and activates only on pages containing a diagram.

**Caveat recorded in `mkdocs.yml`:** Material's loader fetches mermaid 11 from `unpkg.com` at *runtime*, so a diagram is blank for a reader without access to that host. Diagrams therefore explain; they are never the only place a fact is stated.

---

## Deliberately not done

- **The 3.12 CI asymmetry.** `pyproject.toml` advertises `Programming Language :: Python :: 3.12` while `tests.yml` still runs the 3.12 leg as `experimental: true` / `continue-on-error`, and the comment above that leg still reads as though the classifier were pending. So UELer claims support that no blocking leg defends. Either drop `experimental` or drop the classifier — a policy call about what to promise, not a docs fix. Recorded as an open item in `packaging.md` and left for the maintainer.
- **Page titles stay title-cased.** `tutorials/export.md` is titled "Batch Export" for a plugin called "Batch export", consistently with every other nav entry (Basic Usage, User Interface, Map Mode…). The convention is: **titles and nav are title case, inline references to a UI element are exact.** The checker enforces only the latter, by looking at bolded terms rather than headings or link text.

---

## Verification

```bash
make check-docs                  # mkdocs build --strict, then the consistency checker
python -m unittest discover tests
```

- `mkdocs build --strict` — clean.
- `tools/check_docs_consistency.py` — "Docs are consistent with the software."
- Full suite — **1109 tests, OK** (1100 before, plus the 9 new).
- Mermaid verified in the built HTML: `<div class="mermaid">` on both new pages, zero mermaid references on a page without a diagram.
