# Regions of Interest

The **ROI manager** plugin lets you capture, tag, and revisit regions of interest. ROIs persist
across sessions and work **without a cell table**, so this is part of the essentials.

The plugin has two tabs: **ROI browser** (find and revisit ROIs) and **ROI editor** (capture and
edit them).

---

## Capturing an ROI

1. Pan and zoom the main viewer to frame the region you want.
2. Open the **ROI editor** tab.
3. Click **Capture view**.

The capture records the current viewport, the active FOV (or map), your marker set choice, and the
current mask/annotation settings, so re-centering later can restore the same look.

Before or after capturing you can set:

- **Name:** — an optional custom name for the ROI.
- **Add tag:** / **Tags:** — free-form tags (type or pick; new tags are allowed).
- **Comment:** — a free-text note.
- **Marker set:** — which marker set to associate (**Current set**, **None**, or a saved set).

Use **Update** to save edits to the selected ROI, **Delete** to remove it, and **Center** /
**Center with preset** to jump the viewer back to a saved ROI (the latter also re-applies the ROI's
saved marker/mask/annotation presets).

!!! tip "Only current FOV"
    The editor's **Only current FOV** checkbox (on by default) filters the saved-ROI dropdown to the
    active FOV.

---

## Browsing ROIs

The **ROI browser** tab shows a paged thumbnail gallery (12 per page) of your ROIs. Click a tile to
jump to that ROI; with **Apply saved preset on click** enabled, clicking also restores the ROI's
saved presets.

Filter the gallery with:

- **Tags:** and a **Tag logic** toggle — **All tags (AND)** or **Any tag (OR)**.
- **FOVs:** — restrict to specific FOVs.
- **Only current FOV** — restrict to the active FOV (off by default here).

The browser has **simple** and **advanced** filter sub-tabs. The advanced tab is an expression editor:

### Expression-based selection

In the advanced sub-tab, build a boolean tag expression using `&`, `|`, `!`, and parentheses — for
example:

```
(good & figure1) & !excluded
```

Insert operators and tag names with the provided buttons, then click **Apply**. Leave the expression
blank to fall back to the simple tag filter.

---

## Where ROIs are stored

ROIs are saved to:

```
<base_folder>/.UELer/roi_manager.csv
```

You can also **Export** / **Import** the ROI table from the editor tab (paths are resolved relative to
the base folder).

---

## ROIs and Map Mode

ROIs captured in [map mode](map-mode.md) are stored with an empty `fov` and a populated `map_id`, and
are shown with a `[MAP:<id>]` location label. The FOV-scope filters are disabled while map mode is
active.

## Exporting ROI images

To render ROIs to image files (single-FOV or map-mode), use the [Batch Export](export.md) plugin's
**ROIs** tab.
