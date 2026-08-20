# Regions of Interest

The **ROI manager** plugin lets you capture, draw, tag, and revisit regions of interest. ROIs persist
across sessions and work **without a cell table**, so this is part of the essentials.

The plugin has two tabs: **ROI browser** (find and revisit ROIs) and **ROI editor** (capture and
edit them).

!!! info "Two kinds of ROI"
    UELer stores two different things under one name, and the difference matters when you filter or
    export:

    - a **view** ROI is a saved viewport — a bookmark of where you were looking, at what zoom, with
      which markers and overlays;
    - a **shape** ROI is a polyline or polygon you drew on the image, which also carries its
      geometry and a physical length.

    Both live in the same table and the same browser. Everything below about tags, comments,
    filtering and export applies to both.

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

## Drawing a shape ROI

The editor tab also has a shape block for drawing directly on the image — use it to mark a boundary,
trace a structure, or measure a distance.

1. Click **Draw**, then click on the image to place vertices. The summary line under the buttons
   tracks the shape as you go.
2. Tick **Closed shape (polygon)** before finishing if you want the shape closed; leave it clear for
   an open polyline.
3. Correct as you go with **Undo** / **Redo**, or abandon the whole shape with **Cancel**.
4. Click **Finish** to end the shape, then **Save shape** to store it.

A saved shape is an ROI like any other: give it a **Name:**, **Tags:** and a **Comment:**, and it
appears in the browser gallery alongside your captured views.

- **Show shapes on canvas** (on by default) draws the saved shapes over the image. Untick it to get
  an unobstructed view without deleting anything.
- **Edit** loads the selected shape ROI back into the editor so you can move or add vertices and save
  it again.

!!! tip "Shapes measure themselves"
    The summary line reports the shape's length in µm as well as pixels, computed from the **Pixel
    Size (nm):** value in Advanced Settings. If that value is wrong, so is the length — see
    [Display Settings](display-settings.md#1-pixel-size-set-it-or-the-scale-bar-lies).

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
