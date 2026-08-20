# Basic Usage

This tutorial covers the essential steps for launching UELer and navigating image data. None of it
requires a cell table.

---

## 1. Launch the Viewer

After [configuring your paths](../getting-started.md), run the launch cells in
`script/run_ueler.ipynb`. The viewer appears inline in the notebook output.

---

## 2. Select an FOV

Use the **Select Image:** dropdown in the left panel to choose a Field of View (FOV). UELer loads
image data on demand and caches recently accessed FOVs in memory.

The **Cache Size:** field controls how many FOVs are held in memory at once (default 100). Lower it if
memory is tight; raise it for faster back-and-forth navigation.

---

## 3. Select Channels

Channels are chosen with the **Channels:** field. Click it to open a scrollable list of every channel
in the dataset, and type to filter that list (case-insensitive). A counter shows how much you are
looking at — "12 of 148 shown · 2 selected".

- **Select all shown** takes the whole filtered group at once, which is the quick way to grab a family
  of markers: type `CD`, then click it.
- **Clear** drops the entire selection.
- Keyboard: ↑/↓ move, Enter toggles the highlighted channel, Esc closes the list, and Backspace on an
  empty filter box removes the last channel you added.

Each selected channel then appears as a **chip** below the field. Drag a chip by its `⋮⋮` grip to
reorder it — a blue bar marks where it will land — or click a chip and press ← / → to move it, or
Delete to remove it. The chip order sets the order of the per-channel control rows, the channel
legend, and the panes of the [channel grid view](#7-use-the-channel-grid-view).

The displayed image composites all selected channels using their assigned colors and contrast ranges.

!!! tip "Reordering does not change the image"
    Channels are composited **additively**, so the result does not depend on their order and no
    channel covers another. Reorder for your own convenience — see
    [Display Settings](display-settings.md#4-channels-pick-them-then-order-them).

---

## 4. Load a Marker Set

A **marker set** is a named combination of channels, colors, and contrast ranges. If you have one
saved, choose it from the **Marker Set:** dropdown and click **Load Marker Set** to restore that
configuration instantly.

To save your current configuration, type a name in **Set Name:** and click **Save Marker Set**. Use
**Update Marker Set** to overwrite the selected set, or **Delete Marker Set** (with the **Confirm
Deletion** checkbox) to remove one.

---

## 5. Adjust Contrast and Colors

Expand the **Channels** accordion in the left panel. Each selected channel exposes:

- A **color** dropdown to set its display color.
- A visibility checkbox to show/hide it without deselecting.
- **Min** and **Max** contrast sliders to adjust the display range.

These settings are per-session — save them as a marker set to reuse them.

Contrast starts at an automatic range rather than the raw data range, which is what makes a channel look all-black or all-white when the default does not suit it. [Display Settings](display-settings.md#getting-contrast-right) explains where that range comes from and how to correct it.

Enable **Show channel legend** to display a color key for the visible channels.

---

## 6. Overlay Masks and Annotations

If `masks_folder` and/or `annotations_folder` were provided:

…the left panel grows an accordion section for each:

- In the **Masks** section, enable a mask via its checkbox and pick a color from its **Mask
  &lt;name&gt;** dropdown. Adjust **Mask outline px:** to change the outline thickness.
- In the **Pixel annotations** section, enable **Show annotation**, choose an annotation from the
  **Annotation:** dropdown, and adjust **Fill alpha:** for overlay transparency. Use **Edit
  palette…** to customize per-class colors.

If a section is missing, that folder did not yield readable rasters — the sections are created from
the data, not toggled on.

For richer per-class mask coloring (fill vs. outline, per-class opacity, continuous colour scales, and
saved palettes), use the **Mask painter** plugin — see the [User Interface](user-interface.md)
reference. That one needs a cell table.

!!! tip "Masks-only view"
    Enable **No image (masks only)** to hide the channel image and inspect mask/annotation overlays
    on a blank background.

---

## 7. Use the Channel Grid View

Enable **Channel grid view** in the Channels panel to render each visible channel as a separate
labelled pane in a synchronized subplot grid — useful for comparing channels side by side without
toggling selections.

---

## Next Steps

- Explore the full [User Interface](user-interface.md) reference.
- Set the viewer up properly for your dataset — [Display Settings](display-settings.md).
- Capture [Regions of Interest](roi-manager.md).
- Load a cell table to unlock [single-cell analysis](cell-table.md).
