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

Channels are chosen with the **Channels:** field — a tag input. Click the field and pick (or type) a
channel name; each channel you add appears as a removable chip. Remove a channel by deleting its chip.

The displayed image composites all selected channels using their assigned colors and contrast ranges.

!!! note
    The channel picker is a tag/token field, not a scrolling list — there is no Shift- or
    Ctrl-click range selection. Add and remove channels one chip at a time, or load a saved
    **marker set** (below) to apply a whole combination at once.

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

Enable **Show channel legend** to display a color key for the visible channels.

---

## 6. Overlay Masks and Annotations

If `masks_folder` and/or `annotations_folder` were provided:

- In the **Channels** panel, enable a mask via its checkbox and pick a color from its **Mask
  &lt;name&gt;** dropdown. Adjust **Mask outline px:** to change the outline thickness.
- Under **Pixel annotations**, enable **Show annotation**, choose an annotation from the
  **Annotation:** dropdown, and adjust **Fill alpha:** for overlay transparency. Use **Edit
  palette…** to customize per-class colors.

For richer per-class mask coloring (fill vs. outline, per-class opacity, and saved palettes), use the
**Mask painter** plugin — see the [User Interface](user-interface.md) reference.

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
- Capture [Regions of Interest](roi-manager.md).
- Load a cell table to unlock [single-cell analysis](cell-table.md).
