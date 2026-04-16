# Basic Usage

This tutorial covers the essential steps for launching UELer and navigating image data.

---

## 1. Launch the Viewer

After [configuring your paths](../getting-started.md), run all cells in `script/run_ueler.ipynb`. The viewer appears inline at the bottom of the notebook.

---

## 2. Select an FOV

Use the **Select Image** dropdown in the left panel to choose a Field of View (FOV). UELer loads the image data on demand and caches recently accessed FOVs in memory.

The **Cache Size** slider controls how many FOVs can be held in memory at once. Increase it if you have sufficient RAM and want faster back-and-forth navigation.

---

## 3. Select Channels

In the **Channel Selection** list box, click one or more channel names to display them. Hold **Shift** to select a range, or **Ctrl / Cmd** to select individual channels.

The displayed image composites all selected channels using their assigned colors and contrast ranges.

---

## 4. Load a Marker Set

If you have previously saved a **Marker Set** (a named combination of channels, colors, and contrast ranges), select it from the **Marker Set** dropdown to restore that configuration instantly.

To save a new marker set, configure channels to your liking and use the **Save marker set** button.

---

## 5. Adjust Contrast and Colors

Expand the **Channel** accordion in the left panel. Each selected channel shows:

- A color picker to set the display color.
- A contrast range slider to adjust the minimum and maximum display values.

These settings are per-session. Save them as a marker set to reuse them.

---

## 6. Overlay Masks and Annotations

If `masks_folder` and/or `annotations_folder` were set:

- Expand the **Mask** accordion to enable segmentation mask overlays.
- Expand the **Annotation** accordion to enable annotation fills or outlines.

You can choose between outline-only, fill-only, or combined display modes, and adjust fill opacity.

---

## 7. Use the Channel Grid View

Enable **Channel grid view** in the Channels panel to render each visible channel as a separate labelled pane in a synchronized subplot grid. This is useful for comparing channels side by side without toggling selections.
