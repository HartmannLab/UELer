# Cell Sets

A cell set is a named group of cells, with tags, saved with the dataset and available in every later session. You can make one from a selection, or start an empty one and collect cells in it over time, like a shopping cart for interesting cells. The **Cell sets** plugin in the right panel holds them, and it is also where you write a value into the cell table for the selected cells. It appears once a cell table is loaded.

---

## Going back to an earlier selection

Every selection you make in this session is remembered: an image click, Ctrl-click or lasso (with the toggle or Ctrl + drag), a scatter lasso, a histogram gate, a heatmap cluster sent to the Gallery, a set you selected, and unselecting with **✕** or a right click on the image background. **◀** and **▶** at the left of the [selection bar](user-interface.md#the-selection-bar) step back and forward through them.

- **Going back puts the earlier selection where the current one was shown.** If the current selection is outlined in the image, the earlier one is outlined there instead; a selection made in the image is outlined there again. The bar then reads, for example, "35 cells · step 1 (image lasso)".
- **◀ undoes ✕ and a right click on the background.** Unselecting everything is a step too, so ◀ brings the selection back.
- **The last 50 selections can be restored.** Further back, ◀ stops: the cells of older steps are let go to keep memory in check. The record of what was done is kept for the whole session, and it is what a saved set stores (below).
- The history lasts for the session. Saving a set is what keeps a selection.

---

## Saving a set

With cells selected, click **Save as set…** in the selection bar. The bar turns into a **name** field, a **tags** field, and **Save** / **Cancel**. The plugin's **Save the current selection** section does the same.

A saved set keeps:

- **its cells**, by FOV, mask layer and mask label, not by row number, so the set still finds its cells after the cell table is reloaded, filtered or re-sorted;
- **its tags** (next section) and a free-text **Note:**;
- **how the selection was made**: the histogram gates and their ranges, the lasso's FOV and outline, the scatter plots, the heatmap cluster;
- **the session's selection history up to the save**: every step, with what made it and how many cells it had. This is the record of how you arrived at the population;
- **the set it was made from**, if you selected a saved set and saved the result under a new name.

Sets are stored in the dataset's `.UELer` folder, next to the Cell Annotation checkpoints.

The cells of a set change only when you add or remove them (next sections). If you re-segment or replace the cell table, the set keeps the cells it had; selecting it tells you how many of them are no longer in the table.

---

## Carts: collecting cells over time

Every set can be added to and taken from, so a set also works as a cart: pick cells across FOVs and sessions, by whatever means suits each one.

- **Choose the active cart** under **Active cart** in the plugin, or with **Make active cart** on a set. To start from nothing, type a name (and tags) under **Save the current selection** and click **New empty cart**; it becomes the active cart. Nothing is active until you choose.
- **+ Cart** and **− Cart** appear in the [selection bar](user-interface.md#the-selection-bar) while a cart is active. They add the current selection to the cart or take it out, whichever view made it: a lasso, a gate, a scatter lasso, a heatmap cluster. The bar says how many cells were added and how many were in the cart already.
- **Shift + click a cell** in the image to put it in the active cart, or take it out if it is there. The current selection does not change. With no active cart, the bar says so and nothing happens.
- The active cart is marked 🛒 in the list. It is chosen for this session only.
- Every addition and removal is recorded with the set: when, how many cells, and how they were selected. The set's details show how often it was edited.
- A cell is its FOV, mask layer and label: the same label on the cell mask and on the nucleus mask counts as two cells.

---

## Stars: favourite cells in a set

Each set can mark some of its cells with a ★, for example the clearest examples of a cell type.

- Select the cells, pick the set in the list and click **★ Star selection**. A starred cell that was not in the set yet is added to it, since a set's stars are always its own cells. **☆ Unstar** takes the star off; the cells stay in the set.
- **Select ★** makes the starred cells the current selection ("set: ★ CD8 T"), to send to the Gallery or the plots like any selection.
- The list shows how many cells are starred ("★ 5"). Removing a cell from a set removes its star too, and deleting a set deletes its stars.

---

## Snapshots

**Snapshot** saves a frozen copy of a set, with its stars, named after the set and the time ("CD8 T (snapshot 2026-11-26 14:05)"); rename it if you like. A snapshot's cells cannot change: it cannot be the active cart, and starring in it is refused. It is marked 🔒 in the list and records the set it was taken from. Use one before reworking a cart, or to keep the exact cells behind a figure.

---

## Tags

Tags are free-form labels for keeping the list in order: `pilot`, `todo`, `figure 2`, `celltype:CD8 T`. Separate several with commas. They are not written into the cell table; to put a value into a column, use [Writing to the cell table](#writing-to-the-cell-table).

Use **Filter** above the list of saved sets to show only the sets whose name or tags contain the text you type: `tumour`, `pilot`, `CD8`.

---

## Selecting, editing and deleting a set

Pick a set in the list to see how many cells it has, in how many FOVs, what made it and how many selection steps it recorded.

- **Select cells** makes the set the current selection ("1,240 cells · set: CD8 T cells") and outlines it in the image, on the mask layer it was saved on. If the FOV shown holds none of its cells, the view moves: to the FOV with most of them, or to the stitched map when all of its FOVs belong to one [map](map-mode.md). From the bar you can then send it to the Gallery or the plots as any other selection.
- **Save changes** stores a new name, tags or note. The cells stay as they are.
- **Delete** removes the set and its stars from disk after a confirmation dialog.

---

## Writing to the cell table

**Write to the cell table** puts a value into a column for every cell of the current selection, whichever view made it. Enter a **Column:** (an existing one from the list, or a new name) and a **Value:**, then click **Write to current selection**.

- The new or changed column is available straight away, for example as a colour, tooltip or plot column.
- The columns that link the table to the images (FOV, label, X, Y, mask) cannot be written to.
- Writing a text value into a numeric column turns that column into text.
- **A write changes the cell table for this session only.** To keep the cells you wrote to, save the selection as a set: after a write, the save section's tags field is already filled in with `column:value`, so the set says what was written.
- Each write is recorded in the session's selection history, so a set saved afterwards shows that it happened.

This replaces the **Cell Table Editor**, which could only write to cells selected in the image.

For an AnnData cell table, written columns go back into `obs` when you call `viewer.get_cell_table_adata()`; see [Working with a Cell Table](cell-table.md#getting-your-annotations-back-out).
