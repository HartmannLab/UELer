# Cell Sets

A cell set is a selection you keep: a named group of cells, with tags, saved with the dataset and available in every later session. The **Cell sets** plugin in the right panel holds them, and it is also where you write a value into the cell table for the selected cells. It appears once a cell table is loaded.

---

## Going back to an earlier selection

Every selection you make in this session is remembered: an image click or lasso, a scatter lasso, a histogram gate, a heatmap cluster sent to the Gallery, a set you selected, and unselecting with **✕**. **◀** and **▶** at the left of the [selection bar](user-interface.md#the-selection-bar) step back and forward through them.

- **Going back puts the earlier selection where the current one was shown.** If the current selection is outlined in the image, the earlier one is outlined there instead; a selection made in the image is outlined there again. The bar then reads, for example, "35 cells · step 1 (image lasso)".
- **◀ undoes ✕.** Unselecting everything is a step too, so ◀ brings the selection back.
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

The cells of a set never change on their own. If you re-segment or replace the cell table, the set keeps the cells it had; selecting it tells you how many of them are no longer in the table.

---

## Tags

Every tag is written `key:value`, for example `celltype:CD8 T`, `region:tumour` or `qc:exclude`. Separate several with commas. The key is the name of a cell-table column and the value is what the set's cells get in it, which is how cell types and other annotations are usually stored (one `cell_type` column), so they carry straight into pandas, scanpy or R.

A tag without a `:` is refused with a message; for a one-word label, give it a key: `flag:interesting`. A set cannot carry two values for one key.

Use **Filter** above the list of saved sets to show only the sets whose name or tags contain the text you type: `tumour`, `celltype:`, `CD8`.

---

## Selecting, editing and deleting a set

Pick a set in the list to see how many cells it has, in how many FOVs, what made it and how many selection steps it recorded.

- **Select cells** makes the set the current selection ("1,240 cells · set: CD8 T cells") and outlines it in the image, on the mask layer it was saved on. If the FOV shown holds none of its cells, the view moves: to the FOV with most of them, or to the stitched map when all of its FOVs belong to one [map](map-mode.md). From the bar you can then send it to the Gallery or the plots as any other selection.
- **Save changes** stores a new name, tags or note. The cells stay as they are.
- **Delete** removes the set from disk after a confirmation dialog.

---

## Writing to the cell table

**Write to the cell table** puts a value into a column for every cell of the current selection, whichever view made it. Enter a **Column:** (an existing one from the list, or a new name) and a **Value:**, then click **Write to current selection**.

- The new or changed column is available straight away, for example as a colour, tooltip or plot column.
- The columns that link the table to the images (FOV, label, X, Y, mask) cannot be written to.
- Writing a text value into a numeric column turns that column into text.
- **A write changes the cell table for this session only.** To keep the label, save the selection as a set: after a write, the save section's tags field is already filled in with `column:value`.
- Each write is recorded in the session's selection history, so a set saved afterwards shows that it happened.

This replaces the **Cell Table Editor**, which could only write to cells selected in the image.

For an AnnData cell table, written columns go back into `obs` when you call `viewer.get_cell_table_adata()`; see [Working with a Cell Table](cell-table.md#getting-your-annotations-back-out).
