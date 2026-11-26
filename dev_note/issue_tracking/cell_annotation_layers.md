# Cell annotation layers: manual labels and ground truth

Status: **planned 2026-11-26, not started.** It comes after step A of the revised cell sets plan ([cell_sets.md](cell_sets.md#revised-plan-carts-stars-and-the-gallery-as-the-editing-view)), whose Gallery source and editing mechanics it reuses. It is on the long-term list ([long_term_todo.md](../long_term_todo.md)).

---

## Problem

Annotating cells by hand, or revising an existing annotation, is how a user builds ground truth, or pseudo ground truth, for a cell-type classifier, for benchmarking one, or for a figure. UELer has three pieces of this, and none of them is a per-cell record:

- **Cell Annotation** (heatmap meta-clusters, saved as checkpoints) labels **clusters**, not cells. A cell that sits in the wrong cluster cannot be corrected on its own.
- **Write to the cell table** (Cell sets plugin) writes a column for the current selection, but only **for the session**. It also does not record which cells a person looked at.
- **Cell sets with `key:value` tags** (phase 1) were meant to carry labels into the table. They cannot settle a cell that two sets label differently, correcting one cell means editing a set, and they have no notion of "checked". The cell sets plan was revised to drop this role (see [cell_sets.md](cell_sets.md), "Why the plan changed").

Ground truth needs what none of these gives: exactly one value per cell, saved, revised cell by cell, with a record of which cells a person confirmed or changed, and how.

---

## Design

### The annotation layer

An **annotation layer** is a named per-cell label, for example `celltype` or `qc`. It has:

- **A vocabulary**: the allowed values, each with a colour and an optional shortcut key, for example `CD8 T`, `CD4 T`, `Tumour`, `Unknown`. A fixed list keeps typos out of ground truth. New values are added deliberately, in the vocabulary editor, not by typing a new value at a cell.
  - `Unknown` / `Ambiguous` is an ordinary value. It is kept apart from `unlabelled`, which means nobody has assigned a value yet.
- **Per cell, by FOV, mask and label** (the identity sets use):
  - a **value**;
  - a **status**:
    - `unlabelled`;
    - `predicted`: the value came from an import (a column, heatmap clusters) and nobody has checked it;
    - `confirmed`: a person kept the value;
    - `changed`: a person corrected the value.
    
    **Ground truth is the `confirmed` and `changed` cells; the `predicted` cells are pseudo ground truth.**
- **A history**: one entry per change, with the cells, the old value and status, the new value and status, the time, and the recipe of the selection the change was applied to (lasso, gate, set, Gallery). This is the audit trail.
- **Saved automatically** with every change, unlike today's session-only write.

### Assigning and revising

- **From any selection.** An annotation palette shows the active layer's values as coloured buttons. Clicking one, or pressing its number key, applies that value to the current selection and records the cells as `changed`, or as `confirmed` when the value was already theirs. The palette sits in the Cell sets plugin or in its own plugin (open question), and a compact form of it can appear in the selection bar while a layer is active.
- **From a set.** **Apply value to set** labels every cell of a set or cart. The other direction works too: **Set from annotation** makes a set from cells with a given value and status, for example all `predicted` CD8 T cells.
- **Review in the Gallery.** The Gallery's source (cell sets step A) can be "layer `celltype`, value `CD8 T`, status `predicted`". Each tile then gets **✓ confirm** and a value picker to relabel it, and the header counts the cells left to review. Stepping through with **Draw new cells** covers a large group screen by screen.
- **Unbiased sampling.** **Sample cells to annotate…** draws N random cells, stratified by FOV or by predicted value, as a set, and opens it in the Gallery for review. This belongs in the first version: ground truth built only from cells a person found interesting is biased, and cannot be used to benchmark a classifier.

### Importing a starting point

- **From a cell-table column** (a pipeline's `cell_type`, FlowSOM clusters): each distinct value becomes a vocabulary entry, and every cell gets that value as `predicted`. This is what remains of phase 3's **Sets from column…**. Making one set per value stays available in the Cell sets plugin.
- **From Cell Annotation**: the meta-cluster labels of a checkpoint, imported as `predicted`.
- **Empty**: start from a typed vocabulary.

### Seeing it

- **In the image**: colour or outline cells by the active layer's values, with `unlabelled` and `predicted` cells dimmed or hidden. This shows what is still to do. The drawing should share the selection-style work on the long-term list.
- **As a column**: the layer appears in the cell table as two columns, `<layer>` and `<layer>_status`. They are rebuilt from the store when the dataset loads, so they work as colour, tooltip and plot columns in every session without saving the table.
- **Progress**: counts per value and status, per FOV and in total.

### Export

- The two columns go out with the cell table: CSV, and AnnData `obs` through `viewer.get_cell_table_adata()`.
- **Export layer**: the per-cell values and statuses with FOV, mask and label (CSV or Parquet), and the vocabulary. The history goes out as JSON, for training data and for a methods section.

---

## Storage

Beside the cell sets, in the dataset's `.UELer` folder (`dataset_storage_dir()`):

```
<storage>/.UELer/dataset_<sha1>/annotations/
    index.json                 # layers: name, vocabulary (value, colour, key), created, format
    <layer>.parquet            # fov, mask, label, value, status
    <layer>.history.jsonl      # one line per change (append-only)
```

- The per-cell file is rewritten atomically on each change, as cell sets are. The history is append-only, so a change costs one small write plus the rewrite.
- `index.json` carries a `format` version from the start.

---

## Open questions (to settle before building)

1. **Starting point.** Is the usual case revising an existing annotation (from clustering or an external pipeline), or labelling from scratch? This decides whether import or the empty vocabulary is the main path in the UI.
2. **Annotators.** Is there one person, or several whose labels must be compared? Several would mean one layer per annotator (`celltype@name`), plus a comparison: a confusion matrix and inter-annotator agreement (Cohen's κ). Planned for later either way, but it affects whether a layer records who made each change from the start. The cheap choice is to record it anyway.
3. **Label hierarchy.** Is a flat vocabulary enough, or is `T → CD8 T` needed, so cells can be labelled coarsely first and refined later? If so, values become paths, and a query for `T` includes its sub-values.
4. **Where the palette lives.** In the Cell sets plugin, or in a separate Annotation plugin? The Gallery review and the bar's compact palette are the same either way.

---

## Steps

1. **First version.**
   - The store, the layer and its vocabulary editor.
   - Imports from a column and from Cell Annotation.
   - The palette and its shortcut keys.
   - The statuses and the history.
   - The two cell-table columns, rebuilt on load.
   - Colouring the image by layer.
   - Review in the Gallery.
   - **Sample cells to annotate…**
   - Set from annotation, and apply value to set.
   - Progress counts and export.
   
   Tests:
   - store round trip;
   - the status rules (same value gives `confirmed`, different value gives `changed`, import gives `predicted`);
   - history entries;
   - columns rebuilt on load, including cells no longer in the table;
   - stratified sampling sizes;
   - the export matches the store.
   
   Docs: a new tutorial page, "Annotating cells", and a FAQ entry on ground truth versus pseudo ground truth.
2. **Later.**
   - Several annotators and their comparison.
   - A label hierarchy.
   - A query language over sets, tags and annotations (`celltype=CD8 T & status=confirmed & !qc=exclude`), sharing the boolean syntax of the ROI browser's tag filter (`&`, `|`, `!`, parentheses).
   - Suggesting labels from a classifier trained on the confirmed cells, proposed as `predicted` for review.
