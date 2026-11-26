# Cell sets, tags and selection history

Status: **phase 1 done** (history, save / select, tags, quick write; Cell Table Editor removed). **Step A1 done** (editable sets, the active cart, ★ favourites in the store and the plugin, snapshots, free-form tags); A2 (the Gallery as the editing view) and B are open. **The later phases were re-planned on 2026-11-26**: see [Revised plan: carts, stars and the Gallery as the editing view](#revised-plan-carts-stars-and-the-gallery-as-the-editing-view). It replaces phases 2 and 3 below; per-cell labels move to their own plan, [cell_annotation_layers.md](cell_annotation_layers.md). Both steps are on the long-term list ([long_term_todo.md](../long_term_todo.md)). Builds on the global selection (`SelectionHub`) and the in-line selection bar described in [selection_design.md](selection_design.md).

---

## Problem

The viewer now holds one current selection, whichever view made it, but only one. A population built step by step (lasso a region, gate a marker, narrow by a second marker) exists only until the next click. There is no way to keep several populations, to get back to the previous selection after a mis-click, to combine populations, or to say afterwards how a population was obtained. The pieces that keep things for later each cover part of this and none covers cells in general:

- **Cell Table Editor** writes a value into a cell-table column, but only for cells selected in the image (`selected_masks_label`).
- **Cell Annotation checkpoints** keep a parent-child tree of heatmap states under `.UELer/dataset_<id>/checkpoints/`, but they store a clustering, not a cell population.
- **ROI manager** keeps regions (areas, not cells).
- **Marker sets** show the save / load / update / delete pattern users already know, with a confirmation dialog on delete.

---

## Decisions

| Question | Decision |
| --- | --- |
| History persistence | **Session-only.** The history is kept in memory. When a cell set is saved, the history steps that led to it are saved with the set. |
| Combining sets | **In the first version that ships**: union, intersection and difference, recorded in the provenance. |
| Writing to the cell table | **`key:value` tags** (design below); no plain tags. |
| History limit | The log of steps is never truncated within a session; only the cells kept for ◀ ▶ are limited. |
| Where the sets live in the UI | **A new right-panel plugin, "Cell sets".** |
| Cell Table Editor | **Replaced** by the Cell sets plugin (section 6). |

---

## Design

### 1. Selection history (session-only)

- Every change of the current selection appends a **step** (`HistoryStep`) to `SelectionHub.history`: time, source, cell count, the cells, and a **recipe** describing how it was made.
- Recipes per source:
  - *image click*: FOV, mask, the clicked label and whether it added or removed;
  - *image lasso*: FOV, mask(s) searched, the loop's vertices;
  - *scatter*: the x and y columns (the lasso polygon if jupyter-scatter reports it);
  - *histogram*: each gate's channel and range or cutoff;
  - *heatmap*: the cluster column and the cluster label(s);
  - *set*: the set it was loaded from;
  - *combine*: the operation and its operands.
- Recipes are plain dicts so they serialise to JSON without help. Sources pass them to `publish(..., recipe=...)`; a source that passes none still gets a step, with an empty recipe.
- **◀ ▶** in the selection bar move through the history like a browser's back and forward. Moving back re-publishes the earlier step's cells (as the *history* source, pointing at the step), so the bar, Image, Gallery and Plots buttons behave as for any other selection. A new selection after going back does not cut the log: the steps after the cursor stay in it (the trail must be complete), and ▶ simply has nowhere to go. Each move is logged as a *navigation* step, which ◀ ▶ skip. The restored selection replaces the current one in every view that showed it, and also goes back to the view it was made in (an image lasso is outlined again), so ◀ reads as undo.
- **Two parts with different limits.** The *log* (time, source, count, recipe of every step) is kept in full for the whole session: it is small, and a saved set needs all of it (see the trail below). The *cells* of each step, which is what ◀ ▶ need to restore a selection, are kept for the last 50 steps only, because a step can hold hundreds of thousands of cells. A step older than that stays in the log but ◀ stops before it.
- Clearing with ✕ is itself a step, so ◀ undoes it. The confirmation dialog is kept but now says that ◀ brings the selection back.
- Actions that are not selections (a column write, a set saved) are logged as *event* steps with `hub.note`, so the trail shows them in order.

### 2. Cell sets

- **Save as set…** in the selection bar swaps the text and buttons for a name field, a tags field and Save / Cancel in the same row, so the bar keeps its height. The Cell sets plugin offers the same action. After saving, the bar shows "Saved N cells as the cell set …" until the next selection.
- **Cell identity is FOV + mask label**, not the cell-table row index, so a set survives reloading, filtering or re-sorting the cell table. Rows are resolved from `(fov_key, label_key)` when a set is selected; cells that no longer resolve are counted and reported ("3 of 1,240 cells are no longer in the cell table").
- A set stores:
  - `id` (uuid), `name`, `tags`, `created`, `note` (free text);
  - the cells;
  - `recipe` of the step it was saved from;
  - `parents`: ids of the sets it was derived from;
  - `trail`: the history steps that led to it (see below).
- **Selecting a set** publishes it with source *set* ("1,240 cells · set: CD8 T cells"); Image / Gallery / Plots then work as usual.
- **Rename, edit tags, edit note, delete** (delete through `viewer.confirm`, as marker sets do).

**The trail.** The set keeps the complete session log up to the moment it is saved: every step since the session started, with recipes, counts and times but without cell lists. It is complete rather than "since the previous save" because the previous save is often an unrelated population, so its trail would not explain this one; and because the log is never truncated, nothing a set needs can have been dropped. The cost is a few kilobytes per step, repeated across the sets of one session, which is acceptable. History from an earlier session is reached through `parents`: a set made from a set loaded from disk names it, and that set carries its own session's trail.

**The cells are the truth, the recipe is the explanation.** A set never re-runs its recipe on its own; after re-segmentation or a new cell table the recipe might select different cells. A deliberate **Re-apply recipe** action can come later.

### 3. Combining sets

- In the plugin: pick two or more sets and **∪ Union**, **∩ Intersect**, **− Subtract** (first minus the rest). The result becomes the current selection, with a *combine* recipe naming the operation and the operand sets; saving it records them as `parents`.
- With the current selection: **Add to / Narrow / Remove from** the current selection with a chosen set, the same three operations with the current selection as the first operand.
- A later phase may add a "combine mode" in the bar (the next lasso or gate adds to, narrows or removes from the current selection). It is not in the first version: it changes what every selection tool does and needs its own design pass.

### 4. Provenance view

> **Changed (2026-11-26).** The plugin's main tree is now by containment (`container`). The derivation shown here moves into a set's details. Export provenance is unchanged. See the revised plan, step B.

- The plugin lists the sets as a **tree** by `parents`, reusing the approach of the checkpoint tree (`CheckpointTreeWidget`). A set with several parents (a union, an intersection) is shown under its first parent with the others named in its row.
- Each row: name, tags, cell count, a one-line recipe ("∩ histogram gate CD8 > 2.1").
- **Export provenance**: a set's lineage (its trail and its ancestors' trails) as readable Markdown or JSON, for a methods section or a lab notebook.

### 5. Tags

> **Superseded (2026-11-26).** Tags become free-form organisation labels, and writing labels into the cell table moves to annotation layers ([cell_annotation_layers.md](cell_annotation_layers.md)). **Sets from column…** survives as the annotation import and as a way to make sets. The text below is kept as the phase 1 design.

A tag is a label on a **set**, and through the set on its cells. Tags do two jobs: organising the list (filter the plugin by tag) and carrying labels into the cell table.

**Every tag is `key:value`**, for example `celltype:CD8 T`, `region:tumour`, `qc:exclude`, `figure:3`. The key names a cell-table column and the value is what the cells get in it. This is how annotations already look in a cell table (one `cell_type` column), so the result works directly in pandas, scanpy and R. One form keeps one rule for writing; a one-word label is written as a key with a value (`flag:interesting`), and the tag editor refuses a tag without a `:` and says so.

**Writing tags to the cell table** (an explicit action, per tag key or for all):

- A `key:value` tag writes `value` into column `key` for the set's cells. Cells in no set with that key keep their current value (or stay empty in a new column).
- **Overlap is the point of tags, and conflicts are shown, not hidden.** Two sets can overlap freely as long as their tags do not disagree. If a cell is in one set tagged `celltype:CD8 T` and another tagged `celltype:CD4 T`, the write stops and lists the conflict (which sets, how many cells), with three ways out: give priority by set order (drag to reorder), write only the non-conflicting cells, or cancel.
- Writing changes `cell_table` in memory and calls `inform_plugins("on_cell_table_change")`, so the new column is offered straight away as a colour, tooltip or plot column. It does not save the cell table file; the columns persist because the tags persist with the sets, and they are rebuilt from the sets when the dataset loads.
- One vectorised write function (cells as fov / mask / label → rows in one lookup, then one assignment) serves every write.
- Reading back: **Sets from column…** creates one set per value of an existing categorical column, tagged `column:value`. This round-trips, and it brings existing annotations (FlowSOM clusters, an imported `cell_type`) into the sets list.

**Tags in the plugin:** tag chips on each row, a tag filter above the tree, autocompletion from the tags already in use, and a colour per tag key so `celltype:` tags read as a group.

### 6. Cell sets replaces the Cell Table Editor

The Cell Table Editor's **Column** + **Value** is a `key:value` tag written once, so the Cell sets plugin takes over its job and the editor plugin is removed. What the editor did, and what replaces it:

| Cell Table Editor | Cell sets |
| --- | --- |
| Column + Value, **Apply to selected cells** | The same two fields and **Write to current selection**, at the top of the plugin |
| Only cells selected in the image (`selected_masks_label`) | The current selection from any view (the hub) |
| One table lookup per cell | One vectorised write |
| Leaves no record | The write is a step in the session log ("wrote `celltype:CD8 T` to 312 cells"), and **Save as set** next to it keeps the cells with the tag |
| Column lost at the end of the session unless saved elsewhere | A quick write stays in memory, as before; saving it as a set makes it persist |

The editor goes in the same release as the quick write arrives, so there is no version without a way to write a column. Its docs entry (`docs/tutorials/user-interface.md`, the plugin list) points to Cell sets, and the log entry says it was replaced.

---

## Storage

Under the dataset folder the checkpoints already use (`CheckpointStore` resolves it, including the storage-root override):

```
<storage>/.UELer/dataset_<sha1>/cell_sets/
    index.json             # every set's metadata: id, name, tags, note, created, recipe, parents, trail, order
    <set id>.parquet       # the set's cells: fov, mask, label
```

- Metadata in one JSON index so the plugin lists sets without reading the cells; membership in Parquet (pyarrow is already a dependency) because a set can hold hundreds of thousands of cells.
- Written atomically (write to a temp file, then replace), as `CheckpointStore` does. The store is a small `CellSetStore` class beside `CheckpointStore`, sharing its dataset-folder logic rather than copying it.
- `index.json` carries a `format` version from the start.

---

## Where it lives in the UI

- **Selection bar** (stays one line): **◀ ▶** at the left of the text, **Save as set…** among the actions before ✕.
- **Cell sets plugin** (right panel): tag filter; the set tree; per set select, rename, tags, note, delete; multi-select for ∪ ∩ −; export provenance; write tags to the cell table; sets from a column.

---

## Implementation phases

> **Phases 2 and 3 were replaced on 2026-11-26** by steps A and B of the [revised plan](#revised-plan-carts-stars-and-the-gallery-as-the-editing-view) and by [cell_annotation_layers.md](cell_annotation_layers.md). Phase 1 stands as built.

1. **History, save / select, and the quick write.** `SelectionHub.history` with recipes from each source; ◀ ▶ in the bar; `CellSetStore`; the plugin with a flat list, save, select, rename, delete; `key:value` tags with the filter; the vectorised write function and **Write to current selection**; the Cell Table Editor removed. Tests for the hub history, the store round trip (including cells that no longer resolve), the bar buttons, and the write (logged, linking columns refused).
2. **Combining and the tree.** ∪ ∩ − on sets and with the current selection; `parents`; the tree view; the trail saved with each set.
3. **Tags into and out of the cell table, and export.** Writing the sets' tags with conflict reporting; rebuilding the columns from the sets on load; sets from a column; export provenance as Markdown / JSON.

Docs with each phase: a "Cell sets" tutorial page, the selection bar section in `docs/tutorials/user-interface.md`, and a FAQ entry on tag conflicts.

---

## Selecting a set: mask and FOV

- **Mask.** A set remembers the mask layer of each cell (cell and nucleus masks can share labels), and selecting it outlines the cells on that layer only.
- **FOV.** Selecting a set always moves the view to its cells, with one rule: go to the FOV holding most of the set's cells (with one FOV, that is simply its FOV). If the set's FOVs belong to one stitched map, open the map instead, which shows all of them. Nothing moves if the FOV shown already holds some of the set's cells, so a set used while working in one FOV does not jump away. The bar's count ("312 in this FOV") says how much of the set is outside the view.

---

## Correction to the plan: writes are per row, not per mask

An earlier version of section 6 said the Cell Table Editor wrote a label for both cell and nucleus masks and that Cell sets would match FOV + mask + label. That was wrong. A cell-table row describes one cell of the mask layer the table was made from (`mask_key`) and is keyed on `(fov, label)` alone, so there is nothing per mask to write: both tools write cell-table rows. The mask layer matters only for **outlining** a set in the image, which is what `mask` in the stored cells is for. The table row was removed from section 6.

---

## Revised plan: carts, stars and the Gallery as the editing view

Planned 2026-11-26, after phase 1 was in use. Nothing in this section is built yet.

### Why the plan changed

Phase 1 made tags `key:value` so that phase 3 could write them into the cell table. In use, that mixed two different jobs in one mechanism:

- **"What is this cell?"** A label: one value per cell, revised cell by cell, needing a record of which cells a person actually checked. This is manual annotation and the building of ground truth or pseudo ground truth.
- **"Which cells do I want to keep coming back to?"** A collection: a cell can be in any number of them, they grow over time, and they feed the Gallery, figures and comparisons. This works like a shopping cart for interesting cells.

Sets answer the second question well and the first badly. If one cell is in a set tagged `celltype:CD8 T` and in another tagged `celltype:CD4 T`, nothing settles which label is right. Correcting one cell means editing or adding a set. And a set cannot say which of its cells were looked at. So the work splits in two:

- **Sets become carts** (this section): editable collections, with starred favourites and one level of named sub-groups. The Gallery is where you edit them.
- **Per-cell labels become annotation layers**: one value per cell, a fixed list of allowed values, a review status and a per-cell history. Plan: [cell_annotation_layers.md](cell_annotation_layers.md).

Tags stay on sets, but as **free-form organisation labels** (`pilot`, `todo`, `figure 3`), as on ROIs. They are no longer read as column names.

### Decisions

| Question | Decision |
| --- | --- |
| Are sets editable? | **Yes, every set is open.** Cells can be added and removed at any time. **Snapshot** makes a frozen copy, which records the set it came from and refuses edits. One rule, so nobody has to wonder whether a given set can be changed. |
| Tag form | **Free-form.** The `key:value` rule and its refusal are dropped. Existing tags stay valid as plain strings, so nothing needs migrating. |
| Favourites | **A star per cell, stored as a child set** ("★ favourites") of the set it belongs to. It can be opened, sent to the Gallery and exported like any set. |
| Hierarchy | **Containment, one level deep for now.** A child is always a subset of its parent. The storage allows more depth, but the UI and the rules are built and tested for one level only until there is a real need for more. |
| Two kinds of tree | **Containment** (`container`: the set this one is part of) is the main tree in the plugin. **Derivation** (`parents` and the trail: what a set was made from) is provenance, shown in a set's details and in the provenance export. The phase 2 plan showed the derivation tree as the main view; that is replaced. |
| Where cells are edited | **The Gallery is the editing view, and the Cell sets plugin manages the sets.** See "The rule between the Gallery and the plugin" below. |
| Gallery Refresh | **Unchanged: Refresh re-renders the cells already shown.** A separate **Draw new cells** button draws a new sample. A checkbox that changes what Refresh does was considered and rejected, because it is a hidden mode: the same button would sometimes keep the cells and sometimes change them. |
| Saving what the Gallery shows | **Save shown cells as set** makes a child of the Gallery's source set, recording how the sample was drawn. If the source is not a saved set, the result is a top-level set. |

### Data model changes (`CellSetStore`)

- **New metadata fields**, defaulted when an older `index.json` is read, so phase 1 sets load unchanged (`format` goes from 1 to 2 on the next write):
  - `container`: id of the set this one is a subset of, or `null`;
  - `role`: `null` for an ordinary set, `"favourites"` for the ★ child (at most one per set, created on the first star);
  - `frozen`: `true` for a snapshot;
  - `edits`: a log of membership changes, each with time, `add` or `remove`, cell count and the recipe of the selection that was added or removed (so a cart keeps its provenance as it grows);
  - `sample`: for a set saved from the Gallery, the source set id, the sample size and the seed.
- **New store operations:**
  - `add_cells(set_id, cells, recipe)` adds the cells to the set and to its container, so a child stays a subset;
  - `remove_cells(set_id, cells, recipe)` removes the cells from the set and from all its children;
  - `star(set_id, cells)` and `unstar(set_id, cells)` act on the favourites child, creating it when needed. Starring a cell that is not in the set adds it to the set first;
  - `snapshot(set_id, name)` copies the cells and the metadata, sets `frozen` and records the source in `parents`;
  - `children(set_id)`.
  
  Every edit is refused on a frozen set, with a message.
- **Deleting a set that has children** asks whether to delete the children too or move them to the top level. The ★ child is always deleted with its set.
- **Membership is rewritten whole on each edit** (the Parquet file, written atomically as today). A set of 10⁵ cells takes milliseconds to write. If single-cell edits on very large sets turn out slow, the fallback is an append-only edit file folded into the Parquet file on load. That is not built unless measured to be needed.
- **Every edit is also a hub event** (`hub.note`), so the session trail shows "added 12 cells to cart 'cool CD8'".

### Carts: building a set over time

- **The active cart.** The plugin picks one set as the active cart (a dropdown, or **Make active** on a set's details). Nothing is active by default.
- **In the selection bar**, while a cart is active: **+ Cart** adds the current selection to it and **− Cart** removes the current selection from it. The tooltip names the cart. The bar stays one line, so the cart's name is not shown as text.
- **In the image:** **Shift-click** on a cell toggles it in the active cart. Shift-click is not used by the image gestures today. With no active cart, Shift-click says so in the bar and does nothing else.
- **In the plugin:** the same add and remove actions, plus **New empty cart**.
- **Tags** in the save form and in the details become free-form. The prefill after a quick write (`column:value`) stays, now as an ordinary tag.

### The Gallery as the editing view

**The rule between the Gallery and the plugin.** The Cell sets plugin stores and manages sets: naming, tags, notes, combining, deleting, the tree, export. The Gallery is where you look at a set's cells and edit it, because that is where the cells are visible. The Gallery holds no copy: it reads and writes through the same `CellSetStore`, and both views refresh from the store's change notifications. A star added in the Gallery therefore shows in the plugin straight away, and the other way round.

- **A source, named in a header.** The Gallery always shows one source: the current selection, or a saved set (when a set is selected, or opened from the plugin with **Open in Gallery**). A header line names it and gives the counts: "Cart: cool CD8 · showing 24 of 310 · ★ 5". This also covers the open item about saying when the Gallery shows only a sample.
- **Actions on a tile.** A tile click keeps its current meaning (go to the cell in the image). A small **★** in the tile's corner toggles the star. Starred tiles show it filled. With a saved set as the source, each tile also gets a **remove from set** control. An **Add shown cells to cart** button sits in the header.
  - `TileGalleryWidget` gains a per-tile `badges` field and an `action` traitlet carrying `"<action>|<id>|<nonce>"`, beside the existing `clicked`.
  - The ROI browser uses the same widget and ignores the new fields.
- **Show:** all cells, or ★ only.
- **Refresh** re-renders the cells shown, as it does now. The sample is fixed because `_limit_selection` seeds its random draw from the selected row indices, so one selection always gives the same sample.
- **Draw new cells** draws another sample from the source, by adding a draw counter to that seed. By default it prefers cells not yet shown in this session, so a large cart can be gone through screen by screen. Once every cell has been shown, it starts again from all of them. It is disabled when the source fits on one screen.
- **Save shown cells as set** saves exactly the tiles on screen, as a child of the source set when the source is a saved set. It records the sample size and seed in `sample`, so a figure panel can say "these 24 cells, drawn from cart X".

### Combining and named sub-groups (step B)

- ∪ ∩ − between sets, and **Add to / Narrow / Remove from** the current selection with a set, as in section 3. Each result records its operands in `parents`.
- **Named sub-groups:** **New sub-group…** on a set makes an empty child, filled from the Gallery or the bar (where the active cart can be a child). Examples: "examples for figure 2", "doubtful".
- **The plugin's list becomes a tree by `container`**, one level deep: each set with its ★ child and its sub-groups.
- **The derivation history goes into the set's details** ("made from: CD8 T ∩ tumour region"), with **Export provenance** (Markdown / JSON) as in section 4.

### Steps

Step A is built in two parts: **A1**, the store, the active cart, + Cart / − Cart, Shift-click, stars and snapshots in the plugin (done, see the record below); **A2**, everything in the Gallery.

- **A. Carts, stars and the Gallery as the editing view.** Store fields and operations, open sets with snapshots, free-form tags, the active cart in the bar and on Shift-click, the Gallery header with the source and counts, ★ and remove on tiles, ★-only view, Draw new cells, Save shown cells as set. Tests:
  - store round trip with the new fields, and phase 1 files read unchanged;
  - the subset rule when adding to a child and removing from a parent;
  - frozen sets refuse edits;
  - star on a cell outside the set adds it to the set;
  - the Gallery draw counter changes the sample and Refresh does not;
  - the shown-cells set is a child that records its sample.
  
  Docs: rewrite the Tags section of `docs/tutorials/cell-sets.md`; new sections on carts and stars; the Gallery page.
- **B. Combining and named sub-groups.** ∪ ∩ −, the containment tree with sub-groups, the derivation history in the details, provenance export.

Annotation layers ([cell_annotation_layers.md](cell_annotation_layers.md)) come after step A and can run alongside B. They use the active-cart and Gallery-source mechanics from step A for their review step.

---

## **Update Summary: phase 1 — history, saved sets, tags, quick write**

A selection existed only until the next one, nothing kept several populations or recorded how one was made, and writing a label into the cell table worked only for cells selected in the image.

---

### **Actions Taken**
- **Session log in the hub:**
  `SelectionHub.history` holds a `HistoryStep` per selection (time, kind, source, count, recipe, cells). `publish(..., recipe=)` and `clear(..., recipe=)` record selection steps; `note()` records events; `go(±1)` restores a step and records a navigation step. Only the last `KEEP_CELLS` (50) selection steps keep their cells; ◀ stops at the first step without them rather than skipping it, so going back never jumps over a gap. `trail()` is the log as dicts without cells.
- **Recipes from every source:**
  image click (`action` select / add / remove, FOV, mask, label; background and right click logged as such), image lasso (`ImageDisplay._lasso_recipe`: FOV or map, masks, added, the loop thinned to ≤200 vertices at 0.1 px), scatter (the plot titles; jupyter-scatter does not report the lasso polygon), histogram (each channel's gate term, numpy values made plain), heatmap (column, cluster, current-FOV-only), set (`set_id`, name, mask). The FOV-switch clear of an image selection is logged with `reason`.
- **◀ ▶ and the restore rule:**
  `viewer.step_selection` sends the restored selection to every view that showed the current one plus the restored step's own source view, or removes it from them when the restored step is a clear. The restored selection has source `history` and no source view, so every bar button acts on it.
- **Storage:**
  `CellSetStore` in `ueler/viewer/cell_set_store.py`; `dataset_storage_dir()` moved out of `CheckpointStore` so both use one folder rule. Names are unique (case-insensitive). The trail is the whole log at save time; `parents` holds the set the selection came from (directly, or through a ◀ restore of a set step).
- **Cell identity:**
  `cells_from_indices` takes the mask layer from the image selection when the image shows the selection, else `mask_key`; `indices_from_cells` resolves with one merge on `(fov, label)` (labels compared as nullable integers so a string label column still matches) and counts cells no row matches.
- **Outlining on a set's mask:**
  `sync_mask_highlights_from_selection(mask_name=)` and `viewer.linked_selection_mask`, so a FOV switch re-projects on the same layer. The re-projection branch of `_reapply_selection_highlights` now also refreshes the bar's FOV count, which it skipped before by returning early.
- **The Cell sets plugin:**
  save (name, tags), write (Column, Value), filter, list, details (cells, FOVs, source, saved, selection steps recorded), edit name / tags / note, select, delete through `PluginBase.confirm` (refuses and says so without a dialog, as the other disk-deleting actions do). It does not register widget-state observers: typed names and values are not settings. It registers **Save as set…** in the bar only when the store exists (a dataset folder is known).
- **Selecting a set moves the view** only when the FOV (or map) shown holds none of its cells: to the stitched map when the set spans several FOVs of one map, else to the FOV holding most cells (leaving map mode first).
- **Quick write:**
  `write_column` assigns once for all rows; a new column starts as `""`; a non-object column is cast to object first; `system_columns` (fov, label, X, Y, mask keys) are refused. After a write the save tags field is pre-filled with `column:value`.
- **Removed the Cell Table Editor** and its test; docstrings and the AnnData test that cited it now cite the Cell sets write.
- **Docs:** `docs/tutorials/cell-sets.md` (nav, tutorials index), selection bar section and plugin list in `user-interface.md`, `cell-table.md`; Pattern 6 in `dev_note/topic_plugin_development.md`.

### **Not done yet (phases 2 and 3)**
Combining sets (∪ ∩ −), the tree view, writing the sets' tags to the cell table with conflict reporting, rebuilding those columns when the dataset loads, sets from a column, and exporting provenance.

---

### **Files Changed**
- `ueler/viewer/selection.py` — `HistoryStep`, the log, ◀ ▶, notes, trail; `CellSelection.recipe`; labels for `set`, `history`, `clear`.
- `ueler/viewer/cell_set_store.py` — new: store, tags, cell identity, `write_column`.
- `ueler/viewer/checkpoint_store.py` — `dataset_storage_dir()` shared with the cell set store.
- `ueler/viewer/plugin/cell_sets.py` — new plugin.
- `ueler/viewer/plugin/cell_table_editor.py` — removed.
- `ueler/viewer/selection_bar.py` — ◀ ▶, **Save as set…** and its in-row form.
- `ueler/viewer/main_viewer.py` — recipes through `publish_image_selection`, `step_selection`, set mask on **Image**, ✕ dialog wording, FOV-switch logging and bar refresh.
- `ueler/viewer/image_display.py` — click and lasso recipes.
- `ueler/viewer/plugin/_chart_common.py` — `publish_selection(recipe=)`, `sync_mask_highlights_from_selection(mask_name=)`.
- `ueler/viewer/plugin/chart.py`, `histogram.py`, `heatmap_layers.py` — recipes.
- `ueler/cell_table.py` — docstring.
- `tests/test_cell_sets.py` — new; `tests/test_selection_hub.py`, `tests/test_cell_table_anndata.py` — updated; `tests/test_cell_table_editor.py` — removed.
- `docs/tutorials/cell-sets.md` (new), `docs/tutorials/user-interface.md`, `docs/tutorials/cell-table.md`, `docs/tutorials/index.md`, `mkdocs.yml`.

---

### **Tests**
```bash
python -m unittest tests.test_cell_sets tests.test_selection_hub tests.test_lasso_hint tests.test_cell_table_anndata tests.test_docs_consistency
python -m unittest discover -s tests
```
- ✅ All tests passed; `mkdocs build --strict` passes. A smoke run on a real viewer (image lasso → histogram gate → ◀ ▶ → save from the bar → quick write → ✕ → select the set) behaved as described.

---

## **Update Summary: step A1 — editable sets, the active cart, stars, snapshots, free-form tags**

A saved set was a snapshot of one selection, so collecting cells over time meant re-selecting and re-saving, and `key:value` tags mixed organising the list with labelling cells. A1 builds the store side of the revised plan and the cart controls outside the Gallery.

---

### **Actions Taken**
- **Store format 2, read-compatible with format 1:**
  `_load_index` fills `container`, `role`, `frozen`, `edits` (and the other optional fields) with defaults, so phase 1 files need no migration; the next write stamps `format: 2`. Chosen over a one-off migration because nothing in format 1 conflicts with the new fields.
- **Editing and the subset rule:**
  `add_cells` walks up through `container` (a child's cells are added to its parent); `remove_cells` walks down through the children. Cells are compared on `(fov, mask, label)` with a `MultiIndex.isin`, so the same label on two mask layers is two cells. The whole Parquet file is rewritten per edit, as planned; no delta log was needed at this size. A frozen set refuses an edit only when cells would actually change, so a snapshot's own ★ child can be saved under it.
- **Names are unique among siblings** (same `container`), not globally, so every set can have a child named "★ favourites". The plugin shows a favourites child as "★ <set name>" in the bar and status.
- **Stars:**
  `star` creates the favourites child on first use and adds through it, which puts an outside cell into the set by the subset rule; `unstar` removes from the child only.
- **Snapshots:**
  `snapshot` copies cells, tags, note and trail, records the source in `parents` and `recipe.snapshot_of`, and copies the stars as a frozen child. The plugin names it "<set> (snapshot YYYY-MM-DD HH:MM)" instead of asking for a name, to avoid a dialog; it can be renamed.
- **Deleting** removes the set and all its descendants. The "delete or move children" choice in the plan is left for step B, when user-made sub-groups exist; in A1 the only children are stars, which belong to their set.
- **Free-form tags:**
  `parse_tags` collapses whitespace, drops empties and exact repeats, and never reports a problem; its `(tags, problems)` return is kept for the callers.
- **The active cart (plugin, session only, none by default):**
  the **Active cart** dropdown lists top-level sets that are not snapshots; **Make active cart** and **New empty cart** set it. The plugin registers **+ Cart** / **− Cart** with the bar (`SelectionBar.set_cart`, actions `cart_add` / `cart_remove`) and sets `ImageDisplay.shift_click_handler`. Messages go to the plugin status and to the bar (`_tell`), because a Shift-click is made with the eyes on the image.
- **Shift-click** (`ImageDisplay.on_mouse_click`):
  checked after Ctrl, so Ctrl + Shift stays a Ctrl click; a Shift-click on the background does nothing; the selection is never touched.
- **Plugin list:**
  top-level sets only, marked 🛒 (active cart), 🔒 (snapshot) and "★ n"; new buttons **Select ★**, **★ Star selection**, **☆ Unstar**, **Make active cart**, **Snapshot**. Selecting an empty set says it has no cells instead of reporting cells missing from the table.

### **Not done yet**
A2 (the Gallery header with source and counts, ★ and remove on tiles, ★-only view, Draw new cells, Save shown cells as set) and step B.

---

### **Files Changed**
- `ueler/viewer/cell_set_store.py` — format 2, free-form tags, `add_cells`, `remove_cells`, `contains`, `children`, `favourites`, `star`, `unstar`, `snapshot`, recursive delete, sibling-scoped names.
- `ueler/viewer/plugin/cell_sets.py` — active cart, New empty cart, + / − Cart, Shift-click handler, star / unstar / Select ★, Snapshot, list marks and details.
- `ueler/viewer/selection_bar.py` — **+ Cart** / **− Cart** and `set_cart`.
- `ueler/viewer/image_display.py` — Shift-click routed to `shift_click_handler`.
- `tests/test_cell_sets.py`, `tests/test_image_click_gestures.py` — new and updated tests.
- `docs/tutorials/cell-sets.md`, `docs/tutorials/user-interface.md`, `docs/tutorials/cell-table.md`, `README.md`, `doc/log.md`.

---

### **Tests**
```bash
python -m unittest tests.test_cell_sets tests.test_image_click_gestures tests.test_docs_consistency
python -m unittest discover -s tests
```
- ✅ All tests passed (1633); `mkdocs build --strict` passes. Not yet tried by hand in a running viewer.
