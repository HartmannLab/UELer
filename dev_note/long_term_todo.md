# Long-term to-do

Planned work that is designed but deliberately not scheduled yet. Each item points to the document holding its plan; this list only says what is left and why it waits. `dev_note/Todos.md` is the historical packaging checklist and is left as written.

**Format.** One item per bullet: a checkbox, a bold title, what is left, the plan document, and the date it was added. `[ ]` open, `[~]` partly done (say which part), `[x]` done (move it to **Done** with the date and the `doc/log.md` entry it shipped in). Never delete an item; close it or mark it dropped with the reason.

---

## Selection and cell sets

- [ ] **Cell sets, phase 2: combining sets and the tree view.** ∪ Union, ∩ Intersect, − Subtract between saved sets, and Add to / Narrow / Remove from the current selection with a set; each result records its operands as `parents`. A tree of the sets by `parents` in the Cell sets plugin (the approach of `CheckpointTreeWidget`). Phase 1 already stores `parents` and the trail, so nothing saved now needs migrating. Plan: [cell_sets.md](issue_tracking/cell_sets.md) §3–4. Added 2026-11-26.
- [ ] **Cell sets, phase 3: tags into and out of the cell table, and export.** Write the sets' `key:value` tags to the cell table with conflict reporting (two sets giving one cell different values for one key: priority by set order, write the rest, or cancel); rebuild those columns from the sets when the dataset loads, so labels persist without saving the table; **Sets from column…** (one set per value of a categorical column); export a set's provenance (its trail and its ancestors') as Markdown / JSON; a FAQ entry on tag conflicts. Plan: [cell_sets.md](issue_tracking/cell_sets.md) §4–5. Added 2026-11-26.
- [ ] **One selection colour and style.** Amber `#FFB000` by default; a selected cell drawn with a thicker outline plus a light tint, so it differs from a mask outline even when the masks use the same colour; one drawing routine for the image (single FOV, map mode, channel grid); the scatter, histogram and Gallery read the same colour. Plan: [selection_design.md](issue_tracking/selection_design.md) §1 and implementation steps 1–3. Added 2026-11-26.
- [ ] **Selection colours as GUI parameters, and collapsible sections in the inner Advanced Settings tab.** The inner **Advanced Settings** tab becomes an accordion: *Display and performance* (the current four settings) and *Selection and colours* (selection colour, outline width, fill opacity, default mask colour, plot base colour), saved with the widget state. The Gallery's own outline picker stays as an override. Do together with the item above. Plan: [selection_design.md](issue_tracking/selection_design.md) §2. Added 2026-11-26.
- [ ] **One selection model.** A single selection object (row indices, `(fov, mask, id)` triples, source) that every view renders, replacing `selected_masks_label`, the plots' own index sets and `linked_selection_indices`; `SelectionHub` already sits where it would go. Plan: [selection_design.md](issue_tracking/selection_design.md) §4. Added 2026-11-26.
- [ ] **"Gallery ✓" above the Gallery's cell limit.** The Gallery shows a random sample when a selection exceeds **Max displayed cells** and only logs it; the bar should say "Gallery ✓ (20 of 35 shown)". Plan: [selection_design.md](issue_tracking/selection_design.md), "What the experiment showed". Added 2026-11-26.
- [ ] **The heatmap's image highlight does not reach the selection bar.** `highlight_cells` is also the FOV-switch re-apply, so publishing from it needs a flag that tells a user action from a re-apply. Plan: [selection_design.md](issue_tracking/selection_design.md), "What the experiment showed". Added 2026-11-26.
- [ ] **Link boxes re-forward after an unlink.** Unlinking a view in the bar lasts only until the next selection from a plot whose **Linked plugins** box for that view is ticked. Decide whether unlinking should untick the box, or the bar should say the box is on. Plan: [selection_design.md](issue_tracking/selection_design.md). Added 2026-11-26.

---

## Done

_Nothing yet._
