# Cell selection: one look, one bar, one model

## Problem

A cell selection, made by a click, the image lasso, a scatter lasso, a histogram gate or a heatmap cluster, has no identity of its own on screen. Nothing in the image says what can be done with it.

**It looks like a mask outline.** In the image the selected cells get a 1 px outline drawn in hard-coded white, which is also the default colour of every mask outline. On a FOV with masks shown, the selection is indistinguishable from the masks around it. Without masks shown, the user sees nothing at all.

**Every view picks its own colour:**

| View | Selection drawn as | Defined in |
| --- | --- | --- |
| Image, single FOV | white 1 px outline | `image_display.py` `update_patches` (`[1, 1, 1]`) |
| Image, map mode | white outline | `main_viewer.py` `_update_map_mask_highlights` (`[1.0, 1.0, 1.0]`) |
| Channel grid | white outline | `channel_grid_view.py` (`[1.0, 1.0, 1.0]`) |
| Mask outlines (default) | white 1 px | mask colour dropdown defaults to `"White"` |
| Gallery, centred cell | white outline | `ColorPicker` defaulting to `viewer.selection_outline_color`, an attribute that does not exist, so `#FFFFFF` |
| Scatter | red points on blue | `scatter_widget._DEFAULT_SELECTED_COLOR` / `_DEFAULT_POINT_COLOR` |
| Histogram | orange bars on blue; green gate band; red cutoff line | `histogram._OVERLAY_COLOR` / `_BASE_COLOR` / `_BAND_COLOR` |
| Image lasso while drawing | Matplotlib's default line, removed on release | `LassoSelector` defaults |

**Nothing says what to do next.** Everything that can use a selection lives in another panel, behind a checkbox:
- the Gallery;
- the Cell Table Editor;
- the plots' **Follow main viewer** box;
- the ROI manager.

v0.6.1-alpha1 added two separate status lines: one beside Lasso Select and one above the scatter plots. They state counts and causes, but they are two components for one concept, and this design replaces both.

**The selection is stored in three places that are kept in step by hand:**
- `ImageDisplay.selected_masks_label` (`(fov, mask, id)` triples);
- each plot's own index set;
- `viewer.linked_selection_indices` (#119).

The #119 and #135 fixes were all about keeping these copies consistent.

## Design

### 1. One visual identity: colour and form

- **One selection colour,** used wherever a selection is drawn: the image (single FOV, map mode, channel grid), the Gallery's centred-cell outline, the scatter's selected points and the histogram's selected bars. The plots' unselected colours stay neutral.
- **A different form from a mask outline:**
  - a mask outline stays a 1 px line;
  - a selected cell gets a thicker outline (2 px at factor 1, scaled with the downsample factor like mask outlines, via `scale_outline_thickness`) plus a light fill tint of the selection colour.
  
  A selection then stays recognisable even if a user sets their masks to the selection colour.
- **Default colour:** a saturated amber, `#FFB000`. It is not among `PREDEFINED_COLORS`, so it differs from the default white masks and from the pure RGB/CMY channel colours most panels use.

### 2. Colours and selection style as GUI parameters in Advanced Settings

**The outer structure stays:** the **Advanced Settings** accordion with its two tabs, **Data mapping** and **Advanced Settings**. Only the inner **Advanced Settings** tab changes. Today it is a flat `VBox` of four widgets; it becomes collapsible sections, an `Accordion` with one child per group. New parameters then get a group instead of lengthening one list, and closed groups take one line each in the 350 px panel.

| Section (inside the Advanced Settings tab) | Contents |
| --- | --- |
| **Display and performance** | cache size, pixel size, Downsample, frame format (unchanged, moved) |
| **Selection and colours** | selection colour (`ColorPicker`); selection outline width; selection fill opacity (0 turns the tint off); default colour for masks without a saved colour; scatter and histogram base colour for unselected cells |

**Persistence:** the new widgets are saved and restored with the widget state like the existing settings (`widget_restore` already handles `ColorPicker`). Changing a value re-renders the image and recolours open plots.

**The Gallery's own outline picker stays as an override:** it defaults to the global selection colour and follows it until the user changes it in the Gallery. The missing `viewer.selection_outline_color` attribute it reads today becomes the real accessor.

### 3. One selection bar

The toolbar row above the image, which now holds Lasso Select and its hint, becomes the selection bar. It reads the same whichever view made the selection:

```
[Lasso Select]  35 cells · scatter lasso · 12 in this FOV   [Gallery] [Annotate…] [Plots] [✕]
```

- **State text:**
  - the cell count;
  - the source (image click, image lasso, scatter lasso, histogram gate, heatmap cluster, Trace);
  - how many of the cells are in the FOV shown, omitted in map mode.
- **Actions,** each calling code that already exists:
  - **Gallery:** `cell_gallery.set_selected_cells(indices)` and reveal the Gallery tab.
  - **Annotate…:** open the Cell Table Editor with the selection; its Apply button already acts on image selections.
  - **Plots:** select the cells in the scatter and histogram views (the Follow main viewer path).
  - **✕:** clear the selection everywhere.
- **Other states,** carried over from v0.6.1-alpha1:
  - lasso armed;
  - no mask shown;
  - no masks loaded;
  - static canvas without `%matplotlib widget`;
  - a loop or Trace that found nothing (with the reason);
  - the anywidget front-end problem (shown in the scatter panel, where the empty plot is).

### 4. One selection model (second phase)

A `Selection` object on the viewer would hold:
- the cell-table indices;
- the `(fov, mask, id)` triples;
- the source.

All views render it and the bar reads it. It replaces `selected_masks_label`, the plots' own index sets and `linked_selection_indices`. It is done second so the visible part (1–3) ships first; 1–3 are written against a small accessor so phase 2 can swap the storage underneath.

## Implementation steps (phase 1: parts 1–3)

1. **Selection style:** `SELECTION_COLOR` default in `ueler/constants.py`, and a `SelectionStyle` (colour, outline width, fill opacity) on the viewer, read through one accessor.
2. **One drawing routine for the image:** `update_patches`, `_update_map_mask_highlights` and the channel grid's highlight draw the outline and tint with it, replacing the three hard-coded whites.
3. **Plots and Gallery:** scatter, histogram and Gallery read the selection colour; recolour on change.
4. **Advanced Settings tab:** its flat list → collapsible sections (**Display and performance**, **Selection and colours**); the outer accordion and the Data mapping tab are unchanged; persistence.
5. **Selection bar:** replaces `lasso_hint` and the scatter `_selection_status` line, with the v0.6.1-alpha1 cause detection folded in as states.
6. **Tests, docs** (user interface, display settings, cell table, scatter tutorial, FAQ), log, README.

## Decisions

- **Default selection colour:** amber `#FFB000`.
- **The Gallery's outline picker:** kept as an override that defaults to the global selection colour.
- **The per-plugin Main viewer / Gallery link checkboxes:** kept as they are in phase 1. Whether to fold them into the bar is revisited with the selection model (phase 2).
- **The lasso path is not kept on screen after release.** A selection drawn in its own colour, with a thicker outline and a fill, is visible on its own, so the path adds nothing but clutter. **Save loop as ROI** is dropped with it, since it needed the path.
- **Where the colour parameters live:** the outer Advanced Settings accordion and its Data mapping / Advanced Settings tabs stay as they are. The inner Advanced Settings tab is grouped into collapsible sections.

## **Update Summary: Global selection and the in-line bar, built first as an experiment (uncommitted)**

The maintainer asked to try the global selection before the colour work, to see whether a single "current selection" holds up. A selection can bypass the image entirely, for example a histogram gate sent straight to the Gallery, so the bar cannot be built on the image's selection alone. A small version of phase 2's model was therefore needed now. This round builds that model and the in-line bar (design part 3); parts 1–2 (the selection colour and the Advanced Settings sections) are still to do.

---

### **Actions Taken**
- **`SelectionHub` beside the existing copies, not instead of them:**
  `ueler/viewer/selection.py` holds the latest `CellSelection`: cell-table row indices, a source key and the set of targets that already show it (`delivered`). The last publish wins, as the image highlight already does (#119). `selected_masks_label`, the plots' `selected_indices` and `linked_selection_indices` are untouched, and so are the link boxes. The experiment can then be judged, or reverted, without the refactor. `viewer.selection_hub` is a lazily created property, so viewer stubs built with `__new__` get one too.
- **Only user actions publish:**
  - **Image:** click, ctrl-click, background click and right-click (`ImageDisplay.on_mouse_click`), and a lasso (`_on_lasso_selected`), through `viewer.publish_image_selection`. That converts the `(fov, mask, id)` triples with `_chart_common.viewer_selection_indices`.
  - **Scatter:** `_commit_scatter_selection(origin="scatter")`.
  - **Histogram:** `_apply_gate(publish=True)` and `clear_selection`.
  - **Heatmap:** `display_cells`.
  
  Programmatic paths do not publish: `set_mask_ids` from a plot link, `clear_patches` on a FOV switch, the histogram re-apply with `publish=False`, and pushes from the bar itself (`origin="hub"`, `show_external_selection`). Otherwise a FOV switch or a link would take the bar from the selection the user actually made.
- **The bar** (`ueler/viewer/selection_bar.py`) is a plain object holding an `HTML` text and four buttons. It is deliberately not a widget attribute of `ui_component`, so `save_widget_states` never persists its text. That also fixes the earlier `lasso_hint` widget, which would have been persisted.
  - **Layout:** the row has a fixed 32 px height and the text is a single ellipsised line, with the full text as its hover title. Filling or emptying the bar therefore never reflows the image.
  - **Messages:** they take precedence over the summary until removed, and a new selection removes them.
  - **Callbacks:** the buttons call `trigger(target)`, which also makes them testable under the ipywidgets stub.
- **Viewer side** (`main_viewer.py`):
  - **Summary:** `_selection_summary` gives the count, the source label, and `cells_in_shown_fov` (omitted in map mode) with "not outlined in the image" when `image` is not delivered.
  - **Bar actions:** `send_selection_to_image` (`sync_mask_highlights_from_selection`), `send_selection_to_gallery`, `send_selection_to_plots` (scatter and histogram, without pushing back), `clear_selection_everywhere`.
  - **FOV switch:** `_reapply_selection_highlights` clears an image-sourced selection, which `clear_patches` has just dropped, and otherwise refreshes the FOV count.
- **The two v0.6.1-alpha1 lines are folded in:**
  - `set_lasso_hint` writes the bar's message.
  - A successful loop hands over to the selection summary.
  - Trace failures go to the bar (`_chart_common.show_selection_message`).
  - The scatter panel line keeps only the lasso tip and the widget front-end warning.

---

### **What the experiment showed**

- **The global selection works for the case that motivated it.** On the real viewer with a cell table: image lasso → "35 cells · image lasso", Image ✓. Gallery button → Gallery ✓. Then a histogram gate with only the Gallery link → "23 cells · histogram gate · 23 in this FOV, not outlined in the image", Gallery ✓. Image button → the 23 cells outlined, Image ✓. ✕ → everything cleared.
- **The image can still show an older selection.** In the run above, after the histogram gate the image kept the 35 outlines of the earlier lasso while the bar described the 23 histogram cells. The hub is last-writer-wins, but the image's own copy is replaced only when something writes to it. Whether a new selection from elsewhere should clear the image's outlines, or the bar should say "the image shows an older selection", is the first question for the colour work and for phase 2.
- **"Gallery ✓" is only partly true above the Gallery's cell limit.** The Gallery samples a random subset when a selection exceeds **Max displayed cells** and only logs that. The bar should say "Gallery ✓ (20 of 35 shown)".
- **Annotate… is not in the bar yet.** The Cell Table Editor acts on the image's `selected_masks_label` only. A selection from a plot would have to be pushed into the image first, and in single-FOV mode that keeps only the cells of the FOV shown. This needs the editor to accept cell-table indices, which belongs to phase 2.
- **The heatmap's image highlight (`highlight_cells`) does not publish:** the FOV-switch re-apply calls it too, and cannot be told apart from a user action without a flag.

## **Update Summary: Unlink by clicking again; ✕ asks first (uncommitted)**

Follow-up after trying the bar: a selection could be sent to a view but not taken back out of it, and ✕ dropped a selection that may have taken several gates to build, without asking.

---

### **Actions Taken**
- **The target buttons toggle:**
  `toggle_selection_target(target)` sends the selection when the target does not show it, and removes it when it does:
  - **Image:** `clear_patches`.
  - **Gallery:** `set_selected_cells([])`.
  - **Plots:** scatter and histogram emptied with `origin="hub"` / `push_highlight=False`, so nothing is pushed back.
  
  `SelectionHub.mark_removed` drops the target from `delivered`. The selection itself stays in the bar and can be sent again.
- **The source view is not unlinkable:**
  `CellSelection.source_target` maps image sources to `image` and scatter/histogram to `plots` (the heatmap is not a target). That button shows ✓, is disabled, and its tooltip says "Selected in the …". Unlinking a selection from where it was made would delete it, which is what ✕ is for. `shows_in()` counts the source view, so an image-made selection is never called "not outlined in the image".
- **✕ goes through the shared confirmation dialog (#139):**
  "Unselect everything?", naming the cell count and source, with **Unselect** and **Cancel**. Only **Unselect** runs `clear_selection_everywhere`, which now also empties the Gallery when the selection was delivered there. Without a front end (`confirm` returns `False`) it clears directly, following the marker-set delete: the selection lives only in this session, and a dead button would be worse.
- **Known interaction, kept:**
  a plot's link boxes still forward every new selection from that plot. A view unlinked in the bar therefore receives the next selection from a linked plot again. Clearing the plots (unlinking Plots, or ✕) can also empty the Gallery through a plot's Gallery link, because clearing `selected_indices` is forwarded like any other change. Both follow from keeping the link boxes in phase 1.

---

### **Files Changed**
- `ueler/viewer/selection.py` — `SOURCE_TARGETS`, `CellSelection.source_target` / `shows_in`, `SelectionHub.mark_removed`.
- `ueler/viewer/selection_bar.py` — per-button state: ✓, disabled for the source, tooltips for send / remove / selected here.
- `ueler/viewer/main_viewer.py` — `toggle_selection_target`, `remove_selection_from_image` / `_gallery` / `_plots`, `ask_clear_selection`; `clear_selection_everywhere` also clears the Gallery.
- `tests/test_selection_hub.py` — 37 tests (unlink per target, source disabled, ✕ asks / clears without a dialog / does nothing with no selection).
- `docs/tutorials/user-interface.md` — the selection bar section.

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1573 tests, OK
- Real viewer: the toggle sequence and both dialog answers behaved as described in `doc/log.md`.

## **Update Summary: Mouse gestures in the image — background clicks, Ctrl-click, Ctrl-drag lasso, loops that unselect (uncommitted)**

Follow-up after using the bar: a left click on the background cleared the whole selection, so one mis-click lost a population that took several steps to build. Clicks and the lasso also behaved differently: a Ctrl-click toggled one cell, but a loop could only add, and a loop needed the **Lasso Select** toggle every time.

---

### **Actions Taken**
- **One gesture table in `ImageDisplay`:**
  - left click on a cell: select only that cell (unchanged);
  - left click on the background: nothing;
  - Ctrl + left click on a cell: add it, or take it out (unchanged, now decided on release);
  - Ctrl + left drag: a lasso without arming the toggle;
  - right click on the background: unselect everything through `clear_selection_everywhere(recipe=…)`, the same path as ✕ but without the dialog, because ◀ restores it. A right click on a cell does nothing (it used to clear everything too).
- **Ctrl press is decided on release:**
  `on_mouse_click` records a `_ctrl_gesture` (start point, screen pixels, vertices); `on_mouse_move` extends it and, once the mouse has moved `CTRL_DRAG_PIXELS` (5) screen pixels, draws the loop as a `Line2D`; `on_mouse_release` (newly connected to `button_release_event`) turns a moved gesture with ≥3 vertices into a lasso and anything shorter into the Ctrl-click toggle. Starting a `LassoSelector` mid-press was ruled out: it never sees the press that already fired, so it would only arm for the next drag. Ctrl is read from both `event.key` and `event.modifiers`, since front ends report it differently.
- **Loops toggle as a group:**
  `_select_in_loop` (shared by the toggle lasso and Ctrl-drag) unselects the cells inside the loop when every one of them is already selected, and otherwise adds them. Per-cell XOR was ruled out: drawing a larger loop to extend a selection would have unselected the part already selected. The lasso recipe now carries `action` (`add` / `remove`) and `added` or `removed`.
- **History:**
  clicks, Ctrl-clicks and both lassos were already published to the hub; the right-click clear now goes in with its recipe (`{"action": "right click on background"}`) instead of a bare clear, so the trail says what unselected.
- **Hint text:** the armed-lasso and empty-loop messages mention Ctrl + drag and the unselect rule.

---

### **Files Changed**
- `ueler/viewer/image_display.py` — gesture table, `_ctrl_held`, `_click_cell`, `_unselect_everything`, `on_mouse_release`, `_extend_ctrl_gesture` / `_cancel_ctrl_gesture`, `_select_in_loop` / `_report_loop`, `_lasso_recipe(action=)`.
- `ueler/viewer/main_viewer.py` — `clear_selection_everywhere(recipe=None)`; lasso hint wording.
- `tests/test_image_click_gestures.py` — new, 12 tests; `tests/test_selection_hub.py` — the clear recipe in the history; `tests/test_issue119_selection_across_fov.py` — the Ctrl-click test sends the release.
- `docs/tutorials/user-interface.md`, `docs/tutorials/cell-table.md`, `docs/tutorials/cell-sets.md` — the gestures.

---

### **Tests**
```bash
python -m unittest discover -s tests -t .
```
- ✅ 1613 tests, OK; `mkdocs build --strict` passes.
- Not yet tried in a browser: Ctrl + drag through ipympl, and macOS Ctrl-click (which browsers also treat as a context-menu click).
