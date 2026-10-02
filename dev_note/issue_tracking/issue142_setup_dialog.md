# Issue #142 — guided data mapping and advanced settings

[GitHub issue #142](https://github.com/HartmannLab/UELer/issues/142)

## Problem

Two separate failures, both of which look to the user like "UELer does not work with my data".

**The data-mapping keys are free text with hard-coded defaults.** `X key:`, `Y key:`, `Label key:`, `Mask key:` and `Fov key:` are `Text` widgets pre-filled with `centroid-1`, `centroid-0`, `label`, `whole_cell` and `fov` — the column names of one lab's pipeline. A table that calls its coordinates `x`/`y` produces no cells at all, and the only way to discover why is to know that the **Advanced Settings** accordion exists, open it, find the **Data mapping** tab, and type the right names from memory. A typo is indistinguishable from a wrong dataset: both render an empty overlay with nothing in the log.

**Nothing points the user at the accordion in the first place.** `Pixel Size (nm):` defaults to `390`, the MIBI detector pitch, and is applied to every dataset with no warning. A wrong value silently mis-scales the scale bar in the viewer *and* in every batch-exported image, and it is the kind of error nobody notices until a figure is in review.

Both are discovery problems, and the fix for both is to ask once, at the point where the answer is knowable, instead of waiting for the user to go looking.

## Design

### 1. The keys become dropdowns populated from the data

Guessing a column name is the failure; offering the real ones removes it. The data needed is already available and already free:

* **Cell-table columns** — `ImageMaskViewer.cell_table_schema` (#141) is an ordered `{column: dtype}` mapping covering *every* column the table has, including the ones a lazy Parquet source has not materialised yet. That property exists precisely because "every marker, cluster and annotation dropdown in the GUI is built from them", and a Parquet schema read costs 0.13 MB in 2 requests against 91 MB for the file. So the dropdown costs nothing, and because the dtypes come along, `X key:` and `Y key:` offer **numeric columns only**.
* **Mask suffixes** — `load_masks_for_fov` already derives these by stripping the `{fov}_` prefix off each `{fov}_{suffix}.tif[f]` in the masks folder.

**The timing wrinkle, and why there is a new loader function.** `mask_names_set` is filled *as a side effect of reading the TIFFs*, one FOV at a time, so at the moment the dropdown needs its options the set holds the suffixes of however many FOVs have been visited — usually one, sometimes none. `ueler.data_loader.discover_mask_suffixes` re-derives the same list from the same filename convention by globbing alone, opening no images, and the viewer unions it with whatever `mask_names_set` has already accumulated.

### 2. Changing the options must never change the user's answer behind their back

`ipywidgets.Dropdown` rejects a value outside its options, which makes a naive "assign new options" a crash (or a silent retarget) in three places: restoring `widget_states.json`, loading a second cell table, and the images-only session that has no columns at all. `ueler/viewer/data_mapping.py:apply_options` is the single rule:

* the current value stays selected if the data has it;
* otherwise the first **preferred** alias the data does have is selected — this is what pre-fills `x`/`y` for a table that does not use `centroid-1`/`centroid-0`, and it only ever fires when the current value names a column that does not exist, i.e. when the viewer is already broken;
* otherwise the current value is kept as an option of its own, so nothing is silently retargeted and the dialog can say the column was not found;
* an empty discovery leaves the widget untouched, which is what keeps an images-only session working.

`load_widget_states` grows one guard around the same invariant: a saved key that is no longer a column is re-added as an option rather than raising `TraitError` and aborting the whole restore.

### 3. The setup dialog

`ueler/viewer/setup_dialog.py` holds `SetupDialog`, a sibling of `ConfirmDialog` (#139) rather than an extension of it. `ConfirmDialog.ask()` is strictly yes/no, carries no body, and deliberately refuses to open over an open dialog; a multi-step form needs all three of those to be different. What the two share is the mount convention — a plain non-`Widget` holder, a `<style>` block mounted with the view, `position: fixed` arriving as a CSS class because `ipywidgets.Layout` has no `position` trait — and that convention is what `setup_dialog.py` copies.

**Steps are built from what was loaded**, by `build_setup_steps`:

| Step key | Shown when | Fields |
| --- | --- | --- |
| `images` | always | `Cache Size:`, `Pixel Size (nm):`, `Downsample` |
| `masks` | `masks_available` | `Mask key:` |
| `cell_table` | a cell table is attached | `X key:`, `Y key:`, `Label key:`, `Fov key:` |

The dialog shows **the accordion's own widget objects**, not copies. ipywidgets renders a second view of the same model, so the two stay in sync with no mirroring code and the accordion remains the single source of truth — whatever the user picks in the dialog is already the live setting, and remains editable afterwards.

**Suppression is per step, not per session.** `.UELer/setup_dialog.json` records which step keys have been shown. This is what makes the two entry points behave: `run_viewer()` on images alone shows the `images` step and records it, and a later `load_cell_table()` in the same notebook shows only the `cell_table` step rather than asking everything again. Closing the dialog counts as shown, whether the user pressed **Done** or **Skip** — re-asking a question the user explicitly dismissed is worse than not asking.

### 4. Where it fires

`after_all_plugins_loaded` is the only hook both `run_viewer()` and `load_cell_table()` call after `display_ui()`, so it is where the refresh and the dialog go, in that order. The refresh also runs once at the end of `__init__` so that an images-only session gets its mask dropdown populated even when the viewer is constructed directly. Both calls are wrapped: a dataset that defeats discovery must never stop the viewer from opening.

## Implementation steps

1. `ueler/data_loader.py` — add `discover_mask_suffixes(masks_folder, fov_names, limit=...)`.
2. `ueler/viewer/data_mapping.py` — new: `KEY_FIELDS`, `column_options`, `apply_options`, `ensure_option`.
3. `ueler/viewer/setup_dialog.py` — new: `SetupStep`, `SetupDialog`, `build_setup_steps`, `load_seen_steps`, `record_seen_steps`.
4. `ueler/viewer/ui_components.py` — the five key `Text` widgets become `Dropdown`; mount `SetupDialog` beside `ConfirmDialog` and add its view in `display_ui`.
5. `ueler/viewer/main_viewer.py` — `discover_mask_suffixes`, `refresh_data_mapping_options`, `maybe_show_setup_dialog`; call sites in `__init__` and `after_all_plugins_loaded`; the `load_widget_states` guard.
6. `tests/test_issue142_setup_dialog.py`.
7. Docs: `docs/tutorials/user-interface.md`, `docs/tutorials/display-settings.md`, `doc/log.md`, `README.md`.

## Out of scope

* **Making the dialog draggable or turning plugin panels into free-floating windows.** Floating is free with the CSS-class trick already in use, but dragging is not reachable from `ipywidgets` at all — `Layout` has no `position`/`top`/`left`, boxes emit no pointer events, a kernel round trip per `mousemove` is unusable, and a `<script>` inside an `HTML` widget never executes. It needs an `anywidget` `_esm` module and a real window manager (z-order, focus, off-screen recovery), plus coexistence with the ipympl canvas's pointer capture and with `update_wide_plugin_panel`'s pane cache. Separate issue — now planned in [window_frame_floating_panels.md](window_frame_floating_panels.md), which revises the "not reachable" conclusion above: `anywidget` is already a hard dependency and the channel picker already ships a hand-written ESM with drag-and-drop.
* **Validating that the chosen columns are *right*** (that `X key:` really holds coordinates and not cluster ids). The dropdown restricts X/Y to numeric columns and stops there.
