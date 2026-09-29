# Issue #139 — a pop-up confirmation dialog for destructive actions

[GitHub issue #139](https://github.com/HartmannLab/UELer/issues/139)

## Problem

Deleting a marker set (the "channel preset" of the issue text) is a two-step ritual that reads like a bug: the user picks the set in the **Marker Set:** dropdown, clicks **Delete Marker Set** — and nothing happens. The only feedback is a log line, `Please check 'Confirm Deletion' to delete the marker set.`, which is invisible unless the viewer was constructed with `debug=True` and the log console is open. The user then has to find a **Confirm Deletion** checkbox further down the panel, tick it, and click **Delete Marker Set** again.

The guard itself is worth keeping — deleting a marker set throws away a channel/colour/contrast combination that cannot be reconstructed from anything else. What is wrong is that the confirmation is *detached from the action*: it sits in a different row, it stays ticked between actions if the user forgets it, and it never names the thing being deleted.

The relevant code before this change:

* `ueler/viewer/ui_components.py` — `self.delete_confirmation_checkbox` and its slot in `marker_set_controls_panel`.
* `ueler/viewer/main_viewer.py:delete_marker_set` — reads the checkbox, refuses when unticked, resets it after a successful delete.

## Why not `ipyoverlay`

The issue suggests `ipyoverlay`. I evaluated it (0.2.1, the current release) and decided against it:

* **It pulls in two front-end stacks UELer does not have.** `ipyoverlay` declares `ipyvuetify`, `plotly`, `ipywidgets` and `matplotlib`. Only `matplotlib` is already installed in the `ueler` environment — `ipyvuetify` (plus `ipyvue`) and `plotly` would become new hard runtime dependencies, each with its own Jupyter front-end extension that has to be present in every notebook front-end a user runs UELer in. That is a large install-and-support surface for a yes/no prompt.
* **It solves a different problem.** `OverlayContainer` is a `VuetifyTemplate` that wraps one *background* widget and renders children over it at pixel coordinates, with optional leader lines connecting a child to a point on a Matplotlib axes or a Plotly figure. It is built for details-on-demand annotations over a plot, not for a modal that blocks the UI until a question is answered. Adopting it would mean turning the viewer's root container into a Vuetify template and hand-computing the dialog's position.

The dialog itself needs nothing more than a fixed-position element over the viewer, which plain `ipywidgets` can express through a CSS class. So this change adds **no new dependency**.

## Design

### A reusable component, not a one-off

`ueler/viewer/confirm_dialog.py` holds `ConfirmDialog`, a small component with no knowledge of marker sets. Deleting a marker set is the first caller; the saved mask colour sets in the Mask Painter, ROI deletion and the export-config delete are equally destructive and can adopt it later without touching this module.

`ConfirmDialog` is deliberately **not** a `Widget` subclass. `ImageMaskViewer.save_widget_states` walks `vars(self.ui_component)` and persists the `.value` of every `Widget` it finds; an `HTML` widget holding the dialog's stylesheet would otherwise be written into `widget_states.json` as a blob of CSS and restored over itself on the next launch. A plain object holding widgets internally is skipped by that walk.

### How it renders

The dialog is a scrim (a full-viewport translucent backdrop) containing a card. The scrim is a `Box` carrying the CSS class `ueler-confirm-scrim`, whose rule is `position: fixed; inset: 0`. `ipywidgets.Layout` has no `position` trait, so the rule has to arrive as a class: a single `HTML` widget holding a `<style>` block is mounted once with the dialog, and `add_class` attaches the classes to the boxes.

Visibility is `view.layout.display`, toggled between `'none'` and `''`. A `display: none` flex item is out of layout entirely, so a closed dialog contributes neither height nor the root container's 12px gap, and the `<style>` inside it still applies — a `<style>` element is live wherever it sits in the document.

Colours come from the JupyterLab CSS variables (`--jp-layout-color1`, `--jp-ui-font-color1`, `--jp-border-color2`) with literal fallbacks, so the card follows the notebook's light/dark theme.

### Where it is mounted

`build_layout` appends `viewer.ui_component.confirm_dialog.view` to `root_children`. The scrim is `position: fixed`, so it covers the browser viewport rather than the 350px left panel where the button lives — a dialog clipped to the left panel would be unreadable.

### What it asks

The card names the thing being deleted: **Delete marker set "CD8 panel"?** over a one-line explanation that the action cannot be undone, then **Cancel** and a red **Delete**. The set name is HTML-escaped — marker set names are user-supplied strings.

### Behaviour on the Python side

`delete_marker_set(button)` becomes a thin handler: it resolves the selected name, refuses (as before) when nothing is selected, and otherwise opens the dialog. The actual mutation moves to `_delete_marker_set_confirmed(set_name)`, which the dialog's **Delete** button invokes.

Two consequences worth stating:

* **The name is captured when the dialog opens, not when it is confirmed.** If the dropdown changed underneath, the set the user was shown is the set that gets deleted. `_delete_marker_set_confirmed` re-checks membership, so a set deleted by some other path in the meantime is a logged warning, not a `KeyError`.
* **A second click while the dialog is open is ignored.** `ask()` on an open dialog is a no-op, so the user cannot stack two confirmations.

When no dialog is available — the `ipywidgets` fallback shim in `ui_components.py` builds widgets without `add_class` — `delete_marker_set` deletes directly. That environment has no way to present a confirmation, and a permanently dead Delete button is worse than an unconfirmed one.

### What is removed

`delete_confirmation_checkbox` goes away, along with its row in `marker_set_controls_panel`. Old `widget_states.json` files that still carry the key are harmless: `load_widget_states` only restores keys for which `hasattr(self.ui_component, attr_name)` holds, and silently ignores the rest.

## Implementation steps

1. Add `ueler/viewer/confirm_dialog.py` with `ConfirmDialog` (`ask`, `confirm`, `cancel`, `is_open`, `view`) and the stylesheet.
2. Construct `self.confirm_dialog` in `uicomponents.__init__`; drop `delete_confirmation_checkbox` and its slot in `marker_set_controls_panel`.
3. Mount `confirm_dialog.view` in `build_layout`'s `root_children`.
4. Split `ImageMaskViewer.delete_marker_set` into the handler and `_delete_marker_set_confirmed`.
5. Add `tests/test_confirm_dialog.py` covering the component and the marker-set delete flow.
6. Update `docs/tutorials/basic-usage.md` and `docs/tutorials/user-interface.md`, which both document the checkbox.
7. Update `doc/log.md`, `README.md`, `dev_note/topic_viewer_runtime_ui.md` and append the issue report to `dev_note/github_issues.md`.

---

## Follow-up (#139 reply 1) — extending the dialog to destructive actions in plugins

The reply asks for the dialog to cover other destructive actions, naming the saved export config as an example, and asks first for a survey. This section is that survey. **Nothing below is implemented**; the reply asks for confirmation before any of it is.

### Method

Every `Button` in `ueler/` whose description is a destructive verb (`Delete`, `Remove`, `Clear`, `Reset`, `Discard`, `Overwrite`) was located, its handler read, and the handler classified by **what the user loses and whether they can get it back**. That question, not the word on the button, is what decides whether a confirmation earns its interruption: a dialog in front of a cheap, repeatable action trains the user to dismiss dialogs, which costs them the one that matters.

A second sweep looked for destructive work reachable *without* such a button — `path.unlink()`, `os.remove`, `shutil.rmtree` and registry `pop()` — to catch anything a button label hides.

### Tier A — writes to disk, unrecoverable. These are the ones worth a dialog.

Each of these removes or overwrites a file the moment the button is clicked. None of them asks anything today; the marker-set checkbox was the only guard in the codebase and #139 replaced it.

| # | Where | Control | Handler | What is destroyed |
|---|---|---|---|---|
| A1 | **Export FOVs** plugin | **Delete** (saved config) | `plugin/export_fovs.py:1204` `_delete_export_config` → `unlink()` at :1224 | The config file plus its registry entry. *This is the example the reply names.* |
| A2 | **Mask Painter** plugin | **Delete** (saved color set) | `plugin/mask_painter.py:1254` `delete_saved_color_set` → `unlink()` at :1262 | The colour-set file plus its registry entry. |
| A3 | **Cell Annotation** plugin | **Delete selected** (checkpoint) | `plugin/cell_annotation.py:482` `_on_delete_button` → `checkpoint_store.py:174` `delete_checkpoint` → `unlink()` at :190 | An `.h5ad` holding a whole analysis step. The heaviest loss in the list. |
| A4 | **ROI Manager** plugin | **Delete** (selected ROI) | `plugin/roi_manager_plugin.py:2107` `_delete_selected_roi` → `roi_manager.py:442` `delete_roi` | The ROI record. `_set_table` defaults to `persist=True`, so the row is written out of the CSV immediately — the plugin's **Undo** covers shape drawing, not this. |
| A5 | Main viewer (not a plugin, same panel family) | **Delete** (annotation palette) | `main_viewer.py:3668` `delete_saved_annotation_palette` → `unlink()` at :3676 | The palette file plus its registry entry. Listed because it is the same action as A1/A2 and would look arbitrary left out. |

### Tier A′ — silent overwrite of a saved file

Three **Overwrite** buttons replace the contents of a named saved file with no prompt: the annotation palette (`ui_components.py:930`), the heatmap (`plugin/heatmap.py:319`) and the Mask Painter's saved sets (`plugin/mask_painter.py:2786`). The loss is identical to a delete — the previous contents are gone — but the button does not read as destructive, which arguably makes the confirmation *more* valuable here, not less. Worth a decision either way.

### Tier B — irreversible for the session, nothing on disk. Judgement call.

| # | Where | Control | Handler | Note |
|---|---|---|---|---|
| B1 | **Heatmap** | **Remove selected** (meta-cluster) | `plugin/heatmap_layers.py:1434` `remove_meta_cluster` | Drops the cluster's name and colour and reassigns every member cell to unassigned. Hand-curated work, no undo. The strongest Tier-B candidate. |
| B2 | **Scatter / Chart** | **Clear all** | `plugin/chart.py:696` `_clear_all_scatter_views` | Disposes every scatter view and forgets the configured pairs. Rebuildable, but a mis-click after configuring several pairs is genuinely annoying. |

### Tier C — cheap to redo. These should *not* get a dialog.

Chart **Remove** (one scatter, `chart.py:862`), Chart **Clear selection** (`chart.py:876`), Histogram **Clear selection** (`histogram.py:1314`), log console **Clear** (`log_console.py:102`), and the Mask Painter class-list `×` (`mask_painter.py:2423` `_on_remove_requested` — removes a class from the *active list*, touching no data). Each is undone by repeating the action that created the state.

### Three things to settle before implementing

1. **Plugins have no route to the dialog.** `PluginBase.__init__` stores the viewer as `self.viewer`, while every concrete plugin separately assigns `self.main_viewer` (verified across all eleven). Reaching through `self.main_viewer.ui_component.confirm_dialog` at each call site would spread that inconsistency and repeat the `getattr` fallback five times. A single `PluginBase.confirm(message, on_confirm, **kw)` helper would give one call site, one fallback rule, and one place to test.
2. **What should the fallback do when no dialog exists?** For marker sets the answer was "delete anyway", because a dead button is worse than an unconfirmed one and nothing left the process. For Tier A that answer is less comfortable — these unlink files. The alternative is to refuse and say so in the plugin's own status line, which every Tier-A plugin already has. This needs a decision, and it may differ per tier.
3. **One instance, mounted at the viewer root.** Footer and accordion plugins render inside that root, so the `position: fixed` scrim covers them. If a plugin can be displayed standalone in its own notebook cell, it would have no dialog and would hit whichever fallback rule (2) settles on.

### Suggested scope, if this goes ahead

Tier A (five sites) plus the `PluginBase.confirm` helper, as one change with tests per site. Tier A′ and Tier B as a separate decision, since both are more about taste than about data loss. Tier C explicitly left alone, and said so in the log so it does not get "fixed" later.

### What was implemented

Tier A and the helper, as scoped above and confirmed by the developer. Tier A′ (the three **Overwrite** buttons) and Tier B were left for a separate decision, and Tier C was deliberately left alone.

The three open questions were answered as follows:

1. **`PluginBase.confirm` was added**, with `PluginBase._confirm_host` resolving `main_viewer` first and `viewer` second so a call site never has to know which name its plugin uses.
2. **No dialog means refuse.** Every Tier-A handler reports "Cannot delete: no confirmation dialog is available in this environment" in its own status line and leaves the file alone. `delete_marker_set` keeps its original delete-anyway fallback, and the difference is deliberate: marker sets exist only in `viewer.marker_sets` for the life of the session, so nothing there outlives the process, while every Tier-A action unlinks a file.
3. **No plugin is displayed in its own cell**, so the single dialog mounted at the viewer root covers every plugin surface and no second mount point is needed.

One structural change fell out of testing. The dialog lookup started on `ImageMaskViewer.confirm`, but `tests/test_export_fovs_mask_customization.py` installs its own `sys.modules` shims and cannot import the viewer at all, so a test helper could not reach it. The rule now lives in `confirm_dialog.confirm_via(ui_component, ...)`, which `ImageMaskViewer.confirm` delegates to — one definition of "where the dialog lives and what happens when it is missing", reachable without dragging in the viewer.

### Follow-up implementation steps

1. Add `confirm_via` and a module-level `escape_name` to `ueler/viewer/confirm_dialog.py`.
2. Add `ImageMaskViewer.confirm` (delegating to `confirm_via`) and route `delete_marker_set` through it.
3. Add `PluginBase.confirm` and `PluginBase._confirm_host`.
4. Split each Tier-A handler into an asking half and a `_..._confirmed` half that captures its target at ask time.
5. Add `tests/confirm_support.py` and `tests/test_confirm_destructive_actions.py`; extend the export-config tests with the confirm, cancel and no-dialog cases.
6. Update `docs/tutorials/roi-manager.md`, `docs/tutorials/export.md` and `docs/tutorials/clustering-annotation.md`, which document three of the five buttons.
