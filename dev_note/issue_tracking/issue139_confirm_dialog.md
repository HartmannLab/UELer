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
