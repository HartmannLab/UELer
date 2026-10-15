# Viewer Runtime and UI Behavior

## Context
These notes cover the main viewer runtime, downsampling behavior, channel controls, tooltips, and notebook-specific behavior. The key focus is how the viewer loads FOVs, updates its display, and synchronizes UI controls with cached image and overlay data.

## Key decisions
- Maintain a lazy load pipeline for FOVs and overlays, with LRU-based caches for image, mask, and annotation data.
- Use downsampling factors derived from viewport and FOV size to keep navigation responsive.
- Respect dataset-specific keys for tooltips instead of hard-coded column names.
- Offer per-channel visibility toggles, a channel legend, and a channel grid display mode without altering the underlying selection list.
- Provide a VS Code scatter fallback when widget rendering fails.

## Current status
- FOV load cycle is documented end-to-end, including cache and plugin update steps.
- Downsampling flow and `select_downsample_factor` behavior are captured for main viewer and ROI thumbnails.
- Tooltip lookup uses viewer-configured keys. `_lookup_mask_hit` first materialises the tooltip's columns (the displayed channels plus the ticked annotation labels) through `ensure_columns`, so a lazy Parquet cell table has the values to show; each column is read once, on the first hover that needs it. Both tooltips (`image_display.py`, `channel_grid_view.py`) use the freshly resolved `hit.cell_record` on every hover rather than a row cached per cell, which could predate a column materialised since.
- VS Code scatter fallback uses a static Matplotlib plot when widget front-ends fail.
- Per-channel visibility toggles and channel color legend features are implemented and tested.
- Channel grid display mode (Issue #76) is implemented: a "Channel grid view" checkbox renders each visible channel as a separate labelled pane in a synchronised matplotlib subplot grid.
- UI layout hardening for Issue #85 is in place: overflow-prone wrappers now use shrink-safe constraints (`max_width: 99%`, `min_width: 0`, `box_sizing: border-box`) in core viewer panels and selected plugin panels to reduce unnecessary horizontal scrollbars, with follow-up tuning for marker-set and channel slider controls.
- Reply 2 to Issue #85 adds a content-widget policy (`width/max_width: calc(100% - 5px)`) so controls shrink inside the unchanged 99%-width containers.
- Reply 3 to Issue #85 tightens remaining channel-panel row composition (marker-set buttons, confirm-deletion row, legend wrapping, compact checkbox/dropdown spacing, longer usable slider tracks) without changing the 99% container policy.
- Reply 4 to Issue #85 reorganizes each channel into a three-row block (header with checkbox+marker name+color, then Min and Max rows), removing channel-name repetition in slider labels.
- Reply 5 to Issue #85 restores shared scrolling behavior for channel controls (no per-channel internal scrollers) and offsets the color dropdown 5px left in the header row.
- Reply 6 to Issue #85 enforces parent vertical-scroll triggering for long channel lists by preventing grouped channel rows from shrinking to fit the container height.
- Reply 7 to Issue #85 removes marker-set channel duplication at load time by de-duplicating marker channels (save/update/apply) and eliminating redundant channel-control rebuilds in the marker-set apply flow.
- Issue #139 replaces the marker-set **Confirm Deletion** checkbox with a modal: `ueler/viewer/confirm_dialog.py` holds a reusable `ConfirmDialog`, mounted once by `build_layout` and hidden until asked. It is deliberately not a `Widget` subclass, so `save_widget_states` does not persist its stylesheet; it positions itself with a CSS class rather than a `Layout` (which has no `position` trait); and it falls back to an unconfirmed delete where no real front end exists. Other destructive buttons (saved mask colour sets, ROI delete, export-config delete) can adopt it unchanged.
- Issue #139 reply 1 extends that modal to the five destructive actions that write to disk: the saved export config (`plugin/export_fovs.py`), the saved mask colour set (`plugin/mask_painter.py`), the checkpoint (`plugin/cell_annotation.py`), the ROI (`plugin/roi_manager_plugin.py`) and the saved annotation palette (`main_viewer.py`). The lookup rule lives in `confirm_dialog.confirm_via(ui_component, ...)`; `ImageMaskViewer.confirm` delegates to it and `PluginBase.confirm` delegates through that, resolving `main_viewer` before `viewer` because `PluginBase` and the concrete plugins disagree on the name. Where no dialog exists these five refuse and say so, unlike `delete_marker_set`, which still deletes unconfirmed — marker sets never leave the session, these five unlink files. Cheap, repeatable actions (**Clear selection**, log console **Clear**, the class-list `×`) are deliberately not guarded; the **Overwrite** buttons and the two irreversible in-session actions are catalogued in the planning document but not yet wired.
- Issue #104 adds a full-screen button at the top-left of the viewer (`ueler/viewer/fullscreen_toggle.py`). `display_ui` tags the root `VBox` with the `ueler-root` class and puts a `FullscreenToggle` anywidget in a toolbar row as the root's first child. The ESM finds what to enlarge with `el.closest('.ueler-root')` and never walks the DOM by size or framework class, which is what broke the earlier #104 attempts in VS Code. In a VS Code webview (detected by its `vscode-*` body class) the button does not try full screen: it opens an inline notice (`.ueler-fs-notice`, `role="dialog"`) saying full screen is not supported there and pointing to JupyterLab / JupyterHub, with **Enlarge in place instead** and **Close**; Esc or the button also close it. Elsewhere it tries native `requestFullscreen()` first. When that is refused, a VS Code webview (detected by its `vscode-*` body class) gets `ueler-expanded`, which keeps the viewer in the notebook flow. Its frame height is `min(innerHeight, screen.availHeight − 160)`, taken once on entry and never re-measured, because the webview is sized to the notebook content and its `innerHeight` grows with the viewer. Content taller than the frame scrolls inside it (`overflow: auto`, with `flex-shrink: 0` on the root's children). The frame is **not** anchored to the editor: a widget cannot set the notebook's scroll position, and the indirect routes (vetoing or rewriting wheel events, clicking an in-page link) all failed in VS Code. The in-page link one is destructive: it scrolls the webview's own document and blanks every output in it. See the issue #104 tracking document before trying again. A probe run in VS Code settled it: the webview is as tall as the notebook (`innerHeight` = `docScrollHeight`), never scrolls itself (`scrollTop` 0), and the root's `top` inside it never changes, so an output can neither read nor set the notebook's scroll position. Only `intersectionRatio` crosses the boundary. A frame-fit built on it was too small a gain to keep. VS Code draws code-cell editors above the output webview, so an overlay there would sit under them. Any other page gets `ueler-maximized`, a `position: fixed` overlay. Esc leaves every mode. The mode is synced to Python as `fullscreen_toggle.mode`, and the available space as `fullscreen_toggle.viewport`. `attach_canvas_fit` uses that space to resize the ipympl figure (`set_size_inches(..., forward=True)`) and restores it on exit. The button is left out when anywidget cannot render. **Default layout:** the button's ESM reports the root's width in every mode (`host_width`, via `ResizeObserver`), and `attach_canvas_fit` keeps the image figure a square filling the width left by the left panel (350 px) and the side column (576 px when present), so the side column sits at the right edge. While enlarged it fills the `viewport` instead, and returns to the square on exit. Without anywidget the figure keeps its 6 × 6 in size. **Hiding the side column:** `ui_component.side_panel_hidden` (a `ToggleButton`, so `widget_states.json` persists it) is mounted in a 24 px column between the image and the side column, only when there are side plugins. `display_ui` applies it once (the saved value is restored before display) and on every change: it sets `side_plot.layout.display`, swaps the icon, refits through the function `attach_canvas_fit` returns, whose `side_width_px` is a callable (24 + 576 px shown, 24 px hidden), and on show calls `_restore_heatmap`.

## Open items
- Keep UI/UX changes consistent with map mode and plugin rendering updates.
- Validate any future changes to channel controls or tooltip semantics against existing tests.

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/61
- https://github.com/HartmannLab/UELer/issues/64
- https://github.com/HartmannLab/UELer/issues/66
- https://github.com/HartmannLab/UELer/issues/75
- https://github.com/HartmannLab/UELer/issues/76
- https://github.com/HartmannLab/UELer/issues/85
- https://github.com/HartmannLab/UELer/issues/104
- https://github.com/HartmannLab/UELer/issues/139

## Key source links
- [dev_note/FOV_load_cycle.md](dev_note/FOV_load_cycle.md)
- [dev_note/main_viewer.md](dev_note/main_viewer.md)
- [dev_note/issue_tracking/issue139_confirm_dialog.md](dev_note/issue_tracking/issue139_confirm_dialog.md)
- [dev_note/issue_tracking/issue104_fullscreen_mode.md](dev_note/issue_tracking/issue104_fullscreen_mode.md)
