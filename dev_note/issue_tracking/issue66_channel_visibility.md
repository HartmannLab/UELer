# Issue #66 — Per-channel visibility checkbox

## Summary
Add a per-channel on/off checkbox in the channel controls so users can temporarily hide a channel without removing it from the multi-select channel list. Visibility state should be session-scoped and must not alter marker set/channel selection behavior.

## Goals
- Provide a visibility checkbox next to each displayed channel row.
- Hide/show channels immediately in the composite render without altering the channel selection list.
- Preserve channel color and contrast settings when toggling visibility.
- Avoid regressions to marker set loading and channel palette behavior.

## Plan
1. **UI controls**
   - Add `channel_visibility_controls` to `ui_components` to store per-channel checkboxes.
   - Render each channel row as a compact header (`checkbox + color dropdown`) followed by min/max sliders.
   - Default visibility to `True` for newly selected channels.

2. **Viewer rendering**
   - Add a helper on `ImageMaskViewer` to derive `visible_channels` from the selected list + visibility checkboxes.
   - Use `visible_channels` in `update_display` (and map-mode rendering) so hidden channels are excluded from the composite.

3. **Lifecycle clean-up**
   - Remove visibility controls for channels that disappear on FOV change, mirroring existing color/contrast cleanup.

4. **Validation**
   - Manual notebook validation via `/script/run_ueler.ipynb`:
     1. Select multiple channels.
     2. Toggle a channel off; confirm it disappears while selection remains.
     3. Toggle back on; confirm color/contrast settings persist.

## Files to touch
- `ueler/viewer/ui_components.py` — add visibility control registry.
- `ueler/viewer/main_viewer.py` — render checkboxes and filter visible channels.
- `dev_note/github_issues.md` — add planning link and completion summary.
- `doc/log.md` and `README.md` — document the feature.
