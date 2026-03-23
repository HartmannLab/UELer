# Issue #75 — Channel Color Legend

## Summary
Add a color-coded legend that lists displayed channels using the same colors as the rendered image. Provide both an on-image overlay and an adjacent UI legend, with a single toggle to show/hide the legend. The legend must update when channel visibility or colors change.

## Goals
- Show a per-channel legend with the channel name rendered in the matching display color.
- Support both an overlay legend (on the image) and an adjacent legend (in the UI).
- Legend content should reflect only visible channels (respect channel visibility toggles).
- Provide a user-facing toggle to show/hide the legend (single toggle controls both placements).
- Keep the legend legible regardless of background (neutral background box).

## Proposed Approach
1. Add legend controls to the viewer UI (toggle + adjacent legend container).
2. Add a viewer helper that returns legend entries (channel name + RGB) derived from existing channel color controls.
3. Render an overlay legend in `ImageDisplay` as a reusable artist updated on each display refresh.
4. Populate the adjacent legend widget using HTML with a neutral background and colored labels.
5. Add unit tests for legend data generation and display update hooks.

## Implementation Steps
1. **UI wiring**
   - Add a `show_channel_legend` checkbox in `ui_components`.
   - Add a `channel_legend_box` widget (HTML or VBox) for the adjacent legend.
   - Ensure the legend box sits near channel controls and updates when selections change.
2. **Legend data helper**
   - Implement `ImageMaskViewer.get_channel_legend_entries()` returning ordered `(name, rgb)` entries.
   - Use the same color source as rendering (`color_controls` + `predefined_colors`).
   - Filter by `visible_channels` (respect channel visibility checkboxes).
3. **Overlay legend**
   - Add a legend artist to `ImageDisplay` and update it during `update_display`.
   - Use a neutral background box for legibility.
   - Clear the legend when disabled or when no channels are visible.
4. **Adjacent legend**
   - Build HTML entries (colored labels) inside `channel_legend_box`.
   - Keep it in sync with the overlay legend.
5. **Testing**
   - Add unit tests for legend entry generation.
   - Add a display-level test to confirm legend updates when channel colors/visibility change.
6. **Docs and tracking**
   - Update `dev_note/github_issues.md` with the planning link and completion summary.
   - Update `doc/log.md` and `README.md` after tests pass.

## Validation
```bash
python -m unittest tests/test_rendering.py
```

Manual:
1. Launch `/script/run_ueler.ipynb`.
2. Select multiple channels and confirm the overlay + UI legend appear.
3. Toggle a channel off; confirm it disappears from the legend.
4. Change a channel color; confirm the legend updates.
5. Toggle the legend off; confirm both legends hide.

## Files to Touch
- `ueler/viewer/ui_components.py`
- `ueler/viewer/main_viewer.py`
- `ueler/viewer/image_display.py`
- `tests/test_rendering.py`
- `dev_note/github_issues.md`
- `doc/log.md`
- `README.md`
