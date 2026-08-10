# Issue #126 — The selected-channel "tags" are not draggable

> GitHub issue: [#126](https://github.com/HartmannLab/UELer/issues/126)
> Status: implemented (see *Implementation* below)
> Follows on from [#125](issue125_dropdown_clipping.md), which introduced the widget this
> regressed in.

## Problem

> The "tags" in the channel selection UI are not draggable. This was there for reordering the
> selected channels, but it seems that the drag-and-drop functionality is not working. This is
> associated with TagsInput replaced by ChannelPickerWidget (anywidget).
>
> **Expected behavior:** The "tags" in the channel selection UI should be draggable, allowing
> users to reorder the selected channels as needed. The order should be reflected in the channel
> color and scale UI.

### Root cause

Not a broken feature — a **feature that was never carried over**. ipywidgets' `TagsInput`
implements drag-reordering in its own view (`TagsInputView` makes each tag element draggable and
splices `value` on drop). `ChannelPickerWidget`, the anywidget that replaced it for #125, renders
its chips itself:

```js
sel.forEach(function (name) {
  var chip = document.createElement('span');
  chip.className = 'ucp-chip';
  …
  chip.appendChild(label);
  chip.appendChild(x);      // remove button — the only interaction a chip had
  chips.appendChild(chip);
});
```

No `draggable`, no drag listeners. The chips were display-plus-remove only, so the sole way to
change the order was to clear the selection and re-pick every channel in the desired sequence.

### Why the order matters

`value` order is not cosmetic — it drives two things already:

| Consumer | Effect of the order |
| --- | --- |
| [`main_viewer.update_controls`](../../ueler/viewer/main_viewer.py#L2809) | iterates `channel_selector.value` and assigns `channel_controls_box.children` in that order — the per-channel visibility / name / colour header and the Min/Max contrast sliders |
| `_compose_fov_image` (via [`update_display`](../../ueler/viewer/main_viewer.py#L4789)) | channel compositing order |

So the second half of the expected behaviour — *"the order should be reflected in the channel
color and scale UI"* — needs **no new wiring at all**: `update_controls` already rebuilds the rows
in `value` order, and it is already reached from
[`on_channel_selection_change`](../../ueler/viewer/main_viewer.py#L2373), the `names='value'`
observer registered in `ui_components.py`. Committing a permuted `value` is sufficient, and because
`update_controls` reuses the existing per-channel widget objects (keyed by channel name), a reorder
moves the rows **without resetting the user's colour choices or contrast settings**.

## Solution

Make the chips drag-reorderable in the widget's own ESM.

```
Channels:  [⋮⋮ CD45 ×] [⋮⋮ CD3 ×] │[⋮⋮ CD8 ×]      <- │ = drop indicator
                                  ^ dragging CD4 here
```

Design decisions:

1. **The whole chip is the drag source**, with a `⋮⋮` grip glyph as the visual affordance and
   `cursor: grab`. A grip-only handle (as in `mask_class_list_widget`) is unnecessary for an
   11px chip, and making the whole chip draggable gives a far larger target. The `×` button opts
   out (`draggable = false` plus a `dragstart` that stops propagation) so removing a chip is still
   a click, not an aborted drag.
2. **The drop side follows the pointer.** `dragover` compares `e.clientX` against the target
   chip's horizontal midpoint and shows a `drop-before` / `drop-after` indicator. Without this
   only one side of each chip is reachable, so a chip could not be moved to the far end of the
   row in one gesture.
3. **The drop indicator is an absolutely-positioned pseudo-element**, not a border or margin.
   A marker that changes the chip's box would reflow the row mid-drag and move the drop target
   out from under the pointer.
4. **The target index is resolved after the source is spliced out.** This is the one piece of real
   logic, and the place a naive implementation goes wrong: resolving it first (as
   `mask_class_list_widget`'s row reorder does) lands every left-to-right move one slot too far,
   because removing the source shifts the target left. It is isolated as a pure
   `reorderSelection(selection, name, target, after)` function at ESM module scope, between
   `// --- reorder helper (#126) ---` markers, with no DOM access — so the unit tests can extract
   and execute it under `node` (see *Tests*).
5. **Dropping in the empty space after the last chip appends.** A container-level handler covers
   this; the per-chip handlers `stopPropagation()` so it only fires outside every chip.
6. **A keyboard path exists.** Chips are `tabIndex = 0`; ← / → move the focused chip and
   Delete/Backspace removes it, with focus followed to the chip's new position by name (the commit
   re-renders, so the element identity changes). HTML5 drag-and-drop is pointer-only, and some
   notebook hosts intercept drag events before a widget sees them — this keeps reordering reachable
   there, and makes the feature usable without a mouse.
7. **Reordering goes through the existing `commit()`**, i.e. `model.set('value', …)` +
   `save_changes()`. Nothing new is added to the Python surface: the permutation arrives through
   the same trait, so `on_channel_selection_change`, checkpointing, marker sets and the plugin
   pickers all see it with no changes.

All five pickers built by `build_channel_picker` inherit this — the left-panel **Channels**
selector plus the Scatter plot, Histogram, Heatmap and FlowSOM marker/feature pickers — which is
correct: marker order is meaningful in the heatmap and scatter plots too.

### Out of scope (deliberately)

* Reordering the **option list** — it is presented in `allowed_tags` order (markers first for an
  AnnData table, per #123) and is not user-orderable.
* Dragging a chip *out* of the widget, or between two pickers.
* Touch-drag: HTML5 drag-and-drop does not fire for touch input. The keyboard path is the
  fallback; a pointer-events-based implementation would be a larger change and no touch use has
  been reported.

## Implementation

Single source file: `ueler/viewer/plugin/channel_picker_widget.py`.

* `_CSS` — `.ucp-chip` gains `position: relative`, `cursor: grab`, `user-select: none` (a drag
  must not turn into a text selection) and a `:focus-visible` outline; new
  `.ucp-chip.is-dragging` (dimmed source), `.ucp-chip.drop-before::before` /
  `.drop-after::after` (the 2px indicator bar, drawn in the 3px inter-chip gap) and `.ucp-grip`.
* `_ESM` —
  * new module-scope `reorderSelection()` between the marker comments;
  * `renderChips()` builds each chip with the grip, `draggable`, `dataset.name`, `tabIndex`,
    `role="listitem"` + `aria-label`, and the `dragstart` / `dragover` / `dragleave` / `drop` /
    `dragend` / `keydown` handlers;
  * helpers `chipNodes()`, `clearDropMarkers()`, `endDrag()`, `focusChip()`, `moveChip()`;
  * container-level `dragover` / `drop` on `.ucp-chips` for append-to-end;
  * the chips container is `role="list"`.
* Module docstring records that `value` order feeds `update_controls` and compositing.

No other file changes — deliberately. `main_viewer`, `ui_components` and the plugins already
react to a `value` change correctly.

## Tests

`tests/test_issue126_chip_reorder.py` (26 tests, 5 classes):

* **`ReorderArithmeticTestCase`** — extracts `reorderSelection` from `_ESM` and runs it under
  `node`, so the drop arithmetic that is asserted is literally the code the browser executes
  (no Python re-implementation that could drift). Covers both drag directions × both drop sides,
  the append case, first↔last, self-drop, an unknown name, an empty selection, and an exhaustive
  sweep of all `4 × 5 × 2` combinations asserting no channel is ever duplicated or lost.
  The left-to-right case is annotated with the wrong answer a naive implementation gives.
* **`EsmSyntaxTestCase`** — `node --check` on the whole `_ESM`. A syntax error here breaks every
  picker in the viewer at render time and nothing else would catch it: the ESM is an opaque
  string to Python and the headless test fallback never evaluates it.
* **`FrontEndContractTestCase`** — chips draggable and name-tagged, all five drag events wired,
  midpoint-based drop side, the commit path, the `×` opt-out, the indicator cannot reflow the row,
  `user-select: none` + `cursor: grab`, the grip, the keyboard path, append-to-end.
* **`ValueTraitTestCase`** — a permutation survives `_normalise_selection` (it must not be
  re-sorted into `allowed_tags` order) and fires the `value` observers.
* **`ChannelControlOrderTestCase`** — the issue's second requirement, on the Python side:
  `update_controls` renders the channel colour/contrast rows in `value` order, a permuted `value`
  reorders them, and the slider objects are **reused** rather than rebuilt (a reorder must not
  discard the user's contrast settings). Uses a trimmed version of the viewer stub in
  `tests/test_annotation_palettes.AnnotationLayoutTests`.

Both `node`-backed classes skip cleanly when `node` is not on `PATH`; the contract tests still run.

```bash
python -m unittest tests.test_issue126_chip_reorder tests.test_issue125_channel_picker \
    tests.test_annotation_palettes tests.test_channel_legend tests.test_initial_display
python -m unittest discover -s tests -t .
```

→ **804 tests, OK** (778 before, +26).

**Not covered by tests, to confirm in a notebook:** the actual pointer gesture. There is no
browser in the dev environment (no playwright/chromium), so real drag-and-drop, the grip's
appearance and the drop indicator's placement were verified by construction and by the ESM
contract only. Worth checking live: drag a chip in the left-panel **Channels** picker, confirm the
colour/contrast rows below follow the new order and keep their settings, and confirm the same in
one plugin picker (Heatmap or Scatter).
