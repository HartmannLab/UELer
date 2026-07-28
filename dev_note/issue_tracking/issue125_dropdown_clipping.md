# Issue #125 — Marker/feature selection dropdowns are clipped

> GitHub issue: [#125](https://github.com/HartmannLab/UELer/issues/125)
> Status: implemented (see *Implementation* below)

## Problem

> When the dropdown lists for marker/feature selection are too long, they are clipped and the
> user cannot see all the options. […] The dropdown lists should be scrollable or expandable so
> that the user can see all the options. This might be a limitation of the ipywidgets library,
> but we should find a way to work around it.

### Root cause

Every marker/feature picker in UELer was an ipywidgets **`TagsInput`**:

| Location | Widget | Purpose |
| --- | --- | --- |
| `ui_components.uicomponents.channel_selector` | `TagsInput` | left-panel channel picker |
| `_chart_common.build_channel_selector` → `.tags` | `TagsInput` | Scatter plot, Histogram, Heatmap marker pickers |
| `run_flowsom.UiComponent.channel_selector` | `TagsInput` | FlowSOM feature picker |

`TagsInput` does **not** render its own option list.  Its view (ipywidgets 8.1.x,
`TagsInputView`) creates a native `<datalist>` element and points the text input at it with
`list=<uuid>`:

```js
this.taginput.setAttribute("list", this.datalistID),
this.autocompleteList = document.createElement("datalist"), …
```

The suggestion popup is therefore drawn **by the browser**, not by the page, which means:

* its height and the number of visible entries are browser-defined and cannot be styled,
  scrolled, or paged from CSS/JS — with 50–150 markers most options are unreachable;
* the popup is a native/host widget, so in embedded notebook hosts (VS Code notebooks,
  webviews, iframes, Voilà panels) it is clipped at the host container's edge;
* it only appears **while typing** — clicking the field does not list what is available, so
  users cannot browse markers at all.

This is a genuine ipywidgets limitation: no `TagsInput` trait or CSS rule can make a
`<datalist>` popup taller, scrollable, or unclipped.  Ancestor layout (`overflow: hidden` on the
plugin control boxes, `max_height` on `channel_controls_box`, the wide-footer panel from #121)
cannot be blamed and cannot be tuned to fix it.

## Solution

Replace the `<datalist>`-based autocomplete with an **in-DOM, always-scrollable option list**
that UELer owns: a new anywidget widget, `ChannelPickerWidget`
(`ueler/viewer/plugin/channel_picker_widget.py`).

```
Channels:  [ CD45 ×] [ CD3 ×]                     <- chips = current selection, ordered
[ filter markers…                          ] [ ▾ ]
┌──────────────────────────────────────────────┐
│ ✓ CD45                                       │  scrollable list (max-height, overflow-y auto)
│   CD4                                        │  every option reachable, keyboard navigable
│   CD8                                        │
└──────────────────────────────────────────────┘
 12 of 148 shown · 2 selected      Select all shown · Clear
```

Design decisions:

1. **The list lives in the widget's own layout flow** (a plain scrollable `div`), *not* in a
   floating/absolutely-positioned overlay and *not* in a native popup.  An in-flow panel cannot
   be clipped by an ancestor `overflow`, a stacking context, an iframe boundary, or a host
   viewport — which is exactly what went wrong with the native popup.  This is the
   "scrollable or expandable" behaviour the issue asks for: opening the picker expands the
   panel, and the surrounding containers already scroll (`left_panel`, the footer control box).
2. **Drop-in API compatibility with `TagsInput`.** The widget exposes the same two observable
   traits — `value` (ordered list of selected names) and `allowed_tags` (available options) —
   so all existing call sites (`channel_selector.value`, `.allowed_tags = […]`,
   `.observe(…, names='value')`, marker-set loading, checkpoint restore) keep working
   unchanged.  It additionally tolerates the sloppy inputs the old code sent it (a bare string
   value in `run_flowsom`, tuples, duplicates, stale names).
3. **Search-first.** A filter box narrows a 150-marker list to a handful of rows, so long lists
   are usable without scrolling at all.  `Select all shown` turns a filter into a bulk
   selection (e.g. type `CD` → select every CD marker).
4. **Graceful degradation.** If anywidget is unavailable (or stubbed, as in the unit-test
   bootstrap), the module exposes a headless traitlets class with identical traits, and
   `build_channel_picker()` falls back to `TagsInput` when there is no anywidget at all in a
   real runtime, so the viewer never loses its channel picker.

### Out of scope (deliberately)

The plain `Dropdown` (`<select>`) feature pickers — `X:`, `Y:`, `Color:`, `Highlight:`,
`Subset on:`, `Class:`, `Marker set:` — keep their native popups.  A native `<select>` popup is
scrollable in every browser and is not subject to the `<datalist>` height cap, so they do not
reproduce the reported bug.  `ChannelPickerWidget` is written so a single-select mode can be
added later if the same clipping is ever observed for those in an embedded host.

## Implementation

* **New** `ueler/viewer/plugin/channel_picker_widget.py`
  * `ChannelPickerWidget` — anywidget with `value`, `allowed_tags`, `placeholder`,
    `description`, `disabled`, `list_max_height` traits; `_css` + `_esm` implement chips,
    filter box, scrollable option list, footer actions, and keyboard navigation
    (↑/↓ move, Enter toggles, Esc closes, Backspace on an empty box pops the last chip).
  * `value` is normalised on assignment: strings are wrapped, tuples/iterables accepted,
    duplicates removed (order preserved), names not in `allowed_tags` dropped (matching the
    previous `TagsInput` guarantee without raising `TraitError`).
  * `build_channel_picker(...)` — factory used by all call sites; ignores `TagsInput`-only
    kwargs (`allow_duplicates`, `style`) and falls back to `TagsInput` when anywidget is
    genuinely missing.
* `ueler/viewer/plugin/_chart_common.py` — `build_channel_selector` builds a
  `ChannelPickerWidget`; docstring/`ChannelSelector.tags` semantics unchanged.
* `ueler/viewer/ui_components.py` — the left-panel `channel_selector` uses the new picker.
* `ueler/viewer/plugin/run_flowsom.py` — the FlowSOM feature picker uses the new picker
  (and its bare-string initial `value` is now tolerated instead of raising).

## Tests

`tests/test_issue125_channel_picker.py` — trait round-trips and normalisation, observer
firing, `build_channel_picker` fallbacks, the CSS/ESM contract (scrollable list, no
`<datalist>`), and integration with the Scatter/Histogram/Heatmap/FlowSOM pickers.

```bash
python -m unittest tests.test_issue125_channel_picker
python -m unittest tests.test_heatmap_marker_selection tests.test_histogram_plugin \
    tests.test_chart_footer_behavior tests.test_chart_heatmap_footer tests.test_heatmap_footer \
    tests.test_initial_display tests.test_channel_legend tests.test_wide_plugin_panel
```
