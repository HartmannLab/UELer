# Issue #127 — Multi-channel gating in the Histogram plugin

Gate cells on several histograms at once: each channel keeps its own term (a brushed
range **or** an above/below cutoff) and the published selection is the intersection
of every active term. Selecting on one histogram must not disturb the others.

---

## Problem

`HistogramDisplay` had exactly one selection term at a time, in two separate forms:

| Mode | State | Behaviour |
| --- | --- | --- |
| Brush | `_brush_selection = (channel, lo, hi)` | `handle_range()` **overwrote** the tuple on every gesture |
| Cutoff | `cutoff` + `_active_histogram_column` | `highlight_cells()` read only the active channel |

So brushing CD4 and then CD8 selected "cells in the CD8 range", never both. The two
modes could not be combined at all, and `_refresh_cutoff_spans()` deliberately hid
the cutoff line on every channel except the active one — so the plugin also *looked*
single-channel, except for the `_refresh_overlays()` cross-histogram overlay, which
redraws the one selection on every figure and reads as linked filtering when it is
only linked display.

Two further inconsistencies mattered once the modes had to share a gate:

* **Different frames.** A brush filtered `_plot_data` (subset-filtered, NaN-dropped);
  a cutoff filtered `main_viewer.cell_table` directly, so the same threshold meant
  different things depending on which mode set it.
* **Different highlight paths.** A brush called
  `_chart_common.sync_mask_highlights_from_selection` and only when **Main viewer**
  was linked; a cutoff called `image_display.set_mask_ids()` itself, unconditionally.

---

## Design

### One term per channel, ANDed

```python
self._gates: dict = {}   # channel -> (kind, a, b)
#   ("range",  lo,        hi)     — brushed [lo, hi]
#   ("cutoff", direction, value)  — "above"/"below" value
```

`gated_indices()` builds one boolean mask per term and `&`s them. With a single term
the result is bit-for-bit the old single-channel behaviour, which is what keeps the
change backwards compatible for anyone using one channel.

**AND is unconditional, with no `Replace`/`Intersect` toggle.** The open question in
the request was resolved this way because a single brush behaves identically either
way — the modes only diverge once a second channel is gated, which is the feature
being asked for — and because a mode toggle would need its own persisted state and
a third code path through `_apply_gate`. Per-channel clearing (below) covers the
"I only want this one channel" case without a mode.

A term naming a column that is absent from the frame is **skipped**, not treated as
empty. Otherwise a gate would silently select nothing whenever the plotted subset
dropped a column.

### The cutoff joins the same gate

`highlight_cells()` folds the pending `cutoff` / `_active_histogram_column` pair into
`_gates` and then applies the gate like any other term. The above/below direction is
captured **at that moment**, so each channel keeps its own direction and the
`Highlight: below/above` toggle re-applies only the last cutoff's channel.

`cutoff` and `_active_histogram_column` were kept (rather than folded away) because
the viewer re-triggers `histogram_plugin.highlight_cells()` on a FOV change
([main_viewer.py:2408](../../ueler/viewer/main_viewer.py#L2408)) and callers/tests
assign them directly.

### One frame for both modes

`_gate_frame()` returns `_plot_data` when a plot exists and `main_viewer.cell_table`
otherwise. A cutoff therefore now respects the **Subset** tab like a brush does —
the one intentional behaviour change here — while a cutoff set before any plot, or
re-applied on a FOV change, still has the full table to fall back on.

`highlight_cells()` keeps pushing mask highlights **unconditionally**: it is the
explicit "show the gate in the viewer" entry point that the viewer itself calls. A
brush still highlights only when **Main viewer** is linked. Unifying those two
policies would have changed cutoff-mode behaviour for no benefit to this issue.

### Per-histogram state — nothing is replotted on selection

* **Persistent markers.** Each figure owns a `BoxAnnotation` (its range band, in
  `sources[channel]["band"]`) and a `Span` (its cutoff line). Both are ours, so a
  term stays drawn after the user acts on another histogram — Bokeh's own
  box-select overlay is transient and only ever marks the figure being dragged.
* **Pinned selection glyphs.** `selection_glyph` and `nonselection_glyph` are set to
  the base glyph on both quads, so a box-select gesture cannot grey out the
  non-selected bars. Without this, brushing one histogram visibly changes it in a
  way that competes with the gate band.
* **`_apply_gate()` is deliberately narrow.** It publishes `selected_indices`,
  optionally syncs mask highlights, and rewrites only the overlay sources and the
  annotations. It never calls `_render()` / `plot_histograms()`, so a selection
  cannot rebuild the stack or lose a zoom/pan (cf. #109, #119). A test asserts
  `_render` is not called across a brush, a cutoff, a per-channel clear and a full
  clear.
* **The cross-channel overlay still refreshes.** `_refresh_overlays()` touches only
  each figure's `selected` ColumnDataSource — no figures, axes or bin edges — which
  is what makes it safe to keep. It is the documented purpose of the plugin (see the
  module docstring) and, with gating, it is what shows the gated population's
  distribution in every channel. "Don't refresh on selection" is honoured where it
  is destructive (replots, markers, zoom), not for a source update that cannot
  disturb anything.

### Clearing

* **Double-tap a histogram** → `clear_gate(channel)` drops just that channel's term
  and re-applies the rest. Registered in both modes.
* **Clear selection** → `clear_selection()` drops every term *and* the cutoff state
  (it previously cleared only `_brush_selection`, leaving a cutoff live).
* **Replot** → terms on channels that are no longer plotted are dropped, and the
  selection is re-published if any were.
* **`show_external_selection()`** replaces the gate rather than intersecting with it:
  the indices come from another plugin's criteria, so leaving stale terms drawn would
  misrepresent what is selected.

### Readability

`gate_description()` renders the gate as text (`Gate: CD4 ∈ [0.2, 0.9] AND CD8 > 0.5`)
into a `gate_summary` HTML label under the controls. A gate whose terms live on
several figures — one of which may be scrolled out of view — is otherwise invisible.

---

## Files changed

- `ueler/viewer/plugin/histogram.py` — `_gates` replaces `_brush_selection`;
  `_gate_frame`, `_term_mask`, `gated_indices`, `_apply_gate`, `set_gate`,
  `clear_gate`, `gate_description`, `_refresh_gate_markers` (+ `_refresh_gate_summary`);
  `handle_range` / `highlight_cells` / `clear_selection` / `show_external_selection` /
  `_on_above_below_change` / `plot_histograms` rewired; per-figure `BoxAnnotation`,
  pinned selection glyphs, `DoubleTap` handler; `gate_summary` label.
- `tests/test_histogram_plugin.py` — new `TestHistogramGating` (19 tests);
  `test_cutoff_span_shows_only_on_active_channel` → `..._on_gated_channels` (spans now
  show on **every** gated channel); new band/glyph-pinning tests.

## Compatibility notes

- `_refresh_cutoff_spans()` is kept as an alias for `_refresh_gate_markers()`.
- `_on_brush()` remains the alias for `handle_range()`.
- `_brush_selection` is **gone**; nothing outside this module referenced it.
- Behaviour change: a cutoff is now evaluated on the plotted subset when a plot
  exists, and cutoff spans are drawn on every gated channel rather than only the
  active one.

## Tests

```bash
python -m unittest tests.test_histogram_plugin
python -m unittest discover -s tests -t .
```

- ✅ 60 histogram tests pass; full suite **853 tests, OK**.
- Not verified: the pointer gestures themselves (double-tap to clear, the band's
  appearance during a drag) need a notebook — the dev environment has no browser.

---

# Reply 1 — switching Brush ↔ Cutoff still resets the plot

The gating itself is remembered across a mode switch (it lives in `_gates`, not in
any figure), but the **plot** resets: zoom/pan is lost and the stack flashes.

## Cause

`_on_interaction_mode_change()` called `self._render()`, which rebuilds the layout via
`_build_figures()` and swaps in a new `BokehModel`. #127 removed every replot from the
*selection* path but left this one on the *mode* path.

The replot was structurally necessary at the time: `_build_figures()` wired the
gestures conditionally — `brush_mode` chose between a `BoxSelectTool` +
`SelectionGeometry` handler and a `Tap` handler — and Bokeh offers no clean way to
unregister an `on_event` callback, so changing the wiring meant rebuilding the figure.

## Design

**Wire both gestures once; switch only the active drag tool.**

| Piece | Before | After |
| --- | --- | --- |
| `BoxSelectTool` | added only in brush mode | added to every figure |
| `SelectionGeometry` handler | brush mode only | always registered |
| `Tap` handler | cutoff mode only | always registered, ignores taps while brushing |
| Mode switch | `_render()` (full rebuild) | `_apply_interaction_mode()` (property write) |

- `_figures` / `_box_tools` hold the live figures, so `_apply_interaction_mode()` can
  set `toolbar.active_drag` to the figure's box tool (brush) or its `PanTool`
  (cutoff, via `_pan_tool()`, falling back to `"auto"`). Both are writes on models the
  frontend already has — nothing is rebuilt, so bars, markers, overlay and zoom persist.
- `_build_figures()` ends with the same call, so first-render behaviour and the two
  existing `active_drag` tests (#112 reply) are unchanged.
- `_make_tap_handler` guards on `_brush_mode()`: Bokeh raises `Tap` for any click
  whatever the active tool is, and a bare click during a brush must not set a cutoff.
- `_make_range_handler` is deliberately **not** guarded — a box-select drag is
  unambiguous, so activating the tool from the toolbar in cutoff mode gates a range
  instead of being silently dropped.
- `_render()` clears `_figures` / `_box_tools` before its early returns, so a later
  mode switch cannot write to detached figures.
- `_on_bin_slider_change()` still replots; a different bin count is a different set of bars.

## Files changed

- `ueler/viewer/plugin/histogram.py` — `PanTool` import; `_figures` / `_box_tools`
  state; new `_brush_mode`, `_pan_tool`, `_apply_interaction_mode`; `_build_figures`
  wires both gestures unconditionally; `_make_tap_handler` mode guard;
  `_on_interaction_mode_change` no longer renders; `_render` resets the figure registry.
- `tests/test_histogram_plugin.py` — new `TestHistogramInteractionModeSwitch` (7 tests).

## Tests

```bash
python -m unittest tests.test_histogram_plugin
python -m unittest discover -s tests -t .
```

- ✅ 76 histogram tests pass; full suite **882 tests, OK** (875 before).
- Not verified: the gesture itself — dragging a range right after flipping the toggle,
  and the zoom actually surviving — needs a notebook; no browser here.
