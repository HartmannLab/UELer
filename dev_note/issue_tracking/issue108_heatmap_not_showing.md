# Issue #108 — Heatmap renders but never appears

> **Status:** The first fix below (destructive-second-pass / children-reassignment) was a
> real improvement but did **not** resolve the blank — after a kernel restart even a plain
> **Plot** click stayed blank. See **“UPDATE — revised root cause & final fix”** at the
> bottom for the actual cause (the `sns.clustermap` figure was built inside the ipympl
> display context) and the shipped fix.

## Problem

Using the Heatmap plugin produced no visible heatmap, even though the debug log showed the
render **completing** end to end:

```
[heatmap] generate_heatmap: render complete
[heatmap] plot refreshed in wide mode
```

The data and clustering were fine (`sns.clustermap (shape=(4, 103))` succeeded). This was a
display/visibility failure in the widget layer, and it reproduced **even with the
interactive `ipympl` (`%matplotlib widget`) backend active** — so it was not a backend or
data problem.

The user asked to identify *what actually makes the rendering unreliable*, using the
reliably-working Chart plugin — specifically its **histogram**, which renders *and stays
interactive* every time — as the reference.

## The reference: Chart plugin's interactive histogram

`chart.py::_render_histogram` (~lines 289–336) is fully interactive
(`fig.canvas.mpl_connect("button_press_event", onclick)`, `fig.canvas.draw_idle()`) and
never fails to appear, using this idiom:

1. `self._hist_output.clear_output(wait=True)` — clear the output.
2. `fig, ax = plt.subplots(...)` — build the figure.
3. `with self._hist_output:` — draw, wire `mpl_connect`, `fig.tight_layout()`,
   `plt.show(fig)` (explicit figure, inside the block).
4. `self._plot_host.children = [self._hist_output]` — **reassign the container's children**
   to force the frontend to (re)instantiate the Output view.
5. **No second clear/redraw pass** afterward.

Under `ipympl`, `plt.show(fig)` displays the interactive canvas widget — this is how the
histogram stays interactive. No `display(canvas)`/`display(fig)` branching, no backend
detection.

## Root Cause

The Heatmap plugin diverged from that proven idiom in three ways, each breaking rendering
even with `ipympl` active:

1. **A destructive second clear/redraw pass (the main wide-mode killer).** After
   `_setup_layout` displayed the figure, `plot_heatmap` called `restore_footer_canvas()` →
   `redraw_cached_footer_canvas()`, which ran `clear_output(wait=True)` — wiping the
   just-shown canvas — and then re-`display()`ed the *same* already-consumed canvas/figure,
   which paints nothing. The panel was left blank.
2. **No container `.children` reassignment (no repaint), worsened by reparenting.** The
   heatmap rendered into a single persistent `plot_output` (`heatmap.py`) and never
   reassigned any container's `.children`. In wide mode that same `plot_output` is
   reparented into a footer pane built once and cached forever (the plugin never overrode
   `wide_panel_cache_token`), so nothing told the frontend to re-instantiate the view.
3. **Bare `plt.show()` + a second conditional `display()`** instead of the histogram's
   single explicit `plt.show(fig)`.

Five render sites shared the same fragile `with self.plot_output: clear_output;
generate_heatmap()` block and none reassigned `.children`: `plot_heatmap`, `load_heatmap`,
`apply_new_cutoff`, `import_heatmap_state`, and the display inside `_setup_layout`.

## Fix — mirror the interactive histogram idiom

All edits in `ueler/viewer/plugin/heatmap_layers.py` plus one import. Interactivity is
preserved because we use `plt.show(g.fig)` under `ipympl`, exactly like the histogram — the
`ipympl` canvas is displayed, so `mpl_connect` handlers keep working. No static path is
added.

1. Import `Output` from `ipywidgets`.
2. `_setup_layout` now does a single explicit `plt.show(g.fig)`; the bare `plt.show()` and
   the conditional `display()` block were removed. `tight_layout`, the cutoff red line,
   `display_row_colors_as_patches()`, `header_visible = False`, and the click-handler
   `mpl_connect` are unchanged.
3. New `_refresh_plot()` helper: create a fresh `Output`, set `self.plot_output`, run
   `generate_heatmap()` inside `with new_out:`, then swap the new Output into
   `plot_section.children` (`_swap_plot_output_in_section`). A brand-new `Output` guarantees
   an identity change so the container repaints in both the vertical accordion and the
   cached wide/footer pane, and gives `ipympl` a clean canvas host.
4. All five render sites now call `self._refresh_plot()`.
5. Removed the destructive plumbing: deleted `redraw_cached_footer_canvas` and
   `_plot_output_has_widget_view`; simplified `_ensure_plot_canvas_attached` to set
   `plot_section.children = (self.plot_output,)`; reduced `restore_footer_canvas` /
   `restore_vertical_canvas` to thin `_ensure_plot_canvas_attached()` wrappers. Dropped the
   orphaned `_cached_footer_artifacts` dict. `wide_panel_cache_token` is unchanged (repaint
   comes from `plot_section.children` inside the cached pane).
6. `generate_heatmap` now `plt.close()`s the *previous* figure before building a new one,
   so repeated Plot clicks / cutoff drags don't leak figures.

## Files Changed

- `ueler/viewer/plugin/heatmap_layers.py` — render pipeline rewrite (steps 1–6).
- `tests/test_issue108_heatmap_refresh.py` — new regression tests (fresh-Output swap +
  children reassignment; imports the real `ueler` package to avoid the pre-existing broken
  heatmap test harness).

## Tests

```bash
python \
  -m unittest tests.test_issue108_heatmap_refresh tests.test_chart_footer_behavior \
              tests.test_checkpoint_store tests.test_cell_annotation tests.test_log_console
```

- ✅ New regression tests pass; full-suite count unchanged (no new failures/errors vs the
  pre-existing baseline).

## Notes

- `tests/test_heatmap_selection.py` errors on import due to a **pre-existing, unrelated**
  path bug (it loads `viewer/plugin/heatmap.py` instead of `ueler/viewer/plugin/heatmap.py`)
  and is part of the existing error baseline; its `HeatmapCanvasRestoreTests` (which
  exercised the now-removed second-pass methods) never ran. The new regression test was
  therefore placed in a clean standalone module.
- Manual verification (the real test): with `%matplotlib widget` active, `run_viewer(folder,
  debug=True)` → click **Plot** → heatmap appears and stays interactive (click a cell, drag
  the dendrogram cutoff, shift/ctrl-select on the color axis); toggling Horizontal layout
  shows it in the footer tab and re-plotting is reliable.

---

## UPDATE — revised root cause & final fix

The first fix did not solve the blank. After a confirmed kernel restart, even a plain
**Plot** click (no layout switch) still showed nothing. Crucially, the same `%matplotlib
widget` (ipympl) backend renders the Chart histogram and the anywidget galleries reliably —
so the backend is fine; the failure is **specific to the heatmap's interaction with
ipympl**.

### Actual root cause

The Chart histogram builds its figure **outside** the display context and emits it once:

```python
# chart.py::_render_histogram
self._hist_output.clear_output(wait=True)
fig, ax = plt.subplots(...)          # built OUTSIDE the `with` block
with self._hist_output:
    ...; plt.show(fig)               # emitted ONCE, inside
self._plot_host.children = [self._hist_output]
```

The heatmap did the opposite: it built and manipulated the `sns.clustermap` figure
**inside** the `with <output>:` block (`sns.clustermap(...)` + `cax.remove()` +
`ax_row_dendrogram.remove()` + `make_axes_locatable` + `tight_layout`) and then called
`plt.show()`. Under `%matplotlib widget` (interactive mode on), creating a figure inside the
display context makes ipympl **auto-emit** its canvas there, and the following `plt.show()`
emits it **again** → a duplicate/blank ipympl canvas. That is the one structural difference
from the histogram, and it explains the persistent blank.

### Final fix (`ueler/viewer/plugin/heatmap_layers.py`) — keeps interactive ipympl + footer docking

1. **`_setup_layout` is build-only** — the `plt.show()` was removed; it only builds the
   clustermap, axis surgery, cutoff line, row-color patches, and wires the `mpl_connect`
   click handler.
2. **`_refresh_plot` builds outside the Output with auto-emit suppressed, then emits once.**
   It creates a fresh `Output`, runs `generate_heatmap()` while `plt.ioff()` is active and
   outside any `with output:` block (restoring the prior interactive state afterward), then
   `with new_out: display(fig.canvas)` — emitting the **interactive** ipympl canvas exactly
   once — and swaps `new_out` into `plot_section.children`. `display(fig.canvas)` is
   targeted (unlike `plt.show()`, which emits every managed figure) and preserves pan/zoom
   and all click handlers.
3. **Toggle double-render removed.** `on_orientation_toggle` now sets
   `_plot_refresh_inflight` around `refresh_bottom_panel()` so the cached-pane refresh does
   not also render; the single explicit `plot_heatmap()` performs the one render.
4. **Reparented footer canvas repaints after layout.** New `_present_footer_canvas_if_wide`
   does a synchronous `canvas.draw()` backstop (as the main image canvas does in
   `main_viewer`) plus a single-shot `canvas.new_timer(150ms) → draw_idle()` (as
   `image_display` does), so the canvas paints once the footer tab has been laid out.

### Why not static/anywidget

Rejected by design: a chart rendered as a static PNG would have unacceptable
scaling/resolution behavior (the #107 galleries are literally images, and their anywidget
move was for better *click* interactivity on tiles — a different goal). The heatmap needs
the live matplotlib canvas for cell-click, dendrogram-cutoff drag, and color-axis selection.

### Verification

- Notebook isolation check (confirms the mechanism): building `sns.clustermap` inside a
  `with Output(): ...; plt.show()` renders blank/duplicated, whereas building with
  `plt.ioff()` outside and `display(g.fig.canvas)` inside a fresh `Output` renders the
  interactive heatmap.
- End-to-end after Restart Kernel: Plot renders and is interactive; Horizontal-layout
  docks into the footer and re-plots reliably; repeated plotting does not leak figures.
- `tests/test_issue108_heatmap_refresh.py` extended to 7 tests (build runs under `ioff`;
  canvas emitted exactly once via `display`; wide mode issues the synchronous + deferred
  redraws; vertical mode does not). Full suite: no new failures/errors vs. baseline.
