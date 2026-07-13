# Issue #109 — Heatmap should remember its scale (figure size) after updating the tree cut

## Problem

When the user changes the dendrogram **tree cut**, the heatmap re-renders and its **scale
resets to the default**. The user clarified that "scale" is the **figure size** set by
dragging the ipympl resize handle — the **triangle at the bottom-right corner** of the
interactive canvas — not the toolbar zoom/pan.

The cutoff-update path is: dendrogram click (`_make_click_handler`, dend-axis branch) →
`apply_new_cutoff()` → `_refresh_plot()` → `generate_heatmap()` → `sns.clustermap(...)`,
which builds a **brand-new figure** using the default `figsize` from
`HeatmapModeAdapter.build_clustermap_kwargs`, discarding the user's resized dimensions.

> An earlier attempt preserved the axes **zoom** (`xlim`/`ylim`) — that was the wrong
> "scale" and was replaced by figure-size preservation.

## Key facts

- The ipympl resize triangle writes the manual resize back to the figure:
  `ipympl.backend_nbagg.Canvas.handle_resize` calls
  `fig.set_size_inches(x/dpi, y/dpi, forward=False)`, so `fig.get_size_inches()` reflects it
  (verified against ipympl 0.9.8).
- `build_clustermap_kwargs` always sets a `figsize` key (both orientations), so overriding
  it at build time makes `sns.clustermap` build the whole grid — and `tight_layout` — at the
  preserved size, with no post-build reflow.
- `@update_status_bar` forwards `**kwargs`, so `generate_heatmap` can take an optional arg.

## Fix (`ueler/viewer/plugin/heatmap_layers.py`)

Only the **cutoff** path preserves the size; a fresh **Plot** / load uses the adapter
default (cluster/marker counts may differ).

1. **`_capture_heatmap_scale()`** — returns the current `self.data.g.fig.get_size_inches()`
   as `(w, h)` floats, or `None` if there is no figure yet (guards first render / fresh
   load). Try/except-guarded.
2. **`generate_heatmap(self, figsize_override=None)`** — after
   `clustermap_kwargs = self.adapter.build_clustermap_kwargs(...)`, if `figsize_override` is
   set, `clustermap_kwargs['figsize'] = tuple(figsize_override)` before `sns.clustermap`.
3. **`_refresh_plot(self, restore_size=None)`** — passes it into the build:
   `self.generate_heatmap(figsize_override=restore_size)`. Default `None` keeps the
   fresh-Plot / load / import paths at the adapter default.
4. **`apply_new_cutoff`** — captures the size from the OLD figure before `_refresh_plot`
   rebuilds it, then `self._refresh_plot(restore_size=saved_size)`.

`plot_heatmap`, `load_heatmap`, and `import_heatmap_state` keep calling `_refresh_plot()`
with no argument → adapter-default size.

## Files Changed

- `ueler/viewer/plugin/heatmap_layers.py` — `_capture_heatmap_scale`; `figsize_override` on
  `generate_heatmap`; `restore_size` on `_refresh_plot`; capture+pass in `apply_new_cutoff`.
- `tests/test_issue109_heatmap_zoom.py` — reworked for figure-size preservation (capture
  reads/returns `None`; `apply_new_cutoff` passes captured size; fresh refresh passes
  `None`; `generate_heatmap` overrides `clustermap_kwargs['figsize']`; adapter always sets
  `figsize`).
- `tests/test_issue108_heatmap_refresh.py` — fake `generate_heatmap` updated to accept the
  new `figsize_override` kwarg.

## Tests

```bash
python \
  -m unittest tests.test_issue109_heatmap_zoom tests.test_issue108_heatmap_refresh
```

- ✅ Pass; full-suite count unchanged (no new failures/errors vs the pre-existing baseline).
  Verified end-to-end that overriding `figsize` makes real `sns.clustermap` build at the
  requested size.

## Manual verification

With `%matplotlib widget` (after Restart Kernel): Plot → drag the bottom-right triangle to
enlarge the heatmap → click a new dendrogram position to change the tree cut. The heatmap
re-clusters **at the same enlarged size** (does not shrink to default). A fresh **Plot**
uses the default size.
