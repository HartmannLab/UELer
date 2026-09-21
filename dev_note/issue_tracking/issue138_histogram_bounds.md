# Issue #138 — User-adjustable histogram bounds

Let the user restrict each histogram to a value range instead of always binning over the full data extent of the channel.

> Source: [`ueler/viewer/plugin/histogram.py`](https://github.com/HartmannLab/UELer/blob/main/ueler/viewer/plugin/histogram.py)

---

## Problem

`HistogramDisplay._histogram_bin_edges` is the only place the x extent of a histogram is decided:

```python
return np.histogram_bin_edges(self._plot_data[channel], bins=bins)
```

With no `range=`, NumPy spans `[min, max]` of the plotted column, so the bins are always stretched across the full data extent. On a channel with a long right tail — the normal case for marker intensities — nearly every cell lands in the first two or three bins and the interesting structure near zero is compressed into a couple of pixels. Raising **Bins** does not fix it: 200 bins over the full extent still puts the same fraction of the resolution where there is no data.

Bokeh's `pan`/`wheel_zoom`/`reset` tools are already enabled on every figure, so the user *can* magnify that region — but zooming only changes the viewport. The bin edges are computed in Python before the figure exists, so a zoomed-in histogram shows the same three fat bars, just larger. Zoom is therefore not the feature being asked for; the bins themselves have to be recomputed over the chosen range.

There is no state for a per-channel range anywhere in the plugin, and nothing in `_build_figures`, `_render` or `_refresh_overlays` is parameterised by one.

---

## Design

### Per-channel, not per-plugin

Bounds are stored as `self._bounds: dict[str, tuple[float, float]]`, one optional override per channel, exactly like `_gates`. Several channels are plotted at once and they do not share a scale — a DNA channel and a CD4 channel have nothing in common numerically — so a single pair of plugin-wide bounds would be meaningless. A channel absent from `_bounds` keeps today's behaviour (full data extent), which is what makes the change backwards compatible.

### One two-handle slider per figure, inside the Bokeh layout

The request suggested "two sliders below each histogram, one for the lower bound and one for the upper". This ships one Bokeh `RangeSlider` per figure instead, which is the same two handles in one widget and additionally cannot be driven into an inverted `lower > upper` state that would need its own validation and error message.

It is a **Bokeh** widget, placed in the Bokeh `column` directly beneath its figure, rather than an ipywidget in the controls area. The plot host holds a single `BokehModel` whose height is capped and scrolled (#112 reply 2); anything interleaved between the figures has to be inside that model or it cannot scroll with them. Putting the control in the fixed controls area above would instead need a channel picker to say *which* histogram it addresses — a second selector for something the user is already pointing at.

Kernel-side callbacks on Bokeh models are the mechanism this plugin already runs on: `p.on_event(Tap, …)`, `SelectionGeometry` and `DoubleTap` all dispatch into Python through `jupyter_bokeh`'s document sync. `slider.on_change("value_throttled", …)` is the same channel.

### Real time, without a kernel round-trip per mouse move

Two bindings, doing different halves of the job:

- `js_link("value", p.x_range, "start"/"end")` — pure client-side, so the x axis follows the handles **while dragging**, with no kernel involvement.
- `on_change("value_throttled", …)` — fires once the handle is released, and rebins in Python.

Binding the rebin to `value` instead would send one full re-binning round-trip per pixel of drag. The user sees the window move live and the bars resolve when they let go.

### Rebin in place; never `_render()`

`_rebin_channel(channel)` recomputes that one channel's edges and counts and **patches the existing ColumnDataSources**, leaving every figure, tool, gate marker and the other channels alone. This follows the rule the plugin has held since #127: only a change of *what is plotted* rebuilds the layout. Calling `_render()` here would be worse than inefficient — it would destroy and replace the very slider the user is holding.

To patch the base bars, `_build_figures` now keeps the full-counts source in `sources[channel]["full_src"]`; it was previously created and dropped, since nothing had reason to rewrite it. `sources[channel]["edges"]` and `["full"]` are updated in the same step, so the subsequent `_refresh_overlays()` recomputes the selected-subset overlay and the faint-selection tint (#135 reply) on the new grid for free.

### What the bounds do and do not affect

Bounds are a **display and binning** concern only. `_gate_frame()` and `_term_mask()` are untouched, so a gate keeps meaning what it says even if the cells it selects are currently outside the plotted window — the alternative silently ANDs an invisible fourth term into every gate. Values outside the range are excluded from the counts (the standard `np.histogram(range=…)` semantics), not clipped into the edge bins, which would put a spike at each end that does not exist in the data.

### Lifecycle

- A bound survives a **Bins** change and a mode switch, because both go through paths that read `self._bounds` when they rebuild or patch.
- `plot_histograms` drops the bounds of channels that are no longer plotted, alongside the same cleanup already done for `_gates`.
- A channel that *is* still plotted keeps its bound across a replot, but `_bin_range` clamps it to the new data extent, so changing the subset cannot leave a histogram binned over a window that contains no rows.
- A **Full range** button next to each slider drops that channel's override. Bokeh's own `reset` tool is not enough: it restores the viewport, not the binning.

Bounds are in-memory plugin state and are not written to `widget_states.json`, consistent with `_gates` and `_plot_data`.

---

## Implementation steps

1. `HistogramDisplay.__init__` — add `self._bounds: dict = {}` and `self._bound_sliders: dict = {}`.
2. Add `_channel_extent(channel)` (data min/max, with a degenerate-column fallback) and `_bin_range(channel)` (the override clamped to the extent, or `None`).
3. `_histogram_bin_edges` — pass `range=self._bin_range(channel)` to `np.histogram_bin_edges`.
4. `_build_figures` — build a `RangeSlider` + **Full range** `Button` row per channel, wire `js_link` and `on_change`, and stack `column(figure, row)` per channel in the outer column. Store `full_src` and the slider.
5. Add `set_channel_bounds`, `clear_channel_bounds` and `_rebin_channel`, plus the two Bokeh handler factories.
6. `plot_histograms` — drop bounds for dropped channels.
7. `_scroll_height` — account for the extra slider row per channel.
8. Tests in `tests/test_histogram_bounds.py`; update the two tests in `tests/test_histogram_plugin.py` that assume `layout.children` are figures.
