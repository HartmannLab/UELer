# Issue #121 — Allocate plugins to the wide-footer layout permanently

> **Reply correction (current state):** the developer clarified that moving the
> **Histogram** into the footer was a mistake — the **Heatmap** should be the
> permanent wide-footer plugin instead. The permanently-footer set is now
> **Scatter plot + Chart (heatmap) + Heatmap**; the **Histogram** is back in the
> side accordion. See [Reply correction](#reply-correction-histogram-out-heatmap-in)
> at the bottom for the delta. The sections below describe the original (first-pass)
> implementation, kept for history.

## Problem

Three plugins used to **switch** between the wide-footer panel
(`viewer.wide_plugin_panel`, a `Tab` below the image) and the default side layout (an entry in
the left SidePlots accordion):

- **Scatter plot** (`chart.py`, `ChartDisplay`) and **Chart (heatmap)** (`chart_heatmap.py`,
  `ChartDisplay`) moved into the footer only when `_has_multiple_scatter()` was true (≥2 active
  scatters); with 0–1 scatters they rendered in the side accordion and showed a `_wide_notice`
  placeholder in the footer path.
- **Histogram** (`histogram.py`, `HistogramDisplay`) did **not** implement `wide_panel_layout()`
  at all, so it was always side-only.

The state-driven switching was distracting and inconsistent. Since these plugins are most
useful with the extra footer width, #121 asks to make the wide footer their **permanent** home
and drop the side variant, simplifying the UI.

## How placement works

- Every plugin lives on `viewer.SidePlots`. `display_ui()`
  ([ui_components.py](../../ueler/viewer/ui_components.py)) unconditionally adds each plugin with
  `.ui` + `.displayed_name` to the side accordion.
- The footer is a separate `Tab` (`viewer.wide_plugin_tab`). `collect_wide_plugin_entries()`
  calls each plugin's `wide_panel_layout()`; a truthy `{title, control, content}` dict earns a
  footer tab, `None` keeps it side-only. `update_wide_plugin_panel()` (via
  `viewer.refresh_bottom_panel()`) builds/caches panes and shows/hides the footer.

## Design decisions (confirmed with the developer)

1. **Scope:** include `chart_heatmap.py` ("Chart (heatmap)") alongside the scatter and histogram
   plugins named in the issue — all three share the switching logic — for full consistency.
2. **Side panel:** **remove** these plugins' side-accordion entry entirely (footer-only), rather
   than keep a "look in the footer" notice. This is cleaner and lets the dead switching code go.

## Fix

### 1. `footer_only` flag on the plugin base — [plugin_base.py](../../ueler/viewer/plugin/plugin_base.py)
`PluginBase.__init__` sets `self.footer_only = False`. Plugins that should live only in the
footer set it `True`.

### 2. Skip footer-only plugins in the side accordion — [ui_components.py](../../ueler/viewer/ui_components.py)
`display_ui()`'s accordion loop `continue`s when `getattr(attr, 'footer_only', False)`. The
plugin stays on `viewer.SidePlots` (so `collect_wide_plugin_entries()` still finds it) but gets
no accordion child. No other footer-machinery change was needed.

### 3. Scatter plugin — [chart.py](../../ueler/viewer/plugin/chart.py)
- `__init__` sets `self.footer_only = True`.
- `wide_panel_layout()` **always** returns
  `{"title", "control": self.controls_section, "content": self.plot_section}`.
- Removed the switching machinery: `_wide_notice`, `_section_location`, `_has_multiple_scatter()`,
  `_place_sections_vertical/horizontal()`, `_sync_panel_location()` and its four call sites. The
  footer holds the same `plot_section`/`_plot_host` widgets the plugin mutates, so footer content
  updates in place when scatters are added/removed.
- `after_all_plugins_loaded()` calls `refresh_bottom_panel()` once to populate the footer on load.
- `self.ui` (built vertical) is left intact but simply never displayed now.

### 4. Chart (heatmap) plugin — [chart_heatmap.py](../../ueler/viewer/plugin/chart_heatmap.py)
Mirrors chart.py exactly (footer-only flag; always-footer `wide_panel_layout()`; switching code
removed; `after_all_plugins_loaded()` simplified to `refresh_bottom_panel()`). The existing
`refresh_bottom_panel()` calls in the plot/remove/clear paths were kept.

### 5. Histogram plugin — [histogram.py](../../ueler/viewer/plugin/histogram.py)
- `__init__` sets `self.footer_only = True`.
- Added `wide_panel_layout()` returning the footer dict from the existing
  `controls_section`/`plot_section`.
- `after_all_plugins_loaded()` calls `refresh_bottom_panel()` to populate the footer on load.

## Tests

- [test_chart_footer_behavior.py](../../tests/test_chart_footer_behavior.py) — rewrote
  `test_footer_layout_toggles_with_scatter_count` → `test_scatter_plugin_is_permanently_wide_footer`
  (asserts `footer_only` and that `wide_panel_layout()` returns the dict for 0/1/2 scatters).
  Updated `HeatmapFooterPersistenceTests.test_heatmap_survives_chart_refresh` to drive
  `viewer.refresh_bottom_panel()` (the removed `_sync_panel_location`) and expect the chart tab to
  persist (2 footer tabs throughout).
- [test_histogram_plugin.py](../../tests/test_histogram_plugin.py) — new `TestHistogramFooterLayout`.
- [test_chart_heatmap_footer.py](../../tests/test_chart_heatmap_footer.py) — new file (there was
  no chart_heatmap test before).
- [test_wide_plugin_panel.py](../../tests/test_wide_plugin_panel.py) — unchanged; still verifies
  the shared footer machinery (its generic `ToggleFooterPlugin` stub still exercises the
  heatmap's real toggle).

Full-suite failure/error set is identical to the `develop` baseline (31 pre-existing
failures/errors, no new ones). Live rendering to be confirmed in the notebook — in particular
that a **single** scatter renders in the footer without edge cropping / a mis-scaled axis (the
#118 concern; the footer is visible from load, so `width='auto'` should measure correctly).

---

## Reply correction: Histogram out, Heatmap in

The developer's reply to #121 reversed part of the first pass:

> It was my mistake to allocate the histogram plugin to the wide-footer layout permanently.
> Rather the heatmap plugin should be allocated to the wide-footer layout permanently. Please
> move the histogram plugin back to the side and allocate the heatmap plugin to the wide-footer
> layout permanently.

### Decisions (confirmed with the developer)
1. **Which "heatmap":** the real **"Heatmap"** plugin (`heatmap.py` / `heatmap_layers.py`),
   *and* keep **"Chart (heatmap)"** (`chart_heatmap.py`) footer-only as-is (answer: "Both
   heatmap plugins" / "Keep it footer-only").
2. **Heatmap mode:** force **wide/horizontal always** and **remove** the `Horizontal layout`
   checkbox (rather than keeping placement-only and a live orientation toggle). This matches
   #121's "reduce UI complexity" goal and avoids a tall vertical heatmap wasting a wide footer.

### Delta from the first pass
- **Histogram** (`histogram.py`) — reverted to side-only: removed `footer_only = True`, the
  `wide_panel_layout()`, and the `refresh_bottom_panel()` call in `after_all_plugins_loaded()`.
  It now falls back to `PluginBase.wide_panel_layout()` (returns `None`).
- **Heatmap** (`heatmap.py`) — `__init__` sets `footer_only = True` and builds the adapter with
  `mode="wide"` (was `"vertical"`); removed the init-time `_sync_panel_location()` prime and the
  `Horizontal layout` checkbox from `UiComponent`.
- **Heatmap layers** (`heatmap_layers.py`) — `wide_panel_layout()` always returns the footer
  `{title, control, content}` dict (no `is_wide()` gate). Removed `on_orientation_toggle`,
  `on_mode_toggle`, `_sync_panel_location`, `_place_sections_vertical/horizontal`,
  `_wide_notice`, `_section_location`; dropped `horizontal_layout` from the AnnData checkpoint
  export/import. `after_all_plugins_loaded()` still calls `refresh_bottom_panel()` (no more
  `_sync_panel_location()`). The orientation-aware rendering/selection branches (`adapter.is_wide()`)
  are kept — the plugin just never leaves wide mode.
- **Scatter plot** and **Chart (heatmap)** — unchanged (still footer-only).

### Tests (reply)
- `tests/test_heatmap_footer.py` — **new**: the real `HeatmapDisplay` is `footer_only`, its
  adapter `is_wide()` is always true, `wide_panel_layout()` always returns the footer dict, and
  `UiComponent` has no `horizontal_layout_checkbox`.
- `tests/test_histogram_plugin.py` — `TestHistogramFooterLayout` → `TestHistogramSideOnly`
  (histogram is *not* footer-only; `wide_panel_layout()` is `None`).
- Existing heatmap suites (`test_issue108/109`, `test_heatmap_selection`,
  `test_heatmap_marker_selection`, `test_heatmap_adapter`, `test_gallery_heatmap_integration`)
  still pass — the retained `is_wide()` branches keep exercising the orientation logic.
- 121 targeted tests pass; full-suite failure/error set identical to the `develop` baseline
  (31 pre-existing, no new failures). Live rendering (heatmap in the footer, always horizontal)
  to be confirmed in the notebook.
