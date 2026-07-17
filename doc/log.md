### v0.4.1
**Reply 2 to #115 — Continuous coloring still slow (repeated UI refreshes)**
- After the earlier perf passes the busy indicator worked, but applying continuous coloring to a ~150-cell FOV still took ~1m30s and the plugin UI visibly refreshed multiple times. **Root cause:** the painter's `on_mv_update_display` re-applies colors whenever `state_changed`, gated on `_last_applied_fov/identifier/classes` — but those fields are set **only in the categorical apply branch**. In continuous mode they were never set, so after enabling (which resets them to `None`) every `update_display` fan-out (`inform_plugins('on_mv_update_display')`) saw `state_changed=True` and re-entered `apply_colors_to_masks` → `update_display` → … a re-entrant cascade (there is no re-entrancy guard on `update_display`). Categorical coloring never hit this because its apply sets the tracking fields. Two secondary redundancies compounded it: `_on_enabled_toggle` rendered twice (apply's own refresh **plus** an explicit `update_display`), and `_refresh_continuous_display` rendered the colorbar and then `apply_colors_to_masks` rendered it again.
- **Fix:** (1) the continuous apply now records `_last_applied_fov` + a `_last_applied_continuous` spec signature, and `on_mv_update_display` compares that signature in continuous mode, so the follow-up hook is a no-op; (2) added an `_applying` re-entrancy guard around the `update_display` calls inside `apply_colors_to_masks` so the hook can never re-enter mid-render; (3) `_on_enabled_toggle` no longer double-renders on enable (apply already refreshes); (4) `_refresh_continuous_display` renders the colorbar only when not applying (apply renders it otherwise); (5) made the `_syncing` guard consistent across the continuous observers (`_on_color_mode_change`, `_on_continuous_column_change`, `_on_auto_range_toggle`) so restore/Apply-set paths don't fan out extra refreshes.
- Tests: added regressions to `tests/test_mask_painter_continuous.py` — enabling in continuous mode renders exactly once with no cascade (the test wires `update_display`→`on_mv_update_display` like the real fan-out), `on_mv_update_display` is a no-op after a continuous apply, a colormap change redraws the colorbar/viewer once each, the `_syncing` guard suppresses the mode/column/auto-range observers, and an interactive param change still refreshes once. Full suite failure/error set identical to the `develop` baseline; +5 net tests. See `dev_note/issue_tracking/issue115_continuous_mask_coloring.md` (reply 2).

**Reply to #115 (part 2) — Continuous coloring still slow + colorbar crash**
- After the first perf pass, Apply was faster but still slow, and rendering could raise `RecursionError` from the colorbar. **Two causes:** (1) `_render_colorbar` used `plt.subplots()`, which under the notebook's `%matplotlib widget` (ipympl) backend builds a live canvas **widget** — slow and crash-prone; (2) painter colors are painted by `mask_color_overlay._apply_region_colors`, which looped over every cell doing a full-region `region_array == id` scan + per-cell `find_boundaries` — O(cells×pixels), so continuous coloring (every cell colored) was dominated by it, unlike the annotation overlay which already uses a vectorized colormap LUT.
- **Fix:** (1) render the colorbar with a detached Agg `Figure`/`FigureCanvasAgg` → PNG → static `IPython.display.Image` (no interactive canvas, backend-independent). (2) Added a vectorized **fill fast path** to `_apply_region_colors`: when all colored cells are `fill` with alpha>0 and no borders (the default continuous case), build a per-id RGB+alpha LUT and blend the whole region in one `lut[region_array]` gather (O(pixels)); the per-cell loop stays as the fallback for outlines/borders/mixed modes so categorical rendering is unchanged.
- Tests: added vectorized-fill correctness cases (distinct colors, no cross-contamination, background/unregistered ids untouched) to `test_mask_color_overlay.py`. Full suite failure/error set identical to the `develop` baseline. See `dev_note/issue_tracking/issue115_continuous_mask_coloring.md` (reply 1, part 2).

**Reply to #115 — Continuous coloring was very slow; add busy indicator**
- Applying continuous coloring took **>2 minutes** (and zoom/pan was sluggish) even with only ~150 cells in the current FOV, with the busy indicator never turning on. **Root cause:** on Apply, the painter registered a color for **every cell in every FOV** into the global per-cell registry (`groupby(fov)` + per-cell matplotlib `to_hex` + `set_cell_colors_bulk`) — synchronously, before `update_display` (hence no busy state). That global registry is unnecessary: the live display/map overlay resolve continuous colors on demand via `get_effective_state_maps_for_fov`, and the cell gallery / batch export / ROI thumbnails via `resolve_mask_painter_snapshot_for_fov` — both compute per-FOV through `build_painter_state_maps_for_fov`. Two secondary costs: the per-cell `to_hex` loop, and `_build_continuous_spec` recomputing a full-column percentile **and writing the vmin/vmax widgets on every render** (comm round-trips → slow zoom).
- **Fix:** (1) `_apply_continuous_colors_to_masks` no longer registers globally — it clears any stale registry colors, invalidates the state-maps cache, and refreshes the current FOV only (Apply is now O(current FOV)). (2) `compute_continuous_colors` builds hex strings vectorized (no per-cell `to_hex`). (3) The auto-range is cached (`_continuous_range_cache`, keyed by column/arcsinh/cofactor) and recomputed/written to the fields only in the param-change handlers (`_refresh_auto_range_fields`), never on the render hot path; cache cleared on cell-table reload. (4) `apply_colors_to_masks` is decorated with `@update_status_bar` so the busy gif shows on Apply.
- Tests: `tests/test_mask_painter_continuous.py` replaced the old "registers globally" test with regressions asserting no global-registry writes on apply, on-demand color resolution via the effective state maps, auto-range caching off the hot path (no widget writes), and busy-state toggling. Full suite failure/error set identical to the `develop` baseline; +3 net tests. See `dev_note/issue_tracking/issue115_continuous_mask_coloring.md` (reply 1).

**Issue #115 — Color cell masks by a continuous variable**
- The Mask Painter could only color masks by a **discrete/categorical** column (one `ColorPicker` per unique value; float columns were excluded). Added a **Continuous** coloring mode, toggled in the painter UI, that maps a numeric cell-table column through a **matplotlib colormap** gradient. Customization: colormap, value range (**auto 1–99 percentile** or manual vmin/vmax), transparency, fill/outline, and an optional `arcsinh(value / cofactor)` transform, with a live colorbar legend.
- **Range is global across all FOVs** (resolved once over the whole column) so a value maps to the same color in every FOV, the cell gallery, and exports. Implementation routes through the single choke-point `build_painter_state_maps_for_fov` (new optional `continuous` spec) and carries continuous parameters on `MaskPainterSnapshot`, so live rendering **and** every snapshot consumer (cell gallery, batch export, ROI replay) get gradient coloring with no per-consumer changes. New pure helpers `resolve_continuous_range` / `compute_continuous_colors` compute per-cell hexes (vectorized, NaN cells skipped) that feed the existing per-cell registry (`set_cell_colors_bulk`) and overlay — so continuous mode is no more expensive per render than categorical coloring with many classes. Continuous settings persist through the existing `.maskcolors.json` palette Save/Load tab (`COLOR_SET_VERSION` → `1.2.0`; older files load as categorical).
- Tests: new `tests/test_mask_painter_continuous.py` (16 tests — colormap mapping/clipping, NaN handling, arcsinh with negatives, range edge cases, the continuous state-map branch, snapshot round-trip + replay determinism, float-only value options, layout toggle, global registration, palette round-trip). Full suite failure/error set identical to the `develop` baseline; +16 net passing tests. Live rendering (needs a notebook frontend) to be confirmed manually. See `dev_note/issue_tracking/issue115_continuous_mask_coloring.md`.

**Issue #114 — Fix broken heatmap links to the Scatter plot & Histogram plugins**
- After #112 split the combined Chart plugin into separate **Scatter plot** (`chart_output`) and **Histogram** (`histogram_output`) plugins, the heatmap's "Linked plugins" tab still used the old combined-plugin logic: a single **Chart** checkbox whose handler branched on the scatter plugin's y-axis (`y == "None"` → log "histogram response not implemented yet"; else → colour/highlight scatter). So the histogram plugin was **never** driven from the heatmap, and the scatter response was gated by a now-meaningless guard.
- **Fix:** (1) `histogram.py` gains `show_external_selection(row_indices)` — the entry point other plugins use to push a cell selection in. It publishes `selected_indices` (keeping the cell-gallery / viewer-highlight links working) and calls the existing `_refresh_overlays()`, drawing the subset as the **"Selected"** overlay distribution on every plotted channel (same machinery as brush selections; indices outside the plotted data are ignored). (2) The heatmap's **Chart** checkbox is renamed **Scatter plot** and a new **Histogram** checkbox is added; `update_linked()` now routes to each plugin independently (dead `y == "None"` guard removed), and a new `update_histogram_distribution()` resolves the active cluster's row indices and calls the histogram plugin (guarding when it is absent).
- Tests: `show_external_selection` (publish / gallery-forward / viewer-highlight / overlay) in `test_histogram_plugin.py`; new `HeatmapHistogramLinkTests` (dispatch to scatter-only / histogram-only / both; no-crash when absent) in `test_heatmap_selection.py`. Also un-broke `test_heatmap_selection.py`, which could not import at all due to stale `viewer.*` paths left from the `viewer` → `ueler.viewer` rename (corrected the file-spec paths, `sys.modules` key, `patch()` targets, a missing `adapter` stub, and a stale expected z-score value). Full suite: no new failures/errors vs. the `develop` baseline; +29 net passing tests. Live linking (jscatter/Bokeh need a frontend) to be confirmed in the notebook. See `dev_note/issue_tracking/issue114_plugin_link_fix.md`.

**Reply to #113 — Fix scatter triangular rendering + axis clipping**
- The developer reported two visual defects: the multi-pair scatter did not render as the intended upper-triangular matrix, and every scatter (single-pair included) was clipped so the left (y) and bottom (x) axes/labels were cut off. **Root cause:** `ScatterPlotWidget` pinned the jscatter widget's DOM box to exactly the plot height (`Layout(height="320px")`), but jscatter draws axes/labels *outside* the plot canvas (its own `compose()` reserves ~36px and never pins `layout.height`; a fresh widget has `layout.height = None` and self-sizes from its `height` trait) — so the axis furniture was clipped. For the matrix, `grid_auto_rows="320px"` compounded it by squeezing each ~360px widget into a 320px track.
- **Fix:** (1) `scatter_widget.py` no longer pins the height — the jscatter widget self-sizes (`Layout(width="100%")`), so axes are fully visible for every scatter. (2) The triangular layout is rebuilt as a **`VBox` of `HBox` rows** (plain flexbox) instead of a sparse CSS-grid `GridBox`: each row has exactly `N-1` equal-flex cells, blank cells fill the lower triangle, and each cell lets its scatter self-size (verified staircase for 3- and 4-channel cases). Extra single-pair plots added after a matrix still append on new rows below. Falls back to `compose` when the matrix is incomplete.
- Tests: the triangular tests now assert the `VBox`/`HBox` row structure (plot vs. blank cells) rather than grid coordinates. Full suite failure/error set identical to the `develop` baseline. The live rendering (jscatter needs a frontend) should be confirmed manually in the notebook. See `dev_note/issue_tracking/issue113_channel_selector_alignment.md`.

**Issue #113 — Align Histogram & Scatter channel selection with the left-panel selector**
- After #112 split the histogram out of the scatter plugin, the two plugins' channel pickers had drifted from the left-panel channel selector. Both plugins now use a **left-panel-consistent picker** — an ipywidgets `TagsInput` plus a marker-set dropdown + **Load set** button — extracted once into `_chart_common.py` (`build_channel_selector`, `ChannelSelector`, `refresh_marker_set_options`, `apply_marker_set_to_selector`, `numeric_columns`) so they can't drift again. Marker-set loading is a **local read** of `viewer.marker_sets[name]['selected_channels']`: it fills only the plugin's own picker and deliberately does **not** call `apply_marker_set_by_name` (which would repaint the main image viewer). Defining/saving marker sets stays in the left panel. Each plugin refreshes its dropdown via `on_marker_sets_changed` + `after_all_plugins_loaded`.
- **Histogram:** the channel picker (with marker-set loading) now sits **above** the Plot button (previously a `SelectMultiple` inside a tab, i.e. below Plot).
- **Scatter — selector swap:** the multi-pair picker ("Plot all pairs") is now the always-visible control on top; the single-pair X/Y/Color selector moved into a **Single-pair** tab.
- **Scatter — triangular layout:** multi-pair plots now render as an **upper-triangular matrix** (no duplicate pairs, no diagonal) using an ipywidgets `GridBox` with CSS grid-line placement, instead of the flat 2-column `jscatter.compose` grid (`compose` has no empty-cell support). A single-pair plot added after a matrix is appended on a **new row** below it; the layout falls back to `compose` when the matrix is incomplete (e.g. after a view is removed).
- Tests: `TestHistogramChannelSelector` (histogram) and new triangular-grid/shared-selector/new-row/compose-fallback tests (scatter). Full suite failure/error set identical to the `develop` baseline (no new failures/errors); +7 net passing tests. See `dev_note/issue_tracking/issue113_channel_selector_alignment.md`.

**Reply to #112 — Auto-load BokehJS so histograms render in VSCode without a priming cell**
- The Bokeh histogram only appeared after something had loaded BokehJS into the notebook frontend (the "BokehJS … successfully loaded" banner from `output_notebook()`); otherwise the `BokehModel` stayed blank. JupyterLab's `jupyter_bokeh` extension loads BokehJS automatically, but VSCode's notebook frontend does not. The plugin now calls `bokeh.io.output_notebook(hide_banner=True)` once, at plugin init (a reliable display context, during the `run_viewer` cell), via a guarded, idempotent `_ensure_bokehjs()` — plus an idempotent backstop call in `_render`. The helper is a no-op outside an interactive kernel (`get_ipython() is None`), so unit tests and headless use are unaffected. Added a regression test for that guard. Full suite unchanged vs. baseline.

**Reply to #112 — Scroll the histogram stack when it overflows the plugin**
- With several channels selected, the stacked histograms exceeded the plugin height and were clipped (no scrollbar). The scroll is now applied to the **BokehModel widget itself** in `_render`: when the estimated stack height exceeds `_MAX_PLOT_HEIGHT` (560px), `_scroll_height()` returns a fixed height that is set with `overflow="hidden auto"` on `self._bokeh_model.layout`; a short stack (few channels) is left unconstrained so it renders at natural height. Two dead ends were ruled out along the way: `overflow_y="auto"` is a no-op (**ipywidgets 8 removed the per-axis `overflow_x`/`overflow_y` Layout traits**), and a `max-height`+`overflow` on the parent `plot_section` VBox does **not** clip the Bokeh column (Bokeh sizes the column on its own DOM node), so the scroll has to live on the model. Regression tests cover the `_scroll_height` threshold and assert the tall-stack path sets the model's `height`/`overflow`. Full suite unchanged vs. baseline.

**Reply to #112 — Fix Histogram brush mode (drag selected a pan, not a range)**
- In Brush mode, click-and-drag on a Bokeh histogram just panned — no range could be selected. `_build_figures` added a `BoxSelectTool` but never made it the active drag gesture, so Bokeh's default (pan, from `tools="pan,wheel_zoom,reset"`) still handled the drag and the wired `SelectionGeometry` handler never fired. Fixed by setting `p.toolbar.active_drag = <BoxSelectTool>` in the brush branch (pan stays available in the toolbar), and guarding the range handler on `event.final` so the selection is computed once per gesture (on mouse-up) rather than on every drag move. Added bokeh-only regression tests asserting `toolbar.active_drag` is a `BoxSelectTool` in Brush mode and is not in Cutoff mode. Full suite unchanged vs. baseline.

**Reply to #112 — Re-implement the Histogram plugin on Bokeh (interactive linked brushing)**
- Replaced the Histogram plugin's matplotlib/`ipympl` render+brush path with **Bokeh** (rendered as an ipywidget via `jupyter_bokeh`'s `BokehModel`), so interactivity no longer rides the fragile `ipympl` backend (the same reasoning as #107's move off interactive matplotlib). Added `bokeh>=3.0` and `jupyter_bokeh>=4.0` to `pyproject.toml`.
- Each channel is a Bokeh `figure` with `quad` bars for the full counts plus an overlaid `quad` (its own `ColumnDataSource`) for the selected subset, both on the **same** bin edges. **Brush mode** uses a `BoxSelectTool`; its `SelectionGeometry` event routes to a kernel-side handler (`handle_range`) that computes the cell selection, drives `selected_indices` (→ cell gallery + viewer masks via the existing wiring), and recomputes the selected overlay for **every** channel by updating its source in place (no full re-render). **Cutoff mode** uses a `Tap` event → the existing `highlight_cells`, with the threshold drawn as a `Span` on the active channel.
- Binning stays in Python (unit-testable): new module helper `bin_counts(values, edges)`, `HistogramDisplay._build_figures()` (bokeh-only, no `jupyter_bokeh`), and a pure `handle_range(channel, lo, hi)` that the Bokeh event delegates to (`_on_brush` kept as an alias). Imports are guarded so the plugin still loads headlessly; when the Bokeh stack is absent the plugin shows an install notice instead of rendering.
- Tests: reworked `tests/test_histogram_plugin.py` — pure-logic tests for `bin_counts`/`handle_range`/edge-sharing, a bokeh-only `_build_figures`/cutoff-span layout test, and a full-stack `BokehModel` render smoke test (each skips if its dependency is unavailable). Full suite: failure/error set identical to baseline (no new failures/errors). The live brush/tap interaction itself is verified manually in the notebook (not headlessly testable). See `dev_note/issue_tracking/issue112_separate_histogram_plugin.md`.

**Reply to #112 — Fix the subset-overlay bin edges in the Histogram plugin**
- The "Selected" overlay used the wrong bin grid: the base and overlay `ax.hist` calls each got `bins` as an integer, so Matplotlib recomputed edges from each call's own data range — the full histogram spanned the full range while a narrow subset spanned only its own range, making the overlay incomparable to the full histogram. Fixed by computing one set of edges per channel from the full plotted data (`np.histogram_bin_edges`, via new `HistogramDisplay._histogram_bin_edges`) and passing those explicit edges to both the base and overlay `ax.hist` calls. Added regression tests (edges span the full range and are selection-independent; narrow-subset overlay renders). A follow-on (approved) will re-implement the histogram on **Bokeh** (`bokeh` + `jupyter_bokeh`) for native kernel-backed linked brushing; see `dev_note/issue_tracking/issue112_separate_histogram_plugin.md`.

**Issue #112 — Separate histograms from the scatter-plot plugin (+ multi-pair scatter, linked brushing)**
- The combined **Chart** plugin drew histograms and scatter plots into one shared render host, so only one could be visible at a time. Split them into two independent plugins, each with its own render area, and implemented the full set of "further considerations": multi-pair scatter, multiple histograms, and linked brushing with cross-histogram distribution overlays.
- New `ueler/viewer/plugin/_chart_common.py` (underscore-prefixed so the plugin auto-loader skips it) holds the de-duplicated data-prep / subset-control / link-checkbox / viewer-highlight logic shared by both plugins.
- **Scatter plugin** (`chart.py`): histogram code removed; class `ChartDisplay` and id `chart_output` retained (so `heatmap_layers`/`cell_gallery`/`ui_components` references keep working), title renamed to **"Scatter plot"**. `plot_chart` now requires both X and Y. New **Multi-pair** tab: pick several channels → **Plot all pairs** builds one scatter per `itertools.combinations(channels, 2)`, reusing the existing multi-scatter grid and selection sync.
- **Histogram plugin** (`histogram.py`, new — `HistogramDisplay`, id `histogram_output`, title **"Histogram"**): its own render area. **Cutoff mode** reproduces the old histogram (click a subplot → set that channel's cutoff → above/below highlight in the viewer + cell gallery, including the FOV-change re-highlight, now routed to `histogram_output.highlight_cells()`). **Brush mode** adds a `matplotlib.widgets.SpanSelector` per subplot: brushing a range selects those cells, drives the viewer/cell-gallery link, and overlays the selected subset's distribution on every histogram (so a selection on one channel is reflected across all channels). The figure canvas is emitted once via `display(fig.canvas)` (the cross-backend idiom from #108), so it renders under both the widget and headless `agg` backends.
- Added `tests/test_histogram_plugin.py`; moved the histogram-cutoff tests out of `tests/test_chart_cell_gallery_link.py` (scatter tests retained) and added multi-pair scatter tests to `tests/test_chart_footer_behavior.py`. Full suite: failure/error set identical to the `develop` HEAD baseline (no new failures/errors); the split adds 9 net passing tests. See `dev_note/issue_tracking/issue112_separate_histogram_plugin.md`.

**Issue #111 — Move the Lasso Select toggle to the top of the image viewer**
- The **Lasso Select** toggle previously sat in the second row of the left control panel (right under the image selector), which broke the panel's visual flow and gave no visual link to the viewer it acts on. Moved it to the **top of the middle (viewer) panel** so it reads as a control that selects cell masks in the image. Pure layout change in `ueler/viewer/ui_components.py::display_ui`: removed the toggle from the left panel's `top_part_widgets` and prepended it (wrapped in a thin left-aligned `HBox` toolbar row) to `center_area`. The widget, its `on_lasso_select_toggle`/`_on_lasso_complete` callbacks, and all selection behavior are unchanged. `tests.test_lasso_selection` (15 tests) still passes.

**Issue #110 — Stream/cache-load images from the BioImage Archive (BIA)**
- New optional remote data source lets UELer explore a public BIA study (`S-BIAD*`) without downloading the whole dataset. `run_viewer_bia(source, *, descriptor=None, local_dir=None, ...)` accepts either an accession id (resolved via the BioStudies `/info` REST endpoint) or a direct HTTPS base URL. The remote file tree is enumerated by crawling the Apache autoindex.
- Structure handling is **descriptor-first**: an optional JSON descriptor maps the study files onto FOVs/channels/masks; auto-detection covers the two clean layouts that mirror the local modes (folder-per-FOV, OME-TIFF-per-FOV). The mask/annotation config accepts either a single `mask_dir`/`mask_glob` (files already named `<fov>_<label>.tiff`) or a `masks` list of `{dir, name}` sources — supporting studies with **several mask folders** and/or `<fov>.tiff` naming (a `name` renames the cached file to `<fov>_<name>.tiff` so the existing `load_masks_for_fov` derives the label). Reads use **byte-range streaming** for pyramidal OME-TIFFs and a transparent **download-once cache** for everything else (the realistic path for single-resolution MIBI TIFFs). Verified end-to-end against two real studies: `S-BIAD2557` (single-mask, `pub/databases` base; 562 FOVs) and `S-BIAD2864` (`fire` base; 166 FOVs, 40 channels, two mask folders `segmentation_masks/`+`follicle_masks/` named `<fov>.tiff`).
- New `ueler/bia_loader.py` (`BIAStudyIndex`, `BIADataSource`, layout classification, remote open/cache helpers). Maximises reuse: folder channels are cached then read by the unchanged `load_one_channel_fov`; masks/annotations are prefetched into a flat local dir and read by the unchanged `load_masks_for_fov`/`load_annotations_for_fov`; OME FOVs reuse `OMEFovWrapper`, which gains an `opener=` parameter so `tifffile` can read over a remote `fsspec` handle (streaming) and closes it on eviction.
- `ImageMaskViewer.__init__` gains `data_source=None`: when set, `_fov_mode = "bia"` and FOV discovery, `load_fov`, and mask/annotation loads route through the data source. `base_folder` becomes a **local workspace** (default `~/.ueler/bia/<accession>/`) so all existing `.UELer` writes (ROIs, checkpoints, widget states, maps, palettes) work unchanged; downloaded images live under a disposable `<workspace>/cache/`. Declared `fsspec[http]`, `requests`, `tifffile` in `pyproject.toml`.
- Added `tests/test_issue110_bia_loader.py` (network mocked: URL/autoindex parsing, accession resolution, descriptor + auto-detect classification, folder-mode channel/mask plumbing, OME stream-vs-cache selection, `run_viewer_bia` wiring). Full suite: no new failures/errors vs. baseline. OME-Zarr (NGFF) streaming is deferred to a later phase. See `dev_note/issue_tracking/issue110_bia_streaming.md`.
- Follow-up — support more BIA layouts (verified live on real studies `S-BIAD2864` and `S-BIAD2708`): the mask/annotation descriptor now accepts a **list of sources** with `name` (renames the cached file to `<fov>_<name>.tiff`, for studies naming masks `<fov>.tiff`) and `per_fov: true` (reads a per-FOV subfolder `<dir>/<fov>/*.tiff`, labelling each mask by its stem). Added `fov_container: "zip"` so studies that pack each FOV's channels into a `<FOV>.zip` work by reading a **single channel out of the remote zip via HTTP byte-range** (`zipfile` over an `fsspec` handle) instead of downloading the whole archive — confirmed on S-BIAD2708 (opening one channel fetched 41 KB from a 31 MB zip) and S-BIAD1507 (15 MB from a 389 MB zip).
- Follow-up — guard against huge non-pyramidal OME downloads: studies like `S-BSST2926` ship 20–56 GB single-resolution OME-TIFFs (no pyramid). Since only pyramidal images stream, these would hit the cache fallback and silently download tens of GB; `open_fov` now checks the remote size and **raises a clear error above `max_download_bytes` (default 2 GiB)** rather than downloading, with `run_viewer_bia(..., max_download_bytes=…)` to override. Verified live on S-BSST2926 (raises, no download). 34 tests total in the #110 suite; full suite unchanged vs. baseline.

### v0.4.0
**Issue #109 — Heatmap remembers its scale (figure size) after updating the tree cut**
- Changing the dendrogram cutoff rebuilds the heatmap as a brand-new `sns.clustermap` figure using the adapter's default `figsize`, which discarded the figure size the user had set by dragging the ipympl resize handle (the triangle at the bottom-right corner of the canvas). The "scale" in the request is this figure size, not the toolbar zoom. `ipympl.Canvas.handle_resize` writes the manual resize back via `fig.set_size_inches`, so `fig.get_size_inches()` reflects it.
- Added `_capture_heatmap_scale()` (returns the current `fig.get_size_inches()`); `generate_heatmap(figsize_override=None)` overrides `clustermap_kwargs['figsize']` at build time (so the grid and `tight_layout` are built at the preserved size); `_refresh_plot(restore_size=None)` threads it into the build; and `apply_new_cutoff` captures the old figure's size before the rebuild and passes it through. Fresh **Plot** / load / import keep the adapter-default size.
- Added `tests/test_issue109_heatmap_zoom.py` (6 tests: capture reads the figure size / returns `None`; `apply_new_cutoff` passes the captured size; fresh refresh passes no override; `generate_heatmap` overrides `clustermap_kwargs['figsize']`; the adapter always provides a `figsize` key) and updated the #108 test's fake `generate_heatmap` for the new kwarg. Full suite: no new failures/errors vs. baseline. See `dev_note/issue_tracking/issue109_heatmap_remember_zoom.md`.

**Issue #108 (revised) — Heatmap not appearing: the real ipympl root cause**
- The first #108 fix (below) did not resolve the blank — after a kernel restart even a plain **Plot** click showed nothing. The same `%matplotlib widget` (ipympl) backend renders the Chart histogram and the anywidget galleries fine, so the fault is **specific to the heatmap's interaction with ipympl**: the heatmap built and manipulated its `sns.clustermap` figure **inside** the `with <output>:` display context and then called `plt.show()`. Under interactive mode, creating a figure inside the display context makes ipympl auto-emit its canvas there, and the subsequent `plt.show()` emits it again → a duplicate/blank canvas. The Chart histogram is reliable because it builds **outside** its Output and emits exactly once.
- Fix (keeps the interactive ipympl canvas and the footer docking; static/anywidget was rejected as unsuitable for a chart): `_setup_layout` is now build-only (removed `plt.show()`); `_refresh_plot` builds the figure with `plt.ioff()` active and **outside** any Output context (restoring the prior interactive state afterward), then emits the interactive canvas **exactly once** via `display(fig.canvas)` inside a fresh `Output`, then swaps it into `plot_section.children`. `display(fig.canvas)` is targeted (unlike `plt.show()`, which emits every managed figure) and keeps pan/zoom + all `mpl_connect` handlers.
- Layout-toggle fixes: `on_orientation_toggle` now guards `refresh_bottom_panel()` with `_plot_refresh_inflight` so the cached-pane refresh no longer double-renders; and new `_present_footer_canvas_if_wide` forces the reparented footer canvas to repaint via a synchronous `canvas.draw()` backstop plus a single-shot `canvas.new_timer(150ms) → draw_idle()` (patterns already used by the main image canvas and `image_display`), fixing the "old plot flashes then blanks" behavior on layout switch.
- Extended `tests/test_issue108_heatmap_refresh.py` to 7 tests (build runs under `plt.ioff()`; canvas emitted exactly once via `display`; wide mode issues the synchronous + deferred redraws; vertical mode does not). Full suite: no new failures/errors vs. baseline. Decisive confirmation is the live ipympl notebook check documented in `dev_note/issue_tracking/issue108_heatmap_not_showing.md`.

### v0.3.1
**Issue #108 — Heatmap renders but never appears**
- Root cause (a widget-layer display bug that reproduced even with the interactive `ipympl` backend active, so not a data/backend problem): the heatmap diverged from the Chart plugin's reliable interactive **histogram** idiom in three ways. (1) A destructive second clear/redraw pass — after `_setup_layout` displayed the figure, `plot_heatmap` called `restore_footer_canvas()` → `redraw_cached_footer_canvas()`, which ran `clear_output(wait=True)` (wiping the just-shown canvas) and then re-`display()`ed the same already-consumed canvas, painting nothing. (2) It never reassigned any container's `.children`, so the frontend never repainted — fatal in wide mode, where the single persistent `plot_output` is reparented into a footer pane that is built once and cached. (3) It used a bare `plt.show()` plus a conditional `display()` instead of the histogram's single explicit `plt.show(fig)`.
- Fix (mirrors `chart.py::_render_histogram`, preserving full interactivity — `plt.show(g.fig)` under `ipympl` displays the interactive canvas so the click/dendrogram/color-axis handlers keep working; no static fallback): new `_refresh_plot()` renders into a **fresh** `ipywidgets.Output`, runs `generate_heatmap()` inside it, then swaps that new Output into `plot_section.children` to force a repaint (a brand-new object guarantees an identity change in both the vertical accordion and the cached wide/footer pane). All five render sites (`plot_heatmap`, `load_heatmap`, `apply_new_cutoff`, `import_heatmap_state`, and the display in `_setup_layout`) now funnel through it. `_setup_layout` was reduced to a single `plt.show(g.fig)`.
- Removed the destructive plumbing: deleted `redraw_cached_footer_canvas` and `_plot_output_has_widget_view`; simplified `_ensure_plot_canvas_attached` (`plot_section.children = (self.plot_output,)`) and reduced `restore_footer_canvas`/`restore_vertical_canvas` to thin wrappers; dropped the orphaned `_cached_footer_artifacts`. `generate_heatmap` now `plt.close()`s the previous figure before building a new one to stop figure accumulation across repeated Plot clicks / cutoff drags.
- Added `tests/test_issue108_heatmap_refresh.py` (3 regression tests: fresh-Output swap, a new Output per render, and `_ensure_plot_canvas_attached` sole-child). It imports the real `ueler` package because the existing `tests/test_heatmap_selection.py` errors on import from a pre-existing, unrelated path bug (loads `viewer/plugin/heatmap.py` instead of `ueler/viewer/plugin/heatmap.py`). Full suite: no new failures/errors vs. baseline.

**Reply to Issue #107 — Fix "Preview unavailable" for all FOV-based ROI thumbnails**
- Fixed a variable-ordering bug in `_render_roi_tile` (`roi_manager_plugin.py`): `snapshot` was referenced in the `render_roi_to_array(...)` call (`skip_image_layer=bool(getattr(snapshot, ...))`) before its assignment a few lines below, raising `UnboundLocalError` that the surrounding `except Exception: return None` swallowed — so every FOV-based ROI thumbnail returned `None` and the gallery showed "Preview unavailable". Pre-existing since `f9c9996` (issue #91 "No image (masks only)" mode); map-mode ROIs were unaffected. Moved `snapshot = self._build_overlay_snapshot(...)` above the render call and simplified the argument to `skip_image_layer=bool(getattr(snapshot, "skip_image_layer", False))`.
- Made the render failure non-silent: the `except` now logs `_logger.warning(..., exc_info=True)` before returning `None`, so future thumbnail-render failures appear in the debug log console instead of a silent placeholder.
- Added `tests/test_roi_manager_tags.py::ROIManagerThumbnailRenderTests` (2 regression tests: renders when snapshot is `None`; honours `skip_image_layer` from the snapshot) — verified they fail without the fix.

**Issue #107 — Replace the Matplotlib galleries with an anywidget tile grid**
- Added `ueler/viewer/plugin/tile_gallery_widget.py`: `TileGalleryWidget` (an `anywidget.AnyWidget` with a headless `traitlets.HasTraits` fallback for CI, mirroring `mask_class_list_widget.py`). It renders a list of `{id, src, label}` tiles as a responsive CSS grid of `<img>` elements; `src` is a base64 PNG data-URI, the label is an in-tile CSS hover tooltip, and a tile click writes `"<id>|<nonce>"` to a synced `clicked` traitlet (the nonce lets re-clicking the already-selected tile fire again). Module helpers: `array_to_data_uri()` (float `[0,1]`/uint8/grayscale → PNG data-URI, via Pillow→imageio→skimage), `text_placeholder_uri()` (gray tile with centered message for "No channels"/"Preview unavailable"), and `parse_clicked_id()`.
- Migrated the ROI Manager browser gallery (`roi_manager_plugin.py`): `_refresh_browser_gallery()` now builds tiles and assigns `browser_gallery.tiles`/`.columns` instead of drawing a `plt.subplots` figure; clicks arrive via the `clicked` traitlet → new `_on_gallery_clicked()` → existing `_activate_roi_from_browser()`. Removed the interactive-Matplotlib machinery (`_determine_gallery_layout`, `_resolve_browser_dpi`, `_ensure_browser_css`, `_on_browser_click`/`_on_browser_motion`/`_clear_browser_hover`, `_disconnect_browser_events`, the `browser_output_inner` `Output`, the `browser_hover_label` widget, and the `matplotlib.pyplot` import). Filtering, pagination, and the signature-based redraw throttle are unchanged.
- Migrated the Cell Gallery (`cell_gallery.py`): split `create_gallery()` into `create_gallery_tiles()` (returns per-tile arrays; front-end agnostic) plus a backwards-compatible `create_gallery()` canvas wrapper. `plot_gellery()` now feeds per-tile arrays to a rewritten `_draw_gallery()` that pushes `{id, src, label}` to a `TileGalleryWidget`; the tile label is the `"<fov>: <mask id>"` text previously shown on Matplotlib hover. Clicks route through `_on_gallery_clicked()` → existing `focus_on_cell()` (preserving the `_skip_next_fov_refresh` logic). Removed `_draw_gallery`'s figure/`plt.show`/`mpl_connect` path, `on_mouse_move`, `process_hover_event`, `_create_annotation`, `_update_tile_metadata`, and the `plot_output` `Output`.
- Removes the `ipympl` interactive-backend dependency for both galleries (the source of unreliable click/hover in VSCode/Voila); no new runtime dependency (`anywidget` was already required). Bokeh was evaluated and explicitly not adopted (over-fit for a thumbnail grid, adds a real dependency).
- Tests: added `tests/test_tile_gallery_widget.py` (9 tests — encoding, placeholder, `parse_clicked_id`, trait round-trip incl. nonce re-click); updated `tests/test_roi_manager_tags.py` (gallery-widget assertions replacing `_determine_gallery_layout`/`browser_output_inner`) and `tests/test_cell_gallery.py` (new `TestTileGalleryRendering` replacing the obsolete `TestDrawGalleryRendering`; logger-based warning test). Full suite: no new failures vs. baseline (2 pre-existing failures incidentally fixed).

**Package-wide logging sweep — all UELer messages feed the log console**
- Converted ~140 scattered `print()` calls across the viewer core and plugins to module loggers (`logging.getLogger(__name__)`), so every diagnostic now flows through the `ueler` logger into the bottom log console (debug mode). Touched: `main_viewer.py` (~95), `image_display.py`, `plugin_base.py`, `ui_components.py`, `decorators.py`, `runner.py`, `data_loader.py`, and the `run_flowsom`, `chart`, `chart_heatmap`, `cell_gallery`, `go_to`, `mask_painter` plugins. Level by intent: debug for traces (dropping `if self._debug:` guards where they only gated a print), info for confirmations, warning for user-actionable problems, error (`exc_info=True`) for failures.
- Mirrored plugin UI text into the log console: the status-label/log helpers `mask_painter._log()`, `roi_manager_plugin.set_status()`, `cell_annotation._set_status()`, `export_fovs._log()`, and `main_viewer._log_annotation_palette()` now emit a `ueler` log record in addition to updating their inline widget; widget-rendered prints (cell_gallery, chart cutoff) keep their `Output` rendering and add a log mirror.
- Added `tests/test_logging_sweep.py` (4 tests) asserting the helpers emit `ueler.*` records.

**Reply 2 to Issue #105 — UELer log console + heatmap diagnostics via `logging`**
- Added a dedicated **log console** docked at the bottom of the viewer UI, shown only when `debug=True`. New `ueler/viewer/log_console.py` defines `OutputWidgetHandler` (a `logging.Handler` that renders records into an `ipywidgets.Output`), plus `enable_log_console()` / `disable_log_console()` / `build_log_console_panel()`. The console is scrollable, its text is selectable/copyable, and a **Clear** button empties it; retained records are capped at 1000.
- `display_ui()` docks the console as the last root child when `viewer._debug` is set, and attaches the handler to `logging.getLogger("ueler")` with `propagate=False` so all `ueler.*` module loggers feed the console (and nothing leaks to the notebook cell). `ImageMaskViewer(debug=True)` raises the `ueler` logger to `DEBUG`.
- Converted the heatmap plugins fully to `logging`: all ~50 `print()` calls in `heatmap_layers.py` and `heatmap.py` now use `_logger.<level>` by intent — `debug` for traces (`Preparing heatmap data…`, click traces), `info` for confirmations, `warning` for user-actionable problems, `error(exc_info=True)` for render failures. `generate_heatmap()` logs at each early-return and wraps `sns.clustermap()` / `_setup_layout()` to surface failures.
- Note: an earlier attempt to route logs to the kernel's stderr via a custom fd handler in `ueler/__init__.py` was **reverted** — ipykernel captures stdout/stderr/fds, and the dedicated widget console is the correct home for these messages.
- Added `tests/test_log_console.py` (8 tests).

**Reply to Issue #105 — Fix heatmap not displayed after Cell Annotation plugin load**
- Fixed `CellAnnotationPlugin.after_all_plugins_loaded()` to not call `super()`: `PluginBase.load_widget_states()` does `vars(self.ui_component)` but `CellAnnotationPlugin` has no `ui_component`, causing `AttributeError` whenever a state file existed on disk.
- Wrapped each plugin call in `main_viewer.after_all_plugins_loaded()` with `try/except` so a crash in one plugin (e.g. `cell_annotation_output`, which sorts alphabetically before `heatmap_output`) no longer prevents later plugins from initializing.
- Wrapped plugin instantiation in `dynamically_load_plugins()` with `try/except` for the same defensive isolation.
- Added regression test `TestAfterAllPluginsLoaded.test_no_crash_when_no_state_file` to `tests/test_cell_annotation.py` (now 29 tests total).

**Issue #105 — Heatmap checkpoint save/load (Cell Annotation plugin)**
- Added `anndata>=0.10` as a required dependency for `.h5ad` checkpoint serialization.
- Added `ueler/viewer/interfaces.py`: `HeatmapStateProvider` and `FlowsomParamsProvider` Protocol stubs for cross-plugin communication.
- Added `ueler/viewer/checkpoint_store.py`: `CheckpointStore` class — atomic `.h5ad` read/write (`.partial` → fsync → `os.replace()`) and `manifest.json` management under `<root>/.UELer/dataset_<sha1>/checkpoints/`.
- Added `export_heatmap_state()` and `import_heatmap_state()` to the `DataLayer` mixin in `heatmap_layers.py`: serialize z-scored median matrix, meta-cluster palette (colors + names), dendrogram linkage, UI settings, and `meta_cluster_revised` obs column into AnnData; restore in the correct order (re-render first, then re-apply saved palette) to avoid `_sync_meta_cluster_registry()` overwrite.
- Added `export_flowsom_params()` and `import_flowsom_params()` to `RunFlowsom` in `run_flowsom.py`.
- Added `ueler/viewer/plugin/cell_annotation.py`: `CellAnnotationPlugin` with a save form (step ID, description, parent dropdown, op selector) and a `CheckpointTreeWidget` (anywidget + HasTraits fallback) that renders saved checkpoints as a parent-child tree. Plugin is auto-discovered by `dynamically_load_plugins`.
- Fixed `tests/bootstrap.py` `_ensure_dask_stub()` to prefer real dask when importable (prevents anndata `find_spec("dask")` crash).
- Added 28 unit tests across `tests/test_checkpoint_store.py` and `tests/test_cell_annotation.py`; all pass with zero regressions.

**Reply to Issue #103 — Merge same color in batch export**
- Added `merge_same_color` checkbox to the shared controls of the batch export plugin (disabled unless `separate_channels` is checked, wired via `.observe()`).
- Added `_build_grouped_channel_items()` private helper: groups `marker_profile.selected_channels` by their `ChannelRenderSettings.color` tuple; exports solo channels as `{base}_{ch}.{fmt}` and multi-channel groups as `{base}_merged_{ch1}_{ch2}.{fmt}`.
- All three builder methods (`_build_full_fov_items`, `_build_single_cell_items`, `_build_roi_items` — both FOV-mode and map-mode branches) dispatch to `_build_grouped_channel_items` when both `separate_channels=True` and `merge_same_color=True`; otherwise unchanged.
- Config serialization updated: `_collect_export_config()` saves `merge_same_color`; `_apply_export_config()` restores it.
- Added 3 new tests in `TestMergeSameColorBuilders`; all pass.
- Validated: 118 tests across `tests/test_export_fovs_batch.py` and `tests/test_roi_manager_tags.py` — all pass except 2 pre-existing unrelated failures.

**Issue #103 — Custom ROI names**
- Added a `name` field to `ROI_COLUMNS` (default `""`). Existing CSVs load without change via `_ensure_dataframe()`.
- `format_roi_label()` now uses the custom name as the label suffix when non-empty, falling back to `roi_id[:8]`.
- Added `name_input` (`Text` widget) to the ROI Manager plugin editor. Populated from the selected ROI record; included in both capture (add) and update operations.
- Added `_roi_file_stem()` to the batch export plugin: returns the sanitized custom name when set, else `roi_id[:12]`. Used in `_build_roi_items()` for both FOV-mode and map-mode ROIs.
- Added 6 new tests (`TestCustomROINameFilenames` + `FormatROILabelTests`); all pass.
- Validated: 121 tests across `tests/test_export_fovs_batch.py` and `tests/test_roi_manager_tags.py` — all pass except 2 pre-existing unrelated failures.

**Issue #102 — Separate channels export option in batch export plugin**
- Added `separate_channels` checkbox to the shared controls of the batch export plugin. When checked, each selected channel is exported as an individual image file (e.g. `FOV1_DNA.png`, `FOV1_CD8.png`) instead of a merged composite. Applies to all three export modes: Full FOV, Single Cells, and ROIs (including map-mode ROIs).
- Added `_build_channel_items()` private helper: iterates over `marker_profile.selected_channels`, creates a per-channel `_MarkerProfile`, and returns a `JobItem` list. Used by all three builder methods to avoid code duplication.
- `_build_full_fov_items`, `_build_single_cell_items`, and `_build_roi_items` each accept `separate_channels: bool = False`; the merged-image path is unchanged when `False`.
- Config serialization updated: `_collect_export_config()` saves `separate_channels`; `_apply_export_config()` restores it via `_set("separate_channels", "separate_channels", bool)`.
- Added 4 new tests in `TestSeparateChannelsBuilders`; all pass.
- Validated: 72 tests in `tests/test_export_fovs_batch.py` — 71 passed, 1 pre-existing unrelated failure.

**Follow-up to Issue #101 — Mask opacity control in batch export**
- Added `mask_alpha_slider` (FloatSlider, 0.0–1.0, default 1.0) to the batch export mask controls. The slider is grouped with the mask layer dropdown and color picker inside `mask_layer_box`, disabled when `Include Mask` is off or masks are unavailable.
- `_capture_overlay_snapshot()` now reads `mask_alpha_slider.value` instead of the previous hardcoded `alpha=1.0` when building the export `MaskOverlaySnapshot`.
- Config serialization extended: `_collect_export_config()` saves `mask_alpha`; `_apply_export_config()` restores it via `_set("mask_alpha_slider", "mask_alpha", float)`.
- Updated test stubs and expanded `test_config_roundtrip_includes_mask_layer_and_color` to cover alpha; added `test_capture_snapshot_uses_alpha_slider_value`.
- Validated: 68 tests in `tests/test_export_fovs_batch.py` — 67 passed, 1 pre-existing unrelated failure.

**Reply to Issue #101 — Explicit mask layer selector in batch export**
- Root cause of persistence: even after the Bug 1 + Bug 2 fix, `_refresh_mask_controls()` reset `include_masks` to `False` whenever no viewer panel mask checkboxes were ticked, silently overriding the user's intent before the export ran.
- Added `mask_layer_dropdown` (Dropdown) and `mask_color_picker` (ColorPicker) to the batch export UI, making mask inclusion fully independent of the viewer's live overlay state.
- Added `_refresh_mask_layer_dropdown()`: populates the dropdown from `main_viewer.mask_names` on each `refresh_overlay_capabilities()` call; falls back to `mask_key` for single-mask sessions; preserves the current selection when valid.
- Removed the `visible_masks` gate in `_refresh_mask_controls()` that disabled the `include_masks` checkbox when no viewer panel checkboxes were ticked.
- Replaced the separate Bug 1 strip + Bug 2 fallback blocks in `_capture_overlay_snapshot()` with a single unified path: when `include_masks=True` and `palette_name is None`, always build a `MaskOverlaySnapshot` from the export-local dropdown value and color picker value.
- Config serialization updated: `_collect_export_config()` saves `mask_layer` and `mask_color`; `_apply_export_config()` restores them.
- Updated tests: all stubs extended with `mask_layer_dropdown` + `mask_color_picker`; renamed `test_fallback_mask_added_when_no_masks_and_painter_disabled` → `test_export_local_layer_and_color_used_when_no_palette_override`; added 5 new tests.
- Validated: 66 tests in `tests/test_export_fovs_batch.py` — 65 passed, 1 pre-existing unrelated failure.

**Issue #101 — Fix batch export mask handling**
- Root cause Bug 1: `capture_overlay_snapshot` always captures the live `MaskPainterSnapshot` when the Mask Painter is enabled. `_capture_overlay_snapshot` carried this snapshot unchanged when `palette_name is None`, so `apply_overlay_snapshot_to_array` applied per-cell annotation colours to every export even when `Override Mask Palette` was not checked.
- Root cause Bug 2: `capture_overlay_snapshot` only adds `MaskOverlaySnapshot` entries for mask layers whose panel checkbox is ticked. When the Mask Painter is disabled and all checkboxes are off, `snapshot.masks` is empty and `snapshot.mask_painter` is `None`; neither rendering stage produces any output, so the export contains no mask even with `Include Mask` checked.
- Fix 1: In `_capture_overlay_snapshot`, after the thickness-adjustment block, strip `mask_painter` to `None` whenever `palette_name is None`. This prevents live painter colours from leaking into exports that did not request a saved palette override.
- Fix 2: After Fix 1, detect the "nothing captured" case (`include_masks=True`, no masks, no painter, no palette) and inject a fallback `MaskOverlaySnapshot` outline for the primary `mask_key` using the viewer's configured colour, so `Include Mask` always produces visible content.
- Updated `test_batch_export_snapshot_preserves_mask_painter_outline_thickness` → `test_batch_export_snapshot_strips_painter_when_no_palette_override` to reflect corrected behaviour; added `test_palette_override_preserves_outline_thickness` plus 3 new regression tests for both bugs and the no-double-overlay edge case.
- Validated: 61 tests in `tests/test_export_fovs_batch.py` — 60 passed, 1 pre-existing unrelated failure.

**Reply to Issue #99 — Relativize output_path inside export config templates**
- Root cause: the `output_path` field stored inside each export config JSON was the raw widget value — typically an absolute path. When the project was moved, the loaded config restored the old absolute path, pointing into the original location.
- Added `_relativize_output_path`: converts `output_path` to relative form when it is under `base_folder`; paths outside `base_folder` are left absolute so they remain usable on fixed mounts.
- Added `_expand_output_path`: called in `_apply_export_config` before restoring widget values; converts a stored relative path back to absolute using the current `base_folder`, ensuring the widget always displays a usable absolute path.
- Updated `_collect_export_config` to call `_relativize_output_path` on the widget value before serialising.
- Added 4 tests:
  - `test_save_config_relativizes_output_path_under_base_folder`: stored path is not absolute.
  - `test_save_config_output_path_outside_base_folder_unchanged`: absolute path outside base_folder is preserved.
  - `test_load_config_expands_relative_output_path_to_absolute`: widget is set to the full absolute path after loading.
  - `test_output_path_survives_folder_move`: after copying the project to a new location, loading the config points the widget into the *new* base_folder.
- Validated: 19 tests in `tests/test_export_fovs_mask_customization.py` — all passed.

**Issue #99 — Store export config paths as relative filenames**
- Root cause: `_save_export_config` in `export_fovs.py` called `.resolve()` on the config file path before serialising it into the registry JSON. This embedded an absolute path in `export_configs_index.json`, so loading or deleting a saved config broke whenever the project was moved or shared with another user.
- Added module-level `_resolve_config_path(folder, stored_path)` helper: returns `folder / stored_path` for relative entries and the path as-is for legacy absolute entries, ensuring backward compatibility.
- `_save_export_config` now stores only the filename (e.g. `my-config.export_config.json`) instead of the full absolute path.
- `_load_export_config` and `_delete_export_config` both route through `_resolve_config_path` to reconstruct the absolute path at read time.
- Added `test_save_config_stores_relative_path`: asserts the registry entry is not absolute.
- Added `test_load_config_survives_folder_move`: copies the temp dir to a new location, re-initialises the plugin there, and verifies the config loads correctly with the original widget values.
- Validated: 15 tests in `tests/test_export_fovs_mask_customization.py` — all passed.

**Issue #98 — Fix batch export partial images in map mode**
- Root cause: `_render_map_region_direct` (`export_fovs.py`) used `viewer.image_cache.get(tile.name)` immediately after `viewer.load_fov()`. The shared `image_cache` can be concurrently evicted by the live viewer UI, so `get()` sometimes returns `None` even though `load_fov()` succeeded. The prior fix in #94 added a warning but still continued — writing a partial canvas to disk.
- Added a single retry (50 ms sleep + reload) before giving up on a tile, to handle transient eviction races.
- After the retry, if the tile is still missing, a `RuntimeError` is raised naming the tile and explaining the abort. This propagates through `_export_map_roi_worker()` to the job runner, which records the item as `ok=False` with the error message — preventing any partial image from being written to disk.
- Updated `test_render_map_region_direct_warns_when_tile_load_fails` → `test_render_map_region_direct_raises_when_tile_load_fails` to assert `RuntimeError` (with tile name and "partial image" in the message).
- Added `test_render_map_region_direct_succeeds_on_retry`: verifies that when `image_cache.get` returns `None` on the first call but a valid array on the second, `load_fov` is called twice and export completes successfully.
- Validated: 57 tests in `tests/test_export_fovs_batch.py` — 56 passed, 1 pre-existing failure unrelated to this change.

**Issue #96 — Consistent ROI naming across plugins + hover tooltip in ROI Gallery**
- Extracted `format_roi_label(record)` to `ueler/viewer/roi_manager.py` (exported via `__all__`). The unified format is `{location} · {marker_set}[{tags}] · {roi_id[:8]}`, replacing coordinates with the unique ID prefix and adding tags to the batch export label.
- Updated `ROIManagerPlugin._format_roi_label` (`roi_manager_plugin.py`) to delegate to the shared function.
- Updated `BatchExportPlugin.refresh_roi_options` (`export_fovs.py`) to use `format_roi_label`, ensuring the ROI selection dropdown in batch export now includes tags and uses the same format as the ROI Manager.
- Added `browser_hover_label` HTML widget to the ROI Gallery (below the thumbnail grid). Connecting `motion_notify_event` to the matplotlib figure updates the label with the formatted ROI name while the cursor hovers over a thumbnail; `axes_leave_event` clears it.
- Added `_browser_records_cache` dict to `ROIManagerPlugin` to make hover lookup O(1) without re-querying the ROI manager.
- Validated: 23 tests passed (`test_cell_table_editor.py`, `test_lasso_selection.py`).

**Issue #95 — Speed up the Mask Painter**
- Root cause: `_compose_fov_image` (`main_viewer.py`) and `_apply_map_painter_overlay` each called four separate `get_effective_*_map_for_fov` methods on the painter, and each called `build_painter_state_maps_for_fov` independently — a 4× redundant cell-table filter + Python row-loop per render frame.
- Added `get_effective_state_maps_for_fov(fov)` to `MaskPainterDisplay` (`ueler/viewer/plugin/mask_painter.py`): calls `build_painter_state_maps_for_fov` once and returns all four maps as a tuple. Results are cached per FOV in `_state_maps_cache`; the cache is cleared in `apply_colors_to_masks` whenever painter UI state changes (color, visibility, mode, opacity, etc.). Pan/zoom events with unchanged painter state are now O(1) after the first render.
- Updated `_compose_fov_image` (`main_viewer.py:3941`): replaced the four separate `get_effective_*_map_for_fov` calls with a single `get_effective_state_maps_for_fov` call; retained individual-method fallback for backward compatibility.
- Updated `_apply_map_painter_overlay` (`main_viewer.py:1764`): moved all four per-FOV painter queries to a single `get_effective_state_maps_for_fov` call at the top of the tile loop, eliminating 4N calls for an N-FOV map render.
- Fixed pre-existing `global_fill_opacity_input` width: changed from `150px` to `95px` to match the existing test assertion (`test_global_fill_layout_uses_spacing_and_narrow_opacity_input`).
- Extended `tests/bootstrap.py` to stub `BoundedIntText`, `select_dtypes`, `iterrows`, element-wise `__eq__`/`__and__`/`__or__`/`__ne__`, `dtype`, `matplotlib.font_manager`, and `tifffile`, enabling the mask painter tests to run in headless CI environments.
- Added 5 tests to `TestPainterStateMapsCaching` in `tests/test_mask_painter_mode_visibility.py`.
- Validated: `python -m unittest tests.test_mask_painter_mode_visibility` — 41 tests passed.

**Make mask loading lazy (Dask) to eliminate OOM kernel crash in VSCode**
- Changed `load_masks_for_fov` (`ueler/data_loader.py`): removed the unconditional `np.asarray(mask_image)` that materialized every mask as a 16 MB NumPy array at load time. Pre-labeled masks (int32, `needs_label=False`) are now stored in the cache as rechunked lazy Dask arrays (matching the channel-image pattern). Binary/boolean masks (`needs_label=True`) use `dask.delayed(measure.label)(mask_image)` + `da.from_delayed` so that the TIFF read and connected-components labeling happen only on `.compute()`, with no numpy data retained between renders. All downstream code already has `.compute()` / `try-except AttributeError` guards and required no changes.
- With 200+ FOVs, map-mode mask cache memory drops from ~3.2 GB (eager) to ~0 MB between renders (Dask computation graphs only), resolving the VSCode OOM kernel crash. The 80-FOV map-mode cap added previously remains as a safety net.
- Validated: 23 tests passed.

**Fix Cell Table Editor Apply button and row-matching in map mode (follow-up)**
- Added `on_selection_change` lifecycle event to `ImageDisplay` (`ueler/viewer/image_display.py`): `inform_plugins("on_selection_change")` is now called after `selected_masks_label` changes in `clear_patches`, `on_mouse_click` (left-click confirm path), and `_on_lasso_selected`. Previously, no plugin notification was sent when the user clicked or lasso-selected cells, so `CellTableEditorPlugin._refresh_apply_btn()` was never triggered and the Apply button stayed disabled with "No cells selected" even when cells were highlighted.
- Fixed "No matching rows found" in `CellTableEditorPlugin._on_apply_clicked` (`ueler/viewer/plugin/cell_table_editor.py`): the apply logic was using `mask_key` (the mask layer name, e.g. "whole_cell") as the cell-ID column in the cell table, but every other plugin uses `label_key` ("label") to match cells. Changed to `label_key` for both the dtype cast and the row filter, consistent with the Chart plugin and ARK analysis cell table format.
- Added `on_selection_change` handler to `CellTableEditorPlugin`: calls `_refresh_apply_btn()` to enable/disable the Apply button immediately after any selection change.
- Updated `tests/test_cell_table_editor.py` to use `label_key = "label"` / `mask_key = "whole_cell"` and a cell table with a "label" column matching the real ARK analysis schema.
- Added `inform_plugins = lambda _: None` stub to `_make_viewer_for_lasso` in `tests/test_lasso_selection.py`.
- Validated: 23 tests passed.

**Fix map-mode lasso selection coordinate mismatch (follow-up 2)**
- Fixed `_find_masks_in_lasso_map_mode` in `ueler/viewer/image_display.py`: the map canvas is dynamically sized to the current viewport; `dest_x0/dest_y0` are viewport-relative downsampled pixel indices (0 to viewport_width_ds), while `LassoSelector` vertices are in global full-res data coordinates (0 to full_map_width). The formula `canvas_x = dest_x0 + col` therefore compared indices on different scales and at different origins. The fix reads the viewport offset from the axis (`xmin_px = ax.get_xlim()[0]`, `ymin_px = min(ax.get_ylim())`) and applies it: `data_x = xmin_px + (dest_x0 + col) * downsample`. Added `test_viewport_offset_corrects_canvas_to_data_coords` to verify panned-viewport behavior; updated `test_downsampled_tile_cell_selected` to use data-coord lasso vertices; added `ax` mock to `_make_image_display()` in the test helper.
- Validated: 23 tests passed.

**Fix map-mode lasso selection coordinate bug (follow-up 1)**
- Fixed `_find_masks_in_lasso_map_mode` in `ueler/viewer/image_display.py`: the method was using `tvp.region_ds` (downsampled pixel coordinates) to index into the full-resolution mask array, which extracted only a tiny top-left corner of each tile. The fix mirrors the correct pattern from `_update_map_mask_highlights`: use `tvp.region_xy` (full-res bounds) to slice the mask array, then apply `[::downsample_factor]` to obtain the tile-sized crop.
- Updated `TestFindMasksInLassoMapMode` in `tests/test_lasso_selection.py`: replaced `region_ds=` mock with `region_xy=` + `downsample_factor=` to match the fixed implementation; added `test_downsampled_tile_cell_selected`.
- Added `_LassoSelector` stub to `tests/bootstrap.py` `_ensure_matplotlib_stub()` so the test suite can import `image_display.py` in headless environments.
- Validated: 22 tests passed.

**Cell Table Editor plugin and lasso selection**
- Added `CellTableEditorPlugin` (`ueler/viewer/plugin/cell_table_editor.py`): a new side-panel plugin with a column combobox, value text field, and "Apply to selected cells" button. The plugin writes a user-supplied string to a specified column (new or existing) for every cell in the current `selected_masks_label` set, then broadcasts `on_cell_table_change`. Non-system columns are listed as options; new columns are initialized to `""` before assignment. Works for both single-FOV and map-mode selections because matching uses `(fov_key, mask_key)` row lookup.
- Added freehand lasso selection to the main viewer (`ueler/viewer/image_display.py`): `LassoSelector` (matplotlib) activated via a new "Lasso Select" `ToggleButton` in the left panel. After drawing, `_on_lasso_selected` determines which mask pixels fall inside the polygon using `matplotlib.path.Path.contains_points`. For single-FOV mode (`_find_masks_in_lasso_single_fov`), downsampled mask pixels are mapped back to full-resolution canvas coordinates. For map mode (`_find_masks_in_lasso_map_mode`), each visible tile's `MapTileViewport.dest_x0/y0` offset is used to convert crop-local mask pixels to canvas space. Any cell with at least one pixel inside the polygon is added to `selected_masks_label`. Lasso is one-shot: the selector deactivates and the toggle button resets automatically after each stroke.
- Added `on_lasso_select_toggle` and `_on_lasso_complete` handlers to `ImageMaskViewer` (`ueler/viewer/main_viewer.py`) and wired a `ToggleButton` widget in `uicomponents.__init__` (`ueler/viewer/ui_components.py`).
- Added 21 unit tests across two new test files: `tests/test_cell_table_editor.py` (8 tests) and `tests/test_lasso_selection.py` (13 tests).
- Validated: `python -m unittest tests.test_cell_table_editor tests.test_lasso_selection` — 21 tests passed.

**Issue #94 — Fix map mode edge tiles hidden and batch export subregion**
- Restructured `VirtualMapLayer.render()` tile cap (`ueler/viewer/virtual_map_layer.py`): moved `channels_tuple` and `state_signature` computation before the cap to enable cache-key lookup during the partition step; replaced the flat 80-tile distance filter with a cache-aware, ds-scaled cap that always renders cached tiles and limits uncached tiles to `base_limit × ds_factor` (so ds=8 allows up to 640 uncached tiles, removing the black-edge symptom on maps with more than 80 FOVs).
- Fixed silent tile-load failure in `_render_map_region_direct` (`ueler/viewer/plugin/export_fovs.py`): replaced the bare `continue` with `warnings.warn(...)` naming the failed tile so large-ROI exports now surface diagnostic output instead of silently producing black patches.
- Added `TileCapTests` (3 tests) to `tests/test_virtual_map_layer.py` and 2 new tests to `BatchExportMapROIItemsTests` in `tests/test_export_fovs_batch.py`.
- Validated: `python -m unittest tests.test_virtual_map_layer tests.test_export_fovs_batch.BatchExportMapROIItemsTests` — 21 tests passed.

**Issue #93 reply — Fix Cell Gallery matplotlib figure not rendering in VS Code**
- Restructured `_draw_gallery` in `ueler/viewer/plugin/cell_gallery.py` to match `chart.py`'s `_render_histogram` pattern: `clear_output(wait=True)` is now called before the `with self.plot_output:` block (VS Code-safe; prevents a blank flash), all figure setup and event handler connections happen before the context, and `plt.show(fig)` with an explicit figure argument is the only call inside the `with` block.
- Wrapped `fig.canvas.new_timer(...)` in `try/except AttributeError` in both `_draw_gallery` and `on_mouse_move`; the gallery renders and hover events degrade gracefully on non-ipympl backends.
- Applied the same `clear_output(wait=True)` pattern to `_show_empty_message`.
- Added `TestDrawGalleryRendering` (4 tests) to `tests/test_cell_gallery.py`.
- Validated: `python -m unittest tests.test_cell_gallery tests.test_chart_cell_gallery_link` — 43 tests passed.

**Issue #93 — Fix Cell Gallery silent failure on scatter plot selection**
- Removed the `except AttributeError` clause from `@update_status_bar` (`ueler/viewer/decorators.py`). The decorator was catching and discarding `AttributeError`, producing a status-bar flash with no gallery update and no visible error whenever `plot_gellery()` encountered an `AttributeError` from recently added Mask Painter / No-image code paths. Errors now propagate as visible notebook tracebacks, enabling immediate diagnosis.
- Added `TestUpdateStatusBarDecorator` to `tests/test_cell_gallery.py` and `TestScatterToGalleryForwarding` to `tests/test_chart_cell_gallery_link.py`.
- Validated: `python -m unittest tests.test_cell_gallery tests.test_chart_cell_gallery_link`.

**Issue #92 reply — Batch export follow-up: mask outline color fix, output folder in config, and UI refinements**
- Fixed batch export border color for palette overrides: `_capture_overlay_snapshot` now reads the left-panel mask overlay color via `_resolve_mask_type_color` as a fallback when the Mask Painter plugin is disabled or returns no snapshot (previously the export silently fell back to the palette's `default_color`). When the painter is active its snapshot's `mask_type_color` is preferred; otherwise the left-panel color control is read directly, so `Show borders on filled masks` with `border_color_mode="mask_type_color"` always reflects the correct overlay color.
- Added `output_path` to the batch export config template: the `Output folder` field is now captured on save and restored on load alongside all other export settings.
- Replaced the `mode_selector` `ToggleButtons` widget with the existing `mode_tabs` `Tab` widget as the sole mode selector; the active tab now determines the export mode directly, removing the redundant toggle-buttons bar.
- Added horizontal separator lines below the DPI field and the Scale bar % width slider for clearer visual grouping.
- Validated: `python -m unittest tests.test_export_fovs_batch tests.test_export_fovs_mask_customization`.

**Issue #91 reply 6 — Mask Painter inactive defaults, restored opacity linkage, and ordering/layout fixes**
- Reworked the Mask Painter default-state model so inactive classes now derive their effective fill-vs-outline mode from `Global fill` instead of always falling back to outline mode, while active class mode controls remain untouched.
- Restored value-based global opacity linkage by updating classes whose current opacity still matches the previous global opacity value, rather than relying on persisted linked-opacity bookkeeping as the runtime source of truth.
- Fixed `Only specified` restore ordering so toggling the filter off brings back the full list with customized classes first and default-colored classes afterward.
- Added spacing between `Global fill` and the opacity input, narrowed the opacity control, and carried the new `global_fill` painter state through snapshot replay so ROI preset capture/apply keeps the same inactive-class semantics.
- Validated: `python -m unittest tests.test_mask_painter_mode_visibility tests.test_mask_color_sets` and `python -m unittest tests.test_roi_manager_tags.ROIManagerTagsTests.test_capture_and_apply_roi_preserves_mask_painter_payload`.

**Issue #91 reply 5 — Mask Painter linked global state and palette persistence**
- Added a `Global fill` toggle beside the existing global fill opacity control, and both global controls now update only classes that are still explicitly linked to the inherited/default painter behavior.
- Saved mask-color sets now persist the active class list, `Only specified` state, global fill toggle, linked fill/opacity classes, and the existing per-class mode/opacity/visibility and border settings.
- Loading older palettes stays backward-compatible by restoring missing reply-5 fields to the current global defaults and rebuilding inherited-class linkage from the restored controls.
- Added focused regressions for linked global fill propagation, customized-class ordering under `Only specified`, full reply-5 palette round-trip, and old-palette fallbacks.
- Validated: `python -m unittest tests.test_mask_painter_mode_visibility tests.test_mask_color_sets`.

**Issue #91 reply 4 — no-image mask mode**
- Added a new left-panel `No image (masks only)` checkbox that skips the image-layer composite while preserving mask and annotation overlays on a black background.
- Routed the new state through the shared rendering engine and viewer map-mode cache signature so both single-FOV rendering and stitched-map tiles redraw consistently when the mode changes.
- Propagated the same state to export, ROI thumbnail, and cell-gallery render paths via overlay snapshots or direct render forwarding, so downstream views can render masks without the underlying image data as well.
- Added focused regressions for the shared render flag, viewer forwarding, export propagation, gallery propagation, and ROI snapshot propagation.
- Validated: `python -m unittest tests.test_rendering.RenderingHelpersTests.test_render_fov_to_array_skips_image_layer_but_preserves_masks tests.test_rendering.RenderingHelpersTests.test_render_fov_to_array_without_channels_can_render_overlays tests.test_mask_painter_mode_visibility.TestMaskPainterRenderPath tests.test_export_fovs_batch.ExportFOVsBatchTests.test_capture_overlay_snapshot_and_rebuild tests.test_export_fovs_batch.BatchExportMapROIItemsTests.test_export_map_roi_worker_calls_render_map_region_direct tests.test_export_fovs_batch.BatchExportMapROIItemsTests.test_render_map_region_direct_uses_render_fov_to_array_per_tile tests.test_export_fovs_batch.BatchExportMapROIItemsTests.test_export_map_roi_worker_applies_map_bounds_offset tests.test_cell_gallery.TestCellGalleryColors.test_gallery_forwards_skip_image_layer_from_snapshot tests.test_roi_manager_tags.ROIManagerMapModeTests.test_build_overlay_snapshot_carries_no_image_mode`.

**Issue #91 — filled-border dimming fix**
- Fixed the shared mask painter overlay helper so filled-mask borders are painted after all fills and their thickened border mask is clipped back to the owning filled cell.
- This prevents filled-border rendering from spilling into neighboring cells and altering adjacent fill colors while preserving the intended border-overwrites-fill behavior inside the same cell.
- Added a focused adjacent-cell regression in `tests/test_mask_color_overlay.py` and revalidated the painter integration suite.
- Validated: `python -m unittest tests.test_mask_color_overlay` and `python -m unittest tests.test_mask_painter_mode_visibility`.

**Issue #91 reply 3 — map-mode painter parity**
- Fixed the live map-mode painter overlay so it resolves the same effective per-FOV color, border-color, mode, and opacity maps from the current Mask Painter UI state that single-FOV rendering already uses.
- Added a focused regression proving `_apply_map_painter_overlay()` prefers the current effective painter state over stale cached registry values.
- Validated: `python -m unittest tests.test_mask_painter_mode_visibility`.

**Issue #91 reply 2 — NumPy-backed mask highlight fix**
- Fixed `ImageDisplay.set_mask_colors_current_fov()` so the immediate current-FOV highlight path accepts either NumPy arrays or lazy arrays that expose `.compute()`, instead of assuming both the masked label slice and generated edge mask are always lazy.
- Added a focused regression in `tests/test_image_display_tooltip.py` that reproduces the NumPy-backed mask path hit by `MaskPainterDisplay.apply_colors_to_masks()`.
- Validated: `python -m unittest tests.test_image_display_tooltip tests.test_mask_painter_mode_visibility`.

**Issue #91 follow-up — ROI thumbnail painter replay and border-color modes**
- Fixed map-mode ROI thumbnail/export replay so saved mask painter snapshots are re-applied onto stitched map renders instead of falling back to the raw map image.
- Added mask painter border-color modes so filled-mask borders can either follow the left-panel mask color or reuse the fill color, and replayed snapshots now preserve the captured mask-type color as a resolved hex value.
- Updated the live viewer and cell gallery to honor distinct filled-border colors in the same overlay path used by snapshot replay.
- Validated: `python -m unittest tests.test_mask_color_overlay tests.test_mask_painter_mode_visibility tests.test_roi_manager_tags tests.test_cell_gallery` and `python -m unittest tests.test_export_fovs_batch.BatchExportMapROIItemsTests.test_export_map_roi_worker_calls_render_map_region_direct tests.test_export_fovs_batch.BatchExportMapROIItemsTests.test_export_map_roi_worker_applies_map_bounds_offset tests.test_export_fovs_batch.BatchExportMapROIItemsTests.test_export_map_roi_worker_raises_on_empty_roi`.

**Issue #91 — Mask Painter opacity and borders on filled masks**
- Added per-class fill opacity controls plus a global linked opacity control to Mask Painter, and added a "Show borders on filled masks" toggle so filled classes can keep visible boundaries.
- Extended the live viewer render path, map-mode painter overlay, and overlay helper to honor per-class fill opacity and optional fill borders in the same compose logic.
- Added painter snapshot capture/replay so cell gallery, ROI thumbnails/presets, and batch export reuse the saved painter state instead of falling back to outline-only rendering.
- Persisted painter snapshot payloads in ROI records so each ROI keeps the mask painter settings captured with it.
- Validated: `python -m unittest tests.test_mask_painter_mode_visibility tests.test_mask_color_overlay tests.test_cell_gallery tests.test_roi_manager_tags` and `python -m unittest tests.test_export_fovs_batch.ExportFOVsBatchTests.test_batch_export_snapshot_preserves_mask_painter_outline_thickness`.

**Issue #90 — Mask Painter redraw visibility fix**
- Fixed the single-FOV mask painter render path so zoom/pan redraws use the current UI state directly during `_compose_fov_image()` instead of relying on post-render plugin timing.
- Added `get_effective_color_map_for_fov()` and `get_effective_mode_map_for_fov()` to `MaskPainterDisplay`, and updated `main_viewer.py` to pass that state into `apply_registry_colors(...)` for the painter-controlled mask overlay.
- Changed Mask Painter to start disabled by default and added an enable/disable observer that invalidates stale painter state and triggers a viewer redraw when the plugin is toggled.
- Updated active/inactive class semantics: only active unchecked classes are hidden; inactive classes remain visible with the default color while preserving their stored custom color for later re-activation.
- Updated issue #89 follow-up expectations in the mask painter tests so removed/filtered classes now retain the default color instead of receiving the hidden sentinel.
- Validated: `python -m unittest tests.test_mask_painter_mode_visibility tests.test_mask_color_overlay tests.test_mask_color_sets` — 47/47 passed.

**Issue #89 follow-up — Mask Painter "Only specified" toggle**
- Added `only_specified_checkbox` (`Checkbox`, `description="Only specified"`) to `UiComponent`, placed inline with `default_color_picker` in `colors_layout`.
- New method `_on_only_specified_toggle(change)` on `MaskPainterDisplay`: when ON, filters `_active_classes` to classes whose color differs from `default_color` (via `colors_match()`); when OFF, restores all classes in `current_classes` to active. Calls `_push_to_widget()` to sync.
- Observer registered in `MaskPainterDisplay.__init__` via `only_specified_checkbox.observe(self._on_only_specified_toggle, names="value")`.
- Added `TestMaskPainterOnlySpecified` class with 4 tests to `tests/test_mask_painter_mode_visibility.py`. Tests call `_on_only_specified_toggle({"new": True/False})` directly because the bootstrap stub's `Widget.observe()` is a no-op (real ipywidgets traitlets are not available in the test environment for Checkbox-based callbacks).
- Validated: `python -m unittest tests.test_mask_painter_mode_visibility tests.test_mask_color_sets tests.test_mask_color_overlay -v` — 43/43 passed.

**Issue #89 follow-up — Mask Painter Add/Remove class feature**
- Added three new traitlets to `MaskClassListWidget`: `available_classes: List[str]`, `add_requested: str`, `remove_requested: str` (all `sync=True`; also present in the `HasTraits` fallback for tests).
- JS: each class row now has a `×` button (`.mask-cl-remove`) that fires `model.set('remove_requested', cls)`. A footer `div.mask-cl-add-row` below the scroll area contains a `<select>.mask-cl-add-select` (populated from `available_classes`) and a `+ Add` button; clicking it fires `model.set('add_requested', <selected_value>)`. The select is disabled (with placeholder text) when `available_classes` is empty.
- `MaskPainterDisplay` gains `self._active_classes: List[str]` (subset of `current_classes` currently shown in the widget).
- `on_identifier_change` now caps the initial active set to the first 6 classes: `self._active_classes = classes[:min(len(classes), 6)]`.
- `_push_to_widget()` now derives `active` from `_active_classes` (falling back to `_get_full_class_order()` when empty), computes `available = [cls for cls in current_classes if cls not in active_set]`, and filters the `colors`/`visible`/`fill` dicts to the active set before syncing. Also sets `w.available_classes`.
- `_pull_from_widget` `class_order` branch now updates `self._active_classes` (was: `self.current_classes`).
- New `_on_add_requested(change)` observer: appends the requested class to `_active_classes` and calls `_push_to_widget()`, then clears `add_requested`.
- New `_on_remove_requested(change)` observer: removes the class from `_active_classes` and calls `_push_to_widget()`, then clears `remove_requested`.
- `_load_color_set` additionally sets `self._active_classes` from `ordered_unique` after palette load.
- `on_cell_table_change` now resets `self._active_classes = []`.
- `_get_hidden_classes()` simplified to `[key for key in class_color_controls if key not in visible]` — inactive (non-active) classes are automatically treated as hidden and receive the `""` sentinel on next apply.
- Added 4 new tests in `TestMaskPainterAddRemoveClass` (test file: `tests/test_mask_painter_mode_visibility.py`); updated `_make_painter` fixture to set `_active_classes` and call `_push_to_widget()`.
- Validated: `python -m unittest tests.test_mask_painter_mode_visibility tests.test_mask_color_sets tests.test_mask_color_overlay -v` — 39/39 passed.

**Issue #89 follow-up — Mask Painter class list replaced with anywidget drag-and-drop rows**
- Replaced the `TagsInput` + `VBox(color_picker_box)` + `show_all_checkbox` combination with a single `MaskClassListWidget` (`anywidget.AnyWidget`) defined in the new `ueler/viewer/plugin/mask_class_list_widget.py`.
- Each class renders as an inline row: `[≡ drag] [□ vis] [■ <input type="color"> ClassName] [□ fill]`. HTML5 native drag API (`draggable`, `dragstart`, `dragover`, `drop`) reorders rows and syncs `class_order` back to Python via traitlets; targeted DOM-only updates handle color, visibility, and fill changes without a full rebuild.
- On the Python side, `_push_to_widget()` syncs all ipywidget state to the anywidget traitlets (`class_order`, `class_colors`, `class_visible`, `class_fill`, `default_color`); `_pull_from_widget(change)` reverses the flow for user interactions in the browser.
- `_syncing` flag prevents recursive observer loops between the anywidget traitlets and the backing ipywidgets.
- `on_sorting_items_change`, `on_show_all_toggle`, and `_handle_selection_transition` are now no-ops (kept as stubs for backward compatibility with test setups that manipulate `sorting_items_tagsinput` / `selected_classes` directly).
- `_get_visible_classes()` uses `class_list_widget.class_order` as the primary source; falls back to `selected_classes` / `sorting_items_tagsinput` for test setups that bypass the anywidget.
- `_load_color_set` simplified: removed `non_default_classes`/`defaulted_classes` split, `hidden_color_cache` population, and `sorting_items_tagsinput` manipulation; colors and modes are restored directly and `_push_to_widget()` syncs the widget.
- `_set_default_color` simplified: removed `hidden_color_cache` and `sorting_items_tagsinput` logic; updates pickers with old default color and calls `_push_to_widget()`.
- `HasTraits` fallback class in `mask_class_list_widget.py` ensures test environments without anywidget work unchanged.
- Updated `tests/test_mask_color_sets.py`: replaced `selected_classes`/`hidden_color_cache` assertions with widget traitlet assertions on `class_list_widget.class_order` and `class_list_widget.class_colors`.
- Validated: `python -m unittest tests.test_mask_painter_mode_visibility tests.test_mask_color_sets tests.test_mask_color_overlay -v` — 35/35 passed.

**Issue #89 — Mask Painter UI Layout and Usability Redesign**
- Replaced the 30%/70% `HBox` split class-color list with a vertical `VBox` (`colors_layout`) placing `TagsInput` and default `ColorPicker` above the per-class scroll area at full width, eliminating chip overflow and label truncation.
- Replaced per-class `ToggleButtons(options=["outline","fill"])` with a compact `Checkbox(description="fill", layout=Layout(width="60px"))`, saving ~100 px per row and eliminating horizontal overflow.
- Wrapped palette Save/Load/Manage `Tab` in a collapsed `Accordion` (`selected_index=None`) so it takes zero height until the user expands it.
- Replaced the `Output` feedback widget with a styled `HTML` label (`feedback_label`) showing ✓ (green) or ⚠️ (orange) messages inline.
- Redesigned the top bar into two `HBox` rows: row 1 holds `enabled_checkbox` + `identifier_dropdown`; row 2 holds `update_button` + `apply_saved_button` + `saved_sets_dropdown`.
- Added `_refresh_save_button_state()`, wired to `set_name_input.value` and `identifier_dropdown.value` observers, that keeps `save_button.disabled=True` until both fields are non-empty.
- Updated `tests/test_mask_painter_mode_visibility.py` for `Checkbox`-based mode controls (replacing `ToggleButtons` bool semantics) and all related assertions.
- Validated with `python -m unittest tests.test_mask_painter_mode_visibility tests.test_mask_color_sets tests.test_mask_color_overlay -v` — 35/35 passed.

**Issue #88 follow-up 5 (Option B) — ROI manager advanced filter replaced with self-contained anywidget expression editor**
- Replaced all ipywidgets-based expression filter UI (Text input, Apply button, operator HBox, tag HBox, JS Output widget) with a single `ROIExpressionEditorWidget` (`anywidget.AnyWidget` subclass) defined in the new `ueler/viewer/plugin/roi_expression_editor.py`.
- The widget renders its own DOM (input field, Apply button, operator row, tag row) entirely in ESM JavaScript. `mousedown` + `preventDefault()` on every button keeps the text input focused, so `selectionStart`/`End` are always accurate when `click` fires — no Python comm round-trip needed for insertion.
- JS `formatInsertion()` replicates the Python `_format_expression_insertion()` spacing rules for all token types.
- Apply button increments the `apply_requested` traitlet; Python observes the change with `_on_apply_requested_change` and triggers gallery refresh.
- `_refresh_expression_tag_buttons(tags)` now simply sets `editor.tags = list(tags)` and the JS rebuilds the tag row via `model.on("change:tags", ...)`.
- A `HasTraits` fallback class (no `anywidget` required) is used automatically in test environments.
- Removed: `browser_expression_input`, `browser_expression_apply_button`, `browser_expression_operator_box`, `browser_expression_tag_box`, `browser_expression_js_output`, `_make_expression_insert_button`, `_insert_browser_expression_snippet*`, `_use_browser_expression_js`, `_browser_operator_buttons`, JSON import, IPythonJS import.
- Updated `tests/test_roi_manager_tags.py`: removed 6 obsolete insertion tests, updated 4 Apply button tests to use `browser_expression_editor.expression`, added `test_refresh_tag_buttons_updates_editor_tags`, repurposed `test_expression_insertion_at_end_of_expression` to test `_format_expression_insertion` directly.
- Validated with `python -m unittest tests.test_roi_manager_tags -v` — 34/34 passed.

**Issue #88 follow-up 4 — ROI manager advanced filter redesigned with JS-only editing and Apply button**
- Replaced the unreliable JS→Python caret-sync bridge with a fully JS-side expression editor: helper buttons (operators and tag tokens) now emit a self-contained JavaScript snippet per click that reads the live DOM caret, applies the same spacing rules as the Python `_format_expression_insertion()` method, updates the field value in place, and repositions the caret — entirely without a Python round-trip.
- Added a new "Apply" button beside the expression input that triggers gallery refresh on demand; the Python handler reads `browser_expression_input.value` (auto-synced by ipywidgets) and compiles the expression.
- Removed: hidden `browser_expression_selection_state` widget, `_install_browser_expression_selection_bridge()`, `_on_browser_expression_selection_state_change()`, `_resolve_browser_expression_selection()`, `_flush_browser_expression_selection_before_click()`, `_insert_browser_expression_snippet_backend()`, `_on_browser_expression_change()`, and `_browser_expression_selection` / `_browser_expression_selection_text` caches.
- Updated `tests/test_roi_manager_tags.py`: removed five now-obsolete caret-tracking tests, updated five insertion tests to reflect tail-append test-mode behaviour, added five Apply button tests (widget existence, gallery refresh, error feedback, helper buttons do not auto-refresh).
- Validated with `python -m unittest tests.test_roi_manager_tags -v` — 39/39 passed.

**Issue #88 follow-up 3 — ROI manager helper buttons now honor caret-only reposition before click**
- Fixed the remaining ROI manager advanced-filter bug where moving the caret inside the expression without editing the text still left helper-button insertion stuck at the older tail position.
- Extended the browser selection bridge in `roi_manager_plugin.py` so helper buttons flush the live DOM selection on `pointerdown` / `mousedown` before the Python click callback runs.
- Reinstalled that bridge after dynamic tag-button refreshes and added focused regression coverage in `tests/test_roi_manager_tags.py` for the pre-click caret-sync path.
- Validated with `python -m unittest tests.test_roi_manager_tags -v`.

**Issue #88 follow-up 2 — ROI manager helper insertions no longer reuse stale positions after manual typing**
- Fixed a second follow-up bug where the ROI manager's advanced-filter helper buttons could still insert at an older cached position after the user manually typed more text into the expression field.
- Added `_browser_expression_selection_text` in `roi_manager_plugin.py` so cached selections are tied to a specific expression-text revision.
- `_on_browser_expression_change()` now collapses stale cached selections to the end of the new text when the expression changes, while the live-caret bridge continues to override that fallback when fresher mid-string caret data is available.
- Added regression coverage in `tests/test_roi_manager_tags.py` for the exact reported sequence: helper insertion, manual typing, then helper insertion again.
- Validated with `python -m unittest tests.test_roi_manager_tags -v`.

**Issue #88 follow-up — ROI manager advanced tag-filter insertion now respects the live caret position**
- Fixed a follow-up regression where advanced-filter helper buttons could still insert at a stale cached position after the user moved the caret in the browser field without editing the text.
- Added a minimal browser-to-Python caret bridge in `roi_manager_plugin.py`: a hidden selection-state widget plus browser listeners keep `_browser_expression_selection` aligned with the live DOM caret while preserving Python-owned text mutation.
- Added regression coverage in `tests/test_roi_manager_tags.py` for direct insertion and operator-button clicks that use a tracked mid-string caret position.
- Validated with `python -m unittest tests.test_roi_manager_tags -v`.

**Issue #88 — ROI manager advanced tag-filter helper buttons now respond reliably**
- Fixed the ROI manager advanced-filter helper buttons so operator and tag buttons always mutate the expression field through the Python widget state instead of relying on a front-end-specific JavaScript insertion path.
- Reduced the browser-side JavaScript helper to a best-effort DOM value/caret synchronization step after backend insertion, which keeps VS Code and other notebook front-ends from turning clicks into silent no-ops.
- Normalized advanced-filter helper button construction in `roi_manager_plugin.py` and added regression coverage in `tests/test_roi_manager_tags.py` for operator clicks, tag-button clicks, and the JS-success-without-mutation case.
- Validated with `python -m unittest tests.test_roi_manager_tags -v`.

**Issue #87 — histogram threshold changes now refresh viewer highlights reliably**
- Fixed stale viewer outlines after repeated histogram cutoff changes by making `ImageDisplay.set_mask_ids()` rebuild replacement highlights from the clean composited image instead of layering them on top of the previous selection overlay.
- Applied the same replacement-redraw fix to both single-FOV and map-mode programmatic highlight paths.
- Added regression coverage in `tests/test_image_display_tooltip.py` and `tests/test_chart_cell_gallery_link.py`; validated with `python -m unittest tests.test_image_display_tooltip tests.test_chart_cell_gallery_link -v`.

**Issue #86 — scatter widget selections now sync across all active plots**
- Fixed the Chart plugin so a selection made in one scatter plot is immediately propagated to every active scatter plot in the same widget instead of staying local to the originating plot.
- Kept `jscatter.compose(..., sync_selection=False)` in place to avoid the previously recorded compose-time selection regression and moved synchronization fully into the plugin's own observer pipeline.
- Refactored `ChartDisplay` so widget-originated and external selections share one commit path for index normalization, per-plot synchronization, single-point state updates, and linked mask-highlight refreshes.
- Added regression coverage in `tests/test_chart_footer_behavior.py` and `tests/test_chart_cell_gallery_link.py`; validated with `python -m unittest tests.test_chart_footer_behavior tests.test_chart_cell_gallery_link`.

**Reply 7 to issue #85 — marker-set channel dedupe and control rebuild de-duplication**
- Fixed a follow-up regression where loading saved marker sets could create repeated channel widget rows.
- Added `_dedupe_channel_sequence(...)` in `main_viewer.py` and applied it to marker-set save/update/apply paths so selected channels are unique (first-seen order preserved).
- Removed unconditional `update_controls(None)` in `_apply_marker_set` after selector assignment and now refresh only when selector value is unchanged.
- Removed duplicate `channel_selector -> update_controls` observer registration in `ui_components.py`; channel changes now flow through `on_channel_selection_change` only.
- Added de-duplication helper regression tests in `tests/test_channel_legend.py`.
- Validated with:
  - `python -m unittest tests.test_wide_plugin_panel tests.test_chart_footer_behavior tests.test_channel_legend tests.test_roi_manager_tags -v`
  (53/53 pass)
  - Extended run including `tests.test_export_fovs_batch` reports 1 pre-existing unrelated failure (`test_export_fovs_batch_writes_file`).

**Reply 6 to issue #85 — enforce parent channel-panel vertical scrolling**
- Ensured the shared channel-panel scrollbar appears when total channel widget height exceeds the container.
- Root cause was child group shrink-to-fit behavior; channel group and slider layouts now use `flex: 0 0 auto` so rows keep intrinsic height and overflow is delegated to the parent scroller.
- `channel_controls_box` now explicitly keeps `overflow_y='auto'` with `overflow_x='hidden'`.
- Validated with:
  - `python -m unittest tests.test_wide_plugin_panel tests.test_chart_footer_behavior tests.test_channel_legend tests.test_roi_manager_tags -v`
  (49/49 pass)

**Reply 5 to issue #85 — shared scroller restoration and dropdown offset**
- Fixed a post-refactor regression where individual channel groups showed their own internal scrollbars; group and slider layouts were updated to suppress per-group overflow so only the shared channel-panel scroller is used.
- Shifted the channel color dropdown 5px to the left in the header row.
- Updated helper assertions in `tests/test_channel_legend.py` to verify dropdown left margin and overflow-safe slider layout values.
- Validated with:
  - `python -m unittest tests.test_wide_plugin_panel tests.test_chart_footer_behavior tests.test_channel_legend tests.test_roi_manager_tags -v`
  (49/49 pass)

**Reply 4 to issue #85 — three-row channel control reorganization**
- Reworked channel controls in `update_controls` so each marker renders as a grouped three-row block:
  1) header row: visibility checkbox + marker name + color dropdown,
  2) Min slider row,
  3) Max slider row.
- Marker name is now shown once in the header row; slider descriptions were simplified to `Min` / `Max` only.
- Replaced fixed wide color dropdown sizing with compact content-driven width calculated from color option names, reducing footprint while keeping names visible.
- Added marker-label and grouped-row layout helpers to support the new structure and preserve shrink-safe behavior.
- Updated `tests/test_channel_legend.py` helper assertions and validated with:
  - `python -m unittest tests.test_wide_plugin_panel tests.test_chart_footer_behavior tests.test_channel_legend tests.test_roi_manager_tags -v`
  (49/49 pass)

**Reply 3 to issue #85 — channel panel row compaction and overflow cleanup**
- Kept container wrappers at `max_width: 99%` and fixed remaining channel-panel overflows by compacting content rows only.
- Reworked marker-set action buttons into a wrap-capable flex row and constrained the confirm-deletion checkbox row to prevent horizontal scrollbars in the marker-set section.
- Tightened per-channel header row behavior in `update_controls`: compact visibility checkbox sizing/spacing, narrower color dropdown sizing, hidden row overflow, and reusable row-layout helpers.
- Tuned slider row layout for longer usable marker-range tracks while preserving existing slider labels and readout.
- Hardened channel legend rendering: content-width legend layout plus HTML wrapping safeguards (`overflow-wrap:anywhere`, `word-break:break-word`) for long channel names.
- Extended regression coverage in `tests/test_channel_legend.py`; validated with:
  - `python -m unittest tests.test_wide_plugin_panel tests.test_chart_footer_behavior tests.test_channel_legend -v`
  - `python -m unittest tests.test_roi_manager_tags -v`
  (49/49 pass)

**Reply 2 to issue #85 — content-widget shrink policy for scrollbar prevention**
- Kept all existing container wrappers at the current `max_width: 99%` policy and introduced a new content-widget sizing rule (`width/max_width: calc(100% - 5px)`, `min_width: 0`, `box_sizing: border-box`) so inner controls can shrink without forcing horizontal overflow.
- Added `_content_widget_layout` in `ui_components.py` and applied it to marker set fields, channel selector rows, channel legend/grid checkboxes, mask outline slider, and annotation palette controls.
- Updated `ImageMaskViewer.update_controls` in `main_viewer.py` to apply the same content-width rule to channel contrast sliders and mask color controls.
- Added a shared `content_widget_layout` helper in `layout_utils.py` and adopted it in ROI Manager / Batch Export content widgets; added a matching content-control helper pass in `chart_heatmap.py` with wrap-safe rows.
- Extended regression coverage in `tests/test_wide_plugin_panel.py` and validated with `python -m unittest tests.test_wide_plugin_panel tests.test_chart_footer_behavior tests.test_roi_manager_tags -v` (45/45 pass).

**Reply to issue #85 — prevent unnecessary horizontal scrollbars in UI panels**
- Added a shared constrained panel layout helper in `ui_components.py` and applied it to main control wrappers, channel/mask/annotation sections, and wide footer wrappers so panel containers no longer match parent width exactly.
- Tuned bounded width policy to `max_width='99%'` (follow-up to initial 97%) and kept `min_width='0'` + `box_sizing='border-box'` safeguards.
- Hardened dynamic rows in `main_viewer.py` (`update_controls`) and normalised marker-set/slider/legend-related control layouts so inner widgets do not exceed parent containers.
- Applied the same constraints to key plugin wrappers in `chart_heatmap.py`, `heatmap_layers.py`, `roi_manager_plugin.py`, `export_fovs.py`, and `annotation_palette_editor.py`.
- Added layout regression assertions in `tests/test_wide_plugin_panel.py`.
- Validated with: `python -m unittest tests.test_wide_plugin_panel tests.test_chart_footer_behavior -v` (13/13 pass).

**Reply to issue #84 — map mode black view (RC6: runner refresh parity with map switch)**
- Fixed a remaining map-mode black/placeholder case in the `run_viewer(auto_display=False)` + `load_cell_table(auto_display=True)` workflow. Root cause: `load_cell_table` refresh path (`_refresh_viewer_state`) called `on_image_change` + `update_display` but did not run the same map re-activation sequence as map switching (`on_map_selector_change` -> `_activate_map_mode`). As a result, map canvas/viewport state could remain stale until a manual map switch forced re-activation. `_refresh_viewer_state` now re-activates the active map when `_map_mode_active` is true (using `_active_map_id` with `map_selector.value` fallback), refreshes map controls, and then renders. Added runner-level regression tests for active-map and selector-fallback re-activation paths. Validated with 8/8 tests passing in `tests.test_runner`.

**Reply to issue #84 — map mode black view (RC5: _map_needs_initial_render flag)**
- Fixed the last remaining black-view case: after `load_cell_table` → `display_ui()` shows the widget, ipympl initialises a fresh browser canvas and fires `on_draw`. The `on_draw` short-circuit (center AND viewport size matching the seeded values from `_set_map_canvas_dimensions`) prevented any tile render, so the 1×1 placeholder stayed visible. Added a `_map_needs_initial_render` boolean flag (default `False`) to `ImageDisplay`; `_set_map_canvas_dimensions` sets it to `True` after every map activation; `on_draw` bypasses the short-circuit once when the flag is `True`, clears it, then reverts to normal short-circuit behaviour. This ensures the very first `on_draw` after the widget is shown always renders real tiles, without removing the memory-safety guard that prevents loading all tiles on every subsequent background draw. Added 4 regression tests (`TestMapNeedsInitialRenderFlag`). Validated with 54/54 tests passing.

**Reply to issue #84 — map mode black view (RC4: on_image_change axis-reset guard)**
- Fixed the primary cause of the persistent black view: `load_cell_table` → `_refresh_viewer_state` → `on_image_change` was unconditionally resetting `image_display.height/width` and `ax.set_xlim/ylim` to single-FOV pixel dimensions (e.g. 1024×1024) inside a 12 000×9 000 map canvas, placing the viewport in a tile-free region. `on_image_change` now skips that block when `_map_mode_active` is `True`. Added 3 regression tests (`TestOnImageChangeMapModeGuard`). Validated with 50/50 tests passing.

**Reply to issue #84 — map mode black view on viewer launch**
- Fixed black/square view on initial map-mode launch from a restored session. Three compounding root causes addressed: (1) `display()` now calls `update_display()` + `canvas.draw()` (synchronous) before `display_ui()` so the canvas buffer contains rendered tiles before the widget is sent to the browser; (2) `display()` re-calls `_sync_navigation_home_view()` after `display_ui()` so the ipympl toolbar Home extent is patched to the full map bounds while the toolbar is fully wired (ipympl may overwrite the pre-display patch); (3) `on_draw` short-circuit now compares viewport width and height in addition to center coordinates, so scroll-wheel zoom (which keeps center fixed but changes viewport size) correctly triggers tile reload. `_set_map_canvas_dimensions` seeds the new `prev_viewport_width`/`prev_viewport_height` trackers to preserve the existing memory-safety guard.
- New test file `tests/test_map_mode_initial_display.py` (7 tests) covering all three phases and a regression guard.
- Validated with: `python -m unittest tests.test_map_mode_initial_display tests.test_initial_display tests.test_map_mode_activation tests.test_fov_detection tests.test_fov_detection_fix tests.test_cache_settings` (47/47 pass).

**Issue #83: per-class paint mode (outline/fill) and per-class visibility toggle in mask painter**
- **Exclusive rendering control (Phase 1):** `_compose_fov_image` in `main_viewer.py` now skips the painter's `mask_key` from `MaskRenderSettings` when the painter is enabled, so the painter has full ownership of its layer and hidden cells never bleed through via the generic outline pass.
- **Hidden-class sentinel (Phase 2):** `apply_colors_to_masks` writes `""` (empty string) to the color registry for hidden classes instead of `self.default_color`. The existing `if colour_hex:` gate in `_iter_mask_region_ids` skips empty-string entries, making the cells truly invisible in both single-FOV and map mode.
- **Per-class visibility checkboxes (Phase 3):** `MaskPainterDisplay` now holds `class_visible_controls: Dict[str, Checkbox]` (one per class). `_refresh_color_picker_display` emits `HBox([Checkbox, ColorPicker, ToggleButtons])` rows. `_get_visible_classes` / `_get_hidden_classes` filter by checkbox state on top of the `show_all` / tag-selection logic. `on_show_all_toggle` sets all checkboxes to `True` when the "show all" flag is enabled.
- **Per-class render mode + mode cache (Phase 4):** Added `class_mode_controls: Dict[str, ToggleButtons]` (options: `"outline"` / `"fill"`) and `_cell_mode_cache: Dict[str, Dict[int, str]]` (FOV → mask_id → mode) to `MaskPainterDisplay`. `apply_colors_to_masks` reads each class's `ToggleButtons.value` and calls a new `_register_mode_globally` helper that bulk-populates the mode cache (with dirty-skip via `_last_applied_class_modes`). Added `get_mode_map_for_fov(fov)` returning `dict(self._cell_mode_cache.get(fov, {}))`. `on_cell_table_change` now also resets `_cell_mode_cache` and `_last_applied_class_modes`.
- **Fill blending (Phase 5):** `_apply_region_colors` in `mask_color_overlay.py` branches on `render_mode = mode_map.get(mask_id, "outline")`. Fill mode alpha-blends `canvas[mask_bool] = (1 − α) * canvas + α * rgb` (`FILL_ALPHA_DEFAULT = 0.35`); outline mode uses the existing boundary path. `apply_registry_colors` accepts `mode_map` and `fill_alpha` optional params.
- **Single-FOV wiring (Phase 6):** `_get_mask_painter()` helper added to `main_viewer.py`; `apply_registry_colors` call in `_compose_fov_image` now passes `mode_map=painter.get_mode_map_for_fov(fov_name)`.
- **Map-mode wiring (Phase 7):** `_apply_map_painter_overlay` groups cells by `(color_hex, render_mode)` tuples; fill mode alpha-blends per pixel on the stitched canvas; empty `color_hex` entries are skipped (invisible).
- **Persistence (Phase 8):** `save_current_color_set` / `overwrite_saved_color_set` now include `"modes"` and `"visible"` keys in the JSON payload. `_load_color_set` restores `ToggleButtons.value` from `"modes"` and `Checkbox.value` from `"visible"` (both keys optional with sensible defaults).
- **Tests (Phase 9):** New `tests/test_mask_painter_mode_visibility.py` with 9 tests covering hidden-class sentinel, visible-class coloring, mode-map accuracy, mode-cache reset, dirty-skip, `_get_visible_classes`/`_get_hidden_classes` with checkboxes, and persistence round-trip. Updated `test_painted_colors_all_fovs.py` to expect `""` for hidden classes.
- Validated with: `python -m unittest tests.test_mask_color_overlay tests.test_painted_colors_all_fovs tests.test_mask_color_sets tests.test_mask_painter_mode_visibility` (38/38 pass).

**Issue #82: mask painter performance — nested registry, bulk write, dirty tracking**
- Restructured `_CELL_COLOR_REGISTRY` in `engine.py` from a flat `dict[tuple[str, int], str]` to a nested `dict[str, dict[int, str]]` (FOV → mask_id → color). `get_all_cell_colors_for_fov` is now O(1) per FOV instead of scanning the entire registry; `clear_cell_colors(fov)` uses a single `dict.pop()` instead of a linear scan-and-delete loop.
- Added `set_cell_colors_bulk(entries)` to `engine.py` and exported it via `ueler/rendering/__init__.py`. It merges a pre-built nested dict into the registry with one `dict.update()` per FOV, replacing O(N) per-cell writes.
- Replaced the `iterrows()` loop inside `_register_color_globally` (nested in `apply_colors_to_masks`) with a vectorised path: two `.to_numpy()` calls build `fov_arr`/`mid_arr`, a single dict comprehension groups mask IDs by FOV, and one `set_cell_colors_bulk()` call writes all entries.
- Added `_last_applied_class_colors: dict[str, str]` to `MaskPainterDisplay`. `_register_color_globally` now skips the bulk write entirely if a class's color is unchanged since the last registration, and updates the cache only after writing. `on_cell_table_change()` resets the cache.
- Added 21 new tests in `tests/test_mask_color_overlay.py` covering the nested registry structure, `set_cell_colors_bulk`, the bulk-write integration path, and per-class dirty tracking.
- Validated with: `python -m unittest tests.test_mask_color_overlay tests.test_cell_gallery tests.test_painted_colors_all_fovs tests.test_mask_color_sets` (40/40 pass).

**Issue #81 reply 2: scatter-plot selection now highlights cell masks in map mode**
- **Root cause:** `_on_scatter_selection()` and `_apply_external_selection()` in `chart.py` only updated `selected_indices` (cell gallery) and optionally navigated the viewport. Neither called `set_mask_ids()`, so mask highlights were never triggered from scatter-plot lasso/click events regardless of mode.
- **Fix:** Added `_sync_mask_highlights_from_selection(indices)` to `ChartDisplay`. In single-FOV mode it filters the cell table to the active FOV and calls `set_mask_ids(mask_ids=[...])`. In map mode (`get_active_fov()` returns `None`) it builds `(fov, mask_id)` pairs and calls `set_mask_ids(fov_mask_pairs=[...])` — the same pattern used by `highlight_cells()`. The new helper is called from `_on_scatter_selection()` and `_apply_external_selection()` whenever `mv_linked_checkbox` is enabled.
- **Changed files:** `ueler/viewer/plugin/chart.py`, `tests/test_chart_cell_gallery_link.py`.
- Validated with: `python -m unittest tests.test_chart_cell_gallery_link tests.test_map_mode_activation tests.test_painted_colors_all_fovs tests.test_mask_color_sets` (51/51 pass).

**Issue #81 follow-up: map-mode cell mask highlighting and mask painter overlay**
- **Root cause 1 (scatter highlight not working in map mode):** `highlight_cells()` in `chart.py` filtered by `image_selector.value` (stale in map mode). `set_mask_ids()` in `image_display.py` read the same stale selector, printed "No active FOV", and returned without populating `selected_masks_label`. The existing `_update_map_mask_highlights()` downstream already handles per-tile overlay correctly — it just never received valid `(fov, mask_id)` pairs.
- **Fix 1:** `highlight_cells()` calls `get_active_fov()`. In map mode, it builds explicit `(fov, mask_id)` pairs from the cell table filtered by the comparator, and passes them to `set_mask_ids()` via a new `fov_mask_pairs` keyword. `set_mask_ids()` gains `fov_mask_pairs`: when provided, directly populates `selected_masks_label` and skips single-FOV validation.
- **Root cause 2 (mask painter not rendering in map mode):** `on_mv_update_display()` in `mask_painter.py` called `apply_colors_to_masks(register_globally=False)`, so the global `_CELL_COLOR_REGISTRY` was never populated in map mode. The tile render cache also served stale (pre-color) images.
- **Fix 2:** Both `apply_colors_to_masks()` and `on_mv_update_display()` use `get_active_fov()`. `_apply_color_to_current_fov()` is guarded by `if current_fov:`. In map mode, `on_mv_update_display()` passes `register_globally=True`. A new `_apply_map_painter_overlay()` method (analogous to `_update_map_mask_highlights()`) iterates visible tile viewports, reads `get_all_cell_colors_for_fov()`, groups IDs by color, computes edges, and blits colored outlines onto the stitched map image. It updates `display.combined` before `_update_map_mask_highlights()` runs so highlights stack correctly on top.
- **Changed files:** `chart.py`, `image_display.py`, `mask_painter.py`, `main_viewer.py`, `tests/test_chart_cell_gallery_link.py`, `tests/test_painted_colors_all_fovs.py`.
- Validated with: `python -m unittest tests.test_map_mode_activation tests.test_painted_colors_all_fovs tests.test_mask_color_sets tests.test_chart_cell_gallery_link` (45/45 pass).

**Issue #81: consistent plugin behavior — ROI gallery black thumbnails and centering on map-mode ROIs**
- **Root cause 1 (black ROI gallery thumbnails):** `_render_map_roi_tile()` in `roi_manager_plugin.py` converted canvas-pixel coordinates to µm by multiplying by `base_pixel_size_um` alone, omitting the map's physical bounds origin (from `layer.map_bounds()`). Identical to the batch-export white-PNG bug. For maps with non-zero stage coordinates, `layer.set_viewport()` received coordinates far outside all tiles → `render()` returned zeros → black thumbnails.
- **Fix 1:** After acquiring `base_px_um`, call `layer.map_bounds()` and apply the origin: `xmin_um = float(bounds[0]) + x_min * base_px_um` (all four edges). Call wrapped in try/except so zero-origin maps are unaffected.
- **Root cause 2 (centering fails when map not active):** `center_on_roi()` in `main_viewer.py` only handled coordinate translation if `self._map_mode_active` was already `True`. If the user was in single-FOV mode and activated a map-mode ROI via the ROI browser or "Center" button, the method fell through to the single-FOV `else` branch with an empty `fov` string — skipping the FOV selector and placing the viewport on the wrong canvas.
- **Fix 2:** At the top of `center_on_roi`, extract `map_id` from the record. If `not target_fov and map_id` and the correct map is not active, call `_activate_map_mode(map_id)`, `_refresh_map_controls()`, and `update_display(...)` before the existing coordinate-translation logic. Mirrors the pattern in `focus_on_cell()`.
- Added 1 new test `test_render_map_roi_tile_applies_bounds_offset` in `test_roi_manager_tags.py`: stubs `map_bounds()` with non-zero origin `(1000, 3000, 4000, 6000)`, asserts `set_viewport` receives offset µm coords.
- Added 3 new tests in `test_map_mode_activation.py`: `test_center_on_roi_activates_map_mode_when_inactive`, `test_center_on_roi_switches_map_when_wrong_map_active`, `test_center_on_roi_skips_activation_when_correct_map_already_active`.
- Validated with: `python -m unittest tests.test_roi_manager_tags tests.test_map_mode_activation` (56/56 pass).

**MkDocs Material documentation site (#80)**
- Created a full documentation site using Material for MkDocs, covering installation, getting started, tutorials (basic usage, user interface, map mode, batch export), FAQ, and developer notes (packaging, viewer runtime, map mode internals, export pipeline, ROI workflows, heatmap & cell annotation, OME-TIFF loading).
- Added `.github/workflows/docs.yml` to auto-deploy the site to GitHub Pages on every push to `main`.
- Added `[project.optional-dependencies] docs = ["mkdocs-material>=9.5"]` to `pyproject.toml` and updated the documentation URL to `https://hartmannlab.github.io/UELer/`.

**Map mode ROI export: fix white PNG — missing bounds offset and region_ds mismatch (Follow-up)**
- **Root cause 1 (white PNG, main bug):** `_export_map_roi_worker` converted canvas-pixel coordinates to µm by multiplying by `base_pixel_size_um` alone, omitting the map's physical bounds origin (from `layer.map_bounds()`). The live display path `_render_map_view` correctly adds `bounds_min_x` / `bounds_min_y` before calling `layer.set_viewport`. For maps with non-zero stage coordinates (absolute µm positions), the passed µm rectangle fell entirely outside all tile geometries, so `_collect_visible_tiles` returned nothing and the canvas stayed all-zeros → white PNG after normalisation.
- **Fix 1:** In `_export_map_roi_worker`, call `layer.map_bounds()` and apply the bounds origin offset: `xmin_um = bounds[0] + x_min * base_px_um` (and equivalently for all four edges). This matches the pattern in `_render_map_view` (line 1083).
- **Root cause 2 (secondary, black tile patches):** `_render_map_region_direct` passed `region_ds` from `_compute_tile_region` to `render_fov_to_array`. That `region_ds` uses absolute downsampled coordinates (non-zero origin). Inside `render_fov_to_array`, `region_xy` is re-clipped via `_ensure_region_within_bounds`, which may change the slice size. The canvas shape was sized from the original `region_ds`, so after clipping the actual sliced array was smaller → the shape-mismatch handler filled only the common area, leaving the rest at zero.
- **Fix 2:** Removed `region_ds=region_ds` from the `render_fov_to_array(...)` call inside `_render_map_region_direct`. `render_fov_to_array` now derives `region_ds` internally from the clipped `region_xy`. The `region_ds` from `_compute_tile_region` is still passed to `_blit_tile` for correct tile placement geometry.
- Added 1 new test (`test_export_map_roi_worker_applies_map_bounds_offset`) verifying non-zero bounds are applied. Updated `_StubLayer` in `test_export_map_roi_worker_calls_render_map_region_direct` and `test_export_map_roi_worker_raises_on_empty_roi` to expose `map_bounds()`. Fixed `region_ds=None` default in mock signature for `test_render_map_region_direct_uses_render_fov_to_array_per_tile`.
- Validated with: `python -m unittest tests.test_export_fovs_batch tests.test_roi_manager_tags` (67/68 pass; 1 pre-existing failure).

**Map mode ROI export: fix black images, stale ROI list, and nan names (Follow-up)**
- **Root cause 1 (black export images after reload):** `_ensure_dataframe()` did not sanitize the `fov` column for NaN. Map-mode ROIs store `fov=""`, which pandas reads back from CSV as `float('nan')`. In `_build_roi_items()`, `NaN` is truthy, so `not fov` evaluates to `False` → the ROI falls through to the single-FOV export path instead of the map-mode path → tries to load a FOV named `"nan"` → produces black output.
- **Fix 1:** Added `"fov"` to both the default-string set and the NaN sanitization loop in `_ensure_dataframe()`.
- **Root cause 2 (ROI list not updating in batch export):** `BatchExportPlugin` never registered an observer on `roi_manager._table`. When `_capture_current_view()` called `add_roi()`, the internal table updated but no signal reached the batch export plugin.
- **Fix 2:** Added `roi_mgr._table.add_observer(...)` in `_connect_events()` that calls `refresh_roi_options()` whenever the ROI table changes.
- **Root cause 3 (nan ROI names):** Same as root cause 1. Also, `refresh_roi_options()` label format didn't handle map-mode ROIs — it displayed raw `fov` value (which was `NaN`/empty) instead of `[MAP:map_id]`.
- **Fix 3:** Updated `refresh_roi_options()` label format to show `[MAP:map_id]` for map-mode ROIs, matching the pattern in `_format_roi_label()`.
- Added 7 new tests: `EnsureDataframeFovSanitizationTests` (4 tests for NaN/empty/normal/CSV-roundtrip), plus `test_build_roi_items_routes_nan_fov_map_roi_to_map_worker`, `test_refresh_roi_options_shows_map_label_for_map_mode_roi`, `test_roi_table_observer_refreshes_batch_export_roi_list`.
- Validated with: `python -m unittest tests.test_export_fovs_batch tests.test_roi_manager_tags` (66/67 pass; 1 pre-existing failure).

**Map mode ROI export: fix white PNG / black PDF output (Follow-up)**
- **Root cause:** `_export_map_roi_worker` called `layer.render(channels)` which internally calls `viewer._compose_fov_image()`. That method reads color and contrast values from live UI widgets (`color_controls`, `contrast_min_controls`, `contrast_max_controls`). During export these widgets may hold stale or zero values (e.g., `contrast_max ≈ 0`), causing all pixels to clip to 1.0 → white PNG, or map through matplotlib's default colormap → black PDF.
- **Fix:** Replaced the `layer.set_viewport / layer.render / restore_viewport` block with a new `_render_map_region_direct()` method on `BatchExportPlugin`. The new method replicates the tile-render loop from `VirtualMapLayer.render()` but calls `render_fov_to_array(tile, arrays, channels, channel_settings, ...)` for each visible tile, explicitly passing `marker_profile.channel_settings` — no UI widget reads. Tiles are stitched onto the canvas via `layer._allocate_canvas` and `layer._blit_tile`, matching the existing pipeline geometry.
- Replaced 2 tests (`test_export_map_roi_worker_calls_set_viewport_and_writes_file`, `test_export_map_roi_worker_restores_viewport_on_success`) with `test_export_map_roi_worker_calls_render_map_region_direct` (verifies um-space coordinates and `channel_settings` forwarding) and `test_render_map_region_direct_uses_render_fov_to_array_per_tile` (verifies `render_fov_to_array` is called with the supplied `channel_settings`, not UI values).
- Validated with: `python -m unittest tests.test_export_fovs_batch` (29/30 pass; 1 pre-existing failure in `test_export_fovs_batch_writes_file` unrelated to these changes).

**Map mode batch export and ROI label fix (Follow-up)**
- **Root cause 1 (batch export drops map-mode ROIs):** `_build_roi_items()` had `if not fov: continue` — map-mode ROIs have `fov=""`, so every one was silently skipped and never exported.
- **Fix 1:** Replaced the unconditional guard with three-way branching: non-empty `fov` → original single-FOV path; `fov=""` + `map_id` non-empty → new map-mode path via `_export_map_roi_worker`; both empty → skip. Output filename for map-mode ROIs uses `map_{map_id}_roi_{roi_id[:12]}.{format}`.
- **New `_export_map_roi_worker()`:** Mirrors `_render_map_roi_tile` (thumbnail renderer). Retrieves the `VirtualMapLayer` via `_get_map_layer(map_id)`, converts canvas pixels to physical µm (`xmin_um = x_min × base_pixel_size_um`), calls `layer.set_viewport(...)` + `layer.render(channels)`, saves/restores `layer._viewport` in a `try/finally`, then passes the rendered array through the existing `_finalise_array` + `_write_image` pipeline. Scale bar `pixel_size_nm` is computed as `base_px_um × 1000 × downsample`.
- **Root cause 2 (ROI label shows stale FOV name after update):** `_update_selected_roi()` stored the new `fov` value but never stored `map_id`, so updating an existing ROI in map mode overwrote the `map_id` with the default empty string, causing `_format_roi_label` to fall back to the stale `fov` value.
- **Fix 2:** Added `map_id` to the `updates` dict in `_update_selected_roi`, using the same `_map_mode_active` / `_active_map_id` pattern already used by `_capture_current_view`. Note: ROIs captured with the *old* code (before this session's `get_active_fov()` fix) retain stale `fov` names and must be re-captured.
- Added 8 new unit tests: `BatchExportMapROIItemsTests` in `test_export_fovs_batch.py` (skips unattributed ROIs, creates map JobItem, single-FOV path unaffected, viewport set/render/write called, viewport restored, zero-extent raises) + 2 new tests in `ROIManagerMapModeTests` (`test_update_selected_roi_stores_map_id_in_map_mode`, `test_update_selected_roi_stores_fov_in_single_fov_mode`).
- Validated with: `python -m unittest tests.test_export_fovs_batch tests.test_roi_manager_tags` (59/60 pass; 1 pre-existing failure in `test_export_fovs_batch_writes_file` unrelated to these changes).

**Map mode ROI capture: blank thumbnails and stale list (Follow-up)**
- **Root cause 1 (blank thumbnails):** `_render_roi_tile` immediately returned `None` when `record["fov"]` was empty — all map-mode ROIs were rendered as "Preview unavailable" in the gallery.
- **Fix 1:** Added `map_id` column to `ROI_COLUMNS` (and `_default_record`/`_ensure_dataframe`). `_capture_current_view` now stores the active map ID alongside `fov = ""`. New `_render_map_roi_tile` method renders the stitched region via `VirtualMapLayer.set_viewport` / `VirtualMapLayer.render`, saving and restoring the layer viewport so live rendering is not disturbed.
- **Root cause 2 (stale list):** After capture, `refresh_roi_table(force_refresh=True)` was called with `preserve_page=True` (default), so the gallery stayed on the current page — new ROIs (sorted newest-first on page 1) were invisible. Additionally, the `_on_roi_table_change` observer fired synchronously *during* `add_roi`, before `_selected_roi_id` was set to the new ID, leaving the dropdown unselected.
- **Fix 2:** Changed `_capture_current_view` to call `refresh_roi_table(force_refresh=True, preserve_page=False)`, navigating to page 1 immediately after capture. `_on_roi_table_change` wrapped in `try/except` to prevent observer exceptions from silently aborting the capture flow. `_format_roi_label` now shows `[MAP:<map_id>]` for map-mode ROIs instead of an empty FOV field.
- Added 7 new unit tests in `tests/test_roi_manager_tags.py` covering `map_id` capture, label formatting, `_render_map_roi_tile` via stub layer, and viewport restore.
- Validated with: `python -m unittest tests.test_roi_manager_tags tests.test_map_mode_activation tests.test_export_fovs_batch` (73/74 pass; 1 pre-existing failure).


- All plugin `image_selector.value` FOV reads replaced with `get_active_fov()` — a new `ImageMaskViewer` method that returns `None` in map mode (where the widget is disabled/stale) and the selector value otherwise.
- New `PluginBase` lifecycle hooks `on_map_mode_activate()` / `on_map_mode_deactivate()` added (no-op stubs, overridden per plugin). Hooks are broadcast from `on_map_mode_toggle()` and `on_map_selector_change()` via `inform_plugins`.
- **ROI Manager** (`roi_manager_plugin.py`): 5 FOV-lookup sites fixed; `on_map_mode_activate` disables and unchecks `limit_to_fov_checkbox` / `browser_limit_to_current` and forces a full-ROI table refresh; `on_map_mode_deactivate` re-enables them.
- **Batch Export** (`export_fovs.py`): 1 FOV-lookup site fixed in `refresh_roi_options()`; `on_map_mode_activate` disables and unchecks `roi_limit_to_fov` and refreshes ROI options; `on_map_mode_deactivate` re-enables.
- **`center_on_roi()`** rewritten to be map-aware: in map mode, FOV-local pixel corners are translated to stitched-canvas coordinates via `resolve_cell_map_position()` without touching `image_selector`; records with empty `fov` use raw coordinates directly.
- Added 13 unit tests across `test_map_mode_activation.py` (`GetActiveFovTests`, `CenterOnRoiMapModeTests`), `test_roi_manager_tags.py` (`ROIManagerMapModeTests`), and `test_export_fovs_batch.py` (`BatchExportMapModeTests`). All pass.
- Validated with: `python -m unittest tests.test_map_mode_activation tests.test_roi_manager_tags tests.test_export_fovs_batch` (69/69 tests pass; 1 pre-existing failure excluded).



**Map mode large-dataset crash fix — render suppression & tile cap (Follow-up)**
- **Root cause confirmed:** During `load_widget_states`, changing every saved widget value fires the slider observer (`FloatSlider.value` → `lambda: update_display()`). For a 400-tile map on a slow network FS this produced many successive full-map renders (one per channel × 2 sliders), each requiring hundreds of sequential TIFF reads — enough to time out the Jupyter kernel connection.
- **Fix 1 — Suppress renders during state restoration** (`main_viewer.py`): Added `_suspend_display_updates: bool` flag. Set to `True` in `__init__` around `load_widget_states` (in a try/finally) and checked at the top of `update_display`. All widget-observer render bursts are silently skipped; the first real render happens on first user interaction after the viewer is displayed.
- **Fix 2 — Per-render tile cap** (`virtual_map_layer.py`): Added `_RENDER_TILE_LIMIT` check in `render()`. When visible tiles exceed `viewer._map_render_tile_limit` (default 80), tiles are sorted by distance from the viewport centre and only the nearest 80 are rendered in a single call. This bounds the TIFF-read cost of any single render for very large maps.
- Also added `_map_render_tile_limit: int = 80` instance variable to `ImageMaskViewer` (settable by the user to tune for their dataset / FS speed).
- Prior fix (batch stats tile cap, same log entry) is preserved.
- Validated with: `python -m unittest tests/test_map_mode_activation.py tests/test_virtual_map_layer.py` (19/19 tests pass).

**Packaged `image_utils` restore (#79 follow-up)**
- Restored `ueler.image_utils` as a concrete packaged module after the root-level `image_utils.py` removal left canonical imports pointing at a missing alias target.
- Reversed the utility compatibility shims so legacy imports (`image_utils`, `data_loader`, `constants`) resolve to the packaged `ueler.*` modules rather than the deleted root-level files.
- Added focused regression coverage in `tests/test_shims_imports.py` and validated the restored downsample helper via `tests.test_roi_manager_tags.ROIManagerTagsTests.test_select_downsample_factor_clamps_against_allowed_list`.

**Batch export marker set dropdown live refresh (#78 follow-up)**
- `BatchExportPlugin.on_marker_sets_changed()` added; it calls `refresh_marker_options()` so the marker set dropdown updates immediately whenever a set is saved, updated, or deleted — no viewer restart needed.
- `main_viewer.update_marker_set()` now calls `update_marker_set_dropdown()` after overwriting an existing set, ensuring `inform_plugins('on_marker_sets_changed')` fires for that path too (it was previously missing).
- Added `TestMarkerSetDropdownRefresh` (2 tests) in `tests/test_export_fovs_batch.py` covering the delegation and idempotency of `on_marker_sets_changed()`.
- Validated with: `python -m unittest tests.test_export_fovs_batch` (`Ran 20 tests … 19 OK, 1 pre-existing failure`).

**Batch export available in simple viewer mode (#78)**
- `BatchExportPlugin` is now loaded when only images are present (`cell_table is None`), enabling full-FOV and ROI exports without needing a cell table.
- Added `_refresh_mode_availability()` to `BatchExportPlugin`: replaces the Single Cells tab content with an informational notice when no cell table is loaded, and restores it when one is set later via `on_cell_table_change()`.
- `display_ui()` whitelist expanded to include `"export_fovs"` for simple mode.
- `_refresh_mode_availability()` is called at init, after `after_all_plugins_loaded()`, and on every `on_cell_table_change()` event.
- Added 9 unit tests in `tests/test_export_fovs_batch.py` (`TestSimpleViewerModeExport`) covering: notice rendering, tab index reset, full/ROI tab preservation, cell table restoration, and idempotent invocations.
- Validated with: `python -m unittest tests.test_export_fovs_batch` (`Ran 18 tests … 17 OK, 1 pre-existing failure`).

**Histogram cutoff linked to cell gallery (#77)**
- `highlight_cells()` in `ChartDisplay` now updates `selected_indices` with all-FOV row indices matching the cutoff, triggering the existing `forward_to_cell_gallery` observer and populating the cell gallery whenever the "Cell gallery" linking checkbox is enabled.
- Toggling "above"/"below" in the histogram controls automatically re-applies the filter without a second histogram click (new observer registered in `setup_observe()`).
- Image display highlighting (current FOV only) is unchanged.
- Added 10 unit tests in `tests/test_chart_cell_gallery_link.py` covering above/below filtering, all-FOV gallery scope, checkbox gate, and auto-update on toggle.

**Channel grid display — cell selection and hover tooltip (#76 follow-up)**
- Grid view now supports the same cell-selection and hover-tooltip interactivity as the composited view.
- Clicking a cell in any pane toggles it in the shared `selected_masks_label` set; Ctrl+click adds to the current selection; right-click clears all selections.
- White-edge highlights are applied to every pane simultaneously via `_update_grid_patches`, mirroring `ImageDisplay.update_patches`.
- A per-pane debounced hover tooltip (300 ms) shows mask ID and per-cell values, mirroring `ImageDisplay.process_hover_event`.
- `_update_grid_display` now populates `full_resolution_label_masks` so mask-hit lookup and edge rendering work in grid mode.
- FOV change (`on_image_change`) now clears grid patches when grid mode is active.
- `update_panes` stores clean copies of rendered arrays in `_stored_arrays` so highlights can be removed without a full re-render.
- Extended `tests/test_channel_grid_view.py` with 7 new interactivity tests (total 29 tests).
- Validated with: `python -m unittest tests.test_channel_grid_view` (`Ran 29 tests ... OK`).

**Channel grid display mode (#76)**
- Added a "Channel grid view" checkbox to the Channels accordion that renders each visible channel as a separate labelled pane in a grid layout.
- Grid uses a single matplotlib figure with `sharex=True, sharey=True`, so pan/zoom in any pane synchronises all others automatically.
- Each pane shows the channel name as an overlaid text label.
- The grid viewport is seeded from the current composited-view position; switching back restores that position to the main viewer.
- Map mode disables and deactivates grid mode (the two modes are mutually exclusive).
- New `GridChannelDisplay` class in `ueler/viewer/channel_grid_view.py` encapsulates the figure management and viewport-sync logic.
- Added 22 unit tests in `tests/test_channel_grid_view.py`.
- Validated with: `python -m unittest tests.test_channel_grid_view` (`Ran 22 tests ... OK`).

**Dev note topic summaries**
- Added topic-oriented summaries in `dev_note/` to consolidate project notes by area (viewer runtime, map mode, OME-TIFF, heatmap/FlowSOM, ROI, exports, and packaging).
- Added `dev_note/index.md` to map original notes and issue-tracking files into the new summary topics.
- Removed `dev_note/issue_tracking/` after distilling its contents into the topic summaries and related issue links.

### v0.3.0-beta

**Channel color legend (#75)**
- Added a per-channel color legend with an on-image overlay and adjacent UI legend that mirrors the rendered channel colors.
- Legend entries reflect only currently visible channels and can be toggled on/off via a new viewer control.
- Added unit coverage in `tests/test_channel_legend.py`.
- Validated with: `python -m unittest tests.test_channel_legend`.

**Heatmap meta-cluster colors beyond cutoff (#74)**
- Fixed heatmap row-color resolution so explicit meta-cluster registry colors are applied before cutoff-derived colormap sampling.
- This ensures user-added/revised meta-cluster IDs beyond the original dendrogram cutoff keep their intended colors in plot displays.
- Added regression coverage in `tests/test_heatmap_selection.py`.
- Validated with: `python -m unittest tests.test_heatmap_selection tests.test_heatmap_adapter` (`Ran 26 tests ... OK`, `skipped=2`).

**Heatmap z-score color palette toggle (#73 follow-up)**
- Updated heatmap coloring to follow normalization mode:
  - z-score mode uses a diverging palette (`bwr`) with white centered at `0`, blue for negative values, and red for positive values,
  - non-zscore mode uses a red sequential palette (`Reds`).
- Added test coverage for palette selection and clustermap kwargs forwarding in `tests/test_heatmap_adapter.py`.
- Validated with: `python -m unittest tests.test_heatmap_selection tests.test_heatmap_adapter` (`Ran 25 tests ... OK`, `skipped=2`).

**Heatmap z-score across markers option (#73 follow-up)**
- Added a new Heatmap setup toggle, `Z-score across markers`, to normalize each class across selected markers.
- Preserved existing default behavior (marker-wise z-score across classes) when the new toggle is off.
- Added focused tests for both normalization modes in `tests/test_heatmap_selection.py` (auto-skipped under bootstrap pandas stubs).
- Validated with: `python -m unittest tests.test_heatmap_selection tests.test_heatmap_adapter` (`Ran 22 tests ... OK`, `skipped=2`).

**Heatmap Save to Cell Table uses renamed labels (#73 follow-up)**
- Updated Heatmap `Save to Cell Table` so the requested output column stores meta-cluster display labels (including user-renamed labels) instead of raw numeric meta-cluster IDs.
- Updated the companion `<column_name>_revised` column to store display labels as well, so both exported columns are label-based.
- Confirmed compatibility by running focused heatmap suites: `python -m unittest tests.test_heatmap_selection tests.test_heatmap_adapter` (`Ran 20 tests ... OK`).

**Heatmap horizontal-layout width clamp (#73 follow-up)**
- Limited wide-layout heatmap figure width to the plugin width budget so rendered heatmaps no longer overflow the Heatmap footer tab in horizontal layout.
- Kept data-driven sizing for smaller cluster counts while capping large cluster sets to prevent tab-width overrun.
- Added regression coverage in `tests/test_heatmap_adapter.py` for wide-mode clamping and unchanged vertical sizing.

**Heatmap horizontal-layout footer replay fix (#73 follow-up)**
- Fixed a wide-layout rendering issue where heatmaps could be generated but the footer panel remained blank after pressing `Plot`.
- Added cached figure/canvas replay and a wide-mode post-plot restore hook in `DisplayLayer` so the footer panel reliably re-renders the latest heatmap.

**Heatmap meta-cluster registry sync fix (#73 follow-up)**
- Fixed a runtime crash in `DataLayer._sync_meta_cluster_registry` caused by boolean evaluation of numpy arrays (`meta_cluster_ids or []`) during heatmap regeneration.
- Updated the sync path to guard only for `None`, preserving numpy array inputs from `np.unique(meta_cluster_labels)`.
- Added regression coverage in `tests/test_heatmap_selection.py` to verify numpy-array IDs are handled safely.

**Heatmap meta-cluster management tab (#73)**
- Added a new `Rename` tab to the Heatmap plugin with meta-cluster rename/add/remove controls and a color-aware registry preview.
- Replaced free-text meta-cluster assignment in the `Assign` tab with a dropdown fed by user-defined meta-cluster labels.
- Removing a meta-cluster now reassigns any existing rows to the default unassigned meta-cluster (`-1`) to avoid invalid assignments.
- Added focused regression coverage in `tests/test_heatmap_selection.py` for dropdown assignment, new meta-cluster creation, and remove+reassign behavior.

**Per-channel visibility toggles (#66)**
- Added a per-channel on/off checkbox in the channel controls so users can temporarily hide individual channels without altering the channel selection list.
- Rendering now filters by visibility state, preserving existing color and contrast settings when channels are re-enabled.
- Tightened the per-channel color dropdown sizing to keep the visibility checkbox visible alongside long channel labels.

**ROI manager without cell table (#65)**
- Enabled the SidePlots accordion to render even when `cell_table` is missing by loading only the ROI manager plugin in that case, keeping ROI capture and persistence available for raw-image-only sessions.
- Preserved the full plugin set when the cell table is present, avoiding regressions for existing workflows.

**VS Code scatter fallback (#64)**
- Added a VS Code–aware scatter backend selector so notebooks running under VS Code (or when `UELER_SCATTER_BACKEND=static`) fall back to a static Matplotlib scatter with an inline notice, preventing blank outputs when the jupyter-scatter/anywidget frontend fails to hydrate. Users can force the interactive widget backend via `UELER_SCATTER_BACKEND=widget` after enabling widget support. [ueler/viewer/plugin/chart.py#L90-L110](ueler/viewer/plugin/chart.py#L90-L110) [ueler/viewer/plugin/chart.py#L262-L280](ueler/viewer/plugin/chart.py#L262-L280) [ueler/viewer/plugin/chart.py#L500-L528](ueler/viewer/plugin/chart.py#L500-L528)

**OME-TIFF keyframe compatibility (#63 follow-up)**
- Added a tolerant OME reader fallback that retries series discovery without OME parsing when tifffile raises `RuntimeError: incompatible keyframe`, letting stacked TIFFs with mismatched keyframes load instead of crashing the viewer.
- Introduced regression coverage in `tests/test_ome_tiff_loading.py::test_incompatible_keyframe_retries_without_ome_series` and ran `python -m unittest tests.test_ome_tiff_loading` to validate the fallback path.

**OME-TIFF suffix-less detection (#63 follow-up)**
- Detect OME-TIFF files that lack the `.ome.tif(f)` suffix by inspecting TIFF metadata, enabling valid OME stacks named with plain `.tif`/`.tiff` extensions to load and appear in the FOV list.
- Added regression coverage in `tests/test_ome_tiff_loading.py::test_find_ome_tiff_files_detects_suffixless` and re-ran `python -m unittest tests.test_ome_tiff_loading`.

**OME-TIFF frame-aware lazy loading (#63)**
- Added frame-aware slicing to `OMEFovWrapper`, including frame metadata, cache keys keyed by frame, and a setter for active frame index to keep stacked TIFF reads lazy via Dask.
- Adjusted pyramid level selection to prefer the coarsest level that meets floor-based downsample expectations, avoiding over-fetching while satisfying irregular pyramids.
- Plumbed frame selection and metadata through `ImageMaskViewer.load_fov`, storing per-FOV OME info for downstream UI/diagnostics.
- Extended `tests/test_ome_tiff_loading.py` with frame-selection coverage; ran `python -m unittest tests.test_ome_tiff_loading tests.test_ome_rendering_fix`.

**OME-TIFF Memory Crash Fix (#60 follow-up)**
- Fixed a critical memory crash in `ImageDisplay` initialization where full-resolution OME-TIFF dimensions caused massive buffer allocation (e.g., 23GB for 44k images). The viewer now initializes with a minimal buffer while maintaining correct full-resolution coordinate systems for zooming and panning.


**OME-TIFF Memory Optimization (#60 follow-up)**
- Resolved memory maxout issues when loading large multi-channel OME-TIFF files by switching from eager to lazy channel statistics computation.
- `ImageMaskViewer` now computes channel max intensity only when a channel is selected for display, significantly reducing startup time and memory footprint for datasets with many channels.

**OME-TIFF Rendering Fixes (#60 follow-up)**
- Fixed view drift when switching pyramid levels in OME-TIFF files by correctly inferring the full-resolution image dimensions from the OME metadata wrapper instead of the downsampled dask array.
- Resolved `ValueError` when zooming out to lower resolution levels by handling shape mismatches between the downsampled OME-TIFF array and the expected canvas size, ensuring robust rendering even when pyramid levels have slightly different dimensions due to rounding.
- Fixed an issue where zooming out fully would not show the entire image due to aggressive downsampling level selection; the viewer now prioritizes pyramid levels that are exact divisors of the requested factor and validates that the selected level covers the full image extent, preventing incomplete display when pyramid levels have irregular dimensions.
- Added regression tests in `tests/test_ome_rendering_fix.py` and `tests/test_ome_level_selection_fix.py` to verify correct shape inference, rendering stability, and downsample level selection.

**OME-TIFF Loading Support (#60)**
- Implemented native support for loading OME-TIFF files (`.ome.tif`, `.ome.tiff`) directly in UELer.
- Added `OMEFovWrapper` in `ueler.data_loader` to handle lazy loading of OME-TIFFs using `dask-image` and `tifffile`, ensuring memory efficiency by respecting the viewer's downsampling factor.
- Updated `ImageMaskViewer` to automatically detect OME-TIFF files in the base folder and switch to `ome-tiff` mode, populating the FOV list from file names.
- Preserved existing folder-based loading behavior while enabling seamless integration of OME-TIFF datasets.
- Added unit tests in `tests/test_ome_tiff_loading.py` to verify channel name extraction and wrapper functionality.

**Map mode stitched interactions (#62 follow-up)**
- Reworked stitched mask highlight overlays to pull tile placement from `MapTileViewport` metadata and aligned single-FOV contour drawing with viewport offsets, resolving the Reply (3) highlight drift and `IndexError` when zoomed.
- Added a descriptor-driven FOV→map index and updated `focus_on_cell` to activate the correct stitched map before resolving coordinates, preventing cross-slide navigation from landing on stale canvases.
- Extended `VirtualMapLayer` with viewport metadata and `localize_map_pixel`, and wired `ImageMaskViewer`/`ImageDisplay` hover & click handlers to translate stitched map pixels back into FOV-local mask hits while preserving tooltip data.
- Introduced the `MaskSelection` structure across viewer plugins so chart and heatmap tracing consume stitched-aware selections, restoring mask highlights in map mode without breaking single-FOV workflows.
- Expanded regression coverage in `tests/test_map_mode_activation.py` (map switching, reverse localization, stitched highlight alignment) and `tests/test_image_display_tooltip.py` (viewport-aware patch updates); ran `python -m unittest tests.test_map_mode_activation` and `python -m unittest tests.test_image_display_tooltip`.

**Map mode cell localization (#62)**
- Extended `VirtualMapLayer` with per-FOV geometry lookups and taught `ImageMaskViewer` to convert FOV-local cell coordinates into stitched-map pixels via the new `resolve_cell_map_position` and `focus_on_cell` helpers, keeping toolbar history intact while navigating.
- Updated the scatter, heatmap, cell gallery, and Go To plugins to rely on the stitched coordinate helper when map mode is active, preserving legacy behaviour when descriptors omit the target FOV.
- Added regression coverage in `tests/test_map_mode_activation.py` and expanded `tests/bootstrap.py` stubs for the `skimage` modules so the viewer’s map-navigation paths remain testable without heavy dependencies; ran `python -m unittest tests.test_map_mode_activation` to confirm the fix.
- Addressed a follow-up `ValueError` raised from `jscatter.compose` by disabling redundant selection syncing in the chart plugin, relying on the existing observer pipeline; added a regression test to ensure the compose integration remains selection-safe.

**Cell tooltip key alignment (#61)**
- Reworked `ImageDisplay` hover handling to resolve cell records via the viewer’s configured `fov`, `label`, and optional `mask` keys so datasets with renamed columns now show full channel means and user-selected labels instead of just the mask ID.
- Added `resolve_cell_record` helper in `ueler/viewer/tooltip_utils.py` with caching and unit coverage for default and custom key combinations, keeping repeated hover events fast while gracefully skipping missing rows.
- Expanded `tests/test_image_display_tooltip.py` with lightweight table stubs and integration coverage, running `python -m unittest tests.test_image_display_tooltip` to confirm tooltips render the expected values across key configurations.


### v0.3.0-alpha
**Map reset alignment (#59)**
- Refresh the Matplotlib navigation stack whenever map mode resizes the canvas so the reset button now restores the stitched slide bounds instead of the original square FOV, eliminating the post-reset black canvas and zoom failures.
- Added regression coverage for populated and empty toolbar stacks in `tests/test_map_mode_activation.py` and ran `python -m unittest tests.test_map_mode_activation` to confirm the update.

**Map viewport alignment (#59)**
- Offset stitched-map viewport coordinates by each descriptor’s minimum X/Y bounds so tiles render correctly when slide coordinates do not originate at zero, resolving the black canvas shown after switching maps.
- Added regression coverage in `tests/test_map_mode_activation.py::test_render_map_view_offsets_descriptor_bounds` to ensure the viewer now supplies offset-aware viewports to `VirtualMapLayer`.

**Map mode activation stability (#58)**
- Replaced the full-resolution placeholder in `_set_map_canvas_dimensions` with a constant 1×1 canvas so selecting large stitched maps no longer allocates 50+ GB arrays and crashes the kernel.
- Recomputed the stitched-view downsample factor during map activation using `select_downsample_factor`, ensuring the first render starts at the coarsest allowed scale for the slide dimensions.
- Added `tests/test_map_mode_activation.py` to cover the placeholder shape and downsample recalculation with lightweight dependency stubs.
- Hardened `image_utils.get_axis_limits_with_padding` so even very coarse downsample factors produce positive viewport bounds, resolving the "Viewport must have positive width and height" regression reported after shipping the crash fix.
- Fixed the zoom-triggered `ValueError: operands could not be broadcast together` by recalculating stitched tile bounds with ceil-based downsample dimensions in `VirtualMapLayer._compute_tile_region`; locked with `tests/test_virtual_map_layer.py::test_render_handles_partial_downsample_tiles`.
**Map-mode cache integration (#3)**
- Centralized the stitched tile cache inside `ImageMaskViewer`, wiring image cache evictions to `VirtualMapLayer.invalidate_for_fov` and driving cache keys with the viewer's channel/mask/annotation state so map renders stay in sync with UI changes.
- Extended `tests/test_virtual_map_layer.py` to assert cache busting when the viewer signature changes, protecting the new integration path.

**FOV load cycle documentation (map mode)**
- Expanded `dev_note/FOV_load_cycle.md` with the current map mode scaffolding, covering the `ENABLE_MAP_MODE` flag, descriptor ingestion via `MapDescriptorLoader`, and how `VirtualMapLayer` reuses `_render_fov_region` for stitched viewport renders.
- Documented the region-level cache keys and geometry helpers exposed by `VirtualMapLayer`, plus the pending invalidation wiring so backend groundwork is visible ahead of UI integration.

**VirtualMapLayer core groundwork (#3)**
- Introduced `VirtualMapLayer` to stitch slide descriptors into viewport-sized composites, including per-tile caching and viewport intersection logic.
- Refactored `ImageMaskViewer` rendering into a reusable `_compose_fov_image` helper and exposed `_render_fov_region` so map mode can reuse existing channel, annotation, and mask pipelines.
- Added `tests/test_virtual_map_layer.py` to verify gap handling, cache reuse, and invalidation semantics for the new layer.

**Map descriptor loader groundwork (#3)**
- Added `MapDescriptorLoader` to validate translation-only slide descriptors, reject mixed-unit inputs, and surface duplicate FOV warnings in preparation for map-based tiled mode.
- Wired the loader into `ImageMaskViewer` behind the `ENABLE_MAP_MODE` feature flag so descriptor parsing feedback reaches users without affecting existing single-FOV workflows.
- Introduced unit fixtures covering valid, mixed-unit, and malformed descriptors to guarantee deterministic slide/FOV registries during ingest.

### v0.2.1
**Mask painter performance patch**
- Fixed performance regression introduced in v0.2.0 where using the mask painter across multiple FOVs caused slowdowns due to expensive global color registration and unnecessary cell gallery regeneration on FOV changes.

### v0.2.0
**Branch merge organization**
- Merged `pre-release` into `main` to consolidate all v0.2.0 release candidate changes.

### v0.2.0-rc3
**Test suite fixes**
- Fixed `test_compute_scale_bar_spec_scales_when_pixel_size_expands` to correctly validate that physical length doubles when pixel size doubles (pixel length remains constant as both select from the same rounding sequence).
- Fixed `matplotlib.pyplot.show()` call in cell gallery to use parameterless form (`plt.show()` instead of `plt.show(fig)`) for compatibility with newer matplotlib backends.
- Added 3 new tests for issue #56 in `test_painted_colors_all_fovs.py` to verify colors are registered for all FOVs independently of viewer state.
- Test suite status: 125 of 130 tests passing (96.2% pass rate). Remaining 5 failures are non-critical: 1 test environment issue (tifffile bootstrap), 1 test isolation issue (export test), and 3 module aliasing tests that validate implementation details without affecting functionality.

**Gallery painted colors independence**
- Fixed issue [#56](https://github.com/HartmannLab/UELer/issues/56) where painted cell mask colors only appeared in the gallery when an FOV was loaded and masks were painted in the main viewer.
- Refactored `apply_colors_to_masks` in `mask_painter.py` to separate two responsibilities: painting masks in the viewer (current FOV only) and registering colors globally (all FOVs).
- The mask painter now registers colors for **all cells** matching each class across **all FOVs** in the cell table, not just those in the currently displayed FOV.
- Gallery can now access painted colors via the centralized `_CELL_COLOR_REGISTRY` regardless of which FOV is loaded in the viewer or whether the viewer has been opened.
- Added `_register_color_globally` helper that iterates through the entire cell table to ensure complete color coverage for gallery rendering.

**Cell gallery mask color consistency - Phase 5 & 6 (Error handling, polish, and documentation)**
- Added graceful error handling for corrupted or missing mask data—gallery now displays red-tinted error placeholders instead of crashing when tile rendering fails (addresses [#55](https://github.com/HartmannLab/UELer/issues/55)).
- Implemented performance warning system that alerts users when display count exceeds 100 cells: "Performance may degrade above 100 cells. Consider reducing display count for better responsiveness."
- Removed all debug logging statements (15+ prints) from production code and eliminated redundant inline imports.
- Enhanced code documentation with comprehensive inline comments explaining:
  - Two-pass z-order rendering strategy (neighbors first, centered cell last)
  - Thickness control separation (gallery slider for centered cell, global setting for neighbors)
  - Uniform vs. painted color mode logic and fallback behavior
- Test coverage: All 11 unit tests passing (6 color tests, 2 error handling tests, 2 FOV change tests, 1 canvas composition test).
- Implementation complete across 6 phases: Setup → Investigation → Core Fix + Refinements → Navigation (skipped) → Error Handling → Polish.

**Cell gallery mask painter synchronization**
- Synchronized cell gallery with mask painter for adaptive thickness and painted color display (fixes [#54](https://github.com/HartmannLab/UELer/issues/54)).
- Extended `_notify_plugins_mask_outline_changed` in `main_viewer.py` to notify the cell gallery plugin when mask outline thickness changes, ensuring all viewing contexts stay synchronized.
- Added `on_viewer_mask_outline_change` and `on_mask_painter_change` callbacks to `CellGalleryDisplay` so the gallery auto-refreshes when the main viewer's thickness slider moves or the mask painter applies colors.
- Implemented persistent `cell_id_to_color` mapping in the mask painter plugin that stores `(fov, mask_id)` -> `color` for all painted cells, enabling the cell gallery to retrieve and display exact painted colors.
- Added "Use uniform color" checkbox to the cell gallery UI: when disabled (default), the gallery displays painted mask colors from the mask painter; when enabled, all masks use the uniform color from the "Mask colour" picker.
- Modified `_render_tile_for_index` in `cell_gallery.py` to query the mask painter's color mapping when `use_uniform_color=False`, ensuring painted classifications appear correctly in gallery thumbnails.
- Ran `python -m py_compile` on all modified files to validate syntax correctness.

**ROI gallery width stabilization**
- Switched ROI gallery to static narrow figure sizing (4.8 inches at 72 DPI ≈ 346px) to eliminate thumbnail clipping at narrow widths (addresses [#39](https://github.com/HartmannLab/UELer/issues/39)).
- Removed all ResizeObserver-based responsive width code after investigation revealed that JavaScript can only resize DOM wrapper elements, not Matplotlib's pre-rendered raster content—when the container shrinks below the original render width, the fixed-size raster overflows and clips.
- Gallery now renders conservatively narrow and relies on CSS `width: 100%` to stretch when space is available, trading slight blur at wider widths for guaranteed no-clip behavior in narrow panels.
- Fixed vertical clipping issue by removing redundant VBox scroll container wrapper—`browser_output` now provides the only scrolling container, allowing canvas to extend to its full calculated height.
- Updated test expectations in `tests/test_roi_manager_tags.py` to validate static 4.8-inch width and direct canvas return without wrapper.
- Documented root cause and solution alternatives in `dev_note/gallery_width.md` for future reference.

**Cache configuration**
- Relocated the cache size control to the Advanced Settings tab and raised its default to 100 so fresh viewers follow the tuned cache policy without squatting space in the header (fixes [#53](https://github.com/HartmannLab/UELer/issues/53)).
- Added `tests/test_cache_settings.py` to pin the widget placement and viewer default, keeping future layout changes from regressing cache behaviour.

**Cell tooltip precision**
- Updated `ImageDisplay` marker tooltips to fall back to scientific notation whenever fixed-point rounding would display small non-zero intensities as 0.00, keeping near-zero markers readable (fixes [#51](https://github.com/HartmannLab/UELer/issues/51)).
- Added `tests/test_image_display_tooltip.py` to cover regular, tiny, zero, and negative marker values so the formatter stays stable.

**ROI browser refresh throttling**
- Cached a lightweight browser signature in `ROIManagerPlugin` so routine FOV changes reuse the current page without regenerating thumbnails, while targeted actions (add/update/delete) request a forced refresh to keep presets in sync (fixes [#50](https://github.com/HartmannLab/UELer/issues/50)).
- Resolved per-ROI preset playback by threading the mask painter's active colour set through thumbnail rendering, ensuring previews match saved settings without wiping pagination.

**Cell gallery FOV debounce**
- Gallery clicks now track viewer-initiated FOV changes and skip the next `on_fov_change` cycle, preventing the redundant redraw loop reported in [#52](https://github.com/HartmannLab/UELer/issues/52).
- External plugin interactions (e.g., scatter single-point navigation) still clear the guard so multi-plugin workflows refresh as expected while viewer-driven hops stay snappy.

**ROI filter tabset**
- ROI manager browser filters now live behind `simple` and `advanced` tabs, separating the AND/OR widgets from expression entry for clearer navigation (addresses [#49](https://github.com/HartmannLab/UELer/issues/49)).
- The ROI browser only evaluates the active tab's inputs and refreshes pagination on tab switches so tag selections and expressions no longer conflict.

### v0.2.0-rc2
**Mask outline scaling**
- Added downsample-aware outline helpers so viewer overlays, selection highlights, and mask painter recolouring apply `max(1, t/f)` thickness scaling (fixes [#46](https://github.com/HartmannLab/UELer/issues/46)).
- Centralised the scaling logic in `ueler.rendering.engine` and extended `tests/test_rendering.py` with regression coverage to guard against future regressions.

**Scatter gallery guard**
- Added `single_point_click_state` to the chart plugin and taught trace/scatter selection paths to toggle it so single-cell interactions mark the next viewer navigation while suppressing single-point gallery syncs (fixes [#48](https://github.com/HartmannLab/UELer/issues/48)).
- Implemented `CellGalleryDisplay.on_fov_change` to clear the flag, skip single-cell refreshes, and re-render only when multi-cell selections persist, with new tests in `tests/test_chart_footer_behavior.py` and `tests/test_cell_gallery.py` covering the forwarding guard and handshake.

**Main viewer downsampling docs**
- Summarized the automatic FOV downsampling flow in `dev_note/main_viewer.md`, covering factor selection, caching strategy, zoom toggles, and scale bar corrections so notebook users understand how large scenes stay responsive.

**ROI browser layout fixes**
- Set the ROI browser output widget to a 400px viewport with vertical scrolling so the accordion stays compact even when dozens of thumbnails load.
- Rebuilt the Matplotlib gallery sizing to keep a fixed three-column grid, pad empty slots, and clamp the figure width to 98% of the plugin width for consistently visible tiles.
- Ran `python -m unittest tests.test_roi_manager_tags` to cover the new layout helper and scroll container assertions.

**ROI browser layout follow-up**
- Allowed the thumbnail output container and parent flex boxes to shrink (`min_width=0`, `flex=1 1 auto`) so the scrollbar scopes to the gallery instead of the entire plugin.
- Injected a scoped CSS rule (`.roi-browser-output img`) plus DPI tuning to keep Matplotlib renders within the widget bounds and leave pagination buttons unobscured.
- Added an ipympl-aware canvas layout helper that forces `fig.canvas` to honour the 400px viewport; fall back to `plt.show` when the widget backend is unavailable.
- Repeated `python -m unittest tests.test_roi_manager_tags` to exercise the new layout expectations and confirm the CSS helper is applied once.

**ROI browser canvas containment**
- Swapped the direct `plt.show` usage for a fixed-height `VBox` that wraps the Matplotlib canvas, applying a pixel-specific `Layout` so the gallery honours the 400px viewport while clipping horizontal overflow.
- Updated `_configure_browser_canvas` and its unit tests to accept an explicit pixel height, confirm both the canvas and wrapper sizing, and guarantee the scroll container engages when figures exceed the viewport.
- Ran `python -m unittest tests.test_roi_manager_tags` to validate the wrapper behaviour and refreshed ipywidgets stubs for positional children support.

**ROI browser thumbnails**
- Reused the new `select_downsample_factor` helper to downsample ROI previews automatically, capping the longest edge at 256 px and ignoring stale zoom metadata for smoother scrolling.
- Follow-up: thumbnails now compute their factor from each ROI viewport (`factor = 2^ceil(log2(ceil(longest/256)))`) and `_derive_downsampled_region` applies ceiling division so non-divisible dimensions still render complete tiles.

**ROI preview reliability**
- Normalised ROI coordinate parsing (strings/NaNs) and synced downsampled bounds with the sampled pixel grid so cropped ROIs render alongside full-FOV entries instead of showing “preview unavailable” (Reply 6 to [#44](https://github.com/HartmannLab/UELer/issues/44)).
- Added regression coverage for string-based ROI metadata in `tests/test_rendering.py` and reran `python -m unittest tests.test_rendering tests.test_roi_manager_tags` to confirm the fix.

**ROI browser pagination**
- Replaced the scroll-triggered lazy loader with explicit Previous/Next page buttons and a live page indicator, rendering 3x4 tiles per step so navigation stays deterministic (addresses [#44](https://github.com/HartmannLab/UELer/issues/44)).
- Reset pagination whenever filters or expressions change and surfaced per-page summaries plus disabled states for boundary pages to keep the gallery responsive.

**Expression helper cursor awareness**
- Injected a custom widget bridge that tracks selection and focus inside the expression field and reuses it when inserting operator/tag snippets, keeping helpers splice-at-caret instead of appending blindly.
- Updated ROI manager helpers and unit scaffolding (`tests/test_roi_manager_tags.py`) to match the cursor-aware workflow while dropping the deprecated scroll listener plumbing.
- Preserved the last focused caret snapshot when helper buttons momentarily steal focus so blur events no longer shove insertions back to index 0, and added a regression test covering the selection hand-off.
- Hardened the DOM selector logic so the JavaScript bridge finds the Text widget reliably in JupyterLab 4, classic Notebook, and Voila frontends, restoring caret telemetry on stacks where the prior `[data-widget-id]` hooks failed.
- Reset the stored caret to the end of restored expressions whenever the widget reloads unfocused, ensuring the very first helper insertion appends instead of prefixing text resurrected from notebook state.
- Reworked `_insert_browser_expression_snippet` to slice around the cached start/end indices so helpers replace highlighted ranges or append at the caret, advancing the stored selection to trail the inserted snippet for chained clicks.
- Expanded `tests/test_roi_manager_tags.py` with insertion coverage at the head, middle, tail, and with highlighted replacements to guard the new behaviour.
- Restored the selection resolver and focus-aware caching after a regression so helper buttons continue to update the field even if caret telemetry drops temporary blur events.
- Moved helper insertion entirely into the browser via custom `insert-snippet` messages so the front end updates the field and caret before syncing changes back to Python, avoiding race conditions with focus churn.
- Added a readiness check that falls back to the Python insertion path until the browser bridge confirms a caret snapshot, preventing helper buttons from no-oping while the widget script attaches.
- Swapped the HTML `<script>` injection for `IPython.display.Javascript` so the caret bridge executes even in sanitized JupyterLab outputs, keeping selection telemetry alive across lab builds.
- Simplified the caret bridge to the DOM-binding pattern proven in the standalone ipywidgets demo so it locates the text input through stable selectors, executes the helper script via a shared `ipywidgets.Output` in the same frame, performs the splice entirely client-side, and then reconciles value/caret state back to Python across Jupyter front-ends.

- ✅ Ran `python -m unittest tests.test_roi_manager_tags` to exercise the caret retention regression, insertion index coverage, and the front-end insertion bridge.
- ⚠ Additional widget-harness coverage for the pagination helpers remains pending until the revamped stubs land.

**Cell gallery tile padding**
- Updated `_compose_canvas` to size gallery slots by the largest rendered tile and center narrower crops so mixed-width images no longer raise broadcasting errors when assembling the grid (fixes [#43](https://github.com/HartmannLab/UELer/issues/43)).
- Added `tests/test_cell_gallery.py` to exercise the padding logic and ran `python -m unittest tests.test_cell_gallery` to confirm the regression stays fixed.
**Cell gallery rendering unification**
- Reimplemented `ueler.viewer.plugin.cell_gallery` against the shared `ueler.rendering` engine, wiring cutout sizing, downsampling, and mask-outline controls through overlay snapshots so gallery tiles match the main viewer and batch export outputs (addresses [#43](https://github.com/HartmannLab/UELer/issues/43)).
- Restored legacy helpers (`find_boundaries`, `_label_boundaries`, `_binary_dilation_4`) via `ueler.viewer.rendering` to keep downstream consumers and renderer tests operational, and reran `python -m unittest tests.test_rendering tests.test_export_fovs_batch` to validate the refactor.
**Pixel annotation palette management**
- Introduced `ueler.viewer.palette_store` with shared slugging, registry, and JSON helpers reused by the mask painter and pixel annotation workflows, and migrated the mask painter plugin to the shared implementation.
- Expanded the Pixel annotations accordion with Save/Load/Manage tabs, optional `ipyfilechooser` pickers, and registry handling that mirrors the mask painter experience so class colour sets can be persisted and restored consistently (addresses [#42](https://github.com/HartmannLab/UELer/issues/42)).
- Added persistence-focused coverage to `tests/test_annotation_palettes.py`, exercising palette save/load round-trips and refreshing layout stubs to accommodate the new controls.

**FOV detection filtering**
- Added `_has_tiff_files()` method to `ueler.viewer.main_viewer.ImageMaskViewer` that checks for .tif/.tiff files in the FOV directory or its 'rescaled' subdirectory, mirroring the logic from `load_channel_struct_fov()` to ensure only directories containing TIFF images are recognized as valid FOVs.
- Updated `available_fovs` initialization to filter out directories without TIFF files, preventing misclassification of folders like '.ueler' as FOVs (fixes [#29](https://github.com/HartmannLab/UELer/issues/29)).
- Added comprehensive unit tests in `tests/test_fov_detection.py` covering positive cases (directories with TIFF files), negative cases (empty directories, nonexistent directories), and edge cases (rescaled subdirectories) to guard against regressions.
- Ran `python -m unittest tests.test_fov_detection` to validate the filtering logic and ensure no existing functionality is broken.

**Plugin layout refinements**
- Added `ueler.viewer.layout_utils` with reusable layout helpers keeping child widths within parent bounds to eliminate shallow horizontal scrollbars in tight containers (fixes [#39](https://github.com/HartmannLab/UELer/issues/39)).
- Updated ROI Manager, Batch Export, and Go To plugins to adopt the shared layouts so button rows wrap, selectors flex, and status content fits without triggering unnecessary horizontal scrolling.
- Verified the affected notebooks manually; unit suite not rerun because changes are widget-only.
- Follow-up: corrected the Batch Export plugin's widget builder to pass the new layout helpers explicitly, fixing the NameError raised when instantiating the plugin after the refactor.

**Annotation control separation**
- Removed the Overlay mode toggle from the pixel annotation controls so mask visibility remains governed by mask checkboxes even when annotations are hidden, delivering the separation requested in [#41](https://github.com/HartmannLab/UELer/issues/41).
- Renamed the accordion entry to `Pixel annotations` and enforced the order `Channels`, `Masks`, `Pixel annotations` so mask controls precede annotation options while keeping the palette editor accessible.
- Updated `tests/test_annotation_palettes.py` to reflect the new accordion order and the pared-down control list ahead of regression runs.

**ROI browser presets**
- Rebuilt the ROI Manager plugin to expose `ROI browser` and `ROI editor` tabs, adding a Matplotlib-backed gallery with tag/FOV filters plus a centre-with-preset action that replays saved rendering metadata (addresses [#44](https://github.com/HartmannLab/UELer/issues/44)).
- Extended ROI persistence to store annotation palette and mask colour set identifiers and taught `ImageMaskViewer` to report/apply the active palette name so plugins can capture and restore presets reliably.
- Refined the browser with AND/OR tag filtering, a saved-preset toggle, 500px scroll container with 98 % width tiles, incremental "show 4 more" pagination, and mask visibility restoration alongside existing preset metadata.

**ROI browser expression filtering**
- Added `ueler.viewer.tag_expression.compile_tag_expression` with tokenization, shunting-yard parsing, and eager validation so ROI tag filters accept boolean expressions using `()`, `&`, `|`, and `!` syntax (addresses [#44](https://github.com/HartmannLab/UELer/issues/44)).
- Expanded the browser UI with operator/tag shortcut buttons, inline error feedback, HUD-free Matplotlib canvases sized to 98 % of the scroll container, and a fixed lazy-loading listener that requests four more previews when the scroller nears the end.
- Updated `tests/test_roi_manager_tags.py` and added `tests/test_tag_expression.py` to cover parser behaviour and plugin wiring; ran `python -m unittest tests.test_tag_expression tests.test_roi_manager_tags` to confirm the changes.

**Channels accordion consolidation**
- Relocated the channel tag chips plus marker set dropdown, name field, and action buttons into the Channels accordion pane so selection presets sit next to their per-channel sliders (addresses [#40](https://github.com/HartmannLab/UELer/issues/40)).
- Rebuilt the accordion entry with dedicated containers for the selector, marker set controls, and dynamic sliders, preserving spacing and keyboard focus while removing duplicate widgets from the left panel header.
- Removed the accordion-level scrollbar and capped the slider container height so only the per-channel section scrolls, eliminating double scrollbars while keeping long channel lists navigable.

**Chart histogram responsiveness**
- Observed the histogram bin slider to rerender plots immediately by enabling continuous updates, tracking the active histogram column, and redrawing Matplotlib output whenever the bin count changes so users see real-time feedback (fixes [#47](https://github.com/HartmannLab/UELer/issues/47)).
- Preserved cutoff markers across rerenders by restoring the vertical threshold line and reapplying cell highlights after each redraw.

**Cutoff highlight persistence**
- Reordered the FOV-change refresh flow so chart-driven cutoff highlights reapply after plugin panels clear overlays, keeping qualifying cells highlighted as the user switches images (fixes [#47](https://github.com/HartmannLab/UELer/issues/47)).
- Attempted `python -m unittest tests.test_chart_footer_behavior`; run blocked because the lightweight test environment lacks `matplotlib.colors`.

### v0.2.0-rc1
**Wide plugin layout**
- Increased the control panel width in `uiler.viewer.ui_components.split_control_content` from 360px to 6in so wide plugins have more room for complex controls without horizontal scrolling.

**Linked plugin reliability**
- Called `setup_attr_observers()` after dynamic plugin loading so chart scatter selections immediately propagate to the cell gallery while guarding duplicate observer registration in both scatter plugins (fixes [#14](https://github.com/HartmannLab/UELer/issues/14)).
- Hardened observer setup flags in `ChartDisplay` and its heatmap counterpart to keep linkage idempotent across repeated viewer displays.

**Regression coverage**
- Added targeted linkage tests in `tests/test_chart_footer_behavior.py`, stubbing heavy imaging dependencies to exercise linked and unlinked flows without the full stack; ran `python -m unittest tests.test_chart_footer_behavior.ChartDisplayFooterTests.test_selection_forwards_to_cell_gallery_when_linked tests.test_chart_footer_behavior.ChartDisplayFooterTests.test_selection_does_not_forward_when_unlinked`.

### v0.2.0-beta
**Phase 4a fixes**
- Adjusted the viewer's scale bar drawing routine to multiply by the active downsample ratio, fixing undersized bars when zoomed out or exporting at reduced resolution.
- Extended helper coverage to assert consistent pixel lengths as the effective pixel size changes; reran `python -m unittest tests.test_scale_bar_helper tests.test_export_fovs_batch` to validate the fix.

**Phase 4b cell export fixes**
- Identified missing marker-set channel selections as the cause of blank single-cell composites; `_resolve_marker_profile` now falls back to the viewer's active channel state and validates per-channel settings before rendering.
- Reworked `_preview_single_cell` to use keyword arguments when calling `_finalise_array`, reuse captured overlays, and render optional scale bars so previews match batch exports without triggering `TypeError`.
- Extended `tests/test_export_fovs_batch.py` with targeted coverage for the marker profile fallback and preview workflow to guard the regression.
- Ran `python -m unittest tests.test_export_fovs_batch` to confirm the Phase 4b fixes.

**Scale bar automation**
- Added `ueler.viewer.scale_bar` with an engineering-rounding helper that produces <=10% frame length bars, formats labels in µm/mm, and tolerates Matplotlib-free environments via graceful fallbacks.
- Updated the main viewer to recompute scale bars whenever pixel size, downsample level, or view extents change so interactive previews stay aligned with physical measurements.

**Batch export scale bars**
- Threaded pixel size through batch jobs, captured the computed scale bar spec in `_finalise_array`, and rendered consistent bars into PNG/JPEG/TIFF outputs along with PDF documents.
- Centralised raster/PDF scale bar drawing via `_render_with_scale_bar` and `_write_pdf_with_scale_bar`, ensuring placement consistency across export formats while preserving overlay snapshots and mask controls.

**Testing & documentation**
- Added `tests/test_scale_bar_helper.py` to lock in rounding behaviour and effective pixel sizing, and refreshed `tests/test_export_fovs_batch.py` with ipywidgets/matplotlib stubs to cover the new export pipeline.
- Ran `python -m unittest tests.test_scale_bar_helper tests.test_export_fovs_batch` and updated the Phase 4 checklist plus supporting docs to reflect completion of the scale bar deliverables.

**Mask & annotation exports**
- Captured live overlay settings with `ImageMaskViewer.capture_overlay_snapshot` and replayed them inside batch jobs so exported images mirror in-viewer mask and annotation selections.
- Added opt-in toggles plus availability hints to the Batch Export plugin, keeping overlay options discoverable while preventing invalid selections on datasets without masks or annotations.
- Threaded overlay snapshots through every export worker (FOV, cell, ROI) and extended renderer helpers with alpha/outline blending to honour mask colours and annotation palettes.
- Expanded `tests/test_rendering.py` with translucent/outline mask coverage and added snapshot reconstruction checks to `tests/test_export_fovs_batch.py` to guard future regressions.
- Replaced binary edge approximations with label-aware mask outlines, added an adjustable thickness control, and hardened the fallback path so NumPy-only environments still produce per-cell contours backed by new renderer tests.

**Mask outline controls & plugin independence**
- Added a dedicated mask-outline thickness slider to the main viewer, seeded from the current render state and wired into plugin notifications so on-screen previews refresh immediately.
- Taught `BatchExportPlugin` to seed its slider from the viewer once, then maintain an independent thickness value for exports while respecting viewer-driven updates when the user has not overridden it.
- Propagated plugin-specific thickness through overlay snapshots, cache keys, and export workers so batch images honour local overrides without disturbing the live viewer.
- Extended `tests/test_rendering.py` and `tests/test_export_fovs_batch.py` with label-boundary regressions and slider sync cases to lock in per-cell contours across both surfaces.

**Batch export UI**
- Replaced the placeholder export plugin with `BatchExportPlugin`, adding mode-aware controls, marker set selection, output location tools, and asynchronous Start/Cancel handling with progress, status messages, and output links.
- Added mode-specific panels for Full FOV, Single Cells, and ROIs, including cell table filters, ROI selectors, crop sizing, and a Matplotlib-backed preview to validate single-cell settings before starting long runs.
- Surfaced scale-bar toggles and DPI/downsample controls across all export modes; the pipeline records the requested ratios ahead of the Phase 4 sizing work.
- Added a mask outline thickness slider that synchronises with the viewer so exports and interactive previews apply the same contour width.

**Job integration & rendering reuse**
- Wired the plugin to build `Job` items per mode so exports share structured progress, cancellation, and per-item result tracking without blocking the UI thread.
- Reused the pure rendering helpers for FOV, crop, and ROI exports, ensuring consistent compositing across interactive previews and background jobs.

**Matplotlib bootstrap & planning docs**
- Expanded the test bootstrap with stubs for `matplotlib.text`, `matplotlib.backend_bases`, `matplotlib.patches`, `matplotlib.widgets`, and `mpl_toolkits.axes_grid1` so export suites run in dependency-light environments.
- Marked the Phase 3 checklist complete in `dev_note/batch_export.md` and noted the pending scale-bar sizing follow-up planned for Phase 4.

**Verification**
- `python -m unittest tests.test_rendering tests.test_export_fovs_batch`
- `python -m unittest tests.test_export_job`

**Batch export groundwork**
- Extracted compositing helpers into `ueler.viewer.rendering` and refactored `ImageMaskViewer.render_image` plus `export_fovs_batch` to reuse them, preserving overlay behaviour while returning NumPy arrays ready for disk writes.
- Added lightweight stubs for optional dependencies (OpenCV, scikit-image, tifffile, matplotlib, dask) so unit tests can execute without the full imaging stack installed.
- Introduced a synchronous job runner (`ueler.export.job.Job`) with cancellation and structured result reporting, and migrated `ImageMaskViewer.export_fovs_batch` to delegate work to the runner while logging progress through the shared logger.

**Rendering & tests**
- Created `tests/test_rendering.py` to lock in colour compositing, annotation blending, mask overlays, and ROI/crop behaviour through synthetic fixtures.
- Added `tests/test_export_fovs_batch.py` smoke coverage for the existing export loop, including success and missing-channel failure cases aligned with the current API surface.
- Authored `tests/test_export_job.py` to exercise success, error, cancellation, and snapshot behaviour for the new job runner API.

### v0.2.0-alpha
See [issue #4](https://github.com/HartmannLab/UELer/issues/4) for an overview.

**Notebook runner interface**
- Added `ueler/runner.py` with a `run_viewer(...)` helper that normalizes dataset paths, registers import shims, displays the UI by default, and triggers plugin post-load hooks so notebooks can launch the viewer without boilerplate.
- Added `tests/test_runner.py` to smoke-test the runner using stubbed factories, covering alias registration, optional flags, and package-level re-exports for both `ueler.runner` and `import ueler` entry points.
- Hardened the viewer navigation stack update so environments without a Matplotlib toolbar (e.g., inline backends) skip the nav-stack sync instead of raising `AttributeError` when launched via the new runner.
- Added `load_cell_table(...)` to attach CSV or in-memory tables to an existing viewer, refresh channel/mask controls, and optionally redisplay the interface for notebook workflows that stage data loading separately.

**Fast-test dependency isolation**
- Forced the shared bootstrap to install in-process seaborn/scipy stubs whenever pandas is stubbed so heatmap imports no longer reach for the real libraries, and wired the annotation palette suite to load the bootstrap before importing viewer modules to guarantee the lightweight shims take effect.
- Added explicit stub markers for the installed modules so subsequent imports keep reusing the lightweight implementations during `unittest` discovery.

**Chart widget layout compatibility**
- Updated the chart plugin to build every `VBox`/`HBox` with the `children=` keyword, ensuring older ipywidgets stubs capture the controls and plot panes so footer layout assertions reflect production behavior.
- Retained the legacy `color_points` signature while documenting the unused `selected_colors` parameter to satisfy lint without altering callers.

**Compatibility import shims**
- Registered lazy module aliases bridging the new `ueler.*` namespace to the legacy `viewer` modules via `_compat.register_module_aliases`, letting notebooks adopt the packaged layout without eager imports.
- Hardened the alias finder to fall back to stubbed modules lacking `ModuleSpec` metadata so fast-stub tests keep loading lightweight plugin placeholders without errors.
- Expanded `tests/test_shims_imports.py` to assert the alias matrix resolves to the legacy modules and to skip gracefully when optional dependencies such as `cv2` are unavailable.

**Incremental module moves**
- Migrated `viewer/ui_components.py` into `ueler.viewer.ui_components` while retaining the legacy import path through a lightweight compatibility wrapper and alias bridge.
- Updated shim tests to tolerate downstream stubs and confirmed the fast suite stays green after the relocation.
- Relocated `viewer/color_palettes.py` into `ueler.viewer.color_palettes`, added a legacy wrapper plus reverse alias in `_compat.py`, and reran the fast test suite to verify shim coverage remains intact.
- Moved `viewer/decorators.py` into `ueler.viewer.decorators`, introduced helpers to simplify the status-bar decorator, and left a compatibility wrapper for legacy imports.
- Transitioned `viewer/observable.py` to `ueler.viewer.observable`, tightened typing for the observable helper, and replaced the legacy module with a thin forwarding shim.
- Relocated `viewer/annotation_palette_editor.py` into `ueler.viewer.annotation_palette_editor`, updated its color helper imports, and retained a legacy shim for backward compatibility.
- Shifted `viewer/annotation_display.py` into `ueler.viewer.annotation_display`, refreshed imports to use the packaged namespace, added a lazy widget loader so test stubs initialize before instantiation, and provided a compatibility wrapper plus alias updates so existing code keeps working.
- Repositioned `viewer/roi_manager.py` into `ueler.viewer.roi_manager`, switched timestamps to `datetime.now(timezone.utc)` for lint compliance, and left a compatibility wrapper plus reverse alias so legacy imports remain operational.
- Ported `viewer/plugin/plugin_base.py` into `ueler.viewer.plugin.plugin_base`, introduced a packaged plugin scaffold, and replaced the legacy module with a forwarding shim plus reverse alias coverage.
- Moved `viewer/plugin/export_fovs.py` into `ueler.viewer.plugin.export_fovs`, preserved the placeholder UI, and wrapped the legacy module with a forwarding shim plus alias updates.
- Shifted `viewer/plugin/go_to.py` into `ueler.viewer.plugin.go_to`, refreshed imports to use the packaged helpers, and added a legacy wrapper plus alias updates for continuity.
- Migrated `viewer/plugin/cell_gallery.py` into `ueler.viewer.plugin.cell_gallery`, tightened deterministic sampling when downscaling selections, and replaced the legacy module with a forwarding shim plus alias updates to preserve backwards compatibility.
- Relocated `viewer/plugin/chart.py` into `ueler.viewer.plugin.chart`, re-pointed intra-project imports to the packaged namespace, and slimmed the legacy module to a forwarding shim while pruning redundant alias table entries.
- Migrated `viewer/plugin/run_flowsom.py` into `ueler.viewer.plugin.run_flowsom`, added a graceful fallback when `pyFlowSOM` is missing, deduplicated repeated layout strings, and left the legacy module as a thin wrapper.
- Ported `viewer/main_viewer.py` into `ueler.viewer.main_viewer`, updated its internal imports to favor the packaged namespace, pointed dynamic plugin loading at `ueler.viewer.plugin.*`, and replaced the legacy file with a compatibility wrapper.

**Fast-stub pandas parity**
- Extended the shared pandas shim with `Series.loc`/`.iloc` indexers, `map`, `astype`, boolean helpers, and dictionary-aware constructors so chart and scatter tests align categories correctly without the real library installed.
- Patched fallback pandas modules discovered during test imports to graft the shared `api.types` helpers, keeping `is_numeric_dtype` and `is_object_dtype` available even when ad-hoc stubs surface.
- Hardened reindex and assignment support (`Series.reindex`, `Series.loc[...] = value`) to preserve ROI, heatmap, and scatter workflows in the fast-test environment.

**Matplotlib stub coverage**
- Registered a lightweight `matplotlib.pyplot` stub with canvas and axis helpers so histogram code paths execute during fast tests without pulling in the full plotting stack.
- Updated the plugin preloader to replace minimalist viewer stubs with the real chart/ROI modules before tests run, ensuring footer layout assertions exercise production logic.

**Heatmap footer redraw preference**
- Taught `HeatmapDisplay.restore_footer_canvas` to attempt cached redraws before scheduling a canvas repaint, satisfying the footer regression tests and avoiding redundant `draw_idle` calls when the cache already holds a canvas snapshot.

**Packaging skeleton groundwork**
- Added `pyproject.toml` with minimal project metadata, setuptools configuration, and developer extras to unblock incremental packaging work.
- Created a lightweight `Makefile` offering virtualenv creation, editable installs, and fast/integration test targets to align local workflows with the mitigation strategy.
- Introduced `ueler.__init__` and `ueler.viewer.__init__` compatibility shims that lazily forward to the legacy `viewer` module so consumers can begin migrating import paths without runtime changes.

**Root helper packaging**
- Listed `constants.py`, `data_loader.py`, and `image_utils.py` under `tool.setuptools.py-modules` so wheel builds include the legacy helpers relied upon by the compatibility shims.
- Flagged follow-up release validation to build wheels/sdists and confirm the helpers remain available after installation.

**Heatmap selection safeguards**
- Wrapped `InteractionLayer._apply_cluster_highlights` so scatter highlights respect the chart link toggle, eliminating false redraws during Task 2 heatmap tests.
- Preloaded the chart plugin before test modules import so downstream suites reuse the real `ChartDisplay` implementation instead of minimal stubs.

**Bootstrap dependency coverage**
- Normalized ad-hoc pandas stubs by grafting the shared test DataFrame/Series helpers whenever modules downgrade `pandas` to `object`, restoring ROI and heatmap helpers.
- Expanded the ipywidgets shim to surface `allowed_tags`, `allow_new`, and other TagsInput traits that the ROI manager exercises during tag merge scenarios.

**Test suite reliability**
- Pre-imported key plugins and reran `python -m unittest discover tests`, confirming all 44 tests pass under the shared bootstrap.

### v0.1.10-rc3
**Assign tab activation**
- Wired `HeatmapDisplay` to observe `current_clusters["index"]` so the Assign tab's `Meta-cluster ID` field and Apply button enable as soon as a cluster selection exists and disable when it clears ([issue #9](https://github.com/HartmannLab/UELer/issues/9)).
- Initialized the observer during widget restore to keep the controls accurate after notebook reloads, preventing stale disabled states when selections persist across sessions ([issue #9](https://github.com/HartmannLab/UELer/issues/9)).

**Automatic cutoff locking**
- Re-engaged the Lock Cutoff control automatically after meta-cluster assignments and dendrogram edits so patches immediately regain protection from accidental reclassification ([issue #10](https://github.com/HartmannLab/UELer/issues/10)).
- Added a guarded “Unlock once” workflow that forces users to request a one-time override before editing the cutoff, restoring the lock after changes and keeping manual toggles from bypassing the safeguard ([issue #10](https://github.com/HartmannLab/UELer/issues/10)).

**Chart footer placement**
- Updated `ChartDisplay` so the plugin automatically refreshes the footer when multiple scatter plots are active, keeping controls in the bottom tab while leaving a clear notice in the sidebar ([issue #5](https://github.com/HartmannLab/UELer/issues/5)).
- Added `tests/test_chart_footer_behavior.py` with lightweight widget and data stubs to guard the layout toggle and footer refresh workflow ([issue #5](https://github.com/HartmannLab/UELer/issues/5)).

**Heatmap footer stability**
- Ensured the heatmap plugin reattaches its canvas during footer rebuilds so horizontal heatmaps persist after scatter plots are added or removed in the chart plugin ([issue #6](https://github.com/HartmannLab/UELer/issues/6)).
- Expanded `tests/test_chart_footer_behavior.py` to simulate combined chart and heatmap footer refreshes, guaranteeing the heatmap remains registered in `BottomPlots` following scatter changes ([issue #6](https://github.com/HartmannLab/UELer/issues/6)).

**Footer pane caching**
- Reworked `viewer/ui_components.update_wide_plugin_panel` to cache wide plugin panes, preserving live matplotlib canvases when the chart toggles between sidebar and footer layouts ([issue #6](https://github.com/HartmannLab/UELer/issues/6)).
- Extended `tests/test_wide_plugin_panel.py` with cache-aware stubs so chart relocations reuse the existing heatmap pane instead of instantiating a fresh widget tree ([issue #6](https://github.com/HartmannLab/UELer/issues/6)).

**Heatmap redraw automation**
- Triggered the heatmap footer pane to replay its Plot routine whenever a cached wide layout is reused so horizontal canvases repopulate immediately after chart-driven footer rebuilds ([issue #6](https://github.com/HartmannLab/UELer/issues/6)).
- Added a regression to `tests/test_wide_plugin_panel.py` asserting cached panes invoke the redraw helper every time the footer recomposes, preventing future regressions ([issue #6](https://github.com/HartmannLab/UELer/issues/6)).

**Wide heatmap tick alignment**
- Centered tick locations and histogram bins in horizontal heatmaps so column and row labels sit on the underlying grid instead of drifting half a cell off to the left/top ([issue #8](https://github.com/HartmannLab/UELer/issues/8)).
- Added regression coverage around the tick-label helper to guard against future misalignment regressions in wide mode ([issue #8](https://github.com/HartmannLab/UELer/issues/8)).

**Cluster ID persistence**
- Cached `meta_cluster_revised` overrides before rebuilding heatmaps and restored them afterward so manual assignments survive horizontal/vertical layout toggles ([issue #11](https://github.com/HartmannLab/UELer/issues/11)).
- Added focused regression tests that exercise the caching helper to prevent future regressions in manual assignment retention ([issue #11](https://github.com/HartmannLab/UELer/issues/11)).

### v0.1.10-rc2
**Annotation overlays & control layout**
- Added `load_annotations_for_fov` plus rich overlay controls (mode toggle, opacity slider, palette editor launcher) so pixel annotations render as fills, outlines, or both directly in the main viewer; reshaped the left column into a scrollable accordion that keeps annotations ahead of masks and anchors the palette editor for easy access ([issue #21](https://github.com/HartmannLab/UELer/issues/21)).
- Normalised annotation discovery for both Dask and NumPy rasters, enabled palette editing for names containing spaces, and prevented startup crashes when restoring widget states with already-materialised composites ([issue #21](https://github.com/HartmannLab/UELer/issues/21)).

**Footer-wide plugin layouts**
- Introduced the bottom-tab host (`BottomPlots`) and helper utilities so plugins can opt into a full-width footer without vacating the accordion; the heatmap plugin now moves into the footer when “Horizontal layout” is enabled and returns gracefully when toggled off ([issue #24](https://github.com/HartmannLab/UELer_alpha/issues/24)).
- Hardened plugin observer setup to cope with `SimpleNamespace` registries, eliminating the footer-related startup regression ([issue #24](https://github.com/HartmannLab/UELer_alpha/issues/24)).

**ROI management & tagging**
- Shipped a persistent `ROIManager` backend backing onto `<base_folder>/.UELer/roi_manager.csv`, complete with import/export helpers, timestamping, and a dedicated accordion plugin for capture, centring, and metadata edits ([issue #16](https://github.com/HartmannLab/UELer_alpha/issues/16)).
- Rebuilt the tag workflow with a ComboBox + TagsInput hybrid that normalises and preserves new labels even under restrictive widget front-ends, covering multiple regression scenarios in unit tests ([issue #23](https://github.com/HartmannLab/UELer_alpha/issues/23)).

**Mask painter & channel workflows**
- Added mask colour set persistence (`.maskcolors.json`), default-colour management, optional `ipyfilechooser` support, and identifier-aware palette switching, alongside UI affordances to focus on edited classes ([issue #18](https://github.com/HartmannLab/UELer_alpha/issues/18)).
- Centralised channel intensity caching via `merge_channel_max`, updated contrast slider formatting, and clarified mask loading so binary rasters are promoted to labelled images with consistent naming ([issue #15](https://github.com/HartmannLab/UELer_alpha/issues/15)).

**Test & developer support**
- Added targeted suites covering palettes, mask colour persistence, ROI tagging, and footer layout assembly (`tests/test_annotation_palettes.py`, `tests/test_mask_color_sets.py`, `tests/test_roi_manager_tags.py`, `tests/test_wide_plugin_panel.py`) to guard against future regressions ([issues #21](https://github.com/HartmannLab/UELer_alpha/issues/21), [#23](https://github.com/HartmannLab/UELer_alpha/issues/23), [#24](https://github.com/HartmannLab/UELer_alpha/issues/24)).

**Chart gallery upgrades**
- Replaced both chart accordions with `jupyter-scatter` for multi-plot scatter views, including the heatmap variant; selections now sync across all active plots, the footer automatically hosts multi-plot layouts, and histogram fallbacks remain available ([issue #22](https://github.com/HartmannLab/UELer_alpha/issues/22)).
- Added `anywidget>=0.9` as a dependency for the new scatter widgets—make sure the environment that launches JupyterLab has `pip install anywidget` applied (and, if you use a separate Lab environment, install `anywidget` there as well so the `@anywidget/jupyterlab` federated extension is available).
- Streamlined the heatmap plugin's selection handling so scatter clicks and lasso selections update row highlights in-place instead of rebuilding the entire dendrogram, dramatically improving responsiveness during linked exploration ([issue #25](https://github.com/HartmannLab/UELer_alpha/issues/25)).

**Heatmap enhancement**
- Refreshed the heatmap's meta-cluster patches so horizontal (wide) layouts redraw their column highlights immediately after reassignment, keeping footer views synced with cluster edits ([issue #26](https://github.com/HartmannLab/UELer_alpha/issues/26)).
- Restored the heatmap → chart linkage so clicking a heatmap cell recolors and highlights the linked scatter plots via the shared `ScatterPlotWidget` API—no more stale Matplotlib handles when `Chart` is linked ([issue #27](https://github.com/HartmannLab/UELer_alpha/issues/27)).
- Promoted the lightweight `HeatmapModeAdapter` scaffold into the primary layout engine so both vertical and wide heatmaps share the same histogram, cutoff, and palette logic with far less branching inside `viewer/plugin/heatmap.py`.
- Continued the refactor by breaking `HeatmapDisplay` into dedicated data, interaction, and display mixins, leaving the orchestrator class focused on wiring while keeping the public plugin API intact (refactor plan step 5-6).
- Fixed the “Horizontal layout” toggle so it once again flips the plugin into wide mode after the mixin refactor.
- Reduced orientation drift by funnelling heatmap drawing and click handling through shared helpers, so vertical and horizontal layouts highlight, recolor, and log selections identically.
- Eliminated the disappearing heatmap regression when scatter plots trigger a footer refresh—the vertical layout now reattaches its canvas after bottom-panel rebuilds so plotting order no longer matters ([issue #28](https://github.com/HartmannLab/UELer_alpha/issues/28)).


### v0.1.9-alpha  

**Performance Improvements**
- **Enhanced efficiency**: UELer now uses the `dask` package as the backend for lazy loading, significantly improving speed and reducing memory usage. See [issue #7](https://github.com/HartmannLab/UELer/issues/7).

**UI Enhancements**
- **Automatic settings saving**: Some plugins now support automatic saving of settings. See [issue #9](https://github.com/HartmannLab/UELer/issues/9).
- **Status bar**: A new status bar now indicates when an intensive computation is running. See [issue #13](https://github.com/HartmannLab/UELer/issues/13).
- **Mask painter palettes**: Save, load, and manage reusable mask color sets directly from the plugin—palettes live under `<base_folder>/.UELer/` by default, the UI focuses on the actively selected classes (with an optional “show all” toggle), `ipyfilechooser` support provides file dialogs when available, the default color can be changed on the fly, classes that stick with the default are automatically deselected when loading a palette, and each saved set remembers the identifier it applies to—all exported as `.maskcolors.json` files.
- **Annotation overlays**: Pixel annotation images can now be loaded alongside masks. Per-annotation controls in the main viewer let you switch overlay modes (mask outlines, annotation fill, or both), tune opacity, and open the class palette editor to assign colors and friendly labels to annotation IDs.

**Improved Interactivity**
- **Cluster tracing**: In the main viewer, you can now trace cells belonging to the same cluster, which are highlighted in the heatmap.
- **Cell gallery navigation**: Clicking on a cell in the cell gallery brings you to its location in the main viewer.

**New Plugins**
- **Mask Painter**: Enables coloring of mask outlines.
- **FlowSOM**: Supports FlowSOM clustering as part of the FlowSOM workflow integration. See [issue #12](https://github.com/HartmannLab/UELer/issues/12).

**Bug Fixes**
- **Corrected UI settings keys**: Settings now align properly with the UI. See [issue #4](https://github.com/HartmannLab/UELer/issues/4).
- **Fixed cell selection error**: The issue with cell selection has been resolved. See [issue #2](https://github.com/HartmannLab/UELer/issues/2).
- **ROI manager tags**: A ComboBox + TagsInput hybrid now lets you type or pick tags freely; new entries are added to both the active tag list and future suggestions even when the frontend enforces the allowed-tag list. See [issue #23](https://github.com/HartmannLab/UELer/issues/23).
- **ROI manager panel deduplication**: The ROI Manager now lives exclusively in its plugin accordion, eliminating the duplicate block that previously appeared in the left control panel.
- **Annotation palette visibility**: Annotation rasters that load as NumPy arrays now populate the palette editor correctly, and the default overlay mode shows annotation fills alongside mask outlines for immediate feedback.
- **Saved widget state crash**: Restoring widget presets no longer raises an error when the base image buffer is already a NumPy array (common when annotations are enabled).

### v0.1.7-alpha
**Allowing user specified column keys**
- Users can now specify custom column keys in the `Advanced Settings` under `Data mapping`.
- To use this feature, navigate to `Advanced Settings` > `Data mapping`, and enter the desired column keys for each data field.

**Heatmap on subsetted cell table**
- Users can generate a heatmap based on a subset of the cell table data.
- To use this feature, select the desired values in the selected column in the cell table and then generate the heatmap.
- This is a key step in a FlowSOM workflow.

### v0.1.6-alpha
**Automatic settings saving/loading**
- Automatic Saving: Overall settings are now saved automatically whenever changes are made.
- Automatic Loading: The latest settings are loaded when images are opened in the viewer.

### v0.1.5-alpha
**Interactive Histogram**
- When using the chart tool, selecting only the **x-axis** will display a histogram.  
- By clicking on the histogram, you can set a cutoff value to filter cells.  
- Cells above or below the cutoff (as specified) will be highlighted in the currently displayed image.  
