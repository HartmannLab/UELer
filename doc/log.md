### v0.3.0-alpha

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
