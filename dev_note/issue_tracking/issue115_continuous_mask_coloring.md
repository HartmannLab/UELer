# Issue #115 — Color cell masks by a continuous variable

## Problem

The Mask Painter plugin (`ueler/viewer/plugin/mask_painter.py`) colored cell masks only by a
**discrete/categorical** cell-table column: each unique value in the identifier column became a row with its
own `ColorPicker`. Float columns were deliberately excluded from the identifier dropdown, so continuous data
(protein abundance, feature values, …) could not be visualized as a gradient. Users asked to select a numeric
column and apply a **color gradient** to the masks, with customization for colormap, value range, transparency,
and an arcsinh transform.

## Scope

- New **Continuous** coloring mode alongside the existing **Categorical** mode, toggled in the painter UI.
- Customization: matplotlib colormap, value range (auto 1–99 percentile or manual vmin/vmax), transparency
  (opacity %), fill/outline, and an optional `arcsinh(value / cofactor)` transform.
- Colors are computed **globally across all FOVs** (range resolved once over the whole column) so a value maps to
  the same color in every FOV, the cell gallery, and exports.
- Continuous coloring flows through the same per-cell registry / rendering / snapshot machinery as categorical
  coloring, so the cell gallery, batch export, and ROI replay all work with no per-consumer changes.
- Continuous settings persist through the existing `.maskcolors.json` palette Save/Load/Manage tab.

## Approach

The single choke-point `build_painter_state_maps_for_fov` (module-level in `mask_painter.py`) is used both by live
rendering (via the `get_effective_*_map_for_fov` getters) and by every snapshot consumer (via
`MainViewer.resolve_mask_painter_snapshot_for_fov`). Teaching that function a **continuous branch** and carrying
continuous parameters through `MaskPainterSnapshot` gives all consumers gradient coloring for free. The per-cell
color registry (`set_cell_colors_bulk`) and overlay (`apply_registry_colors`) already store one hex per cell, so
continuous mode is just a new way to compute those hexes — no structural change to the apply/render path, and no
new per-render cost versus categorical coloring with many classes.

## Implementation

- **Helpers (module level):**
  - `resolve_continuous_range(values, arcsinh, cofactor, lo_pct=1, hi_pct=99)` — percentile range in the
    (optionally arcsinh-transformed) value space, computed once over the whole column. Guards all-NaN → `(0,1)`
    and constant/degenerate columns (widened).
  - `compute_continuous_colors(values, mask_ids, *, colormap, vmin, vmax, arcsinh, cofactor)` — vectorized
    `cmap(Normalize(clip=True)(v))` → `{mask_id: hex}` (via `matplotlib.colors.to_hex`). NaN/non-finite rows are
    skipped so those cells fall through to the base mask.
  - `_get_colormap` (version-compatible), `_arcsinh_transform`, `_safe_float`.
- **`build_painter_state_maps_for_fov`** gains an optional `continuous` spec; when set it returns
  `color_map`/`mode_map`/`opacity_map` from the continuous computation and ignores the categorical kwargs.
- **UI (`UiComponent`):** a `ToggleButtons` mode selector plus a `continuous_layout` (value-column dropdown,
  colormap dropdown, arcsinh + cofactor, auto/manual vmin/vmax + recompute button, opacity, fill, colorbar
  `Output`). The existing categorical controls are grouped into `categorical_layout`; the two are shown/hidden via
  `.layout.display`. `_initialise_continuous_options` populates the value dropdown from the float columns.
- **Handlers:** `_on_color_mode_change`, `_on_continuous_param_change`, `_on_continuous_column_change`,
  `_on_auto_range_toggle`, `_on_continuous_range_edit`, `_recompute_continuous_range`, `_render_colorbar`,
  `_build_continuous_spec` / `_active_continuous_spec`.
- **Apply path:** `apply_colors_to_masks` branches to `_apply_continuous_colors_to_masks`, which registers colors
  globally with one `groupby(fov)` + one `set_cell_colors_bulk`, gated behind `register_globally`.
- **Snapshot & consumers:** `MaskPainterSnapshot` gains defaulted continuous fields (`color_mode`,
  `continuous_column`, `colormap`, `vmin`, `vmax`, `arcsinh`, `arcsinh_cofactor`, `continuous_opacity`,
  `continuous_fill`). `capture_snapshot`/`apply_snapshot` handle continuous mode (relaxed the per-class guard);
  `resolve_mask_painter_snapshot_for_fov` passes the continuous spec; `roi_manager_plugin._resolve_mask_painter_snapshot`
  deserializes the new fields (backward compatible via `.get`).
- **Palette persistence:** `_build_color_set_payload` writes a `color_mode` + `continuous` block (`COLOR_SET_VERSION`
  bumped to `1.2.0`); `_load_color_set` routes continuous payloads to `_apply_continuous_payload`; save guards
  accept a continuous value column.

## Files changed

- `ueler/viewer/plugin/mask_painter.py` — helpers, continuous state-map branch, UI, handlers, apply path,
  snapshot capture/apply, palette payload/load.
- `ueler/rendering/engine.py` — continuous fields on `MaskPainterSnapshot`.
- `ueler/viewer/main_viewer.py` — continuous branch in `resolve_mask_painter_snapshot_for_fov`.
- `ueler/viewer/plugin/roi_manager_plugin.py` — deserialize continuous snapshot fields.
- `tests/test_mask_painter_continuous.py` — new test module.
- `tests/bootstrap.py` — stub `BoundedFloatText`/`FloatText` so the test harness doesn't feed a stub `Layout` to a
  real widget.
- `tests/test_mask_painter_mode_visibility.py` — layout test updated for the new `categorical_layout` nesting.

## Validation

### Automated
- New `tests/test_mask_painter_continuous.py` (16 tests): colormap mapping (endpoints, hex format, clipping),
  NaN skipping, all-NaN empty, arcsinh with negatives, range resolution (constant/all-NaN/percentile), the
  continuous state-map branch (fill + outline), snapshot field round-trip, replay determinism vs. direct compute,
  float-only value options, layout toggle, global registration, and palette payload round-trip.
- Run: `python -m unittest discover tests`. Full suite failure/error set **identical to the `develop` baseline**
  (no new failures/errors); +16 net passing tests. Regression suites re-checked:
  `test_mask_painter_mode_visibility`, `test_mask_color_overlay`, `test_mask_color_sets`, `test_cell_gallery`,
  `test_export_fovs_batch`, `test_roi_manager_tags`.

### Manual (to confirm in a notebook)
Load a dataset with a float column, switch the painter to **Continuous**, pick the column + colormap, toggle
arcsinh, adjust the range; confirm the gradient renders, the colorbar updates, colors are consistent across FOVs,
and the cell gallery / a batch export / an ROI reflect the gradient. Save a continuous palette, reload it, confirm
settings restore.

## Risks / edge cases

- All-NaN column → finite fallback range, empty color dict (no crash), user-facing warning logged.
- Constant column → widened range (no zero-width `Normalize`).
- Negative values + arcsinh → fine; range computed in transformed space; cofactor bounded > 0.
- Large cell counts → vectorized compute + single bulk write; render cost equals the existing many-class cost.
- Old ROI/palette payloads without `color_mode` → default to `categorical` (unchanged behavior).

---

## Follow-up: reply 1 — continuous coloring is very slow + no busy indicator

### Problem
Pressing **Apply** in continuous mode took >2 minutes even when the current FOV had only ~150 cells; zoom/pan was
also slow; and the busy indicator never turned on (the developer noted the slow work happened *before* the busy
state was set — a strong clue that the cost was in the synchronous Apply handler, ahead of `update_display`).

### Root cause
1. **Redundant global registration (dominant cost).** `_apply_continuous_colors_to_masks` registered a color for
   **every cell in every FOV** (`groupby(fov)` + per-FOV `compute_continuous_colors` + `set_cell_colors_bulk`) on
   Apply, synchronously before `update_display`. A trace showed this registry is unnecessary: the live display and
   map overlay (`get_effective_state_maps_for_fov`), and the cell gallery / batch export / ROI thumbnails
   (`resolve_mask_painter_snapshot_for_fov`) all resolve continuous colors **on demand** via
   `build_painter_state_maps_for_fov`.
2. **Per-cell `to_hex`.** `compute_continuous_colors` converted colors with a per-cell matplotlib `to_hex` call.
3. **Full-column percentile + widget writes on the render hot path.** `_build_continuous_spec` recomputed the
   global range over the whole column *and* wrote the vmin/vmax ipywidgets (comm round-trips) on every state-map
   build — i.e. on every render/zoom.
4. **Busy state never set** by the Mask Painter.

### Fix
1. **Dropped the global registration.** `_apply_continuous_colors_to_masks` now only `clear_cell_colors()` (so
   stale categorical colors can't bleed through the gallery's per-cell fallback), invalidates the state-maps cache,
   redraws the colorbar, notifies the gallery, and refreshes the current FOV. Apply is now O(current FOV).
2. **Vectorized `compute_continuous_colors`** — sample the colormap for finite cells and build hex strings
   vectorized (no per-cell `to_hex`).
3. **Cached the auto-range** in `self._continuous_range_cache` keyed by `(column, arcsinh, cofactor)`;
   `_build_continuous_spec` reads the cache and writes no widgets. The range is recomputed and the fields updated
   only in the param-change handlers / recompute button (`_refresh_auto_range_fields`). Cache cleared on
   `on_cell_table_change`.
4. **Busy indicator** — decorated `apply_colors_to_masks` with `@update_status_bar`.

### Files changed
- `ueler/viewer/plugin/mask_painter.py` — all four fixes.
- `tests/test_mask_painter_continuous.py` — replaced the old "registers globally" test with: no global-registry
  writes on apply, on-demand color resolution, auto-range caching off the hot path, and busy-state toggling.

### Validation
`python -m unittest tests.test_mask_painter_continuous` — 19 tests pass. Full suite failure/error set identical to
the `develop` baseline (no new failures). Manual: Apply now renders in well under a second, zoom/pan is smooth, the
busy gif shows during Apply, and the gradient still appears in the cell gallery, batch export, and saved ROIs.

### Follow-up: reply 1 (part 2) — still slow + colorbar crash

After removing the global registration, Apply was faster but **still slow**, and rendering could raise a
`RecursionError` from `_render_colorbar`. Two further root causes:

1. **Colorbar created a live matplotlib figure.** `_render_colorbar` used `plt.subplots()`, which under the
   notebook's `%matplotlib widget` (ipympl) backend builds an interactive canvas **widget** — slow, and it crashed
   with `RecursionError: maximum recursion depth exceeded` during `Canvas`/`Layout` construction. **Fix:** render
   with a detached Agg `Figure` + `FigureCanvasAgg`, `savefig` to PNG bytes, and show a static `IPython.display.Image`
   in the `colorbar_output` — no interactive canvas, backend-independent.

2. **Mask recolor was O(cells × pixels).** Painter colors are painted by
   `mask_color_overlay._apply_region_colors`, which looped over every distinct cell doing a full-region
   `region_array == id` scan (and a per-cell `find_boundaries`). For continuous coloring every cell is colored, so
   this dominated. Unlike the annotation overlay, which already uses the fast vectorized LUT
   (`engine._apply_annotation_overlay`: `colormap[label_image]`). **Fix:** added a vectorized **fill fast path** —
   when all colored cells are `fill` mode with alpha > 0 and no borders (the default continuous case), build a
   per-id RGB+alpha lookup table and blend the whole region in one gather (`lut[region_array]`), O(pixels). The
   per-cell loop is retained as a fallback for outlines/borders/mixed modes, so categorical behavior (and its tests)
   is unchanged.

**Files changed:** `ueler/viewer/plugin/mask_painter.py` (`_render_colorbar`), `ueler/viewer/mask_color_overlay.py`
(`_apply_region_colors` fast-path guard + `_apply_region_colors_fill_vectorized`).
**Tests:** `tests/test_mask_color_overlay.py` — added vectorized-fill correctness (distinct colors, no
cross-contamination; background/unregistered ids untouched). Full suite failure/error set identical to the
`develop` baseline.
