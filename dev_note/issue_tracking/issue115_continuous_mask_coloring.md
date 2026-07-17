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
