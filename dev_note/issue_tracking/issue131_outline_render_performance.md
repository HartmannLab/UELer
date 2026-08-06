# Issue #131: Mask painter outline rendering is O(cells × pixels)

## Problem

Rendering with the mask painter's `Fill (unchecked = outline)` checkbox **unchecked** is
dramatically slower than with it checked, and the gap widens with the number of painted cells.

Measured through the real `apply_registry_colors(...)` on a synthetic 1024×1024 region:

| painted cells | Fill checked | Fill unchecked | ratio |
| --- | --- | --- | --- |
| 200 | 152 ms | 4.7 s | 31× |
| 1000 | 132 ms | 23.6 s | 179× |
| 5000 | 167 ms | 116.9 s | 701× |

Fill is flat in cell count; outline is linear, ≈23 ms **per cell per render**.
cProfile at 1000 cells: 20.7 s of 22.6 s (92%) inside `find_boundaries`, i.e. 2000
`scipy.ndimage.min_or_max_filter` calls at ~10 ms each.

## Root cause

The two modes are different algorithms, not variants of one.

- `_can_vectorize_fill` (`mask_color_overlay.py:191`) gates a vectorised fast path and
  **fails closed**: it returns `False` if *any* colored cell has `mode != "fill"`. Unchecking
  `Fill` for a single class therefore drops **every** cell in the FOV into the slow loop.
- Fill → `_apply_region_colors_fill_vectorized` (L218): one per-id colour LUT, one gather plus
  one blend over the region. Independent of cell count.
- Outline → the per-cell loop (L284–324). Per cell: a full-region `region_array == id`, two
  `np.any`, `find_boundaries(mask_bool)` (a grey erosion *and* dilation across the **whole**
  region to outline one ~1000-px cell), an optional `thicken_outline`, and a full-region
  boolean-index write (L326–327).

Two secondary defects in the same loop:

- `pending_edges` holds one full-region bool array **per cell** — ~1 GB at 1000 cells × 1 MP,
  ~5 GB at 5000.
- `np.unique(region_array)` is computed twice (`_region_colored_ids` L177,
  `_iter_mask_region_ids` L334).

## Key insight

`find_boundaries(label_image, mode="inner")` is **exactly equivalent** to the per-cell calls,
computed once for every label simultaneously. skimage's `inner` mode is
`(dilation(img) != erosion(img)) & (img != background)`, so:

- per cell: `find_boundaries(region == A, "inner")` = pixels of A adjacent to non-A
- batched: `find_boundaries(region, "inner") & (region == A)` = pixels of A adjacent to a
  *different label* = pixels of A adjacent to non-A

Identical, including at array borders and for excluded/uncoloured neighbours (both treat them
as "not A"). `ueler/rendering/engine.py:346–361` already renders whole-mask outlines this way;
the painter overlay simply never adopted it.

## Approach

Rewrite `_apply_region_colors(...)` so **every** path is O(pixels), removing the fill/outline
fork entirely rather than adding a second fast path.

1. One `np.unique` for the region; partition the coloured ids into three LUT-backed groups —
   *fill* (alpha > 0), *unclipped border* (outline mode), *clipped border* (fill mode with
   `show_borders_on_filled`, or fill at alpha ≤ 0).
2. Blend all fills in one vectorised pass (reusing the existing LUT technique).
3. Compute `find_boundaries` **once** on the label image.
4. Thicken by dilating an *owner-label* image rather than per-cell booleans, so each thickened
   pixel stays attributed to a cell.
5. Paint borders after fills, preserving the issue-#91 ordering fix.

### Preserving thickening semantics

`thicken_outline` → `engine._binary_dilation_4` is 4-connected with idiosyncratic border
handling (edge rows/columns receive only a partial update). A new `_max_dilate_labels_4(...)`
mirrors it shift-for-shift, substituting `np.maximum` for `|`. Because the old code applied
`pending_edges` in ascending id order (last write wins), **the largest mask id won overlaps** —
and a max-dilation reproduces exactly that rule.

### Preserving clipping semantics

Two clipping behaviours must be kept distinct (both from issue #91 follow-ups):

- **fill + border**: thickened borders are clipped back to the owning cell so they cannot dim a
  neighbour's fill. Batched as `owner == region_idx`.
- **outline mode**: edges are *not* clipped (existing behaviour; thick outlines may spill).

The two groups get separate dilations, unclipped painted first. At `outline_thickness = 1`
(dilation 0) edges never leave their own cell, so the groups are disjoint and the result is
bit-identical to the old code. Only at thickness ≥ 2 *with mixed modes on adjacent cells* can
they collide, and there the old result was itself an artefact of loop order.

## Implementation steps

1. Add `_max_dilate_labels_4(...)` to `ueler/viewer/mask_color_overlay.py`.
2. Rewrite `_apply_region_colors(...)` as the three-group vectorised pass; fold
   `_apply_region_colors_fill_vectorized` into it and delete `_can_vectorize_fill` and
   `_iter_mask_region_ids` (no callers outside the module).
3. Rewrite the four tests in `tests/test_mask_color_overlay.py` that patch `find_boundaries` /
   `thicken_outline` with per-cell fakes — they encode the old call shape, not the behaviour.
   Use real geometry and assert the same outcomes.
4. Add regressions: outline/fill parity against a per-cell reference implementation, mixed
   modes in one region, thickened outlines, and largest-id-wins overlap resolution.
5. Re-run the benchmark to confirm the asymptotics.

## Validation

- `python -m unittest tests.test_mask_color_overlay tests.test_mask_painter_mode_visibility`
- `python -m unittest tests.test_mask_painter_continuous tests.test_cell_gallery tests.test_export_fovs_batch tests.test_roi_manager_tags`
- `python -m unittest discover -s tests -t .`
- Benchmark fill vs outline at 200/1000/5000 cells.

## Risks

- **Behaviour drift on thick mixed-mode outlines.** Mitigated by the dilation-0 equivalence
  argument above and by a parity test against a per-cell reference.
- **LUT sizing.** The LUT is `max_id + 1` rows, so very large sparse label values cost memory.
  This exposure is unchanged — the existing fill fast path already indexes LUTs by raw label.
- **Shared blast radius.** `apply_registry_colors(...)` backs live FOV rendering, live map mode,
  ROI replay, gallery, and export (`main_viewer.py:1902, 4156, 4418, 4506`). Covered by running
  the full suite, not just the overlay tests.
