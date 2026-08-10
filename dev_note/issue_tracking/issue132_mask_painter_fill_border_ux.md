# Issue #132 — Mask Painter: consistent Fill / Border UX across discrete and continuous modes

## Problem

The mask painter has two coloring modes (categorical/discrete and continuous, issue #115) whose
controls do not line up, and whose fill/border behaviour is implicit rather than stated.

1. **`Identifier:` sits above the mode toggle**, in the shared control panel
   (`initiate_ui`, `mask_painter.py`), so it reads as applying to both modes. It is only used by the
   categorical branch — `build_painter_state_maps_for_fov` ignores it entirely when a `continuous`
   spec is supplied.
2. **Fill and border are entangled.**
   - Categorical: `Global fill` (a checkbox) decides fill for classes still inheriting the default,
     `Opacity (%)` is the fill opacity, and `Borders on filled` adds an outline *only* on filled
     cells. A cell that is not filled always gets an outline, in its **class** color, and there is no
     way to turn that outline off or change its color.
   - Continuous: a single `Fill (unchecked = outline)` checkbox multiplexes two different renderings.
     There is no border control at all; `build_painter_state_maps_for_fov` returns an empty
     `border_map` for the continuous branch.
   - Border opacity does not exist in either mode — `_apply_region_colors` writes border pixels
     opaquely (`canvas[painted] = ...`).
   - Border color offers two options (`Mask color`, `Same as fill`) and no user-defined color.

## Requested (verbatim from the issue)

- Move `identifier` into the discrete-mode section.
- Control fill and border separately:
  - a `Fill` checkbox followed by its own opacity slider; in discrete mode rename `Global fill`
    to `Fill all classes`.
  - a `Border` checkbox followed by its own opacity slider and color picker. The color can be the
    value to paint, the masks' color, or a user-defined color; the default is the masks' color.

## Approach

Make **Fill** and **Border** orthogonal for every painted cell, in both modes:

| Fill | Border | Result |
| --- | --- | --- |
| on | on | filled at fill opacity, border drawn on top (clipped to the cell) |
| on | off | filled only |
| off | on | outline only — the current default look |
| off | off | cell not painted |

This is the only reading of "control the fill and the border separately" that makes the two modes
consistent, and it subsumes the old `Fill (unchecked = outline)` multiplexer: *unchecked fill* is
just *fill off, border on*.

**Fill stays per-mode** (categorical has per-class fill checkboxes and per-class opacity that
override the global one; continuous has a single fill/opacity pair). **Border is shared** by both
modes and rendered below the mode-specific block, since a border checkbox, opacity and color mean
the same thing regardless of how the cell got its color.

### Border geometry is unchanged

`_apply_region_colors` (rewritten for issue #131) already paints two border groups with different
clipping rules, and those rules are kept exactly:

- a cell **without** fill gets the unclipped, thickenable outline (`_max_dilate_labels_4`),
- a cell **with** fill gets the border clipped to its own cell (`_dilate_within_labels`),

so at `outline_thickness = 1` the geometry is bit-identical to before. Only the *color* and the
*alpha* of those pixels change, plus the new ability to suppress them.

### Border color

`border_map` is currently populated only for filled cells; unfilled cells fall back to the class
color inside the overlay. With an explicit border color control that fallback has to go: the state
builder now emits a border color for **every** painted cell, and the overlay uses
`border_registry.get(id, cell_color)` for both groups. Three modes:

| Dropdown | Constant | Border color |
| --- | --- | --- |
| `Painted color` (default) | `same_as_fill` | the cell's class color (categorical) or colormap color (continuous) |
| `Mask color` | `mask_type_color` | the mask layer's own color |
| `Custom…` | `custom` | the hex from the new `border_color_picker` |

### The default deviates from the issue text, deliberately

The issue text asks for `Mask color` as the border default (which was also the old dropdown
default, back when the dropdown only affected filled cells). Once borders are orthogonal that
default also governs the outline of **unfilled** cells, which had always drawn in their *class*
color — so with the out-of-the-box settings (`Fill all classes` off, `Border` on) it would draw
mask-colored outlines and hide the very colors the painter exists to show. This was raised with the
developer and the default changed to `Painted color` by their decision.

Two things keep the old value reachable and honest:

- `MaskPainterSnapshot.border_color_mode` and the `build_painter_state_maps_for_fov` parameter still
  default to `mask_type_color`. Those are the *legacy* fallbacks for payloads written before the
  field existed, not the UI default, and are commented as such.
- Palettes and ROI/checkpoint snapshots saved before this change stored `mask_type_color`
  explicitly, so they are restored as saved and will draw mask-colored outlines until switched over.
  This is called out in `doc/log.md` and `README.md`.

### Border opacity

New scalar `border_alpha` threaded to the overlay; `_paint_border_group` blends
(`(1-a)*canvas + a*color`) instead of assigning. The gallery/engine path needs no new machinery —
`engine._blend_mask_pixels` already honours `MaskRenderSettings.alpha` for `mode="outline"`.

## Implementation steps

1. **`ueler/viewer/mask_color_overlay.py`**
   - `apply_mask_color_overlay`: `show_borders_on_filled: bool = False` → `show_borders: bool = True`,
     add `border_alpha: float = 1.0`.
   - `_apply_region_colors`: gate both border groups on `show_borders`; source both groups' color
     from `border_registry` with the cell color as fallback; drop the implicit
     "alpha <= 0 forces a border" rule (orthogonal controls express that explicitly).
   - `_paint_border_group`: blend at `alpha`.
2. **`ueler/rendering/engine.py`** — `MaskPainterSnapshot`: `show_borders_on_filled: bool = False`
   → `show_borders: bool = True`; add `border_opacity: int = 100`,
   `border_custom_color: str = DEFAULT`.
3. **`ueler/viewer/plugin/mask_painter.py`**
   - add `BORDER_COLOR_MODE_CUSTOM`.
   - `UiComponent`: move `identifier_dropdown` into `categorical_layout`; rename
     `Global fill` → `Fill all classes`, `Fill (unchecked = outline)` → `Fill`; rename
     `show_fill_borders_checkbox` → `border_checkbox` (`Border`, default **on**); add
     `border_opacity_input` and `border_color_picker`; new shared `border_layout` inside
     `colors_layout`; one `OPACITY_INPUT_WIDTH = "150px"` for all three opacity inputs
     (previously 95px / 95px / 130px), so the two panes line up.
   - `build_painter_state_maps_for_fov`: new `border_custom_color` param, `custom` mode,
     `border_map` for every painted cell, and a `border_map` for the continuous branch.
   - accessors `get_show_borders()` / `get_border_alpha()` / `get_border_custom_color()`;
     `get_border_color_mode()` accepts `custom`.
   - snapshot capture/apply and color-set save/load carry the new fields, reading the legacy
     `show_fill_borders` key when the new one is absent.
4. **Call sites** — `main_viewer.py` (live FOV, live map, ROI replay, batch export),
   `plugin/cell_gallery.py`, `plugin/export_fovs.py`, `plugin/roi_manager_plugin.py`.
   `resolve_mask_painter_snapshot_for_fov` returns a 6-tuple (adds `border_alpha`).
5. **Tests** — update the suites that construct snapshots/payloads, and add coverage for the
   orthogonal matrix, border opacity blending, the custom border color, and the identifier's new
   placement.

## Validation

```bash
python -m unittest tests.test_mask_painter_mode_visibility tests.test_mask_painter_continuous \
    tests.test_mask_color_overlay tests.test_mask_color_sets
python -m unittest tests.test_cell_gallery tests.test_export_fovs_batch tests.test_rendering
python -m unittest discover -s tests -t .
```

## Risks

- **Renamed public-ish names** (`show_borders_on_filled` → `show_borders`,
  `show_fill_borders_checkbox` → `border_checkbox`) touch several suites. Renaming rather than
  aliasing is deliberate: the field's meaning changed from "borders on filled masks" to "borders",
  and keeping the old name would reproduce the confusion this issue is about.
- **Legacy snapshots/palettes** lack `show_borders`; they are read as `True` so outlines keep being
  drawn. Filled classes in those payloads now also gain a border (previously gated by
  `show_fill_borders`), and see the border-color caveat above.
- **`Fill off + Border off` paints nothing.** That is the honest product of orthogonal controls, but
  it is a state the old UI could not reach.
