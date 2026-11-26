"""anywidget-based clickable thumbnail grid shared by the gallery views.

This widget replaces the Matplotlib ``plt.subplots`` galleries previously used by
the ROI manager browser and the cell gallery. Rendering the actual tile pixels stays
in the (front-end agnostic) numpy pipeline; this widget only handles *display* and
*click* events, so it does not depend on the interactive ipympl backend.

Each tile is a pre-rendered PNG shipped as a base64 data-URI. Tiles are laid out in a
responsive CSS grid of ``<img>`` elements; the tile label is shown as a pure CSS/JS
hover tooltip (no Python round-trip). Clicking a tile writes ``"<id>|<nonce>"`` to the
``clicked`` traitlet — the nonce ensures re-clicking the already-selected tile still
fires a traitlet change (traitlets do not emit when a value is set to its current one).

Traitlets (all ``sync=True`` when anywidget is available):
- ``tiles``   — list of ``{"id": str, "src": str, "label": str}`` dicts (``src`` is a
  ``data:image/png;base64,...`` URI); defines the display order.
- ``columns`` — number of grid columns.
- ``clicked`` — JS→Python signal; set to ``"<id>|<nonce>"`` when a tile is clicked.
- ``controls`` — the per-tile buttons to offer: ``"star"`` (☆/★, top left) and
  ``"remove"`` (✕, top right). Empty by default, so a gallery that does not use
  them (the ROI browser) looks as before.
- ``badges`` — ``{tile id: [mark, ...]}``; ``"star"`` shows a filled ★. Kept apart
  from ``tiles`` so a star toggle does not resend every image. A ★ without the
  ``"star"`` control is shown but cannot be clicked (a read-only set).
- ``action`` — JS→Python signal for the buttons: ``"<action>|<id>|<nonce>"``
  (``star`` or ``remove``). A button click is not also a tile click.
"""

from __future__ import annotations

import base64
import io

import numpy as np


def array_to_data_uri(array) -> str:
    """Encode an image array as a base64 PNG ``data:`` URI.

    Accepts float arrays in ``[0, 1]`` or ``uint8`` arrays, and grayscale (2D),
    RGB, or RGBA inputs. Returns an empty string for empty/degenerate input.
    """
    arr = np.asarray(array)
    if arr.size == 0:
        return ""

    if arr.dtype != np.uint8:
        arr = (np.clip(arr.astype(np.float64), 0.0, 1.0) * 255.0).round().astype(np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    png_bytes = _encode_png(np.ascontiguousarray(arr))
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def text_placeholder_uri(message: str, size: int = 160) -> str:
    """Return a data-URI for a light-gray square tile with ``message`` centered.

    Used in place of a rendered thumbnail when a tile cannot be produced (e.g.
    "No channels" / "Preview unavailable"). Falls back to a plain gray tile when
    Pillow is unavailable.
    """
    side = max(16, int(size))
    try:
        from PIL import Image, ImageDraw  # type: ignore

        img = Image.new("RGB", (side, side), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        text = str(message or "")
        try:
            bbox = draw.textbbox((0, 0), text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:  # pragma: no cover - very old Pillow
            tw, th = (len(text) * 6, 11)
        draw.text(((side - tw) / 2, (side - th) / 2), text, fill=(120, 120, 120))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # pragma: no cover - Pillow missing or drawing failure
        return array_to_data_uri(np.full((side, side, 3), 0.94, dtype=np.float64))


def _encode_png(arr: np.ndarray) -> bytes:
    """Encode a uint8 HxWxC array to PNG bytes, trying PIL then imageio then skimage."""
    buf = io.BytesIO()
    try:  # Pillow is the most reliable in-memory PNG encoder.
        from PIL import Image  # type: ignore

        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()
    except Exception:  # pragma: no cover - fallback for environments without PIL
        pass

    try:
        import imageio  # type: ignore

        imageio.imwrite(buf, arr, format="PNG")
        return buf.getvalue()
    except Exception:  # pragma: no cover - fallback for optional dependency
        pass

    from skimage.io import imsave  # type: ignore

    imsave(buf, arr, plugin="imageio", format_str="png")
    return buf.getvalue()


try:
    import anywidget
    import traitlets

    class TileGalleryWidget(anywidget.AnyWidget):  # type: ignore[misc]
        """Clickable thumbnail grid rendered from pre-encoded PNG tiles."""

        tiles: list = traitlets.List().tag(sync=True)
        columns: int = traitlets.Int(2).tag(sync=True)
        clicked: str = traitlets.Unicode("").tag(sync=True)
        max_height: str = traitlets.Unicode("400px").tag(sync=True)
        controls: list = traitlets.List().tag(sync=True)
        badges: dict = traitlets.Dict().tag(sync=True)
        action: str = traitlets.Unicode("").tag(sync=True)

        _css = """
.tg-root {
    overflow-y: auto;
    box-sizing: border-box;
}
.tg-grid {
    display: grid;
    gap: 4px;
    padding: 2px;
    box-sizing: border-box;
}
.tg-tile {
    position: relative;
    cursor: pointer;
    border: 1px solid transparent;
    border-radius: 3px;
    overflow: hidden;
    background: var(--jp-layout-color2, #f0f0f0);
    line-height: 0;
}
.tg-tile:hover {
    border-color: var(--jp-brand-color1, #2196f3);
}
.tg-tile img {
    width: 100%;
    height: auto;
    display: block;
    image-rendering: pixelated;
}
.tg-tip {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    padding: 1px 4px;
    font-family: var(--jp-ui-font-family, sans-serif);
    font-size: 11px;
    line-height: 1.3;
    color: #fff;
    background: rgba(0, 0, 0, 0.6);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    opacity: 0;
    transition: opacity 0.1s ease-in-out;
    pointer-events: none;
}
.tg-tile:hover .tg-tip {
    opacity: 1;
}
.tg-btn {
    position: absolute;
    top: 2px;
    min-width: 18px;
    height: 18px;
    padding: 0 3px;
    border-radius: 9px;
    font-family: var(--jp-ui-font-family, sans-serif);
    font-size: 13px;
    line-height: 18px;
    text-align: center;
    color: #fff;
    background: rgba(0, 0, 0, 0.55);
    cursor: pointer;
    user-select: none;
    opacity: 0;
    transition: opacity 0.1s ease-in-out;
}
.tg-star { left: 2px; }
.tg-remove { right: 2px; }
.tg-tile:hover .tg-btn { opacity: 1; }
.tg-btn:hover { background: rgba(0, 0, 0, 0.8); }
.tg-star.tg-on { opacity: 1; color: #ffc107; }
.tg-star.tg-static { cursor: default; pointer-events: none; }
.tg-star.tg-static:not(.tg-on) { display: none; }
.tg-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 12px;
    font-family: var(--jp-ui-font-family, sans-serif);
    font-size: 12px;
    color: var(--jp-ui-font-color2, #888);
    font-style: italic;
}
"""

        _esm = r"""
export function render({ model, el }) {
  let _nonce = 0;

  function marks(id) {
    const badges = model.get('badges') || {};
    return badges[id] || [];
  }

  function applyBadges() {
    grid.querySelectorAll('.tg-star').forEach(function (star) {
      const on = marks(star.dataset.id).indexOf('star') >= 0;
      star.classList.toggle('tg-on', on);
      star.textContent = on ? '\u2605' : '\u2606';
      star.title = star.classList.contains('tg-static') ? 'Starred' : (on ? 'Unstar' : 'Star');
    });
  }

  function sendAction(name, id) {
    model.set('action', name + '|' + id + '|' + (_nonce++));
    model.save_changes();
  }

  const root = document.createElement('div');
  root.className = 'tg-root';

  const grid = document.createElement('div');
  grid.className = 'tg-grid';

  function applyHeight() {
    root.style.maxHeight = model.get('max_height') || '400px';
  }

  function rebuild() {
    const tiles = model.get('tiles') || [];
    const columns = Math.max(1, parseInt(model.get('columns'), 10) || 1);
    grid.style.gridTemplateColumns = 'repeat(' + columns + ', 1fr)';

    while (grid.firstChild) { grid.removeChild(grid.firstChild); }

    if (!tiles.length) {
      const empty = document.createElement('div');
      empty.className = 'tg-empty';
      empty.textContent = 'No items to display.';
      grid.appendChild(empty);
      return;
    }

    tiles.forEach(function (t) {
      const tile = document.createElement('div');
      tile.className = 'tg-tile';
      tile.dataset.id = t.id != null ? String(t.id) : '';

      const img = document.createElement('img');
      img.loading = 'lazy';
      img.draggable = false;
      if (t.src) { img.src = t.src; }
      img.alt = t.label || '';
      tile.appendChild(img);

      if (t.label) {
        const tip = document.createElement('div');
        tip.className = 'tg-tip';
        tip.textContent = t.label;
        tip.title = t.label;
        tile.appendChild(tip);
      }

      const controls = model.get('controls') || [];
      const id = tile.dataset.id;
      const star = document.createElement('div');
      star.className = 'tg-btn tg-star';
      star.dataset.id = id;
      if (controls.indexOf('star') >= 0) {
        star.addEventListener('click', function (ev) {
          ev.stopPropagation();
          sendAction('star', id);
        });
      } else {
        star.classList.add('tg-static');
      }
      tile.appendChild(star);
      if (controls.indexOf('remove') >= 0) {
        const remove = document.createElement('div');
        remove.className = 'tg-btn tg-remove';
        remove.textContent = '\u2715';
        remove.title = 'Remove from the set';
        remove.addEventListener('click', function (ev) {
          ev.stopPropagation();
          sendAction('remove', id);
        });
        tile.appendChild(remove);
      }

      tile.addEventListener('click', function () {
        model.set('clicked', tile.dataset.id + '|' + (_nonce++));
        model.save_changes();
      });

      grid.appendChild(tile);
    });
    applyBadges();
  }

  model.on('change:tiles', rebuild);
  model.on('change:columns', rebuild);
  model.on('change:controls', rebuild);
  model.on('change:badges', applyBadges);
  model.on('change:max_height', applyHeight);

  applyHeight();
  rebuild();
  root.appendChild(grid);
  el.appendChild(root);
}
"""

except (ImportError, AttributeError):
    # Fallback for environments without anywidget (e.g. unit tests / headless CI).
    # Provides the same traitlet interface so Python code works unchanged.
    import traitlets  # type: ignore[import]

    class TileGalleryWidget(traitlets.HasTraits):  # type: ignore[no-redef]
        """Headless fallback used when anywidget is not installed."""

        tiles: list = traitlets.List()
        columns: int = traitlets.Int(2)
        clicked: str = traitlets.Unicode("")
        max_height: str = traitlets.Unicode("400px")
        controls: list = traitlets.List()
        badges: dict = traitlets.Dict()
        action: str = traitlets.Unicode("")


def parse_clicked_id(value) -> str:
    """Return the tile id from a ``"<id>|<nonce>"`` ``clicked`` payload.

    Splits from the right so ids that themselves contain ``"|"`` are preserved.
    """
    text = str(value or "")
    if not text:
        return ""
    return text.rsplit("|", 1)[0]


def parse_action(value) -> "tuple[str, str]":
    """Return ``(action, tile id)`` from an ``"<action>|<id>|<nonce>"`` payload, or ``("", "")``."""
    text = str(value or "")
    head = text.rsplit("|", 1)[0] if "|" in text else ""
    name, sep, tile_id = head.partition("|")
    return (name, tile_id) if sep and name and tile_id else ("", "")


__all__ = [
    "TileGalleryWidget",
    "parse_action",
    "array_to_data_uri",
    "text_placeholder_uri",
    "parse_clicked_id",
]
