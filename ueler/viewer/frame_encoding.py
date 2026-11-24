"""Image format of the frames the main canvas sends to the browser.

ipympl (through matplotlib's WebAgg core) sends every frame as a PNG: either
the full canvas or a *diff* in which unchanged pixels are transparent. PNG is
lossless, and for a full frame of a multiplexed image it is large: a sharp
1 337 px frame was ~2.5 MB in a real session, several seconds on a slow
remote link.

:func:`install_frame_encoder` lets the viewer send the frames that replace
most of the canvas (after a zoom, a FOV switch, a channel change, or the sharp
frame at the end of a drag) as JPEG or WebP instead, several times smaller and
faster to encode. Everything else keeps the PNG path:

* small updates (a tooltip, a highlight), sent as PNG diffs, which are
  already small and need transparency;
* drag preview frames, whose repeated pixels PNG compresses better than the
  lossy formats do.

The front end labels every frame ``image/png``, but browsers pick the decoder
from the image data itself, so a JPEG or WebP frame displays the same way. A
lossy frame is sent in ``full`` image mode, since it has no transparency to
overlay a diff with.
"""

from __future__ import annotations

import types
from io import BytesIO
from typing import Callable, Dict

import numpy as np

__all__ = ["FRAME_FORMATS", "available_frame_formats", "install_frame_encoder"]

#: value -> label shown in Advanced Settings
FRAME_FORMATS: Dict[str, str] = {
    "png": "PNG (lossless)",
    "jpeg": "JPEG (smaller, lossy)",
    "webp": "WebP (smallest, lossy)",
}
JPEG_QUALITY = 85
WEBP_QUALITY = 80
#: A frame goes lossy only if at least this share of its pixels changed;
#: smaller changes stay PNG diffs.
LOSSY_MIN_CHANGED_FRACTION = 0.25


def available_frame_formats() -> Dict[str, str]:
    """The formats this Pillow build can encode (PNG always)."""
    try:
        from PIL import features
    except Exception:  # pragma: no cover
        return {"png": FRAME_FORMATS["png"]}
    out = {"png": FRAME_FORMATS["png"]}
    if features.check("jpg"):
        out["jpeg"] = FRAME_FORMATS["jpeg"]
    if features.check("webp"):
        out["webp"] = FRAME_FORMATS["webp"]
    return out


def encode_lossy(rgba: np.ndarray, fmt: str) -> bytes:
    """Encode an RGBA frame as JPEG or WebP (alpha dropped)."""
    from PIL import Image

    image = Image.fromarray(np.ascontiguousarray(rgba[..., :3]))
    with BytesIO() as out:
        if fmt == "jpeg":
            image.save(out, format="JPEG", quality=JPEG_QUALITY)
        elif fmt == "webp":
            image.save(out, format="WEBP", quality=WEBP_QUALITY, method=0)
        else:
            raise ValueError(f"not a lossy frame format: {fmt!r}")
        return out.getvalue()


def install_frame_encoder(canvas, get_format: Callable[[], str], is_preview: Callable[[], bool] = lambda: False) -> None:
    """Replace *canvas*'s ``get_diff_image`` with one that can send lossy frames.

    ``get_format()`` returns ``"png"``, ``"jpeg"`` or ``"webp"`` at the time of
    each frame, so a change of setting applies to the next frame.
    ``is_preview()`` returns True while the frame is a drag preview, which
    stays PNG. With ``"png"`` the canvas behaves exactly as before.
    """
    original = canvas.get_diff_image

    def get_diff_image(self):
        if not getattr(self, "_png_is_old", False):
            return original()
        try:
            fmt = get_format()
        except Exception:
            fmt = "png"
        if fmt not in ("jpeg", "webp"):
            return original()
        try:
            if is_preview():
                return original()
        except Exception:
            pass

        pixels = np.asarray(self.get_renderer().buffer_rgba())
        buff = pixels.view(np.uint32).squeeze(2)
        last = getattr(self, "_last_buff", None)
        same_shape = last is not None and buff.shape == last.shape
        must_be_full = getattr(self, "_force_full", False) or not same_shape
        if not must_be_full:
            changed = np.count_nonzero(buff != last) / buff.size
            if changed < LOSSY_MIN_CHANGED_FRACTION:
                return original()  # a small update: PNG diff

        try:
            data = encode_lossy(pixels, fmt)
        except Exception:
            return original()
        self.set_image_mode("full")
        # Same bookkeeping as matplotlib's get_diff_image.
        self._last_buff = buff.copy()
        self._force_full = False
        self._png_is_old = False
        return data

    canvas.get_diff_image = types.MethodType(get_diff_image, canvas)
