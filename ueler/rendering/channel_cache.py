"""In-memory cache of decoded channel rasters for the render path.

In folder mode every channel is a lazy Dask array over a compressed TIFF, and
one TIFF page is one Dask task. Slicing any region of it therefore reads and
decompresses the whole file again, on every pan and zoom (~117 ms per render
for 2048² × 10 channels on a local SSD; more on network storage).

:class:`ChannelLevelCache` materialises each channel once and serves every
later render from memory. It holds two kinds of entry:

* **Full resolution** (``factor == 1``) when the channel fits the per-entry
  cap. Any region at any downsample factor is then a NumPy view of it, so the
  result is exactly ``source[ymin:ymax:f, xmin:xmax:f]`` without a copy.
* **A downsampled level** (``source[::f, ::f]``) for a channel too large to
  keep at full resolution. It serves only regions aligned to ``f``, which is
  what the main viewer requests (``get_axis_limits_with_padding``); anything
  else falls back to the lazy read.

Entries are evicted least recently used once their total exceeds
``max_bytes``. Arrays that are already in memory (NumPy) are not cached, and a
read that the cache cannot serve returns ``None`` so the caller keeps its
original path.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Hashable, Optional, Tuple

import numpy as np

Region = Tuple[int, int, int, int]

#: Default total budget. A 2048² uint16 channel is 8 MiB, so this holds about
#: six FOVs of ten such channels.
DEFAULT_MAX_BYTES = 512 * 1024 * 1024


def _is_lazy(source) -> bool:
    return hasattr(source, "compute") and not isinstance(source, np.ndarray)


def _materialise(array) -> np.ndarray:
    try:
        data = array.compute()
    except AttributeError:
        data = array
    return np.ascontiguousarray(np.asarray(data))


class ChannelLevelCache:
    """Byte-budgeted LRU cache of decoded channel rasters.

    Keys are ``(fov, channel, factor)``. Each entry also remembers the source
    array it was decoded from, so replacing a channel in the viewer's
    ``image_cache`` invalidates the entry instead of serving stale pixels.
    """

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES, max_entry_fraction: float = 1 / 8):
        self._entries: "OrderedDict[Tuple[Hashable, Hashable, int], Tuple[object, np.ndarray]]" = OrderedDict()
        self._nbytes = 0
        self.max_bytes = int(max_bytes)
        self.max_entry_fraction = float(max_entry_fraction)
        self.hits = 0
        self.misses = 0

    # -- budget ---------------------------------------------------------
    @property
    def nbytes(self) -> int:
        """Bytes currently held."""
        return self._nbytes

    @property
    def max_entry_bytes(self) -> int:
        """Largest single raster the cache will keep."""
        return int(self.max_bytes * self.max_entry_fraction)

    def __len__(self) -> int:
        return len(self._entries)

    def set_max_bytes(self, max_bytes: int) -> None:
        """Change the budget and evict down to it; ``0`` disables the cache."""
        self.max_bytes = max(0, int(max_bytes))
        self._evict()

    def clear(self) -> None:
        self._entries.clear()
        self._nbytes = 0

    def drop_fov(self, fov: Hashable) -> None:
        """Forget every entry of one FOV (called when the viewer evicts it)."""
        for key in [k for k in self._entries if k[0] == fov]:
            self._remove(key)

    # -- reads ----------------------------------------------------------
    def read_region(
        self,
        fov: Hashable,
        channel: Hashable,
        source,
        region_xy: Region,
        factor: int,
    ) -> Optional[np.ndarray]:
        """Return ``source[ymin:ymax:f, xmin:xmax:f]`` from memory, or ``None``.

        ``None`` means the cache does not handle this read (an in-memory
        source, a non-2-D array, a raster over the per-entry cap, a
        misaligned region on a downsampled level, or a disabled cache); the
        caller then reads ``source`` itself.
        """
        if self.max_bytes <= 0 or not _is_lazy(source):
            return None
        shape = getattr(source, "shape", None)
        if shape is None or len(shape) != 2:
            return None
        factor = max(1, int(factor))
        xmin, xmax, ymin, ymax = (int(v) for v in region_xy)
        itemsize = np.dtype(getattr(source, "dtype", np.float64)).itemsize
        height, width = int(shape[0]), int(shape[1])

        full = self._lookup(fov, channel, 1, source)
        if full is None and height * width * itemsize <= self.max_entry_bytes:
            self.misses += 1
            full = self._store(fov, channel, 1, source, _materialise(source))
        elif full is not None:
            self.hits += 1
        if full is not None:
            return full[ymin:ymax:factor, xmin:xmax:factor]

        # Too large to keep at full resolution: keep the requested level only.
        if factor == 1 or xmin % factor or ymin % factor:
            return None
        level = self._lookup(fov, channel, factor, source)
        if level is None:
            level_bytes = -(-height // factor) * -(-width // factor) * itemsize
            if level_bytes > self.max_entry_bytes:
                return None
            self.misses += 1
            level = self._store(fov, channel, factor, source, _materialise(source[::factor, ::factor]))
        else:
            self.hits += 1
        rows = -(-(ymax - ymin) // factor)
        cols = -(-(xmax - xmin) // factor)
        y0, x0 = ymin // factor, xmin // factor
        return level[y0 : y0 + rows, x0 : x0 + cols]

    # -- internals ------------------------------------------------------
    def _lookup(self, fov, channel, factor, source) -> Optional[np.ndarray]:
        key = (fov, channel, factor)
        entry = self._entries.get(key)
        if entry is None:
            return None
        cached_source, array = entry
        if cached_source is not source:
            self._remove(key)
            return None
        self._entries.move_to_end(key)
        return array

    def _store(self, fov, channel, factor, source, array: np.ndarray) -> np.ndarray:
        key = (fov, channel, factor)
        self._remove(key)
        self._entries[key] = (source, array)
        self._nbytes += int(array.nbytes)
        self._evict(keep=key)
        return array

    def _remove(self, key) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._nbytes -= int(entry[1].nbytes)

    def _evict(self, keep=None) -> None:
        while self._nbytes > self.max_bytes and self._entries:
            oldest = next(iter(self._entries))
            if oldest == keep:
                # The entry just stored is in use by the caller; evict it
                # last, and only if it alone exceeds the budget.
                if len(self._entries) == 1:
                    self._remove(oldest)
                    break
                self._entries.move_to_end(oldest)
                continue
            self._remove(oldest)
