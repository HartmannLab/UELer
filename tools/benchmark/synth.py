"""Generate synthetic multiplexed-imaging datasets for benchmarking UELer.

The layout is UELer's local folder mode::

    <out>/images/<fov>/<channel>.tiff     one uint16 image per channel
    <out>/masks/<fov>_whole_cell.tiff     int32 label mask
    <out>/cell_table.csv                  one row per cell: fov, label, centroid, mean per channel

Images are cell-like rather than noise: each cell is a soft blob whose brightness
per channel is drawn from a per-channel expression profile, over a low background
with Poisson-like noise. Noise-only images compress badly and would inflate the
PNG payloads ipympl sends to the browser, so they would overstate transfer cost.

Masks are a Voronoi tessellation of the cell seeds clipped to a cell radius, so
cells touch the way segmented tissue does and outline rendering has realistic work.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _cell_seeds(rng, size, n_cells):
    return rng.uniform(4, size - 4, size=(n_cells, 2))


def _label_mask(seeds, size, radius):
    """Voronoi labels clipped to *radius* pixels around each seed."""
    from scipy import ndimage

    seed_img = np.zeros((size, size), dtype=bool)
    ys = np.clip(seeds[:, 0].astype(int), 0, size - 1)
    xs = np.clip(seeds[:, 1].astype(int), 0, size - 1)
    seed_img[ys, xs] = True
    dist, (iy, ix) = ndimage.distance_transform_edt(~seed_img, return_indices=True)
    seed_id = np.zeros((size, size), dtype=np.int32)
    seed_id[ys, xs] = np.arange(1, len(seeds) + 1, dtype=np.int32)
    labels = seed_id[iy, ix]
    labels[dist > radius] = 0
    return labels


def _channel_image(rng, labels, n_cells, expression, size):
    """Smooth cell-like image: per-cell intensity painted into the mask, blurred, plus noise."""
    from scipy import ndimage

    lut = np.concatenate([[0.0], expression]).astype(np.float32)
    img = lut[labels]
    img = ndimage.gaussian_filter(img, sigma=1.5)
    img += 5.0  # background
    img = rng.poisson(img).astype(np.float32)
    return np.clip(img, 0, 65535).astype(np.uint16)


def generate(out, *, n_fovs=4, size=1024, n_channels=5, cells_per_fov=3000, compression="zlib", seed=0):
    import pandas as pd
    import tifffile

    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Cell radius chosen so the requested cell count roughly tiles the field.
    radius = max(3.0, 0.6 * size / np.sqrt(cells_per_fov))
    channels = [f"marker{c:02d}" for c in range(n_channels)]
    rows = []
    t0 = time.perf_counter()
    for f in range(n_fovs):
        fov = f"fov{f:04d}"
        (out / "images" / fov).mkdir(exist_ok=True)
        seeds = _cell_seeds(rng, size, cells_per_fov)
        labels = _label_mask(seeds, size, radius)
        tifffile.imwrite(out / "masks" / f"{fov}_whole_cell.tiff", labels, compression=compression)
        means = {}
        for ch in channels:
            # Bimodal expression: a positive fraction per marker, the rest dim.
            positive = rng.random(cells_per_fov) < rng.uniform(0.1, 0.5)
            expression = np.where(positive, rng.gamma(4.0, 40.0, cells_per_fov), rng.gamma(1.5, 5.0, cells_per_fov))
            tifffile.imwrite(out / "images" / fov / f"{ch}.tiff", _channel_image(rng, labels, cells_per_fov, expression, size), compression=compression)
            means[ch] = expression
        for i in range(cells_per_fov):
            row = {"fov": fov, "label": i + 1, "centroid-0": seeds[i, 0], "centroid-1": seeds[i, 1]}
            row.update({ch: float(means[ch][i]) for ch in channels})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "cell_table.csv", index=False)
    spec = dict(n_fovs=n_fovs, size=size, n_channels=n_channels, cells_per_fov=cells_per_fov, compression=compression, seed=seed, total_cells=n_fovs * cells_per_fov, generate_seconds=round(time.perf_counter() - t0, 2))
    (out / "dataset.json").write_text(json.dumps(spec, indent=2))
    return spec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", help="output directory")
    p.add_argument("--fovs", type=int, default=4)
    p.add_argument("--size", type=int, default=1024, help="image edge length in pixels")
    p.add_argument("--channels", type=int, default=5)
    p.add_argument("--cells", type=int, default=3000, help="cells per FOV")
    p.add_argument("--compression", default="zlib", help="tifffile compression, or 'none'")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    spec = generate(a.out, n_fovs=a.fovs, size=a.size, n_channels=a.channels, cells_per_fov=a.cells, compression=None if a.compression == "none" else a.compression, seed=a.seed)
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
