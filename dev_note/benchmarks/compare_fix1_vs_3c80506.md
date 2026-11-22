# UELer before/after — 3c80506 vs working tree

macOS-26.5.2-arm64-arm-64bit · 12 CPUs · 18.0 GB RAM · Python 3.10.20 · matplotlib 3.10.9 · ipympl 0.10.0
Median of 3 runs per version, interleaved. Latency = kernel time from the user's input until the PNG showing the correct image is sent; network transfer and browser decode come on top.

## 4 FOVs, 1024×1024 px, 5 channels, 3,000 cells/FOV · canvas 600 CSS px × 1 = 600 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 127 | 111 | -12 % | 2 → 1 |
| Pan, one step (zoomed in 4×) | 84 | 54 | -36 % | 2 → 1 |
| Pan, one step (mask outlines shown) | 157 | 140 | -11 % | 2 → 1 |
| Zoom to a rectangle | 77 | 63 | -18 % | 2 → 1 |
| Add a marker channel | 163 | 160 | -2 % | 1 → 1 |
| Show mask outlines | 130 | 124 | -5 % | 1 → 1 |
| Switch FOV (first visit) | 186 | 181 | -3 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 277 | 260 | -6 % | 2 → 1 |
| Switch FOV (visited before) | 124 | 124 | +1 % | 1 → 1 |

## 4 FOVs, 1024×1024 px, 5 channels, 3,000 cells/FOV · canvas 600 CSS px × 2 = 1200 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 247 | 205 | -17 % | 2 → 1 |
| Pan, one step (zoomed in 4×) | 78 | 56 | -29 % | 2 → 1 |
| Pan, one step (mask outlines shown) | 266 | 224 | -16 % | 2 → 1 |
| Zoom to a rectangle | 178 | 141 | -21 % | 2 → 1 |
| Add a marker channel | 250 | 251 | +1 % | 1 → 1 |
| Show mask outlines | 167 | 167 | +0 % | 1 → 1 |
| Switch FOV (first visit) | 285 | 276 | -3 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 377 | 344 | -9 % | 2 → 1 |
| Switch FOV (visited before) | 220 | 212 | -4 % | 1 → 1 |

## 3 FOVs, 2048×2048 px, 10 channels, 12,000 cells/FOV · canvas 600 CSS px × 1 = 600 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 647 | 604 | -7 % | 2 → 1 |
| Pan, one step (zoomed in 4×) | 196 | 168 | -14 % | 2 → 1 |
| Pan, one step (mask outlines shown) | 768 | 726 | -5 % | 2 → 1 |
| Zoom to a rectangle | 248 | 213 | -14 % | 2 → 1 |
| Add a marker channel | 786 | 776 | -1 % | 1 → 1 |
| Show mask outlines | 732 | 716 | -2 % | 1 → 1 |
| Switch FOV (first visit) | 1559 | 1619 | +4 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 1573 | 1620 | +3 % | 2 → 1 |
| Switch FOV (visited before) | 620 | 668 | +8 % | 1 → 1 |

## 3 FOVs, 2048×2048 px, 10 channels, 12,000 cells/FOV · canvas 600 CSS px × 2 = 1200 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 715 | 670 | -6 % | 2 → 1 |
| Pan, one step (zoomed in 4×) | 303 | 218 | -28 % | 2 → 1 |
| Pan, one step (mask outlines shown) | 830 | 781 | -6 % | 2 → 1 |
| Zoom to a rectangle | 324 | 275 | -15 % | 2 → 1 |
| Add a marker channel | 888 | 851 | -4 % | 1 → 1 |
| Show mask outlines | 772 | 748 | -3 % | 1 → 1 |
| Switch FOV (first visit) | 1558 | 1533 | -2 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 1616 | 1538 | -5 % | 2 → 1 |
| Switch FOV (visited before) | 698 | 688 | -1 % | 1 → 1 |

