# UELer before/after — 4ff01d0 vs working tree

macOS-26.5.2-arm64-arm-64bit · 12 CPUs · 18.0 GB RAM · Python 3.10.20 · matplotlib 3.10.9 · ipympl 0.10.0
Median of 3 runs per version, interleaved. Latency = kernel time from the user's input until the PNG showing the correct image is sent; network transfer and browser decode come on top.

## 3 FOVs, 2048×2048 px, 10 channels, 12,000 cells/FOV · canvas 1605 CSS px × 0.833333 = 1337 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after | KB sent until correct, before → after |
| --- | --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 39 | 38 | -2 % | 1 → 1 | 131 → 131 |
| Pan, one step (zoomed in 4×) | 36 | 35 | -2 % | 1 → 1 | 99 → 99 |
| Pan, one step (mask outlines shown) | 58 | 57 | -1 % | 1 → 1 | 115 → 115 |
| Pan, sharp frame after letting go (whole FOV) | 214 | 126 | -41 % | 1 → 1 | 1264 → 423 |
| Zoom to a rectangle | 138 | 50 | -63 % | 1 → 1 | 988 → 233 |
| Add a marker channel | 273 | 167 | -39 % | 1 → 1 | 1459 → 439 |
| Show mask outlines | 298 | 250 | -16 % | 1 → 1 | 1953 → 1017 |
| Switch FOV (first visit) | 647 | 553 | -14 % | 1 → 1 | 1405 → 442 |
| Switch FOV while zoomed in (first visit) | 659 | 545 | -17 % | 1 → 1 | 1326 → 441 |
| Switch FOV (visited before) | 256 | 150 | -41 % | 1 → 1 | 1404 → 442 |

