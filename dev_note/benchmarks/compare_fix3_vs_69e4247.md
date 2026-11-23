# UELer before/after — 69e4247 vs working tree

macOS-26.5.2-arm64-arm-64bit · 12 CPUs · 18.0 GB RAM · Python 3.10.20 · matplotlib 3.10.9 · ipympl 0.10.0
Median of 3 runs per version, interleaved. Latency = kernel time from the user's input until the PNG showing the correct image is sent; network transfer and browser decode come on top.

## 4 FOVs, 1024×1024 px, 5 channels, 3,000 cells/FOV · canvas 600 CSS px × 1 = 600 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 76 | 52 | -32 % | 1 → 1 |
| Pan, one step (zoomed in 4×) | 53 | 32 | -39 % | 1 → 1 |
| Pan, one step (mask outlines shown) | 99 | 78 | -22 % | 1 → 1 |
| Zoom to a rectangle | 56 | 37 | -34 % | 1 → 1 |
| Add a marker channel | 102 | 81 | -21 % | 1 → 1 |
| Show mask outlines | 90 | 70 | -22 % | 1 → 1 |
| Switch FOV (first visit) | 141 | 136 | -4 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 172 | 145 | -16 % | 1 → 1 |
| Switch FOV (visited before) | 84 | 63 | -25 % | 1 → 1 |

## 4 FOVs, 1024×1024 px, 5 channels, 3,000 cells/FOV · canvas 600 CSS px × 2 = 1200 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 162 | 140 | -14 % | 1 → 1 |
| Pan, one step (zoomed in 4×) | 49 | 31 | -38 % | 1 → 1 |
| Pan, one step (mask outlines shown) | 186 | 163 | -12 % | 1 → 1 |
| Zoom to a rectangle | 134 | 115 | -14 % | 1 → 1 |
| Add a marker channel | 197 | 172 | -13 % | 1 → 1 |
| Show mask outlines | 128 | 108 | -16 % | 1 → 1 |
| Switch FOV (first visit) | 233 | 224 | -4 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 257 | 236 | -8 % | 1 → 1 |
| Switch FOV (visited before) | 174 | 157 | -9 % | 1 → 1 |

## 3 FOVs, 2048×2048 px, 10 channels, 12,000 cells/FOV · canvas 600 CSS px × 1 = 600 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 164 | 56 | -66 % | 1 → 1 |
| Pan, one step (zoomed in 4×) | 135 | 32 | -76 % | 1 → 1 |
| Pan, one step (mask outlines shown) | 215 | 100 | -54 % | 1 → 1 |
| Zoom to a rectangle | 153 | 47 | -70 % | 1 → 1 |
| Add a marker channel | 237 | 118 | -50 % | 1 → 1 |
| Show mask outlines | 210 | 98 | -53 % | 1 → 1 |
| Switch FOV (first visit) | 647 | 499 | -23 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 710 | 544 | -23 % | 1 → 1 |
| Switch FOV (visited before) | 186 | 74 | -60 % | 1 → 1 |

## 3 FOVs, 2048×2048 px, 10 channels, 12,000 cells/FOV · canvas 600 CSS px × 2 = 1200 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 319 | 203 | -36 % | 1 → 1 |
| Pan, one step (zoomed in 4×) | 195 | 94 | -52 % | 1 → 1 |
| Pan, one step (mask outlines shown) | 437 | 310 | -29 % | 1 → 1 |
| Zoom to a rectangle | 220 | 116 | -47 % | 1 → 1 |
| Add a marker channel | 443 | 310 | -30 % | 1 → 1 |
| Show mask outlines | 415 | 288 | -31 % | 1 → 1 |
| Switch FOV (first visit) | 888 | 708 | -20 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 859 | 693 | -19 % | 1 → 1 |
| Switch FOV (visited before) | 359 | 232 | -35 % | 1 → 1 |

