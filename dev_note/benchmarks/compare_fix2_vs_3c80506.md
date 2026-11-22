# UELer before/after — 3c80506 vs working tree

macOS-26.5.2-arm64-arm-64bit · 12 CPUs · 18.0 GB RAM · Python 3.10.20 · matplotlib 3.10.9 · ipympl 0.10.0
Median of 3 runs per version, interleaved. Latency = kernel time from the user's input until the PNG showing the correct image is sent; network transfer and browser decode come on top.

## 4 FOVs, 1024×1024 px, 5 channels, 3,000 cells/FOV · canvas 600 CSS px × 1 = 600 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 128 | 73 | -43 % | 2 → 1 |
| Pan, one step (zoomed in 4×) | 84 | 52 | -38 % | 2 → 1 |
| Pan, one step (mask outlines shown) | 156 | 99 | -36 % | 2 → 1 |
| Zoom to a rectangle | 75 | 56 | -26 % | 2 → 1 |
| Add a marker channel | 158 | 103 | -35 % | 1 → 1 |
| Show mask outlines | 133 | 88 | -34 % | 1 → 1 |
| Switch FOV (first visit) | 182 | 138 | -24 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 274 | 174 | -37 % | 2 → 1 |
| Switch FOV (visited before) | 128 | 83 | -35 % | 1 → 1 |

## 4 FOVs, 1024×1024 px, 5 channels, 3,000 cells/FOV · canvas 600 CSS px × 2 = 1200 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 232 | 162 | -30 % | 2 → 1 |
| Pan, one step (zoomed in 4×) | 79 | 50 | -37 % | 2 → 1 |
| Pan, one step (mask outlines shown) | 252 | 189 | -25 % | 2 → 1 |
| Zoom to a rectangle | 179 | 136 | -24 % | 2 → 1 |
| Add a marker channel | 250 | 199 | -20 % | 1 → 1 |
| Show mask outlines | 169 | 127 | -25 % | 1 → 1 |
| Switch FOV (first visit) | 267 | 239 | -10 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 375 | 265 | -29 % | 2 → 1 |
| Switch FOV (visited before) | 216 | 182 | -16 % | 1 → 1 |

## 3 FOVs, 2048×2048 px, 10 channels, 12,000 cells/FOV · canvas 600 CSS px × 1 = 600 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 664 | 164 | -75 % | 2 → 1 |
| Pan, one step (zoomed in 4×) | 191 | 137 | -28 % | 2 → 1 |
| Pan, one step (mask outlines shown) | 777 | 216 | -72 % | 2 → 1 |
| Zoom to a rectangle | 249 | 159 | -36 % | 2 → 1 |
| Add a marker channel | 806 | 240 | -70 % | 1 → 1 |
| Show mask outlines | 729 | 221 | -70 % | 1 → 1 |
| Switch FOV (first visit) | 1521 | 647 | -57 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 1543 | 712 | -54 % | 2 → 1 |
| Switch FOV (visited before) | 632 | 198 | -69 % | 1 → 1 |

## 3 FOVs, 2048×2048 px, 10 channels, 12,000 cells/FOV · canvas 600 CSS px × 2 = 1200 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 747 | 326 | -56 % | 2 → 1 |
| Pan, one step (zoomed in 4×) | 312 | 203 | -35 % | 2 → 1 |
| Pan, one step (mask outlines shown) | 883 | 446 | -49 % | 2 → 1 |
| Zoom to a rectangle | 345 | 227 | -34 % | 2 → 1 |
| Add a marker channel | 905 | 467 | -48 % | 1 → 1 |
| Show mask outlines | 795 | 421 | -47 % | 1 → 1 |
| Switch FOV (first visit) | 1676 | 920 | -45 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 1619 | 896 | -45 % | 2 → 1 |
| Switch FOV (visited before) | 720 | 367 | -49 % | 1 → 1 |

