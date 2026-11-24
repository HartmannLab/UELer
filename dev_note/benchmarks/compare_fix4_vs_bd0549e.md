# UELer before/after — bd0549e vs working tree

macOS-26.5.2-arm64-arm-64bit · 12 CPUs · 18.0 GB RAM · Python 3.10.20 · matplotlib 3.10.9 · ipympl 0.10.0
Median of 3 runs per version, interleaved. Latency = kernel time from the user's input until the PNG showing the correct image is sent; network transfer and browser decode come on top.

## 4 FOVs, 1024×1024 px, 5 channels, 3,000 cells/FOV · canvas 600 CSS px × 1 = 600 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 63 | 57 | -10 % | 1 → 1 |
| Pan, one step (zoomed in 4×) | 40 | 36 | -10 % | 1 → 1 |
| Pan, one step (mask outlines shown) | 96 | 86 | -11 % | 1 → 1 |
| Zoom to a rectangle | 44 | 42 | -3 % | 1 → 1 |
| Add a marker channel | 87 | 77 | -12 % | 1 → 1 |
| Show mask outlines | 76 | 82 | +9 % | 1 → 1 |
| Switch FOV (first visit) | 174 | 142 | -19 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 186 | 138 | -26 % | 1 → 1 |
| Switch FOV (visited before) | 75 | 67 | -11 % | 1 → 1 |

## 4 FOVs, 1024×1024 px, 5 channels, 3,000 cells/FOV · canvas 600 CSS px × 2 = 1200 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 148 | 148 | +0 % | 1 → 1 |
| Pan, one step (zoomed in 4×) | 31 | 31 | +0 % | 1 → 1 |
| Pan, one step (mask outlines shown) | 171 | 171 | +0 % | 1 → 1 |
| Zoom to a rectangle | 122 | 122 | -0 % | 1 → 1 |
| Add a marker channel | 180 | 171 | -5 % | 1 → 1 |
| Show mask outlines | 113 | 111 | -1 % | 1 → 1 |
| Switch FOV (first visit) | 233 | 235 | +1 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 240 | 233 | -3 % | 1 → 1 |
| Switch FOV (visited before) | 159 | 160 | +0 % | 1 → 1 |

## 3 FOVs, 2048×2048 px, 10 channels, 12,000 cells/FOV · canvas 600 CSS px × 1 = 600 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 56 | 57 | +1 % | 1 → 1 |
| Pan, one step (zoomed in 4×) | 33 | 33 | +1 % | 1 → 1 |
| Pan, one step (mask outlines shown) | 104 | 103 | -1 % | 1 → 1 |
| Zoom to a rectangle | 48 | 47 | -2 % | 1 → 1 |
| Add a marker channel | 122 | 106 | -13 % | 1 → 1 |
| Show mask outlines | 104 | 103 | -2 % | 1 → 1 |
| Switch FOV (first visit) | 521 | 501 | -4 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 568 | 498 | -12 % | 1 → 1 |
| Switch FOV (visited before) | 76 | 74 | -2 % | 1 → 1 |

## 3 FOVs, 2048×2048 px, 10 channels, 12,000 cells/FOV · canvas 600 CSS px × 2 = 1200 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after |
| --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 205 | 205 | -0 % | 1 → 1 |
| Pan, one step (zoomed in 4×) | 94 | 94 | -0 % | 1 → 1 |
| Pan, one step (mask outlines shown) | 317 | 314 | -1 % | 1 → 1 |
| Zoom to a rectangle | 116 | 116 | +0 % | 1 → 1 |
| Add a marker channel | 317 | 256 | -19 % | 1 → 1 |
| Show mask outlines | 303 | 295 | -2 % | 1 → 1 |
| Switch FOV (first visit) | 740 | 657 | -11 % | 1 → 1 |
| Switch FOV while zoomed in (first visit) | 732 | 648 | -11 % | 1 → 1 |
| Switch FOV (visited before) | 244 | 240 | -2 % | 1 → 1 |

