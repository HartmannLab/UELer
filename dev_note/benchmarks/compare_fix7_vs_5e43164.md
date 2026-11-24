# UELer before/after — 5e43164 vs working tree

macOS-26.5.2-arm64-arm-64bit · 12 CPUs · 18.0 GB RAM · Python 3.10.20 · matplotlib 3.10.9 · ipympl 0.10.0
Median of 3 runs per version, interleaved. Latency = kernel time from the user's input until the PNG showing the correct image is sent; network transfer and browser decode come on top.

## 4 FOVs, 1024×1024 px, 5 channels, 3,000 cells/FOV · canvas 600 CSS px × 1 = 600 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after | KB sent until correct, before → after |
| --- | --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 52 | 35 | -33 % | 1 → 1 | 356 → 328 |
| Pan, one step (zoomed in 4×) | 33 | 32 | -1 % | 1 → 1 | 282 → 282 |
| Pan, one step (mask outlines shown) | 76 | 38 | -50 % | 1 → 1 | 689 → 589 |
| Pan, sharp frame after letting go (whole FOV) | 15 | 51 | +233 % | 1 → 1 | 1 → 354 |
| Zoom to a rectangle | 37 | 36 | -1 % | 1 → 1 | 288 → 288 |
| Add a marker channel | 70 | 69 | -1 % | 1 → 1 | 305 → 305 |
| Show mask outlines | 70 | 66 | -5 % | 1 → 1 | 554 → 554 |
| Switch FOV (first visit) | 133 | 131 | -2 % | 1 → 1 | 332 → 332 |
| Switch FOV while zoomed in (first visit) | 132 | 130 | -2 % | 1 → 1 | 330 → 330 |
| Switch FOV (visited before) | 61 | 61 | +0 % | 1 → 1 | 332 → 332 |

## 4 FOVs, 1024×1024 px, 5 channels, 3,000 cells/FOV · canvas 600 CSS px × 2 = 1200 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after | KB sent until correct, before → after |
| --- | --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 139 | 31 | -77 % | 1 → 1 | 1283 → 166 |
| Pan, one step (zoomed in 4×) | 30 | 30 | +0 % | 1 → 1 | 150 → 150 |
| Pan, one step (mask outlines shown) | 161 | 40 | -75 % | 1 → 1 | 2038 → 156 |
| Pan, sharp frame after letting go (whole FOV) | 27 | 137 | +403 % | 1 → 1 | 6 → 1299 |
| Zoom to a rectangle | 115 | 115 | +0 % | 1 → 1 | 852 → 852 |
| Add a marker channel | 162 | 161 | -0 % | 1 → 1 | 1126 → 1126 |
| Show mask outlines | 106 | 106 | +0 % | 1 → 1 | 1113 → 1113 |
| Switch FOV (first visit) | 224 | 219 | -2 % | 1 → 1 | 1193 → 1193 |
| Switch FOV while zoomed in (first visit) | 220 | 217 | -1 % | 1 → 1 | 1180 → 1180 |
| Switch FOV (visited before) | 149 | 149 | +0 % | 1 → 1 | 1194 → 1194 |

## 3 FOVs, 2048×2048 px, 10 channels, 12,000 cells/FOV · canvas 600 CSS px × 1 = 600 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after | KB sent until correct, before → after |
| --- | --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 55 | 35 | -36 % | 1 → 1 | 353 → 332 |
| Pan, one step (zoomed in 4×) | 32 | 29 | -11 % | 1 → 1 | 228 → 214 |
| Pan, one step (mask outlines shown) | 100 | 50 | -50 % | 1 → 1 | 792 → 523 |
| Pan, sharp frame after letting go (whole FOV) | 16 | 55 | +239 % | 1 → 1 | 1 → 356 |
| Zoom to a rectangle | 45 | 47 | +4 % | 1 → 1 | 273 → 273 |
| Add a marker channel | 101 | 105 | +4 % | 1 → 1 | 373 → 373 |
| Show mask outlines | 100 | 97 | -3 % | 1 → 1 | 767 → 767 |
| Switch FOV (first visit) | 487 | 470 | -4 % | 1 → 1 | 373 → 373 |
| Switch FOV while zoomed in (first visit) | 468 | 466 | -0 % | 1 → 1 | 362 → 362 |
| Switch FOV (visited before) | 73 | 71 | -3 % | 1 → 1 | 373 → 373 |

## 3 FOVs, 2048×2048 px, 10 channels, 12,000 cells/FOV · canvas 600 CSS px × 2 = 1200 px

| Operation | Before (ms) | After (ms) | Change | Frames until correct image, before → after | KB sent until correct, before → after |
| --- | --- | --- | --- | --- | --- |
| Pan, one step (whole FOV in view) | 201 | 36 | -82 % | 1 → 1 | 1088 → 156 |
| Pan, one step (zoomed in 4×) | 94 | 33 | -65 % | 1 → 1 | 686 → 117 |
| Pan, one step (mask outlines shown) | 310 | 55 | -82 % | 1 → 1 | 2374 → 131 |
| Pan, sharp frame after letting go (whole FOV) | 56 | 188 | +237 % | 1 → 1 | 6 → 1060 |
| Zoom to a rectangle | 116 | 111 | -4 % | 1 → 1 | 843 → 843 |
| Add a marker channel | 248 | 246 | -1 % | 1 → 1 | 1211 → 1211 |
| Show mask outlines | 283 | 286 | +1 % | 1 → 1 | 1790 → 1790 |
| Switch FOV (first visit) | 681 | 637 | -6 % | 1 → 1 | 1176 → 1176 |
| Switch FOV while zoomed in (first visit) | 642 | 613 | -5 % | 1 → 1 | 1110 → 1110 |
| Switch FOV (visited before) | 230 | 224 | -3 % | 1 → 1 | 1175 → 1175 |

