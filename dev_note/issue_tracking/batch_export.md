# Batch Export Design Document

## Original Issue: [#1](https://github.com/your-repo/issues/1)
### Feature Request: Export Modes for Image Export Plugin

Enhance the image export plugin to support multiple export modes:

#### Modes:
- **Full FOV**: Export the entire field of view as an image.
- **Single Cells**: Export a set of cells based on cell table filtering. Users can select/filter cells and export them individually or as a batch.
- **ROIs**: Export entities defined in the ROI manager. Allow:
  - Export respecting each ROI's own settings (image size, marker set, scale bar)
  - Optionally impose shared settings across all ROIs (e.g., image size range +/-100 pixels, marker set, scale bar with dynamic or fixed sizes)

#### Additional Details:
- Provide UI/CLI controls to select the desired export mode.
- Allow configuration of export parameters per mode.
- Ensure compatibility with scale bar and marker set features.
- Allow batch export and customizable output settings.

This feature will make the export functionality more flexible and suited to different analysis workflows.

## Current structure:
The codebase already contains a minimal placeholder plugin and a few helper functions that can be used as the basis for a proper batch export feature. Key pieces to be aware of:

- `ueler/viewer/plugin/export_fovs.py` — currently a placeholder plugin (exposed via the packaged namespace) whose UI simply displays the message "Batch export will be available soon." It intentionally provides only a stub PluginBase subclass so the tab can be shown without functionality.

- `ueler/viewer/main_viewer.py::ImageMaskViewer.export_fovs_batch` — a procedural helper method that can export multiple FOVs programmatically. It accepts a marker set name, optional output directory, a list of FOVs (or single FOV), file format, downsample factor, DPI and figure size. The function:
  - validates the marker set exists and creates an output directory
  - saves and restores the viewer UI state (selected FOV, channels, contrast/mask settings)
  - loops over FOVs, calls `load_fov(...)` and `render_image(...)` to generate combined images, and writes them using `skimage.io.imsave`
  - returns a dict mapping each FOV to True (success) or a string error message

- `ueler/viewer/roi_manager.py::ROIManager.export_to_csv` — an existing export helper for ROI tables used by the ROI manager plugin. This demonstrates the project's export pattern for tabular ROI data (the ROI manager plugin wires a UI export button to call this helper and save CSVs).

Limitations / why the helpers are not practical yet:

- UX and integration: the placeholder plugin provides no UI controls to drive `export_fovs_batch`. There is no visible tab UI to choose marker sets, configure output options, or start/monitor batch exports from the viewer.

- Granularity & export modes: `export_fovs_batch` is focused on exporting full-FOV raster images for a marker set. It doesn't directly support the other export modes requested in the design doc (per-cell crops/tiles, per-ROI exports with per-ROI settings, or imposing a shared settings profile across ROIs).

- Per-ROI/per-cell rendering: the current rendering pipeline (via `render_image`) and `export_fovs_batch` assume a full-FOV composite render. There are no high-level helpers yet to render arbitrary bounding boxes, crop around individual cells, or apply dynamic per-export scale bars and marker overlays.

- Robustness and progress: the helper is synchronous and prints progress via IPython `clear_output`/`print`. For large batches it will block the UI and offers no cancellation or background job handling. Error handling is coarse-grained (stores exception string for a failed FOV) and there is limited logging.

- Performance and memory: `export_fovs_batch` uses the normal viewer caches and rendering code; exporting many large FOVs (or many small crops) may hit memory limits. There are commented dask/distributed lines in the file, but no integrated parallel/export worker implementation.

These existing artifacts are a useful starting point: `export_fovs_batch` already demonstrates the export loop, state save/restore, and image write logic, while the placeholder plugin shows where a user-facing UI should be mounted. The next steps are to design a UI and control model that exposes the requested modes (Full FOV / Single Cells / ROIs), add helpers for crop-based rendering and per-ROI overrides, and wire the UI to run exports asynchronously or with cancellable progress reporting.

## Implementation Plan

This plan describes a phased migration from the current minimal placeholder and helper functions to a user-facing, robust Batch Export UX that supports the requested modes (Full FOV, Single Cells, ROIs). It prioritizes low-risk incremental changes that let us ship value early and iterate.

### High-level contract (what each piece should provide)
- [X] UI plugin (`viewer/plugin/export_fovs.py`): inputs are user-chosen export mode, marker set, output folder, format, optional per-export overrides (crop size, scale bar, marker overlay); outputs are a started export job handle and observable status updates (progress, success/failure per item). The UI should also expose an explicit option to include or omit scale bars for each export mode.
- [ ] Core helpers (`main_viewer.py`, `image_utils.py`): pure-ish functions that render a requested FOV or bounding box to an image array given a marker set and override settings. They should not depend on widget state and must return a serializable result or raise an informative exception.
- [ ] Job runner: a small orchestration layer that schedules export tasks (single-threaded worker initially, pluggable to use threads/processes or Dask later), reports progress, supports cancellation, and persists results to disk.

### Phase 0 — Prep & discovery (quick wins, low risk)
- [X] Add documentation notes (this file) summarizing current code and plan.

### Phase 1 — Make rendering reusable and testable (core refactor)
Goals: separate rendering logic from UI state so exports can be invoked from multiple entry points.

#### Phase 1 Action Plan (2025-10-21)
- [x] Capture current `export_fovs_batch` behaviour with smoke tests covering:
  - [x] Success path
  - [x] Missing-channel failure mode
- [x] Extract pure rendering helpers (`render_fov_to_array`, `render_crop_to_array`, `render_roi_to_array`) so downstream code no longer depends on widget state.
  - [x] Implement these in `ueler/image_utils.py` or `ueler/viewer/rendering.py` with signatures such as:
    - `render_fov_to_array(fov_name, marker_set, downsample_factor, xym=None, xym_ds=None, mask_options=None, scale_bar=None) -> np.ndarray`
    - `render_crop_to_array(fov_name, marker_set, center_xy, size_px, downsample_factor, ...) -> np.ndarray`
    - `render_roi_to_array(fov_name, roi_definition, marker_set, overrides...) -> np.ndarray`
  - [x] Ensure renderer functions return image arrays without displaying figures; any Matplotlib objects must render off-screen or be closed.
- [x] Update `ImageMaskViewer.render_image` to delegate to the new renderer and maintain backward compatibility via a thin shim.
- [x] Back the renderer refactor with unit tests using synthetic fixtures to cover:
  - [x] Colour blending
  - [x] Annotation overlays
  - [x] Mask outlines

### Phase 2 — Build export job runner and API (non-UI)
 Goals: provide a programmatic API that runs exports, reports progress, and supports cancellation.
- [x] Create `ueler/export/job.py` with a simple `Job` class:
  - [x] Inputs: mode, items (list of fov / cell / ROI descriptors), marker_set, output_dir, file_format, overrides
  - [x] Methods: `start()`, `cancel()`, `status()` — `status()` returns per-item states and overall progress
  - [x] Implementation: `start()` runs a worker loop that calls the renderer functions and writes files, catching exceptions and updating per-item results
- [x] Wire `ImageMaskViewer.export_fovs_batch` to use the new Job runner internally (backwards compatible) or provide a thin adapter.
- [x] Add logging and structured error reporting (error type, trace) rather than only string messages.

#### Phase 2 Action Plan (2025-10-21)
- Finalise `Job` lifecycle design (states, progress accounting, structured result payload) and capture it inside the new module docstring.
- Implement synchronous `start()` that iterates items serially while yielding status updates through an observable callback hook; defer multi-threading to Phase 5.
- Define serialisable `ExportResult` records capturing `ok`, `error`, `traceback`, and output path metadata for each item.
- Extend the job runner with `cancel()` and idempotent shutdown semantics so UI threads can stop work safely ahead of future background integration.
- Refactor `ImageMaskViewer.export_fovs_batch` to construct a job instance, subscribe to progress, and return the structured results while keeping the public signature intact.
- Replace ad-hoc `print`/`clear_output` logging with a shared logger plus optional progress callback plumbing, and surface machine-readable error dictionaries for downstream consumers.

### Phase 3 — UI plugin & UX
Goals: provide a user-facing tab to select mode, configure options and start/cancel jobs with progress UI.
- [x] Expand `ueler/viewer/plugin/export_fovs.py` to a full plugin class (rename stub `RunFlowsom` to `BatchExportPlugin` or similar):
  - [x] Controls: mode selector (Full FOV / Single Cells / ROIs), marker set dropdown, output folder chooser (text + browse), file format selector, per-mode options (cell table filter input, ROI selection widget, image size controls, scale bar options), Start and Cancel buttons
  - [x] Status/Output area: job progress bar, per-item log, link to output folder
  - [x] When Start is clicked, create a `Job` and call `start()` in a background thread (or `concurrent.futures.ThreadPoolExecutor`) and subscribe to status updates to refresh UI
  - [x] Implement a small UX for Single Cells: allow applying a cell table filter string or selecting a subset; provide a preview button that renders one sample crop using the renderer API
  - [x] Add explicit controls for scale bar inclusion per-export and a global default. (Computation of recommended scale bar length remains planned for Phase 4.)
- [x] Ensure the plugin runs without blocking the main UI; use IPython's event loop-friendly approaches when updating widgets (use `threading` and send status updates back to the main thread via `ipywidgets.Output`/traitlets safely).

#### Phase 3 Action Plan (2025-10-21)
- [x] Replace `RunFlowsom` stub with `BatchExportPlugin` implementing `PluginBase` contract while preserving tab identity.
- [x] Build UI layout with shared controls (mode, marker set, output path, format, scale bar defaults) and per-mode option containers using traitlets to toggle visibility.
- [x] Integrate job runner by wiring Start/Cancel buttons to create background `Job` instances using `ThreadPoolExecutor` and forward progress via traitlets observers.
- [x] Implement per-mode data binding: Full FOV uses selected FOVs, Single Cells consumes filter text and preview, ROIs binds to ROI manager selection widget.
- [x] Surface progress/log output via `ipywidgets.Output` and status bar, ensuring thread-safe updates on the main event loop.

### Phase 3a — Supporting masks and annotations in exports
Goals: Enable users to include or exclude mask and annotation overlays in exported images, reusing the existing viewer state wherever possible.
- [x] Capture the current mask and annotation settings from the viewer when initiating an export job.
  - [x] Define an overlay snapshot structure that can be reused by the plugin workers and new tests.
  - [x] Add a `capture_overlay_snapshot` helper on the viewer that records annotation state, mask toggles, and colours.
- [x] Update renderer functions to accept options for mask and annotation overlays (e.g., colors, visibility, outlines).
  - [x] Expand mask/annotation render dataclasses if additional attributes (alpha, outline mode) are required by the snapshot.
  - [x] Ensure helper methods gracefully handle missing overlays using the snapshot inputs.
- [x] Extend the plugin UI to let users choose whether to include masks and annotations in their exports.
  - [x] Add opt-in checkboxes for masks/annotations and wire them into the job-builder pipeline.
  - [x] Display an inline hint when masks or annotations are unavailable for the selected dataset.
- [x] Ensure the export job runner correctly passes these options to the renderer.
  - [x] Thread the overlay snapshot through job overrides and worker call sites.
  - [x] Back the new flow with unit tests that assert overlays are respected for FOV, cell, and ROI exports.

### Phase 3b —  Fixing the mask outline rendering
Goals: Ensure that mask outlines are rendered for each cell correctly, not just based on a binary mask of presence/absence. Also the thickness of the outline should be adjustable.

#### Phase 3b Action Plan (2025-10-22)
- [x] Review the current mask overlay pipeline (viewer rendering helpers, snapshot wiring, plugin usage) to confirm where label data is discarded and how outline arrays are constructed today.
- [x] Update renderer logic so outline mode operates on per-label segmentation masks, supports configurable outline thickness, and gracefully degrades when `skimage` is unavailable.
- [x] Thread the outline thickness setting through `MaskOverlaySnapshot`, `MaskRenderSettings`, and the viewer/plugin plumbing so exports and on-screen previews remain consistent.
- [x] Add an outline thickness control to the Batch Export plugin UI (including sensible defaults and validation) and sync it with the viewer overlay settings when applicable.
- [x] Extend the automated test suite to cover per-cell outline rendering across multiple thicknesses (unit tests for the renderer helpers + an integration test for the export job path).
- [x] Update user-facing documentation (`README.md`, `doc/log.md`) and implementation notes (`dev_note/github_issues.md`) after landing the feature.

### Phase 3c — Fixing Mask Outline Rendering (Again)
Now the main viewer and the batch export plugin share the same rendering logic for mask outlines, but both are not rendering outlines per cell correctly. All cells are still being merged into a few large patches instead of each cell having its own individual outline.

Goals:
Individually outline each cell in both the main viewer and the batch export plugin. Additionally, introduce a separate control for adjusting the outline thickness in the main viewer. The outline thickness setting in the batch export plugin should remain local to that plugin, rather than affecting the global settings in the main viewer (but take the global setting by default).

- [x] Update the main viewer's rendering logic to ensure that each cell is outlined individually, rather than merging multiple cells into a single outline patch.
- [x] Add a dedicated control in the main viewer's UI for adjusting the outline thickness of masks.
- [x] Ensure that the batch export plugin continues to use its own outline thickness setting, independent of the main viewer's setting, while defaulting to the main viewer's current thickness value when the plugin is opened.
- [x] Extend the test suite to validate that both the main viewer and the batch export plugin correctly render individual cell outlines with the specified thickness settings.

#### Phase 3c Action Plan (2025-10-22)
- Audit the existing mask outline pipeline (`ueler/viewer/rendering.py`, `MaskRenderSettings`) to pinpoint where label IDs collapse into merged outlines and document current data flow.
- Refactor the renderer to compute per-label contours (e.g., leveraging `skimage.segmentation.find_boundaries`) and thread a configurable thickness parameter through viewer and export paths without regressing fill/alpha behaviour.
- Introduce a persistent outline thickness control in the main viewer UI (likely an `ipywidgets.FloatSlider`) backed by viewer state, ensuring it updates live preview rendering.
- Update the batch export plugin to seed its local thickness control from the viewer value on activation while preserving independent adjustments during a session.
- Expand automated coverage: add unit tests for per-cell outline generation, viewer regression tests for the new control, and export integration tests to confirm plugin isolation.

### Phase 4 — Scale bars support
Goals: support scale bars.
- [ ] Scale bar sizing and PDF format behavior:
  - [ ] When a scale bar is requested, the export pipeline must compute a scale bar length such that the drawn length is <= 10% of the exported image width in pixels. Choose a round/clean physical length (e.g., 1 µm, 2 µm, 5 µm, 10 µm, etc.) that satisfies this constraint and is appropriate for the image resolution and DPI.
    - [ ] Refactor how the viewer handle scale bars to expose a helper that computes recommended scale bar lengths given image pixel size, physical size, and desired max fraction of image width.
    - [ ] Ensure both the export job runner and the main viewer calls this helper for displaying the scale bar.
  - [ ] For non-PDF formats (PNG, JPEG, TIFF), draw the scale bar directly onto the exported image at the bottom-right corner using Matplotlib's `AnchoredSizeBar` or similar logic.
  - [ ] For PDF exports, draw the scale bar as part of the Matplotlib figure prior to saving to PDF, ensuring it scales correctly with figure size and DPI.
- [ ] Add a small helper to compute scale bar pixel size for exported DPI/figure size and optionally draw it into images (reuse Matplotlib `AnchoredSizeBar` logic).


### Phase 5 — Performance, parallelism, and reliability
Goals: make large batch runs efficient and robust.
- [ ] Support configurable parallelism in the Job runner (threads or processes). Start with a thread pool for IO-bound tasks; if CPU-bound rendering is observed, add a process pool or Dask integration.
- [ ] Implement chunked/streamed IO to avoid holding many full-resolution images in memory simultaneously.
- [ ] Add retry policies and rate limiting for heavy IO scenarios.

### Phase 6 — Tests, docs, and release
- [ ] Add tests for Job runner (small batch exports using temporary folders and small fixtures). Include tests for cancellation and error reporting.
- [ ] Update `doc/log.md` and `README.md` with a brief section describing the new Batch Export UI and CLI hooks.
- [ ] Add a small example notebook under `script/` demonstrating the programmatic API (how to run a `Job` from a notebook with progress output).

### Edge cases and concerns to handle during work
- Missing channels or missing FOVs: renderers should raise explicit exceptions the Job runner can classify (recoverable vs fatal).
- Large/bad masks or corrupted images: catch and continue with retries or mark failed items without crashing the whole job.
- Concurrent modifications: if users change marker sets or UI state while export is running, ensure the export uses the snapshot of settings captured at job start.
- Permissions and disk space: detect and report write permission or out-of-space errors early when possible.

### Low-risk extras to add during the refactor
- A small preview button in the plugin to render one sample item (FOV/cell/ROI) so users can validate settings before running the full batch.
- A `--dry-run` flag on the Job API that validates inputs and emits the list of planned output paths without writing files.

### Acceptance criteria (when to call this done)
- The viewer shows a Batch Export tab where users can choose mode, configure options, start a job, and see progress.
- Exports for all three modes (Full FOV, Single Cells, ROIs) are supported end-to-end (image files written and matching requested settings).
- Exports are cancellable and do not block the main viewer UI.
- Tests cover core functionality and CI runs them (basic unit tests + a small integration smoke test).



  - [ ] Implement the PDF gallery output as an optional output format for ROI batches, where individual ROI images are placed into an arrayed PDF (one ROI per page or arranged in a grid). Each ROI entry in the PDF should include its own scale bar (if requested) and caption/metadata (ROI id, source FOV, export settings).
  - [ ] When exporting multiple ROIs into a single gallery PDF (future feature), each ROI may have a different scale bar appropriate for its crop size; the Job runner should support generating a per-ROI scale bar and render each ROI to a PDF page or grid cell. The gallery exporter should accept layout parameters (rows x cols) and pagination behavior.