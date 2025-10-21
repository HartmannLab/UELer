# Refactorize the entire app into a Python package [#4](https://github.com/HartmannLab/UELer/issues/4)

## GitHub issue post
The current structure of the UELer app should be refactored to follow best practices for Python packaging. This includes:
- Organizing code into modules and sub-packages where appropriate
- Adding or updating `setup.py` or `pyproject.toml` for package installation
- Ensuring all dependencies are clearly listed
- Moving scripts and main entry points into a package structure
- Updating documentation to reflect new usage and installation instructions
- Verifying that tests still pass after the refactor

**Important:** The refactoring process must ensure that the app's UI and UX remain unchanged for end users. All existing user interface components, workflows, and interactions should be preserved, and there should be no changes in appearance or usability as a result of this restructuring.

This change will improve maintainability, distribution, and future development.

## Plans

### Action plan
1. **Repository assessment (Phase 0)**
	- Inventory existing modules, scripts, and shared utilities to understand current coupling.
	- Map entry points in `script/` notebooks and any CLI-like usage to define packaging requirements.
	- Confirm current test coverage and identify gaps that could break during refactor.

2. **Package skeleton and configuration (Phase 1)**
	- Introduce a `pyproject.toml` (or update if present) with build system metadata and dependencies.
	- Restructure code into a top-level package (e.g., `ueler/`) preserving module boundaries; add `__init__.py` as needed.
	- Provide minimal packaging docs and ensure existing imports remain backward compatible through shims if required.

3. **Script and entry point migration (Phase 2)**
	- Convert reusable notebook logic into scripts or package modules where appropriate, keeping notebooks as thin wrappers.
	- Define console entry points for common workflows and integrate with packaging configuration.
	- Update internal imports to use new package paths; ensure UI components load without behavioral change.

4. **Documentation and dependency alignment (Phase 3)**
	- Update `README.md`, `doc/log.md`, and any usage guides to reflect new package structure and installation steps.
	- Audit dependencies to ensure `environment.yml`, `pyproject.toml`, and tests align; remove redundancies.

5. **Validation and rollout (Phase 4)**
	- Run full automated test suite plus smoke tests for main UI workflows.
	- Prepare migration notes, tag release candidate, and gather stakeholder feedback before final merge.

### Phase 0 findings (2025-10-16)
**Module inventory**
- Top-level helpers: `constants.py`, `data_loader.py`, `image_utils.py` plus placeholder `__init__.py`.
- Core UI package lives under `viewer/` with submodules for UI assembly (`ui_components.py`, `main_viewer.py`, `image_display.py`, `roi_manager.py`, `annotation_*`, `color_palettes.py`, `observable.py`, `decorators.py`).
- Plugins in `viewer/plugin/` provide optional panels (`heatmap_*`, `scatter_widget.py`, `mask_painter.py`, `cell_gallery.py`, `export_fovs.py`, `roi_manager_plugin.py`, `run_flowsom.py`, etc.), all importing shared viewer utilities directly from the root.
- Tests target viewer behavior under `tests/` and rely on locally defined dependency stubs; no packaging metadata (e.g., `pyproject.toml`) exists yet.

**Entry point review**
- Interaction currently exposed through Jupyter notebooks in `script/` (`run_ueler*.ipynb`, `run_viewer*.ipynb`) that extend `sys.path` then construct `viewer.main_viewer.ImageMaskViewer` instances.
- No standalone CLI or module-level `__main__` entry detected; notebooks embed workflow-specific configuration (paths, dataset presets, multi-FOV layouts).

**Test coverage status**
- Running `unittest` discovery (`python -m unittest discover tests`) executes 44 tests; 34 pass, 2 fail, 8 error.
- Failures stem from stubbed dependencies (`pandas` stub returns `object` so DataFrame expectations break) and missing attributes in plugin mocks (e.g., `viewer.plugin.heatmap_layers.display`, `HeatmapDisplay.adapter`).
- Coverage focuses on annotation palettes, ROI manager tags, plugin panel layout, and scatter/heatmap utilities; no tests cover data loaders, notebook flows, or packaging concerns.

**Risks and gaps before Phase 1**
- Heavy cross-module imports assume flat module layout; restructuring will need compatibility shims or incremental refactors.
- Notebook-first entry points complicate packaging; we must decide how to expose equivalent functionality via scripts or console entry points.
- Test suite requires real `pandas`/widget shims or updated mocks; addressing failing tests early will prevent regressions during package moves.

**Phase 1 readiness**
- Proceed with scaffolding a package (`ueler/` namespace, `pyproject.toml`) while planning import aliases to keep notebooks operational.

### Mitigation strategy for Phase 1 risks (notebook-first, VSCode workflow)
This action plan focuses on keeping the Jupyter notebook as the primary UI, minimizing or eliminating CLI surface, and supporting a VSCode-centric developer workflow (local branches and commits rather than external PRs). The aim is low-risk, incremental change with clear checkpoints you can review inside VSCode.
1) Cross-module import coupling — compatibility shims (status: pending design)
- Status: design work not started. Implementation is planned after the test bootstrap is stabilized.
- Goal: Support both existing flat imports (e.g., `from viewer.main_viewer import ImageMaskViewer`) and a new packaged layout (`from ueler.viewer import ImageMaskViewer`) during the migration.
- Key next actions:
	- Draft a compact shim mapping that lists the few high-value modules to export from `ueler.*` (start with `viewer`, `viewer.plugin`, `viewer.ui_components`, `viewer.main_viewer`).
	- Implement a minimal `ueler/__init__.py` and `ueler/viewer/__init__.py` that re-export the current symbols and add a tiny test that imports both paths.
	- Keep `viewer/__init__.py` working as a compatibility shim while moving modules.

2) Notebook-first entry points — runner first, CLI deferred (status: decided)
- Status: we decided to keep notebooks primary. Implement a small `ueler.runner` now as the canonical programmatic entrypoint and defer a CLI until after packaging stabilizes.
- Key next actions:
	- Create `ueler/runner.py` with a single `run_viewer(...)` function signature to hold extracted notebook logic.
	- Add a smoke-test notebook cell that demonstrates `from ueler.runner import run_viewer` and calls it with minimal args.

3) Brittleness in tests — shared bootstrap (status: completed)
- Status: completed. The project uses `tests/bootstrap.py` (imported from `tests/__init__.py`) to provide lightweight stubs for pandas/ipywidgets/scipy/jscatter and heatmap safety patches.
- What was implemented:
	- `tests/bootstrap.py` provides a robust pandas shim fallback and upgrades minimal test-level pandas stubs where found.
	- An ipywidgets stub was expanded with `allowed_tags`, `allow_new`, and other traits used by ROI-manager tests.
	- A jscatter stub and IPython.display no-op were included.
	- Heatmap runtime guards (redraw/restore/highlight gating) were added plus plugin preloading to avoid test-installed minimalist stubs.
	- Full test discovery (`python -m unittest discover tests`) was run to verify the suite in fast-stub mode.

4) Developer workflow and local review (VSCode-first)
- Status: guidance drafted; small infra items remain.
- Key next actions:
	- Add `pyproject.toml` and a small `Makefile` with targets for dev venv creation, `pip install -e .`, fast tests, and optional integration tests.
	- Add a brief developer guide in `README.md` describing the VSCode flow and how to run the fast test suite.

5) Incremental plan for local branches and commits (ordered)
- Checkpoint 1 (skeleton): create `ueler/` package skeleton with minimal `__init__` and shim mapping; add `pyproject.toml` and `Makefile`.
- Checkpoint 2 (shims & tests): implement `ueler` shims and an import-test (`tests/test_shims_imports.py`) that verifies both `viewer` and `ueler.viewer` imports; keep changes small and reversible.
- Checkpoint 3 (move incrementally): move modules (start with lower-risk modules like `viewer/ui_components.py`), run fast tests and notebooks after each small move.
- Checkpoint 4 (runner & notebooks): extract notebook logic into `ueler.runner`, update notebooks to call it, and run smoke tests.
- Checkpoint 5 (swap re-exports): once everything is stable, make `viewer` re-export from `ueler` and clean up legacy shims.

#### Action plan (2025-10-17) — Fast-stub parity
1. Reproduce the fast-stub failures under `python -m unittest discover tests` to catalogue which pandas/matplotlib behaviours are missing.
2. Extend the shared pandas shim with indexing (`loc`/`iloc`), mapping, casting, and API supplementation so downstream tests can execute without the real dependency.
3. Add a lightweight matplotlib stub and harden the plugin preloader so chart/ROI modules load their production implementations before ad-hoc stubs appear.
4. Iterate on the bootstrap until the full fast-test suite passes and document the outcome.

#### Task 3 execution plan (2025-10-17)
- Proposed solution: introduce a minimal `ueler` package skeleton that re-exports the existing `viewer` API while leaving runtime behavior untouched, and scaffold packaging metadata plus developer automation to unblock later moves.
- Implementation steps:
	1. Populate `pyproject.toml` with basic project metadata, build-system settings, and mark dependencies as dynamic for future refinement.
	2. Author a developer-focused `Makefile` with targets for creating a virtual environment, installing in editable mode, running the fast stubbed tests, and reserving an integration test hook.
	3. Fill in `ueler/__init__.py` and `ueler/viewer/__init__.py` with compatibility shims that lazily forward imports to the current `viewer` package without altering behavior.
	4. Run the fast unit test suite to confirm the skeleton and shims do not destabilize the existing codebase.

#### Task 4 action plan (2025-10-17) — Import shim design
- Proposed solution: map the legacy `viewer` module surface onto the new `ueler` namespace while keeping notebooks and plugins fully backward compatible.
- Implementation steps:
	1. Inventory high-traffic imports across `viewer/`, `viewer/plugin/`, and notebooks to shortlist modules and symbols needing immediate shim coverage.
	2. Draft a routing table that pairs the new `ueler.*` module paths with current `viewer.*` implementations, including nested packages (`viewer.plugin`, `viewer.annotation_*`).
	3. Document the shim routing table in `dev_note/github_issues.md`, highlighting phased moves (core viewer, plugins, utilities) and dependencies.
	4. Outline validation expectations (existing fast tests, targeted smoke tests for notebooks) plus follow-up steps for Task 5 implementation work.

#### Task 4 mapping proposal (2025-10-17)
**Goal:** Provide the compatibility matrix that Task 5 will implement so notebooks and plugins can migrate to `ueler.*` without breaking legacy imports.

**Core viewer exports**

| New import | Legacy target | Notes |
| --- | --- | --- |
| `ueler.viewer.main_viewer` | `viewer.main_viewer` | primary notebook entry point (`ImageMaskViewer`). |
| `ueler.viewer.ui_components` | `viewer.ui_components` | used by plugin layout tests. |
| `ueler.viewer.annotation_display` | `viewer.annotation_display` | ROI highlight utilities. |
| `ueler.viewer.annotation_palette_editor` | `viewer.annotation_palette_editor` | palette editor widget. |
| `ueler.viewer.image_display` | `viewer.image_display` | shared canvas rendering helpers. |
| `ueler.viewer.roi_manager` | `viewer.roi_manager` | ROI tagging, selected heavily by tests. |
| `ueler.viewer.color_palettes` | `viewer.color_palettes` | shared color constants. |
| `ueler.viewer.observable` | `viewer.observable` | event bus used across plugins. |
| `ueler.viewer.decorators` | `viewer.decorators` | status-bar helpers referenced by plugins. |

**Plugin surface (high-priority aliases)**

| New import | Legacy target | Notes |
| --- | --- | --- |
| `ueler.viewer.plugin` | `viewer.plugin` | package alias required for `from ... import heatmap`. |
| `ueler.viewer.plugin.plugin_base` | `viewer.plugin.plugin_base` | base class, pulled by most plugin modules. |
| `ueler.viewer.plugin.chart` | `viewer.plugin.chart` | provides `ChartDisplay`. |
| `ueler.viewer.plugin.chart_heatmap` | `viewer.plugin.chart_heatmap` | heatmap draw utilities. |
| `ueler.viewer.plugin.heatmap` | `viewer.plugin.heatmap` | exposes `HeatmapPlugin`. |
| `ueler.viewer.plugin.heatmap_layers` | `viewer.plugin.heatmap_layers` | layered heatmap adapter (used by tests). |
| `ueler.viewer.plugin.heatmap_adapter` | `viewer.plugin.heatmap_adapter` | backend for heatmap plugin. |
| `ueler.viewer.plugin.scatter_widget` | `viewer.plugin.scatter_widget` | scatter panel widget. |
| `ueler.viewer.plugin.cell_gallery` | `viewer.plugin.cell_gallery` | gallery panel. |
| `ueler.viewer.plugin.export_fovs` | `viewer.plugin.export_fovs` | FOV export helper. |
| `ueler.viewer.plugin.mask_painter` | `viewer.plugin.mask_painter` | mask editing UI. |
| `ueler.viewer.plugin.region_annotation` | `viewer.plugin.region_annotation` | region annotation workflow. |
| `ueler.viewer.plugin.roi_manager_plugin` | `viewer.plugin.roi_manager_plugin` | ROI manager integration. |
| `ueler.viewer.plugin.run_flowsom` | `viewer.plugin.run_flowsom` | FlowSOM integration hook. |
| `ueler.viewer.plugin.go_to` | `viewer.plugin.go_to` | FOV navigation helper. |

**Utilities and helpers**

| New import | Legacy target | Notes |
| --- | --- | --- |
| `ueler.constants` | `constants` | top-level constants referenced by notebooks. |
| `ueler.data_loader` | `data_loader` | dataset loading helpers. |
| `ueler.image_utils` | `image_utils` | shared image transforms (used by plugins). |

**Implementation outline for Task 5**
- Introduce a helper (e.g., `ueler/_compat.py`) that registers proxy modules via `importlib.import_module` and `sys.modules` so dotted imports (both `import` and `from ... import ...`) resolve to the legacy modules.
- Populate alias registrations for each mapping above during package import (`ueler/__init__.py` or a new bootstrap hook) without importing heavy modules eagerly.
- Preserve laziness by deferring module imports until first access (`types.ModuleType` wrappers or `LazyLoader`).
- Maintain `__all__` values where needed (notably for `ueler.viewer` and `ueler.viewer.plugin`).

**Validation plan**
- Extend `tests/test_shims_imports.py` to exercise each alias (core viewer, plugin, and utility modules) via both `import` and `from ... import ...` patterns.
- Run `make test-fast` to confirm the shim registry does not perturb existing tests.
- Execute a notebook smoke test (manual or scripted) that imports `ueler.viewer.main_viewer.ImageMaskViewer` and a representative plugin module to ensure runtime wiring remains intact.

**Follow-up when migrating modules**
- When a module physically moves into `ueler`, keep the alias pointing at the new location and backfill a reverse alias into `viewer` until the notebooks are updated.
- Track alias usage (via logging or optional warnings) once migration reaches Checkpoint 3 to identify remaining legacy importers.

Acceptance criteria before merging the local branch into main:
- Fast-stub unit tests pass in CI (or failing tests are documented and marked as integration tests).
- Representative notebooks run unchanged (aside from replacing `sys.path` hacks with `import ueler` where applicable) and display the viewer as before.
- `pip install -e .` succeeds in the dev venv and `import ueler` works in notebooks.

CI / test matrix recommendation (small, practical)
- Add two CI jobs: `fast-stub` (runs on pull requests) and `integration` (runs nightly or on-demand). This keeps PR feedback fast while ensuring integration coverage.

Notes and assumptions
- Notebook-first remains the default UX; a user-facing CLI is deferred until the package is stable.
- Tests that require full GUI or compiled extensions will be separated and executed in the `integration` job.

I have marked the mitigation doc task completed in the todo list and the bootstrap hardening complete. The next immediate work item is to design and implement the `ueler` import shims and a tiny `ueler/runner` skeleton.


#### Task 5 action plan (2025-10-17) — Import shim implementation
- Proposed solution: implement a reusable lazy-alias registry that wires the `ueler` namespace to existing `viewer` and root modules without importing heavy dependencies until first access.
- Implementation steps:
	1. Introduce `ueler/_compat.py` supplying a `register_lazy_alias` helper plus the mapping tables approved in Task 4.
	2. Update `ueler/__init__.py` to register top-level aliases (`constants`, `data_loader`, `image_utils`) and expose a public `ensure_compat_aliases()` hook for tests.
	3. Expand `ueler/viewer/__init__.py` to load the Task 4 viewer and plugin mappings, delegating alias registration to the shared helper while preserving existing lazy attribute lookup.
	4. Add `tests/test_shims_imports.py` to iterate through the mapping, asserting both `import module` and `from module import symbol` patterns resolve to the legacy modules.
	5. Run `python -m unittest tests.test_shims_imports` and `python -m unittest discover tests` under the fast-stub bootstrap to confirm shim coverage.

#### Task 7 action plan (2025-10-17) — First incremental module move
- Proposed solution: migrate `viewer/ui_components.py` into the `ueler.viewer` package while keeping legacy imports functional through alias shims and a thin compatibility module.
- Implementation steps:
	1. Copy `viewer/ui_components.py` into `ueler/viewer/ui_components.py`, updating intra-package imports to rely on the `ueler.viewer` namespace.
	2. Replace the legacy `viewer/ui_components.py` with a compatibility wrapper that delegates to the new module and preserves `sys.modules` identity.
	3. Extend `_compat.py` to alias `viewer.ui_components` to `ueler.viewer.ui_components`, keeping legacy import paths operational.
	4. Confirm `tests/test_shims_imports.py` still validates the alias matrix and adapt expectations if needed.
	5. Run the fast test suite to verify the migration leaves behaviour unchanged.

#### Task 7a action plan — Risk-sorted module migration (2025-10-17)
Goal: to incrementally move modules from the legacy `viewer.*` layout into `ueler.viewer.*`. The ordering below prioritises low-dependency, high-test-coverage modules first, and delays heavy, GUI- or external-library-dependent modules until the shims and tests are stable.

Acceptance: For each module: copy → update internal imports → add lightweight compatibility wrapper → run fast tests. This process should not stop until item 22 in the **progress** section of **7a. Further incremental module moves** in `dev_note/Todos.md` is checked.

Ordered list (low risk -> high risk)

a. `viewer.color_palettes` — constants and small helpers; few deps and used widely by UI code. Quick wins and minimal runtime surface.
b. `viewer.decorators` — tiny helpers used across the viewer; low coupling and easy to port.
c. `viewer.observable` — event-bus / pub-sub used by plugins; move early so other modules can import the new namespace for wiring.
d. `viewer.annotation_palette_editor` — UI-centric but relies on `ipywidgets`, which the bootstrap stubs cover; medium-low risk.
e. `viewer.annotation_display` — display helpers; moderate risk (uses image buffers) but covered by tests that can detect regressions early.
f. `viewer.roi_manager` — core ROI logic (persistence/CSV) with moderate external I/O coupling; move before the ROI-plugin so plugin code can import updated API.
g. `viewer.plugin.plugin_base` — base plugin class; move before other plugin modules so they can import the new base location without changes.
h. `viewer.plugin.export_fovs`, `viewer.plugin.go_to`, `viewer.plugin.cell_gallery` — small, self-contained plugins with light dependencies; low-to-medium risk and good for validating plugin wiring.
i. `viewer.plugin.chart` — charting glue (matplotlib) used by many tests; medium-to-high risk but important to expose under `ueler.viewer.plugin` early so chart consumers can migrate.
j. `viewer.plugin.scatter_widget` — depends on `jscatter` (or stub); higher risk due to optional front-end integration.
k. `viewer.image_display` and `viewer.images` — image rendering and helper resources; medium-to-high risk (may touch cv2/numpy and file IO).
l. `viewer.plugin.heatmap_adapter`, `viewer.plugin.heatmap_layers`, `viewer.plugin.chart_heatmap`, `viewer.plugin.heatmap` — heatmap stack with scipy/seaborn dependencies and complex drawing logic; high risk, move after chart and basic image plumbing are stable.
m. `viewer.plugin.mask_painter` — mask editing UI with heavy widget interactions; high risk.
n. `viewer.plugin.roi_manager_plugin` — plugin that integrates ROI manager into the UI; move after `viewer.roi_manager` and core plugin base are relocated.
o. `viewer.plugin.run_flowsom` — FlowSOM integration and heavy analysis code; high risk and optional, keep for last (or leave as plugin-only until CI integration is ready).
p. `viewer.main_viewer` — the notebook entrypoint and orchestrator; highest risk (complex wiring, many cross-imports). Migrate only after most dependencies and plugins are moved and the `ueler` shims are fully exercised.

Migration checklist (repeat per module)
- Copy module to `ueler/viewer/<module>.py` and update intra-project imports to `ueler.*` where applicable.
- Add a tiny compatibility wrapper at the original path (e.g., `viewer/<module>.py`) that forwards to the new module via `sys.modules` or attribute delegation.
- Register an alias in `ueler/_compat.py` if necessary so `viewer.<submodule>` is resolved to `ueler.viewer.<submodule>` during the transition.
- Run `python -m unittest tests.test_shims_imports` and `python -m unittest discover tests` under the fast-stub bootstrap and fix any regressions.
- Move to the next module in the ordered list until item 22 in the **progress** section of **7a. Further incremental module moves** in `dev_note/Todos.md` is checked.


Edge cases & notes
- External libs: modules that import heavy native libs (OpenCV, SciPy, FlowSOM bindings) should be moved late; keep the alias in place so notebooks using old import paths remain functional during migration.
- Test stubs: some tests inject their own stub modules (e.g., `viewer.plugin.chart`) — the verification tests will skip strict equivalence checks when stubs are present. During a move, prefer running the full fast suite to expose regressions.
- Backwards compatibility: maintain both forward (`ueler.*`) and legacy (`viewer.*`) paths via the alias registry during the entire migration to avoid breakage in notebooks and plugins.

## Todo Checklist for Packaging Plan
### **Checkpoint 1 — Skeleton and Documentation Setup**
**Goal:** Establish a working package baseline and document the mitigation plan.

- [x] **1. Mitigation strategy doc**  
  - **Description:** Draft and record the mitigation strategy for Phase 1 risks (notebook-first, compatibility shims).  
  - **Files:** `doc/log.md`, `README.md`  
  - **Acceptance:** Both updated and committed.

- [x] **2. Package skeleton**  
  - **Description:** Create the `ueler/` package with minimal structure and shims; add `pyproject.toml` and `Makefile`.  
  - **Acceptance:**  
    - `pip install -e .` succeeds.  
    - `import ueler` works.

- [x] **3. Harden test bootstrap**  
  - **Description:** Improve `tests/bootstrap.py` with fast-stub fallbacks for `pandas`, `ipywidgets`, `jscatter`, and `matplotlib`.  
  - **Acceptance:**  
    - `python -m unittest discover tests` passes in fast-stub mode.  

---

### **Checkpoint 2 — Import Shims Design & Implementation**
**Goal:** Design and implement compatibility shims for legacy imports.

- [x] **4. Design import shims**  
  - **Description:** Draft a compact shim mapping that re-exports high-value modules (`viewer`, `viewer.plugin`, `viewer.main_viewer`) from `ueler.*`.  
  - **Output:** Mapping proposal recorded in `dev_note/github_issues.md` (see "Task 4 mapping proposal").  
  - **Acceptance:** Proposal committed and ready for review.

- [x] **5. Implement import shims + tests**  
  - **Description:** Add lazy import re-exports per the approved design.  
  - **Tests:** Create `tests/test_shims_imports.py` verifying both `viewer` and `ueler.viewer` import paths.  
  - **Acceptance:** Shim tests pass in fast-stub mode.

---

### **Checkpoint 3 — Incremental Module Moves**
**Goal:** Safely refactor modules into the new `ueler.*` namespace.

- [x] **7. Incremental module moves**  
  - **Description:** Move low-risk modules one by one (starting with `viewer/ui_components.py`). Update internal imports accordingly.  
  - **Acceptance:**  
    - All fast tests remain green after each move.  
    - Commit after each move for traceability.

- [x] **7a. Further incremental module moves**
  - **Description:** Move modules one by one according to the order and the plan in `dev_note/github_issues.md`. Don't stop until item 22 in the **progress** list below is checked.
  - **Details & rationale:** See the full action plan in `dev_note/github_issues.md` under the "Task 7a — Risk-sorted module migration order (2025-10-17)" section for the ordered list, checklist, and edge cases.
  - **Acceptance:** 
    - All fast tests remain green after each move.
    - All moves are completed according to the plan.
  - **Progress:**
  Don't stop until item 20 is checked.
    1. [x] `viewer.color_palettes` → `ueler.viewer.color_palettes` (2025-10-17)
    2. [x] `viewer.decorators` → `ueler.viewer.decorators` (2025-10-17)
    3. [x] `viewer.observable` → `ueler.viewer.observable` (2025-10-18)
    4. [x] `viewer.annotation_palette_editor` → `ueler.viewer.annotation_palette_editor` (2025-10-18)
    5. [x] `viewer.annotation_display` → `ueler.viewer.annotation_display` (2025-10-18)
    6. [x] `viewer.roi_manager` → `ueler.viewer.roi_manager` (2025-10-18)
    7. [x] `viewer.plugin.plugin_base` → `ueler.viewer.plugin.plugin_base` (2025-10-18)
    8. [x] `viewer.plugin.export_fovs` → `ueler.viewer.plugin.export_fovs` (2025-10-18)
    9. [x] `viewer.plugin.go_to` → `ueler.viewer.plugin.go_to` (2025-10-18)
    10. [x] `viewer.plugin.cell_gallery` → `ueler.viewer.plugin.cell_gallery` (2025-10-18)
    11. [x] `viewer.plugin.chart` → `ueler.viewer.plugin.chart` (2025-10-18)
    12. [x] `viewer.plugin.scatter_widget` → `ueler.viewer.plugin.scatter_widget` (2025-10-18)
    13. [x] `viewer.image_display` → `ueler.viewer.image_display` (2025-10-18)
    14. [x] `viewer.images` → `ueler.viewer.images` (2025-10-18)
    15. [x] `viewer.plugin.heatmap_adapter` → `ueler.viewer.plugin.heatmap_adapter` (2025-10-18)
    16. [x] `viewer.plugin.heatmap_layers` → `ueler.viewer.plugin.heatmap_layers` (2025-10-18)
    17. [x] `viewer.plugin.chart_heatmap` → `ueler.viewer.plugin.chart_heatmap` (2025-10-18)
    18. [x] `viewer.plugin.heatmap` → `ueler.viewer.plugin.heatmap` (2025-10-18)
    19. [x] `viewer.plugin.mask_painter` → `ueler.viewer.plugin.mask_painter` (2025-10-18)
    20. [x] `viewer.plugin.roi_manager_plugin` → `ueler.viewer.plugin.roi_manager_plugin` (2025-10-18)
    21. [x] `viewer.plugin.run_flowsom` → `ueler.viewer.plugin.run_flowsom` (2025-10-18)
    22. [x] `viewer.main_viewer` → `ueler.viewer.main_viewer` (2025-10-18)

---

### **Checkpoint 4 — Runner & Notebooks**
**Goal:** Extract the notebook entrypoint logic into a clean interface.

- [x] **6. Create `ueler.runner`**  
  - **Description:** Add `ueler/runner.py` exposing `run_viewer(...)`.  
  - **Demo:** Smoke-test notebook cell showing import and invocation.  
  - **Acceptance:**  
    - `from ueler.runner import run_viewer` works in a notebook.  
    - Smoke test added and passes.

---

### **Checkpoint 5 — Final Integration & CI**
**Goal:** Prepare CI and integration-test strategy, finalize shims, and finish documentation.

- Planned: add a CI `fast-stub` job
  - Description: create a `fast-stub` workflow (GitHub Actions or equivalent) that will run the fast-stub test suite on incoming PRs and report results.
  - Acceptance criteria: PRs will show the fast-stub job status and failures will be visible in the PR checks.

- Planned: define integration test triage and job
  - Description: define and add an `integration` workflow for GUI and heavy-dependency tests (scheduled and/or manual runs). Document the cadence and resource requirements.
  - Acceptance criteria: an integration job will be available to run nightly or on demand and will produce logs/artifacts for troubleshooting.