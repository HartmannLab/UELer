## **Checkpoint 1 — Skeleton and Documentation Setup**
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

## **Checkpoint 2 — Import Shims Design & Implementation**
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

## **Checkpoint 3 — Incremental Module Moves**
**Goal:** Safely refactor modules into the new `ueler.*` namespace.

- [x] **7. Incremental module moves**  
  - **Description:** Move low-risk modules one by one (starting with `viewer/ui_components.py`). Update internal imports accordingly.  
  - **Acceptance:**  
    - All fast tests remain green after each move.  
    - Commit after each move for traceability.

- [ ] **7a. Further incremental module moves**
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
  Don't stop until item 22 is checked.
  22. [x] `viewer.main_viewer` → `ueler.viewer.main_viewer` (2025-10-18)

---

## **Checkpoint 4 — Runner & Notebooks**
**Goal:** Extract the notebook entrypoint logic into a clean interface.

- [ ] **6. Create `ueler.runner`**  
  - **Description:** Add `ueler/runner.py` exposing `run_viewer(...)`.  
  - **Demo:** Smoke-test notebook cell showing import and invocation.  
  - **Acceptance:**  
    - `from ueler.runner import run_viewer` works in a notebook.  
    - Smoke test added and passes.

---

## **Checkpoint 5 — Final Integration & CI**
**Goal:** Complete documentation updates, finalize shims, and add CI coverage.

- [ ] **8. Update docs and release notes**  
  - **Description:** Update `doc/log.md`, `README.md`, and append entries to `dev_note/github_issues.md`.  
  - **Acceptance:** Docs reflect latest changes and version notes.

- [ ] **9. CI `fast-stub` job**  
  - **Description:** Add CI job `fast-stub` to run the fast test suite on PRs.  
  - **Acceptance:** PRs show fast-stub test results.

- [ ] **10. Integration test triage**  
  - **Description:** Define and add `integration` CI job for GUI and heavy dependencies; document cadence.  
  - **Acceptance:** Integration job runs nightly or on demand.