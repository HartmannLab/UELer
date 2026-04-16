# Packaging & Project Structure

> Source: [`dev_note/topic_packaging_and_project.md`](https://github.com/HartmannLab/UELer/blob/main/dev_note/topic_packaging_and_project.md)

---

## Context

UELer has been refactored from a notebook-first script layout into a proper Python package (`ueler/`), while keeping notebook-based usage as the primary interface.

---

## Key Decisions

- **Notebooks as primary entrypoint.** `ueler.runner.run_viewer` provides a programmatic entrypoint, but the main audience uses `script/run_ueler.ipynb`.
- **Compatibility shims.** Legacy `viewer.*` imports continue to work during the transition via `_AliasModuleFinder` in `ueler/_compat.py`.
- **Editable install.** `pip install -e .` is the recommended install mode; it makes `git pull` upgrades instant.
- **Fast-stub test bootstrap.** `tests/bootstrap.py` stubs out heavy dependencies (`pandas`, `ipywidgets`, `matplotlib`) so the test suite runs quickly without a full environment.

---

## Package Layout

```
ueler/
├── __init__.py          # Public API surface
├── _compat.py           # Legacy import shims
├── image_utils.py       # Image helper functions
├── runner.py            # Programmatic entrypoint
└── viewer/
    ├── __init__.py
    ├── main_viewer.py
    ├── ui_components.py
    ├── plugin/
    │   ├── export_fovs.py
    │   ├── chart.py
    │   ├── heatmap.py
    │   └── ...
    └── images/          # Bundled UI icons
```

---

## Current Status

- `ueler/` package skeleton, `pyproject.toml`, and `Makefile` are in place.
- Import shims are implemented and tested (`tests/test_shims_imports.py`).
- All module moves from `viewer.*` → `ueler.viewer.*` are complete.
- `ueler.image_utils` is restored as a real packaged module (post-cleanup regression fix).

---

## Open Items

- Define and add a CI fast-stub job.
- Add an integration test workflow for heavier dependencies and GUI paths.

---

## Related Issues

- [#79 — Package UELer as a pip package](https://github.com/HartmannLab/UELer/issues/79)
- [#4 — Packaging plan](https://github.com/HartmannLab/UELer/issues/4)
