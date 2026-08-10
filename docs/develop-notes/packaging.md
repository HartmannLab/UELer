# Packaging & Project Structure

> Source: [`dev_note/topic_packaging_and_project.md`](https://github.com/HartmannLab/UELer/blob/main/dev_note/topic_packaging_and_project.md)

---

## Context

UELer has been refactored from a notebook-first script layout into a proper Python package (`ueler/`), while keeping notebook-based usage as the primary interface.

---

## Key Decisions

- **Notebooks as primary entrypoint.** `ueler.runner.run_viewer` provides a programmatic entrypoint, but the main audience uses `script/run_ueler.ipynb`.
- **No import-time side effects.** `import ueler` registers no `sys.meta_path` finders and claims no top-level module names, so it cannot change how any other import in the session resolves — a hard requirement for a PyPI-distributed package. Enforced by `tests/test_import_namespace_hygiene.py`.
- **Compatibility shims removed.** The `_AliasModuleFinder` / `_PrefixAliasFinder` layer in `ueler/_compat.py` that kept legacy `viewer.*`, `constants`, `data_loader` and `image_utils` imports working through the migration has been deleted; those four names were claimed at `sys.meta_path[0]` for every session that imported UELer. Import from `ueler.*`.
- **Editable install.** `pip install -e .` is the recommended install mode; it makes `git pull` upgrades instant.
- **Fast-stub test bootstrap.** `tests/bootstrap.py` stubs out heavy dependencies (`pandas`, `ipywidgets`, `matplotlib`) so the test suite runs quickly without a full environment.

---

## Package Layout

```
ueler/
├── __init__.py          # Public API surface
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
- The legacy import shims are removed; `import ueler` is side-effect free and asserted so by `tests/test_import_namespace_hygiene.py`.
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
