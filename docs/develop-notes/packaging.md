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
- **`pip install ueler` for users, editable install for developers.** PyPI is the documented install path; `pip install -e .` from a clone is for working on UELer itself, where it makes `git pull` upgrades instant.
- **Fast-stub test bootstrap, opt-in.** `tests/bootstrap.py` stubs out heavy dependencies (`pandas`, `ipywidgets`, `matplotlib`) so the test suite runs quickly without a full environment. The `sitecustomize.py` / `usercustomize.py` startup hooks initialise it **only** when `UELER_TEST_BOOTSTRAP=1` is set — `make test-fast` and `make test-integration` set it for you. Defaulting it on meant any interpreter with the repo root on `PYTHONPATH` could silently run against fake scientific libraries; a bootstrap that was requested and then failed now emits a `RuntimeWarning` instead of being swallowed.
- **Packaged assets must be visible to git.** `.gitignore` has blanket `*.txt` / `*.png` rules, so assets are re-included explicitly (`!LICENSE.txt`, `!ueler/**/*.png`, `!doc/**/*.png`, `!docs/**/*.png`). setuptools builds from the working tree, so an ignored asset ships from a developer's machine and vanishes from a clean-checkout build — add new asset types to those negations.
- **`MANIFEST.in` keeps the sdist to build inputs only.** No tests are shipped: making them runnable would require shipping `bootstrap.py`'s dev-only stub machinery.
- **Supported Python: 3.10–3.11** (`requires-python = ">=3.10,<3.13"`). The bound is deliberately narrow — widen it only once CI has proven the newer minor.

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
- **Gate A of the PyPI release plan is complete** (2026-08-10). The build is reproducible and safe to publish: `python -m build` is clean, `twine check --strict` passes on both artifacts, and wheel and sdist have each been installed into a fresh venv and imported from outside the repository.

### Release targets

```shell
make build         # clean dist/ first, then build sdist + wheel
make check-dist    # twine check --strict
make publish-test  # upload to TestPyPI
make publish       # upload to PyPI — append-only, rehearse first
```

`publish*` depend on `check-dist`, not on `build`, so an upload sends exactly the artifacts that were built and inspected.

---

## Open Items

- Define and add a CI fast-stub job.
- Add an integration test workflow for heavier dependencies and GUI paths.
- Finish PyPI metadata (classifiers, `[project.urls]`), state the license in user-facing docs, and move `ueler/graphify-out/` out of the package directory.
- Add the release workflow (Trusted Publishing), then rehearse on TestPyPI before the first real upload.
- Decide whether Python 3.12 joins the CI matrix or `requires-python` tightens to `<3.12`.

---

## Related Issues

- [#79 — Package UELer as a pip package](https://github.com/HartmannLab/UELer/issues/79)
- [#4 — Packaging plan](https://github.com/HartmannLab/UELer/issues/4)
