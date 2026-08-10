# Packaging and Project Structure

## Context
UELer has been refactored into a package-first layout with `ueler/` as the canonical namespace, while keeping notebook-first usage intact. The migration is complete: the compatibility shims that kept legacy `viewer.*` imports working during it have been removed (see below).

## Key decisions
- Keep notebooks as the primary entrypoint and add `ueler.runner.run_viewer` for programmatic use.
- Use a lightweight fast-stub test bootstrap to keep the test suite runnable without heavy dependencies.
- **`import ueler` must have no global side effects.** It registers no `sys.meta_path` finders and claims no top-level module names, so it cannot change how any other import in the session resolves. This is a hard requirement for distributing on PyPI and is enforced by `tests/test_import_namespace_hygiene.py`.

## Current status
- `ueler/` package skeleton, `pyproject.toml`, and `Makefile` are in place.
- Module moves from `viewer.*` to `ueler.viewer.*` are complete.
- A runner entrypoint exists for notebook usage.
- **The compatibility layer is gone.** `ueler/_compat.py` and `ueler.ensure_compat_aliases()` were deleted, along with the `ensure_aliases=` kwarg on `run_viewer` / `run_viewer_bia` (accepted-and-warned for one cycle via `runner._drop_removed_kwargs`). The shims installed finders at `sys.meta_path[0]` claiming the generic top-level names `viewer`, `constants`, `data_loader` and `image_utils`; nothing in the repo used them any more. `tests/test_shims_imports.py` was replaced by `tests/test_import_namespace_hygiene.py`.

## Open items
- Define and add a CI fast-stub job.
- Add an integration test workflow for heavier dependencies and GUI paths.
- Keep the packaging notes and release documentation aligned as changes land.
- Remaining pre-PyPI items (assessed, not yet done): raise the `build-system` setuptools floor to `>=77` for the PEP 639 `license` field, clear the stale `dist/` artifacts and add a release target, `prune tests` in a `MANIFEST.in` (the sdist currently ships `tests/test_*.py` without `bootstrap.py`, so they cannot run), invert the `sitecustomize.py` / `usercustomize.py` bootstrap to opt-*in*, and fill out the PyPI classifiers and URLs.

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/4

## Key source links
- [dev_note/Packaging_plan.md](dev_note/Packaging_plan.md)
- [dev_note/Todos.md](dev_note/Todos.md)
