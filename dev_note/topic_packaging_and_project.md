# Packaging and Project Structure

## Context
UELer has been refactored into a package-first layout with `ueler/` as the canonical namespace, while keeping notebook-first usage intact. The packaging plan also includes compatibility shims so legacy `viewer.*` imports continue to work during migration.

## Key decisions
- Keep notebooks as the primary entrypoint and add `ueler.runner.run_viewer` for programmatic use.
- Maintain compatibility shims so `viewer.*` and `ueler.*` imports both work during the transition.
- Use a lightweight fast-stub test bootstrap to keep the test suite runnable without heavy dependencies.

## Current status
- `ueler/` package skeleton, `pyproject.toml`, and `Makefile` are in place.
- Import shims are designed and implemented with test coverage (`tests/test_shims_imports.py`).
- Incremental module moves from `viewer.*` to `ueler.viewer.*` are completed.
- A runner entrypoint exists for notebook usage.

## Open items
- Define and add a CI fast-stub job.
- Add an integration test workflow for heavier dependencies and GUI paths.
- Keep the packaging notes and release documentation aligned as changes land.

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/4

## Key source links
- [dev_note/Packaging_plan.md](dev_note/Packaging_plan.md)
- [dev_note/Todos.md](dev_note/Todos.md)
