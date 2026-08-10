# Packaging and Project Structure

## Context
UELer has been refactored into a package-first layout with `ueler/` as the canonical namespace, while keeping notebook-first usage intact. The migration is complete: the compatibility shims that kept legacy `viewer.*` imports working during it have been removed (see below).

## Key decisions
- Keep notebooks as the primary entrypoint and add `ueler.runner.run_viewer` for programmatic use.
- Use a lightweight fast-stub test bootstrap to keep the test suite runnable without heavy dependencies.
- **`import ueler` must have no global side effects.** It registers no `sys.meta_path` finders and claims no top-level module names, so it cannot change how any other import in the session resolves. This is a hard requirement for distributing on PyPI and is enforced by `tests/test_import_namespace_hygiene.py`.
- **The test dependency stubs are opt-in, never on by default.** `sitecustomize.py` / `usercustomize.py` initialise `tests.bootstrap` only when `UELER_TEST_BOOTSTRAP=1` is set (the `make test-*` targets do). The stubs replace `pandas`, `matplotlib` and `ipywidgets`, so defaulting them on meant any interpreter with the repo root on `PYTHONPATH` could silently run against fakes. A requested-but-failed bootstrap now warns rather than passing silently.
- **BSD-3-Clause, relicensed from GPL-3.0-only before the first PyPI upload.** Copyleft on a package that others *import* propagates into their distributed work, which is the wrong shape for a library. The change was available because copyright sits with a single author and every runtime dependency is already permissive (BSD-3 / MIT / Apache-2.0) — nothing obliged the GPL. MPL-2.0 is the fallback if closed-fork protection ever becomes a goal; LGPL was rejected because its "linking" model maps badly onto Python imports.
- **Anything that ships must be visible to git.** `.gitignore` carries blanket `*.txt` / `*.png` rules, so packaged and doc assets are re-included explicitly (`!LICENSE.txt`, `!ueler/**/*.png`, `!doc/**/*.png`, `!docs/**/*.png`). setuptools builds from the working tree, so an ignored asset is present in a local build and absent from a clean-checkout build — add new asset types to those negations.

## Current status
- **Gate A of the PyPI release plan is complete** (`88b0e6e`): `MANIFEST.in` added, `setuptools>=77` floor, `requires-python = ">=3.10,<3.13"`, one settled project description, a PyPI-first README, `.gitignore` negations for packaged assets, opt-in test bootstrap, and `clean-dist`/`build`/`check-dist`/`publish-test`/`publish` Makefile targets.
- **Gate B is complete** (2026-08-10): ten PyPI classifiers and five `[project.urls]` entries, the license stated in the README, the docs-site install page realigned with the PyPI-first flow, and `ueler/graphify-out/` moved out of the package directory. `ipykernel`/`ipympl` stay hard dependencies and there is still no `console_scripts` entry point — both declined for `0.5.0` rather than overlooked. **Gates C and D remain** — see the plan linked under *Open items*.
- `ueler/` package skeleton, `pyproject.toml`, and `Makefile` are in place.
- Module moves from `viewer.*` to `ueler.viewer.*` are complete.
- A runner entrypoint exists for notebook usage.
- **The compatibility layer is gone.** `ueler/_compat.py` and `ueler.ensure_compat_aliases()` were deleted, along with the `ensure_aliases=` kwarg on `run_viewer` / `run_viewer_bia` (accepted-and-warned for one cycle via `runner._drop_removed_kwargs`). The shims installed finders at `sys.meta_path[0]` claiming the generic top-level names `viewer`, `constants`, `data_loader` and `image_utils`; nothing in the repo used them any more. `tests/test_shims_imports.py` was replaced by `tests/test_import_namespace_hygiene.py`.

## Open items
- Define and add a CI fast-stub job.
- Add an integration test workflow for heavier dependencies and GUI paths.
- Keep the packaging notes and release documentation aligned as changes land.
- Remaining pre-PyPI items are tracked as Gates C–D in [issue_tracking/issue79_pypi_release_readiness.md](issue_tracking/issue79_pypi_release_readiness.md): CI test + release workflows (Trusted Publishing), the git-tag backfill (`v0.4.2`–`v0.4.4` never got tags), and the TestPyPI rehearsal plus a live notebook smoke test.
- Decide whether **Python 3.12** joins the CI matrix or `requires-python` tightens to `<3.12` — the current `">=3.10,<3.13"` bound permits a minor version that has never been run. The per-minor classifiers list only 3.10 and 3.11, so the two must move together.
- **Confirm with DKFZ** that the BSD-3 copyright line naming both the author and the institute matches institutional policy — the one part of the relicense that is not a code change.

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/4

## Key source links
- [dev_note/Packaging_plan.md](dev_note/Packaging_plan.md)
- [dev_note/Todos.md](dev_note/Todos.md)
