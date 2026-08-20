# Packaging and Project Structure

## Context
UELer has been refactored into a package-first layout with `ueler/` as the canonical namespace, while keeping notebook-first usage intact. The migration is complete: the compatibility shims that kept legacy `viewer.*` imports working during it have been removed (see below).

## Key decisions
- Keep notebooks as the primary entrypoint and add `ueler.runner.run_viewer` for programmatic use.
- Use a lightweight fast-stub test bootstrap to keep the test suite runnable without heavy dependencies.
- **`import ueler` must have no global side effects.** It registers no `sys.meta_path` finders and claims no top-level module names, so it cannot change how any other import in the session resolves. This is a hard requirement for distributing on PyPI and is enforced by `tests/test_import_namespace_hygiene.py`.
- **The test dependency stubs are opt-in, never on by default.** `sitecustomize.py` / `usercustomize.py` initialise `tests.bootstrap` only when `UELER_TEST_BOOTSTRAP=1` is set (the `make test-*` targets do). The stubs replace `pandas`, `matplotlib` and `ipywidgets`, so defaulting them on meant any interpreter with the repo root on `PYTHONPATH` could silently run against fakes. A requested-but-failed bootstrap now warns rather than passing silently.
- **The PyPI metadata says what UELer *is*, not what it competes with or nearly does.** Two entries were dropped on 2026-08-20. `napari-alternative` left `keywords`: nobody searches PyPI for it, and a keyword field exists to match the words a user types, not to stake out a position against another project. `Topic :: Scientific/Engineering :: Image Processing` left `classifiers`: that trove topic is where people look for libraries that *transform* pixels — filtering, segmentation, registration, morphology — and UELer offers none of it; it loads, links and displays what other tools produced. `Visualization` + `Bio-Informatics` carry the description without over-claiming, and the reason now sits in a comment above the classifier block next to the `License ::` note so neither gets added back as a "fix".
- **BSD-3-Clause, relicensed from GPL-3.0-only before the first PyPI upload.** Copyleft on a package that others *import* propagates into their distributed work, which is the wrong shape for a library. The change was available because copyright sits with a single author and every runtime dependency is already permissive (BSD-3 / MIT / Apache-2.0) — nothing obliged the GPL. MPL-2.0 is the fallback if closed-fork protection ever becomes a goal; LGPL was rejected because its "linking" model maps badly onto Python imports.
- **A skipped test is a failure, not a pass.** CI runs the suite through
  `tools/run_test_suite.py --max-skips 0`, which prints every skip with its reason
  and exits non-zero. `unittest` prints `OK` for a run that quietly dropped 14
  bokeh-gated tests, and that is exactly how the Python 3.11 coverage gap stayed
  invisible until the release audit. A complete environment skips 0 of 913, so the
  gate is a measurement rather than an aspiration.
- **CI installs the real dependency stack; the fast stubs are not a CI substitute.**
  Measured, `tests/bootstrap.py` collects only 671 of 913 tests in a minimal
  environment and errors on 68 — `_ensure_matplotlib_stub()` replaces real
  matplotlib with a stub that has no `matplotlib.path` and no `colors.Normalize`
  whenever `matplotlib.pyplot` is not already imported. The stubs make an
  already-complete environment fast; they do not replace one.
- **Every dependency the code imports directly must be declared, even when it
  arrives transitively anyway.** `pandas` was imported in fourteen modules and
  named nowhere in `pyproject.toml`, arriving only through seaborn and anndata.
  That is what let pip resolve a different pandas *major* per Python minor —
  pandas 3 requires ≥ 3.11, so the CI 3.12 leg got the new default `StringDtype`
  and the 3.10/3.11 legs did not, which surfaced as 19 errors on one leg and
  silence on the others. Declared `pandas>=2.0`; the floor is conservative, not
  measured.
- **Column dtypes are classified through `pandas.api.types`, never
  `np.issubdtype`.** `np.issubdtype` raises `TypeError` on every pandas
  *extension* dtype — nullable `Int64`/`Float64`, `category`, and the
  `StringDtype` pandas 3 gives object string columns by default — and
  `flatten_anndata` produces all of them. `ueler.cell_table` owns
  `is_integer_column_dtype` / `is_float_column_dtype`, which unwrap categoricals
  to `.categories.dtype` (a `category` column of integers must still convert its
  labels to `int`, because `series == "1"` matches no rows where `series == 1`
  matches) and fall back to numpy only for the value-based `api.types` namespace
  in `tests/bootstrap.py`.
- **A tag push can reach TestPyPI but never PyPI.** `release.yml` publishes to the
  real index only on an explicit `workflow_dispatch` from a tag ref, behind a
  GitHub environment reviewer, using Trusted Publishing (OIDC — no API token in the
  repository). A PyPI version can be yanked but never reused, so `git push --tags`
  must not be an irreversible public act.
- **Anything that ships *or runs remotely* must be visible to git.** `.gitignore` carries blanket `*.txt` / `*.png` rules, so packaged and doc assets are re-included explicitly (`!LICENSE.txt`, `!ueler/**/*.png`, `!doc/**/*.png`, `!docs/**/*.png`). setuptools builds from the working tree, so an ignored asset is present in a local build and absent from a clean-checkout build — add new asset types to those negations. The same rule bit CI: `.github/` was ignored wholesale, which is why `workflows/docs.yml` was force-added and why the Gate C workflows were invisible to `git status`. **A negation cannot re-include anything under an ignored directory**, so a blanket directory rule has to be narrowed rather than patched — `.gitignore` now lists the local-only assistant scaffolding (`.github/agents/`, `.github/chatmodes/`, `.github/prompts/`, `.github/copilot-instructions.md`) and leaves `.github/workflows/` tracked.

## Current status
- **Gate A of the PyPI release plan is complete** (`88b0e6e`): `MANIFEST.in` added, `setuptools>=77` floor, `requires-python = ">=3.10,<3.13"`, one settled project description, a PyPI-first README, `.gitignore` negations for packaged assets, opt-in test bootstrap, and `clean-dist`/`build`/`check-dist`/`publish-test`/`publish` Makefile targets.
- **Gate B is complete** (2026-08-10): the PyPI classifier block (ten entries then, nine since the 2026-08-20 metadata trim) and five `[project.urls]` entries, the license stated in the README, the docs-site install page realigned with the PyPI-first flow, and `ueler/graphify-out/` moved out of the package directory. `ipykernel`/`ipympl` stay hard dependencies and there is still no `console_scripts` entry point — both declined for `0.5.0` rather than overlooked.
- **Gate C is complete** (2026-08-10): `.github/workflows/tests.yml` (matrix over 3.10/3.11 with a non-blocking 3.12 leg, full runtime stack, zero-skip gate, plus a `package` job that builds and imports the wheel from outside the repo) and `.github/workflows/release.yml` (Trusted Publishing; tag → TestPyPI, manual dispatch → PyPI). `v0.4.2`–`v0.4.4` are **not** being backfilled — those tag gaps were intentional. **Gate D remains** and needs a human: TestPyPI rehearsal, live notebook smoke test, then the upload. See the plan linked under *Open items*.
- **CI has now run, and it paid for itself on the first push** (2026-08-10): 3.10 ✅, 3.11 ✅, `package` ✅, and the non-blocking 3.12 leg surfaced 19 errors that turned out to be a live pandas bug rather than a 3.12 one — `np.issubdtype` on a pandas extension dtype, which broke mask painting for any AnnData-derived `category` identifier column on *current* pandas too. Fixed with dtype predicates in `ueler/cell_table.py`, plus `pandas` finally declared as a direct dependency; suite now **922 tests, 0 skips**.
- `ueler/` package skeleton, `pyproject.toml`, and `Makefile` are in place.
- Module moves from `viewer.*` to `ueler.viewer.*` are complete.
- A runner entrypoint exists for notebook usage.
- **The compatibility layer is gone.** `ueler/_compat.py` and `ueler.ensure_compat_aliases()` were deleted, along with the `ensure_aliases=` kwarg on `run_viewer` / `run_viewer_bia` (accepted-and-warned for one cycle via `runner._drop_removed_kwargs`). The shims installed finders at `sys.meta_path[0]` claiming the generic top-level names `viewer`, `constants`, `data_loader` and `image_utils`; nothing in the repo used them any more. `tests/test_shims_imports.py` was replaced by `tests/test_import_namespace_hygiene.py`.

## Open items
- ~~Define and add a CI fast-stub job.~~ **Closed as superseded** (2026-08-10): a stub-based job cannot run the suite — see the key decision above. CI installs the real stack instead, which is what a user gets from `pip install ueler` anyway.
- Add an integration test workflow for heavier dependencies and GUI paths. (`tests.yml`'s `unit` job now covers the full dependency stack; what is still missing is the *GUI* half, which needs a browser.)
- Keep the packaging notes and release documentation aligned as changes land.
- Remaining pre-PyPI items are tracked as Gate D in [issue_tracking/issue79_pypi_release_readiness.md](issue_tracking/issue79_pypi_release_readiness.md): configure the Trusted Publishers, rehearse on TestPyPI, run the notebook smoke test from an installed wheel, then upload `0.5.0-alpha`.
- **Act on the 3.12 result** — the leg has now run **once**, and its first reading was not about 3.12 at all: 19 errors, all of them one `np.issubdtype` call meeting the pandas-3 string dtype (fixed; see the key decisions above and the log entry). Tightening `requires-python` to `<3.12` would have hidden that rather than fixed it, since pandas 3 installs on 3.11 too. Re-read the leg after the fix lands; the evidence so far argues for *widening* — add `Programming Language :: Python :: 3.12` and drop `continue-on-error` — and the classifiers and the bound still move together. **Keep the leg either way:** it is the project's only coverage of pandas-3 semantics and it is what caught this.
- **The 3.10 floor is the part of the Python range under real pressure, not the 3.12 ceiling.** pandas 3 requires ≥ 3.11, anndata 0.12 requires ≥ 3.11 and anndata 0.13 requires ≥ 3.12, so a 3.10 install is already pinned to the older half of the stack. Nothing is broken today — the dangerous pandas-3-with-anndata-0.11 pairing is unreachable, because pandas 3 cannot be installed on 3.10 at all — but the floor will need revisiting before it starts costing coverage.
- **Confirm with DKFZ** that the BSD-3 copyright line naming both the author and the institute matches institutional policy — the one part of the relicense that is not a code change.

## Related GitHub issues
- https://github.com/HartmannLab/UELer/issues/4

## Key source links
- [dev_note/Packaging_plan.md](dev_note/Packaging_plan.md)
- [dev_note/Todos.md](dev_note/Todos.md)
