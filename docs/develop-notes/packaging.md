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
- **Packaged assets — and workflow files — must be visible to git.** `.gitignore` has blanket `*.txt` / `*.png` rules, so assets are re-included explicitly (`!LICENSE.txt`, `!ueler/**/*.png`, `!doc/**/*.png`, `!docs/**/*.png`). setuptools builds from the working tree, so an ignored asset ships from a developer's machine and vanishes from a clean-checkout build — add new asset types to those negations. CI hit the same rule from the other side: `.github/` was ignored as a whole directory, so a new workflow file simply never reached GitHub. **A negation cannot re-include anything under an ignored directory**, so the directory rule was narrowed to the local-only assistant scaffolding instead, leaving `.github/workflows/` tracked.
- **`MANIFEST.in` keeps the sdist to build inputs only.** No tests are shipped: making them runnable would require shipping `bootstrap.py`'s dev-only stub machinery.
- **Supported Python: 3.10–3.11** (`requires-python = ">=3.10,<3.13"`). The bound is deliberately narrow — widen it only once CI has proven the newer minor. The per-minor `Programming Language :: Python ::` classifiers list the same two versions, so both have to change together.
- **BSD-3-Clause, relicensed from GPL-3.0-only before the first PyPI upload.** UELer is a library other people import, and copyleft there propagates into the importer's distributed work — the opposite of what a lab tool wants. Every runtime dependency is already permissive (BSD-3 / MIT / Apache-2.0), so nothing obliged the GPL; and copyright sits with a single author, so the change needed no contributor round-up. BSD-3 matches `scikit-image`, `dask`, `bokeh`, `anndata` and `napari`.
- **No `License ::` classifier.** PEP 639 forbids combining one with the `license` SPDX expression that `pyproject.toml` declares; setuptools warns if both are present. The license reaches the metadata as `License-Expression: BSD-3-Clause`.
- **A skipped test is a failure, not a pass.** CI runs the suite through
  `tools/run_test_suite.py --max-skips 0`, which prints every skip with its reason
  before deciding the exit code. Plain `unittest` prints `OK` for a run that
  silently dropped 14 bokeh-gated tests — that is how the Python 3.11 coverage gap
  stayed invisible until the release audit. A complete environment skips 0 of 913
  tests, so zero is a measurement, not an aspiration. `make test-ci` runs the same
  gate locally.
- **CI installs the real dependency stack; the fast stubs are not a substitute for
  one.** A stub-only run collects 671 of 913 tests and errors on 68:
  `_ensure_matplotlib_stub()` replaces real matplotlib whenever `matplotlib.pyplot`
  is not already imported, and the stub has no `matplotlib.path` and no
  `colors.Normalize`. The stubs make an already-complete environment fast; the
  planned "CI fast-stub job" was dropped because it would report green over a third
  of the suite never running.
- **Every directly-imported dependency is declared, even when it would arrive
  transitively.** `pandas` was imported by fourteen modules and named nowhere in
  `pyproject.toml`, arriving only via seaborn and anndata — which is how pip came
  to resolve a different pandas *major* per Python minor (pandas 3 requires
  ≥ 3.11, so the CI 3.12 leg got the new default `StringDtype` and the 3.10/3.11
  legs did not). Now `pandas>=2.0`.
- **Column dtypes are classified through `pandas.api.types`, never
  `np.issubdtype`.** `np.issubdtype` raises `TypeError` on every pandas
  *extension* dtype — nullable `Int64`/`Float64`, `category`, and pandas 3's
  default `StringDtype` — and `flatten_anndata` produces all of them.
  `ueler.cell_table.is_integer_column_dtype` / `is_float_column_dtype` handle both
  families and unwrap categoricals to `.categories.dtype`, because a `category`
  column of integers still needs its labels converted (`series == "1"` matches no
  rows where `series == 1` matches).
- **A tag push can reach TestPyPI but never PyPI.** The real upload takes an
  explicit `workflow_dispatch` from a tag ref, behind a GitHub environment
  reviewer. A PyPI version can be yanked but never reused, so `git push --tags`
  must not be an irreversible public act.
- **Generated caches stay out of `ueler/`.** The graphify output belongs at the repo root; a build tool that globs package data is one `package-data` change away from shipping it, and the `.gitignore` negation trick above cannot rescue a directory rule.

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
- **Gate A of the PyPI release plan is complete.** The build is reproducible and safe to publish: `python -m build` is clean, `twine check --strict` passes on both artifacts, and wheel and sdist have each been installed into a fresh venv and imported from outside the repository.
- **Gate B is complete** (2026-08-10). The release now describes itself: ten PyPI classifiers, `[project.urls]` covering repository / issues / changelog, the license stated in the README, the docs-site install page realigned with the PyPI-first flow, and the stale graphify cache moved out of `ueler/`.
- **Gate C is complete** (2026-08-10). Two workflows: `tests.yml` (unit matrix + a build-and-import-the-wheel job) and `release.yml` (Trusted Publishing). See *Continuous integration* and *Release process* below.
- **CI paid for itself on its first run** (2026-08-10): 3.10, 3.11 and `package` green, and the non-blocking 3.12 leg surfaced 19 errors that were a live pandas bug rather than a 3.12 one — mask painting was broken for any AnnData-derived `category` identifier column on *current* pandas too. Suite now **922 tests, 0 skips**.
- **Remaining: Gate D** — configure the Trusted Publishers, rehearse on TestPyPI, run the notebook end to end from an installed wheel, then publish `0.5.0-alpha`.

### Continuous integration

`.github/workflows/tests.yml` runs on pushes to `main` / `develop` / `nightly` /
`pre-release`, on every pull request, on manual dispatch, and as a reusable
workflow called by `release.yml`. Two jobs:

- **`unit`** — matrix over Python **3.10** and **3.11**, plus a **3.12** leg marked
  `continue-on-error` because `requires-python` permits 3.12 and nothing has ever
  run there. It installs the full runtime stack and runs
  `tools/run_test_suite.py --max-skips 0`. `actions/setup-node` is there on purpose:
  two tests parse the anywidget ESM bundle with node and would otherwise skip.
- **`package`** — builds the sdist and wheel, runs `twine check --strict`,
  cross-checks every version declaration, then installs the wheel into a clean venv
  and imports it **from outside the repository**. That last step is the
  clean-checkout case a developer never sees locally, where a `.gitignore`-swallowed
  asset goes missing. Its artifacts are what `release.yml` publishes.

### Release process

Uploads use **PyPI Trusted Publishing** (OIDC): there is no API token in the
repository's secrets. One-time setup before the first release:

1. On **TestPyPI** → *Account settings* → *Publishing*, add a **pending publisher**:
   owner `HartmannLab`, repository `UELer`, workflow `release.yml`, environment
   `testpypi`.
2. The same on **PyPI**, with environment `pypi`. "Pending" exists for projects that
   are not on the index yet.
3. On GitHub → *Settings* → *Environments* → `pypi`, add a **required reviewer**.

After that there are exactly two paths:

| What you do | What happens |
|---|---|
| push a `v*` tag | tests → build → verify the tag against the artifacts → upload to **TestPyPI** |
| Actions → *Release* → *Run workflow*, ref = **the tag**, `publish_to: pypi` | the same gates, then upload to **PyPI** |

A tag push cannot reach PyPI. `release.yml` publishes what `tests.yml` built in the
same run, so the uploaded artifact is the tested one.

### Release targets

```shell
make test-ci                          # the suite with no skips tolerated
make build                            # clean dist/ first, then build sdist + wheel
make check-dist                       # twine check --strict
make check-release TAG=v0.5.0-alpha   # tag == pyproject == __version__ == dist/
make publish-test                     # upload to TestPyPI
make publish                          # upload to PyPI — append-only, rehearse first
```

`publish*` depend on `check-dist`, not on `build`, so an upload sends exactly the
artifacts that were built and inspected. `check-release` compares PEP 440-normalised
versions, so the repo's SemVer tag spelling (`v0.5.0-alpha`) matches the packaging
spelling (`0.5.0a0`) — `v0.5.0-a0` works too.

---

## Open Items

- ~~Define and add a CI fast-stub job~~ — **closed as superseded**; the skip threshold it asked for exists, but on the real dependency stack (see the key decisions above).
- Add an integration test workflow for the **GUI** paths. The full dependency stack is now covered by `tests.yml`; the widget layer still needs a browser, so it stays manual.
- Rehearse on TestPyPI before the first real upload, and configure the Trusted Publishers described under *Release process*.
- Act on the 3.12 result. The leg has run **once**, and its first reading was not about 3.12: 19 errors, all one `np.issubdtype` call meeting pandas 3's default string dtype — now fixed. Tightening `requires-python` to `<3.12` would have concealed it, since pandas 3 installs on 3.11 too. Re-read after the fix lands; the evidence argues for *widening* (add `Programming Language :: Python :: 3.12`, drop `continue-on-error`), and the classifiers and the bound move together. Keep the leg regardless — it is the only coverage of pandas-3 semantics.
- Revisit the **3.10 floor** before it costs coverage: pandas 3 and anndata 0.12 both require ≥ 3.11, and anndata 0.13 requires ≥ 3.12, so a 3.10 install is pinned to the older half of the stack. Nothing is broken today — pandas 3 cannot be installed on 3.10 at all, so the bad pandas-3-with-anndata-0.11 pairing is unreachable.
- Confirm with DKFZ that naming both the author and the institute in the BSD copyright line matches institutional policy — the only part of the relicense that is not purely a code change.
- Revisit `ipykernel` / `ipympl` as hard runtime dependencies before `1.0`: `pip install ueler` currently installs a Jupyter kernel. Moving them to a `notebook` extra also requires updating `.binder/postBuild`, which runs a bare `pip install .`.

---

## Related Issues

- [#79 — Package UELer as a pip package](https://github.com/HartmannLab/UELer/issues/79)
- [#4 — Packaging plan](https://github.com/HartmannLab/UELer/issues/4)
