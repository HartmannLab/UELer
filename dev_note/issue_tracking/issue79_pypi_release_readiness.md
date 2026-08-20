# Issue #79 — PyPI release readiness plan

> Status: written 2026-08-10 against `develop` @ `377c1bb` + working tree at `v0.5.0-alpha`.
> **Gate A is complete** (`88b0e6e`). **Gate B is complete** (`f9fcfae` — B2 and B3
> consciously declined, plus one item the original list missed). **Gate C is
> complete** (2026-08-10, working tree on `nightly` — C1 landed in a different shape
> than planned and C3 was decided as "don't"). **CI has now run once** (run
> `31407846635`): 3.10, 3.11 and `package` green; the non-blocking 3.12 leg found a
> real bug, fixed in **C5**. **Gate D remains**, and it is the
> half that needs a human: a TestPyPI rehearsal, a live notebook run, then the
> upload. Developer decisions are recorded under
> [Decisions taken](#decisions-taken).
> Related: [#79 — Package UELer as a pip package](https://github.com/HartmannLab/UELer/issues/79),
> [#4 — Packaging plan](https://github.com/HartmannLab/UELer/issues/4),
> [dev_note/topic_packaging_and_project.md](../topic_packaging_and_project.md),
> [docs/develop-notes/packaging.md](../../docs/develop-notes/packaging.md).

---

## Context

The packaging skeleton is finished and works. This document covers only what is
left between here and `pip install ueler` working for a stranger.

### Already verified (do not re-litigate)

Measured on 2026-08-10, not inferred from config:

| Check | Result |
|---|---|
| `python -m build` | ✅ sdist + wheel build clean |
| `twine check` | ✅ PASSED on both artifacts |
| Wheel install in a clean venv | ✅ `import ueler`, `ueler.viewer.main_viewer`, bundled `images/` assets all resolve |
| **sdist** install in a clean venv | ✅ builds and imports (`pip install ueler-0.5.0a0.tar.gz`) |
| Dependency resolution from PyPI | ✅ all 22 core deps; the `[ark]` extra (`ark-analysis==0.7.0`) also resolves |
| PyPI name availability | ⚠️ **partly wrong — re-measured 2026-08-10.** `ueler` is indeed unregistered on **PyPI** (404 on both `/pypi/ueler/json` and `/simple/ueler/`), but *unregistered* is not the same as *allowed*: PyPI's pending-publisher form rejects it with **"This project name isn't allowed"**, which is its message for an administratively **prohibited** name, not for a taken one. See **C4**. And the row only ever checked PyPI — **TestPyPI already holds `ueler` 0.3.1**, uploaded 2026-04-10 (see **D1**). |
| Local-path / machine-info leakage | ✅ `make scan-package` → 0 high, 0 medium, 2 low (docstrings naming `~/.ueler`) |
| Wheel hygiene | ✅ no `__pycache__`, no `ueler/graphify-out/`, no tests, no `_compat.py`; license lands in `dist-info/licenses/` |
| Test suite, Python 3.10 | ✅ 913 tests, OK (5 skips) |
| Test suite, Python 3.11 | ⚠️ 913 tests, OK but **19 skips** — the extra 14 are all `bokeh not available in this environment`, i.e. an artifact of the 3.11 env used, **not** a 3.11 incompatibility. The bokeh/histogram paths are therefore *unexercised* on 3.11. See **C1**. |
| Import-time side effects | ✅ removed this cycle — `import ueler` adds nothing to `sys.meta_path` and claims no top-level names (`tests/test_import_namespace_hygiene.py`) |

The one genuinely dangerous item found in the assessment — `import ueler`
installing meta-path finders that claimed the generic top-level names `viewer`,
`constants`, `data_loader` and `image_utils` for every session — is **fixed**
(see the `v0.5.0-alpha` entry in [doc/log.md](../../doc/log.md)).

### Irreversibility, i.e. why the gates below exist

PyPI is **append-only**. A version number can never be reused, even after
`yank` or delete, and the project name cannot be transferred casually. A wrong
first upload is permanent public record. Everything in Gate A exists to make the
first upload one we would not want back.

---

## Gate A — blocking: must be correct *before any upload* — ✅ **DONE**

All eight items are implemented. Verification recorded per item below; the whole
gate was re-verified together at the end (913 tests OK, clean `python -m build`,
`twine check --strict` PASSED on both artifacts, wheel + sdist installed and
imported in two fresh venvs).

### Decisions taken

- **A1** → **"Unified Exploratory Linked Viewer"**, scoped to *spatial proteomics
  and multiplexed imaging* (not MIBI-specific). This matched the newest edit
  (`mkdocs.yml`) and the tracked `README.md`; `pyproject.toml` and
  `docs/index.md` were the stragglers still saying "Usability Enhanced … MIBI".
- **A3** → **`requires-python = ">=3.10,<3.13"`**. Note the residual gap: this
  permits **3.12**, which has never been run. Closing it is a **C1** job — add
  3.12 to the matrix, or tighten to `<3.12`.
- **B4** → **relicensed GPL-3.0-only → BSD-3-Clause** (2026-08-10, before any
  upload). UELer is imported by other people's code, and copyleft propagates into
  their distributed work — the wrong shape for a library. Two checks established
  the change was actually available: copyright sits with a **single** author
  (`git shortlog -sne --all`: one human, two bot identities committing on the repo
  owner's behalf), so no contributor round-up was required; and **every** runtime
  dependency is permissive (BSD-3 / MIT / Apache-2.0), so the GPL was a standalone
  choice rather than inherited. BSD-3 matches the neighbourhood — scikit-image,
  dask, bokeh, anndata, napari. **MPL-2.0** was the runner-up and is the right
  answer if closed-fork protection ever becomes a goal; **LGPL** was rejected
  because its "linking" model maps badly onto Python `import` and the ambiguity
  deters adopters as effectively as the GPL. Applies going forward; tags through
  `v0.4.1` stay GPL as published, and nothing was ever uploaded to PyPI, so there
  is no released artifact to reconcile.
- **C1** → **CI runs the real dependency stack, not the fast stubs**, and the
  workflow fails on the *first skipped test*. The long-standing "add a CI fast-stub
  job" item is closed as **superseded**: measured, the stub bootstrap collects only
  671 of 913 tests in a minimal environment and errors on 68, so a stub-based job
  would report green over a third of the suite never running. Zero skips is a
  measurement — 0 of 913 in a complete environment — so it can be a hard gate.
- **C1 (3.12)** → **added as a non-blocking matrix leg** rather than settled by
  argument. `requires-python` permits 3.12 and nothing had ever run there; a
  `continue-on-error` leg produces that evidence on the next push at no risk.
  Widening the classifiers then becomes a one-line follow-up backed by data — and
  `classifiers` and `requires-python` still have to move together.
  **Vindicated on the first run, but not in the expected way:** the leg's 19 errors
  were a pandas-3 forward-compatibility bug that was *already live on current
  pandas* for AnnData-derived `category` columns, not a 3.12 incompatibility. Had
  the decision gone the other way — argue it out and tighten to `<3.12` — the bug
  would have shipped. See **C5**.
- **C5 (dtype classification)** → **`pandas.api.types`, never `np.issubdtype`, for
  anything holding a pandas column dtype.** `np.issubdtype` raises `TypeError` on
  every pandas extension dtype. The predicates live in `ueler/cell_table.py`
  because the dtype in question is always a cell-table column's, and categoricals
  are unwrapped to their categories' dtype rather than reported as non-numeric —
  otherwise a loud crash becomes a silently empty selection.
- **C5 (dependencies)** → **declare what the code imports, even when it arrives
  transitively.** `pandas` was imported in fourteen modules and named nowhere, so
  pip was free to resolve a different major per Python minor with nothing in the
  repository recording the expectation. Now `pandas>=2.0`.
- **C2** → **a tag push publishes to TestPyPI only; PyPI needs a manual dispatch.**
  The plan had tags publishing straight to PyPI. Given that a PyPI version can be
  yanked but never reused, making `git push --tags` an irreversible public act is
  the wrong default for a solo-maintained project. The real upload takes an explicit
  `workflow_dispatch` from a tag ref, behind a GitHub environment reviewer.
- **C3** → **no tag backfill.** Developer's call: the `v0.4.2`–`v0.4.4` gaps were
  intentional, not drift. The first tag created through the release workflow is
  `v0.5.0-alpha`, after Gate D.
- **D3** → **the first upload is `0.5.0-alpha`**, not a final `0.5.0`. Stated by the
  developer ("let's start with v0.5.0-a0 after we have done with all the gates").
  pip will not install it without `--pre`, which the README and the docs site both
  already document, so it validates the whole pipeline without becoming the version
  a casual `pip install ueler` picks up.

### A1. Reconcile the project description and keywords — ✅ done

`pyproject.toml` `description` is **"Usability Enhanced Linked Viewer for MIBI
imaging"**. The README title now reads **"Unified Exploratory Linked Viewer"**
and `mkdocs.yml` `site_description` (uncommitted at time of writing) reads
**"Unified Exploratory Linked Viewer for Spatial Proteomics"**. `keywords` still
says `MIBI`.

Why blocking: `description` becomes the one-line summary on the PyPI project
page and in `pip search`-style listings for that release, and the README becomes
the long description verbatim. Shipping three different names for the tool on
day one is the kind of thing that is awkward to walk back.

- **Action:** pick one expansion of the acronym, then make `pyproject.toml`
  (`description`, `keywords`), `README.md` (title line), `mkdocs.yml`
  (`site_name`, `site_description`) and the GitHub repo description agree.
  Broaden `keywords` beyond MIBI if the tool is meant for spatial proteomics
  generally (`imaging-mass-cytometry`, `spatial-proteomics`, `multiplexed-imaging`).
- **Done:** `pyproject.toml` `description` → "Unified Exploratory Linked Viewer
  for spatial proteomics and multiplexed imaging"; `keywords` broadened to
  `spatial proteomics, multiplexed imaging, image viewer, jupyter,
  napari-alternative, MIBI, IMC, bioimage analysis` (MIBI kept as a real search
  term rather than the framing). `docs/index.md` retitled to match. `README.md`
  and `mkdocs.yml` already agreed.
- **Still open (not a packaging blocker):** the **GitHub repo description** is set
  in the GitHub UI, not in the tree — the developer must update it there.
- **Verified:** wheel `METADATA` reads
  `Summary: Unified Exploratory Linked Viewer for spatial proteomics and multiplexed imaging`.
- **Revised (2026-08-20):** `napari-alternative` dropped from `keywords`. It is not a term anyone searches PyPI for, and it defines UELer by what it is not — a positioning claim, in a field whose only job is matching the words a user actually types. The remaining seven keywords are all things the tool *is* or *reads*. Keyword list is now `spatial proteomics, multiplexed imaging, image viewer, jupyter, MIBI, IMC, bioimage analysis`.

### A2. Give the README a `pip install ueler` path — ✅ done

[README.md](../../README.md) §Installation documents only: create a conda env →
`git clone` → `pip install -e .`. The README *is* the PyPI long description, so
the landing page for a pip package would tell readers to clone from git.

- **Action:** add a short "Install from PyPI" block as the *first* installation
  option (`pip install ueler`, plus `pip install "ueler[ark]"` for the
  ark-analysis extra), and keep the existing clone + editable flow retitled as
  "Install for development". Note the conda env is still recommended for the
  binary stack (`hdf5`) on HPC.
- Also fix: the "Upgrade UELer" section says `git pull`, which is wrong for a
  pip install (`pip install --upgrade ueler`).
- **Done:** §Installation is now "Option A — install from PyPI (recommended)"
  (including the `--pre` form, the `[ark]`/`[docs]` extras, and the supported
  Python versions) followed by "Option B — install from source (for development)".
  "Upgrade UELer" now covers both paths. "Getting started" step 2 no longer
  assumes a clone — it links the starter notebook for pip users.
- **Also fixed here (was D1):** every repo-relative link in the README, which
  renders on GitHub but breaks on PyPI. `doc/GUI_preview.png` and both
  `doc/log.md` links are now absolute `https://…githubusercontent/…` /
  `https://github.com/…` URLs.
- **Note:** `pip install ueler` will 404 until the first upload actually happens
  (**D3**). The instruction is correct as of the release, and this branch is
  `develop`, so it is not yet on the public default branch.
- **Verified:** `twine check --strict` PASSED (long-description markup valid) and
  the section was re-read end to end.

### A3. Decide and enforce the supported Python range — ✅ done

`requires-python = ">=3.10"` is an open-ended promise. Only **3.10** and
**3.11** exist locally, and 3.11 has not been exercised against bokeh (see the
table above). pip on Python 3.13 will happily install and then fail in ways we
cannot reproduce.

- **Options:** (a) narrow to `">=3.10,<3.13"` now and widen once CI proves
  otherwise — honest and cheap; (b) keep `>=3.10` and add a CI matrix (**C1**)
  covering 3.10–3.13 *first*. Recommend **(a) now, (b) as the follow-up** so the
  first release cannot over-promise.
- **Action:** set the bound, and add matching `Programming Language :: Python ::
  3.10 / 3.11` classifiers (see **B1**).
- **Done:** `requires-python = ">=3.10,<3.13"`. The matching per-minor
  classifiers are deliberately left to **B1** with the rest of the classifier
  block, so the two stay consistent.
- **Verified:** wheel `METADATA` reads `Requires-Python: <3.13,>=3.10`. Not
  verified against a real 3.12+ interpreter — none is available in this
  environment; pip's `requires-python` enforcement is a well-established path, so
  this is a low-risk gap rather than an untested behaviour.
- **Residual gap:** 3.12 is permitted but untested. See *Decisions taken*.

### A4. Purge stale `dist/` and add a release target — ✅ done

`dist/` currently holds **`ueler-0.3.1`** wheel + tarball from 2026-04-10.

**Corrected after review:** those were the developer's own test artifacts, so the
"would publish 0.3.1 as the first release" framing overstated it — this is
hygiene, not an active hazard. The `clean-dist`-first targets are still worth
having, because `twine upload dist/*` globs whatever happens to be in the
directory.

- **Action:** `rm -rf dist/ build/ ueler.egg-info/`, then add Makefile targets
  that *clean first*:
  ```make
  build: clean-dist
  	$(PYTHON) -m build
  publish-test: build
  	$(PYTHON) -m twine upload --repository testpypi dist/*
  publish: build
  	$(PYTHON) -m twine upload dist/*
  clean-dist:
  	rm -rf dist build ueler.egg-info
  ```
  Add `build`/`twine` to the `dev` extra so `make build` works from a fresh venv.
- **Done:** stale `dist/` purged; `Makefile` gained `clean-dist`, `build`
  (depends on `clean-dist`), `check-dist` (`twine check --strict`),
  `publish-test` and `publish` — the last two depend on `check-dist`, so an
  artifact that fails validation cannot be uploaded. `build>=1.0` and
  `twine>=5.0` added to the `dev` extra. `publish` carries a comment about PyPI
  being append-only.
- **Note:** `publish*` deliberately depend on `check-dist`, **not** on `build`, so
  that `make publish` uploads exactly the artifacts that were built and inspected
  rather than silently rebuilding them.
- **Verified:** after the purge, `python -m build` produced exactly
  `ueler-0.5.0a0-py3-none-any.whl` + `ueler-0.5.0a0.tar.gz` and nothing else.

### A5. Fix `.gitignore` so packaged assets cannot be silently dropped — ✅ done

`.gitignore` contains blanket `*.txt` (line 26) and `*.png` (line 44).
`LICENSE.txt` and `ueler/viewer/images/ready.png` are only in the repo because
someone used `git add -f`. Verified: a **new** `ueler/viewer/images/*.png` asset,
or a new root `*.txt`, is ignored today.

Why blocking: `setuptools` builds from the *filesystem*, so a locally-working
build would ship an asset that a CI build from a clean checkout silently omits —
producing a wheel that raises `FileNotFoundError` in `load_asset_bytes()` for
users only. This is exactly the failure mode that is invisible to the developer.

- **Action:** add negations immediately after the offending rules:
  ```gitignore
  !LICENSE.txt
  !ueler/**/*.png
  !doc/**/*.png
  ```
  (or narrow `*.txt`/`*.png` to the specific directories that need ignoring).
- **Done:** negations appended *below* the blanket rules (last match wins, so
  order is load-bearing): `!LICENSE.txt`, `!ueler/**/*.png`, `!doc/**/*.png`,
  `!docs/**/*.png` — `docs/` added because the mkdocs site images
  (`docs/img/UELer_icon_gw.png`) were exposed to the same trap.
- **Technique note:** negation works per file pattern but **cannot** re-include
  anything under an ignored *directory*. `graphify-out/` is a directory rule, so
  this does nothing for **B5** — that one still needs the relocation.
- **Verified** on the real repo: probe files `ueler/viewer/images/_probe_new_icon.png`
  and `doc/_probe.png` now show as `??` in `git status`, while
  `dev_note/_probe_scratch.png` and a root `_probe_scratch.txt` stay ignored
  (`git check-ignore -v` still attributes them to `.gitignore:44` / `:26`), and
  `LICENSE.txt` is no longer reported as ignored. Separately confirmed the
  clean-checkout case directly: `git archive HEAD ueler/viewer/images` contains
  `__init__.py`, `loading.gif` and `ready.png`, so a CI build has all three.

### A6. Raise the `setuptools` floor to match the metadata — ✅ done

`build-system.requires = ["setuptools>=68", "wheel"]`, but `license =
"GPL-3.0-only"` (a PEP 639 SPDX expression) and `license-files` require
**setuptools >= 77**. It builds here only because the isolated build env happens
to fetch 80.9; a pinned or `--no-build-isolation` build (conda-forge, offline
CI, some corporate mirrors) would fail or silently drop the license metadata.

- **Action:** `requires = ["setuptools>=77"]`. Drop `wheel` — it has not been
  needed since setuptools 70.1.
- **Done:** `requires = ["setuptools>=77"]`, `wheel` dropped.
- **Verified:** wheel `METADATA` contains `Metadata-Version: 2.4`,
  `License-Expression: GPL-3.0-only` and `License-File: LICENSE.txt`.

### A7. `prune tests` from the sdist — ✅ done

Default setuptools sdist rules pull in `tests/test_*.py` (60+ files) but **not**
`tests/__init__.py` or `tests/bootstrap.py`, so the tests shipped in the sdist
cannot even be imported, let alone run.

- **Action:** add a `MANIFEST.in`:
  ```
  prune tests
  prune tests_tmp
  prune dev_note
  prune script
  prune tools
  prune graphify-out
  exclude sitecustomize.py usercustomize.py Makefile mkdocs.yml
  ```
  Shipping *no* tests is the honest choice here; shipping runnable ones means
  also shipping `bootstrap.py` and its stub machinery, which is dev-only.
- **Done:** [MANIFEST.in](../../MANIFEST.in) added — explicit `include` for the
  four build inputs, `recursive-include ueler/viewer/images *`, and `prune` for
  every dev directory (`tests`, `tests_tmp`, `dev_note`, `doc`, `docs`, `script`,
  `tools`, `site`, `env`, `.binder`, `.github`, `.claude`, `.specify`, `specs`,
  `graphify-out`, `ueler/graphify-out`) plus `exclude` for `Makefile`,
  `mkdocs.yml`, `CLAUDE.md`, `sitecustomize.py`, `usercustomize.py`.
- **Consequence worth knowing:** the sdist now carries no tests at all, so a
  downstream packager (conda-forge, Debian) cannot run the suite from it. That is
  the accepted trade — the suite needs `tests/bootstrap.py`'s stub machinery,
  which is dev-only. Revisit if the conda-forge feedstock (Deferred) happens.
- **Verified:** the sdist's 77 files are `ueler/`, `ueler.egg-info/`, `PKG-INFO`,
  `LICENSE.txt`, `README.md`, `pyproject.toml`, `setup.cfg`, `MANIFEST.in` —
  nothing else. The wheel's 64 files contain no strays outside `ueler/` and
  `*.dist-info/`, no `tests`, no `graphify`.

### A8. Invert the `sitecustomize.py` / `usercustomize.py` bootstrap — ✅ done

Both files at the repo root auto-import `tests.bootstrap`, which **replaces
`pandas`, `ipywidgets` and `matplotlib` with stubs**, and swallow every failure
in a bare `except Exception`. They currently opt *out* via
`UELER_SKIP_TEST_BOOTSTRAP=1`.

Verified: they do **not** ship in the wheel, and a modern editable install uses
a finder rather than putting the repo root on `sys.path`, so the common paths are
safe today. But `PYTHONPATH=<repo>` — or any interpreter that gets the repo root
onto `sys.path` before `site` runs — silently gets stubbed scientific libraries,
with no error and no log line. That is a debugging trap for exactly the people
who clone the repo per **A2**'s development flow.

- **Action:** invert to opt-in (`UELER_TEST_BOOTSTRAP=1`), so the files no-op
  unless the test runner asks. Point `Makefile`'s `test-fast` / `test-integration`
  at the new variable. Keep the bare `except` but add a `warnings.warn` so a
  failed bootstrap is at least visible when it *was* requested.
- **Done:** both files now no-op unless `UELER_TEST_BOOTSTRAP=1`.
  `UELER_SKIP_TEST_BOOTSTRAP=1` is still honoured as a hard override so any
  environment already setting it keeps working. The bare `except Exception` now
  emits a `RuntimeWarning` naming the failure, so a requested-but-failed
  bootstrap is visible instead of silent. `Makefile`'s `test-fast` /
  `test-integration` set the new variable.
- **Verified**, and the measurement changed the picture usefully:
  - `PYTHONPATH=. python -c ...` — `sitecustomize` imports, `tests.bootstrap`
    does **not** load without the opt-in, and **does** with it. The gate works
    in both directions.
  - Without `PYTHONPATH`, the repo's `sitecustomize.py` is **never imported at
    all** for `python -c` / `python -m`: the cwd entry lands on `sys.path` after
    `site` has run. So the exposure was narrower than the plan assumed —
    `PYTHONPATH=<repo>` was effectively the only trigger.
  - The suite is **not** load-bearing on this hook either: `tests/__init__.py`
    imports `tests.bootstrap`, which calls `initialize()` at module scope. 913
    tests pass identically with the variable set, unset, and under
    `PYTHONPATH=.` — same count as the pre-change baseline.
  - Also worth recording: in a *complete* environment the stubs are near-inert,
    because every `_ensure_*` helper installs a stub only when the real module is
    missing. `matplotlib` and `ipywidgets` resolve to the real site-packages
    copies either way here. The trap was real but only for incomplete envs.

---

## Gate B — before the release is public / announced — ✅ **DONE**

Implemented 2026-08-10 in the working tree, on top of Gate A (`88b0e6e`).
**B2 and B3 were declined rather than implemented** — see their entries; both were
already carrying a "leave as-is" recommendation, and declining is the action.
**B6 was added during the work** (the docs-site install page still contradicted
the README). Verified together: clean `python -m build`, `twine check --strict`
PASSED on both artifacts, `mkdocs build` exit 0, 913 tests OK.

### B1. Fill out PyPI metadata — ✅ done

Only two classifiers today (`Programming Language :: Python :: 3`,
`Operating System :: OS Independent`). This is the PyPI landing page's sidebar.

- **Action:** add `Development Status :: 4 - Beta` (matching a `0.5.0-alpha`
  line), `Intended Audience :: Science/Research`, `Topic :: Scientific/Engineering
  :: Bio-Informatics`, `Topic :: Scientific/Engineering :: Visualization`,
  `Framework :: Jupyter`, and the per-minor Python classifiers agreed in **A3**.
  Do **not** add a `License ::` classifier — PEP 639 forbids combining it with
  a `license` expression, and setuptools will warn.
- Add to `[project.urls]`: `Repository`, `Issues`
  (`https://github.com/HartmannLab/UELer/issues`), `Changelog`
  (pointing at `doc/log.md` or the docs site).
- **Verify:** `twine check`, then read the rendered page on **TestPyPI** (**D1**).
- **Done:** ten classifiers — `Development Status :: 4 - Beta`, `Intended Audience
  :: Science/Research`, `Framework :: Jupyter`, `Operating System :: OS
  Independent`, `Programming Language :: Python :: 3` / `3.10` / `3.11`, and
  `Topic :: Scientific/Engineering ::` `Bio-Informatics` / `Visualization` /
  `Image Processing` (the last added beyond the plan — UELer is an image tool
  first). A comment above the block records *why* there is no `License ::`
  classifier, so it does not get "fixed" back in later. `[project.urls]` now has
  `Homepage`, `Documentation`, `Repository`, `Issues`, `Changelog`; the two
  pre-existing keys were capitalised, since PyPI uses the key verbatim as the
  sidebar label.
- **Revised (2026-08-20):** `Topic :: Scientific/Engineering :: Image Processing` removed, so the release now carries **nine** classifiers. The trove `Image Processing` topic is where users look for libraries that *transform* pixels — filtering, segmentation, registration, morphology — and UELer offers none of that: it loads, links and displays images and cell tables that other tools produced. `Visualization` and `Bio-Informatics` describe it without over-claiming, and the comment above the block now records this alongside the `License ::` note so it does not get added back.
- **On `Development Status :: 4 - Beta` with a `0.5.0-alpha` version:** kept as
  planned. The `-alpha` token is a pre-release marker *within* the 0.5.0 line, not
  a claim about project maturity — 913 tests and a working Binder deployment are
  not "3 - Alpha". Recorded so the apparent mismatch reads as a choice.
- **Verified:** `twine check --strict` PASSED, which is also what validates
  classifiers against the trove list — an invalid one fails there rather than at
  upload. Wheel `METADATA` shows all ten classifiers and all five `Project-URL`
  lines. The rendered sidebar still needs eyes on it at **D1**.

### B2. Decide on `ipykernel` / `ipympl` as hard runtime dependencies — ⏸️ declined for `0.5.0`

Both are hard `[project.dependencies]`. A library that force-installs a Jupyter
kernel into any environment that merely depends on it is unusual, and it makes
`pip install ueler` heavier than it needs to be for someone importing
`ueler.image_utils` in a script.

- **Action (optional):** move `ipykernel` + `ipympl` to a `notebook` extra and
  make it the documented install (`pip install "ueler[notebook]"`).
- **Careful — this has a second consumer:** [.binder/postBuild](../../.binder/postBuild)
  runs a bare `pip install .` and relies on those two coming in via core deps.
  It must become `pip install ".[notebook]"` in the same commit, or Binder
  breaks. Deliberately calling this out because the coupling is easy to miss.
- **Decision needed:** worth the churn, or leave as-is for the first release?
  Recommend **leaving as-is** for `0.5.0` and revisiting once there is a
  non-notebook user.
- **Outcome: left as-is, no change made.** The recommendation and the no-op are
  the same action, so this is closed for `0.5.0` rather than pending. Reopen it
  when a non-notebook consumer appears — and remember the `.binder/postBuild`
  coupling has to change in the *same* commit.
- **Note for whoever reopens it:** `pip install ueler` currently installs a Jupyter
  kernel into the environment. That is the actual cost, and it is worth revisiting
  before `1.0`, not because it breaks anything but because it is surprising.

### B3. Consider a `console_scripts` entry point — ⏸️ declined for `0.5.0`

There is none, deliberately — UELer is notebook-first and
`ueler.runner.run_viewer` is the programmatic entry. A `ueler` command that
launches `script/run_ueler.ipynb` in Jupyter would be cheap discoverability, but
it is genuinely optional and adds a support surface.

- **Recommend:** skip for `0.5.0`. Noted so the omission is a decision rather
  than an oversight.
- **Outcome: skipped, no entry point added.** Nothing to verify.

### B4. State the license in the user-facing docs — ✅ done

`LICENSE.txt` is GPL-3.0 and `pyproject.toml` declares `GPL-3.0-only`, but
neither `README.md` nor the docs site mentions the license. Users evaluating
whether they can build on UELer should not have to open the repo tree.

- **Action:** one "License" section in the README and a line in the docs index.
  Worth a conscious confirmation that **GPL-3.0-only** is the intended choice
  for a library others will import, since it is copyleft — this is the moment it
  becomes hard to change (every later contributor's work is licensed under it).
- **Correction:** the docs half was **already done** — [docs/index.md](../../docs/index.md)
  has carried a "License" section linking `LICENSE.txt` since before this plan.
  Only the README was missing it. The plan overstated the gap.
- **Done:** `README.md` gained a "License" section naming GPL-3.0-only, with one
  plain-language paragraph distinguishing what copyleft binds (a distributed work
  that incorporates UELer) from what it does not (your data and your results when
  you merely *use* UELer). That distinction is written out deliberately: it is the
  question an evaluator in a commercial or clinical setting actually asks, and
  "released under GPL-3.0" alone does not answer it. An "Issues and contact"
  section was added alongside, since the README is the PyPI landing page and had
  no route to the issue tracker.
- **Resolved 2026-08-10 — relicensed to BSD-3-Clause.** The item above documented
  the *existing* declaration; the choice itself was the open question, and the
  answer was to change it. See [Decisions taken](#decisions-taken) for the
  reasoning and the two facts that made it possible. The README and `docs/index.md`
  sections written for this item were rewritten accordingly.
- **Residual, non-code:** confirm with DKFZ that the copyright line naming both the
  author and the institute matches institutional policy.

### B5. Move `ueler/graphify-out/` out of the package directory — ✅ done

The graphify cache/output lives *inside* `ueler/`. It is gitignored and verified
absent from the wheel (no `__init__.py`, so `packages.find` skips it), so this
is hygiene, not a bug — but a build tool that globs package data is one
`package-data` change away from shipping ~2 MB of graph JSON.

- **Action:** relocate to the repo-root `graphify-out/` that already exists.
- **What it actually was:** a **stale, separate** graph, not a stray copy of the
  root one — `ueler/graphify-out/.graphify_root` pointed at `…/UELer_public/ueler`,
  i.e. someone had once run graphify scoped to the package directory. Last written
  2026-07-31 (11 MB); the repo-root graph is from 2026-08-10 (48 MB) and covers a
  superset. So the two could not be *merged*, only one kept.
- **Done:** moved to `graphify-out/legacy-ueler-scoped-2026-07-31/`. Nesting it
  inside the existing `graphify-out/` keeps it covered by the same `.gitignore`
  directory rule with no new pattern, and the date-stamped name says what it is.
  It is regenerable and superseded — **safe to delete** whenever.
- **Verified:** `ueler/graphify-out` no longer exists; `git check-ignore -v`
  attributes the new path to the same `.gitignore:43` rule; `graphify query` still
  resolves against the root graph from the new layout; and no path containing
  `graphify` appears in either the rebuilt wheel or sdist.

### B6. Align the docs-site installation page with the README — ✅ done

**Not in the original Gate B list** — found while checking B4. [docs/installation.md](../../docs/installation.md)
was still the pre-PyPI page: env → `git clone` → `pip install -e .`, with
"Requirements: Python ≥ 3.10". A reader arriving from the PyPI project page (whose
`Documentation` URL now points straight at the docs site, per **B1**) would have
been told to clone the repository anyway, and given a Python bound that
`requires-python` no longer permits.

- **Done:** an "Option A — Install from PyPI" section now leads the page (`pip
  install ueler`, the `--pre` form, the `[ark]`/`[docs]` extras, `pip install
  --upgrade`), followed by "Option B — Install from Source" introducing the
  existing steps. "Requirements" reads 3.10 or 3.11. "Updating UELer" covers both
  paths. The development section lists `build`/`twine` in the `dev` extra and
  documents the opt-in `UELER_TEST_BOOTSTRAP=1` bootstrap from Gate A's **A8**.
- **Also stated explicitly:** the starter notebook is **not** part of the package,
  with a link to it in the repository. This is the one thing a pip user cannot
  discover on their own, and the same note was added to the README in **A2**.
- **Verified:** `mkdocs build` exits 0 (only the Material insiders banner on
  stderr).

---

## Gate C — infrastructure — ✅ **DONE**

Implemented 2026-08-10 in the working tree, on top of Gate A + B (`f9fcfae`).
All three items are done, but **C1 changed shape once measured** — the
"fast-stub CI job" that had been an open item since the packaging migration turns
out not to be buildable, and the reason is worth reading before someone tries
again. **C3 was decided by the developer: no backfill.**

### How to actually use the two workflows

Recorded here because the useful half of Gate C is operational knowledge, not code.

**`tests.yml`** needs no action. It runs on pushes to `main` / `develop` /
`nightly` / `pre-release`, on every pull request, on manual dispatch, and it is
called by `release.yml`. The local equivalent of its gate is `make test-ci`.

**`release.yml`** needs a one-time setup, because it holds no API token:

1. On **TestPyPI** → *Account settings* → *Publishing* → add a **pending
   publisher**: owner `HartmannLab`, repository `UELer`, workflow `release.yml`,
   environment `testpypi`.
2. The same on **PyPI**, with environment `pypi`. "Pending" is the mechanism for
   a project that does not exist on the index yet — which is our case exactly.
3. On GitHub → *Settings* → *Environments* → `pypi`, add a **required reviewer**
   so the real upload also needs a human click.

Then there are two paths, and only two:

| Action | What happens |
|---|---|
| `git push origin v0.5.0-alpha` | tests → build → verify tag against artifacts → upload to **TestPyPI**. Never touches PyPI. |
| Actions → *Release* → *Run workflow*, ref = **the tag**, `publish_to: pypi` | the same gates, then upload to **PyPI** (behind the environment reviewer) |

Before tagging: `make check-release TAG=v0.5.0-alpha`.

### C1. Add a CI test workflow — ✅ done

There is **no test workflow** — `.github/workflows/` contains only `docs.yml`.
Releasing a package with no automated test gate means the artifact users install
was validated only on one developer machine, on one Python version, with an
environment that happened to be missing bokeh.

- **Action:** add `.github/workflows/tests.yml`: matrix over the Python versions
  settled in **A3**, `pip install -e ".[dev]"`, `python -m unittest discover
  tests`. The fast-stub bootstrap makes this cheap (~6 s for 913 tests).
- **Make the bokeh gap explicit:** add a job (or a matrix leg) that installs the
  full runtime dependency set so the ~14 bokeh-gated tests actually run, and
  make the workflow **fail** if the skip count exceeds a threshold — silent
  skips are how the 3.11 gap stayed invisible.
- This closes the long-standing "Define and add a CI fast-stub job" open item in
  [topic_packaging_and_project.md](../topic_packaging_and_project.md).
- **Done:** [.github/workflows/tests.yml](../../.github/workflows/tests.yml) with two
  jobs. `unit` is a matrix over **3.10** and **3.11** that installs the *full*
  runtime stack (`pip install -e ".[dev]"`) and runs the suite through a new
  skip gate; `package` builds the sdist + wheel, runs `twine check --strict`,
  cross-checks the version declarations, then installs the wheel into a clean
  venv and imports it **from outside the repository** — the clean-checkout case a
  developer never sees, and the one **A5** was about. It uploads the artifacts as
  a `dist` artifact, which **C2** publishes.
- **The skip gate is the point.** [tools/run_test_suite.py](../../tools/run_test_suite.py)
  runs the same discovery `unittest` does, then prints every skipped test with its
  reason and exits non-zero if the count exceeds `--max-skips` (CI passes `0`).
  `make test-ci` runs the identical gate locally. Rationale: `unittest` prints
  `OK` for a run that quietly dropped 14 tests, which is precisely how the 3.11
  bokeh gap in the table above stayed invisible. **Measured: a complete
  environment skips 0 of 913 tests**, so `0` is a fact rather than an aspiration.
- **`actions/setup-node` is deliberate, not incidental.** Two tests in
  `tests/test_issue126_chip_reorder.py` gate on `shutil.which("node")` to parse
  the anywidget ESM bundle. `ubuntu-latest` happens to ship node, but a zero-skip
  gate is only honest if every optional tool is present on purpose.
- **The "fast-stub CI job" is not achievable, and that is the finding.** The open
  item assumed the `tests/bootstrap.py` stubs let the suite run without the heavy
  dependency stack. Measured in a fresh venv:
  - numpy only → **302** of 913 tests even collected, 113 errors
    (`traitlets`, `matplotlib.path`, `anndata` missing).
  - numpy + `traitlets` + `anndata` + real `matplotlib` → **671** collected, 68
    errors, 19 skips.
  - The blocker is structural: `_ensure_matplotlib_stub()` skips itself only when
    `matplotlib.pyplot` is **already in `sys.modules`**. In a full environment
    `seaborn_image` imports it first, so the stub never installs and the real
    library is used. In a minimal environment the stub *does* install — and it
    has no `matplotlib.path` and no `colors.Normalize`, so ~52 tests error out.
  - Conclusion: the stubs are a *speed* optimisation for an already-complete
    environment, not a substitute for one. A stub-based CI job would produce a
    green tick over a suite that never ran a third of itself — worse than no job.
    **The open item is closed as superseded**, not implemented. Reopen it only if
    someone is willing to finish the matplotlib stub, and note the full install is
    ~15 s of wheels anyway, so the payoff is small.
- **3.12 is a non-blocking matrix leg** (`continue-on-error: true`), which turns
  open decision C1 from a guess into a measurement — see *Decisions taken*.
- **Blocker found and fixed on the way in — `.gitignore` ignored `.github/`.** Line
  31 was a blanket directory rule, so both new workflow files were invisible to
  `git status` and would have been committed nowhere and run never. It also explains
  why `docs.yml` is the *only* tracked file under `.github/`: it was force-added.
  This is **A5's failure mode applied to CI instead of to assets**, and A5's own
  technique note predicts the trap — a negation cannot re-include anything under an
  ignored *directory*, so `!.github/workflows/*.yml` would not have worked. Fixed by
  narrowing the rule to the local-only assistant scaffolding
  (`.github/agents/`, `.github/chatmodes/`, `.github/prompts/`,
  `.github/copilot-instructions.md`), leaving `.github/workflows/` visible.
  **Verified both directions:** the two workflows now appear as `??` in
  `git status`, and all four scaffolding paths are still reported ignored by
  `git check-ignore`. `MANIFEST.in` already prunes `.github`, so nothing about this
  reaches the sdist.
- **Verified:** both workflow files parse as YAML; the Python embedded in the
  wheel-check step was extracted and `ast.parse`d, then **run verbatim** against a
  clean venv install of `dist/ueler-0.5.0a0-py3-none-any.whl` from `/tmp` (it
  imports `ueler`, imports the heaviest module `ueler.viewer.main_viewer`, reads
  the packaged `ready.png` through `load_asset_bytes`, and asserts none of the
  four legacy top-level names leaked into `sys.modules`) — exit 0. The skip gate
  was exercised in both directions: exit 0 on the full environment with 0 skips,
  exit 1 in the minimal venv with all 19 skips printed and named.
- **Now verified on GitHub's runners** (run `31407846635`, branch `nightly`):
  `unit (py3.10)` ✅, `unit (py3.11)` ✅, `build + verify the distribution` ✅,
  `unit (py3.12)` ❌ — and the **run as a whole is `success`**, which is the
  `continue-on-error` contract working as designed. The 3.12 result is written up
  in **C5** below. The publish jobs still cannot be exercised without a tag.

### C2. Add a release workflow with Trusted Publishing — ✅ done

- **Action:** `.github/workflows/release.yml` triggered on `v*` tags: build,
  `twine check`, then publish via **PyPI Trusted Publishing** (OIDC, no API
  token in repo secrets — the current best practice). Gate it on **C1** passing.
- Configure the trusted publisher on PyPI *before* the first upload, since it
  must be created against a project that may not exist yet (PyPI supports
  "pending" publishers for exactly this).
- **Done:** [.github/workflows/release.yml](../../.github/workflows/release.yml),
  four jobs: `tests` (calls `tests.yml` as a reusable workflow), `verify`,
  `testpypi`, `pypi`. Both uploads use `pypa/gh-action-pypi-publish` with
  `id-token: write` — **no API token exists in the repository's secrets**.
- **Deliberately asymmetric, because PyPI is append-only.** A tag push gets you as
  far as **TestPyPI** and no further. The real upload requires a
  `workflow_dispatch` with `publish_to: pypi` **and** a tag as the ref, so there
  is no sequence of ordinary git commands that publishes to PyPI by accident. The
  plan's "triggered on `v*` tags: … then publish" would have made
  `git push --tags` an irreversible act; that seemed like the wrong default for
  the one operation in this project that cannot be undone.
- **It publishes what was tested.** `tests.yml` is invoked with `workflow_call`, so
  it runs at the release commit inside the same run, and its `package` job's `dist`
  artifact is what the publish jobs download. Nothing is rebuilt between the test
  and the upload — the same reasoning as `make publish` depending on `check-dist`
  rather than on `build` (**A4**).
- **`verify` checks the tag against the artifacts, not just against the source.**
  [tools/check_release_tag.py](../../tools/check_release_tag.py) compares four
  things — the tag, `pyproject.toml`, `ueler.__version__`, and both `dist/`
  filenames — on **PEP 440-normalised** versions. That normalisation is load-bearing:
  the repo tags in SemVer style (`v0.2.0-alpha` is an existing tag) while setuptools
  writes `0.5.0a0`, so a string comparison would reject a *correct* tag. Both
  `v0.5.0-alpha` and `v0.5.0-a0` validate, since both normalise to `0.5.0a0`.
- **`skip-existing: true` on the TestPyPI upload only.** A rehearsal may already
  have pushed that version by hand, and a workflow re-run should not fail on it.
  The PyPI job has no such flag — there, a duplicate must be an error.
- **Setup the developer must still do by hand** (see the operational table above):
  pending Trusted Publishers on both indexes, plus a required reviewer on the
  `pypi` GitHub environment. Until those exist the publish jobs will fail at the
  OIDC exchange, which is the correct failure.
- **Verified:** YAML parses; job graph is `tests → verify → {testpypi, pypi}`. The
  `verify` step's checker was run locally against the real `dist/` for a correct
  tag (exit 0, both spellings) and a wrong one (`v0.5.0` → exit 1). The publish
  jobs themselves cannot be tested without pushing a tag — that is **D1**.

### C4. **BLOCKER — PyPI will not accept the name `ueler`** 🚨 open

**Not in the original plan; found 2026-08-10 while the developer configured the
Trusted Publishers.** PyPI's pending-publisher form rejects the project name with
**"This project name isn't allowed."**

- **What is and is not established.** Measured here: `ueler` is *unregistered* on
  PyPI — 404 from both `https://pypi.org/pypi/ueler/json` and
  `https://pypi.org/simple/ueler/`. So this is **not** "someone took the name";
  PyPI's message for that case is "This project name is already being used."
  The message the developer got is warehouse's response for a name matching its
  **prohibited-names** list, which is checked against both the PEP 503 normalised
  form and an "ultranormalised" form that collapses visually confusable characters
  (`l`/`I`/`1`, `0`/`O`, and `-_.`). That list is administrative and **not readable
  from outside PyPI**, so the *reason* for the block cannot be determined from here
  — only that the block exists and is not an availability problem.
- **Confirmed 2026-08-10: `ueler` and `UELer` fail identically**, which is the expected
  result — the validator normalises before comparing, so the two are the same name to
  PyPI. The form also shows a generic banner, **"The trusted publisher could not be
  registered"**, above the field-level reason; they are two halves of one failure, not
  two problems.
- **Control test run 2026-08-10: `ueler-viewer` registers successfully.** So the
  rejection is specific to the string `ueler` — not the account, not the pending-publisher
  cap, not the form, not 2FA. `ueler` is unregistered *and* administratively blocked.
  The `ueler-viewer` pending publisher is being kept rather than deleted: it costs
  nothing and it is the ready-made fallback if PyPI declines.
- **Why it is a blocker rather than a nuisance:** the distribution name is the one
  piece of packaging that everything else quotes. `pip install ueler` appears in the
  README, `docs/installation.md`, `docs/index.md` and the Binder instructions, and
  it is what **D3** uploads. Nothing after Gate C can proceed until the name is
  settled.
- **Steps, cheapest first:**
  1. Re-try the form with lowercase `ueler`. Validation normalises the name so case
     should not matter, but eliminate it in ten seconds.
  2. Submit a control name (e.g. `ueler-viewer`) on the same form. If that is
     accepted, the block is specific to `ueler`; delete the control publisher after.
  3. **Ask PyPI to release the name.** Start by email to `admin@pypi.org` — one of
     only two addresses published on [pypi.org/help](https://pypi.org/help/) (the
     other is `security@pypi.org`).
     **↳ Superseded in practice (2026-08-10): the admins replied and asked for a
     `pypi/support` issue, so that is now the route.** File it under the
     **PEP 541 template**, which is the only way in — `pypi/support` sets
     `blank_issues_enabled: false` and its seven templates are network access,
     account recovery, two size limits, mass name squatting, and PEP 541, none of
     which covers a prohibited name. Two things make it survive triage:
     - **State in the first line that this is not a PEP 541 claim**, and name the
       admin who referred you plus the date of their mail. PEP 541 governs claiming
       a name **away from an existing owner**; `ueler` has no owner and no project
       object at all, so without the referral quoted up front the issue reads as a
       malformed claim and gets closed.
     - **Leave the "Maintenance or replacement?" dropdown unset** — it carries no
       `required` validation and neither option is true. Fill "Project to be
       claimed" with the 404ing URL and say it 404s, rather than omitting it.
     Post the issue link back on the email thread so the two records do not drift.
     **Which of PyPI's four documented reasons applies**, per its own
     [`#project-name`](https://pypi.org/help/#project-name) answer — three eliminated
     empirically, so the fourth holds by elimination:

     | Reason | Verdict |
     |---|---|
     | Conflicts with a Python stdlib module (2.5 → present) | ✗ there is no `ueler` module |
     | Too similar to / confusable with an existing project | ✗ every confusable variant 404s — `ueier`, `ue1er`, `u-eler`, `u_eler`, `ue-ler`, `uel-er`, `uele-r`, `ueller`, `uelr`. Only `euler` exists, and character-level confusable folding maps lookalike *characters*, not letter order |
     | Registered by another user with no releases | ✗ `/simple/ueler/` returns 404, so no project object exists — a registered-but-empty project would still have a simple-index page |
     | **Explicitly prohibited by the PyPI administrators** | ✓ by elimination |

     Include in the mail: the exact error, that `ueler` 404s on both
     `/pypi/ueler/json` and `/simple/ueler/`, that `ueler-viewer` registers
     immediately from the same account and form, that no confusable variant exists,
     and the evidence this is a real tool (repository, docs site, Binder deployment,
     the existing TestPyPI project). **Set expectations accordingly:** this is an
     unqueued admin request with no SLA, and a name can be prohibited for a reason
     nobody outside PyPI can see.
  4. **Only if that fails: rename the distribution.** Note the import name is
     independent — `pyproject.toml`'s `name` can become `ueler-viewer` while
     `import ueler` stays exactly as it is, so **no code changes**. What changes is
     `pyproject.toml`, every `pip install ueler` in the README / `docs/installation.md`
     / `docs/index.md`, and the TestPyPI publisher. The mismatch is unremarkable —
     `scikit-image`/`skimage`, `opencv-python`/`cv2`, `pillow`/`PIL`.
- **The name only has to be settled before D3, not before D1.** TestPyPI accepts
  `ueler`, so the rehearsal and the notebook smoke test can run now while the support
  request is open. Decide the final name before the real upload, because the cleanest
  outcome is publishing under one name from the start: if `0.5.0-alpha` goes out as
  `ueler-viewer` and PyPI later grants `ueler`, you end up maintaining two
  distribution names for a while.
- **Do not try to force it with a token upload first.** The prohibited-name check runs
  on upload too, so it fails identically — and if it somehow did not, `0.5.0a0` would
  be on real PyPI before the rehearsal, which is the exact outcome Gate D exists to
  prevent.
- **This is the argument for Gate A existing, arriving on schedule.** The name is the
  one property that cannot be corrected after an upload, and it surfaced during
  configuration rather than during `twine upload`.
- **Not blocked by this:** the TestPyPI rehearsal (**D1**), because TestPyPI already
  accepts the name — see below.
- **RESOLVED (2026-08-19): take step 4 — the distribution is `ueler-viewer` on both indexes.** `pyproject.toml`'s `[project] name` is renamed; the import name stays `ueler`, so no code, no test and not `ueler.__version__` changed. A *dual* name (`ueler` on TestPyPI, `ueler-viewer` on PyPI) was planned first and rejected on cost: the name is baked into the artifact filename, `METADATA` and `RECORD`, so it would take two builds, a `pyproject.toml`-rewriting build tool, two artifacts through `release.yml`, and a wrong-name guard — and the TestPyPI rehearsal would stop validating the artifact PyPI receives. The TestPyPI `ueler` project stays registered and simply stops receiving uploads; TestPyPI does not reclaim names for inactivity, and PyPI's decision about `ueler` does not depend on activity there. Full record, verification and the exit ramp for a later grant of `ueler`: [`issue79_dist_name_rename.md`](issue79_dist_name_rename.md). **This unblocks Gate D:** `release.yml` needed no change, so a `v*` tag push now publishes `ueler-viewer` to TestPyPI automatically, and that first upload is also what claims the name — a pending Trusted Publisher does not reserve it (measured: the publisher existed for a week while `/simple/ueler-viewer/` still 404'd).
- **New finding (2026-08-17) — the prohibition is most likely `euler` typosquat protection.** `euler` is a live PyPI project (1.0.3, 3 releases, "A library for Euler angle computation and conversion") and does not exist on TestPyPI, which is exactly why TestPyPI accepted `ueler` and PyPI does not. This does not overturn the elimination table above, it explains it: the table ruled out row 2 for the **automated** check only — ultranormalisation folds confusable *characters*, not a transposition of letter order — so row 4 still holds, and `euler` is the likely *motive* for that manual entry. Consequence: a prohibition protecting a live project is the kind admins decline to lift, so `ueler-viewer` should be planned as a long-lived name rather than a two-week placeholder.

### C3. Reconcile the git tags with the shipped versions — ✅ decided: no backfill

`git tag --sort=-v:refname` shows `v0.4.1` as the newest tag, but `v0.4.2`,
`v0.4.3` and `v0.4.4` all shipped per `doc/log.md`. If **C2** triggers on tags,
this drift becomes a release mechanism problem rather than a bookkeeping one.

- **Action:** backfill the missing tags against their release commits (or
  consciously decide not to and start clean from `v0.5.0`).
- **Decided by the developer (2026-08-10): do not backfill. The skipped tags were
  intentional.** The premise of this item was wrong — it read the gaps as drift,
  and they were choices. Nothing is broken: `doc/log.md` remains the record of what
  shipped, and tags mark what was *released through the release mechanism*, which
  did not exist for `v0.4.2`–`v0.4.4`.
- **Consequence for C2:** none. `release.yml` triggers on tag *pushes*, and the
  historical tags are already pushed, so no old tag can fire it. The first tag the
  workflow will ever see is the next one.
- **First tag under the new mechanism: `v0.5.0-alpha`**, created after Gate D. That
  spelling matches the existing `v0.2.0-alpha` and the version-bump skill's SemVer
  convention; `v0.5.0-a0` also validates, since the checker normalises both to
  `0.5.0a0`. Either works — `v0.5.0-alpha` is the consistent one.

### C5. The 3.12 leg's first reading — ✅ fixed (it was a pandas bug, not a 3.12 one)

The non-blocking leg existed to turn open decision #3 into a measurement. It ran
on the first push (run `31407846635`) and returned **19 errors**, with 3.10, 3.11
and `package` all green. The finding is not what the open decision anticipated.

**What the 19 errors were.** All of them, without exception:
`TypeError: Cannot interpret '<StringDtype(na_value=nan)>' as a data type`. All of
them passed through the *same four-line closure* — `_convert` inside
`MaskPainterDisplay.apply_colors_to_masks` (`mask_painter.py`), which converts the
class labels the widgets return (always strings) to the identifier column's dtype
so `cell_table[identifier] == value` matches rows. It asked `np.issubdtype`.
Extracted from the tracebacks, the implicated `ueler/` lines were exactly:
`mask_painter.py:1864` ×19 (and `:1870` ×38, the two call frames per failure).

**It is not a Python 3.12 problem — proved by reproducing it on 3.10.** Setting
`pd.options.future.infer_string = True` on this project's own environment (Python
3.10.19, pandas 2.3.3) yields **19 identical errors**, an exact match with CI. The
matrix difference is the *resolver*, not the language: **pandas 3.0 requires
Python ≥ 3.11**, so the 3.12 leg received a pandas that had already made
`StringDtype` the default for object string columns, and `np.issubdtype` raises
`TypeError` on every pandas **extension** dtype.

**So the obvious action would have been the wrong one.** Open decision #3 offered
"add the 3.12 classifier, or tighten to `<3.12`". Tightening would have removed
the *symptom* while leaving the bug: pandas 3 installs happily on 3.11, so the
same 19 errors sat one dependency resolution away from a **supported** Python. The
3.11 leg passing is the resolver's current preference, not evidence of
correctness.

**The bug was already live on current pandas, which is the part that mattered.**
`np.issubdtype` also raises on `Int64Dtype`, `Float64Dtype` and `CategoricalDtype`,
and `flatten_anndata` produces all three — `obs` columns arrive categorical as a
matter of course. Painting a cell table whose identifier column is a `category`
therefore crashed on **today's** pandas, on **every** supported Python. No test
covered it, so nothing said so. This is the strongest argument for the CI gate
that has come up in the whole issue: the job found a user-facing bug on its first
run, in a code path the release audit had not touched.

**Fix.** `is_integer_column_dtype` / `is_float_column_dtype` in
`ueler/cell_table.py` (the cell-table dtype domain, already imported by
`mask_painter.py` for `categorical_columns`). They ask `pandas.api.types` first —
it understands numpy and extension dtypes alike — and fall back to `np.issubdtype`
only for the minimal pandas stand-in in `tests/bootstrap.py`, whose `api.types`
namespace carries a different, *value*-based set of helpers. Unknown objects
return `False` rather than raising.

- **Categoricals are unwrapped to `.categories.dtype`, deliberately.**
  `pd.Series(pd.Categorical([1, 2])) == "1"` matches **0** rows where `== 1`
  matches 1. Classifying `category` as "neither integer nor float" would have
  traded a loud `TypeError` for a silently empty selection — strictly worse,
  because an empty result looks like an answer.
- **`pandas` is now a declared dependency** (`pandas>=2.0`). Fourteen modules
  `import pandas as pd` while nothing in `pyproject.toml` named it; it arrived
  through seaborn and anndata. That omission is *why* pip could hand a different
  pandas major to a different Python minor with no signal. The floor is
  conservative rather than measured, and the comment in `pyproject.toml` says so.
- **`anndata` was checked and deliberately left at `>=0.10`.** Under
  `future.infer_string` this environment shows 24 *further* failures, but they are
  confined to three h5ad modules and fail **inside anndata 0.11.4**
  (`IORegistryError: No method registered for writing
  ArrowStringArrayNumpySemantics`), not in UELer. They never appeared in CI, and
  the pairing is unreachable by a fresh install: pandas 3 needs ≥ 3.11; on 3.11+
  anndata 0.12+ is available and handles it; on 3.10 the newest anndata is 0.11.4
  but pandas 3 cannot be installed at all.
- **Noted for later:** anndata 0.12 requires ≥ 3.11 and anndata 0.13 requires
  ≥ 3.12, and pandas 3 requires ≥ 3.11. The part of this project's Python range
  under real pressure is the **3.10 floor**, not the 3.12 ceiling.

**Tests — 9 new, in `tests/test_cell_table_dtype_predicates.py`.** They cover each
dtype family, assert that `np.issubdtype` *does* raise on the extension dtypes (so
the helper's reason for existing is itself a test), and drive
`apply_colors_to_masks` end to end through `category` and nullable-`Int64`
identifier columns — cases that fail on **every** supported pandas, so the
regression is caught without needing pandas 3 present. The pandas-3 string dtype
joins the parameter list only when constructible, rather than via `skipUnless`,
because CI runs `--max-skips 0` and a skip would be an invisible hole.

**Verified.** Non-vacuous: restoring the old `np.issubdtype` call makes the
end-to-end group fail with 2 errors; the new code makes it pass.
`python tools/run_test_suite.py --max-skips 0` → **922 tests, OK**, "No tests were
skipped", exit 0. Under `future.infer_string`, `StringDtype` errors go **19 → 0**
and only the 24 anndata ones remain. `python -m build` clean; wheel `METADATA`
reads `Requires-Dist: pandas>=2.0`. **Not verified:** the 3.12 leg itself — that
needs the next push.

**Standing reproducer, no new tooling required:**

```bash
python -c "import pandas as pd; pd.options.future.infer_string = True; \
import runpy, sys; sys.argv = ['x', '--max-skips', '0']; \
runpy.run_path('tools/run_test_suite.py', run_name='__main__')"
```

---

## Gate D — rehearsal and publish

### D1. Publish to **TestPyPI** first and install from it

Non-negotiable, given PyPI's append-only nature. TestPyPI is the only way to see
the real rendered page and exercise the real resolver.

```bash
make publish-test
# in a scratch venv, on a machine that has never seen the repo:
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ "ueler[notebook]"
```
The `--extra-index-url` is required: TestPyPI does not mirror the 22 real deps.

**Update 2026-08-10 — TestPyPI already has a `ueler` project, and it is ours.**
`ueler` **0.3.1** was uploaded there on 2026-04-10 by the developer, from the same
stale `dist/` that **A4** later purged. Consequences, none of them harmful but all
worth knowing before the rehearsal:

- **The project exists, so a *pending* publisher is the wrong mechanism there.**
  Pending publishers are for names with no project behind them. Confirm the
  publisher actually attached: test.pypi.org → *Your projects* → `ueler` → *Manage*
  → *Publishing* should list it. If it only appears under *Account settings →
  Publishing* as pending, add it on the project instead, or the OIDC exchange will
  fail at upload.
- **Do not judge the metadata from the 0.3.1 page.** It predates all of Gate A and B:
  `Summary` is the old "Usability Enhanced Linked Viewer for MIBI imaging" and the
  license shows as **Proprietary**. The point of **D1** is to read the *0.5.0a0*
  page.
- **No version collision.** 0.5.0a0 ≠ 0.3.1, so `skip-existing: true` on the
  workflow's TestPyPI step is not doing any work here — it is there for re-runs.
- **The name is accepted on TestPyPI**, which is why **C4**'s block on PyPI did not
  show up in April.

- **Check:** the rendered README, the `Summary` line, classifiers, and the URLs
  sidebar.
- The root-relative-link problem called out here (`/doc/GUI_preview.png` and the
  two `/doc/log.md` links — fine on GitHub, broken on PyPI) was **fixed in Gate A**
  along with **A2**. What remains is confirming visually that they render.

### D2. End-to-end smoke test from the installed package

Install from TestPyPI into a *fresh* conda env on a machine without the repo,
then run `script/run_ueler.ipynb` against a real dataset — open a FOV, toggle a
mask, paint, save an ROI, export. The unit suite runs against stubs and has
never proven the real widget stack works from an installed wheel.

- This is the one step that cannot be automated here: the dev environment has no
  browser, so it needs a live Jupyter session on the developer's side.

### D3. Publish, tag, announce

1. Final `make build` from a clean checkout of the release commit.
2. `twine check dist/*`.
3. `make publish` (or push the tag and let **C2** do it).
4. `git tag v0.5.0` (or the agreed final version) and push.
5. Repoint the Binder badge / docs at the released version if desired.

---

## Deferred — explicitly out of scope for the first release

- conda-forge feedstock (worth doing eventually for the HPC audience given the
  `hdf5`/`opencv` binary stack, but it depends on a PyPI release existing first).
- Loosening the exact `ark-analysis==0.7.0` pin in the `[ark]` extra.
- Wheels per platform — UELer is pure Python, `py3-none-any` is correct.
- SBOM / provenance attestation beyond what Trusted Publishing gives for free.

---

## Suggested sequencing

Gate A landed as **one working-tree change set** rather than the originally
suggested three commits, because the items are individually small and were
verified together against a single rebuild. Splitting at commit time is still
reasonable:

1. **`chore(packaging): make the build reproducible and safe to publish`** —
   A4 (dist purge + Makefile targets), A5 (`.gitignore` negations), A6
   (setuptools floor), A7 (`MANIFEST.in`).
2. **`chore(dev): make the test bootstrap opt-in`** — A8.
3. **`docs(packaging): PyPI-facing metadata and install instructions`** —
   A1, A2, A3, the README link fixes.

Gate A landed as `88b0e6e`. Gate B is likewise one working-tree change set;
split at commit time if you prefer:

4. **`chore(packaging): finish PyPI metadata`** — B1 (classifiers, `[project.urls]`),
   B5 (`graphify-out` relocation).
5. **`docs: state the license and the PyPI install path`** — B4, B6, plus the
   `doc/log.md` and README summary entries.

Gate B landed as `f9fcfae`. Gate C is one working-tree change set on top of it:

6. **`ci: add test and release workflows`** — C1 (`tests.yml`,
   `tools/run_test_suite.py`), C2 (`release.yml`, `tools/check_release_tag.py`),
   the `Makefile` targets, and the C3 decision recorded in the docs.

Then Gate D by hand.

### Gate D, in the order it has to happen

Gate C built the mechanism; nothing has exercised it. The remaining sequence:

1. ~~**Push the Gate C commit** and let `tests.yml` run for the first time.~~ **Done**
   — run `31407846635`. The 3.12 leg's 19 errors turned out to be a pandas bug that
   was already live on current pandas; fixed in **C5**. **Push the C5 fix and read
   the 3.12 leg again**: if it is green, widen the classifiers (`classifiers` and
   `requires-python` move together). Do not tighten to `<3.12` — that was the answer
   the first reading ruled out.
2. **Configure the Trusted Publishers** on TestPyPI and PyPI, plus the `pypi`
   environment reviewer (see the table in Gate C).
3. `make check-release TAG=v0.5.0-alpha`, then **push the tag** → the workflow
   rehearses on **TestPyPI** by itself. That is **D1**.
4. **Install from TestPyPI** into a fresh env and read the rendered page — the
   sidebar, the `Summary`, the long description. Then the notebook smoke test
   (**D2**), which needs a browser this environment does not have.
5. **Dispatch the release** to PyPI (**D3**).

### Open decisions for the developer

*(A1, A3, B4, C1, C2, C3 and D3 are resolved — see
[Decisions taken](#decisions-taken). B2 and B3 are closed as "declined for
`0.5.0`", which needed no code change.)*

1. 🚨 **C4 — `ueler` vs `ueler-viewer` on PyPI.** `ueler` is unregistered but
   administratively **prohibited**; `ueler-viewer` registers fine (control test), so
   the block is name-specific. `admin@pypi.org` was mailed and **the admins replied
   asking for a `pypi/support` issue** — file it under the **PEP 541 template** (the
   only way in; blank issues are disabled), leading with "not a PEP 541 claim" and
   the referring admin's name and date, or triage will close it. Fall back to
   `ueler-viewer` — whose pending publisher already exists — if they decline or go
   quiet. Renaming needs no code changes, only `pyproject.toml` and every
   `pip install ueler` in the README and the docs. **Not blocking D1/D2** (TestPyPI
   accepts `ueler`); it must be settled **before D3**.
2. **B4 leftover (non-code)** — confirm with **DKFZ** that the BSD copyright line
   naming both the author and the institute matches institutional policy. The
   licensing decision itself is made; this is the institutional half of it.
3. **The 3.12 answer — one reading in, and it changed the question.** The leg's
   first run reported 19 errors that were a **pandas** bug, not a 3.12
   incompatibility, and they are now fixed (**C5**). Tightening `requires-python`
   to `<3.12` would have hidden it rather than fixed it. What is left is genuinely
   a reading: re-run after this lands, and if 3.12 is green, add
   `Programming Language :: Python :: 3.12` and drop `continue-on-error`. **Keep
   the leg regardless of the decision** — it is the project's only coverage of
   pandas-3 semantics, and it is what caught this. Classifiers and the bound still
   move together.
4. **A1 leftover** — the **GitHub repo description** still needs updating in the
   GitHub UI; it is not a tracked file.
5. **B5 leftover (trivial)** — `graphify-out/legacy-ueler-scoped-2026-07-31/` is
   kept only because deleting someone else's generated data unasked is rude. It is
   stale and regenerable; delete it when you notice it.
