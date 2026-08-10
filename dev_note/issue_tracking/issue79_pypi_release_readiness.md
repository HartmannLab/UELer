# Issue #79 — PyPI release readiness plan

> Status: written 2026-08-10 against `develop` @ `377c1bb` + working tree at `v0.5.0-alpha`.
> **Gate A is complete** (committed as `88b0e6e`). **Gate B is complete**
> (2026-08-10, working tree — B2 and B3 consciously declined, plus one item the
> original list missed). Gates **C** and **D** remain open. Developer decisions
> taken so far are recorded under [Decisions taken](#decisions-taken).
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
| PyPI name availability | ✅ both `ueler` and `UELer` are free (404 on the JSON API) |
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

## Gate C — infrastructure

### C1. Add a CI test workflow

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

### C2. Add a release workflow with Trusted Publishing

- **Action:** `.github/workflows/release.yml` triggered on `v*` tags: build,
  `twine check`, then publish via **PyPI Trusted Publishing** (OIDC, no API
  token in repo secrets — the current best practice). Gate it on **C1** passing.
- Configure the trusted publisher on PyPI *before* the first upload, since it
  must be created against a project that may not exist yet (PyPI supports
  "pending" publishers for exactly this).

### C3. Reconcile the git tags with the shipped versions

`git tag --sort=-v:refname` shows `v0.4.1` as the newest tag, but `v0.4.2`,
`v0.4.3` and `v0.4.4` all shipped per `doc/log.md`. If **C2** triggers on tags,
this drift becomes a release mechanism problem rather than a bookkeeping one.

- **Action:** backfill the missing tags against their release commits (or
  consciously decide not to and start clean from `v0.5.0`).

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

Remaining:

6. **`ci: add test and release workflows`** — C1, C2, C3.

Then Gate D by hand.

### Open decisions for the developer

*(A1, A3 and B4 are resolved — see [Decisions taken](#decisions-taken). B2 and B3
are closed as "declined for `0.5.0`", which needed no code change.)*

1. **B4 leftover (non-code)** — confirm with **DKFZ** that the BSD copyright line
   naming both the author and the institute matches institutional policy. The
   licensing decision itself is made; this is the institutional half of it.
2. **C1** — add **3.12** to the CI matrix, or tighten `requires-python` to
   `<3.12`? The current `<3.13` bound permits an untested minor. Note the
   per-minor classifiers added in **B1** list only 3.10 and 3.11, so whichever way
   this goes, `classifiers` and `requires-python` must move together.
3. **C3** — backfill `v0.4.2`–`v0.4.4` tags, or start clean at `v0.5.0`?
4. **D3** — is `0.5.0-alpha` the version that goes to PyPI, or should the first
   public upload be a final `0.5.0`? (An `-alpha` upload is a normal way to
   validate the pipeline, and pip will not install it without `--pre`. The README
   documents both forms, so either choice is already covered.)
5. **A1 leftover** — the **GitHub repo description** still needs updating in the
   GitHub UI; it is not a tracked file.
6. **B5 leftover (trivial)** — `graphify-out/legacy-ueler-scoped-2026-07-31/` is
   kept only because deleting someone else's generated data unasked is rude. It is
   stale and regenerable; delete it when you notice it.
