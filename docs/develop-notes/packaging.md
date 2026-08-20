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
- **`pip install ueler-viewer` for users, editable install for developers.** An index is the documented install path; `pip install -e .` from a clone is for working on UELer itself, where it makes `git pull` upgrades instant.
- **The distribution is named `ueler-viewer`; the import name stays `ueler`.** PyPI rejects the project name `ueler` as administratively prohibited — most likely typo-squat protection for the live `euler` project, since `ueler` is a transposition of its first two characters and TestPyPI, which has no `euler`, accepted the name without complaint. The two names are independent: the distribution name is `[project] name` in `pyproject.toml`, the import name is the `ueler/` package directory, so the rename touched no code, no test, and not `ueler.__version__` (a literal, not an `importlib.metadata` lookup). Precedent: `scikit-image`/`skimage`, `opencv-python`/`cv2`, `pillow`/`PIL`. **One name on both indexes** was chosen over publishing `ueler` to TestPyPI and `ueler-viewer` to PyPI: the name is baked into the artifacts, so a dual name means two separate builds, and the TestPyPI rehearsal would then no longer validate the artifact PyPI receives — which is the entire point of having a rehearsal. The TestPyPI `ueler` project stays registered and simply stops receiving uploads; nothing about it expires. Full analysis and the exit ramp for a later grant of `ueler`: `dev_note/issue_tracking/issue79_dist_name_rename.md`.
- **A pending Trusted Publisher does not reserve a project name.** Measured: the `ueler-viewer` pending publisher existed on PyPI for a week while `/simple/ueler-viewer/` still returned 404, so no project object existed and the name was claimable by anyone. The name is created and claimed by the **first upload** — which is why the first tagged release doubles as the reservation.
- **Fast-stub test bootstrap, opt-in.** `tests/bootstrap.py` stubs out heavy dependencies (`pandas`, `ipywidgets`, `matplotlib`) so the test suite runs quickly without a full environment. The `sitecustomize.py` / `usercustomize.py` startup hooks initialise it **only** when `UELER_TEST_BOOTSTRAP=1` is set — `make test-fast` and `make test-integration` set it for you. Defaulting it on meant any interpreter with the repo root on `PYTHONPATH` could silently run against fake scientific libraries; a bootstrap that was requested and then failed now emits a `RuntimeWarning` instead of being swallowed.
- **Packaged assets — and workflow files — must be visible to git.** `.gitignore` has blanket `*.txt` / `*.png` rules, so assets are re-included explicitly (`!LICENSE.txt`, `!ueler/**/*.png`, `!doc/**/*.png`, `!docs/**/*.png`). setuptools builds from the working tree, so an ignored asset ships from a developer's machine and vanishes from a clean-checkout build — add new asset types to those negations. CI hit the same rule from the other side: `.github/` was ignored as a whole directory, so a new workflow file simply never reached GitHub. **A negation cannot re-include anything under an ignored directory**, so the directory rule was narrowed to the local-only assistant scaffolding instead, leaving `.github/workflows/` tracked.
- **`MANIFEST.in` keeps the sdist to build inputs only.** No tests are shipped: making them runnable would require shipping `bootstrap.py`'s dev-only stub machinery.
- **Supported Python: 3.10–3.12** (`requires-python = ">=3.10,<3.13"`), with three per-minor `Programming Language :: Python ::` classifiers matching. `requires-python` and the classifiers are the same claim stated twice, so they move together — a bound that permits a minor the classifiers omit tells an installer and a human two different things. **CI does not yet back the claim symmetrically:** `tests.yml` runs 3.10 and 3.11 as blocking legs and 3.12 as `continue-on-error`, so 3.12 is declared supported but cannot fail the build. Closing that gap means dropping `experimental` from the 3.12 leg, which is a policy call, not a docs one — see the open item below.
- **BSD-3-Clause, relicensed from GPL-3.0-only before the first PyPI upload.** UELer is a library other people import, and copyleft there propagates into the importer's distributed work — the opposite of what a lab tool wants. Every runtime dependency is already permissive (BSD-3 / MIT / Apache-2.0), so nothing obliged the GPL; and copyright sits with a single author, so the change needed no contributor round-up. BSD-3 matches `scikit-image`, `dask`, `bokeh`, `anndata` and `napari`.
- **No `License ::` classifier.** PEP 639 forbids combining one with the `license` SPDX expression that `pyproject.toml` declares; setuptools warns if both are present. The license reaches the metadata as `License-Expression: BSD-3-Clause`.
- **The PyPI metadata says what UELer *is*, not what it competes with or nearly does.** Two entries were dropped on 2026-08-20. `napari-alternative` left `keywords`: nobody searches PyPI for that string, and the field exists to match the words a user actually types rather than to stake out a position against another project — the remaining seven keywords are all things UELer is or reads (`spatial proteomics`, `multiplexed imaging`, `image viewer`, `jupyter`, `MIBI`, `IMC`, `bioimage analysis`). `Topic :: Scientific/Engineering :: Image Processing` left `classifiers`, taking the block from ten entries to nine: that trove topic is where people look for libraries that *transform* pixels — filtering, segmentation, registration, morphology — and UELer offers none of that; it loads, links and displays images and cell tables other tools produced. `Visualization` and `Bio-Informatics` describe it without over-claiming, and the reason is recorded in a comment above the classifier block beside the `License ::` note so neither entry gets added back as a "fix".
- **A skipped test is a failure, not a pass.** CI runs the suite through
  `tools/run_test_suite.py --max-skips 0`, which prints every skip with its reason
  before deciding the exit code. Plain `unittest` prints `OK` for a run that
  silently dropped 14 bokeh-gated tests — that is how the Python 3.11 coverage gap
  stayed invisible until the release audit. A complete environment skips 0 of the
  1109 tests currently in the suite, so zero is a measurement, not an aspiration.
  `make test-ci` runs the same gate locally.
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
- **The tag routes itself: a pre-release goes to TestPyPI, a stable release goes to TestPyPI and then PyPI.** The routing predicate is `packaging.version.Version(tag).is_prerelease`, computed by `tools/release_channel.py` and tested against every tag in the repository's history. It needs no special-casing, because the repo's SemVer spellings (`-alpha`, `-alphaN`, `-rcN`) normalise to PEP 440 pre-release segments while a plain `vX.Y.Z` does not — and `v1.0.0.post1`, correctly, is not a pre-release. This **supersedes** the earlier rule that a tag push could never reach PyPI: that rule made `git push --tags` safe by requiring a second manual act, which proved only that a human clicked. The replacement is a stronger claim about the artifact (the rehearsal gate below) plus a required reviewer on the `pypi` environment, so a stable tag still cannot upload without a human, but the human is approving a run that is already built, verified and rehearsed rather than filling in a form.
- **A stable release must be the promotion of a release candidate TestPyPI already serves.** `tools/check_stable_rehearsal.py` refuses `vX.Y.Z` unless an `rc` tag exists for the same version, the **highest** such rc is served by TestPyPI (asked of the index, since a tag proves nothing about whether its upload succeeded), and everything that ships in the wheel is unchanged between the two tags. "Unchanged" cannot mean the same commit or identical artifacts: `check_release_tag.py` requires the tag, `pyproject.toml`, `ueler.__version__` and both filenames to describe one release, so the rc necessarily declares `0.5.0rc1` where the stable declares `0.5.0`. The comparison is therefore scoped to `ueler/**` and `pyproject.toml`, permitting only the `__version__` and `version` lines to move — and checking that they move from exactly the rc version to exactly the stable one. Freezing `pyproject.toml` matters as much as freezing the code: a dependency floor edited after the rc changes the wheel's metadata and invalidates the rehearsal even though no Python moved. Docs, `doc/log.md` and `tests/**` stay free, which is exactly what the version-bump skill touches on an rc → stable bump. Ancestry is *reported*, not enforced: the content comparison already proves the shipped bytes match, while `git merge-base` can fail on a rebase or a shallow fetch without saying anything about the artifact. The rule applies to the next stable tag onward; nothing already released is revisited.
- **Every publishing run passes through TestPyPI, stable releases included.** `pypi` has `needs: testpypi`, so the cheapest possible pre-flight always runs first and the failure direction is the safe one. It also keeps TestPyPI a complete mirror of release history, which is what makes the *next* release's rehearsal check meaningful.
- **Generated caches stay out of `ueler/`.** The graphify output belongs at the repo root; a build tool that globs package data is one `package-data` change away from shipping it, and the `.gitignore` negation trick above cannot rescue a directory rule.

---

## Package Layout

```
ueler/
├── __init__.py          # Public API surface
├── runner.py            # Programmatic entrypoint (run_viewer, run_viewer_bia)
├── constants.py         # Shared defaults
├── data_loader.py       # TIFF-folder and OME-TIFF ingestion
├── bia_loader.py        # BioImage Archive streaming
├── cell_table.py        # Cell table / AnnData handling
├── image_utils.py       # Image helper functions
├── rendering/
│   └── engine.py        # UI-independent compositor (render_fov_to_array)
├── export/
│   └── job.py           # Job runner for batch export
└── viewer/
    ├── main_viewer.py   # ImageMaskViewer — the god object
    ├── ui_components.py
    ├── virtual_map_layer.py
    ├── roi_manager.py
    ├── checkpoint_store.py
    ├── scale_bar.py
    ├── plugin/          # Auto-discovered PluginBase subclasses
    │   ├── export_fovs.py
    │   ├── chart.py
    │   ├── heatmap.py
    │   └── ...
    └── images/          # Bundled UI icons
```

`ueler/rendering/` and `ueler/export/` exist so that batch export never reads a widget: the compositor
and the job runner are importable without a live viewer, which is what makes export testable.

---

## Current Status

- `ueler/` package skeleton, `pyproject.toml`, and `Makefile` are in place.
- The legacy import shims are removed; `import ueler` is side-effect free and asserted so by `tests/test_import_namespace_hygiene.py`.
- All module moves from `viewer.*` → `ueler.viewer.*` are complete.
- `ueler.image_utils` is restored as a real packaged module (post-cleanup regression fix).
- **Gate A of the PyPI release plan is complete.** The build is reproducible and safe to publish: `python -m build` is clean, `twine check --strict` passes on both artifacts, and wheel and sdist have each been installed into a fresh venv and imported from outside the repository.
- **Gate B is complete** (2026-08-10). The release now describes itself: the PyPI classifier block (ten entries then, nine since the 2026-08-20 metadata trim), `[project.urls]` covering repository / issues / changelog, the license stated in the README, the docs-site install page realigned with the PyPI-first flow, and the stale graphify cache moved out of `ueler/`.
- **Gate C is complete** (2026-08-10). Two workflows: `tests.yml` (unit matrix + a build-and-import-the-wheel job) and `release.yml` (Trusted Publishing). See *Continuous integration* and *Release process* below.
- **CI paid for itself on its first run** (2026-08-10): 3.10, 3.11 and `package` green, and the non-blocking 3.12 leg surfaced 19 errors that were a live pandas bug rather than a 3.12 one — mask painting was broken for any AnnData-derived `category` identifier column on *current* pandas too. Suite now **922 tests, 0 skips**.
- **Gate D is under way.** The TestPyPI rehearsal is done — `v0.5.0-alpha2` published `ueler-viewer 0.5.0a2` there from a tag push on 2026-08-19, claiming the project name on that index, and `0.5.0rc1` followed. What remains before a stable tag is the **PyPI** Trusted Publisher and the `pypi` environment reviewer, neither of which exists yet; see the open items below.

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
3. On GitHub → *Settings* → *Environments* → `pypi`, add a **required reviewer**. This
   one is not optional: it is what keeps a human between a stable tag and an
   irreversible upload now that the tag routes itself.
4. Optionally, a **tag ruleset** (*Settings* → *Rules* → *Rulesets*, target *Tags*,
   pattern `v*`) restricting who may create release tags.

After that the tag decides the route:

| What you push | What happens |
|---|---|
| a pre-release tag (`v0.6.0-alpha1`, `v0.6.0-rc1`) | tests → build → verify → upload to **TestPyPI**, unattended |
| a stable tag (`v0.6.0`) | the same, **plus** the rehearsal check against the highest published `rc`, then TestPyPI, then **PyPI** once the `pypi` environment's reviewer approves |

So a stable release is a two-step act. First tag `v0.6.0-rc1` and let it publish; then,
changing nothing but the version declarations and the documentation, tag `v0.6.0`. If
anything that ships in the wheel changed since the candidate, `check_stable_rehearsal.py`
fails the run and asks for `rc2` — that is the mechanism working, not friction to route
around.

`workflow_dispatch` survives as the way to re-drive an upload (ref = **the tag**,
`publish_to: pypi`). It cannot skip anything: the rehearsal check runs in `verify`, which
both paths share. `release.yml` publishes what `tests.yml` built in the same run, so the
uploaded artifact is the tested one.

### Release targets

```shell
make test-ci                          # the suite with no skips tolerated
make build                            # clean dist/ first, then build sdist + wheel
make check-dist                       # twine check --strict
make check-release TAG=v0.5.0-alpha   # tag == pyproject == __version__ == dist/
make check-rehearsal TAG=v0.6.0       # a stable tag must promote a published rc
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
- ~~Rehearse on TestPyPI before the first real upload~~ — **done** (2026-08-19). `v0.5.0-alpha2` published `ueler-viewer 0.5.0a2` to TestPyPI from a tag push, which also created and claimed the project there. The rehearsal is no longer a manual habit: `check_stable_rehearsal.py` now makes it a precondition of every stable release.
- Configure the **PyPI** Trusted Publisher and the `pypi` environment reviewer. `https://pypi.org/simple/ueler-viewer/` still returns 404, so the project does not exist on PyPI and the publisher must be a *pending* one. Both are preconditions of the first stable tag: without the publisher the upload fails, and without the reviewer nothing stands between a stable tag and an irreversible upload.
- **Finish acting on the 3.12 result — the widening is half-applied.** The leg's first reading was not about 3.12 at all: 19 errors, all one `np.issubdtype` call meeting pandas 3's default string dtype, now fixed. Tightening `requires-python` to `<3.12` would have concealed it, since pandas 3 installs on 3.11 too, so the evidence argued for widening instead. `Programming Language :: Python :: 3.12` **has** been added; the 3.12 leg in `tests.yml` is **still** `experimental: true` / `continue-on-error`, and the comment above it still reads as though the classifier were pending. So UELer currently advertises 3.12 support that no blocking CI leg defends. Either drop `experimental` (and that stale comment) or drop the classifier — the two must say the same thing. Keep the leg either way: it is the only coverage of pandas-3 semantics.
- Revisit the **3.10 floor** before it costs coverage: pandas 3 and anndata 0.12 both require ≥ 3.11, and anndata 0.13 requires ≥ 3.12, so a 3.10 install is pinned to the older half of the stack. Nothing is broken today — pandas 3 cannot be installed on 3.10 at all, so the bad pandas-3-with-anndata-0.11 pairing is unreachable.
- Confirm with DKFZ that naming both the author and the institute in the BSD copyright line matches institutional policy — the only part of the relicense that is not purely a code change.
- Revisit `ipykernel` / `ipympl` as hard runtime dependencies before `1.0`: `pip install ueler-viewer` currently installs a Jupyter kernel. Moving them to a `notebook` extra also requires updating `.binder/postBuild`, which runs a bare `pip install .`.

---

## Related Issues

- [#79 — Package UELer as a pip package](https://github.com/HartmannLab/UELer/issues/79)
- [#4 — Packaging plan](https://github.com/HartmannLab/UELer/issues/4)
