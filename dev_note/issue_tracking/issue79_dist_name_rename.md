# Issue #79 follow-up — the distribution is renamed to `ueler-viewer`, the import name stays `ueler`

**Status:** implemented 2026-08-19 (rename + docs). The automatic tagged release to TestPyPI is unblocked and is the remaining step (§8).
**Parent:** [`issue79_pypi_release_readiness.md`](issue79_pypi_release_readiness.md) — blocker **C4** ("PyPI will not accept the name `ueler`"), Gate D.

**Decision (2026-08-19).** `pyproject.toml`'s `[project] name` becomes **`ueler-viewer`**, used on **both** indexes. The import name remains `ueler`. The TestPyPI `ueler` project stays registered and stops receiving uploads.

An earlier reading of the developer's intent was a *dual* name — `ueler` on TestPyPI, `ueler-viewer` on PyPI — and a plan was written for it. That was set aside once the actual requirement became clear: automatic tagged releases to TestPyPI under `ueler-viewer`. The dual-name analysis is kept in §4 because the reasoning against it is what justifies the current shape.

---

## 1. Why the name is prohibited — the `euler` finding (2026-08-17)

C4 established that `ueler` is *unregistered* on PyPI yet administratively **prohibited**: the pending-publisher form answers "This project name isn't allowed", which is warehouse's response for a name on its prohibited list rather than for a taken one, and the same account registers `ueler-viewer` immediately.

Measured 2026-08-17: **`euler` is a live PyPI project** (1.0.3, 3 releases, "A library for Euler angle computation and conversion"), and it does **not** exist on TestPyPI — which is precisely why TestPyPI accepted `ueler` and PyPI does not.

This explains C4's elimination table rather than contradicting it. The table ruled out "too similar to an existing project" for the **automated** check only, and correctly: warehouse's ultranormalisation folds confusable *characters* (`l`/`I`/`1`, `0`/`O`, `-_.`), never a transposition of letter order, so `ueler` against `euler` cannot trip it. "Explicitly prohibited by the administrators" still holds — and `euler` is the likely *motive* for that manual entry.

**Consequence:** a prohibition that exists to protect a live project is not the kind an admin lifts on request. Treat `ueler-viewer` as the long-lived name and keep the exit ramp (§7) rather than assuming a grant is weeks away.

---

## 2. What changed

- **`pyproject.toml`** — `name = "ueler"` → `name = "ueler-viewer"`, with a comment recording that this is the distribution name only and pointing here.
- **`README.md`** — Option A leads with the install-name/import-name split, uses `ueler-viewer` (and `"ueler-viewer[ark]"` / `"ueler-viewer[docs]"`), collapses the multi-line `pip install` into one line, and gains an "If the install fails" block covering the `scikit-image` error, the `uv` flag and the old-distribution conflict. The upgrade section matches.
- **`docs/installation.md`** — "Option A — Install from PyPI" becomes "Option A — Install from TestPyPI", with an admonition on the name split and two warning admonitions for the same two failure modes. "Updating UELer" no longer claims `pip install --upgrade ueler`.
- **`docs/develop-notes/packaging.md`** — the `pip install` decision now names `ueler-viewer`, plus two new decisions: the rename with its rationale, and the pending-publisher/reservation finding (§5).
- **`doc/log.md`**, **`README.md`** "New Update" — entries under `v0.5.0-alpha2`.

## 3. What did **not** change, and why

The distribution name and the import name are independent: the first is `[project] name`, the second is the `ueler/` package directory.

- **No code.** Every `import ueler` / `from ueler.viewer...` in the package, the tests and `script/run_ueler.ipynb` is untouched.
- **`ueler.__version__`.** Checked specifically, not assumed: it is a literal in `ueler/__init__.py`, and nothing under `ueler/` uses `importlib.metadata` or `pkg_resources`, so no distribution-name lookup existed to break.
- **`tools/check_release_tag.py`.** Its wheel parser splits on `-` and takes index 1; its sdist parser takes `rsplit("-", 1)`. The renamed artifacts normalise to `ueler_viewer-0.5.0a2-py3-none-any.whl` and `ueler_viewer-0.5.0a2.tar.gz`, so both still yield `0.5.0a2`. Confirmed against the real build.
- **`tests/`.** The only name-shaped assertion, `tests/test_import_namespace_hygiene.py`, inspects `spec.origin` paths rather than distribution metadata.
- **`.github/workflows/`, `Makefile`.** Nothing hardcodes the distribution name; the artifacts move as a directory (`dist/`), not by filename.
- **`.binder/postBuild`.** A bare `pip install .`, so it simply builds whatever name `pyproject.toml` declares.
- **`MANIFEST.in`, `[tool.setuptools.packages.find]`.** Both address the *import* package (`ueler`, `ueler.*`).

Precedent for the split: `scikit-image`/`skimage`, `opencv-python`/`cv2`, `pillow`/`PIL`, `beautifulsoup4`/`bs4`.

---

## 4. Rejected alternative — a dual name (`ueler` on TestPyPI, `ueler-viewer` on PyPI)

The distribution name is baked into the artifact filename, `*.dist-info/METADATA`'s `Name:` and `RECORD`, so **one build carries exactly one name**. Publishing different names to the two indexes therefore requires two separate builds, and PEP 621 forbids `dynamic = ["name"]`, so there is no config or environment hook for it — the line has to be rewritten in `pyproject.toml` and put back. That would have meant:

- a `tools/build_dist.py` that rewrites `pyproject.toml`, builds, restores it in a `finally`, and verifies byte-identity afterwards, plus a dirty-file guard so an interrupted run cannot be committed;
- a `--expect-dist-name` check in `tools/check_release_tag.py`, because "the wrong-named wheel reaches PyPI" would become a real and unrecoverable failure mode;
- `Makefile` targets and `clean-dist` handling two output directories;
- `tests.yml` building and validating two artifact sets and uploading two artifacts instead of one;
- `release.yml`'s `verify` checking both sets, with each publish job taking its own;
- a wheel-diff step to prove the two builds differ only in the name.

And it would have cost the thing the rehearsal exists for: the file uploaded to TestPyPI would no longer be the file PyPI receives, so "it installed from TestPyPI" would stop being evidence about the PyPI artifact.

Against that, the only benefit was keeping new uploads appearing under `ueler` on TestPyPI. That is not needed to keep the name: the TestPyPI `ueler` project stays registered with its 0.3.1 and 0.5.0a history, TestPyPI does not reclaim names for inactivity, and PyPI's decision about `ueler` has nothing to do with activity on TestPyPI. Rejected accordingly.

Also rejected, for the record: renaming a built wheel (unsupported — the name lives in `METADATA` and `RECORD`, and `wheel tags` rewrites only compatibility tags), and a second `pyproject.testpypi.toml` (setuptools only ever reads `pyproject.toml`, so it needs the same copy dance plus a file that drifts).

---

## 5. A pending Trusted Publisher does **not** reserve the name

Measured 2026-08-17: the `ueler-viewer` pending publisher had existed on PyPI since 2026-08-10, yet `https://pypi.org/simple/ueler-viewer/` still returned **404**. By the argument C4 used to eliminate "registered by another user with no releases" — a registered-but-empty project would still have a simple-index page — no project object existed, so the name was claimable by anyone. Re-measured 2026-08-19 after the developer configured the TestPyPI publisher: `https://test.pypi.org/simple/ueler-viewer/` also still 404s.

**The name is created and claimed by the first upload.** So the first tagged release doubles as the reservation, on whichever index it reaches — no separate manual upload is needed, and none should be done by hand.

The corollary cuts the other way on PyPI: for as long as the support request is open and nothing has been uploaded, `ueler-viewer` is unheld there. There is no way to hold a PyPI name without publishing to PyPI. The exposure is small (an unremarkable compound name nobody else has reason to want) and the mitigation — an earlier irreversible upload — is worse than the risk, so this is accepted knowingly rather than overlooked.

### Trusted Publisher settings

The form asks for the workflow **filename**, not the workflow's `name:` field (`Release` would be rejected):

| Field | TestPyPI | PyPI |
|---|---|---|
| Project name | `ueler-viewer` | `ueler-viewer` |
| Owner | `HartmannLab` | `HartmannLab` |
| Repository name | `UELer` | `UELer` |
| Workflow name | `release.yml` | `release.yml` |
| Environment name | `testpypi` | `pypi` |

Both are configured. One repository and workflow can back any number of projects, so these coexist with the existing `ueler` publisher on TestPyPI.

---

## 6. Two distributions can own one import package

`ueler` and `ueler-viewer` both install `ueler/`. pip does not know they are the same project, so installing one over the other leaves two `dist-info` directories claiming the same files, and uninstalling either can delete files the other still lists.

This is not hypothetical — the development environment holds an editable `ueler 0.3.0a0`:

```
site-packages/ueler-0.3.0a0.dist-info
site-packages/__editable__.ueler-0.3.0a0.pth
site-packages/__editable___ueler_0_3_0a0_finder.py
```

So a `pip install -e .` after this rename produces exactly that collision. **`pip uninstall ueler` first**, in the development env and in any env where an earlier release was installed. The README and `docs/installation.md` upgrade paths now say so. `Conflicts-Dist` metadata was considered and skipped: pip does not enforce it, so it would document the conflict without preventing it.

---

## 7. Exit ramp if PyPI later grants `ueler`

PyPI cannot rename a project, and a released version cannot be re-uploaded under a different name, so "switching" means publishing a new distribution:

1. Flip `[project] name` back to `ueler`. That is the whole source change — the reward for the name living in exactly one place.
2. Publish a final `ueler-viewer` release that is a **thin shim**: no modules of its own, a single `ueler` dependency pinned at the current version, and a README saying the project moved. The `sklearn` → `scikit-learn` pattern.
3. Point the docs at `ueler`; the shim's PyPI page remains the redirect for anyone who installed the interim name.
4. Do **not** yank the `ueler-viewer` releases — yanking breaks pinned installs.

Bounded, known cost: one shim release plus a docs pass.

---

## 8. Releasing a tagged version automatically

No workflow changes were needed: `release.yml` already uploads to TestPyPI on any `v*` tag push, and reaches PyPI only through a manual `workflow_dispatch` from a tag. What the rename changes is only *which project* receives the upload.

What a tag push does, in order: `tests.yml` runs the full matrix and its `package` job builds the artifacts and uploads them as the `dist` artifact → `verify` re-runs `twine check --strict` and `check_release_tag.py <tag> --require-dist`, so the tag, `pyproject.toml`, `ueler.__version__` and both filenames must describe one release → the `testpypi` job publishes via OIDC with `skip-existing: true`.

Steps for the next release:

1. Merge to `main` — `v0.5.0-alpha1` was tagged on a merge commit there, so match that. The current work is on `nightly`.
2. Confirm the version. `0.5.0-alpha2` is declared and untagged, and it is unused under `ueler-viewer` (a fresh version namespace), so it can be tagged as-is. A bump is the developer's call — use the version-bump skill if the intent is to release something newer than what `doc/log.md` describes.
3. `python tools/check_release_tag.py v0.5.0-alpha2 --no-tag` style pre-check locally, or `make check-release TAG=v0.5.0-alpha2`, then `make build check-dist` to see exactly what would be uploaded.
4. `git tag v0.5.0-alpha2 && git push origin v0.5.0-alpha2` — lightweight, matching every existing tag.
5. Watch the run. The `testpypi` job creates the `ueler-viewer` project on TestPyPI and claims the name (§5).
6. Smoke-test the published release with the one-line command from the README, in a fresh env, and run the starter notebook against it.
7. PyPI stays manual: Actions → *Release* → *Run workflow*, ref = **the tag**, `publish_to: pypi`, then the required reviewer on the `pypi` environment approves. Irreversible — a version can be yanked but never replaced or reused.

---

## 9. Verification performed (2026-08-19)

- `python tools/run_test_suite.py --max-skips 0` → **1073 tests, OK**, "No tests were skipped".
- `python -m build` → `ueler_viewer-0.5.0a2-py3-none-any.whl` and `ueler_viewer-0.5.0a2.tar.gz`, built to a scratch directory so `dist/` was left alone.
- `twine check --strict` → both PASSED.
- Wheel metadata → `Name: ueler-viewer`, `top_level.txt` → `ueler`. The split is real, not assumed.
- `check_release_tag.py --no-tag --require-dist --dist <scratch>` → all four declarations agree on `0.5.0a2`.
- Wheel installed into a fresh venv and imported from outside the repository: `import ueler`, `import ueler.viewer.main_viewer`, and the packaged `ready.png` asset all fine; `importlib.metadata.version("ueler-viewer")` → `0.5.0a2`.

## 10. Not done here

- **The tag push itself** (§8) — the developer's call, and it is the step that claims the name.
- **`dev_note/github_issues.md`** — follow-up entry under the #79 section, written with the commit.
- **Rebuild the graphify graph** if anything under `ueler/` changes later; nothing did here.
