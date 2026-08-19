# Issue #79 follow-up — route pre-release tags to TestPyPI and stable tags to PyPI automatically

**Status:** **implemented** (2026-08-19) — `tools/release_channel.py`, `tools/check_stable_rehearsal.py`, the rewritten `.github/workflows/release.yml`, `make check-rehearsal`, and 29 tests across `tests/test_release_channel.py` and `tests/test_stable_rehearsal.py`. The §7 rehearsal guard is in, with "identical" defined as §7.2 specifies. **Two preconditions remain manual and are not done:** the required reviewer on the `pypi` environment, and the pending Trusted Publisher on PyPI (§4 step 4). Until both exist, a stable tag must not be pushed — the first would fail at the upload, and nothing would stand between a mistyped tag and an irreversible release. Related: [`issue79_pypi_release_readiness.md`](issue79_pypi_release_readiness.md), [`issue79_dist_name_rename.md`](issue79_dist_name_rename.md), [`../../docs/develop-notes/packaging.md`](../../docs/develop-notes/packaging.md).

---

## 1. The request

> Do you think it is possible to push all the pre-release to the test-PyPI and the stable release to PyPI automatically?

Yes, and cleanly — the tag already carries everything needed to decide. The routing rule is one predicate, `packaging.version.Version(tag).is_prerelease`.

The table below runs that predicate over the existing tags purely as **test data** — a free corpus of real spellings to confirm the predicate reads them the way a human would. No past release is revisited, re-uploaded or re-validated; the routing applies to the next tag pushed and every one after it.

| tag | PEP 440 | `is_prerelease` | target index |
| --- | --- | --- | --- |
| `v0.1.10` | `0.1.10` | False | PyPI |
| `v0.1.10-rc2` | `0.1.10rc2` | True | TestPyPI |
| `v0.2.0` | `0.2.0` | False | PyPI |
| `v0.2.0-alpha` | `0.2.0a0` | True | TestPyPI |
| `v0.2.0-rc1` | `0.2.0rc1` | True | TestPyPI |
| `v0.4.0` | `0.4.0` | False | PyPI |
| `v0.4.1` | `0.4.1` | False | PyPI |
| `v0.5.0-alpha1` | `0.5.0a1` | True | TestPyPI |
| `v0.5.0-alpha2` | `0.5.0a2` | True | TestPyPI |

The repository's SemVer tag spelling (`-alpha`, `-alphaN`, `-rcN`) normalises to PEP 440 pre-release segments (`a0`, `a1`, `rc1`) with no special-casing, which is the same normalisation `tools/check_release_tag.py` already relies on. Hypothetical future spellings behave sensibly too: `v0.5.0-beta1` → `0.5.0b1` (pre-release), `v0.5.0-dev1` → `0.5.0.dev1` (pre-release, `is_devrelease` as well), `v1.0.0.post1` → `1.0.0.post1` (**not** a pre-release, so a post-release of a stable version correctly goes to PyPI).

So the mechanism is trivial. The part that deserves care is the guardrail this change removes.

---

## 2. What this changes about the safety model

[`.github/workflows/release.yml`](../../.github/workflows/release.yml) is deliberately asymmetric today, and says so in its header comment:

> There is no path where a `git push --tags` reaches the real index by itself.

That property exists because **PyPI is append-only**: a version can be yanked but never replaced or reused. A mistyped stable tag — `v0.5.0` when `v0.5.0-alpha3` was meant — currently costs nothing, because reaching PyPI needs a separate deliberate `workflow_dispatch`. After this change, that same typo burns the version number permanently.

The plan therefore does not just delete the manual step; it **moves the human gate from "run a second workflow by hand" to "approve a job that is already waiting"**. That keeps one irreversible-action confirmation while making the release one click instead of a form:

* **Layer 1 — required reviewer on the `pypi` GitHub Environment.** The `pypi` job already declares `environment: pypi`. Adding a required reviewer there makes the run pause with the artifacts built and verified, and upload only on approval. A run can sit waiting up to 30 days. **This is the replacement for the removed `workflow_dispatch` gate and must be configured before the workflow change is merged.** If it is not, the first stray stable tag ships.
* **Layer 2 — a tag ruleset (optional, cheap).** Settings → Rules → Rulesets, target *Tags*, pattern `v*`, restrict creation to maintainers. Stops an accidental tag push from a collaborator ever entering the pipeline.
* **Layer 3 — the existing `verify` job, unchanged.** `tools/check_release_tag.py <tag> --require-dist` still fails the run unless the tag, `pyproject.toml`, `ueler.__version__` and both artifact filenames describe one release. Layer 1 protects against a *wrong-but-consistent* version; layer 3 protects against an inconsistent one.
* **Layer 4 — no `skip-existing` on the PyPI job.** Already the case (only `testpypi` sets it). A duplicate PyPI upload should fail loudly rather than pass quietly.
* **Layer 5 — a stable tag must be the promotion of a rehearsed release candidate.** Specified in full in §7. This is the layer that makes the automation *safer* than the current manual dispatch rather than merely as safe: it is a machine-checkable statement that the exact code about to reach PyPI has already been installed from an index.

**Decision to make:** whether layer 1 stays on permanently, or only for the first few stable releases. Recommendation: keep it on. It costs one click and it is the only thing standing between a typo and an unusable version number.

---

## 3. Proposed job graph

```
route ─┐
       ├─> testpypi                 (every v* tag)
tests ─┴─> verify ─┬─> testpypi
                   └─> pypi         (stable v* tags only, after testpypi, gated by the pypi environment)
```

Two deliberate choices:

* **Every tag still uploads to TestPyPI**, stable ones included. The routing decision is only about whether `pypi` *additionally* runs. This keeps the current behaviour intact, keeps TestPyPI a complete mirror of release history, and makes the TestPyPI upload a free pre-flight for every stable release — the last thing that can fail before the irreversible one.
* **`pypi` gets `needs: testpypi`.** If TestPyPI rejects the artifacts, PyPI never sees them. The failure direction is the safe one, and `skip-existing: true` on the TestPyPI leg means re-running the job after a transient failure is harmless.

`route` has no `needs`, so a malformed tag fails in ~20 seconds instead of after the full test matrix.

---

## 4. Implementation steps

### Step 1 — `tools/release_channel.py` (new)

A small script that decides the channel, so the safety-critical predicate is testable rather than buried in a shell one-liner inside YAML. It reads the GitHub context from the environment and writes `key=value` lines suitable for `$GITHUB_OUTPUT`:

* inputs: `EVENT` (`push` / `workflow_dispatch`), `REF_TYPE`, `TAG` (`github.ref_name`), `CHOICE` (`inputs.publish_to`, empty on a push)
* outputs: `channel=testpypi` | `channel=pypi` | `channel=none`, plus `version=<normalised>` and `prerelease=true|false` for the step summary
* rules:
  * `EVENT=push`, `REF_TYPE=tag` → strip a leading `v`, parse with `packaging.version.Version`; pre-release → `testpypi`, otherwise `pypi`
  * `EVENT=workflow_dispatch` → honour `CHOICE` verbatim, but refuse `pypi` unless `REF_TYPE=tag` (the existing rule, preserved)
  * **hard failures** (exit non-zero, so the run stops before `tests` finishes): the tag is not a valid PEP 440 version; the tag carries a local version segment (`+something`), which PyPI rejects anyway and which should never be tagged
* also appends a one-line verdict to `$GITHUB_STEP_SUMMARY` so the run page states which index it is heading for before anything is uploaded

Mirrors the style of `tools/check_release_tag.py`: stdlib plus `packaging`, module docstring explaining *why*, `SystemExit` with an actionable message.

### Step 2 — `.github/workflows/release.yml`

1. Rewrite the header comment. The current text promises an asymmetry that will no longer hold; replacing it is part of the change, not a follow-up. New wording states the actual contract: *a pre-release tag reaches TestPyPI unattended; a stable tag reaches PyPI only after the `pypi` environment's reviewer approves.*
2. Add the `route` job (no `needs`, `outputs.channel`), running `tools/release_channel.py`.
3. `testpypi`: `needs: [route, verify]`, condition becomes `route.outputs.channel != 'none'` for pushes plus the existing dispatch choice — in practice `github.event_name == 'push' || inputs.publish_to == 'testpypi'`, i.e. unchanged.
4. `pypi`: `needs: [route, verify, testpypi]`, condition becomes
   ```
   (github.event_name == 'push' && needs.route.outputs.channel == 'pypi')
   || (github.event_name == 'workflow_dispatch' && inputs.publish_to == 'pypi' && github.ref_type == 'tag')
   ```
   The manual path is kept, not replaced — it is still the way to re-drive an upload after a failure, or to publish a tag that predates this change.
5. Leave `tests`, `verify`, `concurrency` and both `permissions` blocks alone.

### Step 2b — `tools/check_stable_rehearsal.py` (new) and its wiring

The §7 guard. Runs as a step in `verify`, so `verify` gains `needs: [tests, route]` and the step is conditional on `needs.route.outputs.channel == 'pypi'`. The `actions/checkout@v4` in `verify` needs `fetch-depth: 0` — the default shallow, single-ref checkout cannot see the rc tag or diff two trees.

### Step 3 — `tests/test_release_channel.py` (new)

A table-driven `unittest` over the nine historical tags plus the interesting synthetic ones (`v1.0.0.post1` → pypi, `v0.5.0-dev1` → testpypi, `v0.5.0+local` → error, `vnonsense` → error, dispatch with `publish_to=pypi` on a branch → error). There is currently no test for `tools/check_release_tag.py`, so this is new ground for the repo — justified here because the predicate's failure mode is an irreversible upload, and because it is the cheapest possible place to catch a regression in it.

### Step 4 — GitHub settings (manual, outside the repository)

Cannot be scripted from here — `gh` is not installed in this environment, so each must be confirmed in the web UI:

1. **`pypi` environment → required reviewers.** Settings → Environments → `pypi`. See §2 layer 1.
2. **Trusted Publisher for PyPI.** `ueler-viewer` does not exist on PyPI yet, so this must be a **pending** publisher: PyPI → Your account → Publishing → owner `HartmannLab`, repository `UELer`, workflow `release.yml`, environment `pypi`. Note the finding recorded in `issue79_dist_name_rename.md` §5: a pending publisher does **not** reserve the name — the first upload creates the project. Verify this exists before the first stable tag, or the run fails at the upload step.
3. **Tag ruleset** (optional). See §2 layer 2.

### Step 5 — documentation

* `docs/develop-notes/packaging.md` — the decision list still records the manual-only-to-PyPI rule; append the new decision with the reasoning from §2 rather than editing the old entry.
* `README.md` and `docs/installation.md` — currently "Option A — Install from TestPyPI", written when TestPyPI was the only index carrying the project. Once a stable version lands on PyPI, the primary instruction becomes `pip install ueler-viewer` from PyPI, with TestPyPI demoted to "installing a pre-release". **This documentation flip belongs with the first stable release, not with this workflow change** — writing "install from PyPI" while PyPI has no `ueler-viewer` would be wrong until the tag is pushed.
* `doc/log.md` — entry at the top of the current version section.
* `Makefile` — no change needed. `publish-test` / `publish` are the local escape hatches and stay useful when Actions is unavailable.

---

## 5. What this does not solve

* **The first stable release is still a decision, not an automation.** This plan makes `v0.6.0` reach PyPI without a second workflow form; it does not decide when `0.5.0-alphaN` becomes `0.5.0`. Version bumps remain the developer's call via the version-bump skill.
* **`ueler` vs `ueler-viewer`.** Unchanged by this plan. If PyPI later grants `ueler`, the exit ramp in `issue79_dist_name_rename.md` §7 applies and the routing logic is unaffected — it never looks at the distribution name.
* **Yanking.** Nothing here helps after a bad stable upload; that is still a manual yank on PyPI plus a new version. Which is precisely why §2 layer 1 is not optional.

---

## 6. Estimated size

| file | change |
| --- | --- |
| `tools/release_channel.py` | new, ~90 lines with docstring |
| `tools/check_stable_rehearsal.py` | new, ~140 lines with docstring (§7) |
| `.github/workflows/release.yml` | ~45 lines changed (one new job, two conditions, one new step, `fetch-depth: 0`, header comment) |
| `tests/test_release_channel.py` | new, ~70 lines |
| `tests/test_stable_rehearsal.py` | new, ~90 lines (git fixture repository) |
| `docs/develop-notes/packaging.md` | one appended decision |
| `doc/log.md` | one entry |
| GitHub settings | 2 required, 1 optional, all manual |

No production code under `ueler/` is touched, so the risk surface is the release pipeline only, and it is verifiable before any tag exists: `workflow_dispatch` with `publish_to: none` exercises `route` → `tests` → `verify` and uploads nothing.

---

## 7. The rehearsal guard — a stable tag requires a matching, published release candidate

**Accepted.** The rule: `vX.Y.Z` may only reach PyPI if a release-candidate tag for the same `X.Y.Z` exists, is the candidate the stable tag was promoted from, and was actually published to TestPyPI. The rationale is that the rc leg is not merely a version number — it is the only point at which the artifact is installed from a real index by a real resolver, so a stable release that is a pure promotion of a published rc inherits that evidence.

### 7.1 "Identical" cannot mean byte-for-byte, and cannot mean the same commit

The rc and the stable release **differ by construction**, in exactly one respect: the version string. `tools/check_release_tag.py` enforces that the tag, `pyproject.toml [project] version`, `ueler.__version__` and both artifact filenames all describe one release. At the rc commit those read `0.5.0rc1`; at the stable commit they must all read `0.5.0`. So:

* the artifacts cannot be byte-identical — the wheel filename, `METADATA`'s `Version:`, and `ueler/__init__.py`'s `__version__` all change;
* the two tags cannot point at the same commit either, since one commit cannot declare both versions.

The guard therefore has to be defined as **"identical apart from the version declarations"**, with the version-bearing lines named explicitly rather than hand-waved. Any implementation that compares whole files or whole trees will fail on every correct release, so this is the part to get precisely right.

### 7.2 The precise definition

Scope the comparison to **everything that lands in the wheel**, which is two paths, each with one named exception:

| path | must be unchanged between the highest rc and the stable tag | exception |
| --- | --- | --- |
| `ueler/**` | yes | `ueler/__init__.py`, whose diff may contain **only** its `__version__` line |
| `pyproject.toml` | yes | its `version` line only |

Mechanically: `git diff --name-only <rc> <stable> -- ueler/ pyproject.toml` must yield nothing beyond those two files, and for each of them every added/removed line in the diff must match `^\s*(__version__|version)\s*=` with a value that normalises to the rc version on the old side and the stable version on the new side. That last clause matters — it is what stops "only the version line changed" from being satisfied by a version line changed to something *else*.

Freezing `pyproject.toml` is as important as freezing the code: a dependency floor edited between rc and stable changes the wheel's metadata and invalidates the rehearsal even though no Python source moved.

Everything outside those two paths is deliberately free — `doc/log.md`, `README.md`, `docs/**`, `dev_note/**` and `tests/**` are all expected to change on the way to a stable release and none of them affect what a user installs.

This scoping makes the guard fit the existing tooling rather than fight it: the version-bump skill's rc → stable bump touches `pyproject.toml`'s version line, `ueler/__init__.py`'s `__version__`, and documentation. That is exactly the allowed set.

### 7.3 Which release candidate

Among all tags, select those whose PEP 440 `release` tuple equals the stable tag's and whose `pre` segment is an `rc` — so `0.5.0a2` does **not** qualify as a rehearsal; an alpha is a preview, an rc is a candidate. Take the **highest** by PEP 440 ordering and require *that* one to match. Accepting merely *some* rc would let `rc1` vouch for a release that `rc2` changed.

Tag spelling is normalised, not matched textually, so `v0.5.0.rc1`, `v0.5.0-rc1`, `v0.5.0rc1` and even `v0.5.0-c1` all resolve to `0.5.0rc1` — verified. Existing tags here use the hyphen form (`v0.2.0-rc1`, `v0.1.10-rc2`); staying with `-rcN` keeps the tag list uniform, but the guard accepts either.

### 7.4 Proof that the candidate was actually published

A tag existing is not evidence that the TestPyPI upload succeeded — the run could have failed at the publish step, or the tag could predate the workflow. So the guard also asks the index:

```
GET https://test.pypi.org/pypi/ueler-viewer/json   ->  releases{}
```

and requires the normalised rc version to be a key in `releases`. One cheap HTTPS GET, fails closed, and it is what turns the rule from a tagging ritual into a statement about an artifact that exists. Confirmed against the live index: that endpoint currently reports `['0.5.0a2']` for `ueler-viewer`, i.e. the `v0.5.0-alpha2` tag push did publish successfully.

### 7.5 Ancestry: a note, not a failure

`git merge-base --is-ancestor <rc> <stable>` looks like the natural extra check, and under the practice §7.6 establishes it would almost always hold — the release commit sits directly on top of the candidate. It is still not worth enforcing. The content comparison of §7.2 is strictly stronger for this purpose: it proves the shipped bytes match, which is the actual question. Ancestry, by contrast, can fail for reasons that say nothing about the artifact — a rebase, a merge topology where the candidate and the release sit on lines joined later, a shallow fetch. As a gate it would add a failure mode without adding evidence, so it prints as a note: worth seeing, not worth blocking on.

### 7.6 What this changes about release practice

The guard is a commitment about how releases are cut from here on. Nothing already released is revisited or re-validated — the rule applies to the next stable tag and every one after it.

* **After the final rc, no shipped file may change.** Any fix, however small, needs a new rc tag and a new TestPyPI publish. Expect `rc2`, `rc3` — that is the mechanism working, not friction to route around.
* **The stable release commit is a version bump plus documentation, nothing else.**
* **The current `0.5.0` line has alphas only**, so shipping `0.5.0` will require pushing `v0.5.0-rc1` and letting it publish first. The first PyPI upload of `ueler-viewer` — which also creates the project — will therefore always be preceded by a rehearsed candidate.

### 7.7 Escape hatch

The guard runs in `verify`, so it covers the tag-push path and the manual `workflow_dispatch` path alike — a dispatch cannot sidestep it. No bypass input is proposed: the legitimate escape is to push an rc and wait a few minutes, cheap enough that a `force` knob would only ever be used in the situation it exists to prevent. If one is added later it must default to false and stay behind the `pypi` environment reviewer.

### 7.8 Failure messages

Each rejection should name its fix, in the style of `check_release_tag.py`'s failure text:

* no rc at all → *"no release candidate found for 0.5.0; tag and publish v0.5.0-rc1 before releasing v0.5.0"*
* highest rc not on TestPyPI → *"0.5.0rc1 is tagged but absent from TestPyPI; its release run must have failed"*
* content mismatch → the offending path list, plus *"these ship in the wheel and changed after v0.5.0-rc1; tag v0.5.0-rc2 and rehearse again"*
* a version-bearing file changed beyond its version line → the offending hunk, since that is the case most likely to be an accident
