# Developer tools

## `scan_local_info.py`

Scans the codebase for **local / machine-specific information that should not be
shipped or committed**: hardcoded secrets, absolute machine paths, personal or
institution-internal identifiers (emails, usernames), and hardcoded network
endpoints. Home-relative (`~/`) defaults and developer notes (`TODO`, `FIXME`)
are reported as informational only.

Stdlib-only, no dependencies.

By default only **git-tracked** files are scanned — untracked and gitignored
files can't be committed or shipped, so they'd only add noise. Pass
`--include-untracked` to scan every text file on disk instead. Outside a git
repository the scan falls back to all files (with a note).

### Two scopes

| Scope     | What it scans                              | Why                                   |
| --------- | ------------------------------------------ | ------------------------------------- |
| `package` | the `ueler/` package + subpackages         | exactly what is built into the wheel  |
| `project` | the entire repository working tree         | catches leaks in docs, notes, configs |

### Usage

```bash
python tools/scan_local_info.py                    # both scopes
python tools/scan_local_info.py --scope package    # what actually ships
python tools/scan_local_info.py --scope project    # whole repo
python tools/scan_local_info.py --min-severity medium
python tools/scan_local_info.py --include-untracked   # scan untracked files too
python tools/scan_local_info.py --root /path/to/repo --no-color

# or via Makefile
make scan            # both
make scan-package
make scan-project
```

### Severities & exit code

* **HIGH** — hardcoded secrets, absolute machine paths (e.g. `/omics/...`, `C:\...`).
* **MEDIUM** — emails, personal/institution identifiers, network literals.
* **LOW** — `~/` defaults, `TODO`/`FIXME`/`@copilot` notes.

Exit code is **1** if any HIGH finding exists (0 otherwise), so it can gate a
release or run in CI. Only HIGH gates the exit code; MEDIUM/LOW are advisory.

### Suppressing a known-benign line

Append an inline comment:

```python
BASE = "/data/example"   # noqa: scan            # suppress all rules on this line
EMAIL = "a@b.com"        # noqa: scan:email-address   # suppress one rule
```
