# `Unable to find widget 'anywidget' version '~0.9.*' from configured widget sources ["local"]`

User report, no reproduction supplied. Same reporter and, most likely, the same shared HPC environment as the `ipympl` crash in `ipympl_none_manager_crash.md`.

---

## Problem

The message is **not** UELer's, and not Python's. It is emitted by the **VS Code Jupyter extension's** widget manager when it cannot resolve the browser-side JavaScript for a widget model the kernel has just created. Its shape is `Unable to find widget '<module>' version '<range>' from configured widget sources <list>`, where `<list>` is the value of the VS Code setting `jupyter.widgetScriptSources`. The reporter's list is `["local"]`, i.e. every CDN source is disabled and only on-disk lookup is left.

Three facts fix the diagnosis:

- **`anywidget` is load-bearing for UELer, not optional decoration.** Six plugins render their entire UI as an `AnyWidget` subclass: the channel picker (`channel_picker_widget.py`), the ROI expression editor (`roi_expression_editor.py`), the Mask Painter class list (`mask_class_list_widget.py`), the gallery tile grid (`tile_gallery_widget.py`), the cell-annotation checkpoint tree (`cell_annotation.py`), and the scatter plot via `jupyter-scatter`. When the frontend asset is unresolvable these render as empty boxes.
- **The `~0.9.*` in the message is the kernel's `anywidget`, and it is behaving correctly.** `anywidget.AnyWidget._model_module_version` is `~0.9.*` on an 0.9.x install (`~0.11.*` on 0.11, which is what this dev environment has). So the Python side asked for exactly what it should; the failure is on the lookup side.
- **VS Code's "local" source searches one directory, and it is not the one Jupyter searches.** Confirmed against `localWidgetScriptSourceProvider.node.ts` upstream: *"Widget scripts are found in `<python folder>/share/jupyter/nbextensions`"* — the **kernel interpreter's `sys.prefix`**, and nothing else. JupyterLab, by contrast, reads federated extensions from the whole `jupyter_path()` chain, which also includes `~/.local/share/jupyter` and `~/Library/Jupyter`.

That last asymmetry is the trap, and it fits a shared read-only HPC environment exactly. `pip install --user ueler-viewer` (or `pip install --user anywidget`) puts the package on `sys.path` but puts its data files under `~/.local/share/jupyter/{lab,nb}extensions/anywidget/`. **JupyterLab then works and VS Code does not**, from one identical environment — which is why a report like this arrives with no reproduction.

Verified that the asset layout itself is not the variable: both the `anywidget` 0.9.13 and 0.11.0 wheels ship `share/jupyter/labextensions/anywidget/` *and* `share/jupyter/nbextensions/anywidget/{extension,index}.js`. So whenever `anywidget` is installed into the same prefix as the kernel, VS Code's local source finds it. The failure is always about *where*, or about a version mismatch between two installs.

### Why this is worth code, not only a docs note

Nothing in UELer can repair VS Code's resolver, and nothing should try. But the failure is silent on the Python side and near-undiagnosable on the user's side: the widgets come up blank, the kernel logs nothing, and the only clue is a transient VS Code notification that names `anywidget` — a package the user never installed deliberately and has no reason to connect to UELer. Meanwhile every fact needed to diagnose it is available in-process, cheaply, at import time.

`channel_picker_widget.ANYWIDGET_AVAILABLE` already exists but answers a different question — *can Python import it* — and is `True` in exactly this failure.

---

## Approach

A startup self-check, plus documentation. No change to how any widget is built.

**1. `ueler/viewer/widget_frontend_check.py`** — a pure, cheap, never-raising function `check_anywidget_frontend()` returning a `FrontendStatus`. It compares what the kernel will *ask* the browser for against what is discoverable on disk:

- the required range, from `AnyWidget._model_module_version`;
- every `anywidget` labextension on `jupyter_path("labextensions")`, with the version from its `package.json`;
- every `anywidget` nbextension on `jupyter_path("nbextensions")` (no `package.json` there — presence is the signal, and it ships from the same wheel as the Python package, so its version is the Python one).

Codes, each with a different remedy:

| code | condition | who is broken |
| --- | --- | --- |
| `ok` | nbextension under `sys.prefix`, version satisfies the range | nobody |
| `absent` | `anywidget` not importable, or the test bootstrap's stub | nobody — the plugins' fallbacks handle it |
| `missing-assets` | importable, but no asset anywhere on the search path | every frontend |
| `outside-env` | assets exist, none under `sys.prefix` | VS Code only (the `--user` case) |
| `version-skew` | asset under `sys.prefix`, version outside the range | every frontend |
| `undetermined` | the check could not run | nobody |

**The rule that governs every ambiguous case: stay silent.** A false alarm on a healthy JupyterLab install is worse than the original problem, so anything unparseable or unexpected returns a non-warning code.

**2. Call it from `ImageMaskViewer.__init__`**, beside `install_canvas_message_guard()`, emitting a `logging.warning` through the `ueler` logger only for the three real-problem codes. The message names the condition, the paths actually found, and both remedies (install into the kernel's environment; or add CDN entries to `jupyter.widgetScriptSources`).

**3. Docs** — a troubleshooting entry in `docs/installation.md` and `docs/faq.md` covering the message verbatim, so a search for it lands somewhere useful.

### Semver matching

`~0.9.*` is matched by taking the leading numeric components before the `*` and requiring the found version to share that prefix — `~0.9.*` admits `0.9.13` and rejects `0.11.0`. An unrecognised range shape is treated as "satisfied" rather than guessed at, per the silence rule.

---

## Implementation steps

1. Add `ueler/viewer/widget_frontend_check.py` with `FrontendStatus`, `check_anywidget_frontend()` and `warn_about_widget_frontend()`.
2. Call `warn_about_widget_frontend()` from `ImageMaskViewer.__init__`.
3. Add `tests/test_widget_frontend_check.py`: one test per code against a synthetic `jupyter_path`, the silence rule, the semver matcher, and that the viewer calls it.
4. Document in `docs/installation.md` and `docs/faq.md`; update `doc/log.md` and `README.md`.

---

## Out of scope

- **Making the widgets degrade to non-anywidget fallbacks when the frontend is unresolvable.** The existing `ANYWIDGET_AVAILABLE` fallbacks key off a Python import, and Python cannot know whether the browser resolved the asset — the kernel gets no negative acknowledgement. Doing this properly needs a round trip to the frontend, which is a much larger change than the problem warrants.
- **Vendoring or pinning `anywidget`'s frontend.** UELer declares `anywidget>=0.9` and must keep tracking whatever `jupyter-scatter` resolves.
