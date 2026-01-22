## Issue #64 — Jupyter-scatter blank in VS Code

### Context
- Scatter plots (jupyter-scatter/anywidget) intermittently render blank in VS Code notebooks (`1.108.0-insider`, Jupyter extension). Other front-ends (classic/JLab) are unaffected.
- Kernel-side construction succeeds (no ImportError), but the widget view sometimes fails to appear; users lose all scatter interactions. Trust/state in VS Code and widget renderer availability seem to be involved.
- Environment installs `jupyter-scatter[all]` via `environment.yml`, so Python deps are present; the failure appears front-end specific.

### Hypothesis
- VS Code’s widget renderer occasionally fails to hydrate the anywidget frontend, leaving the scatter output empty. When the front-end cannot load custom widgets, we should degrade gracefully instead of leaving a blank cell.

### Plan
1) Add a VS Code–aware guard in the Chart plugin to detect when the notebook runs under VS Code (env `VSCODE_PID`) or when the user opts in via `UELER_SCATTER_BACKEND=static`.
2) When the guard triggers, render a static Matplotlib scatter fallback (same data/axes, optional color column) with a banner explaining that interactive scatter was disabled due to missing widget support and how to re-enable (`UELER_SCATTER_BACKEND=widget`).
3) Keep existing jupyter-scatter behavior for all other environments; no change to selection/linkage when the widget path is active.
4) Document the fallback and the override flag in README and log; add the issue link in `dev_note/github_issues.md`.

### Acceptance
- VS Code users always see a scatter plot (interactive when widgets load; static fallback when not).
- Default behavior for JupyterLab/classic remains unchanged.
- Users can force widget mode via `UELER_SCATTER_BACKEND=widget`.