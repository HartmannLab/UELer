# Installation

There are two ways to install UELer. Pick the first unless you intend to modify UELer itself.

---

## Requirements

- **Python** 3.10 or 3.11
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (recommended) or conda/mamba — needed for the development install, and still the easiest way to get the binary stack (HDF5, OpenCV) on an HPC system
- Git — development install only

---

## Option A — Install from PyPI

```shell
pip install ueler
```

While UELer is on a pre-release version, ask pip for it explicitly:

```shell
pip install --pre ueler
```

That pulls in every runtime dependency. Two optional extras are available:

```shell
pip install "ueler[ark]"    # adds ark-analysis (pinned) for ark-based workflows
pip install "ueler[docs]"   # adds the mkdocs toolchain for building these docs
```

To upgrade later:

```shell
pip install --upgrade ueler
```

The starter notebook is not part of the package — download
[`script/run_ueler.ipynb`](https://github.com/HartmannLab/UELer/blob/main/script/run_ueler.ipynb)
from the repository, or write your own cell calling `ueler.runner.run_viewer`.

---

## Option B — Install from Source

Use this if you want to modify UELer, run the test suite, or track the `develop` branch. The
remaining steps on this page describe that path.

## Step 1 — Set Up the Environment

The easiest way to create a compatible environment is using the provided `environment.yml` file.

=== "micromamba (recommended)"

    ```shell
    # Download the environment file from the repository
    # Then create the environment:
    micromamba env create --name ark-analysis-ueler --file environment.yml
    ```

=== "conda"

    ```shell
    conda env create --name ark-analysis-ueler --file environment.yml
    ```

This installs all required packages, including `ark-analysis`, `ipywidgets`, `jupyter-scatter`, `dask`, and other dependencies.

---

## Step 2 — Clone the Repository

Navigate to the directory where you want to install UELer and clone the repository:

```shell
git clone https://github.com/HartmannLab/UELer.git
cd UELer
```

---

## Step 3 — Activate the Environment

```shell
micromamba activate ark-analysis-ueler
```

---

## Step 4 — Install UELer

Install the package in editable mode so that you can update it with `git pull` without reinstalling:

```shell
pip install -e .
```

---

## Updating UELer

For a source install, navigate to your UELer directory and pull the latest changes:

```shell
cd <path-to-UELer-folder>
git pull
```

No reinstall is needed when using editable mode, unless `environment.yml` or `pyproject.toml`
changed — then re-run `pip install -e .`. For a PyPI install, use `pip install --upgrade ueler`
instead.

---

## Updating Your Environment

If you are upgrading from **v0.1.7-alpha or earlier**, you need to install additional packages:

```shell
micromamba activate ark-analysis-ueler
micromamba install dask
micromamba install dask-image
```

---

## Installing for Development

If you plan to contribute to UELer or run the test suite, install the `dev` extras:

```shell
pip install -e ".[dev]"
```

This adds `pytest`, `pytest-cov`, `build` and `twine` to your environment.

To run the test suite:

```shell
make test-fast                      # or, equivalently:
python -m unittest discover tests
```

`tests/bootstrap.py` carries lightweight stubs for `pandas`, `matplotlib` and
`ipywidgets`, but each one installs only when the real library is missing — in a complete
`dev` environment they are inert and the suite runs against the real stack in about six
seconds. `make test-fast` also sets `UELER_TEST_BOOTSTRAP=1`, which lets
`sitecustomize.py` install the stubs at interpreter startup. That is **opt-in** — without
the variable a plain interpreter always gets the real libraries, so nothing you run outside
the tests is affected.

To run the suite the way CI does, which fails on the first **skipped** test:

```shell
make test-ci                        # or, equivalently:
python tools/run_test_suite.py --max-skips 0
```

A skipped test means an optional dependency is missing, and plain `unittest` still prints
`OK` in that case — so a partially installed environment can look green while a whole
code path goes untested. This target prints every skip with its reason instead.

---

## Installing MkDocs (for documentation contributors)

To build and preview the documentation locally:

```shell
pip install mkdocs-material
mkdocs serve
```

The documentation site is then available at `http://127.0.0.1:8000`.

---

## Troubleshooting

!!! tip "Widget not rendering in VS Code"
    If the interactive scatter plots are not shown in VS Code, UELer automatically falls back to a static Matplotlib figure. See the [FAQ](faq.md) for details.

!!! tip "ModuleNotFoundError on import"
    Make sure you have activated the correct environment and that `pip install -e .` completed without errors.
