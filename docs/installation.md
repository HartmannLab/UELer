# Installation

There are two ways to install UELer. Pick the first unless you intend to modify UELer itself.

---

## Requirements

- **Python** 3.10 or 3.11
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (recommended) or conda/mamba — needed for the development install, and still the easiest way to get the binary stack (HDF5, OpenCV) on an HPC system
- Git — development install only

---

## Option A — Install from TestPyPI

!!! info "The install name is `ueler-viewer`, the import name is `ueler`"

    PyPI administratively prohibits the project name `ueler`, so the distribution is published as
    **`ueler-viewer`**. Only `pip install` is affected — `import ueler` is unchanged, and so is every
    API and notebook. This is the same split as `scikit-image`/`skimage` and `opencv-python`/`cv2`.

While UELer is in pre-release, the releases live on **TestPyPI** rather than PyPI. TestPyPI does not
mirror UELer's runtime dependencies, so the install needs a second index to resolve them — keep the
command on one line:

```shell
pip install --pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ueler-viewer
```

`--pre` is required while the version is a pre-release. That pulls in every runtime dependency. Two
optional extras are available:

```shell
# adds ark-analysis (pinned) for ark-based workflows
pip install --pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "ueler-viewer[ark]"
# adds the mkdocs toolchain for building these docs
pip install --pre --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "ueler-viewer[docs]"
```

To upgrade later, add `--upgrade` to the same command.

!!! warning "`No matching distribution found for scikit-image>=0.19`"

    This means the resolver only consulted TestPyPI, which hosts an empty `scikit-image` project, so
    it found no candidates for the first dependency in the list. Two causes:

    - **`--extra-index-url https://pypi.org/simple/` is missing.** It is what allows the
      dependencies to come from real PyPI, and it is easily lost when a multi-line command is pasted.
    - **You are using `uv`.** Its default `--index-strategy first-index` stops at the first index
      that lists a package at all and never falls back to PyPI:

      ```shell
      uv pip install --prerelease=allow --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match ueler-viewer
      ```

!!! warning "Upgrading from a release installed as `ueler`"

    Run `pip uninstall ueler` before installing `ueler-viewer`. Both distributions install the same
    `ueler/` package and pip does not know they are the same project, so keeping both leaves two
    installs claiming the same files.

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
changed — then re-run `pip install -e .`. For an index install, re-run the
[Option A](#option-a-install-from-testpypi) command with `--upgrade` added.

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

!!! tip "Widget not rendering"
    The interactive scatter is the default in every environment, VS Code included. If widgets do not render at all, check that `%matplotlib widget` ran in the kernel; a static Matplotlib scatter is available as an opt-in fallback. See the [FAQ](faq.md) for details.

!!! tip "ModuleNotFoundError on import"
    Make sure you have activated the correct environment and that `pip install -e .` completed without errors.
