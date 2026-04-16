# Installation

This page covers the full installation of UELer. Follow the steps below to set up a working environment.

---

## Requirements

- **Python** ≥ 3.10
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (recommended) or conda/mamba
- Git

---

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

To update to the latest version, navigate to your UELer directory and pull the latest changes:

```shell
cd <path-to-UELer-folder>
git pull
```

No reinstall is needed when using editable mode.

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

This adds `pytest` and `pytest-cov` to your environment.

To run the test suite:

```shell
python -m unittest discover tests
```

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
