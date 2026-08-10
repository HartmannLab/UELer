# Default developer targets for the UELer packaging skeleton

.PHONY: help venv install test-fast test-integration scan scan-package scan-project docs docs-serve clean

VENV ?= .venv
BIN_DIR := $(if $(filter Windows_NT,$(OS)),Scripts,bin)
PYTHON := $(VENV)/$(BIN_DIR)/python
PIP := $(VENV)/$(BIN_DIR)/pip

help:
	@echo "Available targets:"
	@echo "  make venv              # create a virtual environment in $(VENV)"
	@echo "  make install           # install UELer in editable mode"
	@echo "  make test-fast         # run fast stubbed unit tests"
	@echo "  make test-integration  # placeholder for integration suite"
	@echo "  make scan              # scan pkg + project for local/machine info"
	@echo "  make scan-package      # scan the ueler package only (what ships)"
	@echo "  make scan-project      # scan the whole repository"
	@echo "  make docs              # build the docs once into site/"
	@echo "  make docs-serve        # serve all published doc versions locally"
	@echo "  make clean             # remove the virtual environment"

venv:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip setuptools wheel

install: venv
	$(PIP) install --editable .

test-fast: venv
	$(PYTHON) -m unittest discover tests

test-integration: venv
	@echo "Running integration test placeholder..."
	UELER_TEST_MODE=integration $(PYTHON) -m unittest discover tests

scan:
	python tools/scan_local_info.py --scope both

scan-package:
	python tools/scan_local_info.py --scope package

scan-project:
	python tools/scan_local_info.py --scope project

# Single-version preview of the working tree; no version selector.
docs:
	mkdocs build

# Serves the gh-pages version store, so the version dropdown works locally.
docs-serve:
	mike serve

clean:
	rm -rf $(VENV)
