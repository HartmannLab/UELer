# Default developer targets for the UELer packaging skeleton

.PHONY: help venv install test-fast test-integration clean

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

clean:
	rm -rf $(VENV)
