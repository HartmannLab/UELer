# Default developer targets for the UELer packaging skeleton

.PHONY: help venv install test-fast test-integration test-ci scan scan-package scan-project docs docs-serve check-docs clean clean-dist build check-dist check-release check-rehearsal publish-test publish

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
	@echo "  make test-ci           # run the suite in the active env, no skips allowed"
	@echo "  make scan              # scan pkg + project for local/machine info"
	@echo "  make scan-package      # scan the ueler package only (what ships)"
	@echo "  make scan-project      # scan the whole repository"
	@echo "  make docs              # build the docs once into site/"
	@echo "  make docs-serve        # serve all published doc versions locally"
	@echo "  make check-docs        # the docs must build strictly and describe the real software"
	@echo "  make clean-dist        # remove stale build artefacts from dist/"
	@echo "  make build             # build a fresh sdist + wheel into dist/"
	@echo "  make check-dist        # validate the built artefacts with twine"
	@echo "  make check-release     # cross-check every version declaration (TAG=v0.5.0-alpha)"
	@echo "  make check-rehearsal   # a stable tag must promote a published rc (TAG=v0.6.0)"
	@echo "  make publish-test      # upload dist/* to TestPyPI"
	@echo "  make publish           # upload dist/* to PyPI (irreversible)"
	@echo "  make clean             # remove the virtual environment"

venv:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip setuptools wheel

install: venv
	$(PIP) install --editable .

# UELER_TEST_BOOTSTRAP=1 opts into the dependency stubs installed by
# sitecustomize/usercustomize. It is deliberately off by default so a plain
# interpreter run from the repo root gets the real dependency stack.
test-fast: venv
	UELER_TEST_BOOTSTRAP=1 $(PYTHON) -m unittest discover tests

test-integration: venv
	@echo "Running integration test placeholder..."
	UELER_TEST_BOOTSTRAP=1 UELER_TEST_MODE=integration $(PYTHON) -m unittest discover tests

# What CI runs (.github/workflows/tests.yml): the suite against whatever is
# installed in the active interpreter, failing on the first skipped test. A skip
# means a dependency is missing, and a missing dependency turns a whole code path
# into an untested one while `unittest` still prints OK.
test-ci:
	python tools/run_test_suite.py --max-skips 0

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

# Two different questions, both of which have to be "yes" before docs deploy.
# `--strict` asks whether the site is well formed: every nav entry resolves,
# every internal link lands. The checker asks whether it is *true*: the plugin
# names, module paths, extras and Python range it asserts still match the code.
# Neither subsumes the other — the audit that motivated the checker found a
# strict-clean site asserting a module path that had not existed for months.
check-docs:
	mkdocs build --strict --site-dir $(if $(SITE_DIR),$(SITE_DIR),site)
	python tools/check_docs_consistency.py

# A stale artefact in dist/ is the main way a wrong version reaches PyPI, since
# `twine upload dist/*` globs whatever is there. Every build starts from empty.
clean-dist:
	rm -rf dist build *.egg-info

build: clean-dist
	python -m build

check-dist:
	python -m twine check --strict dist/*

# Run this before creating a release tag. TAG is optional: without it the source
# declarations and dist/ are cross-checked; with it, the tag joins the comparison.
check-release:
	python tools/check_release_tag.py $(if $(TAG),$(TAG),--no-tag)

# Answers "may this stable tag go to PyPI?" the same way release.yml does: the
# highest rc for the same version must be served by TestPyPI, and nothing that
# ships in the wheel may have changed since it. Run it before tagging a stable
# release, not after.
check-rehearsal:
	@$(if $(TAG),,echo "give a stable tag: make check-rehearsal TAG=v0.6.0" >&2; exit 1)
	python tools/check_stable_rehearsal.py $(TAG)

publish-test: check-dist
	python -m twine upload --repository testpypi dist/*

# PyPI version numbers are append-only: a released version can be yanked but
# never replaced or reused. Rehearse with `make publish-test` first.
publish: check-dist
	python -m twine upload dist/*

clean:
	rm -rf $(VENV)
