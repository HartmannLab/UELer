"""Test package bootstrap.

Importing this module sets up lightweight fakes for optional dependencies via
``tests.bootstrap`` before any individual test modules execute.
"""

from . import bootstrap  # noqa: F401
