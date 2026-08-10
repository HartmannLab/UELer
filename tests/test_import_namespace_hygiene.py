"""Guard the import-time contract of the public ``ueler`` namespace.

``import ueler`` used to install two ``MetaPathFinder`` instances at
``sys.meta_path[0]`` that claimed the top-level names ``viewer``, ``constants``,
``data_loader`` and ``image_utils`` for the pre-0.2 notebook layout. That is a
process-wide side effect: once UELer is a pip-installable package, any user
module or notebook-local file with one of those (very common) names could be
shadowed, and the "skip if a real one already exists" guard only ran at
registration time — a module that became importable *after* ``import ueler``
was still hijacked.

The shims are gone. These tests exist so they cannot come back unnoticed.
"""

import subprocess
import sys
import textwrap
import unittest

# Names the removed compatibility layer used to claim. Deliberately spelled out
# rather than imported, so the test keeps meaning after the shim tables are gone.
FORMERLY_ALIASED_NAMES = ("viewer", "constants", "data_loader", "image_utils")


def _run_in_subprocess(body: str) -> subprocess.CompletedProcess:
	"""Execute ``body`` in a pristine interpreter.

	A subprocess is required: the test suite's own bootstrap stubs modules and
	other test modules import ``ueler``, so ``sys.meta_path`` in this process
	says nothing about what a *fresh* ``import ueler`` does.
	"""

	return subprocess.run(
		[sys.executable, "-c", textwrap.dedent(body)],
		capture_output=True,
		text=True,
		env={"PYTHONPATH": "", "UELER_SKIP_TEST_BOOTSTRAP": "1", "PATH": "/usr/bin:/bin"},
	)


class TestImportNamespaceHygiene(unittest.TestCase):
	def test_import_ueler_adds_no_meta_path_finders(self):
		result = _run_in_subprocess(
			"""
			import sys
			before = list(sys.meta_path)
			import ueler
			added = [f for f in sys.meta_path if f not in before]
			print("ADDED:" + repr([type(f).__name__ for f in added]))
			"""
		)
		self.assertEqual(result.returncode, 0, msg=result.stderr)
		self.assertIn("ADDED:[]", result.stdout, msg=result.stdout)

	def test_import_ueler_claims_no_top_level_names(self):
		result = _run_in_subprocess(
			f"""
			import importlib.util
			import ueler
			claimed = []
			for name in {FORMERLY_ALIASED_NAMES!r}:
				try:
					spec = importlib.util.find_spec(name)
				except (ImportError, ValueError):
					spec = None
				# A spec resolving into the ueler package means the name was
				# hijacked; an unrelated third-party 'viewer' would not.
				if spec is not None and "ueler" in (spec.origin or ""):
					claimed.append(name)
			print("CLAIMED:" + repr(claimed))
			"""
		)
		self.assertEqual(result.returncode, 0, msg=result.stderr)
		self.assertIn("CLAIMED:[]", result.stdout, msg=result.stdout)

	def test_legacy_top_level_import_fails_cleanly(self):
		result = _run_in_subprocess(
			"""
			import ueler
			try:
				import viewer.main_viewer  # noqa: F401
			except ModuleNotFoundError:
				print("RAISED")
			else:
				print("RESOLVED")
			"""
		)
		self.assertEqual(result.returncode, 0, msg=result.stderr)
		self.assertIn("RAISED", result.stdout, msg=result.stdout)

	def test_compat_module_is_gone(self):
		with self.assertRaises(ModuleNotFoundError):
			__import__("ueler._compat")

	def test_ensure_compat_aliases_is_not_exported(self):
		import ueler

		self.assertNotIn("ensure_compat_aliases", ueler.__all__)
		self.assertFalse(hasattr(ueler, "ensure_compat_aliases"))


if __name__ == "__main__":  # pragma: no cover - unittest entrypoint
	unittest.main()
