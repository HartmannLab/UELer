import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import ueler
from ueler.runner import run_viewer


class RunnerSmokeTest(unittest.TestCase):
	def setUp(self) -> None:
		self._tmp_dir = TemporaryDirectory()
		self.tmp_root = Path(self._tmp_dir.name)

	def tearDown(self) -> None:
		self._tmp_dir.cleanup()

	def test_run_viewer_instantiates_and_displays(self) -> None:
		base_dir = self.tmp_root / "run_viewer_dataset"
		masks_dir = base_dir / "masks"
		annotations_dir = base_dir / "annotations"
		annotations_dir.mkdir(parents=True, exist_ok=True)
		masks_dir.mkdir(parents=True, exist_ok=True)
		base_dir.mkdir(parents=True, exist_ok=True)

		with patch("ueler.runner.ensure_aliases_loaded") as ensure_aliases, \
			patch("ueler.runner._load_viewer_factory") as load_factory, \
			patch("ueler.runner._load_display_helpers") as load_display:

			viewer_instance = MagicMock()
			factory_calls = {}

			def fake_factory(base_folder, *, masks_folder=None, annotations_folder=None, **kwargs):
				factory_calls["base"] = base_folder
				factory_calls["masks"] = masks_folder
				factory_calls["annotations"] = annotations_folder
				factory_calls["kwargs"] = kwargs
				return viewer_instance

			load_factory.return_value = fake_factory
			display_mock = MagicMock()
			update_mock = MagicMock()
			load_display.return_value = (display_mock, update_mock)

			run_viewer(
				base_dir,
				masks_folder=masks_dir,
				annotations_folder=annotations_dir,
				custom_arg="value",
			)

			ensure_aliases.assert_called_once()
			self.assertEqual(factory_calls["base"], str(base_dir))
			self.assertEqual(factory_calls["masks"], str(masks_dir))
			self.assertEqual(factory_calls["annotations"], str(annotations_dir))
			self.assertEqual(factory_calls["kwargs"], {"custom_arg": "value"})
			display_mock.assert_called_once_with(viewer_instance)
			update_mock.assert_called_once_with(viewer_instance)
			viewer_instance.after_all_plugins_loaded.assert_called_once()

	def test_run_viewer_respects_flags(self) -> None:
		base_dir = self.tmp_root / "run_viewer_flags"
		base_dir.mkdir(parents=True, exist_ok=True)

		with patch("ueler.runner.ensure_aliases_loaded") as ensure_aliases, \
			patch("ueler.runner._load_viewer_factory") as load_factory, \
			patch("ueler.runner._load_display_helpers") as load_display:

			viewer_instance = MagicMock()

			def fake_factory(base_folder, **_kwargs):
				return viewer_instance

			load_factory.return_value = fake_factory

			run_viewer(
				base_dir,
				auto_display=False,
				after_plugins=False,
				ensure_aliases=False,
			)

			ensure_aliases.assert_not_called()
			load_display.assert_not_called()
			viewer_instance.after_all_plugins_loaded.assert_not_called()

	def test_module_reexports_runner(self) -> None:
		self.assertTrue(hasattr(ueler, "run_viewer"))


if __name__ == "__main__":
	unittest.main()
