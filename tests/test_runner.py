import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import ueler
from ueler.runner import load_cell_table, run_viewer


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
		self.assertTrue(hasattr(ueler, "load_cell_table"))

	def _make_viewer_mock(self) -> MagicMock:
		viewer = MagicMock()
		viewer.current_downsample_factor = 8
		viewer.update_marker_set_dropdown = MagicMock()
		viewer.update_controls = MagicMock()
		viewer.on_image_change = MagicMock()
		viewer.update_display = MagicMock()
		viewer.update_keys = MagicMock()
		viewer.refresh_bottom_panel = MagicMock()
		viewer.inform_plugins = MagicMock()
		viewer.after_all_plugins_loaded = MagicMock()
		return viewer

	def test_load_cell_table_from_path_refreshes_ui(self) -> None:
		csv_path = self.tmp_root / "cell_table.csv"
		csv_path.write_text("a,b\n1,2\n")

		viewer = self._make_viewer_mock()
		viewer.load_cell_table_from_path = MagicMock()

		with patch("ueler.runner._load_display_helpers") as load_display:
			load_display.return_value = (MagicMock(), MagicMock())
			load_cell_table(
				viewer,
				cell_table_path=csv_path,
				auto_display=False,
				after_plugins=False,
			)

		load_display.assert_not_called()
		viewer.load_cell_table_from_path.assert_called_once_with(str(csv_path))
		viewer.update_marker_set_dropdown.assert_called_once()
		viewer.update_controls.assert_called_once_with(None)
		viewer.on_image_change.assert_called_once_with(None)
		viewer.update_display.assert_called_once_with(viewer.current_downsample_factor)
		viewer.update_keys.assert_called_once_with(None)
		viewer.refresh_bottom_panel.assert_called_once()
		viewer.inform_plugins.assert_called_once_with('refresh_roi_table')
		viewer.after_all_plugins_loaded.assert_not_called()

	def test_load_cell_table_with_dataframe_and_display(self) -> None:
		viewer = self._make_viewer_mock()
		viewer.set_cell_table = MagicMock()

		fake_df = object()

		with patch("ueler.runner._load_display_helpers") as load_display:
			display_mock = MagicMock()
			panel_mock = MagicMock()
			load_display.return_value = (display_mock, panel_mock)
			load_cell_table(
				viewer,
				cell_table=fake_df,
				auto_display=True,
				after_plugins=True,
			)

		viewer.set_cell_table.assert_called_once_with(fake_df)
		load_display.assert_called_once()
		display_mock.assert_called_once_with(viewer)
		panel_mock.assert_called_once_with(viewer)
		viewer.update_marker_set_dropdown.assert_called_once()
		viewer.update_controls.assert_called_once_with(None)
		viewer.on_image_change.assert_called_once_with(None)
		viewer.update_display.assert_called_once_with(viewer.current_downsample_factor)
		viewer.update_keys.assert_called_once_with(None)
		viewer.refresh_bottom_panel.assert_called_once()
		viewer.inform_plugins.assert_called_once_with('refresh_roi_table')
		viewer.after_all_plugins_loaded.assert_called_once()

	def test_load_cell_table_validation(self) -> None:
		viewer = self._make_viewer_mock()
		viewer.load_cell_table_from_path = MagicMock()
		viewer.set_cell_table = MagicMock()

		with self.assertRaises(ValueError):
			load_cell_table(viewer)

		with self.assertRaises(ValueError):
			load_cell_table(viewer, cell_table_path=self.tmp_root / "foo.csv", cell_table=object())


if __name__ == "__main__":
	unittest.main()
