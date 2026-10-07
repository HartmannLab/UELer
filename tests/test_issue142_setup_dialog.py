"""Guided data mapping and advanced settings (#142).

Three things are under test, and they are independent of each other:

* ``discover_mask_suffixes`` — the list the ``Mask key:`` dropdown needs
  *before* any TIFF has been read, which is the whole reason it exists.
* ``apply_options`` — the rule that lets the dropdowns be repopulated from new
  data without ever silently changing an answer the user already gave. Every
  branch of that rule is a way the viewer could mis-map a column, so each gets
  its own case.
* ``SetupDialog`` / ``build_setup_steps`` — which questions get asked, in what
  order, and the per-step record that stops them being asked twice.

The viewer is a stand-in rather than a real ``ImageMaskViewer``: constructing
one scans an image folder and builds the whole widget tree, which would test the
fixture rather than the feature. The viewer methods are the shipped ones all the
same -- taken off the class and run over a namespace, or bound onto a small
class where the method calls a sibling or reads a property.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import ipywidgets as widgets

from ueler.data_loader import discover_mask_suffixes
from ueler.viewer.data_mapping import (
    CELL_TABLE_FIELDS,
    KEY_FIELDS,
    MASK_FIELD,
    apply_options,
    build_key_widget,
    column_options,
    ensure_option,
)
from ueler.viewer.main_viewer import ImageMaskViewer
from ueler.viewer.setup_dialog import (
    STATE_FILENAME,
    SetupDialog,
    SetupStep,
    build_setup_steps,
    load_seen_steps,
    record_seen_steps,
)


def _touch(path):
    with open(path, "w") as handle:
        handle.write("")


def _key_dropdown(value, description="X key:"):
    return widgets.Dropdown(options=[value], value=value, description=description)


class DiscoverMaskSuffixesTests(unittest.TestCase):
    """The suffix list, from filenames alone."""

    def test_suffixes_come_from_the_filenames_not_the_images(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _touch(os.path.join(folder, "fov1_whole_cell.tiff"))
            _touch(os.path.join(folder, "fov1_nuclear.tif"))

            self.assertEqual(
                discover_mask_suffixes(folder, ["fov1"]),
                ["nuclear", "whole_cell"],
            )

    def test_several_fovs_are_unioned(self) -> None:
        """One FOV may be missing a layer the rest of the set has."""
        with tempfile.TemporaryDirectory() as folder:
            _touch(os.path.join(folder, "fov1_whole_cell.tiff"))
            _touch(os.path.join(folder, "fov2_nuclear.tiff"))

            self.assertEqual(
                discover_mask_suffixes(folder, ["fov1", "fov2"]),
                ["nuclear", "whole_cell"],
            )

    def test_only_the_first_few_fovs_are_scanned(self) -> None:
        """A dataset with thousands of FOVs must not pay a glob for each one."""
        with tempfile.TemporaryDirectory() as folder:
            _touch(os.path.join(folder, "fov1_whole_cell.tiff"))
            _touch(os.path.join(folder, "fov9_late.tiff"))

            found = discover_mask_suffixes(folder, [f"fov{n}" for n in range(1, 10)], limit=1)

            self.assertEqual(found, ["whole_cell"])

    def test_files_belonging_to_another_fov_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _touch(os.path.join(folder, "other_whole_cell.tiff"))

            self.assertEqual(discover_mask_suffixes(folder, ["fov1"]), [])

    def test_a_missing_folder_is_not_an_error(self) -> None:
        self.assertEqual(discover_mask_suffixes("/no/such/folder", ["fov1"]), [])
        self.assertEqual(discover_mask_suffixes(None, ["fov1"]), [])

    def test_a_file_in_place_of_a_folder_is_not_an_error(self) -> None:
        """``scandir`` raises ``NotADirectoryError``; discovery must absorb it."""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "not_a_folder.txt")
            _touch(path)

            self.assertEqual(discover_mask_suffixes(path, ["fov1"]), [])

    def test_the_folder_is_read_once_however_many_fovs_are_scanned(self) -> None:
        """The constructor pays for one listing, not one per FOV (#142 regression).

        The first implementation globbed ``{fov}_*`` per FOV per extension -- six
        full listings of a directory that, for a cohort on shared storage, holds
        tens of thousands of entries.  That is what made the viewer unopenable.
        """
        with tempfile.TemporaryDirectory() as folder:
            for fov in ("fov1", "fov2", "fov3"):
                _touch(os.path.join(folder, f"{fov}_whole_cell.tiff"))
                _touch(os.path.join(folder, f"{fov}_nuclear.tiff"))

            real_scandir = os.scandir
            with mock.patch("ueler.data_loader.os.scandir", side_effect=real_scandir) as scandir:
                found = discover_mask_suffixes(folder, ["fov1", "fov2", "fov3"])

            self.assertEqual(found, ["nuclear", "whole_cell"])
            self.assertEqual(scandir.call_count, 1)

    def test_a_directory_too_slow_to_finish_yields_what_it_found(self) -> None:
        """An exhausted budget returns a partial list rather than stalling."""
        with tempfile.TemporaryDirectory() as folder:
            for index in range(1200):
                _touch(os.path.join(folder, f"fov1_s{index:04d}.tiff"))

            # Expired before the first clock check, which falls on entry 512.
            partial = discover_mask_suffixes(folder, ["fov1"], budget=1e-9)
            complete = discover_mask_suffixes(folder, ["fov1"], budget=None)

            self.assertEqual(len(complete), 1200)
            self.assertLess(len(partial), len(complete))
            self.assertTrue(set(partial) <= set(complete))

    def test_an_uppercase_extension_still_counts(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _touch(os.path.join(folder, "fov1_whole_cell.TIFF"))

            self.assertEqual(discover_mask_suffixes(folder, ["fov1"]), ["whole_cell"])

    def test_a_bare_prefix_with_no_suffix_is_not_offered(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _touch(os.path.join(folder, "fov1_.tiff"))

            self.assertEqual(discover_mask_suffixes(folder, ["fov1"]), [])


class ViewerMaskSuffixTests(unittest.TestCase):
    """``available_mask_suffixes`` unions the lazy set with the glob."""

    def test_the_lazily_accumulated_set_is_included(self) -> None:
        """``mask_names_set`` holds suffixes of FOVs already read; keep them."""
        with tempfile.TemporaryDirectory() as folder:
            _touch(os.path.join(folder, "fov1_whole_cell.tiff"))
            viewer = SimpleNamespace(
                mask_names_set={"membrane"},
                masks_folder=folder,
                available_fovs=["fov1"],
            )

            found = ImageMaskViewer.available_mask_suffixes(viewer)

            self.assertEqual(found, ["membrane", "whole_cell"])

    def test_no_masks_folder_yields_whatever_was_already_known(self) -> None:
        viewer = SimpleNamespace(mask_names_set=set(), masks_folder=None, available_fovs=[])

        self.assertEqual(ImageMaskViewer.available_mask_suffixes(viewer), [])


class KeyWidgetTests(unittest.TestCase):
    """The key field must never be able to strand the user (#142 follow-up).

    The first implementation used a ``Dropdown``, which made each key exactly as
    good as the options discovery happened to find.  When discovery came up
    short -- an unreadable masks folder, a scan that hit its time budget, a
    column no heuristic would guess -- the field held one wrong value and
    offered no way to type the right one, which is strictly worse than the free
    text it replaced.

    The construction kwargs are asserted rather than the resulting traits: the
    headless bootstrap maps every widget class onto one stub, so a stub
    ``Combobox`` would accept an unlisted value even if the shipped code had
    regressed to a ``Dropdown``.  The kwargs are the thing that is actually
    load-bearing, and they are checked against what the factory passes.
    """

    class _Recorder:
        """Stands in for the widget class and keeps what it was built with."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def test_the_options_are_a_suggestion_list_not_a_constraint(self) -> None:
        """``ensure_option=False`` is what keeps an unlisted value assignable."""
        for field in KEY_FIELDS:
            with self.subTest(field=field.attribute):
                built = build_key_widget(self._Recorder, field)

                self.assertIs(built.kwargs["ensure_option"], False)

    def test_the_handler_fires_on_commit_not_per_keystroke(self) -> None:
        """``on_key_change`` reaches ``ensure_columns``; ``fo`` is never a column."""
        for field in KEY_FIELDS:
            with self.subTest(field=field.attribute):
                built = build_key_widget(self._Recorder, field)

                self.assertIs(built.kwargs["continuous_update"], False)

    def test_it_opens_on_the_shipped_default(self) -> None:
        for field in KEY_FIELDS:
            with self.subTest(field=field.attribute):
                widget = build_key_widget(widgets.Combobox, field)

                self.assertEqual(widget.value, field.preferred[0])
                self.assertEqual(widget.description, field.description)

    def test_a_typed_key_survives_a_later_refresh_as_an_option(self) -> None:
        """Discovery that does not know the user's column must not erase it."""
        widget = build_key_widget(widgets.Combobox, CELL_TABLE_FIELDS[2])
        widget.value = "my_label_column"

        apply_options(widget, ["alpha", "beta"], preferred=CELL_TABLE_FIELDS[2].preferred)

        self.assertEqual(widget.value, "my_label_column")
        self.assertIn("my_label_column", widget.options)


class ApplyOptionsTests(unittest.TestCase):
    """Replacing the options must never change a valid answer."""

    def test_a_value_the_data_has_stays_selected(self) -> None:
        widget = _key_dropdown("centroid-1")

        changed = apply_options(widget, ["centroid-0", "centroid-1"], preferred=("centroid-1",))

        self.assertFalse(changed)
        self.assertEqual(widget.value, "centroid-1")

    def test_an_absent_default_falls_back_to_a_preferred_alias(self) -> None:
        """The pre-fill: a table using ``x``/``y`` is mapped on arrival."""
        widget = _key_dropdown("centroid-1")

        changed = apply_options(widget, ["x", "y", "label"], preferred=("centroid-1", "x"))

        self.assertTrue(changed)
        self.assertEqual(widget.value, "x")

    def test_an_unrecognised_key_is_kept_rather_than_guessed_at(self) -> None:
        """Silently retargeting a key the user chose is the failure to avoid."""
        widget = _key_dropdown("my_x")

        changed = apply_options(widget, ["alpha", "beta"], preferred=("centroid-1",))

        self.assertFalse(changed)
        self.assertEqual(widget.value, "my_x")
        self.assertIn("my_x", widget.options)

    def test_empty_options_leave_the_widget_untouched(self) -> None:
        """An images-only session has no columns and must keep working."""
        widget = _key_dropdown("centroid-1")

        self.assertFalse(apply_options(widget, [], preferred=("centroid-1",)))
        self.assertEqual(widget.value, "centroid-1")
        self.assertEqual(tuple(widget.options), ("centroid-1",))

    def test_duplicates_collapse_and_order_is_preserved(self) -> None:
        widget = _key_dropdown("a")

        apply_options(widget, ["a", "b", "a", "c"])

        self.assertEqual(tuple(widget.options), ("a", "b", "c"))

    def test_a_missing_widget_is_tolerated(self) -> None:
        self.assertFalse(apply_options(None, ["a"]))


class ColumnOptionsTests(unittest.TestCase):
    """X and Y offer numeric columns -- unless that would offer nothing."""

    def test_numeric_only_filters_the_schema(self) -> None:
        import numpy as np

        schema = {"centroid-0": np.dtype("float32"), "fov": np.dtype("O")}

        self.assertEqual(column_options(schema, numeric_only=True), ["centroid-0"])
        self.assertEqual(column_options(schema), ["centroid-0", "fov"])

    def test_an_all_object_schema_still_offers_everything(self) -> None:
        """Losing the only control that could fix a bad dtype helps nobody."""
        import numpy as np

        schema = {"x": np.dtype("O"), "y": np.dtype("O")}

        self.assertEqual(column_options(schema, numeric_only=True), ["x", "y"])


class EnsureOptionTests(unittest.TestCase):
    """A stale saved key must not abort the whole widget-state restore."""

    def test_a_value_outside_the_options_is_re_admitted(self) -> None:
        widget = widgets.Dropdown(options=["a", "b"], value="a")

        self.assertTrue(ensure_option(widget, "gone"))
        widget.value = "gone"
        self.assertEqual(widget.value, "gone")

    def test_a_known_value_changes_nothing(self) -> None:
        widget = widgets.Dropdown(options=["a", "b"], value="a")

        self.assertFalse(ensure_option(widget, "b"))
        self.assertEqual(tuple(widget.options), ("a", "b"))


class BuildSetupStepsTests(unittest.TestCase):
    """Which questions this dataset actually needs asked."""

    @staticmethod
    def _viewer(*, masks=False, cell_table=None):
        ui_component = SimpleNamespace(
            cache_size_input=widgets.IntText(value=100),
            pixel_size_inttext=widgets.IntText(value=390),
            enable_downsample_checkbox=widgets.Checkbox(value=True),
            mask_key=_key_dropdown("whole_cell", "Mask key:"),
            x_key=_key_dropdown("centroid-1"),
            y_key=_key_dropdown("centroid-0", "Y key:"),
            label_key=_key_dropdown("label", "Label key:"),
            fov_key=_key_dropdown("fov", "Fov key:"),
        )
        return SimpleNamespace(
            ui_component=ui_component, masks_available=masks, cell_table=cell_table
        )

    def test_images_only_asks_one_question(self) -> None:
        steps = build_setup_steps(self._viewer())

        self.assertEqual([step.key for step in steps], ["images"])

    def test_masks_add_the_mask_key(self) -> None:
        steps = build_setup_steps(self._viewer(masks=True))

        self.assertEqual([step.key for step in steps], ["images", "masks"])

    def test_a_cell_table_adds_the_column_mapping(self) -> None:
        steps = build_setup_steps(self._viewer(masks=True, cell_table=object()))

        self.assertEqual([step.key for step in steps], ["images", "masks", "cell_table"])

    def test_already_answered_steps_are_dropped(self) -> None:
        """``load_cell_table`` later in the session asks only what is new."""
        viewer = self._viewer(masks=True, cell_table=object())

        steps = build_setup_steps(viewer, seen=("images", "masks"))

        self.assertEqual([step.key for step in steps], ["cell_table"])

    def test_the_card_shows_the_accordion_s_own_widgets(self) -> None:
        """Not copies -- so what the user picks here *is* the live setting."""
        viewer = self._viewer(cell_table=object())

        cell_step = build_setup_steps(viewer)[-1]

        self.assertIs(cell_step.widgets[0], viewer.ui_component.x_key)
        self.assertEqual(
            [widget.description for widget in cell_step.widgets],
            [field.description for field in CELL_TABLE_FIELDS],
        )

    def test_the_mask_step_carries_the_mask_key(self) -> None:
        viewer = self._viewer(masks=True)

        mask_step = build_setup_steps(viewer)[-1]

        self.assertEqual(mask_step.widgets[0].description, MASK_FIELD.description)

    def test_a_viewer_without_widgets_asks_nothing(self) -> None:
        self.assertEqual(build_setup_steps(SimpleNamespace(ui_component=None)), [])


class SetupDialogTests(unittest.TestCase):
    """Navigation, and what closing reports."""

    @staticmethod
    def _steps():
        return [
            SetupStep("images", "Images", "intro", (widgets.IntText(value=1),)),
            SetupStep("cell_table", "Cells", "intro", (widgets.Dropdown(options=["a"]),)),
        ]

    def test_opening_shows_the_first_step(self) -> None:
        dialog = SetupDialog()

        self.assertTrue(dialog.open(self._steps()))
        self.assertTrue(dialog.is_open)
        self.assertEqual(dialog.title_label.value, "Images")
        self.assertEqual(dialog.progress_label.value, "Step 1 of 2")
        self.assertEqual(dialog.view.layout.display, "")

    def test_there_is_no_back_from_the_first_step(self) -> None:
        dialog = SetupDialog()
        dialog.open(self._steps())

        self.assertTrue(dialog.back_button.disabled)
        dialog.back()
        self.assertEqual(dialog.title_label.value, "Images")

    def test_next_advances_and_the_last_step_says_done(self) -> None:
        dialog = SetupDialog()
        dialog.open(self._steps())

        dialog.advance()

        self.assertEqual(dialog.title_label.value, "Cells")
        self.assertEqual(dialog.next_button.description, "Done")
        self.assertFalse(dialog.back_button.disabled)

    def test_done_closes_and_reports_every_step(self) -> None:
        dialog = SetupDialog()
        reported = []
        dialog.open(self._steps(), on_close=reported.extend)

        dialog.advance()
        dialog.advance()

        self.assertFalse(dialog.is_open)
        self.assertEqual(dialog.view.layout.display, "none")
        self.assertEqual(reported, ["images", "cell_table"])

    def test_skipping_dismisses_the_remaining_steps_too(self) -> None:
        """Re-asking a question the user dismissed, on every load, is worse."""
        dialog = SetupDialog()
        reported = []
        dialog.open(self._steps(), on_close=reported.extend)

        dialog.skip()

        self.assertEqual(reported, ["images", "cell_table"])

    def test_a_single_step_shows_no_progress_counter(self) -> None:
        dialog = SetupDialog()
        dialog.open(self._steps()[:1])

        self.assertEqual(dialog.progress_label.value, "")
        self.assertEqual(dialog.next_button.description, "Done")

    def test_opening_over_an_open_wizard_is_refused(self) -> None:
        """Both entry points can reach here in one cell."""
        dialog = SetupDialog()
        dialog.open(self._steps())

        self.assertFalse(dialog.open(self._steps()[:1]))
        self.assertEqual(dialog.title_label.value, "Images")

    def test_opening_with_nothing_to_ask_does_not_show_the_card(self) -> None:
        dialog = SetupDialog()

        self.assertFalse(dialog.open([]))
        self.assertEqual(dialog.view.layout.display, "none")

    def test_a_failing_callback_still_leaves_the_dialog_closed(self) -> None:
        """A scrim left over a failed write would block the whole UI."""
        dialog = SetupDialog()

        def explode(_keys):
            raise OSError("read-only settings folder")

        dialog.open(self._steps(), on_close=explode)
        with self.assertLogs("ueler.viewer.setup_dialog", level="ERROR"):
            dialog.skip()

        self.assertFalse(dialog.is_open)
        self.assertEqual(dialog.view.layout.display, "none")


class SeenStepRecordTests(unittest.TestCase):
    """The per-dataset record that stops the wizard repeating."""

    def test_an_absent_record_means_nothing_was_answered(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(load_seen_steps(os.path.join(folder, STATE_FILENAME)), set())
        self.assertEqual(load_seen_steps(None), set())

    def test_steps_accumulate_across_calls(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, STATE_FILENAME)

            record_seen_steps(path, ["images"])
            record_seen_steps(path, ["cell_table"])

            self.assertEqual(load_seen_steps(path), {"images", "cell_table"})

    def test_a_corrupt_record_is_treated_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, STATE_FILENAME)
            with open(path, "w") as handle:
                handle.write("{not json")

            self.assertEqual(load_seen_steps(path), set())

    def test_an_unwritable_record_is_not_an_error(self) -> None:
        """A read-only settings folder costs a repeated dialog, not a session."""
        record_seen_steps("/no/such/folder/setup.json", ["images"])


class MaybeShowSetupDialogTests(unittest.TestCase):
    """The trigger, including every reason it declines to fire."""

    def _viewer(self, folder, *, displayed=True, cell_table=None):
        ui_component = SimpleNamespace(
            setup_dialog=SetupDialog(),
            cache_size_input=widgets.IntText(value=100),
            pixel_size_inttext=widgets.IntText(value=390),
            enable_downsample_checkbox=widgets.Checkbox(value=True),
        )
        return SimpleNamespace(
            ui_component=ui_component,
            settings_folder=folder,
            masks_available=False,
            cell_table=cell_table,
            _widget_displayed=displayed,
        )

    def test_the_first_load_opens_the_wizard(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            viewer = self._viewer(folder)

            self.assertTrue(ImageMaskViewer.maybe_show_setup_dialog(viewer))
            self.assertTrue(viewer.ui_component.setup_dialog.is_open)

    def test_finishing_records_the_steps_and_the_next_load_is_quiet(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            viewer = self._viewer(folder)
            ImageMaskViewer.maybe_show_setup_dialog(viewer)
            viewer.ui_component.setup_dialog.advance()

            with open(os.path.join(folder, STATE_FILENAME)) as handle:
                self.assertEqual(json.load(handle)["seen_steps"], ["images"])

            self.assertFalse(ImageMaskViewer.maybe_show_setup_dialog(viewer))

    def test_a_cell_table_arriving_later_asks_only_about_itself(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            viewer = self._viewer(folder)
            ImageMaskViewer.maybe_show_setup_dialog(viewer)
            viewer.ui_component.setup_dialog.advance()

            viewer.cell_table = object()
            viewer.ui_component.x_key = _key_dropdown("centroid-1")
            viewer.ui_component.y_key = _key_dropdown("centroid-0", "Y key:")
            viewer.ui_component.label_key = _key_dropdown("label", "Label key:")
            viewer.ui_component.fov_key = _key_dropdown("fov", "Fov key:")

            self.assertTrue(ImageMaskViewer.maybe_show_setup_dialog(viewer))
            dialog = viewer.ui_component.setup_dialog
            self.assertEqual(dialog.current_step.key, "cell_table")
            self.assertEqual(dialog.progress_label.value, "")

    def test_nothing_opens_before_the_widget_tree_is_displayed(self) -> None:
        """A modal opened into an undisplayed tree appears over the next render."""
        with tempfile.TemporaryDirectory() as folder:
            viewer = self._viewer(folder, displayed=False)

            self.assertFalse(ImageMaskViewer.maybe_show_setup_dialog(viewer))

    def test_a_viewer_with_no_dialog_mounted_declines(self) -> None:
        viewer = SimpleNamespace(ui_component=SimpleNamespace(), _widget_displayed=True)

        self.assertFalse(ImageMaskViewer.maybe_show_setup_dialog(viewer))


class _MappingViewer:
    """A viewer carrying only what the refresh touches, with the real methods.

    Built as a class rather than a namespace because ``refresh_data_mapping_options``
    calls a sibling method and reads a property; both have to resolve the way
    they do on the real viewer for the test to mean anything.
    """

    available_mask_suffixes = ImageMaskViewer.available_mask_suffixes
    refresh_data_mapping_options = ImageMaskViewer.refresh_data_mapping_options

    def __init__(self, schema, masks_folder=None, fovs=()):
        self.ui_component = SimpleNamespace(
            x_key=_key_dropdown("centroid-1"),
            y_key=_key_dropdown("centroid-0", "Y key:"),
            label_key=_key_dropdown("label", "Label key:"),
            fov_key=_key_dropdown("fov", "Fov key:"),
            mask_key=_key_dropdown("whole_cell", "Mask key:"),
        )
        self.cell_table_schema = schema
        self.mask_names_set = set()
        self.masks_folder = masks_folder
        self.available_fovs = list(fovs)
        self.update_keys_calls = []

    def update_keys(self, *args):
        self.update_keys_calls.append(args)


class _ExplodingSchemaViewer(_MappingViewer):
    """A table whose schema cannot be read -- the mask list must survive it."""

    @property
    def cell_table_schema(self):
        raise RuntimeError("no schema")

    @cell_table_schema.setter
    def cell_table_schema(self, _value):
        return None


class RefreshDataMappingOptionsTests(unittest.TestCase):
    """The viewer-level refresh, over both sources at once."""

    def test_the_column_dropdowns_are_built_from_the_schema(self) -> None:
        import numpy as np

        viewer = _MappingViewer(
            {
                "x": np.dtype("float32"),
                "y": np.dtype("float32"),
                "cell_label": np.dtype("int32"),
                "sample": np.dtype("O"),
            }
        )

        viewer.refresh_data_mapping_options()

        ui_component = viewer.ui_component
        # X and Y take the numeric columns only; the others take all of them.
        self.assertEqual(tuple(ui_component.x_key.options), ("x", "y", "cell_label"))
        self.assertEqual(tuple(ui_component.fov_key.options), ("x", "y", "cell_label", "sample"))
        self.assertEqual(ui_component.x_key.value, "x")
        self.assertEqual(ui_component.y_key.value, "y")
        self.assertEqual(ui_component.label_key.value, "cell_label")
        self.assertEqual(ui_component.fov_key.value, "sample")

    def test_the_mask_dropdown_is_built_from_the_folder(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _touch(os.path.join(folder, "fov1_nuclear.tiff"))
            viewer = _MappingViewer({}, masks_folder=folder, fovs=["fov1"])

            viewer.refresh_data_mapping_options()

            self.assertEqual(tuple(viewer.ui_component.mask_key.options), ("nuclear",))
            self.assertEqual(viewer.ui_component.mask_key.value, "nuclear")

    def test_an_images_only_session_keeps_its_defaults(self) -> None:
        viewer = _MappingViewer({})

        viewer.refresh_data_mapping_options()

        self.assertEqual(viewer.ui_component.x_key.value, "centroid-1")
        self.assertEqual(viewer.ui_component.mask_key.value, "whole_cell")

    def test_the_viewer_attributes_are_resynced_afterwards(self) -> None:
        import numpy as np

        viewer = _MappingViewer({"x": np.dtype("float32")})

        viewer.refresh_data_mapping_options()

        self.assertEqual(len(viewer.update_keys_calls), 1)

    def test_a_schema_that_raises_does_not_stop_the_mask_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _touch(os.path.join(folder, "fov1_nuclear.tiff"))
            viewer = _ExplodingSchemaViewer({}, masks_folder=folder, fovs=["fov1"])

            viewer.refresh_data_mapping_options()

            self.assertEqual(viewer.ui_component.mask_key.value, "nuclear")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
