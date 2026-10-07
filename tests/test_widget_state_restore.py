"""A saved settings file must never stop the viewer opening.

``widget_states.json`` is written by one session and read by another, and the
two need not agree.  The case that prompted this is the ordinary one: a FOV is
removed from the data folder, so the saved ``image_selector`` names an image
that is no longer among the dropdown's options.  Restoring it raised
``TraitError`` out of ``ImageMaskViewer.__init__`` -- not a wrong setting, but
no viewer at all, and a traceback naming a widget rather than the file behind
it.

The rule under test: a value the widget refuses is replaced by the nearest one
it accepts, the substitution is warned about, and the restore carries on.  The
warning matters as much as the recovery -- a setting that silently reverted is
a wrong figure waiting to happen.
"""

import json
import logging
import os
import tempfile
import unittest

from ueler.viewer.widget_restore import restore_widget_value, restore_widget_values


class _Strict:
    """A widget that validates its value, as every ipywidgets trait does.

    The headless test bootstrap maps every widget class onto one permissive
    stub, so a test written against ``ipywidgets.Dropdown`` here would accept
    any value and prove nothing.  This models the behaviour the ladder exists
    for -- measured against real ipywidgets, where ``Dropdown`` (option gone),
    ``Checkbox`` (a string), ``ColorPicker`` (nonsense) and ``FloatSlider``
    (``None``) all raise ``TraitError``, while ``IntText`` casts ``"7"`` and
    ``IntSlider`` clamps for itself.
    """

    def __init__(self, value, *, options=None, minimum=None, maximum=None,
                 default=None, cast=None, validator=None):
        self.options = list(options) if options else []
        self.min = minimum
        self.max = maximum
        self._cast = cast
        self._default = default
        self._validator = validator
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new):
        if self._cast is not None and type(new) is not self._cast:
            raise TypeError(f"expected {self._cast.__name__}, got {type(new).__name__}")
        if self.options and new not in self.options:
            raise ValueError(f"{new!r} is not one of {self.options!r}")
        if isinstance(new, (int, float)) and not isinstance(new, bool):
            if self.min is not None and new < self.min:
                raise ValueError("below min")
            if self.max is not None and new > self.max:
                raise ValueError("above max")
        if self._validator is not None and not self._validator(new):
            raise ValueError(f"{new!r} is not acceptable")
        self._value = new

    def trait_defaults(self, name):
        return {name: self._default}


def _dropdown(value, options):
    return _Strict(value, options=options, default=options[0], cast=str)


def _checkbox(value):
    return _Strict(value, default=False, cast=bool)


def _int_text(value, **kwargs):
    return _Strict(value, default=0, cast=int, **kwargs)


def _combobox(value, options):
    """The data-mapping keys: options are a suggestion list, so no constraint."""
    return _Strict(value, default=options[0], cast=str)


class _CapturedWarnings(logging.Handler):
    """Collects formatted warnings so a test can assert what the user is told."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _logger_with_capture():
    log = logging.getLogger(f"ueler.test.{id(object())}")
    log.propagate = False
    log.setLevel(logging.WARNING)
    handler = _CapturedWarnings()
    log.handlers = [handler]
    return log, handler


class MissingFovTests(unittest.TestCase):
    """The reported case, end to end on the widget that caused it."""

    def _selector(self):
        return _dropdown("fov1", ["fov1", "fov2", "fov3"])

    def test_a_saved_fov_that_no_longer_exists_falls_back_to_the_first_one(self) -> None:
        selector = self._selector()
        log, captured = _logger_with_capture()

        exact = restore_widget_value(selector, "fov_deleted_last_week", label="image_selector", log=log)

        self.assertFalse(exact)
        self.assertEqual(selector.value, "fov1")
        self.assertEqual(len(captured.messages), 1)

    def test_the_warning_names_the_widget_the_value_and_the_substitute(self) -> None:
        selector = self._selector()
        log, captured = _logger_with_capture()

        restore_widget_value(selector, "fov_deleted_last_week", label="image_selector", log=log)

        message = captured.messages[0]
        self.assertIn("image_selector", message)
        self.assertIn("fov_deleted_last_week", message)
        self.assertIn("fov1", message)

    def test_a_saved_fov_that_still_exists_is_restored_untouched(self) -> None:
        selector = self._selector()
        log, captured = _logger_with_capture()

        exact = restore_widget_value(selector, "fov3", label="image_selector", log=log)

        self.assertTrue(exact)
        self.assertEqual(selector.value, "fov3")
        self.assertEqual(captured.messages, [])


class SubstitutionLadderTests(unittest.TestCase):
    """Each rung keeps as much of the saved intent as it can."""

    def test_a_type_that_survived_a_json_round_trip_is_cast(self) -> None:
        widget = _int_text(3)
        log, captured = _logger_with_capture()

        restore_widget_value(widget, "7", label="cache_size_input", log=log)

        self.assertEqual(widget.value, 7)

    def test_a_checkbox_reads_only_unambiguous_spellings(self) -> None:
        """``bool("false")`` is ``True``, so the generic cast is worse than none."""
        for saved, expected in (("true", True), ("false", False), ("0", False), ("on", True)):
            with self.subTest(saved=saved):
                widget = _checkbox(not expected)
                log, _ = _logger_with_capture()

                restore_widget_value(widget, saved, label="enable_downsample_checkbox", log=log)

                self.assertIs(widget.value, expected)

    def test_a_meaningless_checkbox_value_falls_back_rather_than_guessing(self) -> None:
        widget = _checkbox(True)
        log, captured = _logger_with_capture()

        restore_widget_value(widget, "perhaps", label="enable_downsample_checkbox", log=log)

        self.assertIs(widget.value, False)  # the trait default, not a coin flip
        self.assertEqual(len(captured.messages), 1)

    def test_a_value_outside_a_range_is_clamped_into_it(self) -> None:
        widget = _int_text(5, minimum=0, maximum=10)
        log, _ = _logger_with_capture()

        restore_widget_value(widget, 999, label="pixel_size_inttext", log=log)

        self.assertEqual(widget.value, 10)

    def test_a_value_no_rung_accepts_leaves_the_widget_as_it_was(self) -> None:
        widget = _Strict("#ff0000", default="black", cast=str,
                         validator=lambda v: v.startswith("#") or v == "black")
        log, captured = _logger_with_capture()

        restore_widget_value(widget, "not-a-colour", label="mask_color", log=log)

        self.assertEqual(widget.value, "black")  # the default, never a crash
        self.assertEqual(len(captured.messages), 1)

    def test_a_combobox_keeps_a_value_its_options_do_not_contain(self) -> None:
        """The data-mapping keys are deliberately unconstrained (#142)."""
        widget = _combobox("centroid-1", ["centroid-1"])
        log, captured = _logger_with_capture()

        exact = restore_widget_value(widget, "my_own_x_column", label="x_key", log=log)

        self.assertTrue(exact)
        self.assertEqual(widget.value, "my_own_x_column")
        self.assertEqual(captured.messages, [])

    def test_nothing_is_attempted_for_a_widget_that_is_not_there(self) -> None:
        self.assertFalse(restore_widget_value(None, "anything", label="gone"))


class PerChannelMappingTests(unittest.TestCase):
    """The colour/contrast dictionaries, keyed by channel name."""

    def test_a_channel_that_left_the_dataset_is_skipped_not_raised_on(self) -> None:
        mapping = {"CD3": _int_text(0), "CD8": _int_text(0)}
        log, captured = _logger_with_capture()

        restored = restore_widget_values(
            mapping, {"CD3": 5, "CD_removed": 9}, label="channel_min", log=log
        )

        self.assertEqual(restored, 1)
        self.assertEqual(mapping["CD3"].value, 5)
        self.assertEqual(captured.messages, [])

    def test_a_saved_entry_that_is_not_a_mapping_is_reported_not_raised(self) -> None:
        log, captured = _logger_with_capture()

        restored = restore_widget_values({}, ["not", "a", "mapping"], label="channel_min", log=log)

        self.assertEqual(restored, 0)
        self.assertEqual(len(captured.messages), 1)


class LoadWidgetStatesTests(unittest.TestCase):
    """The file-level guards, exercised through the real method."""

    def _viewer(self):
        from ueler.viewer.main_viewer import ImageMaskViewer

        class _Viewer:
            load_widget_states = ImageMaskViewer.load_widget_states
            _restore_state_entry = ImageMaskViewer._restore_state_entry

            def __init__(self):
                self._debug = False
                self.marker_sets = {}
                self.mask_names = []
                self.current_downsample_factor = 1
                self.ui_component = type("UI", (), {})()
                self.ui_component.control_sections = None
                self.calls = []

            def update_marker_set_dropdown(self):
                self.calls.append("update_marker_set_dropdown")

            def update_controls(self, _):
                self.calls.append("update_controls")

            def update_display(self, _):
                raise RuntimeError("the saved FOV is not on disk any more")

            def update_keys(self, _):
                self.calls.append("update_keys")

            def inform_plugins(self, message):
                self.calls.append(f"inform:{message}")

        return _Viewer()

    def _write(self, folder, payload):
        path = os.path.join(folder, "widget_states.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload)
        return path

    def test_a_truncated_file_is_ignored_rather_than_raised_on(self) -> None:
        """A session killed mid-write leaves exactly this."""
        viewer = self._viewer()
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, '{"marker_sets": {"a": ')

            with self.assertLogs("ueler.viewer.main_viewer", level="WARNING") as logs:
                viewer.load_widget_states(path)

        self.assertIn("could not be read", "\n".join(logs.output))

        self.assertEqual(viewer.calls, [])  # returned early, did not crash

    def test_a_file_that_is_not_a_settings_object_is_ignored(self) -> None:
        viewer = self._viewer()
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, "[1, 2, 3]")

            with self.assertLogs("ueler.viewer.main_viewer", level="WARNING"):
                viewer.load_widget_states(path)

        self.assertEqual(viewer.calls, [])

    def test_a_missing_file_is_not_an_error(self) -> None:
        self._viewer().load_widget_states("/no/such/widget_states.json")

    def test_a_refresh_step_that_fails_does_not_stop_the_others(self) -> None:
        """``update_display`` raising is the shape of the reported crash."""
        viewer = self._viewer()
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, json.dumps({"marker_sets": {"set_a": ["CD3"]}}))

            with self.assertLogs("ueler.viewer.main_viewer", level="WARNING") as logs:
                viewer.load_widget_states(path)

        self.assertIn("update_display failed", "\n".join(logs.output))

        self.assertEqual(viewer.marker_sets, {"set_a": ["CD3"]})
        self.assertIn("update_keys", viewer.calls)
        self.assertIn("inform:refresh_roi_table", viewer.calls)

    def test_saved_attributes_of_the_wrong_type_are_rejected(self) -> None:
        viewer = self._viewer()
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, json.dumps({"marker_sets": "not a mapping", "mask_names": 7}))

            with self.assertLogs("ueler.viewer.main_viewer", level="WARNING"):
                viewer.load_widget_states(path)

        self.assertEqual(viewer.marker_sets, {})
        self.assertEqual(viewer.mask_names, [])


if __name__ == "__main__":
    unittest.main()
