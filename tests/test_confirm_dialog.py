"""Tests for the confirmation modal and the marker-set delete flow (#139)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from ipywidgets import Widget

from ueler.viewer.confirm_dialog import ConfirmDialog
from ueler.viewer.main_viewer import ImageMaskViewer


def _click(button_handler_owner, which):
    """Fire a dialog button the way its ``on_click`` handler would.

    ``tests.bootstrap`` replaces ``ipywidgets`` with a stub whose ``on_click``
    is a no-op and whose buttons have no ``click()``, so a real click cannot be
    simulated. Invoking the handler directly exercises the same code path.
    """
    handler = getattr(button_handler_owner, f"_on_{which}_clicked")
    handler(None)


def _make_delete_stub(marker_sets, selected, *, dialog=None):
    """The narrowest stand-in ``delete_marker_set`` actually touches."""
    ui_component = SimpleNamespace(marker_set_dropdown=SimpleNamespace(value=selected))
    if dialog is not None:
        ui_component.confirm_dialog = dialog
    viewer = SimpleNamespace(
        ui_component=ui_component,
        marker_sets=dict(marker_sets),
        update_marker_set_dropdown=MagicMock(),
    )
    viewer._delete_marker_set_confirmed = (
        lambda name: ImageMaskViewer._delete_marker_set_confirmed(viewer, name)
    )
    return viewer


class ConfirmDialogTests(unittest.TestCase):
    def test_starts_closed_and_hidden(self) -> None:
        dialog = ConfirmDialog()
        self.assertFalse(dialog.is_open)
        self.assertEqual(dialog.view.layout.display, "none")

    def test_ask_opens_and_fills_the_card(self) -> None:
        dialog = ConfirmDialog()
        opened = dialog.ask(
            "gone for good",
            lambda: None,
            title="Delete it?",
            confirm_label="Delete",
            cancel_label="Keep",
        )

        self.assertTrue(opened)
        self.assertTrue(dialog.is_open)
        self.assertNotEqual(dialog.view.layout.display, "none")
        self.assertEqual(dialog.confirm_button.description, "Delete")
        self.assertEqual(dialog.cancel_button.description, "Keep")
        self.assertEqual(dialog.confirm_button.button_style, "danger")

    def test_confirm_runs_the_callback_once_and_closes(self) -> None:
        dialog = ConfirmDialog()
        callback = MagicMock()
        dialog.ask("x", callback)

        _click(dialog, "confirm")

        callback.assert_called_once_with()
        self.assertFalse(dialog.is_open)
        self.assertEqual(dialog.view.layout.display, "none")

        # A second click on the now-closed dialog must not replay the action.
        _click(dialog, "confirm")
        callback.assert_called_once_with()

    def test_cancel_closes_without_running_the_callback(self) -> None:
        dialog = ConfirmDialog()
        callback = MagicMock()
        dialog.ask("x", callback)

        _click(dialog, "cancel")

        callback.assert_not_called()
        self.assertFalse(dialog.is_open)
        self.assertEqual(dialog.view.layout.display, "none")

    def test_second_ask_while_open_is_refused(self) -> None:
        dialog = ConfirmDialog()
        first, second = MagicMock(), MagicMock()

        self.assertTrue(dialog.ask("first", first, title="First"))
        self.assertFalse(dialog.ask("second", second, title="Second"))

        # The first question is still the one on screen, and answering it runs
        # the first action -- a repeated click cannot stack confirmations.
        dialog.confirm()
        first.assert_called_once_with()
        second.assert_not_called()

    def test_a_raising_callback_still_closes_the_dialog(self) -> None:
        dialog = ConfirmDialog()

        def boom():
            raise RuntimeError("no")

        dialog.ask("x", boom)
        with self.assertLogs("ueler.viewer.confirm_dialog", level="ERROR"):
            dialog.confirm()

        self.assertFalse(dialog.is_open)
        self.assertEqual(dialog.view.layout.display, "none")

    def test_confirm_when_closed_is_a_no_op(self) -> None:
        dialog = ConfirmDialog()
        dialog.confirm()  # must not raise
        self.assertFalse(dialog.is_open)

    def test_danger_false_uses_the_non_destructive_style(self) -> None:
        dialog = ConfirmDialog()
        dialog.ask("x", lambda: None, danger=False)
        self.assertEqual(dialog.confirm_button.button_style, "primary")

    def test_escape_name_escapes_user_supplied_text(self) -> None:
        self.assertEqual(ConfirmDialog.escape_name("<b>x</b>"), "&lt;b&gt;x&lt;/b&gt;")

    def test_is_not_a_widget_so_widget_state_saving_skips_it(self) -> None:
        # ``save_widget_states`` persists the ``.value`` of every ``Widget`` on
        # ``ui_component``; the dialog holds its stylesheet in an ``HTML``
        # widget, which has no business in widget_states.json.
        self.assertNotIsInstance(ConfirmDialog(), Widget)

    def test_builds_against_widgets_without_add_class(self) -> None:
        # Every test in this file already runs against ``tests.bootstrap``'s
        # ipywidgets stub, which has no ``add_class``; asserting it keeps the
        # tolerance from being removed as dead code.
        from ipywidgets import Button as _Button

        self.assertFalse(hasattr(_Button(), "add_class"))
        ConfirmDialog()  # must not raise

    def test_the_stylesheet_is_mounted_with_the_view(self) -> None:
        dialog = ConfirmDialog()
        style = dialog.view.children[0]
        self.assertIn("ueler-confirm-scrim", style.value)
        self.assertIn("position: fixed", style.value)


class MarkerSetPanelWiringTests(unittest.TestCase):
    """The panel owns a dialog and no longer owns a confirmation checkbox."""

    @staticmethod
    def _build_ui():
        from ueler.viewer.ui_components import uicomponents

        viewer = MagicMock()
        viewer.available_fovs = ["fov0"]
        viewer._map_descriptors = {}
        viewer.channel_names = ["A", "B"]
        return uicomponents(viewer)

    def test_the_panel_owns_a_confirm_dialog(self) -> None:
        self.assertIsInstance(self._build_ui().confirm_dialog, ConfirmDialog)

    def test_the_confirm_deletion_checkbox_is_gone(self) -> None:
        ui = self._build_ui()
        self.assertFalse(hasattr(ui, "delete_confirmation_checkbox"))
        # Pickers + buttons, with no third row for the checkbox.
        self.assertEqual(len(ui.marker_set_controls_panel.children), 2)


class DeleteMarkerSetTests(unittest.TestCase):
    def test_no_selection_does_nothing(self) -> None:
        dialog = ConfirmDialog()
        viewer = _make_delete_stub({"a": {}}, selected=None, dialog=dialog)

        ImageMaskViewer.delete_marker_set(viewer, None)

        self.assertFalse(dialog.is_open)
        self.assertEqual(set(viewer.marker_sets), {"a"})
        viewer.update_marker_set_dropdown.assert_not_called()

    def test_clicking_delete_only_asks(self) -> None:
        dialog = ConfirmDialog()
        viewer = _make_delete_stub({"a": {}, "b": {}}, selected="a", dialog=dialog)

        ImageMaskViewer.delete_marker_set(viewer, None)

        self.assertTrue(dialog.is_open)
        self.assertEqual(set(viewer.marker_sets), {"a", "b"})
        viewer.update_marker_set_dropdown.assert_not_called()

    def test_the_prompt_names_the_set(self) -> None:
        dialog = ConfirmDialog()
        viewer = _make_delete_stub({"CD8 panel": {}}, selected="CD8 panel", dialog=dialog)

        ImageMaskViewer.delete_marker_set(viewer, None)

        self.assertIn("CD8 panel", dialog.message_label.value)
        self.assertEqual(dialog.confirm_button.description, "Delete")

    def test_a_set_name_with_markup_is_escaped_into_the_prompt(self) -> None:
        dialog = ConfirmDialog()
        viewer = _make_delete_stub({"<b>x": {}}, selected="<b>x", dialog=dialog)

        ImageMaskViewer.delete_marker_set(viewer, None)

        self.assertIn("&lt;b&gt;x", dialog.message_label.value)

    def test_confirming_deletes_and_refreshes_the_dropdown(self) -> None:
        dialog = ConfirmDialog()
        viewer = _make_delete_stub({"a": {}, "b": {}}, selected="a", dialog=dialog)

        ImageMaskViewer.delete_marker_set(viewer, None)
        _click(dialog, "confirm")

        self.assertEqual(set(viewer.marker_sets), {"b"})
        viewer.update_marker_set_dropdown.assert_called_once_with()
        self.assertFalse(dialog.is_open)

    def test_cancelling_keeps_the_set(self) -> None:
        dialog = ConfirmDialog()
        viewer = _make_delete_stub({"a": {}, "b": {}}, selected="a", dialog=dialog)

        ImageMaskViewer.delete_marker_set(viewer, None)
        _click(dialog, "cancel")

        self.assertEqual(set(viewer.marker_sets), {"a", "b"})
        viewer.update_marker_set_dropdown.assert_not_called()

    def test_the_set_shown_is_the_set_deleted_even_if_the_dropdown_moves(self) -> None:
        dialog = ConfirmDialog()
        viewer = _make_delete_stub({"a": {}, "b": {}}, selected="a", dialog=dialog)

        ImageMaskViewer.delete_marker_set(viewer, None)
        viewer.ui_component.marker_set_dropdown.value = "b"
        _click(dialog, "confirm")

        self.assertEqual(set(viewer.marker_sets), {"b"})

    def test_confirming_a_set_that_vanished_meanwhile_is_a_warning(self) -> None:
        dialog = ConfirmDialog()
        viewer = _make_delete_stub({"a": {}}, selected="a", dialog=dialog)

        ImageMaskViewer.delete_marker_set(viewer, None)
        del viewer.marker_sets["a"]
        _click(dialog, "confirm")  # must not raise KeyError

        self.assertEqual(viewer.marker_sets, {})
        viewer.update_marker_set_dropdown.assert_not_called()

    def test_without_a_dialog_the_delete_goes_through_directly(self) -> None:
        # The ipywidgets fallback shim cannot render a modal; a permanently dead
        # Delete button would be worse than an unconfirmed one.
        viewer = _make_delete_stub({"a": {}, "b": {}}, selected="a")

        ImageMaskViewer.delete_marker_set(viewer, None)

        self.assertEqual(set(viewer.marker_sets), {"b"})
        viewer.update_marker_set_dropdown.assert_called_once_with()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
