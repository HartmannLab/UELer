"""The confirmation modal in front of the plugin deletes (#139 reply 1).

Every action covered here removes something from disk, so each test asks the
same three questions: does clicking only *ask*, does confirming do the work, and
does the absence of a dialog *refuse* rather than proceed? That last one is the
point of the exercise -- these deletions outlive the session, so an unconfirmed
fallback would destroy work the user cannot get back.

The handlers are driven as unbound methods over narrow stubs. Building real
plugins here would drag in their whole construction path without testing more of
the guard, and the deletion mechanics themselves are covered by each plugin's own
suite (see ``test_export_fovs_mask_customization`` for the one end-to-end case
that really unlinks a file through the dialog).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from ueler.viewer.confirm_dialog import ConfirmDialog, confirm_via
from ueler.viewer.main_viewer import ImageMaskViewer
from ueler.viewer.plugin.cell_annotation import CellAnnotationPlugin
from ueler.viewer.plugin.mask_painter import MaskPainterDisplay
from ueler.viewer.plugin.plugin_base import PluginBase
from ueler.viewer.plugin.roi_manager_plugin import ROIManagerPlugin

from tests.confirm_support import answer, attach_dialog, without_dialog


def _plugin_stub(**attributes):
    """A real ``PluginBase`` carrying only the attributes the handler touches.

    Constructed with ``__new__`` to skip the base ``__init__``, which wants a
    viewer, a width and a height that none of these tests have a use for. Using a
    real instance rather than a namespace means ``confirm`` and ``_confirm_host``
    are the shipped implementations, so the chain under test is the real one.
    """
    stub = PluginBase.__new__(PluginBase)
    for name, value in attributes.items():
        setattr(stub, name, value)
    return stub


class ConfirmViaTests(unittest.TestCase):
    """The one definition of where the dialog lives."""

    def test_no_dialog_means_the_question_was_not_asked(self) -> None:
        self.assertFalse(confirm_via(SimpleNamespace(), "gone?", lambda: None))

    def test_a_mounted_dialog_is_asked(self) -> None:
        dialog = ConfirmDialog()
        ui_component = SimpleNamespace(confirm_dialog=dialog)

        self.assertTrue(confirm_via(ui_component, "gone?", lambda: None, title="T"))
        self.assertTrue(dialog.is_open)
        self.assertEqual(dialog.title_label.value, "T")

    def test_a_second_question_counts_as_asked(self) -> None:
        """The open dialog's scrim covers the button, so there is nothing to report."""
        dialog = ConfirmDialog()
        ui_component = SimpleNamespace(confirm_dialog=dialog)
        confirm_via(ui_component, "first", lambda: None)

        self.assertTrue(confirm_via(ui_component, "second", lambda: None))
        self.assertEqual(dialog.message_label.value, "first")

    def test_the_viewer_method_delegates_here(self) -> None:
        dialog = ConfirmDialog()
        viewer = SimpleNamespace(ui_component=SimpleNamespace(confirm_dialog=dialog))

        self.assertTrue(ImageMaskViewer.confirm(viewer, "gone?", lambda: None))
        self.assertTrue(dialog.is_open)


class PluginBaseConfirmTests(unittest.TestCase):
    """PluginBase.confirm exists so no plugin repeats the lookup or its fallback."""

    def test_main_viewer_is_preferred(self) -> None:
        host = SimpleNamespace(confirm=MagicMock(return_value=True))
        other = SimpleNamespace(confirm=MagicMock(return_value=True))
        plugin = _plugin_stub(main_viewer=host, viewer=other)

        self.assertTrue(plugin.confirm("gone?", lambda: None))
        host.confirm.assert_called_once()
        other.confirm.assert_not_called()

    def test_viewer_is_the_fallback_name(self) -> None:
        """PluginBase stores ``viewer``; concrete plugins add ``main_viewer``."""
        host = SimpleNamespace(confirm=MagicMock(return_value=True))

        self.assertTrue(_plugin_stub(viewer=host).confirm("?", lambda: None))
        host.confirm.assert_called_once()

    def test_a_host_that_cannot_ask_reports_false(self) -> None:
        self.assertFalse(_plugin_stub(viewer=object()).confirm("?", lambda: None))

    def test_no_host_at_all_reports_false(self) -> None:
        self.assertFalse(_plugin_stub().confirm("?", lambda: None))


class MaskPainterColorSetDeleteTests(unittest.TestCase):
    def _stub(self, *, with_dialog=True):
        stub = _plugin_stub(
            main_viewer=SimpleNamespace(),
            ui_component=SimpleNamespace(saved_sets_dropdown=SimpleNamespace(value="Panel A")),
            _get_selected_registry_record=MagicMock(return_value={"path": "/tmp/a.json"}),
            _delete_saved_color_set_confirmed=MagicMock(),
            _log=MagicMock(),
        )
        dialog = attach_dialog(stub.main_viewer) if with_dialog else None
        if not with_dialog:
            without_dialog(stub.main_viewer)
        return stub, dialog

    def test_nothing_selected_asks_nothing(self) -> None:
        stub, dialog = self._stub()
        stub._get_selected_registry_record.return_value = None

        MaskPainterDisplay.delete_saved_color_set(stub, None)

        self.assertFalse(dialog.is_open)
        stub._delete_saved_color_set_confirmed.assert_not_called()

    def test_clicking_only_asks_and_names_the_set(self) -> None:
        stub, dialog = self._stub()

        MaskPainterDisplay.delete_saved_color_set(stub, None)

        self.assertTrue(dialog.is_open)
        self.assertIn("Panel A", dialog.message_label.value)
        stub._delete_saved_color_set_confirmed.assert_not_called()

    def test_confirming_deletes(self) -> None:
        stub, dialog = self._stub()
        MaskPainterDisplay.delete_saved_color_set(stub, None)

        answer(dialog, "confirm")

        stub._delete_saved_color_set_confirmed.assert_called_once()
        self.assertEqual(stub._delete_saved_color_set_confirmed.call_args[0][0], "Panel A")

    def test_cancelling_keeps_the_set(self) -> None:
        stub, dialog = self._stub()
        MaskPainterDisplay.delete_saved_color_set(stub, None)

        answer(dialog, "cancel")

        stub._delete_saved_color_set_confirmed.assert_not_called()
        self.assertFalse(dialog.is_open)

    def test_without_a_dialog_it_refuses(self) -> None:
        stub, _ = self._stub(with_dialog=False)

        MaskPainterDisplay.delete_saved_color_set(stub, None)

        stub._delete_saved_color_set_confirmed.assert_not_called()
        self.assertIn("Cannot delete", stub._log.call_args[0][0])

    def test_a_set_name_with_markup_is_escaped(self) -> None:
        stub, dialog = self._stub()
        stub.ui_component.saved_sets_dropdown.value = "<script>x</script>"

        MaskPainterDisplay.delete_saved_color_set(stub, None)

        self.assertNotIn("<script>", dialog.message_label.value)
        self.assertIn("&lt;script&gt;", dialog.message_label.value)


class CheckpointDeleteTests(unittest.TestCase):
    def _stub(self, *, with_dialog=True, entries=None):
        store = SimpleNamespace(
            list_checkpoints=MagicMock(
                return_value=entries
                if entries is not None
                else [{"id": "ck1", "step_id": "normalise", "description": "d"}]
            )
        )
        stub = _plugin_stub(
            main_viewer=SimpleNamespace(),
            tree_widget=SimpleNamespace(selected_id="ck1"),
            _store=store,
            _delete_checkpoint_confirmed=MagicMock(),
            _set_status=MagicMock(),
        )
        # The label lookup is part of what is under test, so use the real one.
        stub._checkpoint_label = (
            lambda checkpoint_id: CellAnnotationPlugin._checkpoint_label(stub, checkpoint_id)
        )
        dialog = attach_dialog(stub.main_viewer) if with_dialog else None
        if not with_dialog:
            without_dialog(stub.main_viewer)
        return stub, dialog

    def test_nothing_selected_asks_nothing(self) -> None:
        stub, dialog = self._stub()
        stub.tree_widget.selected_id = ""

        CellAnnotationPlugin._on_delete_button(stub)

        self.assertFalse(dialog.is_open)
        stub._delete_checkpoint_confirmed.assert_not_called()

    def test_clicking_only_asks_and_uses_the_step_id(self) -> None:
        stub, dialog = self._stub()

        CellAnnotationPlugin._on_delete_button(stub)

        self.assertTrue(dialog.is_open)
        self.assertIn("normalise", dialog.message_label.value)
        stub._delete_checkpoint_confirmed.assert_not_called()

    def test_an_unnamed_checkpoint_falls_back_to_its_id(self) -> None:
        stub, dialog = self._stub(entries=[{"id": "ck1"}])

        CellAnnotationPlugin._on_delete_button(stub)

        self.assertIn("ck1", dialog.message_label.value)

    def test_confirming_deletes_the_selected_id(self) -> None:
        stub, dialog = self._stub()
        CellAnnotationPlugin._on_delete_button(stub)

        answer(dialog, "confirm")

        stub._delete_checkpoint_confirmed.assert_called_once_with("ck1")

    def test_cancelling_keeps_the_checkpoint(self) -> None:
        stub, dialog = self._stub()
        CellAnnotationPlugin._on_delete_button(stub)

        answer(dialog, "cancel")

        stub._delete_checkpoint_confirmed.assert_not_called()

    def test_without_a_dialog_it_refuses(self) -> None:
        stub, _ = self._stub(with_dialog=False)

        CellAnnotationPlugin._on_delete_button(stub)

        stub._delete_checkpoint_confirmed.assert_not_called()
        self.assertIn("Cannot delete", stub._set_status.call_args[0][0])


class RoiDeleteTests(unittest.TestCase):
    def _stub(self, *, with_dialog=True, record=None):
        main_viewer = SimpleNamespace(
            roi_manager=SimpleNamespace(
                get_roi=MagicMock(
                    return_value=record if record is not None else {"name": "Tumour core"}
                )
            )
        )
        stub = _plugin_stub(
            main_viewer=main_viewer,
            _selected_roi_id="roi-7",
            _delete_roi_confirmed=MagicMock(),
            set_status=MagicMock(),
        )
        dialog = attach_dialog(main_viewer) if with_dialog else None
        if not with_dialog:
            without_dialog(main_viewer)
        return stub, dialog

    def test_nothing_selected_asks_nothing(self) -> None:
        stub, dialog = self._stub()
        stub._selected_roi_id = None

        ROIManagerPlugin._delete_selected_roi(stub, None)

        self.assertFalse(dialog.is_open)
        stub._delete_roi_confirmed.assert_not_called()

    def test_clicking_only_asks_and_names_the_roi(self) -> None:
        stub, dialog = self._stub()

        ROIManagerPlugin._delete_selected_roi(stub, None)

        self.assertTrue(dialog.is_open)
        self.assertIn("Tumour core", dialog.message_label.value)
        stub._delete_roi_confirmed.assert_not_called()

    def test_an_unnamed_roi_falls_back_to_its_id(self) -> None:
        stub, dialog = self._stub(record={"name": "   "})

        ROIManagerPlugin._delete_selected_roi(stub, None)

        self.assertIn("roi-7", dialog.message_label.value)

    def test_confirming_deletes_the_roi_that_was_shown(self) -> None:
        stub, dialog = self._stub()
        ROIManagerPlugin._delete_selected_roi(stub, None)
        # The selection moves on while the dialog is up.
        stub._selected_roi_id = "roi-9"

        answer(dialog, "confirm")

        stub._delete_roi_confirmed.assert_called_once_with("roi-7")

    def test_cancelling_keeps_the_roi(self) -> None:
        stub, dialog = self._stub()
        ROIManagerPlugin._delete_selected_roi(stub, None)

        answer(dialog, "cancel")

        stub._delete_roi_confirmed.assert_not_called()

    def test_without_a_dialog_it_refuses(self) -> None:
        """delete_roi writes the CSV immediately, so refusing is the safe answer."""
        stub, _ = self._stub(with_dialog=False)

        ROIManagerPlugin._delete_selected_roi(stub, None)

        stub._delete_roi_confirmed.assert_not_called()
        self.assertIn("Cannot delete", stub.set_status.call_args[0][0])


class AnnotationPaletteDeleteTests(unittest.TestCase):
    def _stub(self, *, with_dialog=True):
        viewer = SimpleNamespace(
            ui_component=SimpleNamespace(
                annotation_palette_saved_sets_dropdown=SimpleNamespace(value="Immune")
            ),
            _get_selected_annotation_palette_record=MagicMock(
                return_value={"path": "/tmp/p.json"}
            ),
            _delete_saved_annotation_palette_confirmed=MagicMock(),
            _log_annotation_palette=MagicMock(),
        )
        viewer.confirm = lambda *args, **kwargs: ImageMaskViewer.confirm(
            viewer, *args, **kwargs
        )
        dialog = attach_dialog(viewer) if with_dialog else None
        if not with_dialog:
            without_dialog(viewer)
        return viewer, dialog

    def test_nothing_selected_asks_nothing(self) -> None:
        viewer, dialog = self._stub()
        viewer._get_selected_annotation_palette_record.return_value = None

        ImageMaskViewer.delete_saved_annotation_palette(viewer, None)

        self.assertFalse(dialog.is_open)
        viewer._delete_saved_annotation_palette_confirmed.assert_not_called()

    def test_clicking_only_asks_and_names_the_palette(self) -> None:
        viewer, dialog = self._stub()

        ImageMaskViewer.delete_saved_annotation_palette(viewer, None)

        self.assertTrue(dialog.is_open)
        self.assertIn("Immune", dialog.message_label.value)
        viewer._delete_saved_annotation_palette_confirmed.assert_not_called()

    def test_confirming_deletes_the_palette_that_was_shown(self) -> None:
        viewer, dialog = self._stub()
        ImageMaskViewer.delete_saved_annotation_palette(viewer, None)
        # The dropdown moves on while the dialog is up.
        viewer.ui_component.annotation_palette_saved_sets_dropdown.value = "Stroma"

        answer(dialog, "confirm")

        viewer._delete_saved_annotation_palette_confirmed.assert_called_once()
        self.assertEqual(
            viewer._delete_saved_annotation_palette_confirmed.call_args[0][0], "Immune"
        )

    def test_cancelling_keeps_the_palette(self) -> None:
        viewer, dialog = self._stub()
        ImageMaskViewer.delete_saved_annotation_palette(viewer, None)

        answer(dialog, "cancel")

        viewer._delete_saved_annotation_palette_confirmed.assert_not_called()

    def test_without_a_dialog_it_refuses(self) -> None:
        viewer, _ = self._stub(with_dialog=False)

        ImageMaskViewer.delete_saved_annotation_palette(viewer, None)

        viewer._delete_saved_annotation_palette_confirmed.assert_not_called()
        self.assertIn("Cannot delete", viewer._log_annotation_palette.call_args[0][0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
