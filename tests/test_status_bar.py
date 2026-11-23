"""Status bar updates: one batched widget message, and ready once the viewer is shown."""

import inspect
import unittest
from types import SimpleNamespace
import contextlib

from ueler.viewer.decorators import set_status_ready, update_status_bar

_IMAGES = {"processing": b"GIF89a-spinner", "ready": b"\x89PNG-check"}


class _RecordingWidget:
    """Records each trait assignment and whether it happened inside ``hold_sync``."""

    def __init__(self):
        object.__setattr__(self, "assignments", [])
        object.__setattr__(self, "syncs", 0)
        object.__setattr__(self, "_holding", False)

    @contextlib.contextmanager
    def hold_sync(self):
        object.__setattr__(self, "_holding", True)
        try:
            yield
        finally:
            object.__setattr__(self, "_holding", False)
            object.__setattr__(self, "syncs", self.syncs + 1)

    def __setattr__(self, name, value):
        self.assignments.append((name, self._holding))
        object.__setattr__(self, name, value)


def _viewer(status_bar):
    return SimpleNamespace(ui_component=SimpleNamespace(status_bar=status_bar), _status_image=_IMAGES)


class StatusBarUpdateTests(unittest.TestCase):
    def test_state_change_is_sent_as_one_message(self):
        bar = _RecordingWidget()
        set_status_ready(_viewer(bar))
        # All four traits change while hold_sync is held, so ipywidgets sends
        # them to the browser as one update; size and format before the image.
        self.assertEqual(bar.syncs, 1)
        self.assertEqual(bar.assignments, [("width", True), ("height", True), ("format", True), ("value", True)])
        self.assertEqual((bar.value, bar.format, bar.width, bar.height), (_IMAGES["ready"], "png", 30, 30))

    def test_plain_object_without_hold_sync_is_still_updated(self):
        bar = SimpleNamespace(value=None, format=None, width=None, height=None)
        set_status_ready(_viewer(bar))
        self.assertEqual((bar.value, bar.format, bar.width, bar.height), (_IMAGES["ready"], "png", 30, 30))

    def test_decorator_ends_ready_even_when_the_call_fails(self):
        bar = SimpleNamespace(value=None, format=None, width=None, height=None)
        owner = _viewer(bar)

        @update_status_bar
        def work(_self):
            self.assertEqual(bar.format, "gif")
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            work(owner)
        self.assertEqual(bar.format, "png")


class DisplayEndsReadyTests(unittest.TestCase):
    def test_display_ui_switches_the_initial_spinner_to_ready(self):
        # The widget is created showing the spinner, and only view-change
        # renders flip it. display_ui is the path shared by
        # run_viewer(auto_display=True) and viewer.display(), so it must clear
        # the spinner once the widget tree is shown.
        from ueler.viewer import ui_components

        source = inspect.getsource(ui_components.display_ui)
        shown = source.rindex("display(root)")
        self.assertGreater(source.find("set_status_ready(viewer)", shown), shown)

    def test_run_viewer_auto_display_goes_through_display_ui(self):
        from ueler import runner

        display_fn, _ = runner._load_display_helpers()
        from ueler.viewer.ui_components import display_ui

        self.assertIs(display_fn, display_ui)


class LoadingShowsSpinnerTests(unittest.TestCase):
    def test_fov_switch_and_channel_change_show_the_spinner(self):
        # Draws no longer toggle the status, so the handlers that load and
        # render must show it themselves.
        from ueler.viewer.main_viewer import ImageMaskViewer

        for name in ("on_image_change", "on_channel_selection_change"):
            with self.subTest(handler=name):
                handler = getattr(ImageMaskViewer, name)
                self.assertTrue(hasattr(handler, "__wrapped__"), f"{name} is not wrapped by update_status_bar")
                self.assertEqual(handler.__name__, name)


if __name__ == "__main__":
    unittest.main()
