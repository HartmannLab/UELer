"""Previous / next buttons beside the image selector."""
import unittest
from unittest import mock

from ueler.viewer import ui_components
from ueler.viewer.ui_components import build_selector_stepper, step_selector


class _Observable:
    """A widget whose trait changes call observers, like an ipywidget (the suite stubs those)."""

    def __init__(self, **traits):
        object.__setattr__(self, "_observers", [])
        for name, value in traits.items():
            object.__setattr__(self, name, value)

    def observe(self, handler, names=()):
        names = [names] if isinstance(names, str) else list(names)
        self._observers.append((handler, names))

    def __setattr__(self, name, value):
        old = getattr(self, name, None)
        object.__setattr__(self, name, value)
        if old != value:
            for handler, names in list(self._observers):
                if name in names:
                    handler({"name": name, "old": old, "new": value})


class _Dropdown(_Observable):
    def __init__(self, options, value, disabled=False):
        super().__init__(options=list(options), value=value, disabled=disabled)


class _Button(_Observable):
    def __init__(self, **kwargs):
        super().__init__(disabled=False, _click=None, **kwargs)

    def on_click(self, handler):
        object.__setattr__(self, "_click", handler)

    def click(self):
        if not self.disabled:
            self._click(self)


class ImageStepperTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(ui_components, "Button", _Button)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.selector = _Dropdown(options=["fov_a", "fov_b", "fov_c"], value="fov_a")
        self.previous, self.next = build_selector_stepper(self.selector)

    def test_next_and_previous_move_through_the_list(self):
        self.next.click()
        self.assertEqual(self.selector.value, "fov_b")
        self.next.click()
        self.assertEqual(self.selector.value, "fov_c")
        self.previous.click()
        self.assertEqual(self.selector.value, "fov_b")

    def test_buttons_are_disabled_at_the_ends(self):
        self.assertTrue(self.previous.disabled)
        self.assertFalse(self.next.disabled)
        self.selector.value = "fov_c"
        self.assertFalse(self.previous.disabled)
        self.assertTrue(self.next.disabled)
        self.assertFalse(step_selector(self.selector, 1))
        self.assertEqual(self.selector.value, "fov_c")

    def test_buttons_follow_a_disabled_selector(self):
        self.selector.value = "fov_b"
        self.selector.disabled = True
        self.assertTrue(self.previous.disabled)
        self.assertTrue(self.next.disabled)
        self.assertFalse(step_selector(self.selector, 1))
        self.assertEqual(self.selector.value, "fov_b")
        self.selector.disabled = False
        self.assertFalse(self.previous.disabled)
        self.assertFalse(self.next.disabled)

    def test_step_fires_the_selector_observer(self):
        seen = []
        self.selector.observe(lambda change: seen.append(change["new"]), names="value")
        self.next.click()
        self.assertEqual(seen, ["fov_b"])

    def test_single_option_disables_both(self):
        selector = _Dropdown(options=["only"], value="only")
        previous, nxt = build_selector_stepper(selector)
        self.assertTrue(previous.disabled)
        self.assertTrue(nxt.disabled)

    def test_new_options_resync(self):
        self.selector.value = "fov_c"
        self.selector.options = ["fov_c", "fov_d"]
        self.assertTrue(self.previous.disabled)
        self.assertFalse(self.next.disabled)


if __name__ == "__main__":
    unittest.main()
