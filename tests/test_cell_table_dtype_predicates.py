"""Cell-table column dtype classification across numpy and pandas dtypes.

``ueler.viewer.plugin.mask_painter.apply_colors_to_masks`` converts the class
labels it gets from the widgets (always strings) to the dtype of the identifier
column, so that ``cell_table[identifier] == value`` actually matches rows.  It
used to ask ``np.issubdtype``, which raises ``TypeError`` on every pandas
*extension* dtype -- nullable ``Int64``/``Float64``, ``category``, and the
``StringDtype`` that pandas 3 gives object string columns by default.

That surfaced as 19 errors on the CI Python 3.12 leg, where pip resolved a pandas
that had already switched its string default.  It is not a Python 3.12 problem:
the same failure reproduces on 3.10 with ``pd.options.future.infer_string = True``,
and the ``category`` / nullable cases below fail on every supported pandas.
"""

import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ueler.cell_table import is_float_column_dtype, is_integer_column_dtype
from ueler.rendering import clear_cell_colors, get_cell_color


def _pandas_3_string_dtype():
    """The pandas-3 default string dtype, or ``None`` on older pandas.

    Deliberately not a ``skipUnless``: CI runs the suite with ``--max-skips 0``,
    so an unavailable dtype drops out of the parameter list instead of turning
    into a silently skipped test.
    """

    try:
        return pd.StringDtype(na_value=np.nan)
    except TypeError:  # pragma: no cover - pandas < 2.3
        return None


class ColumnDtypePredicateTests(unittest.TestCase):
    """The predicates must classify numpy and pandas dtypes alike."""

    def test_integer_dtypes(self):
        for dtype in (np.dtype("int64"), np.dtype("uint8"), pd.Int64Dtype()):
            with self.subTest(dtype=dtype):
                self.assertTrue(is_integer_column_dtype(dtype))
                self.assertFalse(is_float_column_dtype(dtype))

    def test_float_dtypes(self):
        for dtype in (np.dtype("float32"), np.dtype("float64"), pd.Float64Dtype()):
            with self.subTest(dtype=dtype):
                self.assertTrue(is_float_column_dtype(dtype))
                self.assertFalse(is_integer_column_dtype(dtype))

    def test_string_like_dtypes_are_neither(self):
        dtypes = [np.dtype("O"), pd.CategoricalDtype(["a", "b"]), pd.StringDtype()]
        pandas_3_strings = _pandas_3_string_dtype()
        if pandas_3_strings is not None:
            dtypes.append(pandas_3_strings)
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                self.assertFalse(is_integer_column_dtype(dtype))
                self.assertFalse(is_float_column_dtype(dtype))

    def test_categorical_dtypes_are_classified_by_their_categories(self):
        """A ``category`` column over numbers still needs numeric conversion.

        ``pd.Series(pd.Categorical([1, 2])) == "1"`` matches no rows, while
        ``== 1`` matches one, so leaving the label a string would silently select
        nothing rather than raise.
        """

        self.assertTrue(is_integer_column_dtype(pd.CategoricalDtype([1, 2, 3])))
        self.assertFalse(is_float_column_dtype(pd.CategoricalDtype([1, 2, 3])))
        self.assertTrue(is_float_column_dtype(pd.CategoricalDtype([1.5, 2.5])))
        self.assertFalse(is_integer_column_dtype(pd.CategoricalDtype([1.5, 2.5])))

    def test_numpy_alone_cannot_answer(self):
        """Documents why the helpers exist rather than calling numpy directly."""

        for dtype in (pd.Int64Dtype(), pd.CategoricalDtype(["a"]), pd.StringDtype()):
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeError):
                    np.issubdtype(dtype, np.integer)

    def test_unknown_objects_do_not_raise(self):
        for value in (None, object(), "not a dtype"):
            with self.subTest(value=value):
                self.assertFalse(is_integer_column_dtype(value))
                self.assertFalse(is_float_column_dtype(value))


class ExtensionDtypeIdentifierColumnTests(unittest.TestCase):
    """End to end: painting an identifier column held in an extension dtype."""

    def setUp(self):
        clear_cell_colors()

    def tearDown(self):
        clear_cell_colors()

    def _make_viewer(self, cluster_column):
        cell_table = pd.DataFrame(
            {
                "fov": ["FOV_001", "FOV_001", "FOV_002"],
                "label": [1, 2, 3],
                "cluster": cluster_column,
            }
        )
        viewer = types.SimpleNamespace()
        viewer.cell_table = cell_table
        viewer.fov_key = "fov"
        viewer.label_key = "label"
        viewer.mask_key = "cell"
        viewer.base_folder = Path.cwd()

        ui_component = types.SimpleNamespace()
        ui_component.image_selector = types.SimpleNamespace(value="FOV_001")
        viewer.ui_component = ui_component
        viewer.image_display = types.SimpleNamespace(
            set_mask_colors_current_fov=lambda **kwargs: None
        )
        viewer.get_active_fov = lambda: ui_component.image_selector.value
        return viewer

    def _apply_colors(self, cluster_column):
        """Paint clusters 1 and 2 red and green, driving the widget path."""

        import ipywidgets

        from ueler.viewer.plugin.mask_painter import MaskPainterDisplay

        painter = MaskPainterDisplay(self._make_viewer(cluster_column), width=400, height=300)
        painter.ui_component.identifier_dropdown.options = ["cluster"]
        painter.ui_component.identifier_dropdown.value = "cluster"
        painter.current_identifier = "cluster"
        # The widgets hand back strings whatever the column dtype is -- that is
        # the whole reason apply_colors_to_masks has to convert them.
        painter.current_classes = ["1", "2"]
        painter.class_color_controls = {
            "1": ipywidgets.ColorPicker(description="1", value="#FF0000"),
            "2": ipywidgets.ColorPicker(description="2", value="#00FF00"),
        }
        painter.ui_component.sorting_items_tagsinput.allowed_tags = ["1", "2"]
        painter.ui_component.sorting_items_tagsinput.value = ("1", "2")
        painter.selected_classes = ["1", "2"]
        painter.ui_component.show_all_checkbox.value = False

        painter.apply_colors_to_masks(None, notify_cell_gallery=False)

    def _assert_clusters_coloured(self):
        self.assertEqual(get_cell_color("FOV_001", 1), "#FF0000")
        self.assertEqual(get_cell_color("FOV_001", 2), "#00FF00")
        self.assertEqual(get_cell_color("FOV_002", 3), "#FF0000")

    def test_categorical_integer_identifier(self):
        self._apply_colors(pd.Series([1, 2, 1], dtype="category"))
        self._assert_clusters_coloured()

    def test_nullable_integer_identifier(self):
        self._apply_colors(pd.Series([1, 2, 1], dtype=pd.Int64Dtype()))
        self._assert_clusters_coloured()

    def test_plain_integer_identifier_still_works(self):
        self._apply_colors(pd.Series([1, 2, 1], dtype="int64"))
        self._assert_clusters_coloured()


if __name__ == "__main__":
    unittest.main()
