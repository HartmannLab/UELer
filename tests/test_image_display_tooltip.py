from types import SimpleNamespace
import types
import sys
import unittest

import numpy as np


if "ueler.image_utils" not in sys.modules:
    image_utils_stub = types.ModuleType("ueler.image_utils")

    def _calculate_downsample_factor(*_args, **_kwargs):
        return 1

    class _EdgeComputation:
        def __init__(self, mask):
            self._mask = mask

        def compute(self):
            return self._mask

    def _generate_edges(mask, **_kwargs):
        return _EdgeComputation(mask)

    def _get_axis_limits_with_padding(*_args, **_kwargs):
        return (0, 0, 0, 0, 0, 0, 0, 0)

    image_utils_stub.calculate_downsample_factor = _calculate_downsample_factor  # type: ignore[attr-defined]
    image_utils_stub.generate_edges = _generate_edges  # type: ignore[attr-defined]
    image_utils_stub.get_axis_limits_with_padding = _get_axis_limits_with_padding  # type: ignore[attr-defined]
    sys.modules["ueler.image_utils"] = image_utils_stub


if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")


from ueler.viewer.image_display import ImageDisplay
from ueler.viewer.tooltip_utils import format_tooltip_value, resolve_cell_record


class SimpleMask(list):
    def __and__(self, other):  # pragma: no cover - deterministic helper
        return SimpleMask([bool(a) and bool(b) for a, b in zip(self, other)])


class SimpleColumn(list):
    def __eq__(self, other):  # pragma: no cover - deterministic helper
        return SimpleMask([value == other for value in self])


class SimpleRow(dict):
    def __init__(self, data):
        super().__init__(data)
        self.index = tuple(data.keys())


class _SimpleLoc:
    def __init__(self, table):
        self._table = table

    def __getitem__(self, mask):
        if not isinstance(mask, SimpleMask):
            raise TypeError("Mask must be a SimpleMask instance")
        rows = [row for row, flag in zip(self._table._rows, mask) if flag]
        return SimpleCellTable(rows)


class _SimpleILoc:
    def __init__(self, table):
        self._table = table

    def __getitem__(self, index):
        if isinstance(index, int):
            return SimpleRow(self._table._rows[index])
        raise NotImplementedError("Only integer iloc supported in test stub")


class SimpleCellTable:
    def __init__(self, rows):
        self._rows = list(rows)
        self.columns = list(rows[0].keys()) if rows else []
        self.loc = _SimpleLoc(self)
        self._iloc = _SimpleILoc(self)

    def __contains__(self, column):
        return column in self.columns

    def __getitem__(self, column):
        return SimpleColumn([row.get(column) for row in self._rows])

    @property
    def empty(self):
        return not self._rows

    @property
    def iloc(self):
        return self._iloc

    @property
    def rows(self):
        return list(self._rows)


class TooltipFormattingTests(unittest.TestCase):
    def test_regular_value_uses_fixed_point(self) -> None:
        self.assertEqual(format_tooltip_value(12.345), "12.35")

    def test_small_value_uses_scientific_notation(self) -> None:
        self.assertEqual(format_tooltip_value(1.12e-6), "1.12e-06")

    def test_zero_value_stays_zero(self) -> None:
        self.assertEqual(format_tooltip_value(0.0), "0.00")

    def test_negative_small_value_preserves_sign(self) -> None:
        self.assertEqual(format_tooltip_value(-3.0e-5), "-3.00e-05")


if __name__ == "__main__":
    unittest.main()


class ResolveCellRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.default_table = SimpleCellTable(
            [
                {"fov": "A", "label": 1, "mask": "cell", "CD3": 0.1, "status": "idle"},
                {"fov": "A", "label": 2, "mask": "cell", "CD3": 0.2, "status": "active"},
                {"fov": "B", "label": 1, "mask": "nucleus", "CD3": 0.3, "status": "off"},
            ]
        )

    def test_resolves_using_default_keys(self) -> None:
        record = resolve_cell_record(
            self.default_table,
            fov_value="A",
            mask_name="cell",
            mask_id=2,
            fov_key="fov",
            label_key="label",
            mask_key="mask",
        )
        self.assertIsNotNone(record)
        self.assertEqual(record["status"], "active")

    def test_resolves_when_mask_column_missing(self) -> None:
        df = SimpleCellTable(
            [{key: value for key, value in row.items() if key != "mask"} for row in self.default_table.rows]
        )
        record = resolve_cell_record(
            df,
            fov_value="A",
            mask_name="cell",
            mask_id=1,
            fov_key="fov",
            label_key="label",
            mask_key="mask",
        )
        self.assertIsNotNone(record)
        self.assertEqual(record["label"], 1)

    def test_custom_key_names(self) -> None:
        df = SimpleCellTable(
            [
                {"FOV_ID": "slide-1", "CellID": 42, "mask_name": "cell", "CD4": 5.5},
            ]
        )
        record = resolve_cell_record(
            df,
            fov_value="slide-1",
            mask_name="cell",
            mask_id=42,
            fov_key="FOV_ID",
            label_key="CellID",
            mask_key="mask_name",
        )
        self.assertIsNotNone(record)
        self.assertAlmostEqual(record["CD4"], 5.5)

    def test_returns_none_when_no_match(self) -> None:
        record = resolve_cell_record(
            self.default_table,
            fov_value="Z",
            mask_name="cell",
            mask_id=99,
            fov_key="fov",
            label_key="label",
        )
        self.assertIsNone(record)


class FakeMaskArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = data
        self.shape = data.shape

    class _CellValue:
        def __init__(self, value: int) -> None:
            self._value = value

        def compute(self) -> int:
            return int(self._value)

    def __getitem__(self, index):  # type: ignore[override]
        value = self._data[index]
        return self._CellValue(int(value))


class ImageDisplayTooltipIntegrationTests(unittest.TestCase):
    def test_process_hover_event_uses_configured_keys(self) -> None:
        cell_table = SimpleCellTable(
            [
                {
                    "FOV_ID": "FOV-1",
                    "CellID": 11,
                    "mask_name": "cell",
                    "CD3": 1.234,
                    "CellType": "T cell",
                }
            ]
        )

        mask_data = np.zeros((4, 4), dtype=int)
        mask_data[1, 1] = 11

        image_display = ImageDisplay(width=4, height=4)
        viewer = SimpleNamespace()
        viewer.cell_table = cell_table
        viewer.fov_key = "FOV_ID"
        viewer.label_key = "CellID"
        viewer.mask_key = "mask_name"
        viewer.current_label_masks = {"cell": FakeMaskArray(mask_data)}
        viewer.current_downsample_factor = 1
        viewer.selected_tooltip_labels = ["CellType"]
        viewer.ui_component = SimpleNamespace(
            image_selector=SimpleNamespace(value="FOV-1"),
            channel_selector=SimpleNamespace(value=("CD3",)),
        )

        def _resolve_hit(_x, _y):
            cell_row = resolve_cell_record(
                viewer.cell_table,
                fov_value="FOV-1",
                mask_name="cell",
                mask_id=11,
                fov_key=viewer.fov_key,
                label_key=viewer.label_key,
                mask_key=viewer.mask_key,
            )
            return SimpleNamespace(
                fov_name="FOV-1",
                mask_name="cell",
                mask_id=11,
                local_x_px=1.0,
                local_y_px=1.0,
                map_id=None,
                cell_record=cell_row,
            )

        viewer.resolve_mask_hit_at_viewport = _resolve_hit

        image_display.main_viewer = viewer
        image_display.last_hover_event = SimpleNamespace(xdata=1.0, ydata=1.0)

        image_display.process_hover_event()

        self.assertTrue(image_display.mask_id_annotation.get_visible())
        expected_text = "\n".join(["cell ID: 11", "CD3: 1.23", "CellType: T cell"])
        self.assertEqual(image_display.mask_id_annotation.get_text(), expected_text)
