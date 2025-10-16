import importlib.util
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd


_ipywidgets = sys.modules.get("ipywidgets")


class _Layout:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Widget:
    def __init__(self, *args, **kwargs):
        children = kwargs.get("children")
        if children is None and args:
            children = args[0]
        self.children = tuple(children or ())
        self.value = kwargs.get("value")
        self.options = kwargs.get("options", [])
        self.allowed_tags = kwargs.get("allowed_tags", [])
        self.layout = kwargs.get("layout", _Layout())
        self.description = kwargs.get("description", "")
        self.tooltip = kwargs.get("tooltip", "")
        self.icon = kwargs.get("icon", "")
        self.button_style = kwargs.get("button_style", "")

    def observe(self, *_, **__):
        return None

    def on_click(self, *_, **__):
        return None

    def clear_output(self, *_, **__):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


if _ipywidgets is None:
    _ipywidgets = types.ModuleType("ipywidgets")
    sys.modules["ipywidgets"] = _ipywidgets

if not hasattr(_ipywidgets, "Layout"):
    setattr(_ipywidgets, "Layout", _Layout)

for _name in [
    "SelectMultiple",
    "FloatSlider",
    "Dropdown",
    "VBox",
    "Output",
    "TagsInput",
    "Checkbox",
    "IntText",
    "Text",
    "Button",
    "HBox",
    "IntSlider",
    "Tab",
    "RadioButtons",
    "HTML",
    "Widget",
]:
    if not hasattr(_ipywidgets, _name):
        setattr(_ipywidgets, _name, _Widget)

if "IPython.display" not in sys.modules:
    display_module = types.ModuleType("IPython.display")

    def _noop_display(*_, **__):
        return None

    display_module.display = _noop_display
    sys.modules["IPython.display"] = display_module

if "seaborn" not in sys.modules:
    seaborn_stub = types.ModuleType("seaborn")

    def _clustermap(*_, **__):
        return SimpleNamespace(
            cax=None,
            ax_heatmap=SimpleNamespace(
                set_yticks=lambda *_: None,
                set_yticklabels=lambda *_: None,
                get_xticklabels=lambda: [],
                set_xticklabels=lambda *_: None,
                get_yticks=lambda: [],
                set_ylabel=lambda *_: None,
                figure=SimpleNamespace(canvas=SimpleNamespace(header_visible=False)),
            ),
            ax_row_colors=None,
            ax_row_dendrogram=None,
            ax_col_dendrogram=None,
            dendrogram_row=None,
            dendrogram_col=None,
            fig=SimpleNamespace(
                canvas=SimpleNamespace(
                    header_visible=False, draw_idle=lambda: None, mpl_connect=lambda *_: None
                )
            ),
        )

    seaborn_stub.clustermap = _clustermap
    seaborn_stub.color_palette = lambda *_, **__: []
    seaborn_stub.set_context = lambda *_, **__: None
    sys.modules["seaborn"] = seaborn_stub

_pandas = sys.modules.get("pandas")


class _Axis(list):
    def tolist(self):
        return list(self)


class _Index(_Axis):
    def __init__(self, data):
        super().__init__(data)

    def get_loc(self, key):
        try:
            return self.index(key)
        except ValueError as exc:  # pragma: no cover - defensive
            raise KeyError(key) from exc

    def take(self, positions):
        return [self[pos] for pos in positions]


class _Series(_Axis):
    def dropna(self):
        return _Series([value for value in self if value is not None])


class _LocIndexer:
    def __init__(self, frame):
        self._frame = frame

    def __getitem__(self, key):
        rows, column = key
        if column not in self._frame._data:
            raise KeyError(column)

        if isinstance(rows, (list, tuple)):
            values = [self._frame._data[column][self._frame._index_map[row]] for row in rows]
            return _Series(values)

        return self._frame._data[column][self._frame._index_map[rows]]


class _DataFrame:
    def __init__(self, data=None, index=None):
        data = data or {}
        if index is None:
            first_column = next(iter(data.values()), [])
            index = list(range(len(first_column)))
        self._index = list(index)
        self._index_map = {idx: pos for pos, idx in enumerate(self._index)}
        self._data = {}
        for column, values in (data or {}).items():
            self._data[column] = list(values)

        self.columns = _Axis(list(data.keys()))
        self.index = _Axis(list(self._index))
        self.loc = _LocIndexer(self)

    def copy(self):  # pragma: no cover - defensive helper
        return _DataFrame({name: list(values) for name, values in self._data.items()}, list(self.index))


def _ensure_pandas_module():
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = _DataFrame
    pandas_stub.Series = _Series
    pandas_stub.Index = _Index
    pandas_stub.isna = lambda value: value is None
    pandas_stub.merge = lambda left, right, on=None, how=None: left
    pandas_stub.unique = lambda values: list(dict.fromkeys(values))
    return pandas_stub


if _pandas is None or getattr(_pandas, "DataFrame", object) is object:
    sys.modules["pandas"] = _ensure_pandas_module()
    _pandas = sys.modules["pandas"]

pd = sys.modules.get("pandas", _pandas)

_HEATMAP_PATH = pathlib.Path(__file__).resolve().parents[1] / "viewer" / "plugin" / "heatmap.py"
_heatmap_spec = importlib.util.spec_from_file_location("heatmap_under_test", _HEATMAP_PATH)
_heatmap_module = importlib.util.module_from_spec(_heatmap_spec)
assert _heatmap_spec.loader is not None
_heatmap_spec.loader.exec_module(_heatmap_module)

Data = _heatmap_module.Data
HeatmapDisplay = _heatmap_module.HeatmapDisplay

heatmap_layers_module = sys.modules.get("viewer.plugin.heatmap_layers")
if heatmap_layers_module is None:
    _HEATMAP_LAYERS_PATH = pathlib.Path(__file__).resolve().parents[1] / "viewer" / "plugin" / "heatmap_layers.py"
    _heatmap_layers_spec = importlib.util.spec_from_file_location(
        "heatmap_layers_under_test", _HEATMAP_LAYERS_PATH
    )
    heatmap_layers_module = importlib.util.module_from_spec(_heatmap_layers_spec)
    assert _heatmap_layers_spec.loader is not None
    _heatmap_layers_spec.loader.exec_module(heatmap_layers_module)

_apply_heatmap_tick_labels = heatmap_layers_module._apply_heatmap_tick_labels


class HeatmapScatterSelectionTests(unittest.TestCase):
    def _make_heatmap(self):
        heatmap = HeatmapDisplay.__new__(HeatmapDisplay)

        cell_table = pd.DataFrame(
            {
                "cluster": ["A", "B", "C", "B"],
            },
            index=[1, 2, 3, 4],
        )

        heatmap.main_viewer = SimpleNamespace(cell_table=cell_table)
        heatmap.ui_component = SimpleNamespace(
            high_level_cluster_dropdown=SimpleNamespace(value="cluster"),
            horizontal_layout_checkbox=SimpleNamespace(value=False),
            chart_checkbox=SimpleNamespace(value=True),
        )
        heatmap.orientation_state = {
            "horizontal": False,
            "view": None,
            "cluster_axis": 0,
            "marker_axis": 1,
            "cluster_index": pd.Index(["A", "B", "C"]),
            "marker_index": pd.Index([]),
            "cluster_order_positions": [1, 0, 2],
            "marker_order_positions": [],
            "cluster_leaves": ["B", "A", "C"],
            "marker_leaves": [],
        }

        heatmap.data = Data()
        heatmap.highlight_row_colors = MagicMock()
        heatmap.plot_heatmap = MagicMock()
        heatmap.heatmap_data = pd.DataFrame(
            {"meta_cluster": [0, 1, 2]}, index=["A", "B", "C"]
        )
        heatmap._reset_selection_cache()
        return heatmap

    def test_selection_updates_highlight_without_replot(self):
        heatmap = self._make_heatmap()

        heatmap.on_selected_indices_change({2, 4})

        heatmap.plot_heatmap.assert_not_called()
        heatmap.highlight_row_colors.assert_called_once_with([0])
        self.assertEqual(heatmap.data.current_clusters["index"].value, [0])

        heatmap.highlight_row_colors.reset_mock()
        heatmap.on_selected_indices_change({4, 2})
        heatmap.highlight_row_colors.assert_not_called()

    def test_empty_selection_clears_highlight(self):
        heatmap = self._make_heatmap()
        heatmap.on_selected_indices_change({2})

        heatmap.highlight_row_colors.reset_mock()
        heatmap.on_selected_indices_change(set())

        heatmap.highlight_row_colors.assert_called_once_with([])
        self.assertEqual(heatmap.data.current_clusters["index"].value, [])

    def test_missing_orientation_triggers_replot(self):
        heatmap = self._make_heatmap()
        heatmap.orientation_state["cluster_index"] = None

        heatmap.on_selected_indices_change({2})

        heatmap.plot_heatmap.assert_called_once()
        heatmap.highlight_row_colors.assert_not_called()

    def test_selection_ignored_when_chart_link_disabled(self):
        heatmap = self._make_heatmap()
        heatmap.ui_component.chart_checkbox.value = False

        heatmap.on_selected_indices_change({2, 4})

        heatmap.highlight_row_colors.assert_not_called()
        heatmap.plot_heatmap.assert_not_called()


class PatchStub:
    def __init__(self, start, end, facecolor, orientation):
        self.start = start
        self.end = end
        self.facecolor = facecolor
        self.orientation = orientation
        self.removed = False

    def remove(self):
        self.removed = True


class AxisStub:
    def __init__(self):
        canvas = SimpleNamespace(draw_idle=lambda: None)
        self.figure = SimpleNamespace(canvas=canvas)
        self.collections = []
        self.vspans = []
        self.hspans = []

    def axvspan(self, start, end, facecolor=None, zorder=None):
        patch = PatchStub(start, end, facecolor, "vertical")
        self.vspans.append((start, end, facecolor, zorder))
        return patch

    def axhspan(self, start, end, facecolor=None, zorder=None):
        patch = PatchStub(start, end, facecolor, "horizontal")
        self.hspans.append((start, end, facecolor, zorder))
        return patch

    def get_children(self):
        return []


class TickAxisStub:
    def __init__(self):
        self.xticks = []
        self.yticks = []
        self.xticklabels = []
        self.yticklabels = []
        self.xlabel = None
        self.ylabel = None
        self.xticklabel_kwargs = {}
        self.yticklabel_kwargs = {}

    def set_xticks(self, ticks):
        self.xticks = list(ticks)

    def set_xticklabels(self, labels, **kwargs):
        self.xticklabels = list(labels)
        self.xticklabel_kwargs = dict(kwargs)

    def set_yticks(self, ticks):
        self.yticks = list(ticks)

    def set_yticklabels(self, labels, **kwargs):
        self.yticklabels = list(labels)
        self.yticklabel_kwargs = dict(kwargs)

    def set_xlabel(self, label):
        self.xlabel = label

    def set_ylabel(self, label):
        self.ylabel = label

    def get_xticklabels(self):
        return []


class HeatmapTickAlignmentTests(unittest.TestCase):
    def test_wide_ticks_centered_on_cells(self):
        axis = TickAxisStub()
        adapter = SimpleNamespace(is_wide=lambda: True)
        cluster_leaves = ["B", "A", "C"]
        marker_leaves = ["CD3", "CD8"]

        _apply_heatmap_tick_labels(adapter, axis, cluster_leaves, marker_leaves, "Cluster")

        self.assertEqual(axis.xticks, [0.5, 1.5, 2.5])
        self.assertEqual(axis.yticks, [0.5, 1.5])
        self.assertEqual(axis.xticklabels, cluster_leaves)
        self.assertEqual(axis.yticklabels, marker_leaves)
        self.assertEqual(axis.xlabel, "Cluster")
        self.assertEqual(axis.xticklabel_kwargs.get("rotation"), 45)
        self.assertEqual(axis.xticklabel_kwargs.get("ha"), "right")

    def test_vertical_ticks_keep_existing_ordering(self):
        axis = TickAxisStub()
        adapter = SimpleNamespace(is_wide=lambda: False)
        cluster_leaves = ["B", "A"]

        _apply_heatmap_tick_labels(adapter, axis, cluster_leaves, [], "Cluster")

        self.assertEqual(axis.yticks, [0, 1])
        self.assertEqual(axis.yticklabels, ["A", "B"])


class OutputStub:
    def __init__(self):
        self.outputs = ()
        self.clear_output_calls = []

    def clear_output(self, wait=False):
        self.clear_output_calls.append(wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class CanvasStub:
    def __init__(self, raise_on_draw=False):
        self.raise_on_draw = raise_on_draw
        self.draw_idle_called = False

    def draw_idle(self):
        if self.raise_on_draw:
            raise RuntimeError("draw failure")
        self.draw_idle_called = True


class HeatmapPatchRenderingTests(unittest.TestCase):
    def _build_display(self, horizontal):
        heatmap = HeatmapDisplay.__new__(HeatmapDisplay)

        axis = AxisStub()
        canvas = SimpleNamespace(header_visible=False, draw_idle=lambda: None, mpl_connect=lambda *_, **__: None)
        g = SimpleNamespace(
            ax_heatmap=SimpleNamespace(),
            ax_col_colors=axis if horizontal else None,
            ax_row_colors=axis if not horizontal else None,
            ax_col_dendrogram=None,
            ax_row_dendrogram=None,
            fig=SimpleNamespace(canvas=canvas),
        )

        heatmap.data = Data()
        heatmap.data.g = g
        heatmap.data.meta_cluster_colors = {1: "red", 2: "blue", 3: "green"}
        heatmap.data.cluster_colors = {"A": "red", "B": "blue"}

        heatmap.ui_component = SimpleNamespace(
            high_level_cluster_dropdown=SimpleNamespace(value="cluster"),
            horizontal_layout_checkbox=SimpleNamespace(value=horizontal),
        )

        heatmap.orientation_state = {
            "horizontal": horizontal,
            "cluster_order_positions": [0, 1],
            "marker_order_positions": [],
            "cluster_leaves": ["A", "B"],
            "marker_leaves": [],
            "cluster_index": pd.Index(["A", "B"]),
            "marker_index": pd.Index([]),
        }

        heatmap.heatmap_data = pd.DataFrame(
            {
                "meta_cluster": [1, 2],
                "meta_cluster_revised": [1, 2],
            },
            index=["A", "B"],
        )

        heatmap.highlight_patches = []
        heatmap._cluster_color_patches = []

        return heatmap, axis

    def test_horizontal_patches_update_after_reassignment(self):
        heatmap, axis = self._build_display(horizontal=True)

        heatmap.display_row_colors_as_patches()

        self.assertEqual(len(axis.vspans), 2)
        first_generation = list(heatmap._cluster_color_patches)
        self.assertTrue(all(hasattr(patch, "remove") for patch in first_generation))
        self.assertFalse(axis.hspans)

        heatmap.heatmap_data.loc["B", "meta_cluster_revised"] = 3
        heatmap.display_row_colors_as_patches()

        self.assertTrue(all(patch.removed for patch in first_generation))
        self.assertTrue(any(entry[2] == "green" for entry in axis.vspans[-2:]))

    def test_vertical_layout_uses_horizontal_spans(self):
        heatmap, axis = self._build_display(horizontal=False)

        heatmap.display_row_colors_as_patches()

        self.assertEqual(len(axis.hspans), 2)
        self.assertEqual(len(axis.vspans), 0)


class HeatmapCanvasRestoreTests(unittest.TestCase):
    def _make_heatmap(self):
        heatmap = HeatmapDisplay.__new__(HeatmapDisplay)
        heatmap.adapter = SimpleNamespace(is_wide=lambda: True)
        heatmap.plot_output = OutputStub()
        heatmap.plot_section = SimpleNamespace(children=(heatmap.plot_output,))
        heatmap._restoring_plot_section = False
        heatmap.data = SimpleNamespace(g=None)
        return heatmap

    def test_redraw_returns_false_without_artifacts(self):
        heatmap = self._make_heatmap()
        heatmap._cached_footer_artifacts = None

        with patch("viewer.plugin.heatmap_layers.display") as display_mock:
            result = heatmap.redraw_cached_footer_canvas()

        self.assertFalse(result)
        display_mock.assert_not_called()

    def test_redraw_replays_display_when_output_empty(self):
        heatmap = self._make_heatmap()
        canvas = CanvasStub()
        fig = SimpleNamespace(canvas=canvas)
        heatmap._cached_footer_artifacts = {
            "fig": fig,
            "canvas": canvas,
            "axes": {},
        }

        with patch("viewer.plugin.heatmap_layers.display") as display_mock:
            result = heatmap.redraw_cached_footer_canvas()

        self.assertTrue(result)
        self.assertEqual(heatmap.plot_output.clear_output_calls, [True])
        display_mock.assert_called_once_with(fig)
        self.assertTrue(canvas.draw_idle_called)

    def test_redraw_skips_display_when_widget_view_present(self):
        heatmap = self._make_heatmap()
        canvas = CanvasStub()
        fig = SimpleNamespace(canvas=canvas)
        heatmap._cached_footer_artifacts = {
            "fig": fig,
            "canvas": canvas,
            "axes": {},
        }
        heatmap.plot_output.outputs = (
            {
                "output_type": "display_data",
                "data": {"application/vnd.jupyter.widget-view+json": {"model_id": "abc"}},
            },
        )

        with patch("viewer.plugin.heatmap_layers.display") as display_mock:
            result = heatmap.redraw_cached_footer_canvas()

        self.assertTrue(result)
        self.assertEqual(heatmap.plot_output.clear_output_calls, [])
        display_mock.assert_not_called()
        self.assertTrue(canvas.draw_idle_called)

    def test_restore_footer_canvas_prefers_cached_redraw(self):
        heatmap = self._make_heatmap()
        canvas = CanvasStub()
        heatmap.data = SimpleNamespace(g=SimpleNamespace(fig=SimpleNamespace(canvas=canvas)))
        heatmap.redraw_cached_footer_canvas = MagicMock(return_value=True)

        heatmap.restore_footer_canvas()

        heatmap.redraw_cached_footer_canvas.assert_called_once()
        self.assertFalse(canvas.draw_idle_called)


if __name__ == "__main__":
    unittest.main()