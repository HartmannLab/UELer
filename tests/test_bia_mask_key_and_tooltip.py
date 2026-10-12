"""Regression tests for two BIA streaming-mode failures (follow-up to #141/#142).

1. **Gallery without masks.**  ``S-BIAD2557``'s only mask layer is
   ``cleaned_mask``, which none of the ``Mask key:`` aliases name, so a fresh
   workspace kept the shipped default ``whole_cell``.  The main viewer draws
   masks from its checkboxes and looked fine; the gallery (and the mask
   painter, the linked highlights) look the layer up by ``mask_key`` and found
   nothing.
2. **Empty tooltip.**  A Parquet cell table holds only the spine until a column
   is asked for, so the hovered row had no marker or label values and the
   tooltip showed the mask ID alone.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ueler.cell_table_source import ParquetSource
from ueler.viewer.data_mapping import MASK_FIELD, apply_options
from ueler.viewer.main_viewer import ImageMaskViewer


def _widget(value, options=None):
    return SimpleNamespace(value=value, options=list(options or [value]))


class ApplyOptionsDefaultTests(unittest.TestCase):
    def test_an_absent_shipped_default_takes_the_first_option(self):
        widget = _widget("whole_cell")
        changed = apply_options(
            widget, ["cleaned_mask"], preferred=MASK_FIELD.preferred, default="whole_cell"
        )
        self.assertTrue(changed)
        self.assertEqual(widget.value, "cleaned_mask")
        self.assertEqual(widget.options, ["cleaned_mask"])

    def test_a_preferred_alias_still_wins_over_the_first_option(self):
        widget = _widget("whole_cell")
        apply_options(
            widget, ["boundary", "nuclear"], preferred=MASK_FIELD.preferred, default="whole_cell"
        )
        self.assertEqual(widget.value, "nuclear")

    def test_a_value_the_user_gave_is_still_kept(self):
        widget = _widget("membrane")
        changed = apply_options(
            widget, ["cleaned_mask"], preferred=MASK_FIELD.preferred, default="whole_cell"
        )
        self.assertFalse(changed)
        self.assertEqual(widget.value, "membrane")
        self.assertEqual(widget.options, ["cleaned_mask", "membrane"])

    def test_without_a_default_the_old_rule_applies(self):
        widget = _widget("whole_cell")
        apply_options(widget, ["cleaned_mask"], preferred=MASK_FIELD.preferred)
        self.assertEqual(widget.value, "whole_cell")


class _MappingViewer:
    available_mask_suffixes = ImageMaskViewer.available_mask_suffixes
    refresh_data_mapping_options = ImageMaskViewer.refresh_data_mapping_options

    def __init__(self, masks_folder, fovs):
        self.ui_component = SimpleNamespace(mask_key=_widget("whole_cell"))
        self.cell_table_schema = {}
        self.mask_names_set = set()
        self.masks_folder = masks_folder
        self.available_fovs = list(fovs)

    def update_keys(self, *args):
        self.mask_key = self.ui_component.mask_key.value


class RefreshMaskKeyTests(unittest.TestCase):
    def test_a_study_whose_only_mask_is_unaliased_is_linked_to_it(self):
        with tempfile.TemporaryDirectory() as folder:
            open(os.path.join(folder, "fov1_cleaned_mask.tiff"), "wb").close()
            viewer = _MappingViewer(folder, ["fov1"])

            viewer.refresh_data_mapping_options()

            self.assertEqual(viewer.mask_key, "cleaned_mask")


class _HoverViewer:
    """The surface ``_lookup_mask_hit`` touches, with the real methods bound."""

    _lookup_mask_hit = ImageMaskViewer._lookup_mask_hit
    _tooltip_columns = ImageMaskViewer._tooltip_columns
    _ensure_tooltip_columns = ImageMaskViewer._ensure_tooltip_columns
    _selected_mask_names = ImageMaskViewer._selected_mask_names

    def __init__(self, source, mask, channels=(), labels=()):
        self._source = source
        self.cell_table = source.read_columns(["fov", "cell_id", "X", "Y"])
        self._mask = mask
        self.masks_available = True
        self.fov_key, self.label_key, self.mask_key = "fov", "cell_id", "cleaned_mask"
        self.selected_tooltip_labels = list(labels)
        self.ui_component = SimpleNamespace(
            mask_display_controls={"cleaned_mask": SimpleNamespace(value=True)},
            channel_selector=SimpleNamespace(value=tuple(channels)),
        )

    def _get_mask_array(self, fov_name, mask_name):
        return self._mask

    def ensure_columns(self, names):
        missing = [n for n in names if n not in self.cell_table.columns and n in self._source.schema]
        if missing:
            data = self._source.read_columns(missing)
            for name in missing:
                self.cell_table[name] = data[name].to_numpy()
        return self.cell_table


class TooltipColumnsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ueler-tooltip-")
        frame = pd.DataFrame(
            {
                "fov": ["fov1"] * 3,
                "cell_id": np.array([1, 2, 3], dtype="int64"),
                "X": [1.0, 2.0, 3.0],
                "Y": [1.0, 2.0, 3.0],
                "CD45": np.array([0.5, 1.5, 2.5], dtype="float32"),
                "lineage": ["tumour", "immune", "stroma"],
            }
        )
        cls.path = os.path.join(cls.tmp, "cells.parquet")
        frame.to_parquet(cls.path, index=False)
        cls.mask = np.zeros((4, 4), dtype=np.uint32)
        cls.mask[2, 2] = 2

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _viewer(self, **kwargs):
        return _HoverViewer(ParquetSource(self.path), self.mask, **kwargs)

    def test_the_hovered_row_carries_the_channels_and_labels(self):
        viewer = self._viewer(channels=("CD45",), labels=("lineage",))

        hit = viewer._lookup_mask_hit("fov1", 2.0, 2.0)

        self.assertEqual(hit.mask_id, 2)
        self.assertAlmostEqual(float(hit.cell_record["CD45"]), 1.5)
        self.assertEqual(hit.cell_record["lineage"], "immune")

    def test_a_channel_without_a_column_is_ignored(self):
        viewer = self._viewer(channels=("DAPI",))

        hit = viewer._lookup_mask_hit("fov1", 2.0, 2.0)

        self.assertEqual(int(hit.cell_record["cell_id"]), 2)
        self.assertNotIn("CD45", viewer.cell_table.columns)

    def test_nothing_is_read_when_the_tooltip_shows_nothing(self):
        viewer = self._viewer()

        viewer._lookup_mask_hit("fov1", 2.0, 2.0)

        self.assertEqual(list(viewer.cell_table.columns), ["fov", "cell_id", "X", "Y"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
