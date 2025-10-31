import unittest

import numpy as np

from ueler.viewer.plugin.cell_gallery import _compose_canvas


class ComposeCanvasTestCase(unittest.TestCase):
    def test_compose_canvas_handles_variable_tile_sizes(self):
        columns = 2
        spacing = 4

        large = np.ones((200, 200, 3), dtype=np.float32)
        narrow = np.full((200, 131, 3), 0.5, dtype=np.float32)
        short = np.full((150, 200, 3), 0.25, dtype=np.float32)

        canvas, rows, cols = _compose_canvas([large, narrow, short], columns, spacing)

        slot_height = 200
        slot_width = 200

        self.assertEqual(rows, 2)
        self.assertEqual(cols, columns)
        self.assertEqual(
            canvas.shape,
            (
                slot_height * rows + spacing * (rows - 1),
                slot_width * columns + spacing * (columns - 1),
                3,
            ),
        )

        # First tile fills the entire first slot with no padding.
        self.assertTrue(np.allclose(canvas[0:slot_height, 0:slot_width, :], large))

        # The narrow tile is centered within the second slot.
        second_col_start = slot_width + spacing + (slot_width - narrow.shape[1]) // 2
        second_col_end = second_col_start + narrow.shape[1]
        self.assertTrue(
            np.allclose(canvas[0:slot_height, second_col_start:second_col_end, :], narrow)
        )

        # Padding around the narrow tile remains zero.
        self.assertTrue(np.allclose(canvas[0:slot_height, slot_width:second_col_start, :], 0.0))
        self.assertTrue(
            np.allclose(
                canvas[0:slot_height, second_col_end : slot_width * columns + spacing * (columns - 1), :],
                0.0,
            )
        )

        # The shorter tile is vertically centered in the second row.
        third_row_start = slot_height + spacing + (slot_height - short.shape[0]) // 2
        third_row_end = third_row_start + short.shape[0]
        self.assertTrue(np.allclose(canvas[third_row_start:third_row_end, 0:slot_width, :], short))

        # Padding above and below the shorter tile remains zero.
        self.assertTrue(np.allclose(canvas[slot_height:third_row_start, 0:slot_width, :], 0.0))
        self.assertTrue(
            np.allclose(
                canvas[
                    third_row_end : slot_height * rows + spacing * (rows - 1),
                    0:slot_width,
                    :,
                ],
                0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
