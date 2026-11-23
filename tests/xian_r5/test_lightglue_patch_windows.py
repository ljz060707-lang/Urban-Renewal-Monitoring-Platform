from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.run_xian_mosaic_lightglue_field_diagnostic import _tile_patch_windows


class LightGluePatchWindowTests(unittest.TestCase):
    def test_non_divisible_tile_grid_is_fully_partitioned_with_clipped_edges(self):
        windows = list(_tile_patch_windows(12, 11, 3))

        self.assertEqual(len(windows), 16)
        covered = set()
        for _, _, row0, col0, row_count, col_count in windows:
            for row in range(row0, row0 + row_count):
                for col in range(col0, col0 + col_count):
                    self.assertNotIn((row, col), covered)
                    covered.add((row, col))

        self.assertEqual(len(covered), 12 * 11)
        self.assertEqual(windows[-1], (3, 3, 9, 9, 2, 3))

    def test_one_tile_patches_and_invalid_dimensions(self):
        windows = list(_tile_patch_windows(12, 11, 1))
        self.assertEqual(len(windows), 12 * 11)
        self.assertTrue(all(window[4:] == (1, 1) for window in windows))

        for args in ((0, 11, 3), (12, 0, 3), (12, 11, 0), (2, 2, 3)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                list(_tile_patch_windows(*args))


if __name__ == "__main__":
    unittest.main()
