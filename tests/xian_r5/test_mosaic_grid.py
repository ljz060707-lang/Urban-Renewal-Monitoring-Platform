from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image
import numpy as np
import rasterio
from rasterio.transform import from_origin

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.alignment import MosaicTileGrid, axis_aligned_window_for_bounds
from xian_r5.alignment.mosaic_grid import sha256_file


class MosaicTileGridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.geid = self.root / "scene.geid"
        self.mosaic = self.root / "mosaic.tif"
        self.tile_root = self.root / "scene" / "20"
        self.tile_root.mkdir(parents=True)
        self._write_geid()
        self._write_mosaic(width=512, height=512)
        self._write_tile(10, 20)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_geid(self) -> None:
        self.geid.write_text(
            "[HistoryDate]\nHistoryDate=2019-01-01\n\n"
            "[Zoom]\nZoom1=20\nZoom2=20\n\n"
            "[XY]\nMinX_20=10\nMinY_20=20\nMaxX_20=11\nMaxY_20=21\n\n"
            "[AREA]\nLeft_Longitude_download_20=10\n"
            "Right_Longitude_download_20=12\nTop_Latitude_download_20=6\n"
            "Bottom_Latitude_download_20=4\n",
            encoding="utf-8",
        )

    def _write_mosaic(self, *, width: int, height: int) -> None:
        with rasterio.open(
            self.mosaic,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(10, 6, 2 / 512, 2 / 512),
        ) as raster:
            raster.write(np.zeros((3, height, width), dtype=np.uint8))

    def _write_tile(self, tile_x: int, tile_y: int) -> Path:
        directory = self.tile_root / str(tile_x)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"gesh_{tile_x}_{tile_y}_20.jpg"
        Image.new("RGB", (256, 256), (31, 72, 128)).save(path, format="JPEG")
        return path

    def test_maps_source_tiles_to_exact_stitched_pixel_windows(self) -> None:
        grid = MosaicTileGrid.open(self.geid, self.mosaic, self.tile_root)

        self.assertEqual(grid.nominal_task_date, "2019-01-01")
        self.assertEqual(grid.mosaic_crs, "EPSG:4326")
        self.assertEqual(grid.mosaic_width_px, 512)
        self.assertEqual(grid.mosaic_height_px, 512)
        self.assertEqual(grid.window_for_tile(10, 21).pixel_window, [0, 0, 256, 256])
        self.assertEqual(grid.window_for_tile(11, 20).pixel_window, [256, 256, 256, 256])

    def test_derives_full_tile_model_input_affine_from_exact_mosaic_window(self) -> None:
        grid = MosaicTileGrid.open(self.geid, self.mosaic, self.tile_root)
        model_grid = grid.model_input_grid_for_tile(
            11, 20, model_input_width_px=512, model_input_height_px=512)

        self.assertEqual(model_grid.pixel_crs, "EPSG:4326")
        self.assertEqual(model_grid.source_pixel_window, (256, 256, 256, 256))
        self.assertEqual(model_grid.pixel_to_crs_affine,
                         (1 / 512, 0.0, 11.0, 0.0, -1 / 512, 5.0))

    def test_model_input_grid_rejects_crop_padding_and_invalid_dimensions(self) -> None:
        grid = MosaicTileGrid.open(self.geid, self.mosaic, self.tile_root)
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            grid.model_input_grid_for_tile(
                10, 20, model_input_width_px=512, model_input_height_px=256)
        with self.assertRaisesRegex(ValueError, "positive integers"):
            grid.model_input_grid_for_tile(
                10, 20, model_input_width_px=0, model_input_height_px=512)

    def test_model_input_grid_rejects_sources_changed_after_validation(self) -> None:
        grid = MosaicTileGrid.open(self.geid, self.mosaic, self.tile_root)
        self._write_mosaic(width=510, height=512)
        with self.assertRaisesRegex(RuntimeError, "changed after grid validation"):
            grid.model_input_grid_for_tile(
                10, 20, model_input_width_px=256, model_input_height_px=256)

    def test_component_receipt_requires_tile_actual_date_and_checks_provenance(self) -> None:
        grid = MosaicTileGrid.open(self.geid, self.mosaic, self.tile_root)
        path = self.tile_root / "10" / "gesh_10_20_20.jpg"
        digest = sha256_file(path)

        receipt = grid.component_receipt(
            10,
            20,
            actual_date="2018-11-01",
            actual_date_source_ref="date-audit:row-7",
            expected_sha256=digest,
            expected_path=path,
        )

        self.assertEqual(receipt["nominal_task_date"], "2019-01-01")
        self.assertEqual(receipt["actual_date"], "2018-11-01")
        self.assertEqual(receipt["pixel_window_col_row_width_height"], [0, 256, 256, 256])
        self.assertEqual(receipt["source_tile_sha256"], digest)
        self.assertEqual(receipt["window_bounds_epsg4326"], [10.0, 4.0, 11.0, 5.0])

        with self.assertRaisesRegex(ValueError, "actual ISO acquisition date"):
            grid.component_receipt(
                10, 20, actual_date="2019", actual_date_source_ref="date-audit:row-7"
            )

    def test_rejects_tile_hash_or_path_mismatch(self) -> None:
        grid = MosaicTileGrid.open(self.geid, self.mosaic, self.tile_root)
        args = dict(actual_date="2018-11-01", actual_date_source_ref="date-audit:row-7")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            grid.component_receipt(10, 20, expected_sha256="0" * 64, **args)
        with self.assertRaisesRegex(ValueError, "path mismatch"):
            grid.component_receipt(10, 20, expected_path=self.root / "other.jpg", **args)

    def test_rejects_mosaic_grid_that_disagrees_with_geid_extent(self) -> None:
        self._write_mosaic(width=510, height=512)
        with self.assertRaisesRegex(ValueError, "dimensions do not equal"):
            MosaicTileGrid.open(self.geid, self.mosaic, self.tile_root)

    def test_rejects_tile_outside_geid_grid_and_missing_source(self) -> None:
        grid = MosaicTileGrid.open(self.geid, self.mosaic, self.tile_root)
        with self.assertRaisesRegex(ValueError, "outside .* GEID grid"):
            grid.window_for_tile(12, 20)
        with self.assertRaises(FileNotFoundError):
            grid.component_receipt(
                11,
                21,
                actual_date="2018-11-01",
                actual_date_source_ref="date-audit:row-8",
            )

    def test_maps_projected_bounds_to_clipped_north_up_reference_windows(self) -> None:
        transform = (10.0, 0.0, 0.0, 0.0, -10.0, 100.0)
        self.assertEqual(
            axis_aligned_window_for_bounds([20, 30, 41, 60], transform, 10, 10),
            [2, 4, 3, 3],
        )
        self.assertEqual(
            axis_aligned_window_for_bounds([-10, -10, 20, 20], transform, 10, 10),
            [0, 8, 2, 2],
        )
        self.assertIsNone(
            axis_aligned_window_for_bounds([200, 0, 210, 10], transform, 10, 10)
        )
        with self.assertRaisesRegex(ValueError, "north-up"):
            axis_aligned_window_for_bounds([0, 0, 10, 10],
                                           (10, 1, 0, 0, -10, 100), 10, 10)


if __name__ == "__main__":
    unittest.main()
