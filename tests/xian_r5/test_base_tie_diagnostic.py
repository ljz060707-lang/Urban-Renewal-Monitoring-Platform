from __future__ import annotations

import unittest
import inspect
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from xian_r5.alignment.base_tie import (estimate_component_gradient_translation,
                                        estimate_gradient_translation,
                                        image_shift_to_projected_correction)
from tools.run_xian_base_tie_diagnostic import (
    _read_vector_features,
    _corr_grid,
    _pixel_window_for_bounds,
    _project_polygon_features,
    _projected_bounds_to_source,
    _rasterize_exclusion_mask,
    _source_component_local_tie_diagnostic,
    _summarize_cli_epoch,
    _write_candidate_alignment_tifs,
    run as run_base_tie_diagnostic,
)


class BaseTieDiagnosticTests(unittest.TestCase):
    def test_pixel_center_roof_mask_is_less_inclusive_than_all_touched(self) -> None:
        from shapely.geometry import box

        geometry = box(0.1, 1.1, 1.1, 2.1)
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0)
        all_touched = _rasterize_exclusion_mask(
            [(geometry, 1)], (3, 3), transform, "all_touched")
        pixel_center = _rasterize_exclusion_mask(
            [(geometry, 1)], (3, 3), transform, "pixel_center")

        self.assertEqual(int(pixel_center.sum()), 1)
        self.assertGreater(int(all_touched.sum()), int(pixel_center.sum()))
        self.assertTrue(np.all((pixel_center > 0) <= (all_touched > 0)))

    def test_roof_mask_rejects_unknown_rasterization_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "policy must be one of"):
            _rasterize_exclusion_mask([], (1, 1), Affine.identity(), "guess")

    def test_roof_mask_rasterization_preserves_historical_default(self) -> None:
        default = inspect.signature(run_base_tie_diagnostic).parameters[
            "roof_mask_rasterization_policy"].default
        self.assertEqual(default, "all_touched")

    def test_external_roof_mask_projects_from_source_crs_and_clips_to_b04(self) -> None:
        from pyproj import CRS, Transformer
        from shapely.geometry import box, mapping
        from shapely.ops import transform

        target = box(309730.0, 3802340.0, 311180.0, 3804070.0)
        to_wgs84 = Transformer.from_crs(32649, 4326, always_xy=True)
        source_polygon = transform(to_wgs84.transform, target)
        query_bbox = _projected_bounds_to_source(target.bounds, CRS.from_epsg(4326))
        projected, counts = _project_polygon_features(
            [{"geometry": mapping(source_polygon)}, {"geometry": None}],
            CRS.from_epsg(4326), target.bounds,
        )

        self.assertEqual(len(projected), 1)
        self.assertGreater(projected[0].area, target.area * 0.999)
        self.assertEqual(counts["feature_count_read"], 2)
        self.assertEqual(counts["invalid_or_nonpolygon_count"], 1)
        self.assertLess(query_bbox[0], query_bbox[2])
        self.assertLess(query_bbox[1], query_bbox[3])

    def test_cli_epoch_summary_excludes_expanded_source_tile_provenance(self) -> None:
        summary = _summarize_cli_epoch({
            "actual_date": "2018-04-29",
            "submeter_tiles": {"exact_date_coverage_fraction": 1.0},
            "registration_diagnostic": {
                "preprocessing_backend": "NUMPY_GAUSSIAN_SOBEL_FALLBACK",
                "best": {"corr": 0.7, "dx_px": 1, "dy_px": -1},
            },
            "local_spatial_tie_diagnostic": {
                "cells": [{"best": {"corr": 0.7}}, {"best": None}],
                "field_interpolated": False, "field_applied": False,
            },
            "candidate_alignment_preview": {
                "status": "GENERATED_CANDIDATE_ONLY_NOT_ACCEPTED",
                "candidate_cells_with_numeric_vectors": 1,
                "quality_pixel_counts": {"2": 100},
                "correction_accepted": False, "field_applied": False,
                "roof_to_ground_correction": False,
                "cells": [
                    {"preview_candidate_rendered": False,
                     "source_component_scope": "INCOMPLETE_SOURCE_COMPONENT_COVERAGE",
                     "source_component_ids": [{"tile_id": "large-expanded-lineage"}]},
                ],
            },
        })

        self.assertEqual(summary["local_numeric_tie_count"], 1)
        self.assertEqual(summary["candidate_rendered_cell_count"], 0)
        self.assertEqual(summary["component_scope_counts"], {
            "INCOMPLETE_SOURCE_COMPONENT_COVERAGE": 1,
        })
        self.assertNotIn("source_component_ids", summary)

    def test_pyogrio_vector_reader_supports_spatially_bounded_fallback(self) -> None:
        try:
            import geopandas as gpd
            import pyogrio
        except ImportError:
            self.skipTest("Pyogrio and GeoPandas are unavailable")
        from shapely.geometry import box
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "buildings.gpkg"
            frame = gpd.GeoDataFrame(
                {"case_id": ["inside", "outside"], "role": ["TARGET", "TARGET"]},
                geometry=[box(2, 2, 4, 4), box(30, 30, 40, 40)],
                crs="EPSG:32649",
            )
            pyogrio.write_dataframe(
                frame, path, layer="buildings", driver="GPKG",
                geometry_type="Polygon",
            )

            with patch("tools.run_xian_base_tie_diagnostic.fiona", None):
                crs, features = _read_vector_features(
                    path, "buildings", bbox=(0, 0, 10, 10),
                )

        self.assertEqual(crs.to_epsg(), 32649)
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0]["properties"]["case_id"], "inside")
        self.assertTrue(features[0]["geometry"].equals(box(2, 2, 4, 4)))

    def test_candidate_rgb_and_quality_tifs_share_date_and_uncertainty_tags(self) -> None:
        preview = SimpleNamespace(
            quality=np.array([[1, 2]], dtype=np.uint8),
            rgba=np.zeros((4, 1, 2), dtype=np.uint8),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            rgb_path, quality_path = _write_candidate_alignment_tifs(
                Path(temporary_dir), "case", preview,
                crs="EPSG:32649", transform=Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0),
                actual_date="2019-12-02", reference_year=2018,
                field_id="2019-12-02_to_Xian_2018_TrueColor_DIRECT",
            )

            with rasterio.open(rgb_path) as rgb, rasterio.open(quality_path) as quality:
                rgb_tags = rgb.tags()
                quality_tags = quality.tags()

        for tags in (rgb_tags, quality_tags):
            self.assertEqual(tags["MOVING_ACTUAL_DATE"], "2019-12-02")
            self.assertEqual(tags["REFERENCE_LABEL"], "Xian_2018_TrueColor")
            self.assertEqual(tags["ALIGNMENT_FIELD_ID"],
                             "2019-12-02_to_Xian_2018_TrueColor_DIRECT")
            self.assertEqual(tags["POSITIONAL_UNCERTAINTY"], "UNKNOWN_NOT_ESTIMATED")
            self.assertEqual(tags["ROOF_TO_GROUND_CORRECTION"], "NO")

    def test_bounds_window_covers_pixels_and_clips_to_raster(self) -> None:
        window = _pixel_window_for_bounds(
            (90.0, 160.0, 145.0, 210.0),
            Affine(10.0, 0.0, 100.0, 0.0, -10.0, 200.0),
            width=10,
            height=10,
        )

        self.assertEqual(window, Window(0, 0, 5, 4))

    def test_bounds_window_handles_rotated_affine(self) -> None:
        transform = Affine(10.0, 2.0, 100.0, 1.0, -10.0, 200.0)
        corners = [
            (transform.a * col + transform.b * row + transform.c,
             transform.d * col + transform.e * row + transform.f)
            for col, row in ((2, 3), (5, 3), (2, 6), (5, 6))
        ]
        bounds = (
            min(point[0] for point in corners), min(point[1] for point in corners),
            max(point[0] for point in corners), max(point[1] for point in corners),
        )

        window = _pixel_window_for_bounds(bounds, transform, width=20, height=20)

        self.assertLessEqual(window.col_off, 2)
        self.assertLessEqual(window.row_off, 3)
        self.assertGreaterEqual(window.col_off + window.width, 5)
        self.assertGreaterEqual(window.row_off + window.height, 6)

    def test_pixel_shift_converts_to_moving_to_reference_map_correction(self) -> None:
        correction = image_shift_to_projected_correction(
            1.5, -2.0, (10.0, 0.0, 309740.0, 0.0, -10.0, 3804060.0))
        self.assertEqual(correction, (-15.0, -20.0))

    def test_shared_tie_interface_reports_moving_image_offset_only(self) -> None:
        rng = np.random.default_rng(91)
        gray = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
        reference = np.repeat(gray[:, :, None], 3, axis=2)
        moving = cv2.warpAffine(
            reference,
            np.float32([[1, 0, 2], [0, 1, -1]]),
            (96, 96),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT,
        )

        result = estimate_gradient_translation(
            reference, moving, np.ones((96, 96), dtype=bool), radius=3,
        )

        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(result["shift_convention"],
                         "MOVING_FEATURE_OFFSET_FROM_REFERENCE;ALIGN_MOVING_APPLY_NEGATIVE")
        self.assertEqual([result["best"]["dx_px"], result["best"]["dy_px"]], [2, -1])
        self.assertGreaterEqual(result["maximum_masked_samples_available"],
                                result["minimum_samples_required"])
        self.assertFalse(result["sample_support_limited"])
        self.assertFalse(any(key in result for key in ("applied", "warped_image")))

    def test_numpy_gradient_fallback_recovers_shift_without_cv_bindings(self) -> None:
        from unittest.mock import patch

        rng = np.random.default_rng(119)
        gray = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
        reference = np.repeat(gray[:, :, None], 3, axis=2)
        moving = cv2.warpAffine(
            reference,
            np.float32([[1, 0, 2], [0, 1, -1]]),
            (96, 96),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT,
        )

        with patch.dict(sys.modules, {"cv2": None}):
            result = estimate_gradient_translation(
                reference, moving, np.ones((96, 96), dtype=bool), radius=3,
            )

        self.assertEqual(result["preprocessing_backend"], "NUMPY_GAUSSIAN_SOBEL_FALLBACK")
        self.assertEqual([result["best"]["dx_px"], result["best"]["dy_px"]], [2, -1])

    def test_component_tie_uses_only_one_source_tile_support(self) -> None:
        rng = np.random.default_rng(97)
        gray = rng.integers(30, 230, size=(96, 96), dtype=np.uint8)
        reference = np.repeat(gray[:, :, None], 3, axis=2)
        moving = cv2.warpAffine(
            reference,
            np.float32([[1, 0, 2], [0, 1, -1]]),
            (96, 96),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT,
        )
        component = np.zeros((96, 96), dtype=bool)
        component[16:80, 14:82] = True

        result = estimate_component_gradient_translation(
            reference,
            moving,
            np.ones((96, 96), dtype=bool),
            np.ones((96, 96), dtype=bool),
            component,
            radius=3,
            min_samples=500,
        )

        self.assertEqual(result["component_local_tie_status"],
                         "NUMERIC_COMPONENT_PEAK_AVAILABLE")
        self.assertEqual([result["best"]["dx_px"], result["best"]["dy_px"]], [2, -1])
        self.assertGreater(result["best"]["n"], 500)
        self.assertEqual(result["base_pixel_window"], [9, 11, 87, 85])
        self.assertFalse(result.get("field_applied", False))

    def test_source_component_diagnostic_keeps_each_exact_tile_as_separate_support(self) -> None:
        from shapely.geometry import box, mapping

        rng = np.random.default_rng(98)
        gray = rng.integers(20, 235, size=(96, 96), dtype=np.uint8)
        reference = np.repeat(gray[:, :, None], 3, axis=2)
        moving = cv2.warpAffine(
            reference, np.float32([[1, 0, 2], [0, 1, -1]]), (96, 96),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT,
        )
        transform = Affine(10, 0, 500, 0, -10, 1500)
        entries = [
            {"tile_id": "420796:312172", "actual_date": "2018-04-29",
             "path": "tile-left.jpg", "sha256": "a" * 64,
             "support_polygon_epsg32649": mapping(box(500, 540, 980, 1500))},
            {"tile_id": "420797:312172", "actual_date": "2018-04-29",
             "path": "tile-right.jpg", "sha256": "b" * 64,
             "support_polygon_epsg32649": mapping(box(980, 540, 1460, 1500))},
        ]

        result = _source_component_local_tie_diagnostic(
            reference, moving, np.ones((96, 96), dtype=bool),
            np.ones((96, 96), dtype=bool), transform, entries,
            actual_date="2018-04-29", radius=3, min_samples=20,
        )

        self.assertEqual(result["support_mode"], "EXACT_SOURCE_TILE_COMPONENTS")
        self.assertEqual(result["component_count_with_base_pixels"], 2)
        self.assertEqual([cell["tile_id"] for cell in result["cells"]],
                         ["420796:312172", "420797:312172"])
        self.assertTrue(all(cell["best"] is not None for cell in result["cells"]))
        self.assertTrue(all(cell["support_polygon_epsg32649"]["type"] == "Polygon"
                            for cell in result["cells"]))
        self.assertFalse(result["field_applied"])
        self.assertFalse(result["field_interpolated"])

    def test_local_field_labels_sample_limited_cells_without_lowering_gate(self) -> None:
        rng = np.random.default_rng(93)
        gray = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
        image = np.repeat(gray[:, :, None], 3, axis=2)
        valid = np.zeros((96, 96), dtype=bool)
        for row in range(3):
            for col in range(3):
                valid[row * 32 + 2:row * 32 + 10,
                      col * 32 + 2:col * 32 + 10] = True

        result = _corr_grid(image, image, valid, grid_size=3, radius=2,
                            min_samples=100)

        self.assertEqual(result["minimum_samples_required"], 100)
        self.assertEqual(result["supported_cell_count"], 0)
        self.assertTrue(all(cell["sample_support_status"] == "INSUFFICIENT_MASKED_SAMPLES"
                            for cell in result["cells"]))
        self.assertTrue(all(cell["maximum_masked_samples_available"] < 100
                            for cell in result["cells"]))

    def test_local_sample_threshold_is_explicit_and_still_candidate_only(self) -> None:
        rng = np.random.default_rng(95)
        gray = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
        image = np.repeat(gray[:, :, None], 3, axis=2)
        valid = np.zeros((96, 96), dtype=bool)
        for row in range(3):
            for col in range(3):
                valid[row * 32 + 2:row * 32 + 12,
                      col * 32 + 2:col * 32 + 12] = True

        strict = _corr_grid(image, image, valid, grid_size=3, radius=2,
                            min_samples=100)
        sensitivity = _corr_grid(image, image, valid, grid_size=3, radius=2,
                                 min_samples=30)

        self.assertEqual(strict["minimum_samples_required"], 100)
        self.assertEqual(sensitivity["minimum_samples_required"], 30)
        self.assertEqual(strict["supported_cell_count"], 0)
        self.assertEqual(sensitivity["supported_cell_count"], 9)
        self.assertTrue(all(cell["sample_support_status"] == "NUMERIC_PEAK_AVAILABLE"
                            for cell in sensitivity["cells"]))
        self.assertFalse(sensitivity["field_applied"])
        self.assertFalse(sensitivity["field_interpolated"])

    def test_local_grid_size_gate_uses_radius_and_minimum_possible_overlap(self) -> None:
        rng = np.random.default_rng(94)
        gray = rng.integers(0, 256, size=(87, 78), dtype=np.uint8)
        image = np.repeat(gray[:, :, None], 3, axis=2)
        valid = np.ones((87, 78), dtype=bool)

        result = _corr_grid(image, image, valid, grid_size=5, radius=3,
                            min_samples=100)

        self.assertEqual(result["grid_rows"], 5)
        self.assertEqual(result["grid_cols"], 5)
        self.assertEqual(result["search_radius_10m_pixels"], 3)
        self.assertEqual(len(result["cells"]), 25)
        self.assertTrue(all(cell["maximum_masked_samples_available"] >= 100
                            for cell in result["cells"]))
        with self.assertRaisesRegex(ValueError, "maximum_overlap"):
            _corr_grid(image[:75, :75], image[:75, :75], valid[:75, :75],
                       grid_size=5, radius=3, min_samples=145)

    def test_moving_validity_is_checked_at_each_candidate_offset(self) -> None:
        rng = np.random.default_rng(92)
        gray = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
        reference = np.repeat(gray[:, :, None], 3, axis=2)
        moving = cv2.warpAffine(
            reference,
            np.float32([[1, 0, 2], [0, 1, -1]]),
            (128, 128),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT,
        )
        full = np.ones((128, 128), dtype=bool)
        moving_valid = full.copy()
        moving_valid[:, :80] = False

        unmasked = estimate_gradient_translation(reference, moving, full, radius=3)
        masked = estimate_gradient_translation(
            reference, moving, full, radius=3,
            moving_valid_mask=moving_valid,
        )

        self.assertEqual([masked["best"]["dx_px"], masked["best"]["dy_px"]], [2, -1])
        self.assertLess(masked["best"]["n"], unmasked["best"]["n"])

    def test_local_grid_reports_known_coarse_shift_without_applying_it(self) -> None:
        rng = np.random.default_rng(7)
        gray = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
        reference = np.repeat(gray[:, :, None], 3, axis=2)
        moved = cv2.warpAffine(
            reference,
            np.float32([[1, 0, 1], [0, 1, -1]]),
            (96, 96),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT,
        )

        result = _corr_grid(
            reference,
            moved,
            np.ones((96, 96), dtype=bool),
            grid_size=3,
            radius=2,
        )

        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY_NOT_ACCEPTED")
        self.assertEqual(result["summary"]["median_coarse_shift_px"], [1.0, -1.0])
        self.assertEqual(result["supported_cell_count"], 9)
        self.assertFalse(result["field_interpolated"])
        self.assertFalse(result["field_applied"])

    def test_subpixel_peak_refines_fractional_synthetic_translation(self) -> None:
        rng = np.random.default_rng(11)
        gray = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
        reference = np.repeat(gray[:, :, None], 3, axis=2)
        moved = cv2.warpAffine(
            reference,
            np.float32([[1, 0, 1.25], [0, 1, -0.3]]),
            (96, 96),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        result = _corr_grid(
            reference,
            moved,
            np.ones((96, 96), dtype=bool),
            grid_size=3,
            radius=2,
        )

        candidate = result["subpixel_candidate_summary"]["median_candidate_shift_px"]
        self.assertIsNotNone(candidate)
        self.assertAlmostEqual(candidate[0], 1.25, delta=0.2)
        self.assertAlmostEqual(candidate[1], -0.3, delta=0.2)
        self.assertFalse(result["field_applied"])


if __name__ == "__main__":
    unittest.main()
