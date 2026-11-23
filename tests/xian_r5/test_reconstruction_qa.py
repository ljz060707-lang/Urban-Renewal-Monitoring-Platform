from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box, shape

from tools.qa_xian_r5_grounded_reconstruction import (
    _PyogrioFeatureCollection,
    _candidate_metrics_required,
    _roof_support_quality_code,
    _spatial_quality_codes_valid,
    _spatial_quality_matches_footprint,
    _source_component_quality_code,
    _valid_unknown_scope_spatial_relative_review,
)


class ReconstructionQaSchemaTests(unittest.TestCase):
    def test_candidate_metrics_are_not_required_by_legacy_v2(self) -> None:
        self.assertFalse(_candidate_metrics_required("xian_r5.grounded_products.v2"))

    def test_candidate_metrics_are_required_by_v3_through_v17(self) -> None:
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v3"))
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v4"))
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v5"))
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v6"))
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v7"))
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v8"))
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v9"))
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v10"))
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v11"))
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v12"))
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v13"))
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v14"))
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v15"))
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v16"))
        self.assertTrue(_candidate_metrics_required("xian_r5.grounded_products.v17"))

    def test_unknown_product_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported product manifest schema"):
            _candidate_metrics_required("xian_r5.grounded_products.v999")

    def test_roof_support_codes_are_manifest_driven_and_legacy_safe(self) -> None:
        codes = {
            "insufficient_masked_tie_samples": 8,
            "no_valid_correlation_peak": 9,
            "insufficient_multi_anchor_support": 10,
        }
        self.assertEqual(_roof_support_quality_code("INSUFFICIENT_MASKED_SAMPLES", codes), 8)
        self.assertEqual(_roof_support_quality_code("NO_VALID_CORRELATION_SCORE", codes), 9)
        self.assertEqual(_roof_support_quality_code("INSUFFICIENT_ANCHOR_CANDIDATES", codes), 10)
        self.assertEqual(_roof_support_quality_code("INSUFFICIENT_MASKED_SAMPLES", {}), 3)

    def test_source_component_qa_codes_are_manifest_driven(self) -> None:
        codes = {
            "mosaic_component_boundary_unresolved": 11,
            "source_component_coverage_incomplete": 12,
            "source_component_id_mismatch": 13,
            "source_component_scope_unknown": 14,
        }
        self.assertEqual(_source_component_quality_code(
            "MOSAIC_COMPONENT_BOUNDARY_UNRESOLVED", codes), 11)
        self.assertEqual(_source_component_quality_code(
            "SOURCE_COMPONENT_SCOPE_UNKNOWN", codes), 14)
        self.assertIsNone(_source_component_quality_code("NOT_COMPONENT_STATUS", codes))

    def test_unknown_scope_spatial_cell_requires_exact_relative_review_attestation(self) -> None:
        cell = {
            "method": "SPATIAL_OVERLAP_TRANSLATION_POSE_GRAPH_CANDIDATE",
            "candidate_status": "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY",
            "status": "ACCEPTED_REVIEWED_RELATIVE_IMAGE_TIE",
            "alignment_review_basis": "RELATIVE_REFERENCE_IMAGE_FEATURES",
            "accepted": 1, "reviewed": 1, "applied": 1, "independent_ground": 0,
            "position_unc_m": None, "relative_unc_m": 0.7,
            "relative_ties_within_support_cell": 1,
            "relative_cell_scope_review_authorized": 1,
            "nonbuilding_tie_pairs": 4, "holdout_tie_pairs": 3,
            "source_component_scope": "SOURCE_COMPONENT_SCOPE_UNKNOWN",
            "source_component_ids_json": "[]", "source_component_coverage": None,
            "reference_component_id_status": (
                "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"),
            "source_edge_count": 4, "cycle_closure_max_m": 2.95,
            "fit_dispersion_kind": "POSE_GRAPH_CYCLE_CLOSURE_NOT_POSITIONAL_UNCERTAINTY",
        }
        self.assertTrue(_valid_unknown_scope_spatial_relative_review(
            cell, "xian_r5.grounded_products.v16"))
        self.assertTrue(_valid_unknown_scope_spatial_relative_review(
            cell, "xian_r5.grounded_products.v17"))
        self.assertFalse(_valid_unknown_scope_spatial_relative_review(
            cell, "xian_r5.grounded_products.v15"))
        for name, value in (("relative_ties_within_support_cell", 0),
                            ("independent_ground", 1),
                            ("source_component_scope", "SINGLE_SOURCE_COMPONENT_PER_CELL"),
                            ("candidate_status", "SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_FAILED")):
            with self.subTest(field=name):
                self.assertFalse(_valid_unknown_scope_spatial_relative_review(
                    {**cell, name: value}, "xian_r5.grounded_products.v16"))

    def test_spatial_quality_codes_match_footprint_state_semantics(self) -> None:
        quality_codes = {
            "background": 0, "reviewed_ground": 1, "roof_only_unresolved": 2,
            "unverified_ground_candidate": 3, "ambiguous_ground_overlap": 4,
        }
        footprint_codes = {
            "background": 0, "confirmed_ground": 1, "unresolved": 2, "nodata": 255,
        }
        quality = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
        footprint = np.array([[0, 1, 2, 2, 2]], dtype=np.uint8)
        self.assertTrue(_spatial_quality_codes_valid(quality_codes))
        self.assertTrue(_spatial_quality_matches_footprint(
            quality, footprint, quality_codes, footprint_codes))
        self.assertFalse(_spatial_quality_codes_valid({**quality_codes, "reviewed_ground": 4}))
        self.assertFalse(_spatial_quality_matches_footprint(
            quality, np.array([[0, 2, 2, 3, 2]], dtype=np.uint8),
            quality_codes, footprint_codes))

    @unittest.skipUnless(importlib.util.find_spec("pyogrio")
                         and importlib.util.find_spec("geopandas"),
                         "Pyogrio or GeoPandas is not installed")
    def test_pyogrio_fallback_exposes_schema_and_geojson_features(self) -> None:
        import geopandas as gpd
        import pyogrio

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "qa_layer.gpkg"
            frame = gpd.GeoDataFrame(
                {"uid": ["building-1"], "accepted": [0]},
                geometry=[box(0, 0, 10, 10)], crs="EPSG:32649",
            )
            pyogrio.write_dataframe(frame, path, layer="candidates",
                                    driver="GPKG", geometry_type="Polygon")

            with _PyogrioFeatureCollection(path, "candidates") as source:
                self.assertEqual(len(source), 1)
                self.assertIn("uid", source.schema["properties"])
                feature = next(iter(source))
                self.assertEqual(feature["properties"]["uid"], "building-1")
                self.assertEqual(feature["properties"]["accepted"], 0)
                self.assertTrue(shape(feature["geometry"]).equals(box(0, 0, 10, 10)))


if __name__ == "__main__":
    unittest.main()
