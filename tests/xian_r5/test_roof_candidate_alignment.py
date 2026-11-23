from __future__ import annotations

import sys
import unittest
from pathlib import Path

from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.alignment.field import DisplacementCell, LocalDisplacementField
from xian_r5.alignment.roof_candidate import align_roof_candidate_to_reference_grid


def field(cell):
    return LocalDisplacementField(
        "2018-04-29_to_Xian_2019_TrueColor_DIRECT",
        None,
        "2018-04-29",
        (cell,),
        reference_label="Xian_2019_TrueColor",
        reference_source=r"F:\xian\xian底图\xian搴曞浘\Xian_2019_TrueColor.tif",
        reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN",
    )


def cell(*, cell_id="c0", bounds=(0, 0, 100, 100), dx=2.5, dy=-1.0,
         status="DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED",
         component_scope="SINGLE_SOURCE_COMPONENT_PER_CELL",
         component_tile_id="420796:312172"):
    return DisplacementCell(
        cell_id, box(*bounds), dx, dy, None, status, "TEST_TIE", f"tie:{cell_id}",
        reviewed=False, independent_ground_evidence=False,
        positional_uncertainty_m=None, applied=False,
        metadata={
            "source_component_scope": component_scope,
            "source_component_ids": ([{
                "actual_date": "2018-04-29", "tile_id": component_tile_id,
                "path": "source/420796_312172.jpg", "sha256": "a" * 64,
            }] if component_scope == "SINGLE_SOURCE_COMPONENT_PER_CELL" else []),
            "source_component_coverage_fraction": (
                1.0 if component_scope == "SINGLE_SOURCE_COMPONENT_PER_CELL" else 0.5),
        })


class RoofCandidateAlignmentTests(unittest.TestCase):
    def test_unique_numeric_cell_translates_roof_as_candidate_not_ground(self):
        roof = box(20, 20, 30, 25)

        result = align_roof_candidate_to_reference_grid(
            roof, "2018-04-29", field(cell(dx=2.5, dy=-1.0)))

        self.assertEqual(result.status, "UNACCEPTED_IMAGE_GRID_SHIFT_ROOF_CANDIDATE")
        self.assertTrue(result.candidate_shift_rendered)
        self.assertEqual(result.cell_id, "c0")
        self.assertEqual(result.geometry.bounds, (22.5, 19.0, 32.5, 24.0))
        self.assertEqual(result.geometry.area, roof.area)
        self.assertIsNone(result.grid_positional_uncertainty_m)
        self.assertEqual(result.cell_support_fraction, 1.0)

    def test_missing_vector_preserves_raw_geometry(self):
        roof = box(20, 20, 30, 25)

        result = align_roof_candidate_to_reference_grid(
            roof, "2018-04-29", field(cell(dx=None, dy=None)))

        self.assertEqual(result.status, "NO_NUMERIC_ALIGNMENT_CANDIDATE")
        self.assertFalse(result.candidate_shift_rendered)
        self.assertEqual(result.geometry, roof)

    def test_multi_component_candidate_is_not_shifted(self):
        roof = box(20, 20, 30, 25)

        result = align_roof_candidate_to_reference_grid(
            roof, "2018-04-29", field(cell(component_scope="MULTIPLE_SOURCE_COMPONENTS_PER_CELL")))

        self.assertEqual(result.status, "MOSAIC_COMPONENT_BOUNDARY_UNRESOLVED")
        self.assertFalse(result.candidate_shift_rendered)
        self.assertEqual(result.geometry, roof)

    def test_cell_crossing_is_not_partially_translated(self):
        roof = box(45, 20, 55, 25)
        left = cell(bounds=(0, 0, 50, 100), dx=2.0, dy=0.0)
        right = DisplacementCell(
            "c1", box(50, 0, 100, 100), -3.0, 0.0, None,
            "DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED", "TEST_TIE", "tie:c1")
        pair = LocalDisplacementField(
            "crossing", None, "2018-04-29", (left, right),
            reference_label="Xian_2019_TrueColor", reference_source="test:base",
            reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")

        result = align_roof_candidate_to_reference_grid(roof, "2018-04-29", pair)

        self.assertEqual(result.status, "CROSSES_OR_OVERLAPS_ALIGNMENT_CELLS")
        self.assertFalse(result.candidate_shift_rendered)
        self.assertEqual(result.geometry, roof)

    def test_dominant_unreviewed_cell_emits_whole_building_candidate(self):
        roof = box(45, 20, 55, 25)
        left = cell(bounds=(0, 0, 54, 100), dx=2.0, dy=1.0)
        right = cell(cell_id="c1", bounds=(54, 0, 100, 100), dx=-3.0)
        pair = LocalDisplacementField(
            "dominant", None, "2018-04-29", (left, right),
            reference_label="Xian_2019_TrueColor", reference_source="test:base",
            reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")

        result = align_roof_candidate_to_reference_grid(roof, "2018-04-29", pair)

        self.assertEqual(
            result.status,
            "UNACCEPTED_DOMINANT_CELL_IMAGE_GRID_SHIFT_ROOF_CANDIDATE")
        self.assertTrue(result.candidate_shift_rendered)
        self.assertEqual(result.cell_id, "c0")
        self.assertAlmostEqual(result.cell_support_fraction, 0.9)
        self.assertEqual(result.geometry.bounds, (47.0, 21.0, 57.0, 26.0))
        self.assertEqual(result.geometry.area, roof.area)

    def test_dominant_reviewed_cell_does_not_apply_partial_coverage(self):
        roof = box(45, 20, 55, 25)
        reviewed = DisplacementCell(
            "c0", box(0, 0, 54, 100), 2.0, 1.0, None,
            "ACCEPTED_REVIEWED_GROUND", "TEST_TIE", "tie:c0",
            reviewed=True, independent_ground_evidence=True,
            positional_uncertainty_m=0.5,
            metadata={
                "source_component_scope": "SINGLE_SOURCE_COMPONENT_PER_CELL",
                "source_component_ids": [{
                    "actual_date": "2018-04-29", "tile_id": "420796:312172",
                    "path": "source/420796_312172.jpg", "sha256": "a" * 64,
                }],
                "source_component_coverage_fraction": 1.0,
            })
        right = DisplacementCell(
            "c1", box(54, 0, 100, 100), -3.0, 0.0, None,
            "DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED", "TEST_TIE", "tie:c1")
        pair = LocalDisplacementField(
            "reviewed-dominant", None, "2018-04-29", (reviewed, right),
            reference_label="Xian_2019_TrueColor", reference_source="test:base",
            reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN")

        result = align_roof_candidate_to_reference_grid(roof, "2018-04-29", pair)

        self.assertEqual(result.status, "CROSSES_OR_OVERLAPS_ALIGNMENT_CELLS")
        self.assertFalse(result.candidate_shift_rendered)
        self.assertEqual(result.geometry, roof)

    def test_actual_date_mismatch_preserves_raw_geometry(self):
        roof = box(20, 20, 30, 25)

        result = align_roof_candidate_to_reference_grid(
            roof, "2018-11-01", field(cell()))

        self.assertEqual(result.status, "FIELD_ACTUAL_DATE_MISMATCH")
        self.assertFalse(result.candidate_shift_rendered)
        self.assertEqual(result.geometry, roof)


if __name__ == "__main__":
    unittest.main()
