from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from affine import Affine
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.alignment.field import DisplacementCell, LocalDisplacementField
from xian_r5.alignment.preview import render_candidate_alignment_preview


def make_cell(cell_id, geometry, *, dx=None, dy=None, status="UNRESOLVED", metadata=None):
    if metadata is None:
        metadata = {
            "source_component_scope": "SINGLE_SOURCE_COMPONENT_PER_CELL",
            "source_component_ids": [{
                "actual_date": "2018-04-29",
                "tile_id": "420796:312172",
                "path": f"test/{cell_id}.jpg",
                "sha256": "a" * 64,
            }],
            "source_component_coverage_fraction": 1.0,
        }
    return DisplacementCell(
        cell_id=cell_id,
        support_geometry=geometry,
        correction_dx_m_candidate=dx,
        correction_dy_m_candidate=dy,
        fit_dispersion_m=None,
        status=status,
        method="SYNTHETIC_TEST_TIE",
        source_ref=f"test:{cell_id}",
        reviewed=False,
        independent_ground_evidence=False,
        positional_uncertainty_m=None,
        applied=False,
        metadata=metadata,
    )


def make_field(*cells):
    return LocalDisplacementField(
        "2018-04-29_to_Xian_2019_TrueColor_DIRECT",
        None,
        "2018-04-29",
        tuple(cells),
        reference_label="Xian_2019_TrueColor",
        reference_source="test:base",
        reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN",
    )


class CandidateAlignmentPreviewTests(unittest.TestCase):
    def test_candidate_shift_moves_image_content_with_projected_correction_sign(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[3, 4] = (200, 100, 50)
        valid = np.ones((8, 8), dtype=np.uint8)
        transform = Affine(1, 0, 0, 0, -1, 8)
        field = make_field(make_cell(
            "c0", box(0, 0, 8, 8), dx=-1, dy=1,
            status="DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED"))

        preview = render_candidate_alignment_preview(
            image, valid, transform, image.shape[:2], transform, field)

        np.testing.assert_array_equal(preview.rgba[:3, 2, 3], (200, 100, 50))
        self.assertEqual(int(preview.rgba[3, 2, 3]), 255)
        self.assertEqual(int(preview.quality[2, 3]), 1)
        self.assertEqual(int(preview.quality[0, 0]), 3)
        self.assertFalse(preview.cells[0]["position_accuracy_claimed"])

    def test_unresolved_cells_remain_transparent_and_are_quality_coded(self):
        image = np.full((8, 8, 3), 120, dtype=np.uint8)
        valid = np.ones((8, 8), dtype=np.uint8)
        transform = Affine(1, 0, 0, 0, -1, 8)
        field = make_field(
            make_cell("left", box(0, 0, 4, 8), dx=0, dy=0,
                      status="DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED"),
            make_cell("right", box(4, 0, 8, 8)),
        )

        preview = render_candidate_alignment_preview(
            image, valid, transform, image.shape[:2], transform, field)

        self.assertEqual(int(preview.quality[3, 6]), 2)
        self.assertEqual(int(preview.rgba[3, 3, 6]), 0)
        self.assertEqual(int(preview.rgba[3, 3, 1]), 255)

    def test_overlapping_cells_are_not_warped_and_marked_ambiguous(self):
        image = np.full((8, 8, 3), 90, dtype=np.uint8)
        valid = np.ones((8, 8), dtype=np.uint8)
        transform = Affine(1, 0, 0, 0, -1, 8)
        field = make_field(
            make_cell("left", box(0, 0, 6, 8), dx=0, dy=0,
                      status="DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED"),
            make_cell("right", box(2, 0, 8, 8), dx=1, dy=0,
                      status="DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED"),
        )

        preview = render_candidate_alignment_preview(
            image, valid, transform, image.shape[:2], transform, field)

        self.assertEqual(int(preview.quality[3, 3]), 4)
        self.assertEqual(int(preview.rgba[3, 3, 3]), 0)

    def test_numeric_candidates_without_single_source_component_scope_are_not_rendered(self):
        image = np.full((8, 8, 3), 90, dtype=np.uint8)
        valid = np.ones((8, 8), dtype=np.uint8)
        transform = Affine(1, 0, 0, 0, -1, 8)
        cases = (
            (
                "MOSAIC_COMPONENT_BOUNDARY_UNRESOLVED",
                {"source_component_scope": "MULTIPLE_SOURCE_COMPONENTS_PER_CELL",
                 "source_component_ids": [{"tile_id": "420796:312172"}],
                 "source_component_coverage_fraction": 1.0},
                5,
            ),
            (
                "SOURCE_COMPONENT_COVERAGE_INCOMPLETE",
                {"source_component_scope": "INCOMPLETE_SOURCE_COMPONENT_COVERAGE",
                 "source_component_ids": [],
                 "source_component_coverage_fraction": 0.75},
                6,
            ),
            (
                "SOURCE_COMPONENT_ID_MISMATCH",
                {"source_component_scope": "SOURCE_COMPONENT_ID_MISMATCH",
                 "source_component_ids": [],
                 "source_component_coverage_fraction": 1.0},
                7,
            ),
            (
                "SOURCE_COMPONENT_SCOPE_UNKNOWN",
                {},
                8,
            ),
        )

        for issue, metadata, expected_code in cases:
            with self.subTest(issue=issue):
                field = make_field(make_cell(
                    "c0", box(0, 0, 8, 8), dx=-1, dy=1,
                    status="DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED",
                    metadata=metadata,
                ))
                preview = render_candidate_alignment_preview(
                    image, valid, transform, image.shape[:2], transform, field)

                self.assertFalse(preview.cells[0]["preview_candidate_rendered"])
                self.assertEqual(preview.cells[0]["source_component_issue"], issue)
                self.assertEqual(preview.cells[0]["preview_blocked_reason"], issue)
                self.assertEqual(int(preview.quality[3, 3]), expected_code)
                self.assertFalse(np.any(preview.rgba[3]))


if __name__ == "__main__":
    unittest.main()
