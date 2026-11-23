from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from xian_r5.alignment.field import DisplacementCell, LocalDisplacementField
from xian_r5.alignment.spatial_composition import compose_spatial_alignment_fields


ANCHOR_LABEL = "Xian_2019_TrueColor"
ANCHOR_SOURCE = "F:/xian/xian底图/Xian_2019_TrueColor.tif"


def _direct(actual_date: str, geometry, correction: tuple[float, float], *,
            reviewed: bool = False) -> LocalDisplacementField:
    cell = DisplacementCell(
        cell_id=f"direct-{actual_date}",
        support_geometry=geometry,
        correction_dx_m_candidate=correction[0],
        correction_dy_m_candidate=correction[1],
        fit_dispersion_m=None,
        status="DIRECT_BASE_TIE_CANDIDATE_NOT_REVIEWED",
        method="test-direct-base-tie",
        source_ref=f"direct:{actual_date}",
        reviewed=reviewed,
        metadata={
            "reference_raster_sha256": "a" * 64,
            "reference_component_id_status": (
                "PER_PIXEL_COMPONENT_PROVENANCE_NOT_RECORDED_BY_BASE_TIE_RECEIPT"),
        },
    )
    return LocalDisplacementField(
        field_id=f"direct-field-{actual_date}",
        reference_date=None,
        moving_date=actual_date,
        cells=(cell,),
        reference_label=ANCHOR_LABEL,
        reference_source=ANCHOR_SOURCE,
        reference_date_status="REFERENCE_ACQUISITION_DATE_UNKNOWN",
    )


def _relative(reference_date: str, moving_date: str, geometry,
              correction: tuple[float, float]) -> LocalDisplacementField:
    cell = DisplacementCell(
        cell_id=f"relative-{reference_date}-{moving_date}",
        support_geometry=geometry,
        correction_dx_m_candidate=correction[0],
        correction_dy_m_candidate=correction[1],
        fit_dispersion_m=None,
        status="LOCAL_CANDIDATE_NOT_REVIEWED",
        method="test-relative-image-tie",
        source_ref=f"relative:{reference_date}->{moving_date}",
        metadata={"field_interpolated": False},
    )
    return LocalDisplacementField(
        field_id=f"relative-field-{reference_date}-{moving_date}",
        reference_date=reference_date,
        moving_date=moving_date,
        cells=(cell,),
        reference_label=reference_date,
        reference_source="test-relative-image-pair",
    )


class SpatialCompositionTests(unittest.TestCase):
    def test_composes_only_on_exact_polygon_overlap_and_preserves_candidate_status(self):
        full = box(0, 0, 10, 10)
        left = box(0, 0, 5, 10)
        direct_apr = _direct("2018-04-29", full, (2.0, 3.0))
        direct_nov = _direct("2018-11-01", full, (5.0, 4.0))
        relative_apr_nov = _relative("2018-04-29", "2018-11-01", left, (3.0, 1.0))
        relative_nov_dec = _relative("2018-11-01", "2019-12-02", left, (-1.0, 2.0))

        fields, summary = compose_spatial_alignment_fields(
            [direct_apr, direct_nov], [relative_apr_nov, relative_nov_dec])

        by_date = {field.moving_date: field for field in fields}
        self.assertEqual(set(by_date), {"2018-04-29", "2018-11-01", "2019-12-02"})
        self.assertEqual(len(by_date["2018-04-29"].cells), 1)
        self.assertEqual(len(by_date["2018-11-01"].cells), 1)
        self.assertEqual(len(by_date["2019-12-02"].cells), 1)
        self.assertEqual(
            by_date["2019-12-02"].sample(2.0, 5.0).accepted_correction_m, None)
        december = by_date["2019-12-02"].sample(2.0, 5.0)
        self.assertAlmostEqual(december.correction_dx_m_candidate, 4.0)
        self.assertAlmostEqual(december.correction_dy_m_candidate, 6.0)
        self.assertEqual(by_date["2019-12-02"].sample(8.0, 5.0).status, "OUTSIDE_FIELD")
        self.assertEqual(by_date["2018-04-29"].sample(8.0, 5.0).status, "OUTSIDE_FIELD")
        self.assertTrue(all(not cell.accepted and not cell.applied
                            for field in fields for cell in field.cells))
        self.assertFalse(summary["absolute_ground_position_resolved"])
        self.assertFalse(summary["roof_to_ground_parallax_resolved"])
        self.assertEqual(
            summary["cycle_status_counts"]["SPATIALLY_COMPOSED_CANDIDATE_CYCLE_SCREEN_ONLY"], 1)

    def test_touching_supports_do_not_create_a_false_date_link(self):
        direct = _direct("2018-04-29", box(0, 0, 5, 5), (1.0, 2.0))
        relative = _relative("2018-04-29", "2018-11-01", box(5, 0, 10, 5), (3.0, 4.0))

        fields, summary = compose_spatial_alignment_fields([direct], [relative])

        self.assertEqual(fields, [])
        self.assertEqual(summary["unanchored_component_count"], 1)

    def test_accepts_disjoint_windows_for_the_same_actual_date_pair(self):
        left = _relative("2018-04-29", "2018-11-01", box(0, 0, 5, 10), (3.0, 4.0))
        right = _relative("2018-04-29", "2018-11-01", box(5, 0, 10, 10), (5.0, 6.0))
        right = replace(right, field_id="relative-field-window-2",
                        reference_source="second-local-window")

        fields, summary = compose_spatial_alignment_fields(
            [_direct("2018-04-29", box(0, 0, 10, 10), (1.0, 2.0))], [left, right])

        by_date = {field.moving_date: field for field in fields}
        self.assertEqual(len(by_date["2018-11-01"].cells), 2)
        composed = by_date["2018-11-01"].sample(2.0, 5.0)
        self.assertAlmostEqual(composed.correction_dx_m_candidate, 4.0)
        self.assertAlmostEqual(composed.correction_dy_m_candidate, 6.0)
        self.assertAlmostEqual(
            by_date["2018-11-01"].sample(8.0, 5.0).correction_dx_m_candidate, 6.0)
        self.assertEqual(summary["spatially_composed_atom_count"], 2)

    def test_rejects_reviewed_input_in_candidate_only_composer(self):
        direct = _direct("2018-04-29", box(0, 0, 10, 10), (1.0, 2.0), reviewed=True)
        relative = _relative("2018-04-29", "2018-11-01", box(0, 0, 10, 10), (3.0, 4.0))

        with self.assertRaisesRegex(ValueError, "unreviewed candidates only"):
            compose_spatial_alignment_fields([direct], [relative])


if __name__ == "__main__":
    unittest.main()
