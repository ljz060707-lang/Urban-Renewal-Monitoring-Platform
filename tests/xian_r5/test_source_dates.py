from __future__ import annotations

import unittest

from xian_r5.reconstruction.source_dates import associate_source_tile_dates


class SourceTileDateAssociationTests(unittest.TestCase):
    def test_single_date_requires_complete_tile_coverage(self):
        result = associate_source_tile_dates(
            [(420798, 312173), (420797, 312173), (420798, 312173)],
            {(420797, 312173): "2018-11-01",
             (420798, 312173): "2018-11-01"},
        )

        self.assertEqual(result.status, "SINGLE_ACTUAL_DATE_CANDIDATE")
        self.assertEqual(result.tile_ids, ("420797:312173", "420798:312173"))
        self.assertEqual(result.tile_actual_dates, (
            ("420797:312173", "2018-11-01"),
            ("420798:312173", "2018-11-01"),
        ))
        self.assertEqual(result.candidate_dates, ("2018-11-01",))
        self.assertEqual(result.unknown_tile_ids, ())

    def test_mixed_tile_dates_are_not_collapsed(self):
        result = associate_source_tile_dates(
            [(420797, 312173), (420797, 312174)],
            {(420797, 312173): "2018-11-01",
             (420797, 312174): "2018-11-08"},
        )

        self.assertEqual(result.status, "MULTIPLE_ACTUAL_DATES")
        self.assertEqual(result.tile_actual_dates, (
            ("420797:312173", "2018-11-01"),
            ("420797:312174", "2018-11-08"),
        ))
        self.assertEqual(result.candidate_dates, ("2018-11-01", "2018-11-08"))

    def test_unknown_tile_prevents_single_date_resolution(self):
        result = associate_source_tile_dates(
            [(420797, 312173), (420799, 312173)],
            {(420797, 312173): "2018-11-01"},
        )

        self.assertEqual(result.status, "INCOMPLETE_SOURCE_TILE_DATES")
        self.assertEqual(result.tile_actual_dates, (
            ("420797:312173", "2018-11-01"),
            ("420799:312173", None),
        ))
        self.assertEqual(result.candidate_dates, ("2018-11-01",))
        self.assertEqual(result.unknown_tile_ids, ("420799:312173",))

    def test_empty_intersection_has_explicit_unresolved_status(self):
        result = associate_source_tile_dates([], {})

        self.assertEqual(result.status, "NO_SOURCE_TILE_SUPPORT")
        self.assertEqual(result.candidate_dates, ())

    def test_invalid_date_is_rejected(self):
        with self.assertRaises(ValueError):
            associate_source_tile_dates(
                [(1, 2)], {(1, 2): "2019-1-1"},
            )


if __name__ == "__main__":
    unittest.main()
