import unittest
import numpy as np
from affine import Affine
from shapely.geometry import box, Polygon
from xian_r5.label_transfer import Grid
from xian_r5.manual_roof_reference import manual_reference_binding, manual_roof_fractions


class ManualReferenceTests(unittest.TestCase):
    def setUp(self):
        self.grid = Grid(Affine(10, 0, 0, 0, -10, 10), 2, 1, 'EPSG:32649')

    def test_user_assertion_is_not_numeric_accuracy(self):
        b = manual_reference_binding()
        self.assertTrue(b['user_confirmed_unbiased_basemap'])
        self.assertIsNone(b['metric_localization_bound_m'])
        self.assertEqual(b['absolute_position_accuracy'], 'UNKNOWN')

    def test_not_ground_or_automatic_admission(self):
        b = manual_reference_binding()
        self.assertEqual(b['geometry_semantics'], 'ROOF_OUTLINE_NOT_GROUND_FOOTPRINT')
        self.assertFalse(b['automatic_alignment_acceptance'])
        self.assertFalse(b['automatic_training_admission'])
        self.assertFalse(b['source_geometry_may_snap_to_reference'])

    def test_no_new_unseen_holdout(self):
        self.assertFalse(manual_reference_binding()['external_unseen_test_set'])

    def test_union_not_duplicate_area(self):
        r = manual_roof_fractions({'a': box(0, 0, 7, 10), 'b': box(3, 0, 10, 10)}, self.grid)
        np.testing.assert_array_equal(r['fraction'], [[1., 0.]])
        np.testing.assert_array_equal(r['object_count'], [[2, 0]])
        np.testing.assert_array_equal(r['overlap_conflict'], [[True, False]])

    def test_partial_full_denominator(self):
        r = manual_roof_fractions({'a': box(-5, 0, 4, 10)}, self.grid)
        self.assertAlmostEqual(r['fraction'][0, 0], .4)
        self.assertTrue(r['roi_truncated_object'][0, 0])
        self.assertEqual(r['union_area_m2'], 40)

    def test_no_100m2_filter(self):
        r = manual_roof_fractions({'a': box(0, 0, 2, 2)}, self.grid)
        self.assertAlmostEqual(r['fraction'][0, 0], .04)

    def test_zero_not_negative_evidence(self):
        r = manual_roof_fractions({}, self.grid)
        self.assertFalse(r['known_background'].any())
        self.assertFalse(r['training_admission'].any())
        self.assertEqual(r['union_area_m2'], 0)

    def test_invalid_fails_closed(self):
        with self.assertRaises(ValueError):
            manual_roof_fractions({'a': Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])}, self.grid)

    def test_geometry_unchanged(self):
        g = box(-2, 0, 12, 10); before = g.wkb
        r = manual_roof_fractions({'a': g}, self.grid)
        self.assertEqual(g.wkb, before)
        self.assertLess(r['area_conservation_error_m2'], 1e-9)


if __name__ == '__main__':
    unittest.main()
