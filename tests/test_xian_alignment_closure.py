"""Coordinate conventions and same-region independent structures, not invented ties."""
import unittest
import numpy as np
from affine import Affine
from shapely.geometry import box
from xian_r5.alignment.numerical_closure import content_correction,validate_same_region_design,numerical_gate


class ClosureTests(unittest.TestCase):
    def test_content_inverse_in_north_up_metres(self):
        self.assertEqual(content_correction(1.25,-.75,Affine(10,0,308460,0,-10,3794750)),(-12.5,-7.5))

    def test_gdal_six_tuple_must_be_converted_explicitly(self):
        with self.assertRaises(ValueError):content_correction(1.25,-.75,(308460,10,0,3794750,0,-10))

    def test_half_pixel_does_not_change_translation(self):
        a=Affine(10,0,0,0,-10,0)
        self.assertEqual(content_correction(.25,0,a*Affine.translation(.5,.5)),(-2.5,0))

    def test_cross_resolution_same_metric_content_shift(self):
        self.assertEqual(content_correction(6.25,-3.75,Affine.scale(2,-2)),(-12.5,-7.5))

    def test_group_repetition_is_not_independence(self):
        rows=[dict(region_id='R',independence_group='J',fold='FIT',anchor_xy=[1,1])]*4
        report=validate_same_region_design(rows,'R',box(0,0,10,10))
        self.assertEqual(report['candidate_fit_groups'],1)
        self.assertFalse(report['formal_fit_check_eligible'])

    def test_different_region_is_not_check(self):
        rows=[dict(region_id='OTHER',independence_group='J',fold='CHECK',anchor_xy=[1,1])]
        with self.assertRaises(ValueError):validate_same_region_design(rows,'R',box(0,0,10,10))

    def test_different_edges_of_same_junction_are_not_check(self):
        rows=[dict(region_id='R',independence_group='J',fold=f,anchor_xy=[1,1]) for f in ['FIT','CHECK']]
        with self.assertRaises(ValueError):validate_same_region_design(rows,'R',box(0,0,10,10))

    def test_two_distinct_same_region_candidates_are_not_precision_ties(self):
        rows=[dict(region_id='R',independence_group=g,fold=f,anchor_xy=[x,1])
              for g,f,x in [('J1','FIT',1),('J2','CHECK',8)]]
        report=validate_same_region_design(rows,'R',box(0,0,10,10))
        self.assertTrue(report['same_region_candidate_design'])
        self.assertFalse(report['formal_fit_check_eligible'])

    def test_high_score_never_overrides_failed_numeric_gate(self):
        self.assertFalse(numerical_gate([.8,5.872],convention_error_m=0))

    def test_unknown_error_fails_closed(self):
        self.assertFalse(numerical_gate([np.nan],convention_error_m=0))


if __name__=='__main__':unittest.main()
