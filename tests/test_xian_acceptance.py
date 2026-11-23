"""Synthetic contract regressions; no image, checkpoint, or protected test reads."""
import unittest
import numpy as np
from affine import Affine
from shapely.affinity import translate
from shapely.geometry import box, mapping
from xian_r5.acceptance import (gate, vector, tie_diagnostics, whole_object_field,
                               unresolved_building, sha256_geometry, hard_labels)
from xian_r5.label_transfer import Grid, aggregate, bounded_transport


class AcceptanceTests(unittest.TestCase):
    def ties(self):
        return [dict(tie_id=str(i), FIT_or_HOLDOUT="FIT" if i < 3 else "HOLDOUT",
                     independence_group=str(i), precision_eligible=True,
                     within_exact_support=True, feature_class="STABLE_NONBUILDING_GROUND_FEATURE",
                     ge_x=i, ge_y=i, reference_x=i+.25, reference_y=i-.75,
                     ge_localization_bound_m=.1, reference_localization_bound_m=.2)
                for i in range(6)]

    def record(self):
        g=box(1,1,2,2)
        return dict(observation_id="original", original_id="original", geometry=mapping(g),
                    geometry_sha256=sha256_geometry(g), actual_date=None, source_support_complete=True)

    def test_unknown_not_pass_or_fail(self):
        self.assertEqual(gate({"x": None}), "UNRESOLVED")
        self.assertEqual(gate({}), "UNRESOLVED")
    def test_not_run_distinct(self): self.assertEqual(gate({"x": True},executed=False),"NOT_RUN")
    def test_failure_dominates_missing(self): self.assertEqual(gate({"x":False,"y":None}),"FAIL")
    def test_boolean_not_number(self):
        with self.assertRaises(ValueError): vector(True,0)
    def test_partial_unknown_vector_rejected(self):
        with self.assertRaises(ValueError): vector(None,0)
    def test_null_distinct_from_true_zero(self):
        self.assertIsNone(vector(None,None)); self.assertEqual(vector(0,0),[0.,0.])
    def test_subpixel_not_rounded(self): self.assertEqual(vector(.25,-.75),[.25,-.75])
    def test_coordinate_sign_and_inverse(self):
        a=Affine(10,0,308460,0,-10,3794750)
        x,y=a*(0,0); x1,y1=a*(1.25,-.75)
        self.assertEqual((x1-x,y1-y),(12.5,7.5))
        self.assertEqual((~a*(x1,y1)),(1.25,-.75))
        g=box(x,y-1,x+1,y)
        self.assertTrue(translate(translate(g,12.5,7.5),-12.5,-7.5).equals_exact(g,0))
    def test_additive_uncertainty(self):
        r=tie_diagnostics(self.ties(),.25,-.75)
        self.assertAlmostEqual(r["relative_registration_uncertainty_m"],.3)
        self.assertEqual(r["authorization"],"NOT_AUTHORIZED_BY_DIAGNOSTICS")
    def test_insufficient_groups_null_uncertainty(self):
        r=tie_diagnostics(self.ties()[:5],.25,-.75)
        self.assertIsNone(r["relative_registration_uncertainty_m"])
    def test_missing_ties_null_metrics(self):
        r=tie_diagnostics([],None,None)
        self.assertIsNone(r["check_median_m"]); self.assertEqual(r["numerical_gate"],"UNRESOLVED")
    def test_repeated_structure_not_independent(self):
        t=self.ties();t[-1]["independence_group"]="0"
        with self.assertRaises(ValueError): tie_diagnostics(t,.25,-.75)
    def test_roof_not_ground_feature(self):
        t=self.ties();t[0]["feature_class"]="ROOF"
        with self.assertRaises(ValueError): tie_diagnostics(t,.25,-.75)
    def test_outside_support_rejected(self):
        t=self.ties();t[0]["within_exact_support"]=False
        with self.assertRaises(ValueError): tie_diagnostics(t,.25,-.75)
    def test_negative_bound_rejected(self):
        t=self.ties();t[0]["ge_localization_bound_m"]=-.1
        with self.assertRaises(ValueError): tie_diagnostics(t,.25,-.75)
    def test_nonfinite_rejected(self):
        t=self.ties();t[0]["reference_x"]=float("nan")
        with self.assertRaises(ValueError): tie_diagnostics(t,.25,-.75)
    def test_median_does_not_replace_upper_bound(self):
        t=self.ties();t[-1]["reference_localization_bound_m"]=3
        r=tie_diagnostics(t,.25,-.75)
        self.assertEqual(r["check_median_m"],0);self.assertEqual(r["numerical_gate"],"FAIL")
    def test_forged_derived_uncertainty_fails(self):
        t=self.ties();t[0]["adjusted_bound_m"]=0
        with self.assertRaises(ValueError): tie_diagnostics(t,.25,-.75)
    def test_original_id_and_hash_preserved(self):
        r=self.record();a=unresolved_building(r)
        self.assertEqual(a["original_geometry_sha256"],r["geometry_sha256"])
        self.assertIsNone(a["applied_vector_m"]);self.assertIsNone(a["physical_id"])
    def test_child_does_not_inherit_parent(self):
        r=self.record();r["original_id"]="parent"
        with self.assertRaises(ValueError):unresolved_building(r)
    def test_applied_geometry_smuggling_rejected(self):
        r=self.record();r["registered_geometry"]=r["geometry"]
        with self.assertRaises(ValueError):unresolved_building(r)
    def test_source_and_temporal_reasons_separate(self):
        r=self.record();r["source_support_complete"]=False
        a=unresolved_building(r)
        self.assertEqual(a["application_status"],"UNRESOLVED_SOURCE_SUPPORT")
        self.assertIn("UNRESOLVED_TEMPORAL_COMPATIBILITY",a["unresolved_reasons"])
        self.assertIn("UNRESOLVED_BUILDING_PARALLAX",a["unresolved_reasons"])
    def test_whole_object_crossing_equal_fields(self):
        g=box(0,0,2,1);fs=[("a",box(0,0,1,1),(.25,.5)),("b",box(1,0,2,1),(.25,.5))]
        a=whole_object_field(g,fs);b=whole_object_field(g,list(reversed(fs)))
        self.assertEqual(a,b);self.assertEqual(a["vector_m"],[.25,.5])
        self.assertNotEqual(a["status"],"ACCEPTED_FOR_SHADOW_ALIGNMENT")
    def test_conflict_not_smoothed(self):
        r=whole_object_field(box(0,0,2,1),[("a",box(0,0,1,1),(1,0)),("b",box(1,0,2,1),(2,0))])
        self.assertIsNone(r["vector_m"])
    def test_partial_support_no_extrapolation(self):
        r=whole_object_field(box(0,0,2,1),[("a",box(0,0,1,1),(1,0))])
        self.assertEqual(r["status"],"UNRESOLVED_SOURCE_SUPPORT")
    def test_unknown_overlap_blocks_whole_field(self):
        r=whole_object_field(box(0,0,2,1),[("a",box(0,0,2,1),(1,0)),("b",box(0,0,1,1),None)])
        self.assertIsNone(r["vector_m"])
    def test_full_pixel_denominator_not_support(self):
        grid=Grid(Affine(10,0,0,0,-10,10),1,1,"EPSG:32649")
        r=aggregate(grid,{"Roof":box(0,0,5,10)},box(0,0,5,10))
        self.assertEqual(r["fractions"][0,0,0],.5)
        self.assertEqual(r["valid"][0,0],.5);self.assertEqual(r["unknown"][0,0],.5)
        self.assertEqual(r["area_conservation_error_m2"],0)
    def test_source_support_clipped_before_translation(self):
        moved,valid=bounded_transport({"outside":box(-1,0,0,1)},(1,0),box(0,0,1,1),box(0,0,2,1))
        self.assertEqual(moved["outside"].area,0)
        self.assertEqual(valid.area,1)
    def test_unknown_and_nodata_stay_ignore(self):
        f=np.array([0.,.43,.44,-9999,np.nan]);e=np.array([False,True,True,False,False])
        self.assertEqual(hard_labels(f,e,threshold=.43,operator=">").tolist(),[255,0,1,255,255])
        self.assertEqual(hard_labels(f,e,threshold=.43,operator=">=")[1],1)
    def test_rule_requires_operator(self):
        with self.assertRaises(ValueError):hard_labels(np.array([.5]),np.array([True]),threshold=.43,operator="auto")
    def test_eligible_nodata_fails(self):
        with self.assertRaises(ValueError):hard_labels(np.array([-9999.]),np.array([True]),threshold=.43,operator=">")

if __name__=="__main__":unittest.main()
