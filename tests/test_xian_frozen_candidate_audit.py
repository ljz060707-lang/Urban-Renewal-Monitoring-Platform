"""Frozen-vector replay checks; no tests claim geographic accuracy."""
import unittest
import numpy as np
from xian_r5.alignment.frozen_candidate_audit import audit_frozen_translation


class FrozenCandidateAuditTests(unittest.TestCase):
    def setup_arrays(self):
        ref=np.random.default_rng(18).random((24,24))
        valid=np.ones(ref.shape,dtype=bool)
        return ref,valid

    def test_correct_observed_offset_scores_same_features(self):
        ref,v=self.setup_arrays();mov=np.roll(ref,1,axis=1)
        r=audit_frozen_translation(ref,mov,v,v,v,(1,0),minimum_samples=3,feature_radius=0)
        self.assertAlmostEqual(r['paired_point_support']['candidate_score'],1.)
        self.assertGreater(r['paired_point_support']['candidate_score'],r['paired_point_support']['zero_score'])
        self.assertEqual(r['observed_offset_px'],[1,0])
        self.assertFalse(r['independent_validation'])

    def test_pair_uses_one_common_cohort(self):
        ref,v=self.setup_arrays();mv=v.copy();mv[:,8]=False
        r=audit_frozen_translation(ref,ref,v,mv,v,(1,0),minimum_samples=3,feature_radius=0)
        self.assertEqual(r['paired_point_support']['pixel_count'],24*21)

    def test_no_edge_wraparound(self):
        ref,v=self.setup_arrays();r=audit_frozen_translation(ref,ref,v,v,v,(-2,0),minimum_samples=3,feature_radius=0)
        self.assertEqual(r['paired_point_support']['pixel_count'],24*22)

    def test_complete_feature_kernel_has_less_support(self):
        ref,v=self.setup_arrays();v[:,8]=False
        r=audit_frozen_translation(ref,ref,v,v,np.ones(v.shape,bool),(1,0),minimum_samples=3,feature_radius=4)
        self.assertLess(r['full_kernel_support']['pixel_count'],r['paired_point_support']['pixel_count'])

    def test_empty_support_is_not_a_pass_or_zero_uncertainty(self):
        ref,v=self.setup_arrays();v[:]=False
        r=audit_frozen_translation(ref,ref,v,v,v,(1,0),minimum_samples=3)
        self.assertEqual(r['paired_point_support']['status'],'INSUFFICIENT_COMMON_SUPPORT')
        self.assertIsNone(r['relative_registration_uncertainty_m'])
        self.assertFalse(r['formal_acceptance'])

    def test_zero_shift_self_comparison_is_not_validation(self):
        ref,v=self.setup_arrays();r=audit_frozen_translation(ref,ref,v,v,v,(0,0),minimum_samples=3,feature_radius=0)
        self.assertEqual(r['paired_point_support']['score_difference'],0.)
        self.assertTrue(r['zero_vector_self_comparison'])
        self.assertFalse(r['formal_acceptance'])

    def test_unknowns_and_noninteger_offsets_fail_closed(self):
        ref,v=self.setup_arrays()
        for offset in (None,(1.2,0),(True,0)):
            with self.assertRaises(ValueError):audit_frozen_translation(ref,ref,v,v,v,offset)
        with self.assertRaises(ValueError):audit_frozen_translation(np.ma.array(ref),ref,v,v,v,(0,0))
        with self.assertRaises(ValueError):audit_frozen_translation(ref,ref,v.astype(float),v,v,(0,0))

    def test_constant_features_are_uninformative(self):
        ref,v=self.setup_arrays();ref[:]=1
        r=audit_frozen_translation(ref,ref,v,v,v,(0,0),minimum_samples=3,feature_radius=0)
        self.assertEqual(r['paired_point_support']['status'],'CONSTANT_OR_UNINFORMATIVE_FEATURES')

if __name__=='__main__':unittest.main()
