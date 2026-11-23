import unittest
import numpy as np
from xian_r5.alignment.source_template_alignment import source_template_check

class SourceTemplateTests(unittest.TestCase):
    def fixture(self,shift=(1,-1)):
        source=np.random.default_rng(26).random((64,64,3))*255
        target=np.roll(source,shift,axis=(0,1));mask=np.zeros((64,64),bool);mask[5:-5,5:-5]=True
        return source,target,mask
    def run_case(self,a,b,m):return source_template_check(a,b,m,np.ones(m.shape,bool),np.ones(m.shape,bool))
    def test_source_to_reference_correction_sign(self):
        a,b,m=self.fixture();r=self.run_case(a,b,m)
        self.assertEqual((r['best']['correction_dx_px'],r['best']['correction_dy_px']),(-1,1))
        self.assertTrue(r['engineering_experiment_supported'])
    def test_all_candidates_same_population(self):
        a,b,m=self.fixture();r=self.run_case(a,b,m)
        self.assertEqual(len({x['n'] for x in r['candidate_scores']}),1)
        self.assertEqual(r['candidate_count'],49)
    def test_source_exclusion_invariance(self):
        a,b,m=self.fixture();r=self.run_case(a,b,m);changed=a.copy();changed[~m]=50000
        s=self.run_case(changed,b,m)
        self.assertEqual(r['candidate_scores'],s['candidate_scores'])
    def test_no_target_annotation_mask(self):
        a,b,m=self.fixture();self.assertFalse(self.run_case(a,b,m)['target_annotation_mask_used'])
    def test_constant_no_peak(self):
        m=np.ones((64,64),bool);r=self.run_case(np.ones((64,64,3)),np.ones((64,64,3)),m)
        self.assertIsNone(r['best'])
    def test_insufficient_source_not_filled(self):
        a,b,m=self.fixture();m[:]=False;r=self.run_case(a,b,m)
        self.assertEqual(r['candidate_count'],0)
    def test_search_limit_rejected(self):
        a,b,m=self.fixture((0,3));r=self.run_case(a,b,m)
        self.assertIn('FIT_PEAK_AT_SEARCH_LIMIT',r['reasons'])
    def test_zero_not_nonzero_correction_evidence(self):
        a,_,m=self.fixture();r=self.run_case(a,a,m)
        self.assertIn('ZERO_SELF_COMPARISON_NOT_NONZERO_CORRECTION_EVIDENCE',r['reasons'])
    def test_no_metric_precision_inferred(self):
        a,b,m=self.fixture();r=self.run_case(a,b,m)
        self.assertIsNone(r['relative_registration_uncertainty_m']);self.assertFalse(r['formal_alignment_accepted'])
    def test_numeric_mask_rejected(self):
        a,b,m=self.fixture()
        with self.assertRaises(ValueError):self.run_case(a,b,m.astype(float))

if __name__=='__main__':unittest.main()
