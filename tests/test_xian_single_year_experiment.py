import unittest
import numpy as np
from xian_r5.single_year_experiment import nominal_year_binding,experimental_pseudolabel,experiment_batch,alignment_experiment_readiness

class SingleYearTests(unittest.TestCase):
    def setUp(self):
        self.f=np.array([[.43,.44],[0.,-9999.]])
        self.scope=np.ones((2,2));self.q=np.zeros((10,2,2),np.uint16);self.q[0]=1
        self.valid=np.ones((2,2),bool)
    def build(self):
        return experimental_pseudolabel(self.f,self.scope,self.q,self.valid,
            allow_unverified_geometry=True,allow_assumed_background=True)
    def test_nominal_year_preserves_observed_date(self):
        r=nominal_year_binding(2019,'2018-11-01')
        self.assertEqual(r['observed_date'],'2018-11-01')
        self.assertEqual(r['experiment_year'],2019)
        self.assertFalse(r['permits_temporal_change_inference'])
    def test_wrong_year_rejected(self):
        with self.assertRaises(ValueError):nominal_year_binding(2018,None)
    def test_unknown_date_not_invented(self):
        self.assertIsNone(nominal_year_binding(2019,None)['observed_date'])
    def test_risk_opt_in_required(self):
        with self.assertRaises(ValueError):experimental_pseudolabel(self.f,self.scope,self.q,self.valid)
    def test_strict_threshold_and_ignore(self):
        np.testing.assert_array_equal(self.build()['label'],[[0,1],[0,255]])
    def test_seam_excluded(self):
        self.q[6,0,1]=1
        self.assertEqual(self.build()['label'][0,1],255)
    def test_boundary_excluded(self):
        self.q[5,0,1]=1
        self.assertEqual(self.build()['label'][0,1],255)
    def test_conflict_excluded(self):
        self.q[7,0,1]=1
        self.assertEqual(self.build()['label'][0,1],255)
    def test_partial_scope_excluded(self):
        self.scope[0,1]=.5
        self.assertEqual(self.build()['label'][0,1],255)
    def test_unknown_vector_excluded(self):
        self.q[0,0,1]=2
        self.assertEqual(self.build()['label'][0,1],255)
    def test_input_invalid_excluded(self):
        self.valid[0,1]=False
        self.assertEqual(self.build()['label'][0,1],255)
    def test_experiment_not_formal_gt(self):
        r=self.build()
        self.assertGreater(r['counts']['experimental_samples'],0)
        self.assertEqual(r['counts']['scientifically_qualified_gt'],0)
    def test_read_only_batch_smoke(self):
        p=self.build();r=experiment_batch(np.ones((6,2,2)),p['label'],p['experiment_weight'])
        self.assertEqual(r['target'][1,1],-100)
        self.assertEqual(r['optimization_steps'],0)
    def test_weight_ignore_mismatch_rejected(self):
        p=self.build()
        with self.assertRaises(ValueError):experiment_batch(np.ones((6,2,2)),p['label'],np.ones((2,2)))
    def test_invalid_fraction_rejected(self):
        self.f[0,1]=np.nan
        with self.assertRaises(ValueError):self.build()
    def test_inputs_unchanged(self):
        before=self.f.copy();self.build();np.testing.assert_array_equal(before,self.f)
    def cell(self,kernel=20,dx=0,vector=(0,0)):
        return dict(region_id='a',candidate_correction_m=vector,full_feature_kernel_supported_pixels=kernel,
            correlation_diagnostic={'best':{'dx_px':dx,'dy_px':0},'search_radius_px':3})
    def test_kernel_zero_not_recommended(self):
        self.assertFalse(alignment_experiment_readiness([self.cell(kernel=0)])[0]['candidate_experiment_supported'])
    def test_search_limit_not_recommended(self):
        self.assertFalse(alignment_experiment_readiness([self.cell(dx=3)])[0]['candidate_experiment_supported'])
    def test_unknown_not_zero_readiness(self):
        self.assertIn('NO_NUMERIC_VECTOR',alignment_experiment_readiness([self.cell(vector=None)])[0]['reasons'])
    def test_both_risks_recorded(self):
        self.assertEqual(len(alignment_experiment_readiness([self.cell(kernel=0,dx=-3)])[0]['reasons']),2)
    def test_supported_still_not_formal(self):
        r=alignment_experiment_readiness([self.cell()])[0]
        self.assertTrue(r['candidate_experiment_supported']);self.assertFalse(r['formal_alignment_accepted'])
    def test_negative_kernel_count_rejected(self):
        with self.assertRaises(ValueError):alignment_experiment_readiness([self.cell(kernel=-1)])
    def test_readiness_input_unchanged(self):
        c=self.cell();alignment_experiment_readiness([c]);self.assertEqual(c['candidate_correction_m'],(0,0))

if __name__=='__main__':unittest.main()
