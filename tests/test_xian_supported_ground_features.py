"""Frozen failure loop: modifying excluded pixels must not change a tie."""
import os,unittest
import numpy as np
from xian_r5.alignment.base_tie import estimate_gradient_translation

class SupportedGroundFeatureTests(unittest.TestCase):
    def estimate(self,reference,moving,mask):
        if os.environ.get('XIAN_R5_FEATURE_TEST_BACKEND')=='legacy':
            return estimate_gradient_translation(reference,moving,mask,radius=1,min_samples=3,moving_valid_mask=mask)
        from xian_r5.alignment.supported_ground_features import estimate_supported_translation
        return estimate_supported_translation(reference,moving,mask,mask,np.ones(mask.shape,bool),radius=1,min_samples=3)
    def test_excluded_pixels_cannot_change_matching_features(self):
        rng=np.random.default_rng(11)
        ref=np.zeros((10,10,3),np.float32)
        mask=np.zeros((10,10),bool);mask[5,2:8]=True
        ref[mask]=rng.uniform(20,240,(mask.sum(),3))
        altered=ref.copy();altered[~mask]=rng.uniform(1000,5000,((~mask).sum(),3))
        before=self.estimate(ref,ref.copy(),mask)
        after=self.estimate(ref,altered,mask)
        self.assertIsNotNone(before['best'])
        self.assertIsNotNone(after['best'])
        self.assertAlmostEqual(before['best']['corr'],after['best']['corr'],places=10)
        self.assertEqual((before['best']['dx_px'],before['best']['dy_px']),
                         (after['best']['dx_px'],after['best']['dy_px']))
    def test_axis_stencil_only_reads_available_pixels(self):
        from xian_r5.alignment.supported_ground_features import directional_features
        rgb=np.arange(300,dtype=float).reshape(10,10,3)
        mask=np.zeros((10,10),bool);mask[5,2:8]=True
        values,valid=directional_features(rgb,mask)
        self.assertEqual(int(valid[0].sum()),4);self.assertEqual(int(valid[1].sum()),0)
        self.assertFalse(values[~valid].any())
    def test_edges_are_not_reflected_or_wrapped(self):
        from xian_r5.alignment.supported_ground_features import directional_features
        _,valid=directional_features(np.ones((8,8,3)),np.ones((8,8),bool))
        self.assertFalse(valid[0,:,0].any());self.assertFalse(valid[1,0].any())
    def test_masked_unknowns_rejected(self):
        from xian_r5.alignment.supported_ground_features import directional_features
        with self.assertRaises(ValueError):directional_features(np.ones((8,8,3)),np.ma.array(np.ones((8,8),bool)))
    def test_confidence_mask_rejected(self):
        from xian_r5.alignment.supported_ground_features import directional_features
        with self.assertRaises(ValueError):directional_features(np.ones((8,8,3)),np.ones((8,8)))
    def test_constant_image_no_peak(self):
        image=np.ones((20,20,3));mask=np.ones((20,20),bool)
        self.assertIsNone(self.estimate(image,image,mask)['best'])
    def test_mask_preserves_unknown(self):
        image=np.random.default_rng(5).random((20,20,3));mask=np.zeros((20,20),bool)
        self.assertIsNone(self.estimate(image,image,mask)['best'])
    def fixture(self,shift=(1,-1)):
        rng=np.random.default_rng(3);image=rng.random((64,64,3))*255
        moving=np.roll(image,shift,axis=(0,1));mask=np.zeros((64,64),bool);mask[5:-5,5:-5]=True
        return image,moving,mask
    def test_known_offset_and_fixed_check(self):
        from xian_r5.alignment.supported_ground_features import internal_spatial_check
        a,b,m=self.fixture()
        r=internal_spatial_check(a,b,m,m,np.ones(m.shape,bool),radius=3,min_samples=100)
        self.assertEqual((r['fit']['best']['dx_px'],r['fit']['best']['dy_px']),(-1,1))
        self.assertGreater(r['check']['n'],100);self.assertGreater(r['check']['gain'],.01)
        self.assertTrue(r['engineering_experiment_supported'])
    def test_check_not_external_precision(self):
        from xian_r5.alignment.supported_ground_features import internal_spatial_check
        a,b,m=self.fixture();r=internal_spatial_check(a,b,m,m,np.ones(m.shape,bool))
        self.assertFalse(r['formal_fit_holdout_review']);self.assertFalse(r['independent_external_validation'])
        self.assertIsNone(r['relative_registration_uncertainty_m'])
    def test_zero_is_not_correction_evidence(self):
        from xian_r5.alignment.supported_ground_features import internal_spatial_check
        a,_,m=self.fixture();r=internal_spatial_check(a,a,m,m,np.ones(m.shape,bool))
        self.assertIn('ZERO_VECTOR_SELF_COMPARISON_NOT_CORRECTION_EVIDENCE',r['reasons'])
    def test_search_limit_fails_closed(self):
        from xian_r5.alignment.supported_ground_features import internal_spatial_check
        a,b,m=self.fixture((0,3));r=internal_spatial_check(a,b,m,m,np.ones(m.shape,bool))
        self.assertIn('FIT_PEAK_AT_SEARCH_LIMIT',r['reasons'])
    def test_same_pixel_two_axes_not_double_sample_count(self):
        a,_,m=self.fixture();r=self.estimate(a,a,m)
        self.assertLessEqual(r['best']['n'],int(m.sum()))
        self.assertGreater(r['best']['scalar_axis_observations'],r['best']['n'])
    def test_inputs_remain_immutable(self):
        a,b,m=self.fixture();before=a.copy();self.estimate(a,b,m);np.testing.assert_array_equal(a,before)

if __name__=='__main__':unittest.main()
