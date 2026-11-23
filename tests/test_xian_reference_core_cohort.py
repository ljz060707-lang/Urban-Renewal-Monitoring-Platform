import unittest
import numpy as np
from xian_r5.reference_core_cohort import reference_core_cohort


class ReferenceCoreTests(unittest.TestCase):
    def data(self):
        return [np.array([[.9,.9,.9,.3]]),np.array([[.9,.7,.9,.9]]),
                np.array([[False,False,True,False]]),np.zeros((1,4),bool),np.ones((1,4),bool)]

    def test_corroborated_not_raw_positive(self):
        r=reference_core_cohort(*self.data())
        np.testing.assert_array_equal(r['mask'],[[True,False,False,False]])
        self.assertEqual(r['source_prior_pixels'],3)

    def test_truncation_exclusion(self):
        a=self.data();a[3][0,0]=True;self.assertFalse(reference_core_cohort(*a)['mask'].any())

    def test_valid_exclusion(self):
        a=self.data();a[4][0,0]=False;self.assertFalse(reference_core_cohort(*a)['mask'].any())

    def test_exact_08_inclusive(self):
        a=self.data();a[0][0,0]=.8;a[1][0,0]=.8
        self.assertTrue(reference_core_cohort(*a)['mask'][0,0])

    def test_unseen_accuracy_not_claimed(self):
        r=reference_core_cohort(*self.data());self.assertFalse(r['external_unseen_validation'])
        self.assertFalse(r['native_source_roof_purity_certified'])
        self.assertTrue(r['manual_position_used_to_select_source_population'])

    def test_grid_mismatch_fails(self):
        a=self.data();a[1]=np.zeros((2,4))
        with self.assertRaises(ValueError):reference_core_cohort(*a)

    def test_nonfinite_fails(self):
        a=self.data();a[1][0,0]=np.nan
        with self.assertRaises(ValueError):reference_core_cohort(*a)

    def test_immutable_inputs(self):
        a=self.data();b=[x.copy() for x in a];reference_core_cohort(*a)
        for left,right in zip(a,b):np.testing.assert_array_equal(left,right)


if __name__=='__main__':unittest.main()
