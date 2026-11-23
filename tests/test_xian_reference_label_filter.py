import unittest
import numpy as np
from xian_r5.reference_label_filter import reference_label_filter


class ReferenceFilterTests(unittest.TestCase):
    def data(self):
        return [np.array([[1,0,1,255]],np.uint8),np.array([[1,1,1,0]],np.float32),
                np.array([[.8,.2,.6,-9999]]),np.array([[.7,.1,.1,1.]]),
                np.zeros((1,4),bool),np.zeros((1,4),bool)]

    def test_only_removes(self):
        r=reference_label_filter(*self.data())
        np.testing.assert_array_equal(r['label'],[[1,0,255,255]])
        self.assertEqual(r['counts']['new_samples_outside_inherited_selection'],0)

    def test_positive_only_has_no_background(self):
        r=reference_label_filter(*self.data())
        np.testing.assert_array_equal(r['positive_only_label'],[[1,255,255,255]])

    def test_background_is_assumption(self):
        r=reference_label_filter(*self.data())
        self.assertIn('NOT_VERIFIED_ABSENCE',r['background_semantics'])
        self.assertFalse(r['quality'][7:].any())

    def test_overlap_excluded(self):
        a=self.data();a[4][0,0]=True
        r=reference_label_filter(*a);self.assertEqual(r['label'][0,0],255)

    def test_truncated_excluded(self):
        a=self.data();a[5][0,1]=True
        r=reference_label_filter(*a);self.assertEqual(r['label'][0,1],255)

    def test_combined_reasons_not_double_counted(self):
        a=self.data();a[4][0,2]=True;a[5][0,2]=True
        r=reference_label_filter(*a);self.assertEqual(r['exclusion_flags'][0,2],7)
        self.assertEqual(r['counts']['union_reference_exclusions'],1)

    def test_equal_tau_is_background(self):
        a=self.data();a[2][0,1]=.43;a[3][0,1]=.43
        self.assertEqual(reference_label_filter(*a)['label'][0,1],0)

    def test_invalid_cached_label_fails(self):
        a=self.data();a[0][0,0]=0
        with self.assertRaises(ValueError):reference_label_filter(*a)

    def test_invalid_reference_fails(self):
        a=self.data();a[3][0,0]=np.nan
        with self.assertRaises(ValueError):reference_label_filter(*a)

    def test_input_unchanged(self):
        a=self.data();before=[v.copy() for v in a];reference_label_filter(*a)
        for left,right in zip(a,before):np.testing.assert_array_equal(left,right)


if __name__=='__main__':unittest.main()
