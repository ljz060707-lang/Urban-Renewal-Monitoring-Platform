import unittest
import numpy as np
from xian_r5.transfer_admission import transfer_admission


class TransferAdmissionTests(unittest.TestCase):
    def fixture(self):
        raw = np.array([[.2, .6], [.5, .1]], dtype=np.float32)
        candidate = np.array([[.6, .2], [-9999, .1]], dtype=np.float32)
        q = np.zeros((10, 2, 2), np.uint16); q[0] = [[1, 1], [2, 1]]; q[2] = [[1, 1], [0, 1]]
        return [raw, candidate, q, np.array([[1, 0], [255, 0]], np.uint8),
                np.array([[1, 1], [0, 1]], np.float32), np.ones((2, 2), bool)]

    def test_all_pixels_have_exclusive_disposition(self):
        r = transfer_admission(*self.fixture())
        self.assertEqual(sum(r['summary']['disposition_counts_exclusive'].values()), 4)
        self.assertEqual(r['summary']['experimental_selected'], 3)

    def test_two_way_label_comparison_not_certification(self):
        r = transfer_admission(*self.fixture())['summary']
        self.assertEqual(r['threshold_flips'], 2); self.assertEqual(r['label_unchanged_comparison_pixels'], 1)
        self.assertFalse(r['label_stability_is_validated_error_envelope']); self.assertEqual(r['measured_gt_eligible'], 0)

    def test_unknown_not_background(self):
        r = transfer_admission(*self.fixture()); self.assertEqual(r['disposition'][1, 0], 1)
        self.assertEqual(r['difference'][1, 0], -9999)

    def test_selection_may_not_override_seam(self):
        args = self.fixture(); args[2][6, 0, 0] = 1
        with self.assertRaises(ValueError): transfer_admission(*args)

    def test_no_formal_gt_flag(self):
        args = self.fixture(); args[2][9, 0, 0] = 1
        with self.assertRaises(ValueError): transfer_admission(*args)

    def test_wrong_threshold_fail_closed(self):
        args = self.fixture(); args[3][0, 0] = 0
        with self.assertRaises(ValueError): transfer_admission(*args)

    def test_weight_binding_fail_closed(self):
        args = self.fixture(); args[4][1, 0] = 1
        with self.assertRaises(ValueError): transfer_admission(*args)

    def test_bad_fraction_fail_closed(self):
        args = self.fixture(); args[1][0, 0] = 1.4
        with self.assertRaises(ValueError): transfer_admission(*args)

    def test_shape_mismatch_fail_closed(self):
        args = self.fixture(); args[2] = args[2][:9]
        with self.assertRaises(ValueError): transfer_admission(*args)

    def test_no_samples_does_not_fake_gt(self):
        args = self.fixture(); args[3][:] = 255; args[4][:] = 0
        r = transfer_admission(*args)['summary']; self.assertEqual(r['experimental_selected'], 0)
        self.assertIsNone(r['maximum_absolute_fraction_difference_diagnostic'])


if __name__ == '__main__': unittest.main()
