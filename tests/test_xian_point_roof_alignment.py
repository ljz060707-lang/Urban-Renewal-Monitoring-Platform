import unittest
import numpy as np
from xian_r5.alignment.point_roof_alignment import point_roof_check, reconcile_roof_descriptors


class PointRoofTests(unittest.TestCase):
    def fixture(self, shift=(1, -1)):
        a = np.random.default_rng(17).random((64, 64, 3)) * 255
        b = np.roll(a, shift, axis=(0, 1)); m = np.zeros((64, 64), bool); m[4:-4, 4:-4] = True
        return a, b, m

    def check(self, a, b, m):
        return point_roof_check(a, b, m, np.ones(m.shape, bool), np.ones(m.shape, bool))

    def test_known_source_correction(self):
        a, b, m = self.fixture(); r = self.check(a, b, m)
        self.assertEqual((r['best']['correction_dx_px'], r['best']['correction_dy_px']), (-1, 1))
        self.assertTrue(r['engineering_experiment_supported'])

    def test_equal_population_all_offsets(self):
        a, b, m = self.fixture(); r = self.check(a, b, m)
        self.assertEqual(r['candidate_count'], 49); self.assertEqual(len({c['n'] for c in r['candidate_scores']}), 1)

    def test_excluded_source_values_invariant(self):
        a, b, m = self.fixture(); r = self.check(a, b, m); a[~m] = np.nan
        self.assertEqual(r, self.check(a, b, m))

    def test_band_affine_radiometry_invariant(self):
        a, b, m = self.fixture(); r = self.check(a, b, m)
        s = self.check(a, b * np.array([2., 3., .5]) + np.array([150., -100., 4.]), m)
        self.assertEqual(r['best']['correction_dx_px'], s['best']['correction_dx_px'])
        self.assertAlmostEqual(r['check']['gain'], s['check']['gain'], places=12)

    def test_constant_colours_not_spatial_evidence(self):
        a = np.ones((64, 64, 3)) * np.array([10, 100, 200]); m = np.ones((64, 64), bool)
        self.assertIsNone(self.check(a, a, m)['best'])

    def test_zero_not_correction(self):
        a, b, m = self.fixture(); r = self.check(a, a, m)
        self.assertIn('ZERO_SELF_COMPARISON_NOT_NONZERO_CORRECTION_EVIDENCE', r['reasons'])

    def test_boundary_not_recommended(self):
        a, b, m = self.fixture((0, 3)); r = self.check(a, b, m)
        self.assertIn('FIT_PEAK_AT_SEARCH_LIMIT', r['reasons'])

    def test_no_target_mask_no_precision(self):
        a, b, m = self.fixture(); r = self.check(a, b, m)
        self.assertFalse(r['target_annotation_mask_used']); self.assertIsNone(r['relative_registration_uncertainty_m'])
        self.assertFalse(r['formal_alignment_accepted'])

    def test_no_fill_when_no_source(self):
        a, b, m = self.fixture(); m[:] = False
        self.assertEqual(self.check(a, b, m)['candidate_count'], 0)

    def test_bad_mask_fail_closed(self):
        a, b, m = self.fixture()
        with self.assertRaises(ValueError): self.check(a, b, m.astype(float))

    def test_conflict_never_cherry_picked(self):
        a, b, m = self.fixture(); p = self.check(a, b, m)
        d = dict(p); d['best'] = dict(p['best'], correction_dx_px=1)
        self.assertFalse(reconcile_roof_descriptors(p, d)['experimental_recommendation'])
        self.assertIn('ROOF_DESCRIPTOR_DIRECTION_CONFLICT', reconcile_roof_descriptors(p, d)['reasons'])

    def test_matching_descriptor_does_not_override_failed_point_check(self):
        a, b, m = self.fixture(); p = self.check(a, b, m)
        failed = dict(p, reasons=['CHECK_DID_NOT_IMPROVE_BASELINE'])
        self.assertFalse(reconcile_roof_descriptors(failed, p)['experimental_recommendation'])


if __name__ == '__main__': unittest.main()
