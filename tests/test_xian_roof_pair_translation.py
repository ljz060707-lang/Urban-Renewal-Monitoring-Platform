import copy, unittest
from xian_r5.roof_pair_translation import roof_pair_translation


class PairTranslationTests(unittest.TestCase):
    def rows(self):
        return [{'component_key':str(i),'source_fid':str(i+100),
                 'source_centroid_xy':[x,y],'reference_centroid_xy':[x+1,y-2],
                 'frozen_pair_vector_m':[1,-2]} for i,(x,y) in enumerate(
                     [(x,y) for y in (100,250,400,550) for x in (100,250,400,550)])]

    def test_constant_and_sign(self):
        r=roof_pair_translation(self.rows(),[0,0,640,640])
        self.assertTrue(r['uniform_translation_diagnostically_consistent'])
        self.assertEqual(r['diagnostic_candidate_vector_m'],[1.,-2.])
        self.assertEqual(r['internal_check_max_centroid_residual_m'],0)

    def test_not_formal_or_uncertainty(self):
        r=roof_pair_translation(self.rows(),[0,0,640,640])
        self.assertFalse(r['formal_alignment_accepted']);self.assertFalse(r['training_admitted'])
        self.assertIsNone(r['relative_registration_uncertainty_m'])
        self.assertFalse(r['candidate_applied'])

    def test_no_input_mutation(self):
        rows=self.rows();before=copy.deepcopy(rows)
        roof_pair_translation(rows,[0,0,640,640]);self.assertEqual(rows,before)

    def test_outlier_is_not_removed(self):
        rows=self.rows();rows[0]['reference_centroid_xy'][0]+=8;rows[0]['frozen_pair_vector_m'][0]+=8
        r=roof_pair_translation(rows,[0,0,640,640])
        self.assertFalse(r['uniform_translation_diagnostically_consistent'])
        self.assertEqual(len(r['records']),16)
        self.assertGreater(max(x['candidate_residual_m'] for x in r['records']),2)

    def test_reference_reuse_fails(self):
        rows=self.rows();rows[1]['source_fid']=rows[0]['source_fid']
        with self.assertRaises(ValueError):roof_pair_translation(rows,[0,0,640,640])

    def test_source_reuse_fails(self):
        rows=self.rows();rows[1]['component_key']=rows[0]['component_key']
        with self.assertRaises(ValueError):roof_pair_translation(rows,[0,0,640,640])

    def test_recompute_fail_closed(self):
        rows=self.rows();rows[0]['frozen_pair_vector_m']=[0,0]
        with self.assertRaises(ValueError):roof_pair_translation(rows,[0,0,640,640])

    def test_insufficient_counts_no_fit(self):
        r=roof_pair_translation(self.rows()[:5],[0,0,640,640])
        self.assertIsNone(r['diagnostic_candidate_vector_m'])

    def test_coverage_not_replace_count(self):
        rows=self.rows()
        for i,r in enumerate(rows):
            r['source_centroid_xy']=[100+i,100];r['reference_centroid_xy']=[101+i,98]
        result=roof_pair_translation(rows,[0,0,640,640])
        self.assertFalse(result['uniform_translation_diagnostically_consistent'])
        self.assertIn('CALIBRATION_SPATIAL_COVERAGE_INSUFFICIENT',result['reasons'])

    def test_nonfinite_fails(self):
        rows=self.rows();rows[0]['source_centroid_xy'][0]=float('nan')
        with self.assertRaises(ValueError):roof_pair_translation(rows,[0,0,640,640])

    def test_outside_support_fails(self):
        rows=self.rows();rows[0]['source_centroid_xy'][0]=-1
        with self.assertRaises(ValueError):roof_pair_translation(rows,[0,0,640,640])


if __name__=='__main__':unittest.main()
