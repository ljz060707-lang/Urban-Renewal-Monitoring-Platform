import unittest
from xian_r5.reference_preflight import reference_preflight,EVIDENCE_GATES

class ReferencePreflightTests(unittest.TestCase):
    def row(self,**kw):
        return dict(source_fid='1',geometry_valid=True,fully_inside_roi=True,
                    crosses_source_cell=False,known_history_exposed=False,**kw)
    def gates(self,v=None):return {k:v for k in EVIDENCE_GATES}
    def test_inventory_not_readiness(self):
        r=reference_preflight([self.row()],self.gates())
        self.assertEqual(r['counts']['STRUCTURAL_CANDIDATES'],1)
        self.assertFalse(r['records'][0]['positive_reference_ready'])
    def test_false_and_unknown_distinct(self):
        a=self.gates();a['TIME_OR_CONTINUITY']=False
        r=reference_preflight([self.row()],a)
        self.assertIn('TIME_OR_CONTINUITY_FAILED',r['counts'])
        self.assertIn('TARGET_IMAGE_FRAME_BINDING_MISSING',r['counts'])
    def test_positive_not_complete_background(self):
        a=self.gates(True);a['BACKGROUND_COMPLETENESS']=None
        r=reference_preflight([self.row()],a)
        self.assertTrue(r['records'][0]['positive_reference_ready'])
        self.assertFalse(r['records'][0]['complete_background_reference_ready'])
    def test_known_exposure_not_overruled(self):
        row=self.row();row['known_history_exposed']=True
        r=reference_preflight([row],self.gates(True))
        self.assertFalse(r['records'][0]['positive_reference_ready'])
    def test_partial_support_not_ready(self):
        row=self.row();row['fully_inside_roi']=False
        self.assertFalse(reference_preflight([row],self.gates(True))['records'][0]['positive_reference_ready'])
    def test_seam_not_ready(self):
        row=self.row();row['crosses_source_cell']=True
        self.assertFalse(reference_preflight([row],self.gates(True))['records'][0]['positive_reference_ready'])
    def test_geometry_validity_not_proxy(self):
        row=self.row();row['geometry_valid']=False
        self.assertFalse(reference_preflight([row],self.gates(True))['records'][0]['positive_reference_ready'])
    def test_no_implicit_split_or_training(self):
        r=reference_preflight([self.row()],self.gates(True))
        self.assertEqual(r['calibration_objects_selected'],0)
        self.assertEqual(r['validation_objects_selected'],0)
        self.assertFalse(r['training_admission_performed'])
        self.assertIsNone(r['validated_error_envelope'])
    def test_evidence_integer_rejected(self):
        with self.assertRaises(ValueError):reference_preflight([self.row()],self.gates(1))
    def test_missing_gate_rejected(self):
        a=self.gates();a.pop('TIME_OR_CONTINUITY')
        with self.assertRaises(ValueError):reference_preflight([self.row()],a)
    def test_duplicate_fid_rejected(self):
        with self.assertRaises(ValueError):reference_preflight([self.row(),self.row()],self.gates())
    def test_empty_pool_remains_unselected(self):
        r=reference_preflight([],self.gates())
        self.assertEqual(r['counts']['REFERENCE_INVENTORY'],0)
        self.assertEqual(r['validation_objects_selected'],0)
    def test_historical_alias_best_external_index(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
        from run_xian_reference_preflight import prior_fids
        self.assertEqual(prior_fids({'matches':[{'best_external_index':'42','roof_index':'99'}]}),{'42'})
    def test_historical_nulls_do_not_make_fake_fids(self):
        from run_xian_reference_preflight import prior_fids
        self.assertEqual(prior_fids({'best_external_index':None,'source_fid':'7','fid_count':20}),{'7'})

if __name__=='__main__':unittest.main()
