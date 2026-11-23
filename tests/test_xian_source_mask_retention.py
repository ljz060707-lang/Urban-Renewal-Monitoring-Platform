import hashlib,unittest
from shapely.geometry import box,mapping
from xian_r5.source_mask_retention import retain_source_masks


class SourceMaskRetentionTests(unittest.TestCase):
    def fixture(self):
        geom=box(0,0,10,10);records=[]
        for index in range(3):records.append({'candidate_id':'g_MASK_'+str(index),'group_id':'g',
            'source_geometry':mapping(geom),'source_geometry_wkb_sha256':hashlib.sha256(geom.wkb).hexdigest(),
            'positive_prompt_pixels_missing':0,'negative_prompt_pixels_included':0,'physical_id':None,
            'complete_roof_verified':False,'ground_footprint':False,'source_parent_component':'parent',
            'source_image_sha256':'a'*64,'source_window':[0,0,20,20],'source_observed_date':'2018-11-01',
            'experiment_year':2019,'model_score':index*.4})
        choices=[{'group_id':'g','source_role':'VISIBLE_EXTENT_CANDIDATE','mask_index':0,
            'source_visual_basis':'Observed source roof outline; higher-score alternatives omit substantial roof portions.',
            'reference_or_target_position_used':False}]
        return records,choices

    def test_explicit_visual_selection_not_highest_score(self):
        r=retain_source_masks(*self.fixture());self.assertEqual(r['records'][0]['retained_candidate_id'],'g_MASK_0')
        self.assertFalse(r['records'][0]['model_score_used_to_select'])

    def test_candidate_not_ground_truth(self):
        r=retain_source_masks(*self.fixture())['records'][0]
        self.assertFalse(r['may_generate_training_label']);self.assertFalse(r['formal_acceptance'])
        self.assertFalse(r['full_physical_roof_verified'])

    def test_partial_surface_not_complete_shape_evidence(self):
        args=self.fixture();args[1][0]['source_role']='SURFACE_ONLY'
        self.assertFalse(retain_source_masks(*args)['records'][0]['shape_diagnostic_eligible'])

    def test_extent_risk_not_complete_shape_evidence(self):
        args=self.fixture();args[1][0]['source_role']='EXTENT_RISK'
        self.assertFalse(retain_source_masks(*args)['records'][0]['shape_diagnostic_eligible'])

    def test_partial_extent_excluded(self):
        args=self.fixture();args[1][0]['source_role']='PARTIAL_EXTENT'
        self.assertFalse(retain_source_masks(*args)['records'][0]['shape_diagnostic_eligible'])

    def test_unresolved_no_selected_geometry(self):
        args=self.fixture();args[1][0].update(source_role='UNRESOLVED',mask_index=None)
        self.assertEqual(retain_source_masks(*args)['retained_masks'],0)

    def test_all_groups_required(self):
        args=self.fixture();args[1].clear()
        with self.assertRaises(ValueError):retain_source_masks(*args)

    def test_target_guided_choice_rejected(self):
        args=self.fixture();args[1][0]['reference_or_target_position_used']=True
        with self.assertRaises(ValueError):retain_source_masks(*args)

    def test_prompt_violation_rejected(self):
        args=self.fixture();args[0][0]['negative_prompt_pixels_included']=1
        with self.assertRaises(ValueError):retain_source_masks(*args)

    def test_geometry_drift_fail_closed(self):
        args=self.fixture();args[0][0]['source_geometry']=mapping(box(0,0,11,10))
        with self.assertRaises(ValueError):retain_source_masks(*args)

    def test_uncertainty_and_physical_id_not_invented(self):
        r=retain_source_masks(*self.fixture())['records'][0]
        self.assertIsNone(r['localization_bound_m']);self.assertIsNone(r['physical_id'])


if __name__=='__main__':unittest.main()
