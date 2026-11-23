import unittest
from copy import deepcopy
from xian_r5.source_prompt_refinement import refine_source_prompt


class SourcePromptRefinementTests(unittest.TestCase):
    def fixture(self):
        base={'group_id':'SAM_ROOF_CANDIDATE_1','source_parent_component':'p','image_shape':[100,100],
            'reference_or_target_positions_used':False,'point_coords_pixel_index_xy':[[14.5,14.5],[50,50]],
            'point_labels':[1,0],'surface_observation_ids':['s']}
        surface={'observation_id':'s','source_parent_component':'p','target_reference_position_used':False,
            'native_vertices_xy':[[10,10],[30,10],[30,30],[10,30]]}
        refinement={'refined_group_id':'SAM_ROOF_CANDIDATE_1_SOURCE_REFINE_01','target_or_reference_position_used':False,
            'source_visual_basis':'Visible source roof portion omitted; add a point inside the frozen source roof surface, not the GT.',
            'additional_positive':[{'source_surface_observation_id':'s','pixel_index_xy':[25,25]}]}
        return base,refinement,[surface]

    def test_base_preserved_extra_source_evidence_added(self):
        args=self.fixture();before=deepcopy(args[0]);r=refine_source_prompt(*args)
        self.assertEqual(before,args[0]);self.assertEqual(r['point_labels'],[1,1,0])

    def test_foreground_outside_roof_evidence_fail_closed(self):
        args=self.fixture();args[1]['additional_positive'][0]['pixel_index_xy']=[35,35]
        with self.assertRaises(ValueError):refine_source_prompt(*args)

    def test_negative_inside_known_surface_fail_closed(self):
        args=self.fixture();args[1]['additional_negative_pixel_index_xy']=[[15,15]]
        with self.assertRaises(ValueError):refine_source_prompt(*args)

    def test_source_negative_allowed_not_ground_tie(self):
        args=self.fixture();args[1]['additional_negative_pixel_index_xy']=[[40,40]]
        r=refine_source_prompt(*args);self.assertEqual(r['point_labels'],[1,1,0,0]);self.assertIsNone(r['reference_point'])

    def test_reference_guided_refinement_prohibited(self):
        args=self.fixture();args[1]['target_or_reference_position_used']=True
        with self.assertRaises(ValueError):refine_source_prompt(*args)

    def test_other_parent_prohibited(self):
        args=self.fixture();args[2][0]['source_parent_component']='other'
        with self.assertRaises(ValueError):refine_source_prompt(*args)

    def test_no_uncertainty_or_review_created(self):
        r=refine_source_prompt(*self.fixture());self.assertIsNone(r['localization_bound_m']);self.assertFalse(r['formal_review'])

    def test_refinement_id_must_bind_parent_group(self):
        args=self.fixture();args[1]['refined_group_id']='unrelated'
        with self.assertRaises(ValueError):refine_source_prompt(*args)


if __name__=='__main__':unittest.main()
