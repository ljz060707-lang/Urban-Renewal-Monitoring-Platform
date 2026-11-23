import unittest
import numpy as np
from xian_r5.sam_source_candidates import sam_source_candidate


class SamSourceCandidateTests(unittest.TestCase):
    def fixture(self):
        mask=np.zeros((20,20),np.uint8);mask[4:15,4:15]=1;mask[7:10,7:10]=0
        packet={'source_parent_component':'p','image_sha256':'a'*64,'source_window':[0,0,20,20],
            'local_image_shape':[20,20],'source_raster_affine':[.00001,0,109,0,-.00001,34.3],
            'source_raster_crs':'EPSG:4326','native_resampling':'NONE','source_observed_date':'2018-11-01','experiment_year':2019}
        prompt={'group_id':'SAM_ROOF_CANDIDATE_1','source_parent_component':'p','image_sha256':'a'*64,
            'reference_or_target_positions_used':False,'point_coords_pixel_index_xy':[[5,5],[8,8]],'point_labels':[1,0]}
        return mask,packet,prompt,0,.9,'b'*64

    def test_pixel_area_and_holes_preserved(self):
        record,g=sam_source_candidate(*self.fixture())
        self.assertEqual(record['native_union_area_px2'],112);self.assertEqual(len(g.geoms[0].interiors),1)

    def test_no_displacement_or_ground_precision(self):
        r,_=sam_source_candidate(*self.fixture());self.assertIsNone(r['candidate_displacement_m'])
        self.assertIsNone(r['localization_bound_m']);self.assertFalse(r['ground_footprint'])

    def test_score_not_roof_verification(self):
        r,_=sam_source_candidate(*self.fixture());self.assertFalse(r['complete_roof_verified'])
        self.assertFalse(r['prompt_consistency_is_roof_semantic_verification']);self.assertFalse(r['selected'])

    def test_positive_and_negative_checked(self):
        r,_=sam_source_candidate(*self.fixture());self.assertEqual(r['positive_prompt_pixels_missing'],0)
        self.assertEqual(r['negative_prompt_pixels_included'],0)

    def test_disconnected_parts_not_physical_buildings(self):
        args=list(self.fixture());args[0][17:19,17:19]=1
        r,_=sam_source_candidate(*args);self.assertEqual(r['native_connected_polygon_parts'],2)
        self.assertIsNone(r['physical_id'])

    def test_empty_not_background_label(self):
        args=list(self.fixture());args[0][:]=0;r,g=sam_source_candidate(*args)
        self.assertIsNone(g);self.assertEqual(r['status'],'EMPTY_MODEL_MASK_NOT_LABEL')

    def test_unknown_mask_values_rejected(self):
        args=list(self.fixture());args[0][0,0]=255
        with self.assertRaises(ValueError):sam_source_candidate(*args)

    def test_shape_drift_rejected(self):
        args=list(self.fixture());args[0]=args[0][:19]
        with self.assertRaises(ValueError):sam_source_candidate(*args)

    def test_image_binding_drift_rejected(self):
        args=list(self.fixture());args[2]['image_sha256']='c'*64
        with self.assertRaises(ValueError):sam_source_candidate(*args)

    def test_target_guidance_rejected(self):
        args=list(self.fixture());args[2]['reference_or_target_positions_used']=True
        with self.assertRaises(ValueError):sam_source_candidate(*args)

    def test_original_mask_not_mutated(self):
        args=self.fixture();original=args[0].copy();sam_source_candidate(*args)
        self.assertTrue(np.array_equal(original,args[0]))


if __name__=='__main__':unittest.main()
