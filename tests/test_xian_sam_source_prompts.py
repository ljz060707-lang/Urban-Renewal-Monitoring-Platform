import unittest
from xian_r5.sam_source_prompts import source_segmentation_prompts


class SamSourcePromptTests(unittest.TestCase):
    def fixture(self):
        records=[{'observation_id':'o1','source_parent_component':'p','target_reference_position_used':False,
            'complete_instance':False,'source_packet_image_sha256':'a'*64,'native_vertices_xy':[[10,10],[20,10],[20,20],[10,20]]}]
        packets={'p':{'local_image_shape':[100,100],'image_path':'raw.png','image_sha256':'a'*64,
            'native_rgb_pixel_sha256':'b'*64,'source_window':[0,0,100,100]}}
        groups=[{'group_id':'SAM_ROOF_CANDIDATE_1','source_parent_component':'p','surface_observation_ids':['o1'],
            'negative_points_pixel_index':[[30,30]],'negative_feature_basis':'Visible non-target source-image region; not a stable-ground registration tie.'}]
        return records,packets,groups

    def test_source_edge_to_pixel_index(self):
        r=source_segmentation_prompts(*self.fixture())[0]
        self.assertEqual(r['point_coords_pixel_index_xy'][0],[14.5,14.5]);self.assertEqual(r['point_labels'],[1,0])

    def test_not_registration_or_ground_precision(self):
        r=source_segmentation_prompts(*self.fixture())[0]
        self.assertIsNone(r['reference_point']);self.assertIsNone(r['localization_bound_m']);self.assertFalse(r['formal_review'])

    def test_target_guidance_rejected(self):
        args=self.fixture();args[0][0]['target_reference_position_used']=True
        with self.assertRaises(ValueError):source_segmentation_prompts(*args)

    def test_other_parent_rejected(self):
        args=self.fixture();args[0][0]['source_parent_component']='other'
        with self.assertRaises(ValueError):source_segmentation_prompts(*args)

    def test_other_packet_rejected(self):
        args=self.fixture();args[0][0]['source_packet_image_sha256']='c'*64
        with self.assertRaises(ValueError):source_segmentation_prompts(*args)

    def test_no_full_instance_promotion(self):
        args=self.fixture();args[0][0]['complete_instance']=True
        with self.assertRaises(ValueError):source_segmentation_prompts(*args)

    def test_outside_negative_rejected(self):
        args=self.fixture();args[2][0]['negative_points_pixel_index']=[[101,30]]
        with self.assertRaises(ValueError):source_segmentation_prompts(*args)

    def test_duplicate_groups_rejected(self):
        args=self.fixture();args[2].append(args[2][0])
        with self.assertRaises(ValueError):source_segmentation_prompts(*args)


if __name__=='__main__':unittest.main()
