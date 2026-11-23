"""Public mask-attribution and evidence-group gates; no mocked matching."""
import unittest
import numpy as np
from affine import Affine
from shapely.geometry import box
from unittest.mock import patch
from xian_r5.label_transfer import Grid,bounded_transport
from xian_r5.source_partition import SourceRegion,source_partition_transport
from xian_r5.coherent_support import coalesce_equal_field_regions
from xian_r5.alignment.base_tie import image_shift_to_projected_correction
from xian_r5.alignment.supported_ground_features import estimate_supported_translation
from xian_r5.alignment.feature_recovery import exclusion_stages
from xian_r5.alignment.ground_trial import candidate_ground_support, independent_groups


class ExclusionTests(unittest.TestCase):
    def test_nested_search_and_derivative_ledger(self):
        roof = np.zeros((15, 15), bool)
        roof[7, 7] = True
        stages = exclusion_stages(roof, roof, radius_px=3)
        self.assertEqual([int(v.sum()) for v in stages.values()], [1, 1, 49, 81])

    def test_actual_ground_patch_near_roof_is_not_blanket_halo_excluded(self):
        road=np.zeros((15,15),bool);road[7,7]=True
        roof=np.zeros_like(road);roof[7,5]=True
        support=candidate_ground_support(road, roof, np.ones_like(road))
        self.assertTrue(support[7,7])
        self.assertFalse(exclusion_stages(roof,roof)['derivative_halo'][7,7] == False)

    def test_roof_facade_and_unknown_never_enter_candidate_ground(self):
        road=np.ones((3,3),bool);bad=np.zeros_like(road);bad[0,0]=True;bad[0,1]=True
        known=np.ones_like(road);known[1,1]=False
        support=candidate_ground_support(road,bad,known)
        self.assertEqual(int(support.sum()),6)

    def test_same_intersection_cannot_be_fit_and_check(self):
        rows=[dict(independence_group='J1',fold='FIT'),dict(independence_group='J1',fold='CHECK')]
        with self.assertRaises(ValueError): independent_groups(rows)

    def test_repeated_points_count_as_one_structure(self):
        rows=[dict(independence_group='J1',fold='FIT')]*4
        self.assertEqual(independent_groups(rows),dict(FIT=['J1'],CHECK=[]))

    def test_source_support_precedes_shift(self):
        moved,valid=bounded_transport({'outside':box(-5,0,-1,5)},(5,0),box(0,0,10,10),box(0,0,20,20))
        self.assertTrue(moved['outside'].is_empty)

    def test_unresolved_region_never_changes_original(self):
        original=box(2,2,8,8);before=original.wkb
        result=source_partition_transport({'roof':original},[SourceRegion('r','2018-11-01',box(0,0,10,10))],box(0,0,20,20))
        self.assertEqual(result['candidate_geometries'],{})
        self.assertEqual(original.wkb,before)

    def test_subpixel_map_shift_is_not_rounded(self):
        self.assertEqual(image_shift_to_projected_correction(.25,-.75,tuple(Affine(10,0,0,0,-10,0))),(-2.5,-7.5))

    def test_complete_roof_crossing_equal_computational_regions(self):
        regions=[SourceRegion('L','2018-11-01',box(0,0,10,10),(.25,.5),'a'*64),
                 SourceRegion('R','2018-11-01',box(10,0,20,10),(.25,.5),'b'*64)]
        coherent,ledger=coalesce_equal_field_regions(regions)
        result=source_partition_transport({'whole':box(8,2,12,6)},coherent,box(0,0,25,25))
        self.assertEqual(result['candidate_geometries']['whole'].bounds,(8.25,2.5,12.25,6.5))
        self.assertEqual(result['candidate_geometries']['whole'].area,16)

    def test_cross_resolution_and_gdal_affine_mapping(self):
        coarse=Grid(Affine(10,0,308460,0,-10,3794750),256,256,'EPSG:32649')
        fine=coarse.nested(10);xy=coarse.pixel_to_map(80.5,56.5)
        self.assertEqual(xy,(309265,3794185))
        self.assertEqual(fine.map_to_pixel(*xy),(805,565))
        self.assertEqual(Affine.from_gdal(*coarse.affine.to_gdal()),coarse.affine)

    def test_unknown_mask_cannot_be_implicit_ground(self):
        with self.assertRaises(ValueError):
            candidate_ground_support(np.ones((3,3)),np.zeros((3,3),bool),np.ones((3,3),bool))

    def test_numeric_matching_does_not_read_GT(self):
        rng=np.random.default_rng(9);a=rng.integers(0,255,(12,12,3)).astype('uint8');mask=np.ones((12,12),bool)
        with patch('builtins.open',side_effect=AssertionError('No GT/filesystem allowed')):
            result=estimate_supported_translation(a,a,mask,mask,mask,radius=1,min_samples=3)
        self.assertFalse(result['formal_acceptance'])


if __name__ == '__main__':
    unittest.main()
