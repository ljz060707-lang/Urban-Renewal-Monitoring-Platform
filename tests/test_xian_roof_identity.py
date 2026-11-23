import unittest
from shapely.geometry import box
from shapely.affinity import translate
from xian_r5.roof_identity import candidates, validate_table


class RoofIdentityTests(unittest.TestCase):
    def test_geometry_is_unchanged(self):
        g=box(0,0,10,20); before=g.wkb
        edges,status=candidates({'obs':g},{'gt':translate(g,xoff=5)})
        self.assertEqual(g.wkb,before)
        self.assertEqual(status['obs'],'PLAUSIBLE')
        self.assertEqual(edges[0]['shape_iou'],1.)
        self.assertFalse(edges[0]['GT_USED_FOR_GEOMETRY'])

    def test_identical_shape_does_not_confirm(self):
        _,status=candidates({'obs':box(0,0,10,20)},{'gt':box(0,0,10,20)})
        self.assertNotEqual(status['obs'],'CONFIRMED')

    def test_competing_references_remain_ambiguous(self):
        g=box(0,0,10,20)
        _,status=candidates({'obs':g},{'a':g,'b':translate(g,xoff=4)})
        self.assertEqual(status['obs'],'AMBIGUOUS')

    def test_merge_is_not_forced(self):
        _,status=candidates({'merged':box(0,0,22,20)},
                           {'a':box(0,0,10,20),'b':box(12,0,22,20)})
        self.assertNotEqual(status['merged'],'CONFIRMED')

    def test_no_reference_retains_denominator(self):
        edges,status=candidates({'a':box(0,0,2,2)},{})
        self.assertEqual(edges,[]);self.assertEqual(status,{'a':'UNRESOLVED'})

    def test_shape_not_nearest_centroid(self):
        g=box(0,0,10,20)
        edges,_=candidates({'obs':g},{'near_wrong':box(0,0,20,10),'far_right':translate(g,xoff=15)})
        best=next(e for e in edges if e['is_best_diagnostic'])
        self.assertEqual(best['reference_id'],'far_right')

    def test_direction_and_boundary_recorded(self):
        edges,_=candidates({'a':box(0,0,10,20)},{'b':box(2,2,12,22)})
        self.assertEqual(edges[0]['orientation_difference_rad'],0.)
        self.assertEqual(edges[0]['centered_boundary_hausdorff_m'],0.)
        self.assertEqual(edges[0]['temporal_continuity'],'NOT_VERIFIED')

    def test_no_match_does_not_imply_demolition(self):
        _,status=candidates({'a':box(0,0,10,20)},{'b':box(1000,1000,1010,1020)})
        self.assertEqual(status['a'],'UNRESOLVED')

    def test_dangling_id_fails(self):
        row={'roof_obs_id':'o','roof_gt_id':'missing','match_status':'PLAUSIBLE'}
        with self.assertRaises(ValueError):validate_table([row],{'o'},{'g'},set(),set())

    def test_confirmed_requires_explicit_evidence(self):
        row={'roof_obs_id':'o','match_status':'CONFIRMED'}
        with self.assertRaises(ValueError):validate_table([row],{'o'},set(),set(),set())

    def test_gt_geometry_use_forbidden(self):
        row={'roof_obs_id':'o','match_status':'PLAUSIBLE','GT_USED_FOR_GEOMETRY':True}
        with self.assertRaises(ValueError):validate_table([row],{'o'},set(),set(),set())

    def test_missing_observation_fails(self):
        with self.assertRaises(ValueError):validate_table([],{'o'},set(),set(),set())

    def test_candidate_track_id_must_exist(self):
        row={'roof_obs_id':'o','match_status':'PLAUSIBLE','physical_id':'invented'}
        with self.assertRaises(ValueError):validate_table([row],{'o'},set(),set(),set())

    def test_zero_unknown_confidence_not_inferred(self):
        rows=[{'roof_obs_id':'o','match_status':'UNRESOLVED','physical_id':None}]
        counts=validate_table(rows,{'o'},set(),set(),set())
        self.assertEqual(counts['UNRESOLVED'],1)


if __name__=='__main__':unittest.main()
