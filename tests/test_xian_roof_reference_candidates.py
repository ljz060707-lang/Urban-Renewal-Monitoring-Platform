import unittest
from shapely.geometry import box,Polygon
from shapely.affinity import translate
from xian_r5.roof_reference_candidates import roof_reference_candidates,RoofMatchPolicy

class RoofReferenceTests(unittest.TestCase):
    def setUp(self):self.regions={'a':box(0,0,200,200)}
    def match(self,s,t):return roof_reference_candidates(s,t,self.regions)
    def test_known_shape_offset_is_hypothesis_only(self):
        g=box(50,50,70,60);r=self.match({'s':g},{'r':translate(g,xoff=5,yoff=4)})
        self.assertEqual(r['hypotheses'][0]['roof_to_reference_candidate_m'],[5,4])
        self.assertFalse(r['hypotheses'][0]['physical_identity_verified'])
        self.assertEqual(r['ground_ties_created'],0)
    def test_repeated_roofs_remain_ambiguous(self):
        g=box(50,50,70,60);r=self.match({'s':g},{'r1':g,'r2':translate(g,yoff=20)})
        self.assertEqual(len(r['hypotheses']),0)
        self.assertIn('SOURCE_COMPETITOR_AMBIGUITY',r['decisions'][0]['reasons'])
    def test_many_to_one_not_promoted(self):
        g=box(50,50,70,60);r=self.match({'s1':g,'s2':translate(g,yoff=20)},{'r':g})
        self.assertEqual(len(r['hypotheses']),0)
    def test_nearest_position_does_not_break_shape_tie(self):
        g=box(50,50,70,60);r=self.match({'s':g},{'near':translate(g,xoff=1),'far':translate(g,xoff=20)})
        self.assertEqual(len(r['hypotheses']),0)
    def test_outside_search_scope_ignored(self):
        g=box(50,50,70,60);r=self.match({'s':g},{'r':translate(g,xoff=40)})
        self.assertEqual(r['candidate_pair_count'],0)
    def test_partial_roof_not_selected(self):
        g=box(-1,20,20,40);r=self.match({'s':g},{'r':translate(g,xoff=5)})
        self.assertEqual(r['source_eligible_count'],0)
    def test_seam_not_pieced_into_independent_buildings(self):
        self.regions={'a':box(0,0,100,200),'b':box(100,0,200,200)}
        g=box(90,50,110,70);r=self.match({'s':g},{'r':g})
        self.assertEqual(r['source_eligible_count'],0)
    def test_no_new_min_area_lowering(self):
        g=box(10,10,15,15);r=self.match({'s':g},{'r':g})
        self.assertEqual(r['source_structure']['s']['status'],'BELOW_PARENT_AREA_FILTER')
    def test_invalid_geometry_fail_closed(self):
        with self.assertRaises(ValueError):self.match({'s':Polygon([(0,0),(10,10),(0,10),(10,0)])},{})
    def test_no_reference_returns_unresolved(self):
        r=self.match({'s':box(50,50,70,60)},{})
        self.assertEqual(len(r['hypotheses']),0)
    def test_original_geometry_is_not_snapped(self):
        g=box(50,50,70,60);before=g.wkb;self.match({'s':g},{'r':translate(g,xoff=10)})
        self.assertEqual(g.wkb,before)
    def test_overlap_source_regions_fail_closed(self):
        self.regions={'a':box(0,0,100,100),'b':box(50,0,150,100)}
        with self.assertRaises(ValueError):self.match({}, {})
    def test_nonfinite_policy_rejected(self):
        with self.assertRaises(ValueError):RoofMatchPolicy(maximum_axis_distance_m=float('nan'))

if __name__=='__main__':unittest.main()
