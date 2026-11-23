import unittest
from shapely.geometry import box
from xian_r5.roof_group_correspondence import roof_group_correspondence


class RoofGroupTests(unittest.TestCase):
    def compare(self,source,refs):return roof_group_correspondence(source,refs,{'r':box(-100,-100,100,100)})

    def test_two_reference_parts_not_two_stable_buildings(self):
        r=self.compare({'s':box(0,0,40,10)},{'a':box(0,0,20,10),'b':box(20,0,40,10)})
        self.assertEqual(r['hypotheses'],1);self.assertEqual(r['records'][0]['relation'],'1:N')
        self.assertEqual(r['stable_independent_pairs_verified'],0)

    def test_single_reference_still_supported(self):
        r=self.compare({'s':box(0,0,20,10)},{'a':box(0,0,20,10)})
        self.assertEqual(r['records'][0]['relation'],'1:1')

    def test_nearest_without_overlap_not_grouped(self):
        r=self.compare({'s':box(0,0,20,10)},{'a':box(21,0,41,10)})
        self.assertEqual(r['hypotheses'],0)

    def test_tiny_competitor_goes_to_strongest_source(self):
        r=self.compare({'s1':box(0,0,20,10),'s2':box(19,0,39,10)},{'a':box(0,0,20,10)})
        self.assertEqual(r['reference_assignments'][0]['source_candidate_id'],'s1')

    def test_overlap_ownership_tie_abstains(self):
        r=self.compare({'s1':box(0,0,20,10),'s2':box(10,0,30,10)},{'a':box(5,0,25,10)})
        self.assertEqual(r['reference_assignments'][0]['status'],'DIRECT_OVERLAP_OWNERSHIP_AMBIGUOUS')

    def test_duplicate_reference_not_independent_parts(self):
        r=self.compare({'s':box(0,0,20,10)},{'a':box(0,0,20,10),'b':box(0,0,20,10)})
        self.assertEqual(r['hypotheses'],0);self.assertIn('REFERENCE_GROUP_OVERLAP_CONFLICT',r['records'][0]['reasons'])

    def test_source_positions_unchanged(self):
        source={'s':box(0,0,40,10)};before=source['s'].wkb
        self.compare(source,{'a':box(0,0,20,10),'b':box(20,0,40,10)});self.assertEqual(before,source['s'].wkb)

    def test_area_threshold_not_lowered(self):
        r=self.compare({'s':box(0,0,40,20)},{'a':box(0,0,20,10)})
        self.assertIn('GROUP_AREA_MISMATCH',r['records'][0]['reasons'])

    def test_no_reference_subset_search(self):
        r=self.compare({'s':box(0,0,20,20)},{'a':box(0,0,20,20),'b':box(10,0,60,20)})
        self.assertEqual(set(r['records'][0]['reference_fids']),{'a','b'})
        self.assertEqual(r['hypotheses'],0)

    def test_no_registration_vector(self):
        r=self.compare({'s':box(0,0,40,10)},{'a':box(1,0,21,10),'b':box(21,0,41,10)})
        self.assertFalse(r['records'][0]['registration_vector_generated'])


if __name__=='__main__':unittest.main()
