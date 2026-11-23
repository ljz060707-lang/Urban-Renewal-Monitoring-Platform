import unittest
from shapely.geometry import box, Polygon
from xian_r5.instance_evidence import component_evidence, InstanceEvidencePolicy


class InstanceEvidenceTests(unittest.TestCase):
    def evidence(self, source, old):
        return component_evidence(source,old,{'cell':box(0,0,100,100)})

    def test_independent_parts_form_merge_hypothesis_only(self):
        r=self.evidence({'s':box(0,0,20,10)},{'1':box(0,0,10,10),'2':box(10,0,20,10)})['records'][0]
        self.assertEqual(r['status'],'POSSIBLE_MERGED_SOURCE_COMPONENT_NOT_VERIFIED')
        self.assertFalse(r['verified_clean_roof']);self.assertFalse(r['physical_identity_verified'])

    def test_duplicate_auxiliary_not_two_independent_buildings(self):
        r=self.evidence({'s':box(0,0,20,10)},{'1':box(0,0,20,10),'2':box(0,0,20,10)})['records'][0]
        self.assertEqual(r['status'],'HISTORICAL_GEOMETRY_OVERLAP_CONFLICT')
        self.assertAlmostEqual(r['historical_union_covered_fraction'],1)

    def test_edge_contact_not_overlap(self):
        r=self.evidence({'s':box(0,0,10,10)},{'1':box(10,0,20,10)})['records'][0]
        self.assertEqual(r['historical_candidate_count'],0)

    def test_coincidence_not_identity_or_displacement(self):
        r=self.evidence({'s':box(0,0,10,10)},{'1':box(0,0,10,10)})
        self.assertEqual(r['records'][0]['status'],'SINGLE_GEOMETRIC_COINCIDENCE_NOT_IDENTITY')
        self.assertEqual(r['new_displacements'],0)

    def test_unexplained_not_automatically_nonroof(self):
        r=self.evidence({'s':box(0,0,20,20)},{'1':box(0,0,5,5)})['records'][0]
        self.assertEqual(r['status'],'UNEXPLAINED_SOURCE_AREA_NOT_CONFIRMED_CONTAMINATION')

    def test_empty_auxiliary_known_zero_not_match(self):
        r=self.evidence({'s':box(0,0,10,10)},{})
        self.assertEqual(r['historical_objects'],0)

    def test_no_nearest_object_fabricated(self):
        r=self.evidence({'s':box(0,0,10,10)},{'1':box(50,50,60,60)})['records'][0]
        self.assertEqual(r['relations'],[])

    def test_immutable_inputs(self):
        source={'s':box(0,0,10,10)};before=source['s'].wkb
        self.evidence(source,{'1':box(0,0,10,10)});self.assertEqual(source['s'].wkb,before)

    def test_invalid_geometry_fail_closed(self):
        with self.assertRaises(ValueError):self.evidence({'s':Polygon([(0,0),(10,10),(0,10),(10,0)])},{})

    def test_bad_policy_fail_closed(self):
        with self.assertRaises(ValueError):InstanceEvidencePolicy(diagnostic_covered_fraction=-1)

    def test_support_overlap_fail_closed(self):
        with self.assertRaises(ValueError):component_evidence({}, {}, {'a':box(0,0,10,10),'b':box(5,0,15,10)})


if __name__=='__main__':unittest.main()
