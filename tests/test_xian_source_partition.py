"""Synthetic source-transport checks, not registration accuracy tests."""
import unittest
from shapely.geometry import box
from xian_r5.source_partition import SourceRegion, source_partition_transport

class SourcePartitionTests(unittest.TestCase):
    def region(self, key, bounds, vector=None):
        return SourceRegion(key, '2018-11-01', box(*bounds), vector,
                            'a'*64 if vector is not None else None)

    def test_unknown_vector_is_not_zero(self):
        r=source_partition_transport({'b':box(2,2,4,4)},
            [self.region('a',(0,0,10,10))],box(-20,-20,30,30))
        self.assertEqual(r['objects']['b']['status'],'UNRESOLVED_NO_VECTOR')
        self.assertNotIn('b',r['candidate_geometries'])

    def test_two_sources_move_independently(self):
        r=source_partition_transport({'b':box(2,2,4,4),'c':box(12,2,14,4)},
            [self.region('a',(0,0,10,10),(3,2)),
             self.region('z',(10,0,20,10),(-2,-1))],box(-20,-20,30,30))
        self.assertTrue(r['candidate_geometries']['b'].equals(box(5,4,7,6)))
        self.assertTrue(r['candidate_geometries']['c'].equals(box(10,1,12,3)))
        self.assertEqual(r['formal_corrected_objects'],0)

    def test_cross_source_roof_is_not_silently_torn(self):
        r=source_partition_transport({'b':box(9,2,11,4)},
            [self.region('a',(0,0,10,10),(3,2)),
             self.region('z',(10,0,20,10),(-2,-1))],box(-20,-20,30,30))
        self.assertEqual(r['objects']['b']['status'],'UNRESOLVED_SOURCE_SEAM')
        self.assertNotIn('b',r['candidate_geometries'])

    def test_outside_fragment_does_not_move_into_support(self):
        r=source_partition_transport({'b':box(-5,2,-2,4)},
            [self.region('a',(0,0,10,10),(5,0))],box(-20,-20,30,30))
        self.assertEqual(r['objects']['b']['status'],'UNRESOLVED_OUTSIDE_SOURCE_SUPPORT')
        self.assertFalse(r['candidate_geometries'])

    def test_partial_source_observation_is_explicit(self):
        r=source_partition_transport({'b':box(-1,2,2,4)},
            [self.region('a',(0,0,10,10),(5,0))],box(-20,-20,30,30))
        self.assertEqual(r['objects']['b']['status'],'UNRESOLVED_PARTIAL_SOURCE_OBJECT')
        self.assertFalse(r['candidate_geometries'])

    def test_target_clipping_is_not_complete_correction(self):
        r=source_partition_transport({'b':box(6,2,9,4)},
            [self.region('a',(0,0,10,10),(3,0))],box(0,0,10,10))
        self.assertEqual(r['objects']['b']['status'],'CANDIDATE_TARGET_EDGE_CLIPPED')
        self.assertAlmostEqual(r['candidate_geometries']['b'].area,2)

    def test_overlapping_supports_fail_closed(self):
        with self.assertRaises(ValueError):
            source_partition_transport({},[self.region('a',(0,0,10,10)),
                self.region('z',(5,0,20,10))],box(0,0,30,30))

    def test_duplicate_region_ids_fail_closed(self):
        with self.assertRaises(ValueError):
            source_partition_transport({},[self.region('a',(0,0,10,10)),
                self.region('a',(10,0,20,10))],box(0,0,30,30))

    def test_vector_requires_finite_values_and_evidence(self):
        for v,h in [((float('nan'),0),'a'*64),((1,0),None),((True,0),'a'*64)]:
            with self.assertRaises(ValueError):SourceRegion('a','2018-11-01',box(0,0,1,1),v,h)

    def test_no_absolute_or_training_accuracy_claim(self):
        r=source_partition_transport({'b':box(2,2,4,4)},
            [self.region('a',(0,0,10,10),(0,0))],box(0,0,10,10))
        self.assertEqual(r['position_semantics'],'RELATIVE_IMAGE_GRID_ONLY')
        self.assertEqual(r['absolute_position_accuracy'],'UNKNOWN')
        self.assertFalse(r['training_gt_eligible'])

if __name__=='__main__':unittest.main()
