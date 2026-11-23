import unittest
from shapely.geometry import box
from xian_r5.native_fragment_evidence import native_fragment_evidence


class NativeFragmentTests(unittest.TestCase):
    def test_shared_row_boundary_not_independent_roofs(self):
        r=native_fragment_evidence({'a':box(10,1000,50,1024),'b':box(10,1024,50,1040)})
        self.assertEqual(r['status'],'SHARED_NATIVE_STRIP_EDGE_EVIDENCE')
        self.assertEqual(r['fragment_groups'][0]['historical_object_ids'],['a','b'])
        self.assertEqual(r['fragment_groups'][0]['native_union_parts'],1)

    def test_corner_not_shared_line(self):
        r=native_fragment_evidence({'a':box(0,1000,10,1024),'b':box(10,1024,20,1040)})
        self.assertEqual(r['fragment_groups'],[])

    def test_non_strip_roof_contact_not_strip_proof(self):
        r=native_fragment_evidence({'a':box(0,1000,10,1010),'b':box(0,1010,10,1040)})
        self.assertEqual(r['fragment_groups'],[])

    def test_subpixel_geometry_not_forced_to_native_grid(self):
        r=native_fragment_evidence({'a':box(.5,1000,10,1024),'b':box(0,1024,10,1040)})
        self.assertEqual(r['status'],'NATIVE_INTEGER_GRID_ALIGNMENT_NOT_ESTABLISHED')

    def test_roundtrip_noise_not_meter_accuracy(self):
        r=native_fragment_evidence({'a':box(0.00000001,1000,10,1024),'b':box(0,1024,10,1040)})
        self.assertIsNone(r['formal_precision_bound_m']);self.assertFalse(r['native_view_may_replace_source_geometry'])

    def test_source_input_unchanged(self):
        a=box(0,1000,10,1024);b=box(0,1024,10,1040);before=a.wkb
        native_fragment_evidence({'a':a,'b':b});self.assertEqual(before,a.wkb)

    def test_distant_never_joined(self):
        r=native_fragment_evidence({'a':box(0,1000,10,1024),'b':box(20,1024,30,1040)})
        self.assertEqual(r['fragment_groups'],[])

    def test_bad_tolerance_fail_closed(self):
        with self.assertRaises(ValueError):native_fragment_evidence({'a':box(0,0,10,10)},serialization_tolerance_px=1)


if __name__=='__main__':unittest.main()
