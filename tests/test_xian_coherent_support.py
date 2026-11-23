import unittest
import hashlib
from shapely.geometry import box
from xian_r5.coherent_support import coalesce_equal_field_regions
from xian_r5.source_partition import SourceRegion
from xian_r5.label_transfer import Grid
from xian_r5.fraction_shadow import build_fraction_shadow
from affine import Affine


class CoherentSupportTests(unittest.TestCase):
    def region(self, name, geometry, vector=(0,-10), date='2018-11-01'):
        return SourceRegion(name, date, geometry, vector, None if vector is None else hashlib.sha256(name.encode()).hexdigest())

    def test_adjacent_equal_field_coalesced_without_mutation(self):
        a=self.region('a',box(0,0,20,20)); b=self.region('b',box(20,0,40,20))
        before=[r.support.wkb for r in (a,b)]; out,groups=coalesce_equal_field_regions([a,b])
        self.assertEqual(len(out),1);self.assertEqual(len(groups[0]['members']),2)
        self.assertEqual(before,[r.support.wkb for r in (a,b)])

    def test_corner_contact_not_coalesced(self):
        out,_=coalesce_equal_field_regions([self.region('a',box(0,0,10,10)),self.region('b',box(10,10,20,20))])
        self.assertEqual(len(out),2)

    def test_different_vectors_not_coalesced(self):
        out,_=coalesce_equal_field_regions([self.region('a',box(0,0,10,10)),self.region('b',box(10,0,20,10),(10,0))])
        self.assertEqual(len(out),2)

    def test_different_dates_not_coalesced(self):
        out,_=coalesce_equal_field_regions([self.region('a',box(0,0,10,10)),self.region('b',box(10,0,20,10),date='2019-12-02')])
        self.assertEqual(len(out),2)

    def test_unknown_keeps_exact_support_and_null(self):
        a=self.region('a',box(0,0,10,10)); b=self.region('b',box(10,0,20,10),None)
        out,_=coalesce_equal_field_regions([a,b]);self.assertIs(out[1],b)

    def test_gap_not_filled(self):
        out,_=coalesce_equal_field_regions([self.region('a',box(0,0,10,10)),self.region('b',box(11,0,21,10))])
        self.assertEqual(len(out),2)

    def test_l_shape_not_convex_hull(self):
        rs=[self.region('a',box(0,0,10,10)),self.region('b',box(10,0,20,10)),self.region('c',box(0,10,10,20))]
        out,_=coalesce_equal_field_regions(rs);self.assertEqual(out[0].support.area,300)
        self.assertEqual(out[0].support.intersection(box(10,10,20,20)).area,0)

    def test_overlap_fail_closed(self):
        with self.assertRaises(ValueError):coalesce_equal_field_regions([self.region('a',box(0,0,11,10)),self.region('b',box(10,0,20,10))])

    def test_internal_grid_seam_removed_but_unknown_seam_preserved(self):
        rs=[self.region('a',box(0,0,20,40),(0,0)),self.region('b',box(20,0,40,40),(0,0)),self.region('u',box(40,0,60,40),None)]
        merged,_=coalesce_equal_field_regions(rs);grid=Grid(Affine(10,0,0,0,-10,40),6,4,'EPSG:32649')
        objects={'inside':box(15,15,25,25),'outside':box(35,15,45,25)}
        old=build_fraction_shadow(objects,rs,grid);new=build_fraction_shadow(objects,merged,grid)
        self.assertTrue(old['quality'][6,1,1]);self.assertFalse(new['quality'][6,1,1])
        self.assertTrue(new['quality'][6,1,3]);self.assertEqual(new['candidate_fraction'][1,4],-9999)

    def test_no_formal_or_acquisition_promotion(self):
        _,groups=coalesce_equal_field_regions([self.region('a',box(0,0,10,10))])
        self.assertFalse(groups[0]['formal_alignment_accepted']);self.assertFalse(groups[0]['camera_acquisition_unit_verified'])


if __name__=='__main__':unittest.main()
