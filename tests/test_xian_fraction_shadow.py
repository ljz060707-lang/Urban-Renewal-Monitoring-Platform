import unittest
import numpy as np
from affine import Affine
from shapely.geometry import box, Polygon
from xian_r5.label_transfer import Grid, aggregate
from xian_r5.source_partition import SourceRegion
from xian_r5.fraction_shadow import build_fraction_shadow,compare_fraction_probes,NODATA

class FractionShadowTests(unittest.TestCase):
    def setUp(self):
        self.grid=Grid(Affine(10,0,0,0,-10,20),3,2,'EPSG:32649')
        self.support=box(0,0,30,20)
    def region(self,vector=(0.,0.),support=None,name='a'):
        return SourceRegion(name,'2018-11-01',support if support is not None else self.support,
                            vector,None if vector is None else 'a'*64)
    def build(self,objects,regions=None):
        return build_fraction_shadow(objects,regions or [self.region()],self.grid)
    def test_exact_half_pixel(self):
        r=self.build({'x':box(0,10,5,20)})
        self.assertEqual(r['raw_fraction'][0,0],.5)
        self.assertEqual(r['candidate_fraction'][0,0],.5)
    def test_full_denominator_not_partial_renormalized(self):
        r=self.build({'x':box(0,10,5,20)},[self.region(support=box(0,0,5,20))])
        self.assertEqual(r['candidate_fraction'][0,0],.5)
        self.assertEqual(r['scope_fraction'][1,0,0],.5)
        self.assertFalse(r['full_compare'][0,0])
    def test_unknown_is_not_zero(self):
        r=self.build({'x':box(0,10,5,20)},[self.region(None)])
        self.assertTrue((r['candidate_fraction']==NODATA).all())
        self.assertTrue((r['quality'][0]==2).all())
        self.assertEqual(r['raw_fraction'][0,0],.5)
    def test_zero_is_not_verified(self):
        r=self.build({})
        self.assertFalse(r['quality'][3].any())
        self.assertFalse(r['quality'][4].any())
        self.assertFalse(r['quality'][9].any())
        self.assertTrue(r['quality'][8].all())
    def test_subpixel_shift_exact_intersection(self):
        r=self.build({'x':box(5,10,10,20)},[self.region((2.5,0.))])
        self.assertAlmostEqual(r['candidate_fraction'][0,0],.25)
        self.assertAlmostEqual(r['candidate_fraction'][0,1],.25)
    def test_source_clip_before_shift(self):
        r=self.build({'outside':box(-10,10,0,20)},[self.region((10.,0.))])
        self.assertEqual(float(r['raw_fraction'].sum()),0)
        self.assertFalse((r['candidate_fraction']>0).any())
    def test_destination_clips_no_wrap(self):
        r=self.build({'x':box(25,10,30,20)},[self.region((10.,0.))])
        self.assertFalse((r['candidate_fraction']>0).any())
        self.assertEqual(r['candidate_fraction'][0,0],NODATA)
    def test_union_not_double_count(self):
        r=self.build({'x':box(0,10,7,20),'y':box(3,10,10,20)})
        self.assertAlmostEqual(r['raw_fraction'][0,0],1.)
        self.assertEqual(r['component_count'][0,0,0],2)
        self.assertEqual(r['quality'][7,0,0],1)
        self.assertEqual(r['quality'][0,0,0],4)
    def test_seam_component_risk(self):
        regions=[self.region(support=box(0,0,10,20)),self.region((0.,0.),box(10,0,30,20),'b')]
        r=self.build({'x':box(5,10,15,20)},regions)
        self.assertTrue(r['quality'][6,0,:2].all())
        self.assertTrue(r['component_ledger'][0]['source_seam_component'])
    def test_area_matches_existing_aggregate(self):
        objects={'x':box(1,2,17,16),'y':box(20,3,29,19)}
        from shapely.ops import unary_union
        r=self.build(objects)
        expected=aggregate(self.grid,{'ROOF':unary_union(list(objects.values()))},self.support)
        np.testing.assert_allclose(r['raw_fraction'],expected['fractions'][0],atol=1e-12)
        self.assertLess(max(r['area_conservation_error_m2'].values()),1e-8)
    def test_input_immutable(self):
        objects={'x':box(5,10,10,20)};before=objects['x'].wkb
        self.build(objects,[self.region((2.5,0.))])
        self.assertEqual(objects['x'].wkb,before)
    def test_overlap_source_fail_closed(self):
        with self.assertRaises(ValueError):
            self.build({},[self.region(),self.region(name='b')])
    def test_invalid_geometry_fail_closed(self):
        with self.assertRaises(ValueError):
            self.build({'x':Polygon([(0,0),(10,10),(0,10),(10,0)])})
    def test_crs_fail_closed(self):
        with self.assertRaises(ValueError):
            build_fraction_shadow({},[self.region()],Grid(self.grid.affine,3,2,'EPSG:4326'))
    def test_threshold_strict_and_frozen(self):
        r=self.build({'x':box(0,10,4.3,20)})
        p=compare_fraction_probes(r)
        self.assertEqual(p['probe_codes'][0,0],0)
        with self.assertRaises(ValueError):compare_fraction_probes(r,threshold=.5)
    def test_old_binary_is_comparison_only(self):
        r=self.build({'x':box(0,10,8,20)},[self.region((10.,0.))])
        old=np.zeros((2,3),np.uint8)
        p=compare_fraction_probes(r,old)
        self.assertEqual(p['statistics']['old_binary_comparison']['old_vs_raw_probe_mismatches'],1)
        self.assertFalse(r['quality'][9].any())

if __name__=='__main__':unittest.main()
